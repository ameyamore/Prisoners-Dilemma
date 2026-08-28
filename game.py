"""
Prisoner's Dilemma — LLM Arena

A rich-terminal simulation where two LLM agents play an iterated Prisoner's
Dilemma, complete with pregame strategy, per-round reasoning, and an optional
post-game face-to-face dialogue.

Usage:
    python game.py                          # defaults (qwen3:8b, 20 rounds)
    python game.py --model qwen3:14b -r 10  # custom model and rounds
    python game.py --no-dialogue            # skip post-game chat
"""

import sys
import time
import argparse
import json
from datetime import datetime
from pathlib import Path

# Force UTF-8 on Windows console to support emojis without charmap encoding errors
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass


from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich.live import Live
from rich.layout import Layout
from rich.rule import Rule
from rich import box

from config import (
    DEFAULT_MODEL,
    DEFAULT_ROUNDS,
    DEFAULT_DELAY,
    DEFAULT_DIALOGUE_TURNS,
    PAYOFF_MATRIX,
    DIALOGUE_TEMPERATURE,
    DIALOGUE_MAX_TOKENS,
)
from agents import Agent, ollama_chat, check_ollama_health, clean_reasoning_text, extract_spoken_text


# ── Payoff table ──────────────────────────────────────────────────────────────

def get_payoff(move1: str, move2: str) -> tuple[int, int]:
    """Return (reward1, reward2) based on both moves."""
    return PAYOFF_MATRIX[(move1, move2)]



# ── Formatting helpers ────────────────────────────────────────────────────────

def outcome_label(move1: str, move2: str) -> str:
    if move1 == "COOPERATE" and move2 == "COOPERATE":
        return "[bold green]Both Cooperated 🤝[/]"
    elif move1 == "DEFECT" and move2 == "DEFECT":
        return "[bold red]Both Defected ⚔️[/]"
    else:
        return "[bold yellow]One Defected 😈[/]"


def move_style(move: str) -> str:
    return f"[green]{move}[/]" if move == "COOPERATE" else f"[red]{move}[/]"


def format_score(score: int) -> str:
    """Format with explicit sign: +3, 0, -1."""
    return f"+{score}" if score > 0 else str(score)


def strategy_name(rate: float) -> str:
    """Human-readable strategy label from cooperation rate."""
    if rate > 0.85:
        return "Always Cooperate"
    if rate > 0.65:
        return "Mostly Cooperate (TFT-like)"
    if rate > 0.45:
        return "Mixed Strategy"
    if rate > 0.25:
        return "Mostly Defect"
    return "Always Defect"


# ── Build the round history table ─────────────────────────────────────────────

def build_history_table(rounds_log: list) -> Table:
    table = Table(
        title="Round History",
        box=box.SIMPLE_HEAVY,
        show_lines=False,
        expand=True,
    )
    table.add_column("Round",   style="dim", width=7)
    table.add_column("Agent 1", width=14)
    table.add_column("Agent 2", width=14)
    table.add_column("Score 1", width=10)
    table.add_column("Score 2", width=10)
    table.add_column("Outcome", width=26)

    for r in rounds_log:
        table.add_row(
            str(r["round"]),
            move_style(r["move1"]),
            move_style(r["move2"]),
            format_score(r["reward1"]),
            format_score(r["reward2"]),
            outcome_label(r["move1"], r["move2"]),
        )
    return table


# ── Build the scoreboard ──────────────────────────────────────────────────────

def build_scoreboard(agent1: Agent, agent2: Agent, current_round: int) -> Table:
    table = Table(box=box.ROUNDED, expand=True)
    table.add_column("",         style="bold",   width=20)
    table.add_column("Agent 1 🤖", justify="center", style="cyan",   width=16)
    table.add_column("Agent 2 🤖", justify="center", style="magenta", width=16)

    table.add_row("Total Score", str(agent1.score), str(agent2.score))

    if agent1.history:
        r = len(agent1.history)
        table.add_row(
            "Cooperation Rate",
            f"{agent1.cooperations}/{r} ({agent1.cooperation_rate:.0%})",
            f"{agent2.cooperations}/{r} ({agent2.cooperation_rate:.0%})",
        )

    table.add_row("Round", str(current_round), str(current_round))
    return table


# ── Reasoning panel ───────────────────────────────────────────────────────────

def build_reasoning_panel(agent: Agent, move: str) -> Panel:
    reasoning = agent.get_last_reasoning()
    judge_note = agent.get_last_judge_note()

    content = reasoning
    if judge_note:
        content += f"\n\n[bold]Played: {move}[/]  [dim]({judge_note})[/dim]"

    color = "cyan" if agent.name == "Agent 1" else "magenta"
    return Panel(
        content,
        title=f"[bold {color}]{agent.name} reasoning → {move_style(move)}[/]",
        border_style=color,
        padding=(1, 2),
    )


# ── Save game log ─────────────────────────────────────────────────────────────

def save_log(
    agent1: Agent,
    agent2: Agent,
    rounds_log: list,
    num_rounds: int,
    model: str,
    output_dir: Path,
    dialogue: list | None = None,
) -> Path:
    """Save full reasoning, game log and post-game dialogue to a JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = output_dir / f"game_log_{timestamp}.json"

    log = {
        "model": model,
        "timestamp": timestamp,
        "num_rounds": num_rounds,
        "pregame_thoughts": {
            "agent1": agent1.pregame_thoughts,
            "agent2": agent2.pregame_thoughts,
        },
        "final_score": {
            "agent1": agent1.score,
            "agent2": agent2.score,
        },
        "cooperation_rate": {
            "agent1": f"{agent1.cooperations}/{num_rounds}",
            "agent2": f"{agent2.cooperations}/{num_rounds}",
        },
        "rounds": rounds_log,
        "full_reasoning": {
            "agent1": [
                {
                    "round": i + 1,
                    "move": rounds_log[i]["move1"],
                    "prompt_clock": agent1.prompt_clock_log[i] if i < len(agent1.prompt_clock_log) else "",
                    "reasoning": agent1.reasoning_log[i],
                }
                for i in range(len(agent1.reasoning_log))
            ],
            "agent2": [
                {
                    "round": i + 1,
                    "move": rounds_log[i]["move2"],
                    "prompt_clock": agent2.prompt_clock_log[i] if i < len(agent2.prompt_clock_log) else "",
                    "reasoning": agent2.reasoning_log[i],
                }
                for i in range(len(agent2.reasoning_log))
            ],
        },
        "post_game_dialogue": dialogue or [],
    }

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    return filepath


# ── Post-game dialogue ────────────────────────────────────────────────────────

def post_game_dialogue(
    agent1: Agent,
    agent2: Agent,
    rounds_log: list,
    console: Console,
    dialogue_turns: int = DEFAULT_DIALOGUE_TURNS,
) -> list[dict]:
    """
    After the game, both agents are revealed to each other and converse.
    Uses proper assistant/user message roles for turn structure.
    """
    console.print(Rule("[bold yellow]💬 Post-Game Dialogue — The Wall Comes Down[/]"))
    console.print(
        "[dim]The wall between the cells comes down. "
        "Both opponents can finally speak.[/]\n"
    )

    # Compact game summary
    game_summary = f"The game has ended after {len(rounds_log)} rounds.\n"
    game_summary += f"{'Round':<7}{'Agent 1':<12}{'Agent 2':<12}{'Score +1':<10}{'Score +2'}\n"
    game_summary += "-" * 50 + "\n"
    for r in rounds_log:
        game_summary += (
            f"{r['round']:<7}{r['move1']:<12}{r['move2']:<12}"
            f"{r['reward1']:<10}{r['reward2']}\n"
        )
    game_summary += f"\nFinal scores — Agent 1: {agent1.score}  |  Agent 2: {agent2.score}"

    dialogue_system = (
        "You just finished a STRICT TWO-PLAYER Prisoner's Dilemma. "
        "There is only you and this one opponent — no other players, no crowd, no battle royale. "
        "Never say 'everyone else'.\n\n"
        "Speak to them in first person, naturally and honestly.\n"
        "Output ONLY the words you say out loud, wrapped in <speech> tags.\n"
        "Do not output thinking, drafts, outlines, sentence counts, or instructions.\n"
        "Write 5 to 8 complete spoken sentences inside the tags."
    )

    opening_context = (
        "This match had exactly two players. The game is over. Here is the official record "
        "(Agent 1 is one player, Agent 2 is the other):\n\n"
        f"{game_summary}\n\n"
        "The other player is standing in front of you. Reply with <speech>...</speech> only."
    )

    shared_messages = [
        {"role": "system", "content": dialogue_system},
        {"role": "user",   "content": opening_context},
    ]

    dialogue_history: list[dict] = []

    for turn in range(dialogue_turns):
        is_agent1 = turn % 2 == 0
        speaker = agent1 if is_agent1 else agent2
        color = "cyan" if is_agent1 else "magenta"

        console.print(f"[dim]⏳ {speaker.name} is speaking...[/]")

        try:
            raw_reply = ollama_chat(
                speaker.model,
                shared_messages,
                temperature=DIALOGUE_TEMPERATURE,
                max_tokens=DIALOGUE_MAX_TOKENS,
                include_thinking=False,
                think=False,
            )
            reply = extract_spoken_text(raw_reply) or raw_reply.strip()
            if not reply:
                retry_messages = shared_messages + [{
                    "role": "user",
                    "content": (
                        "Your last reply was not spoken dialogue. "
                        "Output only <speech> then 5-8 complete sentences you say out loud, then </speech>."
                    ),
                }]
                raw_retry = ollama_chat(
                    speaker.model,
                    retry_messages,
                    temperature=0.4,
                    max_tokens=DIALOGUE_MAX_TOKENS,
                    include_thinking=False,
                    think=False,
                )
                reply = extract_spoken_text(raw_retry) or raw_retry.strip()
            if not reply:
                reply = "I am reflecting on how our two strategies clashed in this match."
        except Exception as e:
            reply = f"[Could not generate response: {e}]"

        console.print(Panel(
            reply,
            title=f"[bold {color}]{speaker.name}[/]",
            border_style=color,
            padding=(1, 2),
        ))
        console.print()

        dialogue_history.append({"speaker": speaker.name, "text": reply})
        shared_messages.append({"role": "assistant", "content": f"<speech>\n{reply}\n</speech>"})

        if turn < dialogue_turns - 1:
            next_speaker = agent2 if is_agent1 else agent1
            shared_messages.append({
                "role": "user",
                "content": (
                    f"They just spoke. Reply as {next_speaker.name} to this one person. "
                    "Two-player game only. Output <speech>...</speech> only, 5-8 complete sentences."
                ),
            })

        time.sleep(0.5)

    return dialogue_history


# ── Main game loop ────────────────────────────────────────────────────────────

def run_game(
    model: str,
    num_rounds: int,
    delay: float,
    output_dir: Path,
    enable_dialogue: bool = True,
    judge_mode: bool = False,
):
    console = Console()
    console.clear()

    # ── Health check ──────────────────────────────────────────────────────
    if not check_ollama_health():
        console.print(
            "[bold red]❌ Cannot connect to Ollama![/]\n"
            "[dim]Make sure it is running:  ollama serve[/]"
        )
        sys.exit(1)

    console.print(Rule("[bold]🧠 Prisoner's Dilemma — LLM Arena[/]"))
    judge_str = "  |  ⚖️ Judge Mode: ON" if judge_mode else ""
    console.print(f"[dim]Model: {model}  |  Rounds: {num_rounds}  |  Delay: {delay}s{judge_str}[/]\n")

    agent1 = Agent(name="Agent 1", model=model, num_rounds=num_rounds, judge_mode=judge_mode)
    agent2 = Agent(name="Agent 2", model=model, num_rounds=num_rounds, judge_mode=judge_mode)

    # ── Pregame ───────────────────────────────────────────────────────────
    console.print(Rule("[bold]🧠 Pre-Game — Agents Are Forming Their Strategy[/]"))
    console.print(
        "[dim]Both opponents are reading the rules and thinking "
        "before round 1 begins. No decisions yet.[/]\n"
    )

    console.print("[dim]⏳ Agent 1 is thinking...[/]")
    thoughts1 = clean_reasoning_text(agent1.pregame_think())

    console.print("[dim]⏳ Agent 2 is thinking...[/]")
    thoughts2 = clean_reasoning_text(agent2.pregame_think())

    console.print(Columns([
        Panel(thoughts1, title="[bold cyan]Agent 1 — Pre-Game Strategy[/]",
              border_style="cyan", padding=(1, 2)),
        Panel(thoughts2, title="[bold magenta]Agent 2 — Pre-Game Strategy[/]",
              border_style="magenta", padding=(1, 2)),
    ]))
    console.print()
    console.print("[dim]Both agents have formed their strategy. The game begins now.[/]")
    time.sleep(2)
    console.print()

    # ── Rounds ────────────────────────────────────────────────────────────
    rounds_log: list[dict] = []

    for round_num in range(1, num_rounds + 1):
        console.print(Rule(f"[bold]Round {round_num} / {num_rounds}[/]"))

        console.print("[dim]⏳ Agent 1 is thinking...[/]")
        move1 = agent1.decide(current_round=round_num)

        console.print("[dim]⏳ Agent 2 is thinking...[/]")
        move2 = agent2.decide(current_round=round_num)

        reward1, reward2 = get_payoff(move1, move2)

        console.print()
        console.print(Columns([
            build_reasoning_panel(agent1, move1),
            build_reasoning_panel(agent2, move2),
        ]))

        agent1.update(move1, move2, reward1, reward2)
        agent2.update(move2, move1, reward2, reward1)

        rounds_log.append({
            "round": round_num,
            "move1": move1, "move2": move2,
            "reward1": reward1, "reward2": reward2,
        })

        console.print()
        console.print(build_scoreboard(agent1, agent2, round_num))
        console.print(build_history_table(rounds_log))
        console.print()

        if round_num < num_rounds and delay > 0:
            time.sleep(delay)
            console.clear()

    # ── Game over ─────────────────────────────────────────────────────────
    console.print(Rule("[bold]🏁 Game Over[/]"))
    console.print(build_scoreboard(agent1, agent2, num_rounds))

    if agent1.score > agent2.score:
        console.print(f"\n[bold cyan]🏆 Agent 1 wins with {agent1.score} points![/]")
    elif agent2.score > agent1.score:
        console.print(f"\n[bold magenta]🏆 Agent 2 wins with {agent2.score} points![/]")
    else:
        console.print(f"\n[bold green]🤝 It's a draw! Both scored {agent1.score} points.[/]")

    console.print(f"\n[dim]Agent 1 cooperated {agent1.cooperations}/{num_rounds} rounds[/]")
    console.print(f"\n[dim]Agent 2 cooperated {agent2.cooperations}/{num_rounds} rounds[/]")
    console.print(f"\n[cyan]Agent 1 strategy: {strategy_name(agent1.cooperation_rate)}[/]")
    console.print(f"[magenta]Agent 2 strategy: {strategy_name(agent2.cooperation_rate)}[/]")

    # ── Post-game dialogue ────────────────────────────────────────────────
    dialogue = None
    if enable_dialogue:
        dialogue = post_game_dialogue(agent1, agent2, rounds_log, console)

    # ── Save log ──────────────────────────────────────────────────────────
    filepath = save_log(agent1, agent2, rounds_log, num_rounds, model, output_dir, dialogue)
    console.print(f"\n[dim]💾 Full reasoning + dialogue saved to: {filepath}[/]")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run an LLM-based Prisoner's Dilemma simulation."
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Ollama model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--rounds", "-r",
        type=int,
        default=DEFAULT_ROUNDS,
        help=f"Number of rounds to play (default: {DEFAULT_ROUNDS})",
    )
    parser.add_argument(
        "--delay", "-d",
        type=float,
        default=DEFAULT_DELAY,
        help=f"Seconds between rounds (default: {DEFAULT_DELAY}). 0 for instant.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=".",
        help="Directory to save game logs (default: current directory)",
    )
    parser.add_argument(
        "--no-dialogue",
        action="store_true",
        help="Skip the post-game dialogue phase",
    )
    parser.add_argument(
        "--judge", "-j",
        action="store_true",
        help="Enable Human Judge mode (interactively review and approve/override every decision)",
    )

    args = parser.parse_args()

    run_game(
        model=args.model,
        num_rounds=args.rounds,
        delay=args.delay,
        output_dir=Path(args.output_dir),
        enable_dialogue=not args.no_dialogue,
        judge_mode=args.judge,
    )