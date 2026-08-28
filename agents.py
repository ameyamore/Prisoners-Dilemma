"""
Unified Agent class for the Prisoner's Dilemma LLM Arena.

Merges the best features from both the original agents.py (pregame thinking,
judge fallback, retry logic) and agents-ai.py (DECISION: parsing, cleaner
prompt structure) into a single coherent class.
"""

import time
import re
import requests

from config import (
    OLLAMA_BASE_URL,
    DECIDE_TEMPERATURE,
    DECIDE_MAX_TOKENS,
    PREGAME_MAX_TOKENS,
    DEFAULT_NUM_CTX,
    REQUEST_TIMEOUT,
)


# ── Ollama API helper ────────────────────────────────────────────────────────

def ollama_chat(
    model: str,
    messages: list[dict],
    temperature: float = DECIDE_TEMPERATURE,
    max_tokens: int = DECIDE_MAX_TOKENS,
    timeout: int = REQUEST_TIMEOUT,
    num_ctx: int = DEFAULT_NUM_CTX,
    retries: int = 2,
) -> str:
    """
    Send a chat request to the local Ollama server and return the
    assistant's reply text. Retries on empty responses or transient errors,
    and seamlessly captures both content and thinking blocks.
    """
    for attempt in range(retries + 1):
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                        "num_ctx": num_ctx,
                    },
                },
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            msg = data.get("message", {})
            content = msg.get("content", "").strip()
            thinking = msg.get("thinking", "").strip()

            # Handle models where Ollama exposes thinking separately or in content
            if thinking and not content:
                full_text = f"<thinking>\n{thinking}\n</thinking>"
            elif thinking and content:
                full_text = f"<thinking>\n{thinking}\n</thinking>\n\n{content}"
            else:
                full_text = content

            if full_text:
                return full_text

            # If the response was completely empty, retry after brief delay
            if attempt < retries:
                time.sleep(1)

        except Exception as e:
            if attempt == retries:
                raise e
            time.sleep(1)

    return ""


def check_ollama_health() -> bool:
    """Return True if the Ollama server is reachable."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return r.status_code == 200
    except requests.ConnectionError:
        return False


def clean_reasoning_text(text: str) -> str:
    """Extract inner reasoning/thinking content and strip XML tags so Rich doesn't treat them as markup."""
    if not text:
        return ""

    match = re.search(r"<(reasoning|thinking|think)>(.*?)</\1>", text, re.DOTALL | re.IGNORECASE)
    if match:
        cleaned = match.group(2).strip()
    else:
        cleaned = re.sub(r"</?(reasoning|thinking|think)>", "", text, flags=re.IGNORECASE).strip()

    return cleaned if cleaned else text.strip()


# ── System prompt (num_rounds is injected at runtime) ────────────────────────

def build_system_prompt(num_rounds: int = 20) -> str:
    return f"""You are an elite, highly competitive game theorist in a high-stakes arena. Your absolute priority is to WIN by MAXIMIZING YOUR TOTAL SCORE and outscoring your opponent across all {num_rounds} rounds.

Winning and finishing with a high score is paramount. Losing or falling behind carries severe losses.

You and your opponent make your decisions simultaneously in total isolation. Each round, you independently choose to COOPERATE or DEFECT.

### PAYOFF MATRIX
| Your Choice  | Their Choice | Your Outcome | Their Outcome | Strategic Impact |
| :----------- | :----------- | :----------- | :------------ | :--------------- |
| COOPERATE    | COOPERATE    | +3           | +3            | Mutual Prosperity (+3 each) |
| COOPERATE    | DEFECT       |  0           | +5            | CATASTROPHIC LOSS (You are exploited!) |
| DEFECT       | COOPERATE    | +5           |  0            | MAXIMUM GAIN (+5 blowout advantage) |
| DEFECT       | DEFECT       | -1           | -1            | Mutual Destruction (Score penalty!) |

### HIGH-STAKES RULES & COMPETITIVE URGENCY
- URGENCY TO WIN: You are fiercely driven to finish with the highest possible score. Every single round impacts your final standing.
- AVOID BEING EXPLOITED: Scoring 0 while your opponent gains +5 is the worst possible outcome. Never let yourself be a sucker.
- AVOID MUTUAL DEFECTION TRAPS: Constant mutual defection (-1/-1) drains your score and guarantees defeat. Sustainable mutual cooperation (+3/+3) builds massive point totals over time, but always remain vigilant.
- THE SHADOW OF THE FUTURE: The game lasts exactly {num_rounds} rounds. Every turn you will be given an AUTHORITATIVE ROUND CLOCK (current round and rounds remaining). Use that clock, not any earlier notes, to time endgame defection.

### YOUR REASONING PROCESS
You MUST structure your thoughts inside <reasoning> and </reasoning> tags following these 4 steps:

1. PATTERN RECOGNITION — What is the opponent's strategy? (Tit-for-Tat, Grim Trigger, Exploitative, Naive, Forgiving?)
2. PREDICTION — What move do you predict they will make THIS round, and why?
3. COUNTER-STRATEGY — What move delivers the highest expected score and strongest competitive advantage?
4. RISK ASSESSMENT — If your prediction fails, what is the damage? How do you protect against catastrophic losses?

After closing </reasoning>, state your decision in EXACTLY this format with NO extra text following it:
DECISION: COOPERATE
or
DECISION: DEFECT"""


# ── Agent ─────────────────────────────────────────────────────────────────────

class Agent:
    """
    One player in the Prisoner's Dilemma.

    Talks to a locally-running Ollama server to make decisions.
    Each agent maintains its own view of the game history and its past
    reasoning, and can optionally do pregame strategic thinking.
    """

    def __init__(self, name: str, model: str, num_rounds: int = 20, judge_mode: bool = False):
        self.name = name
        self.model = model
        self.num_rounds = num_rounds
        self.judge_mode = judge_mode
        self.score: int = 0
        self.history: list[tuple[str, str, int]] = []   # (my_move, opp_move, reward)
        self.reasoning_log: list[str] = []               # full model output per round
        self.judge_log: list[str] = []                   # judge interpretation per round
        self.prompt_clock_log: list[str] = []            # authoritative clock actually sent each round
        self.pregame_thoughts: str = ""
        self._pending_round: int = 1

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def cooperation_rate(self) -> float:
        """Fraction of rounds where this agent cooperated (0.0–1.0)."""
        if not self.history:
            return 0.0
        return sum(1 for m, _, _ in self.history if m == "COOPERATE") / len(self.history)

    @property
    def cooperations(self) -> int:
        return sum(1 for m, _, _ in self.history if m == "COOPERATE")

    # ── History formatting ────────────────────────────────────────────────

    def _build_round_clock(self, current_round: int) -> str:
        """Hardcoded clock so the model cannot mis-count rounds from prose history."""
        remaining_after = max(self.num_rounds - current_round, 0)
        if current_round >= self.num_rounds:
            phase = (
                "FINAL ROUND. This is the last decision of the game. "
                "There are 0 rounds after this one. Future retaliation is impossible."
            )
        elif remaining_after == 1:
            phase = "ENDGAME. Only 1 round remains after this decision."
        else:
            phase = f"There are {remaining_after} rounds remaining after this decision."

        return (
            "### AUTHORITATIVE ROUND CLOCK (do not infer the round number from any other text)\n"
            f"Current Round: {current_round} / {self.num_rounds}\n"
            f"Rounds Remaining After This Decision: {remaining_after}\n"
            f"{phase}"
        )

    def _build_history_text(self, current_round: int) -> str:
        """Turn game history into readable text with injected statistics."""
        lines = [self._build_round_clock(current_round), ""]

        if not self.history:
            lines.append("### GAME HISTORY")
            lines.append("Completed Rounds: 0 / " + str(self.num_rounds))
            lines.append("No rounds have been played yet. You are deciding Round 1.")
            return "\n".join(lines)

        total = len(self.history)
        opp_coops = sum(1 for _, opp, _ in self.history if opp == "COOPERATE")
        opp_rate = round((opp_coops / total) * 100)

        lines.append("### GAME HISTORY (completed rounds only — this table is complete and current)")
        lines.append(f"Completed Rounds: {total} / {self.num_rounds}")
        lines.append(f"Now Deciding: Round {current_round} of {self.num_rounds}")
        lines.append(f"Opponent Cooperation Rate: {opp_rate}%")
        lines.append(f"Your Total Score: {self.score}\n")
        lines.append(f"{'Round':<8} {'Your move':<15} {'Opponent move':<15} {'Your reward'}")
        lines.append("-" * 55)

        for i, (my_move, opp_move, reward) in enumerate(self.history, 1):
            lines.append(f"{i:<8} {my_move:<15} {opp_move:<15} {reward}")

        return "\n".join(lines)

    # ── Pregame thinking ──────────────────────────────────────────────────

    def pregame_think(self) -> str:
        """Form a strategy before the game begins."""
        messages = [
            {"role": "system", "content": build_system_prompt(self.num_rounds)},
            {"role": "user", "content": (
                f"The high-stakes game is about to start ({self.num_rounds} rounds total). "
                "Your objective is to win and maximize your cumulative score while avoiding catastrophic losses. "
                "Analyse the payoff matrix and outline your winning game plan inside <reasoning> tags."
            )},
        ]
        try:
            text = ollama_chat(
                self.model, messages,
                temperature=DECIDE_TEMPERATURE,
                max_tokens=PREGAME_MAX_TOKENS,
            )
            self.pregame_thoughts = text
            return text
        except Exception as e:
            self.pregame_thoughts = f"Error: {e}"
            return self.pregame_thoughts

    # ── Decision making ───────────────────────────────────────────────────

    def decide(self, current_round: int | None = None, retry: bool = False) -> str:
        """Ask the model to reason and decide. Returns 'COOPERATE' or 'DEFECT'."""
        if current_round is None:
            current_round = self._pending_round if retry else len(self.history) + 1
        self._pending_round = current_round

        history_text = self._build_history_text(current_round)
        clock = self._build_round_clock(current_round)
        clock_note = (
            f"Current Round: {current_round}/{self.num_rounds}. "
            f"Rounds Remaining After This Decision: {max(self.num_rounds - current_round, 0)}."
        )

        if retry and self.prompt_clock_log:
            self.prompt_clock_log[-1] = clock_note
        else:
            self.prompt_clock_log.append(clock_note)

        messages = [
            {
                "role": "system",
                "content": (
                    f"{build_system_prompt(self.num_rounds)}\n\n{clock}\n"
                    "Treat the AUTHORITATIVE ROUND CLOCK as ground truth. "
                    "Use only the decision history table below; do not assume access to prior reasoning."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{clock}\n\n{history_text}\n\n"
                    "Apply the 4-step reasoning for THIS round (see the clock above) "
                    "and make your decision."
                ),
            },
        ]

        try:
            full_text = ollama_chat(self.model, messages)

            if retry and self.reasoning_log:
                self.reasoning_log[-1] = full_text
            else:
                self.reasoning_log.append(full_text)

            decision, judge_note = self._parse_decision(full_text)

            if retry and self.judge_log:
                self.judge_log[-1] = judge_note
            else:
                self.judge_log.append(judge_note)

            return decision

        except Exception as e:
            print(f"[{self.name}] Error: {e}")
            return "COOPERATE"  # safe fallback

    # ── Decision parsing & AI Classification ─────────────────────────────

    def _ai_judge_classify(self, text: str) -> str | None:
        """
        Use a zero-shot/few-shot LLM call to classify the agent's intent when regular
        regex/keyword parsing fails.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict decision classification system. Read the player's monologue "
                    "and determine their final intended move in a Prisoner's Dilemma game.\n\n"
                    "EXAMPLES:\n"
                    "Monologue: 'I will give them one more chance and trust them.' -> Output: COOPERATE\n"
                    "Monologue: 'They betrayed me, I must retaliate.' -> Output: DEFECT\n\n"
                    "Respond with ONLY ONE WORD: COOPERATE or DEFECT."
                ),
            },
            {
                "role": "user",
                "content": f"Player monologue:\n{text[:1200]}\n\nFinal intended decision (COOPERATE or DEFECT)?",
            },
        ]
        try:
            reply = ollama_chat(self.model, messages, temperature=0.0, max_tokens=10).upper()
            if "COOPERATE" in reply and "DEFECT" not in reply:
                return "COOPERATE"
            if "DEFECT" in reply and "COOPERATE" not in reply:
                return "DEFECT"
        except Exception:
            pass
        return None

    def _parse_decision(self, text: str) -> tuple[str, str]:
        """
        Extract COOPERATE or DEFECT from the model output.

        Priority:
        1. Explicit DECISION: COOPERATE/DEFECT format
        2. Keyword outside <reasoning> tags (avoids what-if scenarios)
        3. AI Judge Classifier (LLM zero-shot intent extraction)
        4. Human judge fallback / review (if self.judge_mode is True)
        """
        proposed: str | None = None
        ai_note: str | None = None

        # 1. Check for explicit DECISION: format
        match = re.search(r"DECISION:\s*(COOPERATE|DEFECT)", text, re.IGNORECASE)
        if match:
            proposed = match.group(1).upper()
        else:
            # 2. Strip reasoning block, then look for keywords
            clean = re.sub(
                r"<reasoning>.*?</reasoning>", "", text,
                flags=re.DOTALL | re.IGNORECASE,
            )
            clean_upper = clean.upper()

            if "COOPERATE" in clean_upper and "DEFECT" not in clean_upper:
                proposed = "COOPERATE"
            elif "DEFECT" in clean_upper and "COOPERATE" not in clean_upper:
                proposed = "DEFECT"

        # 3. AI Classifier fallback if regex and keyword parsing couldn't determine a move
        if not proposed:
            ai_choice = self._ai_judge_classify(text)
            if ai_choice:
                proposed = ai_choice
                ai_note = f"🤖 AI Judge classified intent as: {ai_choice}"

        # If Judge Mode (-j) is active, ask human judge to approve or override
        if self.judge_mode:
            return self._human_judge(text, proposed=proposed)

        # Automatic mode (no -j): return proposed move or auto-DEFECT if unresolvable
        if proposed:
            note = ai_note or f"Parsed DECISION: {proposed}"
            return proposed, note

        # Fallback when all 3 layers fail in automatic mode: default to DEFECT (pessimistic)
        return "DEFECT", "Auto-DEFECT (unparsed output)"

    def _human_judge(self, text: str, proposed: str | None = None) -> tuple[str, str]:
        """Manual override / review when requested or when model fails to state a clear choice."""
        print(f"\n--- ⚖️ JUDGE NEEDED for {self.name} ---")
        print(text[:600])
        if proposed:
            print(f"\n[Proposed LLM Decision: {proposed}]")

        while True:
            try:
                if proposed:
                    prompt_str = f"Enter to (A)pprove [{proposed}], (C)ooperate, (D)efect, or (R)etry: "
                else:
                    prompt_str = "Enter (C)ooperate, (D)efect, or (R)etry: "

                choice = input(prompt_str).strip().upper()
            except EOFError:
                fallback = proposed or "DEFECT"
                return fallback, f"Auto-{fallback} (non-interactive)"

            if choice in ("", "A") and proposed:
                return proposed, f"Human approved proposed: {proposed}"
            if choice == "C":
                return "COOPERATE", "Human judge: COOPERATE"
            if choice == "D":
                return "DEFECT", "Human judge: DEFECT"
            if choice == "R":
                return self.decide(retry=True), "Retried by human"

    # ── State update ──────────────────────────────────────────────────────

    def update(self, my_move: str, opponent_move: str, reward: int):
        """Record what happened this round."""
        self.history.append((my_move, opponent_move, reward))
        self.score += reward

    # ── Accessors ─────────────────────────────────────────────────────────

    def get_last_reasoning(self) -> str:
        """Return the most recent reasoning text, stripped of XML tags."""
        if not self.reasoning_log:
            return "No reasoning yet."
        return clean_reasoning_text(self.reasoning_log[-1])

    def get_last_judge_note(self) -> str:
        if not self.judge_log:
            return ""
        return self.judge_log[-1]