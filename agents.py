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
    include_thinking: bool = True,
    think: bool | None = None,
) -> str:
    """
    Send a chat request to the local Ollama server and return the
    assistant's reply text. Retries on empty responses or transient errors,
    and seamlessly captures both content and thinking blocks.
    """
    for attempt in range(retries + 1):
        try:
            payload: dict = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "num_ctx": num_ctx,
                },
            }
            if think is not None:
                payload["think"] = think

            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            msg = data.get("message", {})
            content = msg.get("content", "").strip()
            thinking = msg.get("thinking", "").strip()

            if not include_thinking:
                visible = extract_spoken_text(content) or extract_spoken_text(thinking)
                if not visible:
                    visible = extract_spoken_text(f"{thinking}\n{content}".strip())
                if visible:
                    return visible
                if attempt < retries:
                    time.sleep(1)
                    continue
                return ""

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


_META_DIALOGUE_RE = re.compile(
    r"(thinking process|analyze the request|drafting the|draft \d|"
    r"sentence count|count check|constraints?:|analyze the game|"
    r"let me (work|carefully|draft)|sentence \d+\.|exactly 7)",
    re.IGNORECASE,
)


def extract_spoken_text(text: str) -> str:
    """Keep only words meant to be spoken; drop CoT, drafts, and sentence-counting."""
    if not text:
        return ""

    tagged = re.search(r"<speech>(.*?)</speech>", text, re.DOTALL | re.IGNORECASE)
    if tagged:
        speech = tagged.group(1).strip()
        return speech if speech else ""

    marked = re.search(r"(?:^|\n)\s*SPEECH:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if marked:
        speech = marked.group(1).strip()
        speech = re.sub(r"</?speech>", "", speech, flags=re.IGNORECASE).strip()
        if speech and not _META_DIALOGUE_RE.search(speech.split("\n")[0]):
            return speech

    # Drop hidden / tagged thought blocks entirely (do not promote them to speech)
    stripped = re.sub(
        r"<(reasoning|thinking|think)>.*?</\1>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    stripped = re.sub(r"</?(reasoning|thinking|think|speech)>", "", stripped, flags=re.IGNORECASE)
    stripped = stripped.strip()

    if not stripped or _META_DIALOGUE_RE.search(stripped):
        return ""
    return stripped


# ── System prompt (num_rounds is injected at runtime) ────────────────────────

def build_system_prompt(num_rounds: int = 20) -> str:
    return f"""You are an elite, highly competitive game theorist. Your absolute priority is to WIN by MAXIMIZING YOUR TOTAL SCORE and outscoring your one opponent across all {num_rounds} rounds.

This is a STRICT TWO-PLAYER game. There is only you and one opponent. There are no other players, no crowd, and no battle royale. Never talk as if "everyone else" is playing.

You and your opponent make your decisions simultaneously in total isolation. Each round, you independently choose to COOPERATE or DEFECT.

### PAYOFF MATRIX (read BOTH columns every time)
| Your Choice  | Their Choice | YOUR points | THEIR points | What happened |
| :----------- | :----------- | :---------- | :----------- | :------------ |
| COOPERATE    | COOPERATE    | +3          | +3           | Mutual cooperation |
| COOPERATE    | DEFECT       | 0           | +5           | You were exploited: you 0, they +5 |
| DEFECT       | COOPERATE    | +5          | 0            | You exploited them: you +5, they 0 |
| DEFECT       | DEFECT       | -1          | -1           | Mutual defection |

Worked example: if YOU played DEFECT and THEY played COOPERATE, then YOU scored +5 and THEY scored 0. They did NOT also get +5. Only the defector gets +5; the cooperator gets 0.

### HIGH-STAKES RULES & COMPETITIVE URGENCY
- URGENCY TO WIN: You are fiercely driven to finish with the highest possible score. Every single round impacts your final standing.
- When you defect and they cooperate, record it as +5 for you and 0 for them — never as +5/+5.
- Relative standing matters: being exploited (you 0, they +5) widens the gap against you. Mutual defection is -1 each.
- AVOID MUTUAL DEFECTION TRAPS: Constant mutual defection (-1/-1) drains your score. Sustainable mutual cooperation (+3/+3) builds point totals over time, but always remain vigilant.
- THE SHADOW OF THE FUTURE: The game lasts exactly {num_rounds} rounds. Every turn you will be given an AUTHORITATIVE ROUND CLOCK (current round and rounds remaining). Use that clock, not any earlier notes, to time endgame defection.

### YOUR REASONING PROCESS
You MUST structure your thoughts inside <reasoning> and </reasoning> tags following these 4 steps:

1. PATTERN RECOGNITION — What is the opponent's strategy? (Tit-for-Tat, Grim Trigger, Exploitative, Naive, Forgiving?)
2. PREDICTION — What move do you predict they will make THIS round, and why?
3. COUNTER-STRATEGY — What move delivers the highest expected score and strongest competitive advantage?
4. RISK ASSESSMENT — If your prediction fails, what is the damage? How do you protect against catastrophic losses?

After closing </reasoning>, output exactly one line (fill in only one word, do not print both options):
DECISION: <COOPERATE or DEFECT>"""


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
        self.history: list[tuple[str, str, int, int]] = []  # (my_move, opp_move, my_reward, opp_reward)
        self.reasoning_log: list[str] = []               # full model output per round
        self.judge_log: list[str] = []                   # judge interpretation per round
        self.prompt_clock_log: list[str] = []            # authoritative clock actually sent each round
        self.pregame_thoughts: str = ""
        self._pending_round: int = 1
        self.opponent_score: int = 0

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def cooperation_rate(self) -> float:
        """Fraction of rounds where this agent cooperated (0.0–1.0)."""
        if not self.history:
            return 0.0
        return sum(1 for m, *_ in self.history if m == "COOPERATE") / len(self.history)

    @property
    def cooperations(self) -> int:
        return sum(1 for m, *_ in self.history if m == "COOPERATE")

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
        opp_coops = sum(1 for _, opp, *_ in self.history if opp == "COOPERATE")
        opp_rate = round((opp_coops / total) * 100)

        lines.append("### GAME HISTORY (completed rounds only — this table is complete and current)")
        lines.append(f"Completed Rounds: {total} / {self.num_rounds}")
        lines.append(f"Now Deciding: Round {current_round} of {self.num_rounds}")
        lines.append(f"Opponent Cooperation Rate: {opp_rate}%")
        lines.append(f"Your Total Score: {self.score}")
        lines.append(f"Opponent Total Score: {self.opponent_score}")
        lines.append(
            "Payoff reminder: DEFECT vs COOPERATE is +5 for the defector and 0 for the cooperator, "
            "never +5 for both.\n"
        )
        header = f"{'Round':<8} {'Your move':<15} {'Opponent move':<15} {'Your points':<14} {'Opponent points'}"
        lines.append(header)
        lines.append("-" * 70)

        for i, (my_move, opp_move, my_reward, opp_reward) in enumerate(self.history, 1):
            lines.append(
                f"{i:<8} {my_move:<15} {opp_move:<15} {my_reward:<14} {opp_reward}"
            )

        return "\n".join(lines)

    # ── Pregame thinking ──────────────────────────────────────────────────

    def pregame_think(self) -> str:
        """Form a strategy before the game begins."""
        messages = [
            {"role": "system", "content": build_system_prompt(self.num_rounds)},
            {"role": "user", "content": (
                f"The game is about to start. It lasts exactly {self.num_rounds} rounds and has "
                "exactly TWO players: you and one opponent. "
                "Your objective is to win and maximize your cumulative score. "
                "Finish a complete game plan, including expected scores for the full match, "
                "inside <reasoning> tags. Do not stop mid-sentence."
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

    def _thought_conclusion(self, text: str) -> str | None:
        """Last clear I-will-cooperate/defect in the thought block, if any."""
        thought = ""
        match = re.search(
            r"<(reasoning|thinking|think)>(.*?)</\1>",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if match:
            thought = match.group(2)
        else:
            thought = text
        tail = thought[-800:] if thought else ""
        hits = [
            m.group(1).upper()
            for m in re.finditer(
                r"(?:i(?:'m going to|'ll| will)|final (?:decision|verdict|choice)\s*(?:is|:)?|"
                r"i (?:choose|choose to))\s+(?:to\s+)?(cooperate|defect)",
                tail,
                re.IGNORECASE,
            )
        ]
        return hits[-1] if hits else None

    def _ai_judge_classify(self, text: str) -> str | None:
        """
        Same local model, used only as a referee on the *end* of the monologue
        (the conclusion), never the opening 1200 characters.
        """
        tail = text[-1800:] if text else ""
        messages = [
            {
                "role": "system",
                "content": (
                    "You classify the player's FINAL intended move. "
                    "Ignore early brainstorming and copied templates. "
                    "Weight the last few sentences most.\n"
                    "Respond with ONLY ONE WORD: COOPERATE or DEFECT."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Player monologue (end of output):\n{tail}\n\n"
                    "Final intended decision (COOPERATE or DEFECT)?"
                ),
            },
        ]
        try:
            reply = ollama_chat(
                self.model, messages, temperature=0.0, max_tokens=10, think=False,
            ).upper()
            if "COOPERATE" in reply and "DEFECT" not in reply:
                return "COOPERATE"
            if "DEFECT" in reply and "COOPERATE" not in reply:
                return "DEFECT"
        except Exception:
            pass
        return None

    def _stated_decisions(self, text: str) -> list[str]:
        """All DECISION: COOPERATE/DEFECT hits, in order of appearance."""
        return [
            m.group(1).upper()
            for m in re.finditer(r"DECISION:\s*(COOPERATE|DEFECT)", text, re.IGNORECASE)
        ]

    def _extract_stated_decision(self, text: str) -> str | None:
        """
        Use the last real DECISION line, not the first.

        Models often echo the prompt examples:

            DECISION: COOPERATE
            or
            DECISION: DEFECT

        then later write the actual move. Taking the first hit would always
        pick COOPERATE. Prefer lines after the last thought block.
        """
        if not text:
            return None

        after_thoughts = re.split(
            r"</(?:reasoning|thinking|think)\s*>",
            text,
            flags=re.IGNORECASE,
        )
        tail = after_thoughts[-1] if after_thoughts else text

        for chunk in (tail, text):
            hits = self._stated_decisions(chunk)
            if not hits:
                continue
            # Drop a leading echoed "COOPERATE or DEFECT" example pair
            if (
                len(hits) >= 2
                and hits[0] == "COOPERATE"
                and hits[1] == "DEFECT"
            ):
                hits = hits[2:]
            if hits:
                return hits[-1]
        return None

    def _parse_decision(self, text: str) -> tuple[str, str]:
        """
        Extract COOPERATE or DEFECT from the model output.

        Priority:
        1. Last explicit DECISION: COOPERATE/DEFECT (ignoring echoed examples)
        2. Keyword outside thought blocks (avoids what-if scenarios)
        3. AI Judge Classifier (LLM zero-shot intent extraction)
        4. Human judge fallback / review (if self.judge_mode is True)
        """
        proposed: str | None = None
        ai_note: str | None = None

        proposed = self._extract_stated_decision(text)
        thought_end = self._thought_conclusion(text)

        # Same-model referee only when the closing thought disagrees with DECISION:
        if proposed and thought_end and proposed != thought_end:
            ai_choice = self._ai_judge_classify(text)
            if ai_choice:
                proposed = ai_choice
                ai_note = (
                    f"🤖 AI Judge resolved {thought_end} (thought) vs "
                    f"{self._extract_stated_decision(text)} (DECISION) → {ai_choice}"
                )
            else:
                proposed = thought_end
                ai_note = f"Thought conclusion overrode DECISION line: {thought_end}"

        if not proposed:
            # Strip thought blocks, then look for a one-sided keyword
            clean = re.sub(
                r"<(reasoning|thinking|think)>.*?</\1>",
                "",
                text,
                flags=re.DOTALL | re.IGNORECASE,
            )
            clean_upper = clean.upper()

            if "COOPERATE" in clean_upper and "DEFECT" not in clean_upper:
                proposed = "COOPERATE"
            elif "DEFECT" in clean_upper and "COOPERATE" not in clean_upper:
                proposed = "DEFECT"

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
            note = ai_note or f"Parsed last DECISION line: {proposed}"
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

    def update(self, my_move: str, opponent_move: str, my_reward: int, opponent_reward: int):
        """Record what happened this round, including both players' payoffs."""
        self.history.append((my_move, opponent_move, my_reward, opponent_reward))
        self.score += my_reward
        self.opponent_score += opponent_reward

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