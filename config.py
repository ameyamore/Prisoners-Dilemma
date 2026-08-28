"""
Configuration constants for the Prisoner's Dilemma LLM Arena.

Centralises all magic numbers, URLs, and defaults so they can be
changed in one place or overridden via CLI arguments.
"""

# ── Ollama connection ─────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"

# ── Game defaults ─────────────────────────────────────────────────────────────
DEFAULT_MODEL = "qwen3:8b"
DEFAULT_ROUNDS = 20
DEFAULT_DELAY = 2.0
DEFAULT_DIALOGUE_TURNS = 4

# ── Payoff matrix ─────────────────────────────────────────────────────────────
# Keys: (my_move, opponent_move)  →  Values: (my_reward, opponent_reward)
PAYOFF_MATRIX = {
    ("COOPERATE", "COOPERATE"): (3, 3),
    ("COOPERATE", "DEFECT"):    (0, 5),
    ("DEFECT",    "COOPERATE"): (5, 0),
    ("DEFECT",    "DEFECT"):    (-1, -1),
}

# ── LLM parameters ───────────────────────────────────────────────────────────
DECIDE_TEMPERATURE = 0.2     # Low for analytical decisions
DIALOGUE_TEMPERATURE = 0.7   # Natural conversation without rambling drafts
DECIDE_MAX_TOKENS = 2000     # Chain-of-thought plus DECISION line
DIALOGUE_MAX_TOKENS = 1200  # Room for hidden thinking plus a full spoken reply
PREGAME_MAX_TOKENS = 2500    # Full pregame plan including expected-score math
DEFAULT_NUM_CTX = 16384      # Prompt + generation share this window; keep history from being truncated
REQUEST_TIMEOUT = 300        # seconds


