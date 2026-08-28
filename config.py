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
DIALOGUE_TEMPERATURE = 0.8   # Higher for natural conversation
DECIDE_MAX_TOKENS = 1500     # Ample tokens for deep chain-of-thought reasoning + decision
DIALOGUE_MAX_TOKENS = 800    # Enough tokens for full 7-8 sentence speeches
PREGAME_MAX_TOKENS = 1000    # Enough tokens for comprehensive pre-game planning
DEFAULT_NUM_CTX = 16384      # Prompt + generation share this window; keep history from being truncated
REQUEST_TIMEOUT = 300        # seconds


