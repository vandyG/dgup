from __future__ import annotations

# ---------------------------------------------------------------------------
# Column name constants
# ---------------------------------------------------------------------------
COL_DATE = "Date"
COL_NOM = "Nom"
COL_DELIVERY = "Delivery"
COL_USAGE_1 = "Usage - 1"
COL_USAGE_2 = "Usage - 2"
COL_USAGE_2_1 = "Usage - 2_1"
COL_TOTAL = "total_usage"
COL_NET_FLOW = "net_flow"
COL_INJECTION = "injection"
COL_WITHDRAWAL = "withdrawal"
COL_INVENTORY = "inventory"
COL_DAILY_PENALTY = "daily_penalty"
COL_MONTHLY_PENALTY = "monthly_penalty"

# ---------------------------------------------------------------------------
# Storage bank parameters
# ---------------------------------------------------------------------------
INITIAL_INVENTORY: float = 81569.123976  # therms; starting balance before data begins
STORAGE_CAPACITY: float = 144841.0  # therms; total SBS capacity

# ---------------------------------------------------------------------------
# Forecast horizon
# ---------------------------------------------------------------------------
FORECAST_HORIZON: int = 7  # days

# ---------------------------------------------------------------------------
# Daily storage activity limits (fraction of STORAGE_CAPACITY per day)
# Index 1..12 → Jan..Dec
# ---------------------------------------------------------------------------
MAX_INJECTION_PCT: dict[int, float] = {
    1: 0.0030,
    2: 0.0030,
    3: 0.0030,
    4: 0.0030,
    5: 0.0045,
    6: 0.0050,
    7: 0.0045,
    8: 0.0070,
    9: 0.0070,
    10: 0.0070,
    11: 0.0030,
    12: 0.0030,
}

MAX_WITHDRAWAL_PCT: dict[int, float] = {
    1: 0.0100,
    2: 0.0085,
    3: 0.0060,
    4: 0.0030,
    5: 0.0030,
    6: 0.0030,
    7: 0.0030,
    8: 0.0030,
    9: 0.0030,
    10: 0.0030,
    11: 0.0040,
    12: 0.0085,
}

# Precomputed absolute daily limits (therms)
MAX_INJECTION: dict[int, float] = {m: pct * STORAGE_CAPACITY for m, pct in MAX_INJECTION_PCT.items()}
MAX_WITHDRAWAL: dict[int, float] = {m: pct * STORAGE_CAPACITY for m, pct in MAX_WITHDRAWAL_PCT.items()}

# ---------------------------------------------------------------------------
# Month-end inventory bands (fraction of STORAGE_CAPACITY)
# Index 1..12 → Jan..Dec
# ---------------------------------------------------------------------------
MONTHLY_MIN_PCT: dict[int, float] = {
    1: 0.35,
    2: 0.10,
    3: 0.00,
    4: 0.00,
    5: 0.10,
    6: 0.20,
    7: 0.30,
    8: 0.50,
    9: 0.70,
    10: 0.85,
    11: 0.75,
    12: 0.55,
}

MONTHLY_MAX_PCT: dict[int, float] = {
    1: 0.45,
    2: 0.25,
    3: 0.10,
    4: 0.10,
    5: 0.20,
    6: 0.30,
    7: 0.40,
    8: 0.60,
    9: 0.80,
    10: 1.00,
    11: 0.90,
    12: 0.70,
}

# Precomputed absolute month-end inventory bands (therms)
MONTHLY_MIN: dict[int, float] = {m: pct * STORAGE_CAPACITY for m, pct in MONTHLY_MIN_PCT.items()}
MONTHLY_MAX: dict[int, float] = {m: pct * STORAGE_CAPACITY for m, pct in MONTHLY_MAX_PCT.items()}
