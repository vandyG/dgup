from __future__ import annotations

from dataclasses import dataclass

_STORAGE_CAPACITY = 144_841.0
_INITIAL_GAS_IN_BANK = 81_569.123976

_MONTH_ORDER = tuple(range(1, 13))

_MAX_INJECTION_RATES = {
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

_MAX_WITHDRAWAL_RATES = {
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

_MIN_INVENTORY_RATES = {
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

_MAX_INVENTORY_RATES = {
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


@dataclass(frozen=True)
class _TariffMonth:
    month: int
    max_injection_rate: float
    max_withdrawal_rate: float
    min_inventory_rate: float
    max_inventory_rate: float


def _validate_month(month: int) -> None:
    if month not in _MONTH_ORDER:
        msg = f"month must be between 1 and 12, got {month}"
        raise ValueError(msg)


def _tariff_month(month: int) -> _TariffMonth:
    _validate_month(month)
    return _TariffMonth(
        month=month,
        max_injection_rate=_MAX_INJECTION_RATES[month],
        max_withdrawal_rate=_MAX_WITHDRAWAL_RATES[month],
        min_inventory_rate=_MIN_INVENTORY_RATES[month],
        max_inventory_rate=_MAX_INVENTORY_RATES[month],
    )


def _daily_storage_limits(month: int, capacity: float = _STORAGE_CAPACITY) -> tuple[float, float]:
    tariff_month = _tariff_month(month)
    return tariff_month.max_injection_rate * capacity, tariff_month.max_withdrawal_rate * capacity


def _month_end_inventory_limits(month: int, capacity: float = _STORAGE_CAPACITY) -> tuple[float, float]:
    tariff_month = _tariff_month(month)
    return tariff_month.min_inventory_rate * capacity, tariff_month.max_inventory_rate * capacity
