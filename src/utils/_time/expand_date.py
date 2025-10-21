import re
from datetime import date, timedelta
from typing import List, Tuple

DAY_PAT = re.compile(r"^\d{4}-\d{2}-\d{2}$")
MONTH_PAT = re.compile(r"^\d{4}-\d{2}$")

def _parse_day(s: str) -> date:
    """Parse YYYY-MM-DD into a date object with a helpful error message."""
    try:
        y, m, d = map(int, s.split("-"))
        return date(y, m, d)
    except Exception as e:
        raise ValueError(f"Invalid day string '{s}'. Expected 'YYYY-MM-DD'.") from e

def _parse_month(s: str) -> Tuple[int, int]:
    """Parse YYYY-MM into (year, month) with a helpful error message."""
    try:
        y, m = map(int, s.split("-"))
        if not (1 <= m <= 12):
            raise ValueError
        return y, m
    except Exception as e:
        raise ValueError(f"Invalid month string '{s}'. Expected 'YYYY-MM'.") from e

def _iterate_months(y1: int, m1: int, y2: int, m2: int) -> List[str]:
    """Inclusive month iterator from (y1,m1) to (y2,m2)."""
    # Normalize order
    if (y1, m1) > (y2, m2):
        y1, m1, y2, m2 = y2, m2, y1, m1

    out = []
    y, m = y1, m1
    while (y < y2) or (y == y2 and m <= m2):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            m = 1
            y += 1
    return out

def expand_date_or_month_range(start: str, end: str) -> List[str]:
    """
    Return an inclusive list of either days or months, depending on the input format.
    - If both inputs are 'YYYY-MM-DD', returns all dates in that range (inclusive).
    - If both inputs are 'YYYY-MM',    returns all months in that range (inclusive).
    - Mixed granularities are not allowed.
    """
    start = start.strip()
    end = end.strip()

    is_day = DAY_PAT.match(start) and DAY_PAT.match(end)
    is_month = MONTH_PAT.match(start) and MONTH_PAT.match(end)

    if not (is_day or is_month):
        raise ValueError("Both inputs must be either 'YYYY-MM-DD' or 'YYYY-MM'. Mixed granularities are not supported.")

    if is_day:
        d1, d2 = _parse_day(start), _parse_day(end)
        # Normalize order
        if d1 > d2:
            d1, d2 = d2, d1
        out = []
        cur = d1
        while cur <= d2:
            out.append(cur.isoformat())
            cur += timedelta(days=1)
        return out

    # month mode
    y1, m1 = _parse_month(start)
    y2, m2 = _parse_month(end)
    return _iterate_months(y1, m1, y2, m2)
