"""Player display names: the nickname a player goes by plus last name ("Pat Mahomes", "Cam Ward").

nflverse rosters carry ``football_name`` (the name used on broadcasts), ``first_name`` (legal),
``last_name`` and ``player_name``; in current-season weekly rosters ``player_name`` holds only the
first name, so it is a fallback, never the display name.
"""
from __future__ import annotations

import pandas as pd


def _text(value) -> str:
    return "" if value is None or pd.isna(value) else str(value).strip()


def display_name(football_name=None, last_name=None, first_name=None, player_name=None, fallback="") -> str:
    first = _text(football_name) or _text(first_name)
    last = _text(last_name)
    if first and last:
        return f"{first} {last}"
    return _text(player_name) or first or last or str(fallback)


def row_display_name(row) -> str:
    """``display_name`` for a roster record (namedtuple, Series or dict)."""
    get = row.get if isinstance(row, (dict, pd.Series)) else (lambda column: getattr(row, column, None))
    return display_name(get("football_name"), get("last_name"), get("first_name"), get("player_name"),
                        get("player_id") or get("gsis_id") or "")


def roster_display_names(roster: pd.DataFrame) -> dict[str, str]:
    """player_id -> display name, preferring each player's most recent roster record."""
    identity = "player_id" if "player_id" in roster else "gsis_id"
    rows = roster.dropna(subset=[identity])
    order = [column for column in ("season", "week") if column in rows]
    if order:
        rows = rows.sort_values(order)
    rows = rows.drop_duplicates(identity, keep="last")
    return {str(row[identity]): row_display_name(row) for row in rows.to_dict("records")}
