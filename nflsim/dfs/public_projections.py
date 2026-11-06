"""Free public fantasy-point projection sources used to build an independent "field"
mean for the DFS optimizer, instead of reusing our own simulation means for both sides.

All sources are unofficial/undocumented hidden APIs or scraped pages (no key required, but
the response schema can change without notice). Every fetch is best-effort: network or
schema errors return an empty Series rather than raising, so the caller can fall back to
real-sim means.

Verified against live responses on 2026-09-07 / 2026-09-10:
- Sleeper's projections endpoint embeds full player identity + ready-made fantasy point
  totals (`pts_ppr`/`pts_half_ppr`/`pts_std`) -- reliable source.
- RotoWire's optimizer backs onto `/daily/nfl/api/slate-list.php?siteID=1` (1 = DraftKings)
  and `/daily/nfl/api/players.php?slateID=...`, which returns each player's DK salary and
  DK-scored projection (`pts`) plus projected rostership. No login needed. (The separate
  `projections.php` endpoint is keyed by an internal slate id and is paywalled.)
- DailyFantasyFuel's projections page embeds every row's numbers in `data-*` attributes
  (`data-ppg_proj` is the DK-format projection); `/nfl/projections/draftkings/<YYYY-MM-DD>/`
  returns that date's full slate, while the bare URL only returns the next slate.
- ESPN's public (no-league) players endpoint is reachable but only returned raw per-statId
  values, not a ready "appliedTotal" fantasy total, for the views tried -- computing an
  ESPN total would require ESPN's internal statId->points weight table, which isn't
  implemented here. `fetch_espn_projections` is wired up to use `appliedTotal` if a future
  view/response includes it, but will return empty in practice until that's added.
"""

from __future__ import annotations

import html
import json
import re
from datetime import date as calendar_date
from typing import Optional

import pandas as pd
import requests

_REQUEST_TIMEOUT = 15

_SLEEPER_PROJECTIONS_URL = "https://api.sleeper.app/projections/nfl/{season}/{week}"
_ESPN_PLAYERS_URL = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{season}/players"
_ESPN_POSITION_MAP = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "DST"}

_ROTOWIRE_SLATES_URL = "https://www.rotowire.com/daily/nfl/api/slate-list.php"
_ROTOWIRE_PLAYERS_URL = "https://www.rotowire.com/daily/nfl/api/players.php"
_ROTOWIRE_DK_SITE_ID = 1
_DFF_PROJECTIONS_URL = "https://www.dailyfantasyfuel.com/nfl/projections/{site}/{date}"
_DFF_ROW_PATTERN = re.compile(r"<tr[^>]*class=\"[^\"]*projections-listing[^\"]*\"([^>]*)>")
_DFF_ATTR_PATTERN = re.compile(r"data-([a-z_0-9]+)=\s*\"([^\"]*)\"")
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
    ),
    "Accept": "application/json, text/html;q=0.9, */*;q=0.8",
}


def _short_name(first: str, last: str, position: str, team: str) -> Optional[str]:
    """Match `dfs_optimizer`'s short_name convention (TEAM.Kicker/TEAM.Defense or F.Last)."""
    if not team:
        return None
    if position in ("K", "DST"):
        label = "Kicker" if position == "K" else "Defense"
        return f"{team}.{label}"
    first = html.unescape(first or "").strip()
    last = html.unescape(last or "").strip()
    if not first or not last:
        return None
    # DK's key is Name.split()[1], so a multi-token surname ("St. Brown") keys off "St.".
    return f"{first[:3]}.{last.split()[0]}"


def fetch_sleeper_projections(season: int, week: int) -> pd.Series:
    """Points-per-player Series keyed by short_name; empty on any failure."""
    try:
        response = requests.get(
            _SLEEPER_PROJECTIONS_URL.format(season=season, week=week),
            params={"season_type": "regular"},
            timeout=_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        rows = response.json() or []
    except (requests.RequestException, ValueError):
        return pd.Series(dtype=float, name="sleeper")

    records = {}
    for row in rows:
        player = row.get("player") or {}
        stats = row.get("stats") or {}
        points = stats.get("pts_ppr")
        if points is None:
            continue
        name = _short_name(player.get("first_name", ""), player.get("last_name", ""), player.get("position"), player.get("team"))
        if name:
            records[name] = points
    return pd.Series(records, dtype=float, name="sleeper")


def fetch_espn_projections(season: int, week: int) -> pd.Series:
    """Points-per-player Series keyed by short_name; empty on any failure or if the
    response doesn't include a usable projected point total (see module docstring)."""
    try:
        espn_filter = {
            "players": {
                "limit": 600,
                "filterStatsForTopScoringPeriodIds": {
                    "value": 16,
                    "additionalValue": [f"00{season}", f"10{season}"],
                },
            }
        }
        headers = {"x-fantasy-filter": json.dumps(espn_filter)}
        response = requests.get(
            _ESPN_PLAYERS_URL.format(season=season),
            params={"scoringPeriodId": week, "view": "kona_player_info"},
            headers=headers,
            timeout=_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        rows = response.json() or []
    except (requests.RequestException, ValueError):
        return pd.Series(dtype=float, name="espn")

    records = {}
    for row in rows:
        position = _ESPN_POSITION_MAP.get(row.get("defaultPositionId"))
        if not position:
            continue
        applied_total = None
        for stat_entry in row.get("stats", []):
            if stat_entry.get("statSourceId") == 1 and stat_entry.get("scoringPeriodId") == week:
                applied_total = stat_entry.get("appliedTotal")
                break
        if applied_total is None:
            continue
        team = str(row.get("proTeamId", ""))
        name = _short_name(row.get("firstName", ""), row.get("lastName", ""), position, team)
        if name:
            records[name] = applied_total
    return pd.Series(records, dtype=float, name="espn")


def fetch_rotowire_slates(site_id: int = _ROTOWIRE_DK_SITE_ID) -> pd.DataFrame:
    """DK slates RotoWire currently has projections for; empty frame on any failure."""
    try:
        response = requests.get(
            _ROTOWIRE_SLATES_URL,
            params={"siteID": site_id},
            headers=_BROWSER_HEADERS,
            timeout=_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        slates = (response.json() or {}).get("slates") or []
    except (requests.RequestException, ValueError, AttributeError):
        return pd.DataFrame(columns=["slateID", "contestType", "slateName", "startDate"])

    frame = pd.DataFrame(slates)
    if frame.empty:
        return pd.DataFrame(columns=["slateID", "contestType", "slateName", "startDate"])
    return frame[["slateID", "contestType", "slateName", "startDate"]].sort_values("startDate")


def fetch_rotowire_projections(slate_id: int) -> pd.Series:
    """RotoWire's DK-scored projection for every player on a slate, keyed by short_name."""
    try:
        response = requests.get(
            _ROTOWIRE_PLAYERS_URL,
            params={"slateID": int(slate_id)},
            headers=_BROWSER_HEADERS,
            timeout=_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        rows = response.json() or []
    except (requests.RequestException, ValueError, TypeError):
        return pd.Series(dtype=float, name="rotowire")

    records = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            points = float(row.get("pts"))
        except (TypeError, ValueError):
            continue
        position = "DST" if row.get("isTeamDefense") else row.get("rotoPos")
        team = (row.get("team") or {}).get("abbr", "")
        name = _short_name(row.get("firstName", ""), row.get("lastName", ""), position, team)
        if name:
            records[name] = points
    return pd.Series(records, dtype=float, name="rotowire")


def fetch_dailyfantasyfuel_projections(date: Optional[str] = None, site: str = "draftkings") -> pd.Series:
    """DailyFantasyFuel's DK projections (`data-ppg_proj`), keyed by short_name.

    `date` is a YYYY-MM-DD slate date; without it the page only returns the next slate.
    On the current date, DailyFantasyFuel exposes the main slate at the dated URL and
    the evening slate at the bare URL, so combine both pages for the dashboard's daily
    view."""
    pages = [date or ""]
    if date == calendar_date.today().isoformat():
        pages.append("")

    records = {}
    for page_date in pages:
        try:
            response = requests.get(
                _DFF_PROJECTIONS_URL.format(site=site, date=page_date),
                headers=_BROWSER_HEADERS,
                timeout=_REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            page = response.text
        except requests.RequestException:
            continue

        for attribute_blob in _DFF_ROW_PATTERN.findall(page):
            attributes = dict(_DFF_ATTR_PATTERN.findall(attribute_blob))
            try:
                points = float(attributes.get("ppg_proj", ""))
            except ValueError:
                continue
            name = _short_name(
                attributes.get("fn", ""),
                attributes.get("ln", ""),
                attributes.get("pos", ""),
                attributes.get("team", ""),
            )
            if name:
                records[name] = points
    return pd.Series(records, dtype=float, name="dailyfantasyfuel")


def blend_projections(*sources: pd.Series) -> pd.Series:
    """Average whichever projection sources returned data; empty Series if all failed."""
    usable = [s for s in sources if s is not None and not s.empty]
    if not usable:
        return pd.Series(dtype=float, name="blended")
    return pd.concat(usable, axis=1).mean(axis=1).rename("blended")
