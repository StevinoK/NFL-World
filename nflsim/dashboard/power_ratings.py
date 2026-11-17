"""Outside team ratings, in points against an average team, from the sources that publish them openly.

FPI, nfelo, Inpredictable, Sagarin and TeamRankings all rate teams in expected points on a neutral field. Each
source is centered on its league average so the columns compare directly. A source that fails or doesn't return
all 32 teams is left out and noted, without affecting the others. FTN, PFF and Unexpected Points need a
subscription, so they aren't here.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import pandas as pd
import requests

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126 Safari/537.36"}
RATINGS_FILE = "power_ratings.csv"
NOTES_FILE = "power_ratings.json"
# Team: nickname and how TeamRankings writes it.
TEAMS = {
    "ARI": ("Cardinals", "Arizona"), "ATL": ("Falcons", "Atlanta"), "BAL": ("Ravens", "Baltimore"), "BUF": ("Bills", "Buffalo"),
    "CAR": ("Panthers", "Carolina"), "CHI": ("Bears", "Chicago"), "CIN": ("Bengals", "Cincinnati"), "CLE": ("Browns", "Cleveland"),
    "DAL": ("Cowboys", "Dallas"), "DEN": ("Broncos", "Denver"), "DET": ("Lions", "Detroit"), "GB": ("Packers", "Green Bay"),
    "HOU": ("Texans", "Houston"), "IND": ("Colts", "Indianapolis"), "JAX": ("Jaguars", "Jacksonville"), "KC": ("Chiefs", "Kansas City"),
    "LA": ("Rams", "LA Rams"), "LAC": ("Chargers", "LA Chargers"), "LV": ("Raiders", "Las Vegas"), "MIA": ("Dolphins", "Miami"),
    "MIN": ("Vikings", "Minnesota"), "NE": ("Patriots", "New England"), "NO": ("Saints", "New Orleans"), "NYG": ("Giants", "NY Giants"),
    "NYJ": ("Jets", "NY Jets"), "PHI": ("Eagles", "Philadelphia"), "PIT": ("Steelers", "Pittsburgh"), "SEA": ("Seahawks", "Seattle"),
    "SF": ("49ers", "San Francisco"), "TB": ("Buccaneers", "Tampa Bay"), "TEN": ("Titans", "Tennessee"), "WAS": ("Commanders", "Washington"),
}
ABBREVIATIONS = {"ARZ": "ARI", "JAC": "JAX", "LAR": "LA", "OAK": "LV", "WSH": "WAS"}
BY_NICKNAME = {nickname: team for team, (nickname, _) in TEAMS.items()}
BY_PLACE = {place: team for team, (_, place) in TEAMS.items()}
NUMBER = re.compile(r"^[+-]?\d+\.\d+$")


class _Tables(HTMLParser):
    """Rows of every table on a page, as text."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: List[List[List[str]]] = []
        self.row = None
        self.cell = None

    def handle_starttag(self, tag, attrs) -> None:
        if tag == "table":
            self.tables.append([])
        elif tag == "tr":
            self.row = []
        elif tag in ("td", "th"):
            self.cell = []

    def handle_endtag(self, tag) -> None:
        if tag in ("td", "th") and self.cell is not None and self.row is not None:
            self.row.append(" ".join(" ".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.row is not None and self.tables:
            self.tables[-1].append(self.row)
            self.row = None

    def handle_data(self, data) -> None:
        if self.cell is not None:
            self.cell.append(data)


def our_team(code: str) -> str:
    """The code a team has in TEAMS, whichever spelling a source or sim run uses."""
    return ABBREVIATIONS.get(code, code)


def largest_table(html: str) -> List[List[str]]:
    parser = _Tables()
    parser.feed(html)
    return max(parser.tables, key=len) if parser.tables else []


def _get(url: str) -> requests.Response:
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response


def parse_fpi(payload: dict) -> Tuple[Dict[str, float], str]:
    ratings = {}
    for row in payload["teams"]:
        abbreviation = row["team"]["abbreviation"]
        category = next(item for item in row["categories"] if item["name"] == "fpi")
        ratings[ABBREVIATIONS.get(abbreviation, abbreviation)] = float(category["values"][0])
    return ratings, str(payload.get("lastUpdated", ""))[:10]


def parse_nfelo(html: str) -> Tuple[Dict[str, float], str]:
    rows = largest_table(html)
    header = next(row for row in rows if "Value" in row)
    column = header.index("Value")
    ratings = {}
    for row in rows:
        match = re.search(r"\b([A-Z]{2,3}) \d+-\d+", row[1]) if len(row) > column else None
        if match:
            ratings[ABBREVIATIONS.get(match.group(1), match.group(1))] = float(row[column])
    return ratings, ""


def parse_inpredictable(html: str) -> Tuple[Dict[str, float], str]:
    ratings = {}
    for row in largest_table(html):
        if len(row) > 2 and row[0].isdigit():
            # GPF is the first plain number after the team; sparkline cells hold comma-separated lists.
            ratings[ABBREVIATIONS.get(row[1], row[1])] = float(next(cell for cell in row[2:] if NUMBER.match(cell)))
    stamp = re.search(r"As of ([A-Za-z]+ \d{1,2}, \d{4})", html)
    return ratings, dt.datetime.strptime(stamp.group(1), "%B %d, %Y").date().isoformat() if stamp else ""


def parse_teamrankings(html: str) -> Tuple[Dict[str, float], str]:
    ratings = {}
    for row in largest_table(html):
        place = re.sub(r"\s*\(.*\)$", "", row[1]) if len(row) > 2 else ""
        if place in BY_PLACE:
            ratings[BY_PLACE[place]] = float(row[2])
    return ratings, ""


def parse_sagarin(html: str) -> Tuple[Dict[str, float], str]:
    text = re.sub(r"<[^>]+>", " ", html)
    ratings: Dict[str, float] = {}
    # The ranked list is printed more than once; the first printing holds every team.
    for line in re.findall(r"^\s*\d+\s+[A-Z0-9].*=.*\|.*$", text, re.M):
        name, parts = line.split("=")[0].split(), line.split("|")
        team = {"Redskins": "WAS"}.get(name[-1]) or BY_NICKNAME.get(name[-1]) or BY_PLACE.get(" ".join(name[1:]))
        if team and team not in ratings and len(parts) > 2:
            ratings[team] = float(parts[2].split()[0])
    stamp = re.search(r"through games of ([A-Za-z]+ \d{1,2})[^\n]*", text)
    return ratings, stamp.group(0).replace("through games of ", "through ").strip() if stamp else ""


@dataclass(frozen=True)
class Source:
    label: str
    page: str
    pull: Callable[[], Tuple[Dict[str, float], str]]


SOURCES = (
    Source("FPI", "https://www.espn.com/nfl/fpi", lambda: parse_fpi(_get(
        "https://site.web.api.espn.com/apis/fitt/v3/sports/football/nfl/powerindex?limit=32&lang=en&region=us").json())),
    Source("nfelo", "https://www.nfeloapp.com/nfl-power-ratings/", lambda: parse_nfelo(_get("https://www.nfeloapp.com/nfl-power-ratings/").text)),
    Source("Inpredictable", "https://stats.inpredictable.com/rankings/nfl.php", lambda: parse_inpredictable(_get("https://stats.inpredictable.com/rankings/nfl.php").text)),
    # Sagarin's HTTPS certificate is self-signed, so its page is read over plain HTTP.
    Source("Sagarin", "http://sagarin.com/sports/nflsend.htm", lambda: parse_sagarin(_get("http://sagarin.com/sports/nflsend.htm").text)),
    Source("TeamRankings", "https://www.teamrankings.com/nfl/ranking/predictive-by-other", lambda: parse_teamrankings(_get("https://www.teamrankings.com/nfl/ranking/predictive-by-other").text)),
)


def _pull(source: Source) -> Tuple[Dict[str, float], str, str]:
    try:
        ratings, as_of = source.pull()
    except Exception as error:
        return {}, "", f"{type(error).__name__}: {error}"[:200]
    if set(ratings) != set(TEAMS):
        return {}, "", f"returned {len(ratings)} of {len(TEAMS)} teams"
    return ratings, as_of, ""


def fetch_ratings(sources=SOURCES) -> Tuple[pd.DataFrame, dict]:
    """Ratings by team and source, centered on the league average, plus notes on where they came from."""
    with ThreadPoolExecutor(len(sources)) as pool:
        results = list(pool.map(_pull, sources))
    columns, used, failed = {}, [], []
    for source, (ratings, as_of, error) in zip(sources, results):
        if error:
            failed.append({"label": source.label, "error": error})
            continue
        series = pd.Series(ratings, dtype=float)
        columns[source.label] = series - series.mean()
        used.append({"label": source.label, "page": source.page, "as_of": as_of})
    frame = pd.DataFrame(columns, index=sorted(TEAMS))
    frame.index.name = "team"
    return frame, {"pulled": dt.date.today().isoformat(), "sources": used, "failed": failed}


def save_ratings(folder: Path, sources=SOURCES) -> Tuple[pd.DataFrame, dict]:
    ratings, notes = fetch_ratings(sources)
    ratings.to_csv(folder / RATINGS_FILE)
    (folder / NOTES_FILE).write_text(json.dumps(notes, indent=2), encoding="utf-8")
    return ratings, notes


def load_ratings(folder: Path) -> Tuple[pd.DataFrame, dict]:
    return pd.read_csv(folder / RATINGS_FILE, index_col="team"), json.loads((folder / NOTES_FILE).read_text(encoding="utf-8"))
