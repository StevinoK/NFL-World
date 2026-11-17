"""What forecasts said for past open-air games, from Open-Meteo's archive of its weather models' forecasts.

Sim models train on these, not on what the weather turned out to be, because a pregame sim can only know a forecast.
The archive keeps each model's newest forecast for an hour, so these are close-to-kickoff forecasts. Nothing is archived
for 2017, and those games stay unknown. The table is kept in the repository and only grows.

    python -m nflsim.features.forecast_archive --schedule nflsim/artifacts/refresh_inputs/2026_20261101/schedule.parquet
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

from nflsim.features.forecast_weather import GAME_HOURS, STADIUMS, TIME_ZONE, at_kickoff, game_precipitation
from nflsim.features.weather import OUTDOOR_ROOFS
from nflsim.paths import ROOT

URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
TABLE = ROOT / "config/game_forecasts.csv"
FIRST_SEASON = 2018
COLUMNS = ["game_id", "temp", "wind", "precipitation"]
RETRIES, RATE_LIMIT_WAIT = 5, 70
COORDINATES = {
    **STADIUMS,
    "LAX97": (33.8644, -118.2611), "LAX99": (34.0141, -118.2879), "OAK00": (37.7516, -122.2005), "SDG00": (32.7831, -117.1196),
    "FRA00": (50.0686, 8.6455), "GER00": (48.2188, 11.6247), "LON01": (51.4560, -0.3415), "RIO00": (-22.9121, -43.2302),
    "SAO00": (-23.5453, -46.4742), "ATL97": (33.7554, -84.4008), "DAL00": (32.7473, -97.0945), "HOU00": (29.6847, -95.4107),
    "IND00": (39.7601, -86.1639), "PHO00": (33.5276, -112.2626), "MAD01": (40.4531, -3.6883),
}


def load_table(path: Path = TABLE) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"game_id": str}) if path.is_file() else pd.DataFrame(columns=COLUMNS)


def pregame_weather(frame: pd.DataFrame, path: Path = TABLE) -> pd.DataFrame:
    """Past games' temperature, wind and precipitation as forecast. Recorded weather is dropped; a game not in the table has none."""
    table = load_table(path).set_index("game_id")
    ids = frame.game_id.astype(str)
    return frame.assign(**{column: ids.map(table[column]).astype(float).to_numpy() for column in ("temp", "wind", "precipitation")})


def games_to_fetch(schedule: pd.DataFrame, known: set[str]) -> pd.DataFrame:
    """Played open-air games missing from the table, at stadiums with known coordinates."""
    roof = schedule.roof.astype("string").str.strip().str.lower()
    played = pd.to_numeric(schedule.home_score, errors="coerce").notna()
    return schedule[roof.isin(OUTDOOR_ROOFS) & played & schedule.stadium_id.isin(COORDINATES) & ~schedule.game_id.astype(str).isin(known)]


def game_rows(games: pd.DataFrame, hourly: dict) -> pd.DataFrame:
    """One row per game with complete forecast hours; a game with any hour missing is left out."""
    table = pd.DataFrame(hourly).assign(time=lambda frame: pd.to_datetime(frame.time)).set_index("time")
    rows = []
    for game in games.itertuples():
        kickoff = pd.Timestamp(f"{pd.Timestamp(game.gameday).date()} {game.gametime}")
        hours = table.loc[kickoff.floor("h"):kickoff.floor("h") + pd.Timedelta(hours=GAME_HOURS + 1)]
        if len(hours) == GAME_HOURS + 2 and hours.notna().all().all():
            rows.append({"game_id": str(game.game_id), "temp": round(at_kickoff(hours.temperature_2m, kickoff), 1),
                         "wind": round(at_kickoff(hours.wind_speed_10m, kickoff), 1),
                         "precipitation": round(game_precipitation(hours.precipitation, kickoff), 2)})
    return pd.DataFrame(rows, columns=COLUMNS)


def fetch(games: pd.DataFrame, pause: float = 1.0) -> pd.DataFrame:
    """Request one stadium and season at a time, since the free service limits long requests."""
    frames = []
    seasons = pd.to_datetime(games.gameday).dt.year.where(pd.to_datetime(games.gameday).dt.month > 2, pd.to_datetime(games.gameday).dt.year - 1)
    for (stadium, _), group in games.groupby(["stadium_id", seasons]):
        latitude, longitude = COORDINATES[stadium]
        days = pd.to_datetime(group.gameday)
        for attempt in range(RETRIES):
            response = requests.get(URL, timeout=120, params={
                "latitude": latitude, "longitude": longitude, "start_date": f"{days.min().date()}",
                "end_date": f"{(days.max() + pd.Timedelta(days=1)).date()}", "hourly": "temperature_2m,wind_speed_10m,precipitation",
                "temperature_unit": "fahrenheit", "wind_speed_unit": "mph", "timezone": TIME_ZONE})
            if response.status_code != 429:
                break
            time.sleep(RATE_LIMIT_WAIT * (attempt + 1))
        response.raise_for_status()
        frames.append(game_rows(group, response.json()["hourly"]))
        time.sleep(pause)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)


def update_table(schedule: pd.DataFrame, path: Path = TABLE) -> pd.DataFrame:
    table = load_table(path)
    games = games_to_fetch(schedule[schedule.season.ge(FIRST_SEASON)], set(table.game_id))
    table = pd.concat([table, fetch(games)], ignore_index=True).sort_values("game_id")
    table.to_csv(path, index=False)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--schedule", type=Path, required=True)
    args = parser.parse_args()
    table = update_table(pd.read_parquet(args.schedule))
    print(f"{len(table)} games in {TABLE}")


if __name__ == "__main__":
    main()
