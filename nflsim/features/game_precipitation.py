"""Rain and snow that actually fell during past open-air games, from Open-Meteo's weather archive.

The table is kept in the repository and only grows, so feature builds don't depend on the network. It describes games
already played and suits the ridge's look back at past plays. It must not feed a pregame model input, which can only
know a forecast.

    python -m nflsim.features.game_precipitation --schedule nflsim/artifacts/refresh_inputs/2026_20261101/schedule.parquet
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

from nflsim.features.forecast_archive import COORDINATES, RATE_LIMIT_WAIT, RETRIES, games_to_fetch
from nflsim.paths import ROOT

URL = "https://archive-api.open-meteo.com/v1/archive"
TABLE = ROOT / "config/game_precipitation.csv"
TIME_ZONE = "America/New_York"  # The schedule's gametime is Eastern.
GAME_HOURS = 3  # Kickoff hour plus the next three.
COLUMNS = ["game_id", "rain_mm", "snow_cm"]


def load_table(path: Path = TABLE) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"game_id": str}) if path.is_file() else pd.DataFrame(columns=COLUMNS)


def game_totals(games: pd.DataFrame, hourly: dict) -> pd.DataFrame:
    """Rain and snow from each game's kickoff hour through the next `GAME_HOURS`."""
    table = pd.DataFrame(hourly).assign(time=lambda frame: pd.to_datetime(frame.time)).set_index("time")
    rows = []
    for game in games.itertuples():
        kickoff = pd.Timestamp(f"{pd.Timestamp(game.gameday).date()} {game.gametime}").floor("h")
        hours = table.loc[kickoff:kickoff + pd.Timedelta(hours=GAME_HOURS)]
        if len(hours) == GAME_HOURS + 1 and hours[["rain", "snowfall"]].notna().all().all():
            rows.append({"game_id": str(game.game_id), "rain_mm": round(float(hours.rain.sum()), 2), "snow_cm": round(float(hours.snowfall.sum()), 2)})
    return pd.DataFrame(rows, columns=COLUMNS)


def fetch(games: pd.DataFrame, pause: float = 1.0) -> pd.DataFrame:
    frames = []
    for stadium, group in games.groupby("stadium_id"):
        latitude, longitude = COORDINATES[stadium]
        days = pd.to_datetime(group.gameday)
        for attempt in range(RETRIES):
            response = requests.get(URL, timeout=120, params={
                "latitude": latitude, "longitude": longitude, "start_date": f"{days.min().date()}",
                "end_date": f"{(days.max() + pd.Timedelta(days=1)).date()}", "hourly": "rain,snowfall", "timezone": TIME_ZONE})
            if response.status_code != 429:
                break
            time.sleep(RATE_LIMIT_WAIT * (attempt + 1))  # The free service limits use per minute and per hour.
        response.raise_for_status()
        frames.append(game_totals(group, response.json()["hourly"]))
        time.sleep(pause)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)


def update_table(schedule: pd.DataFrame, path: Path = TABLE) -> pd.DataFrame:
    table = load_table(path)
    new = fetch(games_to_fetch(schedule, set(table.game_id)))
    table = pd.concat([table, new], ignore_index=True).sort_values("game_id")
    table.to_csv(path, index=False)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--schedule", type=Path, required=True)
    args = parser.parse_args()
    table = update_table(pd.read_parquet(args.schedule))
    print(f"{len(table)} games in {TABLE}; {(table.rain_mm >= 1).sum()} with at least 1 mm of rain, {(table.snow_cm >= 0.5).sum()} with snow")


if __name__ == "__main__":
    main()
