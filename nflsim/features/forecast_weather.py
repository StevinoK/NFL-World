"""Kickoff forecasts for unplayed outdoor games, from Open-Meteo (free, no key, about 16 days ahead).

The schedule only gets temperature and wind after a game is played, so without this every upcoming outdoor game has
`weather_missing` set. Past games use the archived forecasts in `forecast_archive`. A forecast is saved once per day and reused, so runs on the same day see the same weather.
"""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from nflsim.features.weather import OUTDOOR_ROOFS
from nflsim.paths import ROOT

URL = "https://api.open-meteo.com/v1/forecast"
FOLDER = ROOT / "outputs/weather"
TIME_ZONE = "America/New_York"  # The schedule's gametime is Eastern.
HORIZON_DAYS = 16
GAME_HOURS = 3  # Precipitation counts the kickoff hour and the next three.
COLUMNS = ["game_id", "temp", "wind", "precipitation", "fetched_on"]
# Latitude and longitude of every open-air stadium. Indoor and retractable venues get no forecast.
STADIUMS = {
    "BAL00": (39.2780, -76.6227), "BOS00": (42.0909, -71.2643), "BUF00": (42.7738, -78.7870), "BUF01": (42.8780, -78.8390),
    "CAR00": (35.2258, -80.8528), "CHI98": (41.8623, -87.6167), "CIN00": (39.0954, -84.5160), "CLE00": (41.5061, -81.6995),
    "DEN00": (39.7439, -105.0201), "GNB00": (44.5013, -88.0622), "JAX00": (30.3239, -81.6373), "KAN00": (39.0489, -94.4839),
    "MEX00": (19.3029, -99.1505), "MIA00": (25.9580, -80.2389), "NAS00": (36.1665, -86.7713), "NYC01": (40.8135, -74.0745),
    "PHI00": (39.9008, -75.1675), "PIT00": (40.4468, -80.0158), "SEA00": (47.5952, -122.3316), "SFO01": (37.4030, -121.9700),
    "TAM00": (27.9759, -82.5033), "WAS00": (38.9076, -76.8645), "LON00": (51.5560, -0.2796), "LON02": (51.6043, -0.0664),
}


def games_needing_forecast(schedule: pd.DataFrame, today: date | None = None) -> pd.DataFrame:
    """Unplayed open-air games with no weather yet, kicking off within the forecast horizon."""
    today = pd.Timestamp(today or date.today())
    roof = schedule.roof.astype("string").str.strip().str.lower()
    days = (pd.to_datetime(schedule.gameday, errors="coerce") - today).dt.days
    unplayed = pd.to_numeric(schedule.home_score, errors="coerce").isna()
    empty = pd.to_numeric(schedule.temp, errors="coerce").isna() | pd.to_numeric(schedule.wind, errors="coerce").isna()
    return schedule[roof.isin(OUTDOOR_ROOFS) & unplayed & empty & days.between(0, HORIZON_DAYS - 1)]


def at_kickoff(series: pd.Series, kickoff: pd.Timestamp) -> float:
    """Value at kickoff, between the two surrounding hours."""
    series = series.astype(float)
    return float(series.reindex(series.index.union([kickoff])).interpolate(method="time").loc[kickoff])


def game_precipitation(series: pd.Series, kickoff: pd.Timestamp) -> float:
    """Rain and melted snow forecast during the game, in millimeters. Unknown if any hour is missing."""
    start = kickoff.floor("h")
    hours = series.astype(float).loc[start:start + pd.Timedelta(hours=GAME_HOURS)]
    return float(hours.sum()) if len(hours) == GAME_HOURS + 1 and hours.notna().all() else np.nan


def parse_forecast(games: pd.DataFrame, responses: list[dict], stadiums: list[str], fetched_on: str) -> pd.DataFrame:
    """One forecast row per game, read from one API response per stadium."""
    by_stadium = dict(zip(stadiums, responses))
    rows = []
    for game in games.itertuples():
        hourly = by_stadium[game.stadium_id]["hourly"]
        times = pd.Series(pd.to_datetime(hourly["time"]))
        kickoff = pd.Timestamp(f"{pd.Timestamp(game.gameday).date()} {game.gametime}")
        if kickoff < times.iloc[0] or kickoff > times.iloc[-1]:
            continue
        series = {column: pd.Series(hourly[column], index=times.to_numpy(), dtype=float) for column in ("temperature_2m", "wind_speed_10m", "precipitation")}
        values = {"temp": at_kickoff(series["temperature_2m"], kickoff), "wind": at_kickoff(series["wind_speed_10m"], kickoff),
                  "precipitation": game_precipitation(series["precipitation"], kickoff)}
        if np.isfinite(list(values.values())).all():
            rows.append({"game_id": str(game.game_id), **{key: round(value, 1) for key, value in values.items()}, "fetched_on": fetched_on})
    return pd.DataFrame(rows, columns=COLUMNS)


def fetch_forecast(games: pd.DataFrame, fetched_on: str, timeout: float = 30) -> pd.DataFrame:
    stadiums = sorted(set(games.stadium_id) & set(STADIUMS))
    if not stadiums:
        return pd.DataFrame(columns=COLUMNS)
    response = requests.get(URL, timeout=timeout, params={
        "latitude": ",".join(str(STADIUMS[stadium][0]) for stadium in stadiums),
        "longitude": ",".join(str(STADIUMS[stadium][1]) for stadium in stadiums),
        "hourly": "temperature_2m,wind_speed_10m,precipitation", "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
        "timezone": TIME_ZONE, "forecast_days": HORIZON_DAYS,
    })
    response.raise_for_status()
    payload = response.json()
    return parse_forecast(games[games.stadium_id.isin(stadiums)], payload if isinstance(payload, list) else [payload], stadiums, fetched_on)


def load_forecast(schedule: pd.DataFrame, *, today: date | None = None, refresh: bool = False, folder: Path = FOLDER) -> pd.DataFrame:
    """Today's saved forecast, fetched if needed. If the fetch fails, the newest saved one, else an empty table."""
    today = today or date.today()
    path = folder / f"forecast-{today:%Y%m%d}.csv"
    if path.is_file() and not refresh:
        return pd.read_csv(path, dtype={"game_id": str, "fetched_on": str})
    try:
        forecast = fetch_forecast(games_needing_forecast(schedule, today), f"{today:%Y-%m-%d}")
    except (requests.RequestException, KeyError, ValueError) as error:
        print(f"[weather] forecast unavailable: {error}", flush=True)
        saved = sorted(folder.glob("forecast-*.csv"))
        return pd.read_csv(saved[-1], dtype={"game_id": str, "fetched_on": str}) if saved else pd.DataFrame(columns=COLUMNS)
    folder.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(path, index=False)
    return forecast


def apply_forecast(schedule: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame:
    """Fill weather on games that have none. A game already holding an archived forecast keeps it."""
    schedule = schedule.copy()
    if "precipitation" not in schedule:
        schedule["precipitation"] = np.nan
    table = forecast.drop_duplicates("game_id").set_index("game_id")
    ids = schedule.game_id.astype(str)
    empty = pd.to_numeric(schedule.temp, errors="coerce").isna() | pd.to_numeric(schedule.wind, errors="coerce").isna()
    for column in ("temp", "wind", "precipitation"):
        schedule[column] = pd.to_numeric(schedule[column], errors="coerce").where(~empty, ids.map(table[column]).astype(float))
    schedule["weather_forecast"] = empty & ids.isin(table.index)
    return schedule


def forecast_tag(forecast: pd.DataFrame) -> str:
    """Short name for this weather, so cached simulations are never compared across different forecasts."""
    if forecast.empty:
        return "wxnone"
    text = forecast[["game_id", "temp", "wind", "precipitation"]].sort_values("game_id").to_csv(index=False)
    return "wx" + hashlib.sha256(text.encode()).hexdigest()[:8]
