from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


WEATHER_COLUMNS = (
    "weather_outdoor",
    "weather_temp_delta",
    "weather_wind_speed",
    "weather_missing",
)
OUTDOOR_ROOFS = {"open", "outdoor", "outdoors"}
INDOOR_ROOFS = {"closed", "dome", "indoor", "indoors"}


def _number(value) -> float:
    if value is None or pd.isna(value):
        return np.nan
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def weather_features(game: Mapping) -> dict[str, float]:
    return weather_frame(pd.DataFrame([dict(game)])).iloc[0].to_dict()


def weather_frame(games: pd.DataFrame) -> pd.DataFrame:
    roof = games.get("roof", pd.Series("", index=games.index)).astype("string").str.strip().str.lower().fillna("")
    outdoor = roof.isin(OUTDOOR_ROOFS)
    indoor = roof.isin(INDOOR_ROOFS)
    temperature = pd.to_numeric(games.get("temp", pd.Series(np.nan, index=games.index)), errors="coerce")
    wind = pd.to_numeric(games.get("wind", pd.Series(np.nan, index=games.index)), errors="coerce")
    complete = np.isfinite(temperature) & np.isfinite(wind)
    valid = outdoor & complete
    return pd.DataFrame({
        "weather_outdoor": outdoor.astype(float),
        "weather_temp_delta": (((temperature - 60.0) / 20.0).clip(-2.0, 2.0)).where(valid, 0.0),
        "weather_wind_speed": (wind / 15.0).clip(0.0, 2.0).where(valid, 0.0),
        "weather_missing": ((~(outdoor | indoor)) | (outdoor & ~complete)).astype(float),
    }, index=games.index, columns=WEATHER_COLUMNS)