"""Play-level expected outcomes from game state and weather, for ridge targets measured over expectation.

A ridge regression on raw outcomes credits a defense for a low completion rate in a windy,
rainy game and charges a running back for long-yardage situations. Instead, each target's
outcome is first predicted from context that is not in the ridge design: down, distance,
field position, clock, score, formation, play direction, roof/surface and weather. These
effects are non-linear, so a gradient-boosted model handles them; the ridge then solves for
player/team effects on ``outcome - expected``.

Expectations are strictly out of sample by season: season S plays are scored by a model
trained on seasons before S. The first season has no history, so it is cross-fitted by game.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from nflsim.ridge.metrics import TARGETS_BY_NAME, RidgeTarget

STATE_FEATURES = (
    "down", "ydstogo", "yardline_100", "qtr", "half_seconds_remaining", "score_differential",
    "shotgun", "no_huddle", "goal_to_go",
)
WEATHER_FEATURES = (
    "ctx_outdoor", "ctx_temp", "ctx_wind", "ctx_humidity", "ctx_rain", "ctx_snow", "ctx_turf",
)
PASS_FEATURES = ("air_yards", "pass_left", "pass_middle")
RUN_FEATURES = ("run_left", "run_middle", "run_end", "run_tackle")
PRECIPITATION_RAIN = re.compile(r"\b(rain|showers?|drizzle|thunder|storms?)\b", re.IGNORECASE)
PRECIPITATION_SNOW = re.compile(r"\b(snow|flurr(?:y|ies)|sleet|wintry)\b", re.IGNORECASE)
NO_PRECIPITATION = re.compile(r"\b(no rain|chance of rain|0% chance)\b", re.IGNORECASE)
WEATHER_TEMP = re.compile(r"Temp:\s*(-?\d+)")
WEATHER_WIND = re.compile(r"Wind:\s*[A-Za-z]*\s*(\d+)\s*mph", re.IGNORECASE)
WEATHER_HUMIDITY = re.compile(r"Humidity:\s*(\d+)%")
INDOOR_ROOFS = {"dome", "closed", "indoor", "indoors"}
TURF = re.compile(r"turf|astro|matrix|sport", re.IGNORECASE)


@dataclass(frozen=True)
class ExpectationSpec:
    """Which raw outcome a ridge target is measured against, and the context its expectation uses."""

    target: str
    outcome: str
    kind: str
    features: tuple[str, ...]


EXPECTATIONS: dict[str, ExpectationSpec] = {
    spec.target: spec for spec in (
        ExpectationSpec("yac_oe", "yards_after_catch", "normal", STATE_FEATURES + WEATHER_FEATURES + PASS_FEATURES),
        ExpectationSpec("first_down", "first_down", "binary", STATE_FEATURES + WEATHER_FEATURES + PASS_FEATURES),
        ExpectationSpec("yards_per_target", "yards_gained", "normal", STATE_FEATURES + WEATHER_FEATURES + PASS_FEATURES),
        ExpectationSpec("cpoe", "complete_pass", "binary", STATE_FEATURES + WEATHER_FEATURES + PASS_FEATURES),
        *(ExpectationSpec(f"cpoe_{bucket}", "complete_pass", "binary", STATE_FEATURES + WEATHER_FEATURES + PASS_FEATURES)
          for bucket in ("short", "medium", "deep")),
        ExpectationSpec("sack", "sack", "binary", STATE_FEATURES + WEATHER_FEATURES),
        ExpectationSpec("interception", "interception", "binary", STATE_FEATURES + WEATHER_FEATURES),
        ExpectationSpec("rush_yards", "yards_gained", "normal", STATE_FEATURES + WEATHER_FEATURES + RUN_FEATURES),
        ExpectationSpec("explosive_rush", "explosive", "binary", STATE_FEATURES + WEATHER_FEATURES + RUN_FEATURES),
        ExpectationSpec("scramble_yards", "yards_gained", "normal", STATE_FEATURES + WEATHER_FEATURES),
        ExpectationSpec("pass_epa", "epa", "normal", STATE_FEATURES + WEATHER_FEATURES),
        ExpectationSpec("rush_epa", "epa", "normal", STATE_FEATURES + WEATHER_FEATURES),
    )
}


# Targets the feature build measures over expectation: pooled 2022-25 test seasons showed a
# significant gain and no significantly worse season (``nflsim/outputs/variant_bakeoff_xoe``).
# CPOE (deep is worse), interceptions, EPA, scramble yards and YAC stay raw.
PRODUCTION_XOE_TARGETS = ("first_down", "yards_per_target", "rush_yards", "sack", "explosive_rush")
EXPECTATION_INPUTS = (
    "game_id", "season", "weather", "roof", "surface", "temp", "wind", "pass_location", "run_location", "run_gap",
    "complete_pass", "target", "pass_attempt", "rush_attempt", "yards_gained", "first_down", "sack", "explosive",
)


def production_targets(targets: Iterable[RidgeTarget]) -> tuple[RidgeTarget, ...]:
    """Ridge targets as the feature build fits them: adopted targets keep their name and outputs but read ``__xoe``."""
    return tuple(replace(target, value_col=f"{target.name}__xoe") if target.name in PRODUCTION_XOE_TARGETS else target
                 for target in targets)


def with_production_expectations(pbp: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """``pbp`` plus the ``__expected``/``__xoe`` columns of the production over-expectation targets.

    Expectations are fit on a slim copy of the needed columns, so the full play table is not duplicated.
    """
    if all(f"{name}__xoe" in pbp for name in PRODUCTION_XOE_TARGETS):
        return pbp
    columns = [column for column in dict.fromkeys(EXPECTATION_INPUTS + STATE_FEATURES + PASS_FEATURES) if column in pbp]
    added = add_expectations(pbp[columns], PRODUCTION_XOE_TARGETS, seed=seed)
    new = [column for name in PRODUCTION_XOE_TARGETS for column in (f"{name}__expected", f"{name}__xoe")]
    return pbp.assign(**{column: added[column] for column in new})


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    """float64 column (nullable pandas dtypes become NaN, which the boosted trees handle)."""
    if column not in frame:
        return pd.Series(np.nan, index=frame.index)
    values = pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(values.to_numpy(dtype=float, na_value=np.nan), index=frame.index)


def _parsed(text: pd.Series, pattern: re.Pattern) -> pd.Series:
    return pd.to_numeric(text.str.extract(pattern, expand=False), errors="coerce")


def context_features(pbp: pd.DataFrame) -> pd.DataFrame:
    """Weather, surface and play-direction features; numeric weather falls back to the text report."""
    text = pbp["weather"].astype("string").fillna("") if "weather" in pbp else pd.Series("", index=pbp.index, dtype="string")
    roof = pbp["roof"].astype("string").str.lower().fillna("") if "roof" in pbp else pd.Series("", index=pbp.index)
    outdoor = ~roof.isin(INDOOR_ROOFS)
    temp = _numeric(pbp, "temp").fillna(_parsed(text, WEATHER_TEMP))
    wind = _numeric(pbp, "wind").fillna(_parsed(text, WEATHER_WIND))
    humidity = _parsed(text, WEATHER_HUMIDITY)
    clear = text.str.contains(NO_PRECIPITATION, na=False)
    surface = pbp["surface"].astype("string").fillna("") if "surface" in pbp else pd.Series("", index=pbp.index)
    pass_location = pbp["pass_location"].astype("string") if "pass_location" in pbp else pd.Series(pd.NA, index=pbp.index)
    run_location = pbp["run_location"].astype("string") if "run_location" in pbp else pd.Series(pd.NA, index=pbp.index)
    run_gap = pbp["run_gap"].astype("string") if "run_gap" in pbp else pd.Series(pd.NA, index=pbp.index)
    return pd.DataFrame({
        "ctx_outdoor": outdoor.astype(float),
        # Indoor games are played in controlled conditions: neutral temperature, no wind or precipitation.
        "ctx_temp": temp.where(outdoor, 70.0),
        "ctx_wind": wind.where(outdoor, 0.0),
        "ctx_humidity": humidity.where(outdoor, 50.0),
        "ctx_rain": (outdoor & text.str.contains(PRECIPITATION_RAIN, na=False) & ~clear).astype(float),
        "ctx_snow": (outdoor & text.str.contains(PRECIPITATION_SNOW, na=False) & ~clear).astype(float),
        "ctx_turf": surface.str.contains(TURF, na=False).astype(float),
        "pass_left": pass_location.eq("left").astype(float).where(pass_location.notna()),
        "pass_middle": pass_location.eq("middle").astype(float).where(pass_location.notna()),
        "run_left": run_location.eq("left").astype(float).where(run_location.notna()),
        "run_middle": run_location.eq("middle").astype(float).where(run_location.notna()),
        "run_end": run_gap.eq("end").astype(float).where(run_gap.notna()),
        "run_tackle": run_gap.eq("tackle").astype(float).where(run_gap.notna()),
    }, index=pbp.index)


def _feature_matrix(context: pd.DataFrame, features) -> pd.DataFrame:
    """Plain float64 features (nullable pandas dtypes become NaN, which the boosted trees handle)."""
    return pd.DataFrame({feature: _numeric(context, feature).to_numpy(dtype=float, na_value=np.nan)
                         for feature in features}, index=context.index)


def _season(pbp: pd.DataFrame) -> pd.Series:
    if "season" in pbp:
        return pd.to_numeric(pbp["season"], errors="coerce")
    return pd.to_numeric(pbp["game_id"].astype(str).str[:4], errors="coerce")


def _model(kind: str, seed: int):
    common = dict(max_iter=200, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=200,
                  l2_regularization=1.0, random_state=seed)
    return HistGradientBoostingClassifier(**common) if kind == "binary" else HistGradientBoostingRegressor(**common)


def _predict(model, kind: str, x: np.ndarray) -> np.ndarray:
    return model.predict_proba(x)[:, 1] if kind == "binary" else model.predict(x)


def expected_outcomes(pbp: pd.DataFrame, spec: ExpectationSpec, mask: pd.Series, seed: int = 0) -> pd.Series:
    """Out-of-sample expected outcome for every masked play (NaN elsewhere)."""
    context = pbp.join(context_features(pbp)) if "ctx_outdoor" not in pbp else pbp
    frame = _feature_matrix(context, spec.features)
    outcome = _numeric(pbp, spec.outcome)
    usable = mask.fillna(False).astype(bool) & outcome.notna()
    seasons = _season(pbp)
    expected = pd.Series(np.nan, index=pbp.index)
    ordered = sorted(int(season) for season in seasons[usable].dropna().unique())
    for position, season in enumerate(ordered):
        current = usable & seasons.eq(season)
        if position == 0:
            games = pbp.loc[current, "game_id"].astype(str)
            fold = games.map({game: index % 2 for index, game in enumerate(sorted(games.unique()))})
            for held_out in (0, 1):
                train, score = current & fold.reindex(pbp.index).ne(held_out), current & fold.reindex(pbp.index).eq(held_out)
                model = _model(spec.kind, seed).fit(frame[train].to_numpy(), outcome[train].to_numpy())
                expected[score] = _predict(model, spec.kind, frame[score].to_numpy())
            continue
        train = usable & seasons.lt(season)
        model = _model(spec.kind, seed).fit(frame[train].to_numpy(), outcome[train].to_numpy())
        expected[current] = _predict(model, spec.kind, frame[current].to_numpy())
    return expected


def over_expectation_target(target: RidgeTarget) -> RidgeTarget:
    """The same ridge target measured as ``outcome - expected`` (columns added by ``add_expectations``)."""
    return replace(target, name=f"{target.name}_xoe", value_col=f"{target.name}__xoe")


def expected_target(target: RidgeTarget) -> RidgeTarget:
    """Aggregates the expectation itself, used to put residual predictions back on the outcome scale."""
    return replace(target, name=f"{target.name}_expected", value_col=f"{target.name}__expected")


def raw_target(target: RidgeTarget) -> RidgeTarget:
    """Ridge on the unadjusted outcome (baseline for the over-expectation variant)."""
    return replace(target, name=f"{target.name}_raw", value_col=EXPECTATIONS[target.name].outcome)


def add_expectations(pbp: pd.DataFrame, targets: Iterable[str], seed: int = 0) -> pd.DataFrame:
    """Add ``<target>__expected`` and ``<target>__xoe`` play columns for each named ridge target."""
    pbp = pbp.copy()
    context = context_features(pbp)
    pbp[list(context.columns)] = context
    for name in targets:
        spec = EXPECTATIONS[name]
        target = TARGETS_BY_NAME[name]
        mask = target.mask(pbp).fillna(False).astype(bool)
        expected = expected_outcomes(pbp, spec, mask, seed=seed)
        pbp[f"{name}__expected"] = expected
        pbp[f"{name}__xoe"] = _numeric(pbp, spec.outcome) - expected
    return pbp


def weather_effects(pbp: pd.DataFrame, name: str, seed: int = 0) -> pd.DataFrame:
    """Average expectation shift for a few weather scenarios relative to a calm 60F outdoor game."""
    spec = EXPECTATIONS[name]
    target = TARGETS_BY_NAME[name]
    context = pbp.join(context_features(pbp)) if "ctx_outdoor" not in pbp else pbp
    mask = target.mask(pbp).fillna(False).astype(bool) & _numeric(pbp, spec.outcome).notna()
    frame = _feature_matrix(context, spec.features)[mask]
    model = _model(spec.kind, seed).fit(frame.to_numpy(), _numeric(pbp, spec.outcome)[mask].to_numpy())
    sample = frame.sample(min(len(frame), 20000), random_state=seed)
    base = sample.assign(ctx_outdoor=1.0, ctx_temp=60.0, ctx_wind=5.0, ctx_humidity=50.0, ctx_rain=0.0, ctx_snow=0.0)
    scenarios = {
        "calm_60F": {}, "wind_20mph": {"ctx_wind": 20.0}, "cold_25F": {"ctx_temp": 25.0},
        "rain": {"ctx_rain": 1.0}, "snow": {"ctx_snow": 1.0, "ctx_temp": 30.0},
        "dome": {"ctx_outdoor": 0.0, "ctx_temp": 70.0, "ctx_wind": 0.0},
    }
    baseline = _predict(model, spec.kind, base.to_numpy()).mean()
    rows = []
    for label, changes in scenarios.items():
        value = _predict(model, spec.kind, base.assign(**changes).to_numpy()).mean()
        rows.append({"target": name, "outcome": spec.outcome, "scenario": label, "expected": value, "shift": value - baseline})
    return pd.DataFrame(rows)
