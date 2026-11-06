"""One table saying how every model input in a bundle is built.

Names and config files mislead. 23 inputs, including eight called ``*_bayes``, are ridge estimates. A paired defense
stat such as ``rush_yards_per_carry`` is read from its offense stat's fit, so the settings tuned under its own name
never apply. The remaining smoothed inputs all use one fixed decay, not ``bayesian_decay_rates.json``, which only the
research comparisons read. Read this table, not the names, before describing an input.

Packaging and the weekly refresh write ``input_sources.csv`` into each new bundle. Bundles are hash-sealed, so for an
older one pass ``--output`` to write the table elsewhere:

    python -m nflsim.features.input_sources <bundle> --output input_sources.csv
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from nflsim.features.bayesian_history import _config_for_metric
from nflsim.ridge.bakeoff import PENALTY_GRID, RECENCY_GRID
from nflsim.ridge.metrics import TARGETS

FILE_NAME = "input_sources.csv"
SLOT = re.compile(r"_(QB|RB|WR|TE)\d$")
RIDGE_COLUMN = re.compile(r"^ridge_(.+)_(venue_effect|offense_effect|opponent_defense_effect|defense_mu)$")
GRIDS = {**RECENCY_GRID, **PENALTY_GRID}
SETTINGS = tuple(GRIDS)
SCHEDULE_COLUMNS = {"days_rest", "opp_days_rest"}
TEAM_SCOPE = {target.name for target in TARGETS if target.scope == "team"}


def _fit_settings(metric: str, parameters: dict) -> tuple[str, dict]:
    """The selection entry a metric's values come from: a defense stat paired with an offense stat uses the offense fit."""
    spec = parameters[metric]
    if spec["side"] == "defense":
        for name, other in parameters.items():
            if other["side"] == "subject" and other["target"] == spec["target"]:
                return name, other
    return metric, spec


def _settings_columns(spec: dict) -> dict:
    applicable = [name for name in SETTINGS if not (name == "lambda_offense_team" and spec["target"] in TEAM_SCOPE)]
    edges = [f"{name}={'max' if spec[name] == max(GRIDS[name]) else 'min'}" for name in applicable
             if spec[name] in (min(GRIDS[name]), max(GRIDS[name]))]
    return {**{name: spec[name] for name in applicable}, "target": spec["target"], "at_grid_edge": ", ".join(edges)}


def _describe(base: str, features: dict, parameters: dict, smoothed: set[str], last_tuning_season: int | None) -> dict:
    match = RIDGE_COLUMN.match(base)
    fitted = [name for name, spec in parameters.items() if match and spec["target"] == match.group(1)]
    if fitted:
        fit, spec = _fit_settings(fitted[0], parameters)
        built_by = "ridge defense block" if match.group(2) == "defense_mu" else "ridge context"
        return {"built_by": built_by, "settings_from": fit, "last_tuning_season": last_tuning_season, **_settings_columns(spec)}
    if base in parameters:
        fit, spec = _fit_settings(base, parameters)
        if parameters[base]["side"] == "subject":
            built_by = "ridge subject"
        else:
            built_by = "ridge defense block" if fit != base else "ridge defense fit"
        return {"built_by": built_by, "settings_from": fit, "last_tuning_season": last_tuning_season, **_settings_columns(spec)}
    if base in features.get("usage", {}).get("columns", ()):
        return {"built_by": "usage shares", "settings_from": "usage_params.json"}
    if base in features.get("player_value", {}).get("columns", ()):
        return {"built_by": "player value", "settings_from": "player_value.csv"}
    if base.removeprefix("kick_").removeprefix("ret_") in features.get("special_teams", {}).get("columns", ()):
        return {"built_by": "special teams", "settings_from": "special_teams_params.json"}
    if base in features.get("weather", {}).get("columns", ()):
        return {"built_by": "weather"}
    if base in SCHEDULE_COLUMNS:
        return {"built_by": "schedule"}
    if base in features.get("retained_player_metrics", ()) or base in features.get("retained_team_metrics", ()):
        if base in smoothed:
            shared = _config_for_metric("normal", base, None)
            return {"built_by": "bayesian smoother", "settings_from": "fixed defaults in retained.py", "smoother_decay": shared.decay,
                    "smoother_season_carryover": shared.season_carryover}
        return {"built_by": "retained, not smoothed"}
    return {"built_by": "game state"}


def input_sources(directory: Path) -> pd.DataFrame:
    directory = Path(directory)
    models = json.loads((directory / "model_manifest.json").read_text())["models"]
    features = json.loads((directory / "feature_manifest.json").read_text())
    selection = json.loads((directory / "ridge_selection_parameters.json").read_text())
    parameters = selection["parameters"]
    diagnostics = pd.read_csv(directory / "retained_diagnostics.csv")
    smoothed = set(diagnostics.loc[diagnostics.reason.eq("season_frozen_prior_preweek_update"), "metric"])
    consumers: dict[str, set[str]] = {name: set() for name in parameters}
    slots: dict[str, set[str]] = {}
    for model, spec in models.items():
        for column in [*spec["columns"], *(spec.get("context_columns") or [])]:
            base = SLOT.sub("", column)
            consumers.setdefault(base, set()).add(model)
            if base != column:
                slots.setdefault(base, set()).add(column.removeprefix(f"{base}_"))
    rows = [{"input": base, "model_count": len(users), "models": ", ".join(sorted(users)), "slots": ", ".join(sorted(slots.get(base, ()))),
             **_describe(base, features, parameters, smoothed, selection.get("max_selection_season"))}
            for base, users in consumers.items()]
    order = ["input", "built_by", "target", "settings_from", "at_grid_edge", "last_tuning_season", *SETTINGS, "smoother_decay",
             "smoother_season_carryover", "model_count", "models", "slots"]
    return pd.DataFrame(rows).reindex(columns=order).sort_values(["built_by", "input"]).reset_index(drop=True)


def write_input_sources(directory: Path) -> None:
    """Add the table to a bundle with flat models and a ridge selection; call it before sealing."""
    directory = Path(directory)
    if (directory / "model_manifest.json").is_file() and (directory / "ridge_selection_parameters.json").is_file():
        input_sources(directory).to_csv(directory / FILE_NAME, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    input_sources(args.bundle).to_csv(args.output, index=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
