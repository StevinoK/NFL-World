"""Draw team and player ratings once per simulated game from their ridge posterior (an experiment).

The bundle's ratings are point estimates, so simulated games carry no uncertainty about team strength and the
score distributions come out narrower than the forecast errors. With NFLSIM_STRENGTH_DRAWS naming a settings JSON,

    {"sd_table": "<csv from python -m nflsim.ridge.uncertainty>", "scale": 1.0}

each simulated game draws one standard normal per ridge coefficient in the table (fit, block, label) and moves
every column that reads it by ``scale x sd x draw``: a team's defense rating, its own defense-mu context and the
opponent's copies move together. Games or columns absent from the table keep their point estimates. ``scale``
multiplies the SDs (the 2023-25 pass-EPA check in docs/analysis/2026-11-accuracy/exp_strength_uncertainty.py
put the calibrated variance scale near 0.7, i.e. scale ~0.85).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

STRENGTH_DRAWS_ENV = "NFLSIM_STRENGTH_DRAWS"
_SETTINGS: dict = {}
_TABLES: dict = {}


def settings() -> dict | None:
    path = os.environ.get(STRENGTH_DRAWS_ENV)
    if not path:
        return None
    if path not in _SETTINGS:
        _SETTINGS[path] = json.loads(Path(path).read_text(encoding="utf-8"))
    return _SETTINGS[path]


def sd_table(path) -> dict[str, pd.DataFrame]:
    """The SD table by game_id (read once per process)."""
    key = str(path)
    if key not in _TABLES:
        table = pd.read_csv(path, dtype={"game_id": str, "entity": str, "label": str})
        _TABLES[key] = {game_id: rows.reset_index(drop=True) for game_id, rows in table.groupby("game_id")}
    return _TABLES[key]


def apply(game_id: str, teams, rng: np.random.Generator, config: dict) -> int:
    """Shift the ratings of ``teams`` (simulator Team objects with their players) for one simulated game.
    Returns how many columns moved."""
    rows = sd_table(config["sd_table"]).get(str(game_id))
    if rows is None:
        return 0
    scale = float(config.get("scale", 1.0))
    keys = sorted(set(zip(rows["fit"], rows["block"], rows["label"])))
    draws = dict(zip(keys, rng.standard_normal(len(keys))))
    by_team = {str(team.name): team for team in teams}
    by_player = {str(player.player_id): player for team in teams for player in team.players}
    own_copy = set()  # attributes copied before their first change, so shared inputs stay untouched

    def attributes_of(holder):
        if id(holder) not in own_copy:
            holder.attributes = holder.attributes.copy()
            holder.__dict__.pop("_values", None)  # Player.value's cache of the old attributes
            own_copy.add(id(holder))
        return holder.attributes

    moved = 0
    for row in rows.itertuples(index=False):
        holder = by_team.get(row.entity) if row.kind == "team" else by_player.get(row.entity)
        if holder is None or row.column not in holder.attributes:
            continue
        attributes = attributes_of(holder)
        delta = scale * float(row.sd) * draws[(row.fit, row.block, row.label)]
        attributes[row.column] = attributes[row.column].astype(float) + delta if row.kind == "player" \
            else float(attributes[row.column]) + delta
        moved += 1
    return moved
