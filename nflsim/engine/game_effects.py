"""Shared game-level random effects for the play models.

Every play model m gets a logit shift that is constant within a team-game:

    shift_m = b_m + G_m u_game + T_m u_team + s_m * v_team,m + rain_m * r,
    u_game ~ N(0, I), u_team ~ N(0, I), v ~ N(0, I).

u_game is shared by both teams (pace, officiating, game script), u_team by all
of one offense's plays in that game (a good or bad day that moves completion,
sacks, interceptions and yardage together), and v is model-specific. The
loadings are fit by marginal likelihood on games the play models never saw
(see nflsim.models.game_effects), so the simulator draws effects exactly as
they were estimated: no outcome-derived inputs and no double-counted noise.

r is the game's forecast rain and rain_m comes from nflsim.models.rain_term. Bundles without it have no rain term.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

FILE_NAME = "game_effects.json"
RAIN_CAP = 1.0986  # 8 mm on the weather_precipitation scale. Real scoring stops falling about there, and few games go past it.


def head_width(head: str, width: int) -> int:
    """Number of shift parameters for a head: one for binary/CORN, one per class for multiclass."""
    return int(width) if head == "multiclass" else 1


def expand_shift(head: str, width: int, shift: np.ndarray) -> np.ndarray:
    """Map a head's shift parameters onto the raw network output (logit) layout."""
    shift = np.asarray(shift, dtype=np.float32).reshape(-1)
    if head == "binary":
        return np.array([0.0, shift[0]], dtype=np.float32)
    if head == "corn":
        return np.full(int(width), shift[0], dtype=np.float32)
    return shift.astype(np.float32)


class GameEffects:
    def __init__(self, spec: dict):
        self.spec = spec
        self.game_dims = int(spec["game_dims"])
        self.team_dims = int(spec["team_dims"])
        self.models: Dict[str, dict] = {}
        for model, entry in spec["models"].items():
            bias = np.asarray(entry["bias"], dtype=np.float64)
            size = bias.shape[0]
            self.models[model] = {
                "head": entry["head"],
                "width": int(entry["width"]),
                "bias": bias,
                "game": np.asarray(entry["game_loadings"], dtype=np.float64).reshape(size, self.game_dims),
                "team": np.asarray(entry["team_loadings"], dtype=np.float64).reshape(size, self.team_dims),
                "private": np.asarray(entry["private_scale"], dtype=np.float64).reshape(size),
                "rain": np.asarray(entry.get("rain", np.zeros(size)), dtype=np.float64).reshape(size),
            }
        self.has_rain = any("rain" in entry for entry in spec["models"].values())
        # Optional per-network temperature from the effects fit (nflsim.models.game_effects --temperature).
        self.temperatures = {model: float(entry["temperature"]) for model, entry in spec["models"].items()
                             if "temperature" in entry}
        # Optional team-strength term: fixed coefficients on standardized team ratings, own offense and opponent defense
        # (an experiment; bundles without it are unchanged).
        strength = spec.get("strength")
        self.strength = None
        if strength:
            self.strength = {"offense": list(strength["offense"]), "defense": list(strength["defense"]),
                             "mean": np.asarray(strength["mean"], dtype=np.float64),
                             "scale": np.asarray(strength["scale"], dtype=np.float64)}
            columns = len(self.strength["mean"])
            for model, entry in spec["models"].items():
                self.models[model]["strength"] = np.asarray(
                    entry.get("strength", np.zeros((len(entry["bias"]), columns))), dtype=np.float64).reshape(len(entry["bias"]), columns)

    def strength_inputs(self, team_attributes, opponent_attributes) -> Optional[np.ndarray]:
        """The standardized ratings of the strength term for one offense against one defense (missing ratings sit at the
        mean), or None when the bundle has no strength term."""
        if self.strength is None:
            return None
        def value(attributes, column):
            try:
                number = float(attributes.get(column, np.nan))
            except (TypeError, ValueError):
                number = np.nan
            return number
        raw = np.array([value(team_attributes, column) for column in self.strength["offense"]]
                       + [value(opponent_attributes, column) for column in self.strength["defense"]], dtype=np.float64)
        standardized = (raw - self.strength["mean"]) / self.strength["scale"]
        return np.where(np.isfinite(standardized), standardized, 0.0)

    @classmethod
    def load(cls, directory: Path) -> Optional["GameEffects"]:
        path = Path(directory) / FILE_NAME
        if not path.is_file():
            return None
        return cls(json.loads(path.read_text()))

    def draw_game(self, rng: np.random.Generator) -> np.ndarray:
        return rng.standard_normal(self.game_dims)

    def team_shifts(self, game_draw: np.ndarray, rng: np.random.Generator, rain: float = 0.0,
                    strength: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Draw one team's effects for a game and return each model's raw-logit shift."""
        rain = min(float(rain), RAIN_CAP)
        team_draw = rng.standard_normal(self.team_dims)
        shifts = {}
        for model, entry in self.models.items():
            private = rng.standard_normal(entry["private"].shape[0])
            shift = entry["bias"] + entry["game"] @ game_draw + entry["team"] @ team_draw + entry["private"] * private + entry["rain"] * rain
            if strength is not None and "strength" in entry:
                shift = shift + entry["strength"] @ strength
            shifts[model] = expand_shift(entry["head"], entry["width"], shift)
        return shifts
