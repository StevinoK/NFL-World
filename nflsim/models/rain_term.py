"""Fit one rain effect per play model, because too few games are wet for each network to learn its own rain curve.

The simulator holds the networks' rain input at zero, so they predict a dry game, and each play's logits move by

    shift_m = rain_m * min(weather_precipitation, RAIN_CAP)

The curve can't turn upward in a downpour, and every wet game informs the same number. Only wet games carry information,
so only their plays are read, and dry-game predictions don't change. A coefficient is shrunk by how far it stands above
game-to-game noise, using a standard error that treats a game's plays as one sample, so noise fits to nothing.

    python -m nflsim.models.rain_term --model-dir <refit> --effects <game_effects.json> --output <dir> --inputs <dir> --player-table <csv> --team-table <csv>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from nflsim.engine.game_effects import FILE_NAME, RAIN_CAP
from nflsim.models.fit_minimal import REFERENCE
from nflsim.models.game_effects import DTYPE, ModelRows, load_inputs, local_terms, model_rows, refit_specs
from nflsim.paths import write_json

# Which player gets the ball is a lineup question, and eleven shares per model would fit mostly noise.
EXCLUDED = {"target", "rusher"}
NEWTON_STEPS = 12


def wet_games(prepared: pd.DataFrame) -> pd.Series:
    """Each wet game's rain input, capped."""
    rain = prepared.groupby("game_id").weather_precipitation.max().clip(upper=RAIN_CAP)
    return rain[rain > 0]


def tensors(item: ModelRows, rain: pd.Series, bias: np.ndarray):
    base = torch.as_tensor(item.base, dtype=DTYPE)
    label = torch.as_tensor(item.label, dtype=torch.long)
    amount = torch.as_tensor(pd.Series(item.games).map(rain).to_numpy(dtype=float), dtype=DTYPE)
    capped = None
    if item.limit is not None:
        capped = torch.as_tensor(item.values[item.label] >= item.limit)
    return base, label, amount, torch.as_tensor(bias, dtype=DTYPE)[None, :], capped


def fit_model(item: ModelRows, rain: pd.Series, bias: np.ndarray) -> dict:
    base, label, amount, offset, capped = tensors(item, rain, bias)
    size = offset.shape[1]
    beta = torch.zeros(size, dtype=DTYPE)
    ridge = 1e-6 * torch.eye(size, dtype=DTYPE)  # Multiclass logits are only defined up to a constant.

    def terms(current):
        _, gradient, weight = local_terms(item.head, base, label, offset + amount[:, None] * current[None, :], capped=capped)
        return amount[:, None] * gradient, ((amount ** 2)[:, None, None] * weight).sum(0) + ridge

    for _ in range(NEWTON_STEPS):
        scores, hessian = terms(beta)
        step = torch.linalg.solve(hessian, scores.sum(0))
        beta = beta + step
        if float(step.abs().max()) < 1e-5:
            break
    scores, hessian = terms(beta)
    per_game = pd.DataFrame(scores.numpy()).groupby(item.games).sum().to_numpy()
    inverse = torch.linalg.pinv(hessian - ridge, rcond=1e-8, hermitian=True).numpy()
    covariance = inverse @ (per_game.T @ per_game) @ inverse * len(per_game) / max(len(per_game) - 1, 1)
    raw = beta.numpy() - beta.numpy().mean() if size > 1 else beta.numpy()
    error = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    shrunk = raw * np.clip(1.0 - error ** 2 / np.maximum(raw ** 2, 1e-12), 0.0, None)
    return {"raw": raw, "error": error, "rain": shrunk, "rows": len(label), "games": len(per_game)}


def likelihood_gain(item: ModelRows, rain: pd.Series, bias: np.ndarray, coefficient: np.ndarray) -> float:
    base, label, amount, offset, capped = tensors(item, rain, bias)
    moved = offset + amount[:, None] * torch.as_tensor(coefficient, dtype=DTYPE)[None, :]
    return float((local_terms(item.head, base, label, moved, False, capped)[0] - local_terms(item.head, base, label, offset, False, capped)[0]).sum())


def wet_rows(specs: dict, seasons, args):
    prepared, players, roster = load_inputs(REFERENCE, tuple(seasons), args.player_table, args.team_table, args.inputs)
    rain = wet_games(prepared)
    prepared = prepared.assign(weather_precipitation=0.0)  # What the simulator feeds the networks.
    return model_rows(args.model_dir, specs, prepared, players, roster, keep=lambda ids: pd.Series(ids).isin(rain.index).to_numpy()), rain


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--effects", type=Path, required=True, help="The game_effects.json the rain term is added to")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--player-table", type=Path)
    parser.add_argument("--team-table", type=Path)
    parser.add_argument("--seasons", type=int, nargs=2, default=(2018, 2025))
    parser.add_argument("--test-season", type=int, default=2026)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    spec = json.loads(args.effects.read_text())
    specs = {model: value for model, value in refit_specs(args.model_dir).items() if model not in EXCLUDED}
    rows, rain = wet_rows(specs, args.seasons, args)
    print(f"{len(rain)} wet games, {args.seasons[0]}-{args.seasons[1]}", flush=True)
    test, test_rain = wet_rows(specs, (args.test_season, args.test_season), args)
    table = []
    for model, entry in spec["models"].items():
        size = len(entry["bias"])
        if model not in rows:
            entry["rain"] = [0.0] * size
            continue
        bias = np.asarray(entry["bias"], dtype=float)
        result = fit_model(rows[model], rain, bias)
        entry["rain"] = [float(value) for value in result["rain"]]
        gain = likelihood_gain(test[model], test_rain, bias, result["rain"]) if model in test else np.nan
        for index in range(size):
            table.append({"model": model, "term": index, "raw": result["raw"][index], "standard_error": result["error"][index],
                          "rain": result["rain"][index], "rows": result["rows"], "games": result["games"], "test_season_gain": gain})
        print(table[-1], flush=True)
    spec["fit"]["rain"] = {"seasons": list(args.seasons), "wet_games": len(rain), "cap": RAIN_CAP, "excluded": sorted(EXCLUDED)}
    pd.DataFrame(table).to_csv(args.output / "rain_terms.csv", index=False)
    write_json(args.output / FILE_NAME, spec)
    print("rain term done", flush=True)


if __name__ == "__main__":
    main()
