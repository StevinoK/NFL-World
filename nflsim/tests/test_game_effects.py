import numpy as np
import torch

from nflsim.engine.game_effects import GameEffects, expand_shift
from nflsim.models.game_effects import (
    Data, ModelRows, dispersion_report, effect_logits, fit, importance_marginal, stat_draws,
)


def synthetic_rows(games=240, seed=0, loading=0.35):
    """Binary, CORN and multiclass plays sharing one team-game effect with known loadings."""
    rng = np.random.default_rng(seed)
    game_ids = [f"2020_01_G{index:03d}" for index in range(games)]
    effects = {(game, team): rng.standard_normal() for game in game_ids for team in ("AAA", "BBB")}
    rows = {}
    specs = {"comp": ("binary", 2, 50), "rush": ("corn", 4, 25), "play": ("multiclass", 4, 60)}
    loadings = {"comp": np.array([loading]), "rush": np.array([-loading]), "play": np.array([0.0, loading, 0.0, -loading])}
    for model, (head, width, plays) in specs.items():
        base, labels, gs, ts = [], [], [], []
        for (game, team), u in effects.items():
            if head == "binary":
                f = rng.normal(0.3, 0.5, size=(plays, 1))
                p = 1 / (1 + np.exp(-(f[:, 0] + loadings[model][0] * u)))
                y = (rng.uniform(size=plays) < p).astype(int)
            elif head == "corn":
                f = np.tile(np.array([1.0, 0.0, -1.0]), (plays, 1)) + rng.normal(0, 0.2, size=(plays, 1))
                survive = 1 / (1 + np.exp(-(f + loadings[model][0] * u)))
                y = np.zeros(plays, dtype=int)
                alive = np.ones(plays, dtype=bool)
                for step in range(3):
                    alive &= rng.uniform(size=plays) < survive[:, step]
                    y += alive
            else:
                f = rng.normal(0, 0.5, size=(plays, width))
                logits = f + loadings[model] * u
                p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
                y = np.array([rng.choice(width, p=row) for row in p])
            base.append(f)
            labels.append(y)
            gs += [game] * plays
            ts += [team] * plays
        width_out = 3 if head == "corn" else width
        rows[model] = ModelRows(model, head, width_out, np.concatenate(base), np.concatenate(labels),
                                np.arange(4.0) if head == "corn" else None, np.array(gs), np.array(ts))
    return rows


def test_shared_effect_recovers_loadings_and_beats_independent():
    torch.manual_seed(0)
    data = Data(synthetic_rows())
    train, test = data.subset(data.games[:160]), data.subset(data.games[160:])
    shared = fit(train, 0, 1, True, iterations=150, learning_rate=0.05, log_every=0)
    independent = fit(train, 0, 0, False, iterations=100, learning_rate=0.05, log_every=0)
    comp, rush = shared.team["comp"].detach()[0, 0], shared.team["rush"].detach()[0, 0]
    assert 0.2 < abs(float(comp)) < 0.5 and float(comp * rush) < 0  # opposite-signed shared effect
    gain = importance_marginal(shared, test).sum() - importance_marginal(independent, test).sum()
    assert gain > 20

    # The independent model is under-dispersed at the team-game level; the shared one is close to calibrated.
    for effects, low, high in ((independent, 1.4, 99), (shared, 0.7, 1.35)):
        names, observed, means, variances = stat_draws(test, effect_logits(effects if effects.latent_dims else None, 100, len(test.games)), 100)
        table, implied, actual = dispersion_report(names, observed, means, variances)
        ratio = float(table.loc[table.stat == "completions", "variance_ratio"].iloc[0])
        assert low < ratio < high, (effects.latent_dims, ratio)


def test_runtime_draws_match_exported_spec():
    data = Data(synthetic_rows(games=40))
    effects = fit(data, 1, 1, True, iterations=5, log_every=0)
    runtime = GameEffects(effects.export())
    shifts = runtime.team_shifts(runtime.draw_game(np.random.default_rng(0)), np.random.default_rng(1))
    assert shifts["comp"].shape == (2,) and shifts["comp"][0] == 0.0
    assert shifts["rush"].shape == (3,) and np.allclose(shifts["rush"], shifts["rush"][0])
    assert shifts["play"].shape == (4,)
    assert np.allclose(expand_shift("corn", 5, np.array([0.5])), 0.5)
