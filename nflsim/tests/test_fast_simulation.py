from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from nflsim.engine.fast_inputs import CompiledProcessor
from nflsim.engine.simulation_runtime import PlayRow, build_frame_from_values, get_probs


def bundle():
    from nflsim.sim.season import latest_bundle
    try:
        return latest_bundle()
    except FileNotFoundError:
        pytest.skip("no weekly-refreshed bundle available")


@pytest.fixture(scope="module")
def runtime():
    from nflsim.bundle import verify_bundle_hash
    from nflsim.remote.worker import get_runtime
    directory = bundle()
    return get_runtime(str(directory), verify_bundle_hash(directory))


def test_get_probs_matches_the_corn_loop():
    logits = np.random.default_rng(1).normal(size=(40, 9)).astype(np.float32)
    probas = torch.cumprod(torch.sigmoid(torch.from_numpy(logits)), dim=1).float().numpy()
    expected = np.array([1.0 - probas[:, 0], *[probas[:, k - 1] - probas[:, k] for k in range(1, 9)], probas[:, 8]]).T
    assert np.array_equal(get_probs(logits), expected)


def test_play_rows_keep_column_order_and_frame_like_the_old_builder():
    row = build_frame_from_values([1.0, "home"], ["down", "posteam_type", "yardline_100"])
    assert isinstance(row, PlayRow) and list(row) == ["down", "posteam_type", "yardline_100"]
    expected = pd.DataFrame([{"down": 1.0, "posteam_type": "home", "yardline_100": np.nan}])
    pd.testing.assert_frame_equal(row.frame(), expected)


def test_compiled_preprocessing_matches_sklearn_bit_for_bit(runtime):
    rng = np.random.default_rng(7)
    checked = 0
    for key, processor in runtime.processor.items():
        if not CompiledProcessor.supports(processor):
            continue
        compiled = CompiledProcessor(processor, None)
        for trial in range(25):
            row = {column: float(rng.normal(processor.named_transformers_["num"].mean_[index],
                                            processor.named_transformers_["num"].scale_[index] * 3))
                   for index, column in enumerate(compiled.numeric)}
            categories = [list(values) for values in processor.named_transformers_["cat"].categories_]
            for column, values in zip(compiled.categorical, categories):
                row[column] = values[rng.integers(len(values))] if trial % 5 else "unseen-category"
            if trial % 3 == 0:
                row[compiled.numeric[rng.integers(len(compiled.numeric))]] = np.nan
            if trial % 7 == 0:
                row[compiled.numeric[0]] = None
            if trial % 4 == 0:
                del row[compiled.numeric[-1]]  # absent columns are filled with 0.0, as the frame path reindexes
            frame = pd.DataFrame([row]).reindex(columns=compiled.columns, fill_value=0.0)
            fast, missing = compiled.transform(row)
            assert np.array_equal(fast, processor.transform(frame)[0], equal_nan=True), key
            assert missing == {column: 1 for column in frame.columns if frame[column].isna().iloc[0]}
            checked += 1
    assert checked > 0


def test_batched_simulations_reproduce_sequential_ones(runtime):
    from nflsim.bundle import verify_bundle_hash
    from nflsim.features.game_inputs import build_real_game
    from nflsim.remote.worker import simulate_chunk
    from nflsim.sim.execution import build_ridge_tasks
    from nflsim.sim.synthetic_games import _load_inputs

    directory = bundle()
    schedule, roster, depth = _load_inputs(2026, directory / "weekly_inputs")
    row = schedule[schedule.home_score.isna()].sort_values("game_id").iloc[0]
    game = build_real_game(runtime, runtime.player_stats_wide, runtime.team_stats_wide, roster, depth, schedule, row, 3)
    task = build_ridge_tasks([game], str(directory), verify_bundle_hash(directory), "fast-test", 3, 1)[0]
    task = {**task, "start_simulation": 1, "count": 3, "simulation_ids": [1, 2, 3]}
    sequential = simulate_chunk({**task, "batched": False})
    batched = simulate_chunk({**task, "batched": True})
    for key in ("game_rows", "player_rows", "nan_rows"):
        assert pd.DataFrame(sequential[key]).equals(pd.DataFrame(batched[key])), key
    # Batched runs draw from private generators and leave the global ones bound and untouched.
    import random
    before = (np.random.get_state()[1].copy(), random.getstate(), torch.get_rng_state())
    simulate_chunk({**task, "batched": True})
    assert np.random.choice.__self__ is np.random.mtrand._rand and random.random.__self__ is random._inst
    assert np.array_equal(before[0], np.random.get_state()[1]) and before[1] == random.getstate()
    assert torch.equal(before[2], torch.get_rng_state())
