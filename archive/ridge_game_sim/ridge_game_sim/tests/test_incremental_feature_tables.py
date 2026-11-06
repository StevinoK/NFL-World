import json

import pandas as pd

from ridge_game_sim import build_features


def test_incremental_build_does_not_append_retained_history_twice(tmp_path, monkeypatch):
    previous_game = "2025_01_A_B"
    current_game = "2026_01_A_B"

    def long_rows(entity, game_id, source, value):
        return pd.DataFrame([{
            entity: "P1" if entity == "player_id" else "A",
            "game_id": game_id, "game_date": game_id[:4] + "-09-01",
            "source": source, "average": value, "bayes_estimate_mean": value,
        }])

    prior = tmp_path / "prior"
    prior.mkdir()
    for entity, filename in (("player_id", "player_stats.csv"), ("team", "Team_Stats.csv")):
        pd.concat([
            long_rows(entity, previous_game, "retained", 10.0),
            long_rows(entity, previous_game, "ridge", 20.0),
            long_rows(entity, current_game, "retained", -1.0),
            long_rows(entity, current_game, "ridge", -2.0),
        ], ignore_index=True).to_csv(prior / filename, index=False)
    context = pd.DataFrame({"game_id": [previous_game, current_game], "team": ["A", "A"]})
    context.to_csv(prior / "ridge_context.csv", index=False)

    monkeypatch.setattr(build_features, "selection_input_hashes", lambda *args: {})
    monkeypatch.setattr(build_features, "skill_roster", lambda frame: (frame, pd.DataFrame()))
    monkeypatch.setattr(build_features, "build_estimates", lambda *args, **kwargs: (
        long_rows("player_id", current_game, "ridge", 30.0),
        long_rows("team", current_game, "ridge", 40.0),
        context.iloc[1:].copy(), pd.DataFrame(),
    ))
    monkeypatch.setattr(build_features, "regenerate_retained_features", lambda *args, **kwargs: (
        pd.concat([long_rows("player_id", previous_game, "retained", 10.0),
                   long_rows("player_id", current_game, "retained", 50.0)], ignore_index=True),
        pd.concat([long_rows("team", previous_game, "retained", 10.0),
                   long_rows("team", current_game, "retained", 60.0)], ignore_index=True),
        pd.DataFrame(),
    ))
    parameters = tmp_path / "parameters.json"
    parameters.write_text(json.dumps({"max_selection_season": 2024, "validation_season": 2024,
                                     "input_hashes": {}, "parameters": {"ridge": {}}}))
    output = tmp_path / "features"
    build_features.build_feature_tables(
        pd.DataFrame(), pd.DataFrame({"game_id": [previous_game, current_game]}),
        pd.DataFrame(), pd.DataFrame(), parameters, output,
        prediction_seasons=[2026], prior_bundle=prior,
    )
    for entity, filename, retained_value, ridge_value in (
        ("player_id", "player_stats.csv", 50.0, 30.0),
        ("team", "Team_Stats.csv", 60.0, 40.0),
    ):
        result = pd.read_csv(output / filename)
        assert not result.duplicated([entity, "game_id", "source"]).any()
        assert len(result) == 4
        assert result.loc[result.game_id.eq(previous_game), "average"].tolist() == [10.0, 20.0]
        current = result.loc[result.game_id.eq(current_game)].set_index("source")
        assert current.loc["retained", "average"] == retained_value
        assert current.loc["ridge", "average"] == ridge_value