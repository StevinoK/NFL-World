import copy
import json

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from nflsim.engine.cascade_trainer import CascadeConfig, CascadeMLP, ResidualGridNet
from nflsim.engine.model_architecture import build_network
from nflsim.bundle import seal_bundle, validate_bundle, verify_bundle_hash
from nflsim.configuration import snapshot
from nflsim.paths import digest, write_json
from nflsim.features.player_value import PV_DEFENSE_COLUMNS
from nflsim.sim.runtime import RidgeGame
from nflsim.engine.simulation_runtime import Player, SimulationRuntime, Team


class RecordingProcessor:
    def __init__(self, wrapped):
        self.wrapped, self.inputs = wrapped, []

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def transform(self, frame):
        self.inputs.append(frame.copy())
        return self.wrapped.transform(frame)


def create_synthetic_bundle(directory):
    manifest = snapshot(destination=directory / "config")
    torch.manual_seed(71)
    rng = np.random.default_rng(71)
    state = {"benchmark_ready": False, "synthetic_only": True, "context_map": {}, "trained": {}}
    for model, spec in manifest["models"].items():
        columns = list(dict.fromkeys(spec["columns"] + spec["context_columns"]))
        state["context_map"][model] = spec["context_columns"]
        values = {column: rng.normal(size=40) for column in columns}
        cats = {"roof": ["outdoors", "dome"], "posteam_type": ["home", "away"], "position": ["QB", "WR", "RB", "TE"], "play_type_cat": ["rush", "sack", "complete_pass", "incomplete_pass", "punt", "interception", "field_goal_attempt", "no_play_penalty"], "play_outcome": ["run", "completion", "sack"], "penalty_context": ["on_play", "no_play"]}
        for column in columns:
            if column in cats:
                values[column] = np.resize(cats[column], 40)
        frame = pd.DataFrame(values)
        sidecar = copy.deepcopy(spec["sidecar"])
        if sidecar["kind"] == "cascade":
            frame["residual_com"] = rng.normal(size=40)
        categorical = [column for column in frame if column in cats]
        numerical = [column for column in frame if column not in categorical]
        processor = ColumnTransformer([("num", StandardScaler(), numerical), ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical)]).fit(frame)
        width = len(processor.get_feature_names_out())
        prefix = spec["prefix"]
        kind = sidecar["kind"]
        state["trained"][model] = {"kind": kind}
        joblib.dump(processor, directory / f"{prefix}_preprocessor.pkl")
        joblib.dump(spec["groups"], directory / f"{prefix}_columns.pkl")
        if kind == "cascade":
            config = CascadeConfig(**sidecar["config"])
            config.hidden_dims = (8,)
            config.stage2_hidden_dims = (8,)
            context = list(dict.fromkeys(sidecar["context_columns"] + spec["context_columns"]))
            scaler = StandardScaler().fit(rng.normal(size=(40, len(context))))
            stage3 = CascadeMLP(width, sidecar["n_classes"], config)
            stage2 = ResidualGridNet(len(context), sidecar["n_bins"], config)
            stage1 = CascadeMLP(width-1, sidecar["n_classes"], config)
            for stage, network in ((1, stage1), (2, stage2), (3, stage3)):
                torch.save(network.state_dict(), directory / f"{model}_cascade_stage{stage}_model.pt")
            joblib.dump(processor, directory / f"{model}_cascade_stage3_preprocessor.pkl")
            joblib.dump(scaler, directory / f"{model}_cascade_stage2_preprocessor.pkl")
            stage1_proc = ColumnTransformer([("num", StandardScaler(), [column for column in numerical if column != "residual_com"]), ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical)]).fit(frame.drop(columns="residual_com"))
            joblib.dump(stage1_proc, directory / f"{model}_cascade_stage1_preprocessor.pkl")
            sidecar["context_columns"] = context
            sidecar["config"]["hidden_dims"] = [8]
            sidecar["config"]["stage2_hidden_dims"] = [8]
            sidecar["arch"].update(stage3_hidden_dims=[8], stage2_hidden_dims=[8])
            sidecar["dims"].update(stage3_input_dim=width, stage2_context_dim=len(context))
            write_json(directory / f"{model}_cascade_arch.json", sidecar)
            joblib.dump(spec["groups"], directory / f"{model}_cascade_columns.pkl")
        else:
            assert kind in {"corn", "flat"}
            sidecar["dims"]["input_dim"] = width
            sidecar["arch"] = {"hidden_dims": [8], "dropout": [0.0]}
            network = build_network(kind, sidecar["dims"], sidecar["arch"])
            torch.save(network.state_dict(), directory / f"{prefix}_model.pt")
            write_json(directory / f"{prefix}_arch.json", sidecar)
            if kind == "corn":
                bins = np.arange(sidecar["dims"]["output_dim"])
                if model == "time":
                    bins = bins + 5
                pd.DataFrame(bins).to_csv(directory / f"{prefix}_bins.csv")
    team_columns = list(dict.fromkeys(SimulationRuntime.team_stats_cols + SimulationRuntime.def_cols + SimulationRuntime.pregame_context_columns + SimulationRuntime.qb_context_columns + [column for columns in state["context_map"].values() for column in columns]))
    player_columns = list(dict.fromkeys(SimulationRuntime.passer_columns + SimulationRuntime.receiver_columns + SimulationRuntime.rusher_columns + SimulationRuntime.qb_context_columns))
    pd.DataFrame([{"team": team, "game_id": "2025_01_B_A", "game_date": "2025-09-01", **{column: 0.3 if team == "A" else -0.2 for column in team_columns}} for team in ("A", "B")]).to_csv(directory / "team_stats_wide.csv", index=False)
    pd.DataFrame([{"player_id": "P", "player_name": "P", "game_id": "2025_01_B_A", "game_date": "2025-09-01", **{column: 0.3 for column in player_columns}}]).to_csv(directory / "player_stats_wide.csv", index=False)
    write_json(directory / "feature_manifest.json", {"tables": {name: digest(directory/name) for name in ("team_stats_wide.csv", "player_stats_wide.csv")}})
    write_json(directory / "training_manifest.json", state)


def test_strict_full_bundle_all_inference_and_kernel(tmp_path):
    torch.set_num_threads(1)
    create_synthetic_bundle(tmp_path)
    runtime = validate_bundle(tmp_path, require_ready=False)
    assert set(runtime.model_dict) == set(SimulationRuntime.models)
    checksum = seal_bundle(tmp_path)
    assert verify_bundle_hash(tmp_path) == checksum
    def team(name, role):
        attrs = runtime.team_stats_wide.set_index("team").loc[name]
        players = [Player(name+slot, slot, name+slot, runtime.player_stats_wide.copy()) for slot in ("QB1", "WR1", "WR2", "WR3", "WR4", "RB1", "RB2", "RB3", "TE1", "TE2", "TE3")]
        return Team(name, players, role, attrs, runtime)
    away, home = team("B", "away"), team("A", "home")
    runtime.seed(17)
    game = RidgeGame(away, home, "outdoors", runtime)
    for key, network in runtime.model_dict.items():
        if runtime.context_map.get(key):
            runtime.processor[key] = RecordingProcessor(runtime.processor[key])
        if hasattr(network, "stage2_preprocessor"):
            network.stage2_preprocessor = RecordingProcessor(network.stage2_preprocessor)
    game._resample_prior_embeddings()
    assert set(home.model_latents) == {"play", "comp", "target", "rusher"}
    for key, columns in runtime.context_map.items():
        if not columns:
            continue
        for role, effect in (("home", 0.3), ("away", -0.2)):
            categories = {"roof": "outdoors", "posteam_type": role, "position": "WR", "play_type_cat": "rush", "play_outcome": "run", "penalty_context": "no_play"}
            values = {column: categories.get(column, 0.0) for column in runtime.processor[key].feature_names_in_ if not column.startswith("ridge_")}
            runtime.forward_model_logits(key, pd.DataFrame([values]), {"posteam_type": role}, np.zeros(1))
            captured = runtime.processor[key].inputs[-1]
            for column in columns:
                expected = (-0.2 if role == "home" else 0.3) if column in PV_DEFENSE_COLUMNS else effect
                assert np.isclose(captured[column].iloc[0], expected)
        network = runtime.model_dict[key]
        if hasattr(network, "stage2_preprocessor"):
            for captured, effect in zip(network.stage2_preprocessor.inputs, (-0.2, 0.3)):
                for column in columns:
                    expected = (0.1 - effect) if column in PV_DEFENSE_COLUMNS else effect
                    assert np.isclose(captured[0, network.context_columns.index(column)], expected)
    runtime.forward_model_logits("sack", pd.DataFrame([{column: "home" if column == "posteam_type" else "outdoors" if column == "roof" else 0.0 for column in runtime.processor["sack"].feature_names_in_ if not column.startswith("ridge_")}]), {"posteam_type": "home"})
    assert not runtime.schema_issues
    game.simulate_game()
    assert home.score >= 0 and away.score >= 0
    assert game.diagnostics_row()
    assert not runtime.structural_schema_issues()

def test_game_effects_shift_every_team_game_and_play(tmp_path):
    from nflsim.engine.game_effects import FILE_NAME
    torch.set_num_threads(1)
    create_synthetic_bundle(tmp_path)
    runtime = validate_bundle(tmp_path, require_ready=False)
    effects = {}
    for key in ("sack", "int", "rush", "time"):
        width = int(runtime.model_dict[key].out.weight.shape[0])
        head = "corn" if key in {"rush", "time"} else "binary"
        effects[key] = {"head": head, "width": width, "bias": [0.0], "game_loadings": [[0.0]],
                        "team_loadings": [[0.7]], "private_scale": [0.0]}
    write_json(tmp_path / FILE_NAME, {"version": 1, "game_dims": 1, "team_dims": 1, "models": effects})
    state = json.loads((tmp_path / "training_manifest.json").read_text())
    state["game_effects"] = {"file": FILE_NAME, "sha256": digest(tmp_path / FILE_NAME)}
    write_json(tmp_path / "training_manifest.json", state)
    runtime = validate_bundle(tmp_path, require_ready=False)

    def team(name, role):
        attrs = runtime.team_stats_wide.set_index("team").loc[name]
        players = [Player(name+slot, slot, name+slot, runtime.player_stats_wide.copy()) for slot in ("QB1", "WR1", "WR2", "WR3", "WR4", "RB1", "RB2", "RB3", "TE1", "TE2", "TE3")]
        return Team(name, players, role, attrs, runtime)
    away, home = team("B", "away"), team("A", "home")
    runtime.seed(5)
    game = RidgeGame(away, home, "outdoors", runtime)
    game._resample_prior_embeddings()
    for side in (home, away):
        assert side.model_latents["sack"].shape == (2,) and side.model_latents["sack"][0] == 0.0
        assert np.allclose(side.model_latents["rush"], side.model_latents["rush"][0])
        # one team draw moves every loaded model together (same loading, same sign)
        assert np.isclose(side.model_latents["sack"][1], side.model_latents["rush"][0])
    assert not np.isclose(home.model_latents["sack"][1], away.model_latents["sack"][1])
    frame = pd.DataFrame([{column: "home" if column == "posteam_type" else "outdoors" if column == "roof" else 0.0
                           for column in runtime.processor["sack"].feature_names_in_}])
    plain = runtime.forward_model_logits("sack", frame, {"posteam_type": "home"})
    shifted = runtime.forward_model_logits("sack", frame, {"posteam_type": "home"}, home.model_latents["sack"])
    assert torch.allclose(shifted - plain, torch.as_tensor(home.model_latents["sack"]))
    game.simulate_game()
    assert not runtime.structural_schema_issues()
