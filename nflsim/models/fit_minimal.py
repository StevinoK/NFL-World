from __future__ import annotations

import argparse
import copy
import html.parser
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error, roc_auc_score

from nflsim.engine.cascade_trainer import CascadeConfig, fit_cascade_general, save_cascade_artifacts
from nflsim.paths import ROOT, digest, output_dir, write_json
from nflsim.semantics import ISOLATED_MU_CONTEXT_V2
from nflsim.features.goal_line import GOAL_CAPPED_MODELS, GOAL_LINE_COLUMNS, with_goal_line_inputs
from nflsim.features.training_data import ATTEMPT_MODELS, model_frame, prepare_frame
from nflsim.models.train import cascade_metric_record, fit_network
from nflsim.configuration import TARGETS_BY_NAME, V2_MODEL_CONTEXT
from nflsim.engine.simulation_runtime import SimulationRuntime

# Frozen reference for model architectures and the training-manifest template (archived with the superseded bundles).
REFERENCE = ROOT.parent / "archive/nflsim_bundles/pv-isolated-mu-neutral-pilot-r2-2025"
INPUTS = ROOT / "artifacts/inputs"
PLAYER_GROUP_COLUMNS = {
    "passer_columns": ["cpoe_bayes", "turnover_bayes", "sack_bayes", "cpoe_short", "cpoe_medium", "cpoe_deep"],
    "receiver_columns": ["yac_bayes", "fd_bayes", "ypt"],
    "rusher_columns": ["ypc_bayes", "explosive_bayes", "qb_scramble_ypc_bayes"],
}
PLAYER_GROUP_BY_MODEL = {
    "play": "passer_columns",
    "ay": "passer_columns",
    "comp": "passer_columns",
    "int": "passer_columns",
    "sack": "passer_columns",
    "scramble": "passer_columns",
    "target": "receiver_columns",
    "yac": "receiver_columns",
    "rusher": "rusher_columns",
    "rush": "rusher_columns",
    "scramble_yards": "rusher_columns",
}
STRUCTURAL_COLUMNS = {"air_yards", "position", "play_type_cat", "play_outcome", "penalty_context"}
GAME_STATE_COLUMNS = set(SimulationRuntime.game_state_cols)
WEATHER_COLUMNS = {"weather_outdoor", "weather_temp_delta", "weather_wind_speed", "weather_missing"}
PV_COLUMNS = {"pv_off_qb", "pv_off_wr", "pv_off_rb", "pv_off_te", "pv_def_dl"}
# Deep completion over expected rests on about 2.6 attempts per game and barely repeats (full-season reliability
# 0.08 for defenses, 0.17 for passers), but its rare extreme values moved simulated scoring by several points.
NOISY_COLUMNS = {"deep_cpoe_allowed", "cpoe_deep", "ridge_cpoe_deep_venue_effect", "ridge_cpoe_deep_offense_effect",
                 "ridge_cpoe_deep_opponent_defense_effect", "ridge_cpoe_deep_defense_mu"}


def ridge_context_columns(model: str) -> list[str]:
    columns = []
    for target_name in dict.fromkeys(V2_MODEL_CONTEXT.get(model, [])):
        target = TARGETS_BY_NAME[target_name]
        columns.append(f"ridge_{target_name}_venue_effect")
        if target.scope == "player":
            columns.append(f"ridge_{target_name}_offense_effect")
        if target.defense_output is None or target.scope == "team":
            columns.append(f"ridge_{target_name}_opponent_defense_effect")
    return list(dict.fromkeys(columns))


def redundant_defense_columns(model: str) -> set[str]:
    """Return legacy names duplicating explicit same-fit opponent-defense mus."""
    columns = set()
    for target_name in V2_MODEL_CONTEXT.get(model, []):
        target = TARGETS_BY_NAME[target_name]
        output = target.defense_output
        has_explicit_opponent_effect = output is None or target.scope == "team"
        if output and has_explicit_opponent_effect:
            columns.add(output)
            if output == "def_pass_epa_rate":
                columns.add("opp_def_pass_epa_rate")
            elif output == "def_rush_epa_rate":
                columns.add("opp_def_rush_epa_rate")
    return columns


class _MetricHTMLParser(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows: list[list[str]] = []
        self.current: list[str] | None = None
        self.in_cell = False
        self.text: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.current = []
        elif tag in {"td", "th"}:
            self.in_cell = True
            self.text = []

    def handle_data(self, data):
        if self.in_cell:
            self.text.append(data)

    def handle_endtag(self, tag):
        if tag in {"td", "th"} and self.in_cell:
            if self.current is not None:
                self.current.append(" ".join("".join(self.text).split()))
            self.in_cell = False
        elif tag == "tr" and self.current is not None:
            if len(self.current) >= 2:
                self.rows.append(self.current)
            self.current = None


def reduced_spec(model: str, spec: dict) -> dict:
    duplicate_defense = redundant_defense_columns(model)
    selected_columns = [column for column in spec["columns"] if column not in duplicate_defense]
    groups = list(spec["groups"])
    if model == "play":
        selected_columns = list(dict.fromkeys([*selected_columns, *SimulationRuntime.passer_columns]))
        if "passer_columns" not in groups:
            groups.append("passer_columns")
    context_columns = [
        column for column in spec.get("context_columns", [])
        if (column.startswith("ridge_") and not column.endswith("_available"))
        or column in PV_COLUMNS or column in WEATHER_COLUMNS
    ]
    context_columns = [column for column in context_columns if column not in duplicate_defense]
    context_columns = list(dict.fromkeys([*context_columns, *ridge_context_columns(model)]))
    selected_columns = [column for column in selected_columns if column not in NOISY_COLUMNS]
    context_columns = [column for column in context_columns if column not in NOISY_COLUMNS]
    result = copy.deepcopy(spec)
    if model in GOAL_CAPPED_MODELS:
        selected_columns = list(dict.fromkeys([*selected_columns, *GOAL_LINE_COLUMNS]))
        result["sidecar"]["goal_capped"] = True
    result["columns"] = selected_columns
    result["groups"] = groups
    result["context_columns"] = context_columns
    result["prior_columns"] = []
    result["sidecar"]["prior_columns"] = []
    if result["sidecar"].get("kind") == "cascade":
        result["sidecar"]["context_columns"] = list(context_columns)
    return result


CASCADE_CLASSES = {"play": 4, "target": 11, "rusher": 11, "comp": 2}
# Special-teams CORN models (nflsim.features.training_data.special_teams_rows) and their runtime column groups.
SPECIAL_TEAMS_GROUPS = {
    "punt_distance": ["game_state_cols", "special_teams_cols"],
    "punt_return": ["game_state_cols", "punt_landing_cols", "special_teams_cols"],
    "kickoff": ["game_state_cols", "special_teams_cols", "kickoff_rule_cols"],
}


def game_effects_specs(specs: dict, model_manifest: dict) -> dict:
    """Flat models in place of the cascades (game-level variance moves to game_effects.json), plus sack_yards."""
    result = copy.deepcopy(specs)
    template = model_manifest["models"]["sack"]["sidecar"]
    corn = model_manifest["models"]["scramble_yards"]["sidecar"]
    keep = ("arch", "hyperparameters", "mode", "schema_version")
    for model, width in CASCADE_CLASSES.items():
        sidecar = {key: copy.deepcopy(template[key]) for key in keep}
        sidecar.update(kind="flat", artifact_prefix=model, model_name=model, dims={"input_dim": 0, "output_dim": width})
        if model == "play":
            sidecar["dims"]["class_labels"] = ["kick", "pass", "punt", "rush"]
        result[model]["sidecar"] = sidecar
        result[model]["prior_columns"] = []
    sack = copy.deepcopy(model_manifest["models"]["sack"])
    sack["prefix"] = "sack_yards"
    sack["sidecar"] = {key: copy.deepcopy(corn[key]) for key in keep}
    sack["sidecar"].update(kind="corn", artifact_prefix="sack_yards", model_name="sack_yards",
                           dims={"input_dim": 0, "output_dim": 0})
    sack.pop("source_hashes", None)
    result["sack_yards"] = reduced_spec("sack_yards", sack)
    punt = model_manifest["models"]["punt"]
    for model, groups in SPECIAL_TEAMS_GROUPS.items():
        spec = copy.deepcopy(punt)
        spec.pop("source_hashes", None)
        spec.update(prefix=model, groups=groups, prior_columns=[],
                    columns=[column for group in groups for column in getattr(SimulationRuntime, group)],
                    context_columns=[column for column in punt["context_columns"] if column in WEATHER_COLUMNS])
        spec["sidecar"] = {key: copy.deepcopy(punt["sidecar"][key]) for key in keep}
        spec["sidecar"].update(kind="corn", artifact_prefix=model, model_name=model, dims={"input_dim": 0, "output_dim": 0})
        result[model] = spec
    return result


def model_rows(model: str, spec: dict, prepared: pd.DataFrame, players: pd.DataFrame, roster: pd.DataFrame):
    """Rows, target and input columns for one model, built exactly as the refit trains them."""
    row_id = "__minimal_fit_row_id"
    prepared_for_model = prepared.copy()
    prepared_for_model[row_id] = np.arange(len(prepared_for_model), dtype=np.int64)
    ridge_columns = [column for column in prepared_for_model if column.startswith("ridge_")]
    model_inputs = prepared_for_model.drop(columns=ridge_columns)
    frame, target = model_frame(model, model_inputs, players, roster, semantics=None)
    if model in GOAL_CAPPED_MODELS:
        frame = with_goal_line_inputs(model, frame)
    if model == "play":
        frame = attach_pregame_qb_mus(frame, players, roster)
    source_ids = frame[row_id].to_numpy(dtype=np.int64)
    selected_context = prepared_for_model.set_index(row_id).reindex(source_ids)
    for column in spec["context_columns"]:
        if column not in frame:
            frame[column] = selected_context[column].to_numpy()
    columns = list(dict.fromkeys(spec["columns"] + spec["context_columns"]))
    if any(column.endswith("_available") for column in columns):
        raise ValueError(f"{model}: availability flags must not enter the refit")
    if set(columns) & redundant_defense_columns(model):
        raise ValueError(f"{model}: duplicate legacy defense aliases remain in the refit")
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{model}: reduced inputs missing from prepared frame: {missing}")
    group_column = "game_team_id" if "game_team_id" in frame else "game_id"
    if len(target) != len(frame) or not np.array_equal(source_ids, frame[row_id].to_numpy(dtype=np.int64)):
        raise ValueError(f"{model}: feature construction changed row/target alignment")
    reduced = frame.loc[:, list(dict.fromkeys([*columns, "season", "game_id", "posteam", group_column]))].copy()
    # Optional unavailable Ridge effects are neutral zeros; no availability flags enter the model.
    numeric_context = [column for column in spec["context_columns"] if column not in {"roof", "posteam_type"}]
    reduced.loc[:, numeric_context] = reduced[numeric_context].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return reduced, target, columns


def attach_pregame_qb_mus(frame: pd.DataFrame, players: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    """Attach the pregame roster QB1's Ridge mus to every offensive play row."""
    required = {"game_id", "team", "player_id", "position"}
    if not required <= set(roster):
        raise ValueError(f"Pregame roster lacks QB1 keys: {sorted(required - set(roster))}")
    qbs = roster.loc[roster.position.eq("QB"), [column for column in roster if column in {
        "game_id", "team", "player_id", "depth_team", "status", "report_status",
    }]].copy()
    if "report_status" in qbs:
        qbs = qbs.loc[~qbs.report_status.astype("string").str.casefold().eq("out").fillna(False)]
    if "status" in qbs:
        active = qbs.status.astype("string").str.upper().eq("ACT")
        active_keys = qbs.loc[active, ["game_id", "team"]].drop_duplicates()
        if not active_keys.empty:
            qbs = qbs.merge(active_keys.assign(__active=True), on=["game_id", "team"], how="left")
            qbs = qbs.loc[qbs.__active.fillna(False)].drop(columns="__active")
    if "depth_team" in qbs:
        qbs["depth_team"] = pd.to_numeric(qbs.depth_team, errors="coerce")
        qbs = qbs.sort_values(["game_id", "team", "depth_team", "player_id"], na_position="last")
    qbs = qbs.drop_duplicates(["game_id", "team"], keep="first")
    mus = SimulationRuntime.passer_columns
    table = players[["game_id", "player_id", *mus]].copy()
    table["game_id"] = table.game_id.astype(str)
    table["player_id"] = table.player_id.astype(str)
    qbs["game_id"] = qbs.game_id.astype(str)
    qbs["player_id"] = qbs.player_id.astype(str)
    qbs = qbs.merge(table, on=["game_id", "player_id"], how="left", validate="many_to_one")
    lookup = qbs.set_index(["game_id", "team"])[mus]
    keys = pd.MultiIndex.from_arrays([frame.game_id.astype(str), frame.posteam.astype(str)])
    result = frame.copy()
    for column in mus:
        result[column] = lookup[column].reindex(keys).to_numpy()
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0)
    return result


def metric_record(target, probabilities, classes):
    target = np.asarray(target, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    record = {
        "rows": int(len(target)),
        "accuracy": float(accuracy_score(target, np.argmax(probabilities, axis=1))),
        "log_loss": float(log_loss(target, probabilities, labels=np.arange(probabilities.shape[1]))),
    }
    if probabilities.shape[1] == 2 and len(np.unique(target)) == 2:
        record["auc"] = float(roc_auc_score(target, probabilities[:, 1]))
        record["rmse_prob"] = float(np.sqrt(mean_squared_error(target, probabilities[:, 1])))
    if classes and np.issubdtype(np.asarray(classes).dtype, np.number):
        values = np.asarray(classes, dtype=float)
        estimate = probabilities @ values
        actual = values[target]
        record["expected_value_rmse"] = float(np.sqrt(mean_squared_error(actual, estimate)))
        record["expected_value_mae"] = float(mean_absolute_error(actual, estimate))
    return record


def old_report_metrics(model: str, kind: str) -> tuple[str, dict]:
    if kind == "cascade":
        sidecar = json.loads((REFERENCE / f"{model}_cascade_arch.json").read_text())
        metrics = sidecar["metrics"]
        return "validation", {
            "stage1": metrics["stage1"]["val_metrics"],
            "stage3": metrics["stage3"]["val_metrics"],
        }
    report = ROOT / "training" / REFERENCE.name / f"{model}_report.html"
    if not report.exists():
        return "test", {}
    parser = _MetricHTMLParser()
    parser.feed(report.read_text(encoding="utf-8"))
    found = {}
    for key, value in parser.rows:
        try:
            found[key] = float(value)
        except ValueError:
            continue
    return "test", found


def test_comparison(model: str, kind: str, metrics: dict) -> dict | None:
    old_split, old_metrics = old_report_metrics(model, kind)
    if old_split != "test" or kind == "cascade":
        return None
    return {
        "model": model,
        "kind": kind,
        "evaluation": "test",
        "old_accuracy": old_metrics.get("accuracy"),
        "new_accuracy": metrics.get("accuracy"),
        "old_auc": old_metrics.get("auc"),
        "new_auc": metrics.get("auc"),
        "old_rmse_prob": old_metrics.get("rmse_prob"),
        "new_rmse_prob": metrics.get("rmse_prob"),
        "old_log_loss": old_metrics.get("log_loss"),
        "new_log_loss": metrics.get("log_loss"),
        "old_expected_value_rmse": old_metrics.get("rmse"),
        "new_expected_value_rmse": metrics.get("expected_value_rmse"),
    }


def main():
    parser = argparse.ArgumentParser(description="Fit reduced Ridge-only Torch feature models; no game simulation")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/minimal-ridge-refit-20261021")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--game-effects", action="store_true",
                        help="Fit flat models in place of the cascades and add sack_yards (see nflsim.models.game_effects)")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse completed model fits in this output directory after verifying their schema and artifacts")
    parser.add_argument("--summary-only", action="store_true",
                        help="Rebuild comparison outputs from verified existing fit metrics without fitting")
    parser.add_argument("--unchanged-models", nargs="*", default=[],
                        help="Models with unchanged selected inputs; reuse old test metrics instead of refitting")
    parser.add_argument("--preflight-only", action="store_true",
                        help="Validate reduced feature columns and row/target alignment without fitting")
    parser.add_argument("--player-table", type=Path, default=REFERENCE / "player_stats_wide.csv",
                        help="player_stats_wide.csv to train on, e.g. one rewritten by `nflsim.features.usage write`")
    parser.add_argument("--team-table", type=Path, default=REFERENCE / "team_stats_wide.csv",
                        help="team_stats_wide.csv to train on, e.g. one with `nflsim.features.special_teams write` columns")
    parser.add_argument("--seasons", type=int, nargs=2, default=(2017, 2024), metavar=("FIRST", "LAST"),
                        help="Training seasons (the held-out test split is drawn within them)")
    parser.add_argument("--inputs", type=Path, default=INPUTS, help="Directory with pbp.parquet and roster.parquet")
    args = parser.parse_args()
    output = output_dir(args.output_dir)
    model_manifest = json.loads((REFERENCE / "model_manifest.json").read_text())
    training_manifest = json.loads((REFERENCE / "training_manifest.json").read_text())
    selected = args.models or list(model_manifest["models"])
    specs = {model: reduced_spec(model, model_manifest["models"][model]) for model in model_manifest["models"]}
    if args.game_effects:
        specs = game_effects_specs(specs, model_manifest)
    specs = {model: specs[model] for model in (args.models or specs)}
    if args.summary_only:
        summaries, comparison = [], []
        for model, spec in specs.items():
            metrics_path = output / f"{model}_fit_metrics.json"
            if metrics_path.is_file():
                saved = json.loads(metrics_path.read_text(encoding="utf-8"))
                summaries.append(saved)
                record = test_comparison(model, spec["sidecar"]["kind"], saved.get("new_metrics", {}))
                if record is not None:
                    comparison.append(record)
                continue
            if model not in args.unchanged_models:
                raise ValueError(f"{model}: no saved fit metrics; cannot summarize incomplete refit")
            split, old_metrics = old_report_metrics(model, spec["sidecar"]["kind"])
            if split != "test" or spec["sidecar"]["kind"] == "cascade":
                raise ValueError(f"{model}: unchanged metric reuse requires an old held-out test report")
            reused = {**old_metrics, "reused_from_reference": True}
            summaries.append({"model": model, "kind": spec["sidecar"]["kind"],
                              "evaluation": "unchanged_reused_test", "new_metrics": reused,
                              "selected_columns": list(dict.fromkeys(spec["columns"] + spec["context_columns"]))})
            comparison.append({"model": model, "kind": spec["sidecar"]["kind"],
                               "evaluation": "unchanged_reused_test", "status": "unchanged_reused",
                               "old_accuracy": old_metrics.get("accuracy"), "new_accuracy": old_metrics.get("accuracy"),
                               "old_auc": old_metrics.get("auc"), "new_auc": old_metrics.get("auc"),
                               "old_rmse_prob": old_metrics.get("rmse_prob"), "new_rmse_prob": old_metrics.get("rmse_prob"),
                               "old_log_loss": old_metrics.get("log_loss"), "new_log_loss": old_metrics.get("log_loss"),
                               "old_expected_value_rmse": old_metrics.get("rmse"),
                               "new_expected_value_rmse": old_metrics.get("rmse")})
        pd.DataFrame(comparison).to_csv(output / "test_fit_comparison.csv", index=False)
        write_json(output / "fit_summary.json", {
            "reference_bundle": str(REFERENCE),
            "reference_training_manifest_sha256": digest(REFERENCE / "training_manifest.json"),
            "seed": args.seed, "epochs": args.epochs,
            "note": "Comparison CSV contains ordinary-model held-out test metrics only. Cascades use saved 10% validation metrics and are excluded from the test table. Explicit unchanged models reuse the reference test score.",
            "models": summaries,
        })
        print(f"[done] comparison={output / 'test_fit_comparison.csv'}", flush=True)
        return
    players, teams = [
        pd.read_csv(path, dtype={"game_id": str, "player_id": str})
        for path in (args.player_table, args.team_table)
    ]
    pbp = pd.read_parquet(args.inputs / "pbp.parquet")
    roster = pd.read_parquet(args.inputs / "roster.parquet")
    prepared = prepare_frame(pbp, players, teams, ISOLATED_MU_CONTEXT_V2, seasons=tuple(args.seasons))
    summaries, comparison = [], []
    if args.resume:
        for model, spec in specs.items():
            metrics_path = output / f"{model}_fit_metrics.json"
            if not metrics_path.is_file():
                continue
            saved = json.loads(metrics_path.read_text(encoding="utf-8"))
            expected_columns = list(dict.fromkeys(spec["columns"] + spec["context_columns"]))
            if saved.get("selected_columns") != expected_columns:
                raise ValueError(f"{model}: saved fit schema differs; use a fresh output directory")
            kind = spec["sidecar"]["kind"]
            prefix = spec["prefix"]
            artifact_names = (
                [f"{model}_cascade_stage{stage}_model.pt" for stage in (1, 2, 3)]
                if kind == "cascade" else [f"{prefix}_model.pt", f"{prefix}_preprocessor.pkl"]
            )
            if not all((output / name).is_file() for name in artifact_names):
                raise ValueError(f"{model}: saved fit metrics exist but artifacts are incomplete")
            summaries.append(saved)
            record = test_comparison(model, kind, saved.get("new_metrics", {}))
            if record is not None:
                comparison.append(record)
            print(f"[fit] {model}: verified saved fit; skipping", flush=True)

    write_json(output / "reduced_model_manifest.json", {
        "reference_bundle": str(REFERENCE),
        "reference_bundle_hash": digest(REFERENCE / "bundle_manifest.json"),
        "player_table": str(args.player_table),
        "player_table_sha256": digest(args.player_table),
        "team_table": str(args.team_table),
        "team_table_sha256": digest(args.team_table),
        "seasons": list(args.seasons),
        "inputs": str(args.inputs),
        "feature_profile": "original_features_plus_qb1_and_direct_player_ridge_mus_pv_weather_and_ridge_context_no_flags_v3",
        "models": {model: {"columns": spec["columns"], "context_columns": spec["context_columns"],
                            "groups": spec["groups"], "kind": spec["sidecar"]["kind"]}
                   for model, spec in specs.items()},
    })

    for model, spec in specs.items():
        reduced, target, columns = model_rows(model, spec, prepared, players, roster)
        sidecar = copy.deepcopy(spec["sidecar"])
        kind = sidecar["kind"]
        print(f"[fit] {model}: rows={len(reduced)} selected={len(columns)} kind={kind}", flush=True)
        if args.preflight_only:
            continue
        if args.resume and any(record.get("model") == model for record in summaries):
            continue
        if args.summary_only:
            continue
        if kind == "cascade":
            if model in ATTEMPT_MODELS:
                raise ValueError(f"{model}: cascades cannot record that they were trained on throws only; use --game-effects")
            cascade_config = CascadeConfig(**sidecar["config"])
            cascade_config.epochs = args.epochs
            cascade_config.seed = args.seed
            cascade_context = list(dict.fromkeys(spec["sidecar"]["context_columns"]))
            reduced["__label"] = target.astype(str).to_numpy()
            labels = ["kick", "pass", "punt", "rush"] if model == "play" else [str(i) for i in range(11)] if model in {"target", "rusher"} else ["0", "1"]
            artifacts = fit_cascade_general(
                reduced, "__label", spec["columns"], cascade_context,
                sidecar.get("categorical_columns", []), labels, cascade_config,
                group_column="game_team_id", model_label=f"minimal/{model}",
            )
            save_cascade_artifacts(artifacts, output, model, feature_groups=spec["groups"])
            metrics = artifacts.metrics
            old_split, old_metrics = old_report_metrics(model, "cascade")
            metrics_for_json = cascade_metric_record(metrics)["metrics"]
            summaries.append({"model": model, "kind": kind, "evaluation": "validation", "new_metrics": metrics_for_json})
        else:
            network, processor, probabilities, actual, classes, history, prior = fit_network(
                reduced, target, columns, sidecar, args.epochs, args.seed,
            )
            prefix = spec["prefix"]
            torch_path = output / f"{prefix}_model.pt"
            torch.save(network.state_dict(), torch_path)
            joblib.dump(processor, output / f"{prefix}_preprocessor.pkl")
            joblib.dump(spec["groups"], output / f"{prefix}_columns.pkl")
            ordinal = sidecar["kind"] == "corn" or sidecar["dims"].get("ordinal_output", False) or sidecar["dims"].get("target_mode") == "ordinal"
            if ordinal:
                pd.DataFrame(classes).to_csv(output / f"{prefix}_bins.csv", index=False)
            if sidecar["kind"] in {"flat", "corn"}:
                sidecar["dims"] = {"input_dim": network.fc1.in_features, "output_dim": len(classes), **({"class_labels": sidecar["dims"]["class_labels"]} if "class_labels" in sidecar["dims"] else {})}
            elif sidecar["kind"] == "hierarchical":
                sidecar["dims"].update(
                    play_input_dim=network.fc1.in_features-network.latent_dim,
                    prior_input_dim=network.prior_encoder.hidden[0].in_features,
                    latent_dim=network.latent_dim, output_dim=len(classes),
                    ordinal_output=network.ordinal_output,
                )
            elif sidecar["kind"] == "ngmm":
                sidecar["dims"].update(
                    input_dim=network.fixed[0].in_features,
                    context_dim=network.context[0].in_features,
                    latent_dim=network.latent_dim, output_dim=network.output_dim,
                )
            sidecar.update(seasons=sorted(reduced.season.unique().astype(int).tolist()), seed=args.seed)
            if model in ATTEMPT_MODELS:
                sidecar["row_scope"] = "attempts"
            write_json(output / f"{prefix}_arch.json", sidecar)
            metrics = metric_record(actual, probabilities, classes)
            summaries.append({"model": model, "kind": kind, "evaluation": "test", "new_metrics": metrics,
                              "best_validation_loss": min((item.get("valid_loss", np.inf) for item in history), default=None)})
            record = test_comparison(model, kind, metrics)
            if record is not None:
                comparison.append(record)
        summaries[-1]["selected_columns"] = columns
        summaries[-1]["rows"] = len(reduced)
        summaries[-1]["years"] = sorted(reduced.season.astype(int).unique().tolist())
        write_json(output / f"{model}_fit_metrics.json", summaries[-1])
        print(f"[fit] {model}: complete", flush=True)

    pd.DataFrame(comparison).to_csv(output / "test_fit_comparison.csv", index=False)
    write_json(output / "fit_summary.json", {
        "reference_bundle": str(REFERENCE),
        "reference_training_manifest_sha256": digest(REFERENCE / "training_manifest.json"),
        "seed": args.seed,
        "epochs": args.epochs,
        "note": "Comparison CSV includes only old/new ordinary-model held-out test metrics. Cascade validation results are retained in per-model summaries and excluded from the test comparison because the reference bundle has no saved cascade test fit.",
        "models": summaries,
    })
    print(f"[done] comparison={output / 'test_fit_comparison.csv'}", flush=True)


if __name__ == "__main__":
    main()
