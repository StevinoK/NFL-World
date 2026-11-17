import json
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from nflsim.features import build as build_features
from nflsim.models import refit_isolated_mu as runner
from nflsim.paths import digest


@pytest.fixture(autouse=True)
def local_json_writes(monkeypatch):
    def write_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    monkeypatch.setattr(runner, "write_json", write_json)
    monkeypatch.setattr(build_features, "write_json", write_json)


@pytest.fixture
def reference_models():
    return {f"model_{index}": {"seasons": [2017, 2018, 2024]} for index in range(20)}


def test_coverage_accepts_all_models(reference_models):
    coverage = runner.validate_training_coverage(reference_models, reference_models)
    assert len(coverage) == 20
    assert all(not record["missing"] for record in coverage.values())


@pytest.mark.parametrize("defect", ["model", "season", "reference"])
def test_coverage_rejects_missing_models_and_seasons(reference_models, defect):
    trained = {model: dict(record) for model, record in reference_models.items()}
    if defect == "model":
        trained.pop("model_0")
    elif defect == "season":
        trained["model_0"]["seasons"] = [2024]
    else:
        reference_models.pop("model_0")
    with pytest.raises(ValueError, match="20 expected models|Training coverage shrank"):
        runner.validate_training_coverage(trained, reference_models)


def test_stage_rejects_changed_code_even_when_completed(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "implementation_hashes", lambda: {"runner.py": "new"})
    state = {"completed": ["features"], "identity": {"implementation_hashes": {"runner.py": "old"}}}
    with pytest.raises(ValueError, match="source code changed"):
        runner.run_stage("features", [], tmp_path, state)


@pytest.fixture
def retained_fixture(tmp_path):
    source, output = tmp_path / "retained", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    frozen_input = tmp_path / "pbp.csv"
    implementation = tmp_path / "retained_implementation.py"
    frozen_input.write_text("game_id,value\n2024_01_A_B,1\n")
    implementation.write_text("version = 1\n")
    tables = {
        "retained_player_stats.csv": "game_id,player_id,value\n2024_01_A_B,001,2\n2025_01_A_B,002,3\n",
        "retained_team_stats.csv": "game_id,team,value\n2024_01_A_B,A,4\n2025_01_A_B,B,5\n",
        "retained_diagnostics.csv": "metric,rows\nretained,2\n",
    }
    for name, text in tables.items():
        (source / name).write_text(text)
    manifest = {"status": "complete", "input_hashes": {str(frozen_input): digest(frozen_input)},
                "prediction_seasons": [2024, 2025], "excluded_sources": ["cpoe"],
                "implementation_hashes": {str(implementation): digest(implementation)},
                "tables": {name: digest(source / name) for name in tables}}
    (source / "retained_manifest.json").write_text(json.dumps(manifest))
    return SimpleNamespace(source=source, output=output, frozen_input=frozen_input,
                           implementation=implementation, manifest=manifest)


def test_retained_reuse_copies_verified_tables_without_regeneration(monkeypatch, retained_fixture):
    def no_regeneration(*args, **kwargs):
        pytest.fail("Retained regeneration must not run")

    monkeypatch.setattr(build_features, "regenerate_retained_features", no_regeneration)
    fixture = retained_fixture
    players, teams, diagnostics = build_features.reuse_retained_tables(
        fixture.source, fixture.output, [fixture.frozen_input], [2024], ["cpoe"])
    assert players.player_id.tolist() == ["001"]
    assert teams.team.tolist() == ["A"]
    assert diagnostics.rows.tolist() == [2]
    for name, checksum in fixture.manifest["tables"].items():
        assert digest(fixture.output / name) == checksum
    assert json.loads((fixture.output / "retained_manifest.json").read_text())["reused_from"] == str(fixture.source.resolve())


@pytest.mark.parametrize("changed,message", [
    ("input", "inputs do not match"), ("table", "Retained table changed"),
    ("implementation", "Retained implementation changed"),
    ("seasons", "does not cover"), ("excluded", "different ridge metrics"),
    ("status", "incomplete"),
])
def test_retained_reuse_rejects_changed_provenance(retained_fixture, changed, message):
    fixture = retained_fixture
    seasons, excluded = [2024], ["cpoe"]
    if changed in {"input", "table", "implementation"}:
        path = {"input": fixture.frozen_input, "table": fixture.source / "retained_player_stats.csv",
                "implementation": fixture.implementation}[changed]
        path.write_text(path.read_text() + "changed\n")
    elif changed == "seasons":
        seasons = [2023]
    elif changed == "excluded":
        excluded = ["yac"]
    else:
        fixture.manifest["status"] = "running"
        (fixture.source / "retained_manifest.json").write_text(json.dumps(fixture.manifest))
    with pytest.raises(ValueError, match=message):
        build_features.reuse_retained_tables(fixture.source, fixture.output, [fixture.frozen_input], seasons, excluded)


@pytest.fixture
def pilot_results(tmp_path):
    frame = pd.DataFrame([
        {"game_id": f"2025_01_{index}", "week": 1, "home_team": "BUF", "away_team": "NYJ",
         "actual_home_score": 24, "actual_away_score": 20, "actual_spread": 4, "actual_total": 44,
         "sim_spread": 3, "sim_total": 43, "sim_home_win_prob": 0.6,
         "sim_home_median": 23, "sim_away_median": 20, "simulations_run": 200}
        for index in range(2)])
    candidate, reference = tmp_path / "candidate.csv", tmp_path / "reference.csv"
    frame.to_csv(candidate, index=False)
    frame.iloc[::-1].to_csv(reference, index=False)
    return candidate, reference


def test_pilot_comparison_persists_paired_metrics_and_preserves_reference(tmp_path, pilot_results):
    candidate, reference = pilot_results
    before = digest(reference)
    runner.compare_pilot(candidate, reference, tmp_path, 2)
    summary = json.loads((tmp_path / "pilot_comparison_summary.json").read_text())
    assert summary["common_games"] == 2
    assert summary["unequal_simulation_counts"] is False
    assert summary["seed"] == 42
    assert summary["reference_results_sha256"] == before == digest(reference)
    comparison = pd.read_csv(tmp_path / "pilot_comparison.csv")
    assert comparison.delta_ridge_minus_baseline.eq(0).all()


@pytest.mark.parametrize("defect,message", [
    ("games", "game_id sets differ"), ("count", "exactly 2 unique games"),
    ("duplicate", "exactly 2 unique games"), ("simulations", "must all equal 200"),
    ("both_simulations", "must all equal 200"), ("actuals", "Incompatible paired games"),
])
def test_pilot_rejects_unpaired_results(tmp_path, pilot_results, defect, message):
    candidate, reference = pilot_results
    frame = pd.read_csv(candidate)
    if defect == "games":
        frame.loc[0, "game_id"] = "different"
    elif defect == "count":
        frame = frame.iloc[:1]
    elif defect == "duplicate":
        frame.loc[1, "game_id"] = frame.loc[0, "game_id"]
    elif defect in {"simulations", "both_simulations"}:
        frame.loc[0, "simulations_run"] = 199
        if defect == "both_simulations":
            frame.to_csv(reference, index=False)
    else:
        frame.loc[0, ["actual_home_score", "actual_spread", "actual_total"]] = [25, 5, 45]
    frame.to_csv(candidate, index=False)
    with pytest.raises(ValueError, match=message):
        runner.compare_pilot(candidate, reference, tmp_path, 2)
    assert not (tmp_path / "pilot_comparison_summary.json").exists()


@pytest.fixture
def runner_setup(monkeypatch, tmp_path, reference_models):
    root = tmp_path / "nflsim"
    reference = root / "artifacts/bundles/reference"
    inputs = root / "artifacts/inputs"
    pm = root / "pm_source"
    for directory in (reference, inputs, pm):
        directory.mkdir(parents=True)
    def output_dir(path):
        path.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "DEFAULT_PM_DIR", pm)
    monkeypatch.setattr(runner, "output_dir", output_dir)
    code = {"runner.py": "original"}
    monkeypatch.setattr(runner, "implementation_hashes", lambda: dict(code))
    model_manifest = reference / "model_manifest.json"
    model_manifest.write_text(json.dumps({"models": {model: {} for model in reference_models}}))
    hashes = {}
    for name in ("pbp.parquet", "schedule.parquet", "roster.parquet", "snap_counts.parquet"):
        (inputs / name).write_text("tiny frozen input\n")
        hashes[name.removesuffix(".parquet")] = digest(inputs / name)
    for group in runner.GROUP_LIMITS:
        name = runner.PM_FILE.format(group=group)
        (pm / name).write_text("tiny frozen PM\n")
        hashes[name] = digest(pm / name)
    training = {"trained": reference_models, "lineage": {"config": digest(model_manifest), "epochs": 100, "seed": 42}}
    runner.write_json(reference / "feature_manifest.json", {"input_hashes": hashes})
    runner.write_json(reference / "training_manifest.json", training)
    runner.write_json(reference / "bundle_manifest.json", {"immutable": True})
    monkeypatch.setattr(runner, "semantics_of", lambda path: runner.ISOLATED_MU_V1)
    monkeypatch.setattr(runner, "training_preflight", lambda *args: reference_models)
    stages, comparisons = [], []

    def run_stage(stage, arguments, work, state):
        stages.append((stage, arguments))
        bundle = root / "artifacts/bundles/experiment"
        if stage == "features":
            output_dir(bundle)
            (work / "config/ridge_parameters.json").write_text("{}")
        if stage == "training":
            runner.write_json(bundle / "training_manifest.json", training)
        state["completed"].append(stage)

    monkeypatch.setattr(runner, "run_stage", run_stage)
    monkeypatch.setattr(runner, "compare_pilot", lambda *args: comparisons.append(args))
    argv = ["refit", "--reference", str(reference), "--name", "experiment"]
    monkeypatch.setattr(sys, "argv", argv)
    return SimpleNamespace(root=root, reference=reference, stages=stages, comparisons=comparisons, argv=argv, code=code)


@pytest.mark.parametrize("full", [False, True])
def test_runner_paired_pilot_and_full_backtest_opt_in(runner_setup, full):
    fixture = runner_setup
    before = {path.name: digest(path) for path in fixture.reference.iterdir()}
    if full:
        fixture.argv.append("--full-backtest")
    runner.main()
    stages = dict(fixture.stages)
    assert ("backtest" in stages) is full
    assert list(stages).index("training") < list(stages).index("reference_pilot")
    for label in ("reference", "candidate"):
        arguments = stages[f"{label}_pilot"]
        for flag, value in (("--max-games", "18"), ("--simulations", "200"), ("--seed", "42"),
                            ("--max-containers", "4"), ("--run-id", f"experiment-{label}-pilot-200")):
            assert arguments[arguments.index(flag) + 1] == value
    assert stages["reference_pilot"][stages["reference_pilot"].index("--artifact-dir") + 1] == str(fixture.reference.resolve())
    assert len(fixture.comparisons) == 1
    assert fixture.comparisons[0][0].parent.name == "experiment-candidate-pilot-200"
    assert fixture.comparisons[0][1].parent.name == "experiment-reference-pilot-200"
    assert before == {path.name: digest(path) for path in fixture.reference.iterdir()}


def test_runner_resume_rejects_code_and_pilot_size_changes(runner_setup):
    fixture = runner_setup
    runner.main()
    fixture.stages.clear()
    fixture.argv.append("--resume")
    fixture.code["runner.py"] = "changed"
    with pytest.raises(ValueError, match="resume identity differs"):
        runner.main()
    fixture.code["runner.py"] = "original"
    fixture.argv.extend(["--pilot-games", "3"])
    with pytest.raises(ValueError, match="resume identity differs"):
        runner.main()
    assert fixture.stages == []


def test_runner_preflight_stops_before_training(monkeypatch, runner_setup, reference_models):
    fixture = runner_setup
    reduced = {model: dict(record) for model, record in reference_models.items()}
    reduced["model_0"] = {"seasons": [2024], "rows_by_season": {"2024": 2}}
    monkeypatch.setattr(runner, "training_preflight", lambda *args: reduced)
    with pytest.raises(ValueError, match="Training coverage shrank"):
        runner.main()
    assert [stage for stage, _ in fixture.stages] == ["features"]
    work = fixture.root / "outputs/refits/experiment"
    assert json.loads((work / "run_state.json").read_text())["stage"] == "preflight"
    assert json.loads((work / "training_preflight.json").read_text())["model_0"]["rows_by_season"] == {"2024": 2}


def test_preflight_uses_training_frames_and_records_real_counts(monkeypatch, tmp_path, reference_models):
    bundle, inputs = tmp_path / "bundle", tmp_path / "inputs"
    bundle.mkdir()
    inputs.mkdir()
    runner.write_json(bundle / "model_manifest.json", {"models": {model: {} for model in reference_models}})
    for name in ("player_stats_wide.csv", "team_stats_wide.csv"):
        (bundle / name).write_text("game_id,player_id\n2024_01_A_B,001\n")
    runner.write_json(bundle / "feature_manifest.json", {"tables": {
        name: digest(bundle / name) for name in ("player_stats_wide.csv", "team_stats_wide.csv")}})
    monkeypatch.setattr(runner, "semantics_of", lambda path: runner.ISOLATED_MU_CONTEXT_V2)
    monkeypatch.setattr(pd, "read_parquet", lambda path: pd.DataFrame({"input": [path.name]}))
    calls = []

    def prepare_frame(pbp, players, teams, semantics):
        assert semantics == runner.ISOLATED_MU_CONTEXT_V2
        assert pbp.input.tolist() == ["pbp.parquet"]
        assert players.player_id.tolist() == ["001"]
        return pd.DataFrame({"prepared": [True]})

    def model_frame(model, pbp, players, roster, semantics):
        calls.append(model)
        assert pbp.prepared.all()
        assert roster.input.tolist() == ["roster.parquet"]
        assert semantics == runner.ISOLATED_MU_CONTEXT_V2
        return pd.DataFrame({"season": [2017, 2018, 2018, 2024]}), None

    monkeypatch.setitem(sys.modules, "nflsim.features.training_data",
                        SimpleNamespace(prepare_frame=prepare_frame, model_frame=model_frame))
    records = runner.training_preflight(bundle, inputs, reference_models)
    assert set(calls) == set(reference_models)
    assert records["model_0"] == {"seasons": [2017, 2018, 2024],
                                  "rows_by_season": {"2017": 1, "2018": 2, "2024": 1}}
    runner.validate_training_coverage(records, reference_models)


def test_failed_pilot_comparison_blocks_full_backtest(monkeypatch, runner_setup):
    fixture = runner_setup
    fixture.argv.append("--full-backtest")

    def reject(*args):
        raise ValueError("Pilot game_id sets differ")

    monkeypatch.setattr(runner, "compare_pilot", reject)
    with pytest.raises(ValueError, match="game_id sets differ"):
        runner.main()
    assert "backtest" not in dict(fixture.stages)
    work = fixture.root / "outputs/refits/experiment"
    assert json.loads((work / "run_state.json").read_text())["status"] == "failed_pilot_comparison"