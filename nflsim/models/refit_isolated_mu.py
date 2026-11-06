"""Rebuild, train and benchmark an isolated-mu experiment without overwriting a reference."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from nflsim.configuration import v2_model_manifest
from nflsim.paths import REPO, ROOT, digest, output_dir, write_json
from nflsim.features.player_value import DEFAULT_PM_DIR, GROUP_LIMITS, PM_FILE
from nflsim.semantics import ISOLATED_MU_CONTEXT_V2, ISOLATED_MU_V1, semantics_of


def implementation_hashes() -> dict[str, str]:
    paths = list(REPO.glob("*.py"))
    for directory in (ROOT, REPO / "archive" / "ridge_game_sim" / "ridge_game_sim"):
        paths.extend(path for path in directory.rglob("*.py") if "tests" not in path.parts)
    return {path.relative_to(REPO).as_posix(): digest(path) for path in sorted(set(paths))}


def validate_training_coverage(trained: dict, reference: dict) -> dict:
    if len(reference) != 20:
        raise ValueError("Reference must contain all 20 expected models")
    missing_models = sorted(set(reference) - set(trained))
    unexpected_models = sorted(set(trained) - set(reference))
    coverage = {
        model: {"expected": record["seasons"],
                "actual": trained.get(model, {}).get("seasons", []),
                "missing": sorted(set(record["seasons"]) - set(trained.get(model, {}).get("seasons", [])))}
        for model, record in reference.items()
    }
    if missing_models or unexpected_models or any(not record["expected"] or record["missing"] for record in coverage.values()):
        raise ValueError(f"Training coverage shrank: missing models={missing_models}, "
                         f"unexpected models={unexpected_models}, coverage={coverage}")
    return coverage


def training_preflight(bundle: Path, inputs: Path, reference: dict) -> dict:
    from nflsim.features.training_data import model_frame, prepare_frame

    manifest = json.loads((bundle / "model_manifest.json").read_text())
    validate_training_coverage({model: reference.get(model, {}) for model in manifest["models"]}, reference)
    features = json.loads((bundle / "feature_manifest.json").read_text())
    for name, checksum in features["tables"].items():
        if digest(bundle / name) != checksum:
            raise ValueError(f"Frozen feature table changed: {name}")
    players, teams = [pd.read_csv(bundle / name, dtype={"game_id": str, "player_id": str})
                      for name in ("player_stats_wide.csv", "team_stats_wide.csv")]
    semantics = ISOLATED_MU_CONTEXT_V2 if semantics_of(bundle) == ISOLATED_MU_CONTEXT_V2 else None
    roster = pd.read_parquet(inputs / "roster.parquet")
    pbp = prepare_frame(pd.read_parquet(inputs / "pbp.parquet"), players, teams, semantics)
    records = {}
    for model in manifest["models"]:
        frame, _ = model_frame(model, pbp, players, roster, semantics)
        counts = frame.season.value_counts().sort_index()
        records[model] = {"seasons": [int(season) for season in counts.index],
                          "rows_by_season": {str(int(season)): int(count) for season, count in counts.items()}}
    return records


def compare_pilot(candidate: Path, reference: Path, work: Path, games: int) -> None:
    from nflsim.sim.compare_2025 import compare

    left, right = [pd.read_csv(path, dtype={"game_id": str}) for path in (candidate, reference)]
    for label, frame in (("candidate", left), ("reference", right)):
        if len(frame) != games or frame.game_id.isna().any() or frame.game_id.duplicated().any():
            raise ValueError(f"{label}: pilot must contain exactly {games} unique games")
        if not frame.simulations_run.eq(200).all():
            raise ValueError(f"{label}: pilot simulations must all equal 200")
    if set(left.game_id) != set(right.game_id):
        raise ValueError("Pilot game_id sets differ")
    comparison, summary = compare(left, right, seed=42)
    summary.update(candidate_results_sha256=digest(candidate), reference_results_sha256=digest(reference),
                   selection="First N provider schedule games via identical --max-games", pilot_games=games)
    comparison.to_csv(work / "pilot_comparison.csv", index=False)
    write_json(work / "pilot_comparison_summary.json", summary)


def run_stage(stage: str, arguments: list[str], work: Path, state: dict) -> None:
    if state["identity"]["implementation_hashes"] != implementation_hashes():
        raise ValueError("Refit source code changed; use a fresh experiment")
    if stage in state["completed"]:
        return
    state.update(stage=stage, status="running", updated_at=datetime.now(timezone.utc).isoformat())
    write_json(work / "run_state.json", state)
    print(f"[{stage}] {' '.join(arguments)}", flush=True)
    with (work / f"{stage}.log").open("a", encoding="utf-8") as log:
        result = subprocess.run([sys.executable, *arguments], cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        state.update(status="failed", returncode=result.returncode)
        write_json(work / "run_state.json", state)
        raise RuntimeError(f"{stage} failed; inspect {work / (stage + '.log')}")
    state["completed"].append(stage)
    write_json(work / "run_state.json", state)
    print(f"[{stage}] complete", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ROOT.parent / "archive/nflsim_bundles/pv-weather-qb-tuned-2025")
    parser.add_argument("--name")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--opponent-context", action="store_true",
                        help="isolated_mu_context_v2: same-fit defense/venue/opponent context (default name pv-isolated-mu-context-2025)")
    parser.add_argument("--selection-from", type=Path,
                        help="Reuse a verified 2024 ridge selection (input hashes are re-checked) instead of --select-2024")
    parser.add_argument("--retained-from", type=Path)
    parser.add_argument("--pilot-games", type=int, default=18)
    parser.add_argument("--full-backtest", action="store_true")
    args = parser.parse_args()
    semantics = ISOLATED_MU_CONTEXT_V2 if args.opponent_context else ISOLATED_MU_V1
    args.name = args.name or ("pv-isolated-mu-context-2025" if args.opponent_context else "pv-isolated-mu-2025")
    if args.opponent_context and args.name == "pv-isolated-mu-2025":
        parser.error("the v1 experiment name is reserved; v2 needs its own name")
    if Path(args.name).name != args.name or args.name in {".", ".."}:
        parser.error("name must be a single directory name")
    work_path = ROOT / "outputs/refits" / args.name
    if work_path.exists() and not args.resume:
        parser.error("experiment already exists; use --resume")
    work = output_dir(work_path)
    state_path = work / "run_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"completed": []}
    reference = args.reference.resolve()
    features = json.loads((reference / "feature_manifest.json").read_text())
    training = json.loads((reference / "training_manifest.json").read_text())
    bundle = ROOT / "artifacts/bundles" / args.name
    if bundle.resolve() == reference:
        parser.error("candidate bundle must differ from immutable reference")
    validate_training_coverage(training["trained"], training["trained"])
    config = output_dir(work / "config")
    inputs = output_dir(work / "inputs")
    pm_dir = output_dir(work / "pm")
    hashes = {}
    manifest_name = "reference_model_manifest.json" if args.opponent_context else "model_manifest.json"
    sources = {manifest_name: reference / "model_manifest.json"}
    sources.update({name: ROOT / "artifacts/inputs" / name for name in
                    ("pbp.parquet", "schedule.parquet", "roster.parquet", "snap_counts.parquet")})
    sources.update({PM_FILE.format(group=group): DEFAULT_PM_DIR / PM_FILE.format(group=group)
                    for group in GROUP_LIMITS})
    for name, source in sources.items():
        checksum = digest(source)
        key = name.removesuffix(".parquet")
        expected = training["lineage"]["config"] if name == manifest_name else features["input_hashes"][key]
        if checksum != expected:
            raise ValueError(f"Reference input changed: {source}")
        hashes[name] = checksum
        destination = (config if name == manifest_name else inputs if name.endswith(".parquet") else pm_dir) / name
        if destination.exists():
            if digest(destination) != checksum:
                raise ValueError(f"Experiment snapshot changed: {destination}")
        else:
            shutil.copy2(source, destination)
    identity = {"reference": str(reference), "source_hashes": hashes,
                "implementation_hashes": implementation_hashes(), "pilot_games": args.pilot_games,
                "reference_manifests": {name: digest(reference / name) for name in
                                        ("training_manifest.json", "feature_manifest.json", "bundle_manifest.json")},
                "feature_semantics": semantics, "bundle": str(bundle)}
    if args.opponent_context:
        derived = v2_model_manifest(json.loads((config / manifest_name).read_text()))
        target = config / "model_manifest.json"
        if target.exists():
            if json.loads(target.read_text()) != derived:
                raise ValueError(f"Experiment V2 manifest changed: {target}")
        else:
            write_json(target, derived)
        identity["model_manifest_v2"] = digest(target)
    feature_args = []
    selection = config / "ridge_parameters.json"
    if args.selection_from is not None:
        checksum = digest(args.selection_from)
        if selection.exists() and digest(selection) != checksum:
            raise ValueError(f"Experiment selection snapshot changed: {selection}")
        if not selection.exists():
            shutil.copy2(args.selection_from, selection)
        identity["selection_from"] = {"path": str(args.selection_from.resolve()), "sha256": checksum}
    else:
        feature_args.append("--select-2024")
    if args.retained_from is not None:
        feature_args += ["--retained-from", str(args.retained_from)]
        identity["retained_source"] = {"path": str(args.retained_from.resolve()),
                                       "manifest_sha256": digest(args.retained_from / "retained_manifest.json")}
    if args.pilot_games < 1:
        parser.error("pilot-games must be positive")
    if "identity" in state and state["identity"] != identity:
        raise ValueError("Refit resume identity differs")
    state["identity"] = identity
    write_json(state_path, state)
    run_stage("features", ["-m", "nflsim.features.build",
              "--pbp", str(inputs / "pbp.parquet"), "--schedule", str(inputs / "schedule.parquet"),
              "--roster", str(inputs / "roster.parquet"), "--snap-counts", str(inputs / "snap_counts.parquet"),
              "--pm-dir", str(pm_dir), "--parameters", str(selection),
              *feature_args, "--feature-semantics", semantics, "--output-dir", str(bundle)], work, state)
    if semantics_of(bundle) != semantics:
        raise ValueError(f"Feature build did not publish {semantics}")
    if not (bundle / "bundle_manifest.json").exists():
        for name in ("model_manifest.json", "ridge_parameters.json"):
            shutil.copy2(config / name, bundle / name)
    preflight = training_preflight(bundle, inputs, training["trained"])
    write_json(work / "training_preflight.json", preflight)
    try:
        validate_training_coverage(preflight, training["trained"])
    except ValueError:
        state.update(status="failed_training_coverage", stage="preflight")
        write_json(state_path, state)
        raise
    train_args = ["-m", "nflsim.models.train", "--pbp", str(inputs / "pbp.parquet"),
                  "--roster", str(inputs / "roster.parquet"), "--artifact-dir", str(bundle),
                  "--model-manifest", str(config / "model_manifest.json"),
                  "--epochs", str(training["lineage"]["epochs"]),
                  "--seed", str(training["lineage"]["seed"]), "--resume"]
    if training["lineage"].get("stage2_epochs") is not None:
        train_args += ["--stage2-epochs", str(training["lineage"]["stage2_epochs"])]
    run_stage("training", train_args, work, state)
    trained = json.loads((bundle / "training_manifest.json").read_text())["trained"]
    write_json(work / "training_coverage.json", {"trained": trained, "reference": training["trained"]})
    try:
        coverage = validate_training_coverage(trained, training["trained"])
    except ValueError:
        state.update(status="failed_training_coverage", stage="coverage")
        write_json(state_path, state)
        raise
    write_json(work / "training_coverage.json", coverage)
    run_stage("validation", ["-m", "nflsim.bundle", "--artifact-dir", str(bundle)], work, state)
    run_stage("smoke", ["-m", "nflsim.sim.benchmark_2025", "--backend", "local",
              "--artifact-dir", str(bundle), "--simulations", "5", "--max-games", "1", "--seed", "42",
              "--max-containers", "4", "--run-id", args.name + "-smoke", "--resume"], work, state)
    pilot_runs = {}
    for label, artifact_dir in (("reference", reference), ("candidate", bundle)):
        run_id = args.name + f"-{label}-pilot-200"
        run_stage(f"{label}_pilot", ["-m", "nflsim.sim.benchmark_2025", "--backend", "local",
                  "--artifact-dir", str(artifact_dir), "--simulations", "200", "--seed", "42",
                  "--max-containers", "4", "--max-games", str(args.pilot_games),
                  "--run-id", run_id, "--resume"], work, state)
        pilot_runs[label] = ROOT / "benchmarks/2025" / run_id / "backtest_2025_results.csv"
    try:
        compare_pilot(pilot_runs["candidate"], pilot_runs["reference"], work, args.pilot_games)
    except ValueError:
        state.update(status="failed_pilot_comparison", stage="pilot_comparison")
        write_json(state_path, state)
        raise
    if args.full_backtest:
        run_stage("backtest", ["-m", "nflsim.sim.benchmark_2025", "--backend", "local",
                  "--artifact-dir", str(bundle), "--simulations", "200", "--seed", "42",
                  "--max-containers", "4", "--run-id", args.name + "-200", "--resume"], work, state)
    state.update(status="complete", stage="complete", updated_at=datetime.now(timezone.utc).isoformat())
    write_json(state_path, state)


if __name__ == "__main__":
    main()