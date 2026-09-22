import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


MODEL_TO_SUMMARY_KEY = {
    "corn": "corn_status_quo_crps",
    "condor": "condor_coral_crps",
    "ce": "ce_softmax_crps",
    "crps": "crps_trained_model_crps",
    "soft": "soft_label_ordinal_crps",
    "xgb": "xgb_ordinal_crps",
}


@dataclass
class RunSpec:
    train_start: int
    train_end: int
    test_season: int
    sigma: float
    dropout_rate: float
    models: str
    output_prefix: str


def parse_windows(raw: str) -> List[Tuple[int, int]]:
    windows: List[Tuple[int, int]] = []
    for chunk in [item.strip() for item in raw.split(",") if item.strip()]:
        if "-" not in chunk:
            raise ValueError(f"Invalid train window '{chunk}'. Expected format like 2022-2024.")
        left, right = chunk.split("-", 1)
        start = int(left)
        end = int(right)
        if start > end:
            raise ValueError(f"Invalid train window '{chunk}': start must be <= end.")
        windows.append((start, end))

    if not windows:
        raise ValueError("At least one train window is required.")
    return windows


def parse_sigmas(raw: str) -> List[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one sigma value is required.")
    return values


def sigma_token(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def parse_dropouts(raw: str) -> List[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one dropout value is required.")
    for val in values:
        if val < 0.0 or val >= 1.0:
            raise ValueError(f"Dropout value must be in [0, 1). Got {val}")
    return values


def build_run_specs(
    windows: List[Tuple[int, int]],
    test_season: int,
    sigmas: List[float],
    dropouts: List[float],
    baseline_sigma: float,
    sigma_sweep_dropout: float,
    base_prefix: str,
    include_soft_sigma_runs: bool,
) -> List[RunSpec]:
    specs: List[RunSpec] = []

    for train_start, train_end in windows:
        for dropout_rate in dropouts:
            all_prefix = (
                f"{base_prefix}_{train_start}_{train_end}_drop_{sigma_token(dropout_rate)}_all"
            )
            specs.append(
                RunSpec(
                    train_start=train_start,
                    train_end=train_end,
                    test_season=test_season,
                    sigma=baseline_sigma,
                    dropout_rate=dropout_rate,
                    models="corn,condor,ce,crps,soft,xgb",
                    output_prefix=all_prefix,
                )
            )

        if include_soft_sigma_runs:
            for sigma in sigmas:
                if abs(sigma - baseline_sigma) < 1e-12:
                    continue
                soft_prefix = (
                    f"{base_prefix}_{train_start}_{train_end}_drop_{sigma_token(sigma_sweep_dropout)}"
                    f"_soft_sigma_{sigma_token(sigma)}"
                )
                specs.append(
                    RunSpec(
                        train_start=train_start,
                        train_end=train_end,
                        test_season=test_season,
                        sigma=sigma,
                        dropout_rate=sigma_sweep_dropout,
                        models="soft",
                        output_prefix=soft_prefix,
                    )
                )

    return specs


def run_compare_script(compare_script: Path, python_exec: str, max_epochs: int, batch_size: int, seed: int, spec: RunSpec) -> None:
    command = [
        python_exec,
        str(compare_script),
        "--train-start",
        str(spec.train_start),
        "--train-end",
        str(spec.train_end),
        "--test-season",
        str(spec.test_season),
        "--max-epochs",
        str(max_epochs),
        "--batch-size",
        str(batch_size),
        "--seed",
        str(seed),
        "--soft-label-sigma",
        str(spec.sigma),
        "--dropout-rate",
        str(spec.dropout_rate),
        "--models",
        spec.models,
        "--output-prefix",
        spec.output_prefix,
    ]

    print(
        f"\n[run] window={spec.train_start}-{spec.train_end} models={spec.models} "
        f"sigma={spec.sigma} dropout={spec.dropout_rate} prefix={spec.output_prefix}"
    )
    subprocess.run(command, check=True)


def load_summary(summary_path: Path) -> Dict:
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def build_leaderboard_rows(specs: List[RunSpec], summaries: Dict[str, Dict]) -> List[Dict]:
    baseline_by_window: Dict[Tuple[int, int, float], float] = {}

    for spec in specs:
        if "corn" in spec.models and "condor" in spec.models and "ce" in spec.models and "crps" in spec.models:
            summary = summaries[spec.output_prefix]
            baseline = summary.get("corn_status_quo_crps")
            if baseline is not None:
                baseline_by_window[(spec.train_start, spec.train_end, spec.dropout_rate)] = float(baseline)

    rows: List[Dict] = []
    for spec in specs:
        summary = summaries[spec.output_prefix]
        baseline = baseline_by_window.get((spec.train_start, spec.train_end, spec.dropout_rate))

        for model_key, metric_key in MODEL_TO_SUMMARY_KEY.items():
            metric_value = summary.get(metric_key)
            if metric_value is None:
                continue

            metric_value = float(metric_value)
            delta_vs_corn = None
            pct_improvement_vs_corn = None
            if baseline is not None:
                delta_vs_corn = metric_value - baseline
                if baseline != 0:
                    pct_improvement_vs_corn = ((baseline - metric_value) / baseline) * 100.0

            rows.append(
                {
                    "output_prefix": spec.output_prefix,
                    "train_start": spec.train_start,
                    "train_end": spec.train_end,
                    "test_season": spec.test_season,
                    "models_run": spec.models,
                    "soft_label_sigma": spec.sigma,
                    "dropout_rate": spec.dropout_rate,
                    "model": model_key,
                    "crps": metric_value,
                    "baseline_corn_crps": baseline,
                    "delta_vs_corn": delta_vs_corn,
                    "pct_improvement_vs_corn": pct_improvement_vs_corn,
                }
            )

    rows.sort(
        key=lambda row: (
            -1e9 if row["pct_improvement_vs_corn"] is None else -row["pct_improvement_vs_corn"],
            row["crps"],
        )
    )
    return rows


def write_leaderboard(output_dir: Path, leaderboard_rows: List[Dict], manifest_rows: List[Dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "sweep_manifest.json"
    leaderboard_json_path = output_dir / "sweep_leaderboard.json"
    leaderboard_csv_path = output_dir / "sweep_leaderboard.csv"

    manifest_path.write_text(json.dumps(manifest_rows, indent=2), encoding="utf-8")
    leaderboard_json_path.write_text(json.dumps(leaderboard_rows, indent=2), encoding="utf-8")

    if leaderboard_rows:
        fieldnames = list(leaderboard_rows[0].keys())
        with leaderboard_csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(leaderboard_rows)
    else:
        leaderboard_csv_path.write_text("", encoding="utf-8")

    print(f"\nWrote manifest: {manifest_path}")
    print(f"Wrote leaderboard JSON: {leaderboard_json_path}")
    print(f"Wrote leaderboard CSV: {leaderboard_csv_path}")


def write_failures(output_dir: Path, failures: List[Dict]) -> None:
    if not failures:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    failures_path = output_dir / "sweep_failures.json"
    failures_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
    print(f"Wrote failures: {failures_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YAC ordinal recency/objective/sigma sweep and build leaderboard.")
    parser.add_argument("--compare-script", type=str, default="compare_yac_ordinal_crps.py")
    parser.add_argument("--python-exec", type=str, default=sys.executable)
    parser.add_argument("--train-windows", type=str, default="2016-2024")
    parser.add_argument("--test-season", type=int, default=2025)
    parser.add_argument("--soft-sigmas", type=str, default="0.5,1.0,1.5,2.0,3.0")
    parser.add_argument("--dropout-rates", type=str, default="0.1,0.3,0.5")
    parser.add_argument("--baseline-sigma", type=float, default=1.5)
    parser.add_argument("--sigma-sweep-dropout", type=float, default=0.3)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="yac_ordinal_sweep_runs")
    parser.add_argument("--output-prefix-base", type=str, default="yac_ordinal_sweep")
    parser.add_argument("--skip-soft-sigma-runs", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue sweep when an individual run fails and log failures to sweep_failures.json.",
    )
    args = parser.parse_args()

    compare_script = Path(args.compare_script).resolve()
    if not compare_script.exists():
        raise FileNotFoundError(f"compare script not found: {compare_script}")

    sigmas = parse_sigmas(args.soft_sigmas)
    dropouts = parse_dropouts(args.dropout_rates)
    if args.sigma_sweep_dropout < 0.0 or args.sigma_sweep_dropout >= 1.0:
        raise ValueError("--sigma-sweep-dropout must be in [0, 1).")

    fixed_train_window = (2016, 2024)
    fixed_test_season = 2025
    if args.train_windows != "2016-2024" or args.test_season != fixed_test_season:
        print(
            f"[info] Overriding custom split {args.train_windows}/{args.test_season} to the fixed benchmark split "
            f"{fixed_train_window[0]}-{fixed_train_window[1]}/{fixed_test_season}."
        )
    args.train_windows = "2016-2024"
    args.test_season = fixed_test_season

    windows = [fixed_train_window]
    summaries: Dict[str, Dict] = {}
    manifest_rows: List[Dict] = []
    failed_rows: List[Dict] = []

    specs = build_run_specs(
        windows=windows,
        test_season=args.test_season,
        sigmas=sigmas,
        dropouts=dropouts,
        baseline_sigma=args.baseline_sigma,
        sigma_sweep_dropout=args.sigma_sweep_dropout,
        base_prefix=args.output_prefix_base,
        include_soft_sigma_runs=not args.skip_soft_sigma_runs,
    )

    print(f"Using compare script: {compare_script}")
    print(f"Using python executable: {args.python_exec}")
    print(f"Using fixed split: train={fixed_train_window}, test={fixed_test_season}")

    for spec in specs:
        try:
            run_compare_script(
                compare_script=compare_script,
                python_exec=args.python_exec,
                max_epochs=args.max_epochs,
                batch_size=args.batch_size,
                seed=args.seed,
                spec=spec,
            )

            summary_path = Path(f"{spec.output_prefix}_summary.json")
            summary = load_summary(summary_path)
            summaries[spec.output_prefix] = summary
            manifest_rows.append(
                {
                    "output_prefix": spec.output_prefix,
                    "summary_path": str(summary_path),
                    "train_start": spec.train_start,
                    "train_end": spec.train_end,
                    "test_season": spec.test_season,
                    "models": spec.models,
                    "soft_label_sigma": spec.sigma,
                    "dropout_rate": spec.dropout_rate,
                }
            )
        except subprocess.CalledProcessError as exc:
            failed_rows.append(
                {
                    "output_prefix": spec.output_prefix,
                    "train_start": spec.train_start,
                    "train_end": spec.train_end,
                    "test_season": spec.test_season,
                    "models": spec.models,
                    "soft_label_sigma": spec.sigma,
                    "dropout_rate": spec.dropout_rate,
                    "returncode": int(exc.returncode),
                    "command": exc.cmd,
                }
            )
            print(
                f"[warn] run failed: prefix={spec.output_prefix} returncode={exc.returncode}."
            )
            if not args.continue_on_error:
                raise

    output_dir = Path(args.output_dir)
    write_failures(output_dir=output_dir, failures=failed_rows)

    if not summaries:
        raise RuntimeError(
            "All sweep runs failed. Check sweep_failures.json and retry with a verified python executable."
        )

    leaderboard_rows = build_leaderboard_rows(specs=specs, summaries=summaries)

    write_leaderboard(output_dir=output_dir, leaderboard_rows=leaderboard_rows, manifest_rows=manifest_rows)

    print("\nTop 10 by percent improvement vs baseline CORN:")
    for idx, row in enumerate(leaderboard_rows[:10], start=1):
        pct = row["pct_improvement_vs_corn"]
        pct_txt = "n/a" if pct is None else f"{pct:.3f}%"
        print(
            f"{idx:>2}. model={row['model']:<5} window={row['train_start']}-{row['train_end']} "
            f"sigma={row['soft_label_sigma']:<4} dropout={row['dropout_rate']:<4} crps={row['crps']:.6f} "
            f"delta_vs_corn={row['delta_vs_corn']} pct={pct_txt}"
        )


if __name__ == "__main__":
    main()
