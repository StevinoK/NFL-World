import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    base_dir = Path("yac_ordinal_sweep_runs_condor_dropout")
    leaderboard_path = base_dir / "sweep_leaderboard.csv"
    if not leaderboard_path.exists():
        raise FileNotFoundError(f"Missing leaderboard: {leaderboard_path}")

    df = pd.read_csv(leaderboard_path)

    # Keep full-model comparison runs separate from soft sigma-only runs.
    full_runs = df[df["models_run"] == "corn,condor,ce,crps,soft"].copy()
    soft_sigma_runs = df[df["models_run"] == "soft"].copy()

    if full_runs.empty:
        raise ValueError("No full model-comparison runs found.")

    full_runs["dropout_rate"] = pd.to_numeric(full_runs["dropout_rate"], errors="coerce")
    full_runs["delta_bp_vs_corn"] = full_runs["delta_vs_corn"] * 10000.0

    # Build a wide summary table for quick review.
    table_rows = []
    for dropout in sorted(full_runs["dropout_rate"].dropna().unique()):
        slice_df = full_runs[full_runs["dropout_rate"] == dropout]
        row = {"dropout_rate": float(dropout)}
        for model in ["corn", "condor", "ce", "crps", "soft"]:
            model_df = slice_df[slice_df["model"] == model]
            if model_df.empty:
                continue
            row[f"{model}_crps"] = float(model_df["crps"].iloc[0])
            row[f"{model}_delta_vs_corn"] = float(model_df["delta_vs_corn"].iloc[0])
            row[f"{model}_pct_improvement_vs_corn"] = float(model_df["pct_improvement_vs_corn"].iloc[0])
        table_rows.append(row)

    summary_table = pd.DataFrame(table_rows).sort_values("dropout_rate")
    summary_table_path = base_dir / "model_vs_baseline_by_dropout.csv"
    summary_table.to_csv(summary_table_path, index=False)

    # Build a clean leaderboard table for all runs.
    leaderboard_pretty = df[[
        "model",
        "dropout_rate",
        "soft_label_sigma",
        "crps",
        "baseline_corn_crps",
        "delta_vs_corn",
        "pct_improvement_vs_corn",
        "output_prefix",
    ]].copy()
    leaderboard_pretty = leaderboard_pretty.sort_values(
        ["pct_improvement_vs_corn", "crps"], ascending=[False, True]
    )
    leaderboard_pretty_path = base_dir / "model_vs_baseline_full_table.csv"
    leaderboard_pretty.to_csv(leaderboard_pretty_path, index=False)

    # Chart 1: grouped bars for full runs by dropout (delta vs corn in bps).
    plot_models = ["condor", "ce", "crps", "soft"]
    dropouts = sorted(full_runs["dropout_rate"].dropna().unique())
    x = np.arange(len(dropouts))
    width = 0.2

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, model in enumerate(plot_models):
        model_vals = []
        for d in dropouts:
            sub = full_runs[(full_runs["dropout_rate"] == d) & (full_runs["model"] == model)]
            model_vals.append(float(sub["delta_bp_vs_corn"].iloc[0]) if not sub.empty else np.nan)
        offset = (i - (len(plot_models) - 1) / 2.0) * width
        ax.bar(x + offset, model_vals, width=width, label=model.upper())

    ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d:.1f}" for d in dropouts])
    ax.set_xlabel("Dropout Rate")
    ax.set_ylabel("Delta vs CORN (basis points, lower is better)")
    ax.set_title("YAC Ordinal: Model Delta vs CORN Baseline by Dropout")
    ax.legend(title="Model")
    fig.tight_layout()
    chart_path = base_dir / "model_vs_baseline_by_dropout.png"
    fig.savefig(chart_path, dpi=160)
    plt.close(fig)

    # Chart 2: soft sigma sensitivity (dropout fixed by sweep config).
    if not soft_sigma_runs.empty:
        soft_sigma_runs = soft_sigma_runs.copy()
        soft_sigma_runs["soft_label_sigma"] = pd.to_numeric(soft_sigma_runs["soft_label_sigma"], errors="coerce")
        soft_sigma_runs["delta_bp_vs_corn"] = pd.to_numeric(soft_sigma_runs["delta_vs_corn"], errors="coerce") * 10000.0
        soft_sigma_runs = soft_sigma_runs.sort_values("soft_label_sigma")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            soft_sigma_runs["soft_label_sigma"],
            soft_sigma_runs["delta_bp_vs_corn"],
            marker="o",
            linewidth=2,
            color="#1f77b4",
        )
        ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
        ax.set_xlabel("Soft-Label Sigma")
        ax.set_ylabel("Delta vs CORN (basis points, lower is better)")
        ax.set_title("Soft-Label Sigma Sensitivity vs CORN Baseline")
        fig.tight_layout()
        sigma_chart_path = base_dir / "soft_sigma_vs_baseline.png"
        fig.savefig(sigma_chart_path, dpi=160)
        plt.close(fig)
    else:
        sigma_chart_path = None

    report = {
        "leaderboard_source": str(leaderboard_path),
        "full_table": str(leaderboard_pretty_path),
        "dropout_table": str(summary_table_path),
        "dropout_chart": str(chart_path),
        "soft_sigma_chart": str(sigma_chart_path) if sigma_chart_path else None,
    }
    report_path = base_dir / "model_vs_baseline_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
