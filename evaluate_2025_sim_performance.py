from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


WORKSPACE_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_PATH = WORKSPACE_DIR / "backtest_2025_results.csv"
DEFAULT_ATTEMPT_DIAG_PATH = WORKSPACE_DIR / "backtest_2025_attempt_diagnostics.csv"
DEFAULT_ATTEMPT_SUMMARY_PATH = WORKSPACE_DIR / "backtest_2025_attempt_summary.json"
DEFAULT_PLAYCALL_SUMMARY_PATH = WORKSPACE_DIR / "backtest_2025_playcall_bucket_summary.json"
DEFAULT_TEAM_CALIB_PATH = WORKSPACE_DIR / "backtest_2025_team_offense_calibration.csv"
DEFAULT_BACKTEST_SUMMARY_PATH = WORKSPACE_DIR / "backtest_2025_summary.json"
DEFAULT_RAW_RESULTS_DIR = WORKSPACE_DIR / "backtest_2025_game_results"

DEFAULT_OUT_DIR = WORKSPACE_DIR / "performance_eval_2025"


@dataclass
class Paths:
    results: Path
    attempt_diag: Path
    attempt_summary: Path
    playcall_summary: Path
    team_calib: Path
    backtest_summary: Path
    raw_results_dir: Path
    out_dir: Path


def load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def safe_div(num: float, den: float) -> float:
    if den == 0 or not np.isfinite(den):
        return float("nan")
    return float(num / den)


def safe_mean(values: Iterable[float]) -> float:
    arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def metric_row(model: str, metric: str, value: object, note: str = "") -> Dict[str, object]:
    return {
        "model": model,
        "metric": metric,
        "value": value,
        "note": note,
    }


def crps_empirical(samples: np.ndarray, observation: float) -> float:
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0 or not np.isfinite(observation):
        return float("nan")
    a = float(np.mean(np.abs(samples - observation)))
    pairwise = np.abs(samples[:, None] - samples[None, :])
    b = float(np.mean(pairwise))
    return a - 0.5 * b


def load_actual_ordinal_reference(results: pd.DataFrame) -> Dict[str, Optional[float]]:
    if results.empty or "game_id" not in results.columns:
        return {}

    try:
        import nfl_data_py as nfl  # type: ignore
    except Exception:
        return {}

    seasons: List[int] = []
    if "week" in results.columns:
        # Evaluate script is scoped to one season run in practice; default to 2025 if absent.
        seasons = [2025]
    if not seasons:
        seasons = [2025]

    try:
        pbp = nfl.import_pbp_data(seasons)
    except Exception:
        try:
            pbp = nfl.import_pbp_data(years=seasons)
        except Exception:
            return {}
    if pbp is None or pbp.empty:
        return {}

    game_ids = set(results["game_id"].astype(str))
    pbp = pbp[pbp["game_id"].astype(str).isin(game_ids)].copy()
    if pbp.empty:
        return {}

    ref: Dict[str, Optional[float]] = {}

    # Air yards per pass attempt.
    if {"pass_attempt", "air_yards"}.issubset(pbp.columns):
        pass_mask = pd.to_numeric(pbp["pass_attempt"], errors="coerce").fillna(0.0) > 0
        air = pd.to_numeric(pbp.loc[pass_mask, "air_yards"], errors="coerce")
        air = air[np.isfinite(air)]
        ref["actual_air_yards_per_attempt"] = float(air.mean()) if not air.empty else None

    # YAC yards per completion.
    yac_col = None
    for candidate in ("yards_after_catch", "yac_yards"):
        if candidate in pbp.columns:
            yac_col = candidate
            break
    if yac_col is not None and "complete_pass" in pbp.columns:
        comp_mask = pd.to_numeric(pbp["complete_pass"], errors="coerce").fillna(0.0) > 0
        yac = pd.to_numeric(pbp.loc[comp_mask, yac_col], errors="coerce")
        yac = yac[np.isfinite(yac)]
        ref["actual_yac_yards_per_completion"] = float(yac.mean()) if not yac.empty else None

    # Complete pass yards per completion.
    if {"complete_pass", "yards_gained"}.issubset(pbp.columns):
        comp_mask = pd.to_numeric(pbp["complete_pass"], errors="coerce").fillna(0.0) > 0
        comp_yds = pd.to_numeric(pbp.loc[comp_mask, "yards_gained"], errors="coerce")
        comp_yds = comp_yds[np.isfinite(comp_yds)]
        ref["actual_complete_pass_yards_per_completion"] = float(comp_yds.mean()) if not comp_yds.empty else None

    # Punt yards per punt.
    punt_col = None
    if "punt_yards" in pbp.columns:
        punt_col = "punt_yards"
    elif "kick_distance" in pbp.columns:
        # nflverse often stores punt distance under kick_distance.
        punt_col = "kick_distance"
    if punt_col is not None:
        if "punt_attempt" in pbp.columns:
            punt_mask = pd.to_numeric(pbp["punt_attempt"], errors="coerce").fillna(0.0) > 0
        elif "play_type" in pbp.columns:
            punt_mask = pbp["play_type"].astype(str).str.lower().eq("punt")
        else:
            punt_mask = pd.Series(False, index=pbp.index)
        punts = pd.to_numeric(pbp.loc[punt_mask, punt_col], errors="coerce")
        punts = punts[np.isfinite(punts)]
        ref["actual_punt_yards_per_punt"] = float(punts.mean()) if not punts.empty else None

    # Time elapsed per timed play from within-game deltas in game_seconds_remaining.
    if "game_seconds_remaining" in pbp.columns:
        sec = pd.to_numeric(pbp["game_seconds_remaining"], errors="coerce")
        work = pbp[["game_id"]].copy()
        work["sec"] = sec
        work = work[np.isfinite(work["sec"])].copy()
        if not work.empty:
            work = work.sort_values(["game_id", "sec"], ascending=[True, False])
            work["next_sec"] = work.groupby("game_id")["sec"].shift(-1)
            work["elapsed"] = work["sec"] - work["next_sec"]
            elapsed = work["elapsed"]
            # Keep plausible play-to-play deltas only.
            elapsed = elapsed[(elapsed > 0) & (elapsed <= 120)]
            elapsed = elapsed[np.isfinite(elapsed)]
            ref["actual_time_elapsed_per_timed_play"] = float(elapsed.mean()) if not elapsed.empty else None

    return ref


def build_sim_team_game_draws(raw_results_dir: Path) -> pd.DataFrame:
    if not raw_results_dir.exists():
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []
    for path in sorted(raw_results_dir.glob("*_players.csv")):
        df = load_csv(path)
        if df.empty:
            continue
        game_id = path.stem.replace("_players", "")
        df["game_id"] = str(game_id)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    sim_players = pd.concat(frames, ignore_index=True)
    required_cols = [
        "game_id",
        "simulation",
        "team",
        "pass_attempts",
        "rushing_attempts",
        "passing_yards",
        "rushing_yards",
        "punts",
        "punt_yards",
    ]
    for col in required_cols:
        if col not in sim_players.columns:
            if col in {"game_id", "team"}:
                sim_players[col] = ""
            else:
                sim_players[col] = 0.0

    for col in [
        "pass_attempts",
        "rushing_attempts",
        "passing_yards",
        "rushing_yards",
        "punts",
        "punt_yards",
    ]:
        sim_players[col] = pd.to_numeric(sim_players[col], errors="coerce").fillna(0.0)

    team_draws = (
        sim_players.groupby(["game_id", "simulation", "team"], as_index=False)[
            ["pass_attempts", "rushing_attempts", "passing_yards", "rushing_yards", "punts", "punt_yards"]
        ]
        .sum(numeric_only=True)
        .rename(
            columns={
                "pass_attempts": "sim_pass_attempts",
                "rushing_attempts": "sim_rushing_attempts",
                "passing_yards": "sim_passing_yards",
                "rushing_yards": "sim_rushing_yards",
                "punts": "sim_punts",
                "punt_yards": "sim_punt_yards",
            }
        )
    )
    return team_draws


def build_model_performance_summary(
    attempt_summary: Dict[str, object],
    playcall_summary: Dict[str, object],
    backtest_summary: Dict[str, object],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    if playcall_summary:
        rows.append(metric_row("play_call", "weighted_pass_rate_mae", playcall_summary.get("weighted_pass_rate_mae"), "bucket calibration"))
        rows.append(metric_row("play_call", "mean_pass_rate_error", playcall_summary.get("mean_pass_rate_error"), "positive means over-calling pass"))
        rows.append(metric_row("play_call", "overlap_buckets_after_min_support", playcall_summary.get("overlap_buckets_after_min_support"), "evaluation support"))

    pass_outcomes = attempt_summary.get("pass_outcome_comparison", {}) if attempt_summary else {}
    outcome_pairs = [
        ("sack", "sim_sack_rate", "actual_sack_rate"),
        ("completion", "sim_completion_rate", "actual_completion_rate"),
        ("interception", "sim_interception_rate", "actual_interception_rate"),
        ("scramble", "sim_scramble_rate", None),
        ("incomplete", "sim_incomplete_rate", "actual_incomplete_rate"),
    ]
    for label, sim_key, actual_key in outcome_pairs:
        sim_val = pass_outcomes.get(sim_key)
        actual_val = pass_outcomes.get(actual_key) if actual_key else None
        rows.append(metric_row(label, "sim_rate", sim_val, "from attempt diagnostics"))
        if actual_key:
            rows.append(metric_row(label, "actual_rate", actual_val, "from PBP"))
            if sim_val is not None and actual_val is not None:
                rows.append(metric_row(label, "rate_delta", float(sim_val) - float(actual_val), "sim - actual"))

    kick = attempt_summary.get("kick_comparison", {}) if attempt_summary else {}
    for leg in ("fg", "pat"):
        sim_key = f"sim_{leg}_pct"
        actual_key = f"actual_{leg}_pct"
        sim_val = kick.get(sim_key)
        actual_val = kick.get(actual_key)
        rows.append(metric_row("kick_success", sim_key, sim_val, "from diagnostics"))
        rows.append(metric_row("kick_success", actual_key, actual_val, "from PBP"))
        if sim_val is not None and actual_val is not None:
            rows.append(metric_row("kick_success", f"{leg}_pct_delta", float(sim_val) - float(actual_val), "sim - actual"))

    play_volume = attempt_summary.get("play_volume_vs_scoring", {}) if attempt_summary else {}
    if play_volume:
        rows.append(metric_row("time_elapsed_proxy", "mean_sim_plays_per_game", play_volume.get("mean_sim_plays_per_game"), "pace proxy"))
        rows.append(metric_row("time_elapsed_proxy", "mean_actual_plays_per_game", play_volume.get("mean_actual_plays_per_game"), "pace proxy"))
        rows.append(metric_row("time_elapsed_proxy", "mean_plays_delta", play_volume.get("mean_plays_delta"), "sim - actual"))
        rows.append(metric_row("time_elapsed_proxy", "sim_points_per_100_plays", play_volume.get("sim_points_per_100_plays"), "efficiency proxy"))
        rows.append(metric_row("time_elapsed_proxy", "actual_points_per_100_plays", play_volume.get("actual_points_per_100_plays"), "efficiency proxy"))

    if backtest_summary:
        rows.append(metric_row("totals", "mean_total_bias", backtest_summary.get("mean_total_bias"), "sim_total - actual_total"))
        rows.append(metric_row("totals", "total_rmse", backtest_summary.get("total_rmse"), "game-level total error"))
        rows.append(metric_row("spread", "mean_spread_bias", backtest_summary.get("mean_spread_bias"), "sim_spread - actual_spread"))

    return pd.DataFrame(rows)


def build_attempt_volume_deltas(team_calib: pd.DataFrame) -> pd.DataFrame:
    if team_calib.empty:
        return pd.DataFrame()
    cols = [
        "game_id",
        "team",
        "sim_pass_attempts",
        "actual_pass_attempts",
        "pass_attempts_delta",
        "sim_rushing_attempts",
        "actual_rushing_attempts",
        "rush_attempts_delta",
        "sim_plays",
        "actual_plays",
        "plays_delta",
    ]
    available = [c for c in cols if c in team_calib.columns]
    frame = team_calib[available].copy()
    return frame.sort_values(["game_id", "team"]).reset_index(drop=True)


def summarize_efficiency(
    name: str,
    sim_values: pd.Series,
    actual_values: pd.Series,
) -> Dict[str, object]:
    sim = pd.to_numeric(sim_values, errors="coerce").to_numpy(dtype=float)
    actual = pd.to_numeric(actual_values, errors="coerce").to_numpy(dtype=float)
    input_rows = int(max(sim.size, actual.size))
    mask = np.isfinite(sim) & np.isfinite(actual)
    if not np.any(mask):
        return {
            "metric": name,
            "rows": 0,
            "input_rows": input_rows,
            "dropped_rows": input_rows,
            "mean_sim": None,
            "mean_actual": None,
            "mean_bias": None,
            "mae": None,
            "rmse": None,
        }

    sim = sim[mask]
    actual = actual[mask]
    diff = sim - actual
    return {
        "metric": name,
        "rows": int(sim.size),
        "input_rows": input_rows,
        "dropped_rows": int(input_rows - sim.size),
        "mean_sim": float(np.mean(sim)),
        "mean_actual": float(np.mean(actual)),
        "mean_bias": float(np.mean(diff)),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
    }


def build_ordinal_efficiency_summary(
    team_calib: pd.DataFrame,
    attempt_diag: pd.DataFrame,
    team_draws: pd.DataFrame,
    actual_ordinal_ref: Optional[Dict[str, Optional[float]]] = None,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    actual_ordinal_ref = actual_ordinal_ref or {}

    if not team_calib.empty:
        sim_rush_eff = pd.to_numeric(team_calib.get("sim_rushing_yards", np.nan), errors="coerce") / pd.to_numeric(
            team_calib.get("sim_rushing_attempts", np.nan), errors="coerce"
        ).replace(0, np.nan)
        actual_rush_eff = pd.to_numeric(team_calib.get("actual_rushing_yards", np.nan), errors="coerce") / pd.to_numeric(
            team_calib.get("actual_rushing_attempts", np.nan), errors="coerce"
        ).replace(0, np.nan)
        rows.append(summarize_efficiency("rush_yards_per_attempt", sim_rush_eff, actual_rush_eff))

        sim_pass_eff = pd.to_numeric(team_calib.get("sim_passing_yards", np.nan), errors="coerce") / pd.to_numeric(
            team_calib.get("sim_pass_attempts", np.nan), errors="coerce"
        ).replace(0, np.nan)
        actual_pass_eff = pd.to_numeric(team_calib.get("actual_passing_yards", np.nan), errors="coerce") / pd.to_numeric(
            team_calib.get("actual_pass_attempts", np.nan), errors="coerce"
        ).replace(0, np.nan)
        rows.append(summarize_efficiency("pass_yards_per_attempt_proxy_air_yac", sim_pass_eff, actual_pass_eff))

    if not attempt_diag.empty:
        sim_comp_eff = pd.to_numeric(attempt_diag.get("complete_pass_yards_sum", np.nan), errors="coerce") / pd.to_numeric(
            attempt_diag.get("complete_pass_plays", np.nan), errors="coerce"
        ).replace(0, np.nan)
        rows.append(
            {
                "metric": "complete_pass_yards_per_completion_sim_only",
                "rows": int(pd.Series(sim_comp_eff).notna().sum()),
                "input_rows": int(len(attempt_diag)),
                "dropped_rows": int(len(attempt_diag) - pd.Series(sim_comp_eff).notna().sum()),
                "mean_sim": safe_mean(pd.Series(sim_comp_eff).dropna()),
                "mean_actual": (
                    float(actual_ordinal_ref.get("actual_complete_pass_yards_per_completion"))
                    if actual_ordinal_ref.get("actual_complete_pass_yards_per_completion") is not None
                    and np.isfinite(float(actual_ordinal_ref.get("actual_complete_pass_yards_per_completion")))
                    else None
                ),
                "mean_bias": (
                    float(
                        safe_mean(pd.Series(sim_comp_eff).dropna())
                        - float(actual_ordinal_ref.get("actual_complete_pass_yards_per_completion"))
                    )
                    if actual_ordinal_ref.get("actual_complete_pass_yards_per_completion") is not None
                    and np.isfinite(float(actual_ordinal_ref.get("actual_complete_pass_yards_per_completion")))
                    and np.isfinite(float(safe_mean(pd.Series(sim_comp_eff).dropna())))
                    else None
                ),
                "mae": None,
                "rmse": None,
            }
        )

        for sim_col, plays_col, label in [
            ("air_yards_sum", "air_plays", "air_yards_per_attempt"),
            ("yac_yards_sum", "yac_plays", "yac_yards_per_completion"),
            ("punt_yards_sum", "punt_plays", "punt_yards_per_punt"),
            ("time_elapsed_sum", "time_plays", "time_elapsed_per_timed_play"),
        ]:
            if sim_col in attempt_diag.columns and plays_col in attempt_diag.columns:
                rate = pd.to_numeric(attempt_diag.get(sim_col, np.nan), errors="coerce") / pd.to_numeric(
                    attempt_diag.get(plays_col, np.nan), errors="coerce"
                ).replace(0, np.nan)
                mean_sim_val = safe_mean(pd.Series(rate).dropna())
                actual_key = f"actual_{label}"
                actual_val = actual_ordinal_ref.get(actual_key)
                mean_bias_val = None
                if actual_val is not None and np.isfinite(float(actual_val)) and np.isfinite(float(mean_sim_val)):
                    mean_bias_val = float(mean_sim_val - float(actual_val))
                rows.append(
                    {
                        "metric": f"{label}_sim_only",
                        "rows": int(pd.Series(rate).notna().sum()),
                        "input_rows": int(len(attempt_diag)),
                        "dropped_rows": int(len(attempt_diag) - pd.Series(rate).notna().sum()),
                        "mean_sim": mean_sim_val,
                        "mean_actual": float(actual_val) if actual_val is not None and np.isfinite(float(actual_val)) else None,
                        "mean_bias": mean_bias_val,
                        "mae": None,
                        "rmse": None,
                    }
                )

    if not team_draws.empty and not team_calib.empty:
        actual_lookup = team_calib[["game_id", "team", "actual_rushing_yards", "actual_rushing_attempts", "actual_passing_yards", "actual_pass_attempts"]].copy()
        merged = team_draws.merge(actual_lookup, on=["game_id", "team"], how="inner")
        if not merged.empty:
            merged["sim_rush_eff"] = merged["sim_rushing_yards"] / merged["sim_rushing_attempts"].replace(0, np.nan)
            merged["sim_pass_eff"] = merged["sim_passing_yards"] / merged["sim_pass_attempts"].replace(0, np.nan)
            merged["actual_rush_eff"] = merged["actual_rushing_yards"] / merged["actual_rushing_attempts"].replace(0, np.nan)
            merged["actual_pass_eff"] = merged["actual_passing_yards"] / merged["actual_pass_attempts"].replace(0, np.nan)

            for metric_name, sim_col, actual_col in [
                ("rush_yards_per_attempt", "sim_rush_eff", "actual_rush_eff"),
                ("pass_yards_per_attempt_proxy_air_yac", "sim_pass_eff", "actual_pass_eff"),
            ]:
                crps_values: List[float] = []
                for _, grp in merged.groupby(["game_id", "team"]):
                    y = pd.to_numeric(grp[actual_col], errors="coerce").dropna()
                    if y.empty:
                        continue
                    obs = float(y.iloc[0])
                    sim_samples = pd.to_numeric(grp[sim_col], errors="coerce").to_numpy(dtype=float)
                    score = crps_empirical(sim_samples, obs)
                    if np.isfinite(score):
                        crps_values.append(score)
                if crps_values:
                    rows.append(
                        {
                            "metric": f"{metric_name}_crps_team_game",
                            "rows": int(len(crps_values)),
                            "input_rows": int(merged.groupby(["game_id", "team"]).ngroups),
                            "dropped_rows": int(merged.groupby(["game_id", "team"]).ngroups - len(crps_values)),
                            "mean_sim": None,
                            "mean_actual": None,
                            "mean_bias": None,
                            "mae": None,
                            "rmse": None,
                            "crps": float(np.mean(crps_values)),
                        }
                    )

    out = pd.DataFrame(rows)
    if "input_rows" not in out.columns:
        out["input_rows"] = np.nan
    if "dropped_rows" not in out.columns:
        out["dropped_rows"] = np.nan
    if "crps" not in out.columns:
        out["crps"] = np.nan
    return out


def build_row_coverage_audit(
    results: pd.DataFrame,
    attempt_diag: pd.DataFrame,
    team_calib: pd.DataFrame,
    team_draws: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    result_games = set(results.get("game_id", pd.Series(dtype=str)).astype(str)) if not results.empty else set()
    attempt_games = set(attempt_diag.get("game_id", pd.Series(dtype=str)).astype(str)) if not attempt_diag.empty else set()
    calib_games = set(team_calib.get("game_id", pd.Series(dtype=str)).astype(str)) if not team_calib.empty else set()
    draws_games = set(team_draws.get("game_id", pd.Series(dtype=str)).astype(str)) if not team_draws.empty else set()

    expected_attempt_rows = 0
    expected_team_calib_rows = 0
    expected_team_draw_rows = 0
    if not results.empty and {"game_id", "simulations_run"}.issubset(results.columns):
        sim_runs = pd.to_numeric(results["simulations_run"], errors="coerce").fillna(0.0)
        expected_attempt_rows = int(sim_runs.sum())
        expected_team_calib_rows = int(len(results) * 2)
        expected_team_draw_rows = int(sim_runs.sum() * 2)

    rows.append(
        {
            "dataset": "results",
            "expected_rows": len(results),
            "observed_rows": len(results),
            "missing_rows": 0,
            "expected_games": len(result_games),
            "observed_games": len(result_games),
            "missing_games": 0,
            "details": "baseline",
        }
    )
    rows.append(
        {
            "dataset": "attempt_diagnostics",
            "expected_rows": expected_attempt_rows,
            "observed_rows": len(attempt_diag),
            "missing_rows": int(max(expected_attempt_rows - len(attempt_diag), 0)),
            "expected_games": len(result_games),
            "observed_games": len(attempt_games),
            "missing_games": int(max(len(result_games - attempt_games), 0)),
            "details": "one row per game per simulation",
        }
    )
    rows.append(
        {
            "dataset": "team_calibration",
            "expected_rows": expected_team_calib_rows,
            "observed_rows": len(team_calib),
            "missing_rows": int(max(expected_team_calib_rows - len(team_calib), 0)),
            "expected_games": len(result_games),
            "observed_games": len(calib_games),
            "missing_games": int(max(len(result_games - calib_games), 0)),
            "details": "two rows per game (home/away)",
        }
    )
    rows.append(
        {
            "dataset": "team_draws",
            "expected_rows": expected_team_draw_rows,
            "observed_rows": len(team_draws),
            "missing_rows": int(max(expected_team_draw_rows - len(team_draws), 0)),
            "expected_games": len(result_games),
            "observed_games": len(draws_games),
            "missing_games": int(max(len(result_games - draws_games), 0)),
            "details": "two team rows per game per simulation",
        }
    )

    return pd.DataFrame(rows)


def build_runtime_sanity_checks(
    attempt_summary: Dict[str, object],
    playcall_summary: Dict[str, object],
    backtest_summary: Dict[str, object],
    runtime_source: Path,
) -> pd.DataFrame:
    checks: List[Dict[str, object]] = []

    schema_issues = backtest_summary.get("schema_issues", []) if backtest_summary else []
    checks.append(
        {
            "check": "schema_issues_count",
            "value": len(schema_issues) if isinstance(schema_issues, list) else 0,
            "status": "warn" if schema_issues else "ok",
            "details": "; ".join(str(x) for x in schema_issues[:5]) if isinstance(schema_issues, list) else "",
        }
    )

    inv = attempt_summary.get("invariant_violations") if attempt_summary else None
    checks.append(
        {
            "check": "pass_attempt_invariant_violations",
            "value": inv,
            "status": "ok" if inv == 0 else "warn",
            "details": "expects pass_attempt_delta + scramble == 0",
        }
    )

    wmae = playcall_summary.get("weighted_pass_rate_mae") if playcall_summary else None
    checks.append(
        {
            "check": "playcall_weighted_pass_rate_mae",
            "value": wmae,
            "status": "ok" if isinstance(wmae, (int, float)) and float(wmae) < 0.08 else "warn",
            "details": "bucket calibration error",
        }
    )

    source_text = ""
    if runtime_source.exists():
        try:
            source_text = runtime_source.read_text(encoding="utf-8")
        except Exception:
            source_text = ""

    heuristics = {
        "red_zone_pass_flip_present": "red_zone_pass_flip" in source_text,
        "pass_outcome_reweight_present": "comp_mult =" in source_text and "sack_mult =" in source_text,
        "rush_boost_present": "rush_boost" in source_text,
        "yac_boost_present": "yac_boost" in source_text,
        "punt_distance_cap_present": "self.yards_gained = 80 - self.yardline" in source_text,
    }
    for key, present in heuristics.items():
        checks.append(
            {
                "check": key,
                "value": int(bool(present)),
                "status": "info",
                "details": "1 means logic is present in runtime",
            }
        )

    return pd.DataFrame(checks)


def save_html_report(
    out_path: Path,
    model_perf: pd.DataFrame,
    ordinal_perf: pd.DataFrame,
    sanity: pd.DataFrame,
    attempts: pd.DataFrame,
    coverage: pd.DataFrame,
) -> None:
    def frame_to_html(df: pd.DataFrame) -> str:
        if df.empty:
            return "<p>No rows.</p>"
        return df.to_html(index=False, border=0)

    sections = [
        ("Model Performance Summary", model_perf),
        ("Ordinal Efficiency Summary", ordinal_perf),
        ("Runtime Sanity Checks", sanity),
        ("Attempt Volume Deltas", attempts),
        ("Row Coverage Audit", coverage),
    ]

    body_parts = []
    for title, frame in sections:
        body_parts.append(f"<h2>{html.escape(title)}</h2>")
        body_parts.append(frame_to_html(frame))

    html_doc = f"""<!doctype html>
<html>
<head>
<meta charset=\"utf-8\" />
<title>2025 Simulation Performance Evaluation</title>
<style>
body {{ font-family: Segoe UI, Arial, sans-serif; margin: 20px; color: #1f2937; }}
h1 {{ margin-bottom: 0.2rem; }}
p.meta {{ margin-top: 0; color: #6b7280; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
th, td {{ border: 1px solid #d1d5db; padding: 6px 8px; font-size: 13px; }}
th {{ background: #f3f4f6; text-align: left; }}
</style>
</head>
<body>
<h1>2025 Simulation Performance Evaluation</h1>
<p class=\"meta\">Generated from existing backtest artifacts.</p>
{''.join(body_parts)}
</body>
</html>
"""
    out_path.write_text(html_doc, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 2025 simulation model performance from backtest artifacts.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--attempt-diagnostics", type=Path, default=DEFAULT_ATTEMPT_DIAG_PATH)
    parser.add_argument("--attempt-summary", type=Path, default=DEFAULT_ATTEMPT_SUMMARY_PATH)
    parser.add_argument("--playcall-summary", type=Path, default=DEFAULT_PLAYCALL_SUMMARY_PATH)
    parser.add_argument("--team-calibration", type=Path, default=DEFAULT_TEAM_CALIB_PATH)
    parser.add_argument("--backtest-summary", type=Path, default=DEFAULT_BACKTEST_SUMMARY_PATH)
    parser.add_argument("--raw-results-dir", type=Path, default=DEFAULT_RAW_RESULTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = Paths(
        results=Path(args.results),
        attempt_diag=Path(args.attempt_diagnostics),
        attempt_summary=Path(args.attempt_summary),
        playcall_summary=Path(args.playcall_summary),
        team_calib=Path(args.team_calibration),
        backtest_summary=Path(args.backtest_summary),
        raw_results_dir=Path(args.raw_results_dir),
        out_dir=Path(args.out_dir),
    )

    paths.out_dir.mkdir(parents=True, exist_ok=True)

    results = load_csv(paths.results)
    attempt_diag = load_csv(paths.attempt_diag)
    team_calib = load_csv(paths.team_calib)

    attempt_summary = load_json(paths.attempt_summary)
    playcall_summary = load_json(paths.playcall_summary)
    backtest_summary = load_json(paths.backtest_summary)

    team_draws = build_sim_team_game_draws(paths.raw_results_dir)
    actual_ordinal_ref = load_actual_ordinal_reference(results)

    model_perf = build_model_performance_summary(attempt_summary, playcall_summary, backtest_summary)
    ordinal_perf = build_ordinal_efficiency_summary(team_calib, attempt_diag, team_draws, actual_ordinal_ref)
    sanity = build_runtime_sanity_checks(
        attempt_summary,
        playcall_summary,
        backtest_summary,
        WORKSPACE_DIR / "torch_game_simulation_runtime.py",
    )
    attempt_deltas = build_attempt_volume_deltas(team_calib)
    coverage = build_row_coverage_audit(results, attempt_diag, team_calib, team_draws)

    model_perf.to_csv(paths.out_dir / "model_performance_summary.csv", index=False)
    ordinal_perf.to_csv(paths.out_dir / "ordinal_efficiency_summary.csv", index=False)
    sanity.to_csv(paths.out_dir / "runtime_sanity_checks.csv", index=False)
    attempt_deltas.to_csv(paths.out_dir / "attempt_volume_deltas.csv", index=False)
    coverage.to_csv(paths.out_dir / "row_coverage_audit.csv", index=False)

    save_html_report(
        paths.out_dir / "performance_report.html",
        model_perf,
        ordinal_perf,
        sanity,
        attempt_deltas,
        coverage,
    )

    summary = {
        "rows": {
            "results": int(len(results)),
            "attempt_diagnostics": int(len(attempt_diag)),
            "team_calibration": int(len(team_calib)),
            "team_draws": int(len(team_draws)),
            "model_performance": int(len(model_perf)),
            "ordinal_efficiency": int(len(ordinal_perf)),
            "runtime_sanity": int(len(sanity)),
            "attempt_deltas": int(len(attempt_deltas)),
            "coverage_audit": int(len(coverage)),
        },
        "outputs": {
            "model_performance_summary_csv": str(paths.out_dir / "model_performance_summary.csv"),
            "ordinal_efficiency_summary_csv": str(paths.out_dir / "ordinal_efficiency_summary.csv"),
            "runtime_sanity_checks_csv": str(paths.out_dir / "runtime_sanity_checks.csv"),
            "attempt_volume_deltas_csv": str(paths.out_dir / "attempt_volume_deltas.csv"),
            "row_coverage_audit_csv": str(paths.out_dir / "row_coverage_audit.csv"),
            "performance_report_html": str(paths.out_dir / "performance_report.html"),
        },
    }
    (paths.out_dir / "performance_eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Wrote performance evaluation artifacts:")
    for key, value in summary["outputs"].items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
