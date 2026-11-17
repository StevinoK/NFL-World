from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import nfl_data_py as nfl
import numpy as np
import pandas as pd

from nflsim.engine.simulation_runtime import Game, Player, ROOF_TYPE, SKILL_SLOT_ORDER, SimulationRuntime, Team
from nflsim.features.slots import USAGE_COLUMNS, is_empty_slot


# Legacy Bayesian backtest outputs; the ridge benchmark redirects these per run.
WORKSPACE_DIR = Path(__file__).resolve().parents[2] / "archive" / "bayesian_sim"
RESULTS_PATH = WORKSPACE_DIR / "backtest_2025_results.csv"
MISSING_PATH = WORKSPACE_DIR / "backtest_2025_missing_games.csv"
SUMMARY_PATH = WORKSPACE_DIR / "backtest_2025_summary.json"
RAW_RESULTS_DIR = WORKSPACE_DIR / "backtest_2025_game_results"
HTML_REPORT_PATH = WORKSPACE_DIR / "backtest_2025_report.html"
ATTEMPT_DIAGNOSTICS_PATH = WORKSPACE_DIR / "backtest_2025_attempt_diagnostics.csv"
ATTEMPT_SUMMARY_PATH = WORKSPACE_DIR / "backtest_2025_attempt_summary.json"
PLAYCALL_BUCKET_DIAGNOSTICS_PATH = WORKSPACE_DIR / "backtest_2025_playcall_bucket_diagnostics.csv"
PLAYCALL_BUCKET_SUMMARY_PATH = WORKSPACE_DIR / "backtest_2025_playcall_bucket_summary.json"
PLAYCALL_BUCKET_COMPARISON_PATH = WORKSPACE_DIR / "backtest_2025_playcall_bucket_comparison.csv"
TEAM_OFFENSE_CALIBRATION_PATH = WORKSPACE_DIR / "backtest_2025_team_offense_calibration.csv"
TEAM_OFFENSE_CALIBRATION_SUMMARY_PATH = WORKSPACE_DIR / "backtest_2025_team_offense_calibration_summary.json"
NAN_INPUT_LOG_PATH = WORKSPACE_DIR / "backtest_2025_nan_input_log.csv"
PLAYCALL_BUCKET_MIN_SUPPORT = 5
ACTIVE_STATUSES = {"ACT"}
SLOT_LIMITS = {"QB": 1, "WR": 4, "RB": 3, "TE": 3}
SKILL_POSITIONS = tuple(SLOT_LIMITS.keys())


def distance_bucket(distance: int) -> str:
    if distance <= 3:
        return "short"
    if distance <= 7:
        return "medium"
    return "long"


def score_bucket(score_diff: float) -> str:
    if score_diff <= -9:
        return "trailing_big"
    if score_diff <= -1:
        return "trailing"
    if score_diff >= 9:
        return "leading_big"
    if score_diff >= 1:
        return "leading"
    return "tied"


def clock_bucket(quarter: int, clock: int) -> str:
    if quarter >= 4 and clock <= 300:
        return "q4_late"
    if quarter >= 3:
        return "second_half"
    if quarter == 2 and clock <= 300:
        return "q2_late"
    return "first_half"


def playcall_bucket_key(down: int, distance: int, score_diff: float, quarter: int, clock: int) -> str:
    down_int = int(np.clip(int(down), 1, 4))
    return (
        f"down_{down_int}|dist_{distance_bucket(int(distance))}"
        f"|score_{score_bucket(float(score_diff))}|clock_{clock_bucket(int(quarter), int(clock))}"
    )


def implied_prob_from_moneyline(value: float) -> float:
    if not np.isfinite(value):
        return float("nan")
    if value < 0:
        return 1 / ((-100 / value) + 1)
    return 1 / ((value / 100) + 1)


def cap_rest_days(value: float, maximum: float = 21.0) -> float:
    if not np.isfinite(value):
        return float("nan")
    return float(min(value, maximum))


class SkipGameError(RuntimeError):
    pass


_WORKER_RUNTIME: Optional[SimulationRuntime] = None


def _init_sim_worker(data_dir: str, strict_schema: bool = False) -> None:
    global _WORKER_RUNTIME
    _WORKER_RUNTIME = SimulationRuntime(Path(data_dir), strict_schema=bool(strict_schema))


def _build_player_from_snapshot(player_snapshot: Dict[str, object]) -> Player:
    player_columns = {"player_id": player_snapshot["player_id"], "player_name": player_snapshot["name"]}
    player_columns.update(player_snapshot["attributes"])
    return Player(
        name=str(player_snapshot["name"]),
        position=str(player_snapshot["position"]),
        player_id=str(player_snapshot["player_id"]),
        attributes=pd.DataFrame([player_columns]),
    )


def _simulate_single_game_run(task: Dict[str, object]) -> Dict[str, object]:
    runtime = _WORKER_RUNTIME
    if runtime is None:
        runtime = SimulationRuntime(Path(str(task["data_dir"])), strict_schema=bool(task.get("strict_schema", False)))
    runtime.reset_nan_input_log()

    simulation_number = int(task["simulation"])
    game_id = str(task["game_id"])
    home_team = str(task["home_team"])
    away_team = str(task["away_team"])
    roof = str(task["roof"])
    team_features = task["team_features"]

    away_players = [_build_player_from_snapshot(snapshot) for snapshot in task["away_players"]]
    home_players = [_build_player_from_snapshot(snapshot) for snapshot in task["home_players"]]

    away = Team(away_team, away_players, "away", pd.Series(team_features[away_team]), runtime)
    home = Team(home_team, home_players, "home", pd.Series(team_features[home_team]), runtime)
    game = Game(away, home, roof, runtime)
    summary = game.simulate_game(
        print_summary=bool(task["print_game_summaries"]),
        summary_prefix=f"[{game_id} sim {simulation_number}/{int(task['simulations'])}]",
        live_pbp=bool(task.get("live_pbp", False)),
    )
    attempt_diagnostics = None
    playcall_bucket_rows: List[Dict[str, object]] = []
    if bool(task.get("attempt_diagnostics", False)):
        attempt_diagnostics = game.diagnostics_row()
        attempt_diagnostics.update(
            {
                "game_id": game_id,
                "simulation": simulation_number,
                "home_team": home_team,
                "away_team": away_team,
            }
        )
        playcall_bucket_rows = game.diagnostics_bucket_rows()
        for row in playcall_bucket_rows:
            row.update(
                {
                    "game_id": game_id,
                    "simulation": simulation_number,
                    "home_team": home_team,
                    "away_team": away_team,
                }
            )

    sim_row: Dict[str, object] = {
        "simulation": simulation_number,
        "home": home.score,
        "away": away.score,
        "qb": summary["qb"].name if summary.get("qb") else "",
        "receiver": summary["receiver"].name if summary.get("receiver") else "",
        "rusher": summary["rusher"].name if summary.get("rusher") else "",
    }

    player_rows: List[Dict[str, object]] = []
    for team in [home, away]:
        for player in team.players:
            row: Dict[str, object] = {
                "simulation": simulation_number,
                "team": team.name,
                "player_name": player.name,
                "player_id": player.player_id,
                "position": player.position,
            }
            row.update(player.stats)
            player_rows.append(row)

    return {
        "home_score": int(home.score),
        "away_score": int(away.score),
        "sim_row": sim_row,
        "player_rows": player_rows,
        "attempt_diagnostics": attempt_diagnostics,
        "playcall_bucket_rows": playcall_bucket_rows,
        "nan_input_log_rows": runtime.get_nan_input_log_rows(),
    }


class GameBacktester:
    def __init__(
        self,
        season: int = 2025,
        simulations: int = 200,
        workers: int = 1,
        week_start: Optional[int] = None,
        week_end: Optional[int] = None,
        resume: bool = True,
        reset_artifacts: bool = False,
        print_game_summaries: bool = True,
        strict_schema: bool = False,
        attempt_diagnostics: bool = False,
        raw_results_dir: Path = RAW_RESULTS_DIR,
        game_id: Optional[str] = None,
        live_pbp: bool = False,
    ):
        self.season = season
        self.simulations = simulations
        self.workers = max(1, min(int(workers), 6))
        self.week_start = week_start
        self.week_end = week_end
        self.resume = resume
        self.reset_artifacts = reset_artifacts
        self.print_game_summaries = print_game_summaries
        self.strict_schema = bool(strict_schema)
        self.attempt_diagnostics = bool(attempt_diagnostics)
        self.live_pbp = bool(live_pbp)
        if self.live_pbp and (self.workers > 1 or self.simulations > 1):
            print("--live-pbp requested: forcing --workers 1 and --simulations 1 so play-by-play output stays readable.")
            self.workers = 1
            self.simulations = 1
        self.raw_results_dir = raw_results_dir
        self.game_id = str(game_id).strip() if game_id is not None and str(game_id).strip() else None
        if self.reset_artifacts:
            self._reset_outputs()
        self.raw_results_dir.mkdir(parents=True, exist_ok=True)

        self.runtime = SimulationRuntime(strict_schema=self.strict_schema)
        if self.runtime.schema_issues:
            print(f"Detected {len(self.runtime.schema_issues)} runtime schema issue(s).")
            for issue in self.runtime.schema_issues:
                print(f"  - {issue}")
        self.schedule = self._load_schedule()
        self.rosters = self._load_rosters()
        self.depth_charts = self._load_depth_charts()
        self.player_stats = self.runtime.player_stats_wide.copy()
        self.player_stats["game_date"] = pd.to_datetime(self.player_stats["game_date"])
        self.team_stats = self.runtime.team_stats_wide.copy()
        self.team_stats["game_date"] = pd.to_datetime(self.team_stats["game_date"])
        self._pbp_cache: Optional[pd.DataFrame] = None
        self._actual_player_weekly_cache: Optional[pd.DataFrame] = None
        self._actual_player_game_stats_cache: Optional[pd.DataFrame] = None
        self._actual_team_game_stats_cache: Optional[pd.DataFrame] = None
        self._actual_game_fumble_stats_cache: Optional[pd.DataFrame] = None
        self._actual_team_penalty_stats_cache: Optional[pd.DataFrame] = None
        self.results = self._load_table(RESULTS_PATH)
        self.missing = self._load_table(MISSING_PATH)

    def _get_pbp_data(self) -> pd.DataFrame:
        if self._pbp_cache is not None:
            return self._pbp_cache
        try:
            pbp = nfl.import_pbp_data([self.season], downcast=True, cache=False).copy()
        except Exception:
            self._pbp_cache = pd.DataFrame()
            return self._pbp_cache
        if pbp.empty:
            self._pbp_cache = pd.DataFrame()
            return self._pbp_cache
        if "season_type" in pbp.columns:
            pbp = pbp[pbp["season_type"] == "REG"].copy()
        if "game_id" in pbp.columns:
            pbp["game_id"] = pbp["game_id"].astype(str)
        if "posteam" in pbp.columns:
            pbp["posteam"] = pbp["posteam"].astype(str)
        self._pbp_cache = pbp
        return self._pbp_cache

    def _get_actual_player_weekly(self) -> pd.DataFrame:
        if self._actual_player_weekly_cache is not None:
            return self._actual_player_weekly_cache
        try:
            weekly = nfl.import_weekly_data([self.season]).copy()
        except Exception:
            self._actual_player_weekly_cache = pd.DataFrame()
            return self._actual_player_weekly_cache

        if "season_type" in weekly.columns:
            weekly = weekly[weekly["season_type"] == "REG"].copy()
        weekly["player_id"] = weekly["player_id"].astype(str)
        weekly["recent_team"] = weekly["recent_team"].astype(str)
        self._actual_player_weekly_cache = weekly
        return self._actual_player_weekly_cache

    def _get_actual_player_game_stats(self) -> pd.DataFrame:
        if self._actual_player_game_stats_cache is not None:
            return self._actual_player_game_stats_cache

        # Preferred source: weekly player stats.
        weekly = self._get_actual_player_weekly()
        if not weekly.empty:
            expected_cols = [
                "game_id",
                "week",
                "recent_team",
                "player_id",
                "passing_yards",
                "rushing_yards",
                "receiving_yards",
                "passing_tds",
                "rushing_tds",
                "attempts",
                "carries",
                "completions",
                "receptions",
                "interceptions",
            ]
            available_cols = [column for column in expected_cols if column in weekly.columns]
            if {"week", "recent_team", "player_id"}.issubset(set(available_cols)):
                actual = weekly[available_cols].copy()
                actual["player_id"] = actual["player_id"].astype(str)
                actual["recent_team"] = actual["recent_team"].astype(str)
                if "carries" in actual.columns and "rushing_attempts" not in actual.columns:
                    actual = actual.rename(columns={"carries": "rushing_attempts"})
                elif "rushing_attempts" not in actual.columns:
                    actual["rushing_attempts"] = 0.0
                if "rushing_tds" not in actual.columns:
                    actual["rushing_tds"] = 0.0
                self._actual_player_game_stats_cache = actual
                return self._actual_player_game_stats_cache

        # Fallback source: aggregate play-by-play to game-level player box stats.
        try:
            pbp = nfl.import_pbp_data([self.season], downcast=True, cache=False).copy()
        except Exception:
            self._actual_player_game_stats_cache = pd.DataFrame()
            return self._actual_player_game_stats_cache

        if pbp.empty:
            self._actual_player_game_stats_cache = pd.DataFrame()
            return self._actual_player_game_stats_cache

        if "season_type" in pbp.columns:
            pbp = pbp[pbp["season_type"] == "REG"].copy()

        for column in ["passer_player_id", "rusher_player_id", "receiver_player_id", "posteam"]:
            if column in pbp.columns:
                pbp[column] = pbp[column].astype(str)

        def _group_sum(df: pd.DataFrame, id_col: str, value_col: str, output_col: str) -> pd.DataFrame:
            required_cols = {id_col, "game_id", "week", "posteam", value_col}
            if not required_cols.issubset(df.columns):
                return pd.DataFrame(columns=["game_id", "week", "recent_team", "player_id", output_col])
            subset = df[[id_col, "game_id", "week", "posteam", value_col]].copy()
            subset = subset.rename(columns={id_col: "player_id", "posteam": "recent_team", value_col: output_col})
            subset["player_id"] = subset["player_id"].astype(str)
            subset = subset[subset["player_id"].str.len() > 0]
            return subset.groupby(["game_id", "week", "recent_team", "player_id"], as_index=False)[output_col].sum(numeric_only=True)

        pass_yards = _group_sum(pbp, "passer_player_id", "passing_yards", "passing_yards")
        rush_yards = _group_sum(pbp, "rusher_player_id", "rushing_yards", "rushing_yards")
        rush_attempts = _group_sum(pbp, "rusher_player_id", "rush_attempt", "rushing_attempts")
        recv_yards = _group_sum(pbp, "receiver_player_id", "receiving_yards", "receiving_yards")
        pass_tds = _group_sum(pbp, "passer_player_id", "pass_touchdown", "passing_tds")
        rush_tds = _group_sum(pbp, "rusher_player_id", "rush_touchdown", "rushing_tds")
        pass_attempts = _group_sum(pbp, "passer_player_id", "pass_attempt", "attempts")
        pass_completions = _group_sum(pbp, "passer_player_id", "complete_pass", "completions")
        recs = _group_sum(pbp, "receiver_player_id", "complete_pass", "receptions")
        ints = _group_sum(pbp, "passer_player_id", "interception", "interceptions")

        merged: Optional[pd.DataFrame] = None
        for frame in [pass_yards, rush_yards, rush_attempts, recv_yards, pass_tds, rush_tds, pass_attempts, pass_completions, recs, ints]:
            if frame.empty:
                continue
            if merged is None:
                merged = frame
            else:
                merged = merged.merge(frame, on=["game_id", "week", "recent_team", "player_id"], how="outer")

        if merged is None:
            self._actual_player_game_stats_cache = pd.DataFrame()
            return self._actual_player_game_stats_cache

        for column in ["passing_yards", "rushing_yards", "rushing_attempts", "receiving_yards", "passing_tds", "rushing_tds", "attempts", "completions", "receptions", "interceptions"]:
            if column not in merged.columns:
                merged[column] = 0.0
        merged[["passing_tds", "rushing_tds", "attempts", "completions", "receptions", "interceptions", "rushing_attempts"]] = merged[["passing_tds", "rushing_tds", "attempts", "completions", "receptions", "interceptions", "rushing_attempts"]].fillna(0.0)
        merged["recent_team"] = merged["recent_team"].astype(str)
        merged["player_id"] = merged["player_id"].astype(str)

        self._actual_player_game_stats_cache = merged
        return self._actual_player_game_stats_cache

    def _get_actual_team_game_stats(self) -> pd.DataFrame:
        if self._actual_team_game_stats_cache is not None:
            return self._actual_team_game_stats_cache

        actual_stats = self._get_actual_player_game_stats()
        if actual_stats.empty or "game_id" not in actual_stats.columns or "recent_team" not in actual_stats.columns:
            self._actual_team_game_stats_cache = pd.DataFrame()
            return self._actual_team_game_stats_cache

        frame = actual_stats.copy()
        frame["game_id"] = frame["game_id"].astype(str)
        frame["recent_team"] = frame["recent_team"].astype(str)
        for column in ["attempts", "rushing_attempts", "passing_yards", "rushing_yards", "passing_tds", "rushing_tds"]:
            if column not in frame.columns:
                frame[column] = 0.0
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)

        grouped = (
            frame.groupby(["game_id", "recent_team"], as_index=False)[
                ["attempts", "rushing_attempts", "passing_yards", "rushing_yards", "passing_tds", "rushing_tds"]
            ]
            .sum(numeric_only=True)
            .rename(
                columns={
                    "recent_team": "team",
                    "attempts": "actual_pass_attempts",
                    "rushing_attempts": "actual_rushing_attempts",
                    "passing_yards": "actual_passing_yards",
                    "rushing_yards": "actual_rushing_yards",
                    "passing_tds": "actual_passing_tds",
                    "rushing_tds": "actual_rushing_tds",
                }
            )
        )
        grouped["actual_plays"] = grouped["actual_pass_attempts"] + grouped["actual_rushing_attempts"]
        self._actual_team_game_stats_cache = grouped
        return self._actual_team_game_stats_cache

    def _get_actual_game_fumble_stats(self, game_ids: Optional[List[str]] = None) -> pd.DataFrame:
        cache = self._actual_game_fumble_stats_cache
        if cache is None:
            pbp = self._get_pbp_data()
            if pbp.empty or "game_id" not in pbp.columns:
                self._actual_game_fumble_stats_cache = pd.DataFrame()
            else:
                value_col = None
                for candidate in ["fumble_lost", "fumble"]:
                    if candidate in pbp.columns:
                        value_col = candidate
                        break
                if value_col is None:
                    self._actual_game_fumble_stats_cache = pd.DataFrame()
                else:
                    frame = pbp[["game_id", value_col]].copy()
                    frame["game_id"] = frame["game_id"].astype(str)
                    frame[value_col] = pd.to_numeric(frame[value_col], errors="coerce").fillna(0.0)
                    self._actual_game_fumble_stats_cache = (
                        frame.groupby("game_id", as_index=False)[value_col]
                        .sum(numeric_only=True)
                        .rename(columns={value_col: "actual_fumbles"})
                    )
            cache = self._actual_game_fumble_stats_cache

        if cache is None or cache.empty or not game_ids:
            return cache if isinstance(cache, pd.DataFrame) else pd.DataFrame()
        game_id_set = set(str(x) for x in game_ids)
        return cache[cache["game_id"].astype(str).isin(game_id_set)].copy()

    def _get_actual_team_penalty_stats(self, game_ids: Optional[List[str]] = None) -> pd.DataFrame:
        cache = self._actual_team_penalty_stats_cache
        if cache is None:
            pbp = self._get_pbp_data()
            if pbp.empty or "game_id" not in pbp.columns or "penalty_yards" not in pbp.columns:
                self._actual_team_penalty_stats_cache = pd.DataFrame()
            else:
                team_col = "penalty_team" if "penalty_team" in pbp.columns else ("posteam" if "posteam" in pbp.columns else None)
                if team_col is None:
                    self._actual_team_penalty_stats_cache = pd.DataFrame()
                else:
                    frame = pbp[["game_id", team_col, "penalty_yards"]].copy()
                    frame = frame.rename(columns={team_col: "team"})
                    frame["game_id"] = frame["game_id"].astype(str)
                    frame["team"] = frame["team"].astype(str).str.strip()
                    frame = frame[frame["team"].str.len() > 0]
                    frame["penalty_yards"] = pd.to_numeric(frame["penalty_yards"], errors="coerce").fillna(0.0)
                    self._actual_team_penalty_stats_cache = (
                        frame.groupby(["game_id", "team"], as_index=False)["penalty_yards"]
                        .sum(numeric_only=True)
                        .rename(columns={"penalty_yards": "actual_penalty_yards"})
                    )
            cache = self._actual_team_penalty_stats_cache

        if cache is None or cache.empty or not game_ids:
            return cache if isinstance(cache, pd.DataFrame) else pd.DataFrame()
        game_id_set = set(str(x) for x in game_ids)
        return cache[cache["game_id"].astype(str).isin(game_id_set)].copy()

    def _reset_outputs(self) -> None:
        for path in [
            RESULTS_PATH,
            MISSING_PATH,
            SUMMARY_PATH,
            ATTEMPT_DIAGNOSTICS_PATH,
            ATTEMPT_SUMMARY_PATH,
            PLAYCALL_BUCKET_DIAGNOSTICS_PATH,
            PLAYCALL_BUCKET_SUMMARY_PATH,
            PLAYCALL_BUCKET_COMPARISON_PATH,
            TEAM_OFFENSE_CALIBRATION_PATH,
            TEAM_OFFENSE_CALIBRATION_SUMMARY_PATH,
            NAN_INPUT_LOG_PATH,
        ]:
            if path.exists():
                path.unlink()
        if self.raw_results_dir.exists():
            shutil.rmtree(self.raw_results_dir)
        if HTML_REPORT_PATH.exists():
            HTML_REPORT_PATH.unlink()

    def _load_schedule(self) -> pd.DataFrame:
        schedule = nfl.import_schedules([self.season]).copy()
        schedule = schedule[(schedule["game_type"] == "REG") & schedule["week"].between(1, 18)]
        schedule = schedule[schedule["home_score"].notna() & schedule["away_score"].notna()].copy()
        schedule["gameday"] = pd.to_datetime(schedule["gameday"])
        if self.week_start is not None:
            schedule = schedule[schedule["week"] >= self.week_start]
        if self.week_end is not None:
            schedule = schedule[schedule["week"] <= self.week_end]
        if self.game_id is not None:
            target_id = self.game_id.lower()
            schedule = schedule[schedule["game_id"].astype(str).str.lower() == target_id].copy()
            if schedule.empty:
                raise ValueError(f"No scheduled game found for game_id={self.game_id}")
        return schedule.sort_values(["week", "gameday", "game_id"]).reset_index(drop=True)

    def _load_rosters(self) -> pd.DataFrame:
        rosters = nfl.import_weekly_rosters([self.season]).copy()
        rosters = rosters[rosters["position"].isin(SKILL_POSITIONS)].copy()
        rosters["player_id"] = rosters["player_id"].astype(str)
        return rosters

    def _load_depth_charts(self) -> pd.DataFrame:
        depth_charts = nfl.import_depth_charts([self.season]).copy()
        depth_charts["dt"] = pd.to_datetime(depth_charts["dt"], utc=True).dt.tz_convert(None)
        depth_charts = depth_charts[depth_charts["pos_abb"].isin(SKILL_POSITIONS)].copy()
        depth_charts["gsis_id"] = depth_charts["gsis_id"].astype(str)
        return depth_charts

    @staticmethod
    def _load_table(path: Path) -> pd.DataFrame:
        if not path.exists() or not path.read_text(encoding="utf-8", errors="ignore").strip():
            return pd.DataFrame()
        return pd.read_csv(path)

    @staticmethod
    def _safe_group_rmse(frame: pd.DataFrame, actual_col: str, predicted_col: str) -> float | None:
        if frame.empty or actual_col not in frame.columns or predicted_col not in frame.columns:
            return None
        actual = pd.to_numeric(frame[actual_col], errors="coerce")
        predicted = pd.to_numeric(frame[predicted_col], errors="coerce")
        mask = np.isfinite(actual.to_numpy(dtype=float)) & np.isfinite(predicted.to_numpy(dtype=float))
        if not mask.any():
            return None
        diff = actual.to_numpy(dtype=float)[mask] - predicted.to_numpy(dtype=float)[mask]
        return float(np.sqrt(np.mean(diff ** 2)))

    @staticmethod
    def _safe_group_mean(frame: pd.DataFrame, column: str) -> float | None:
        if frame.empty or column not in frame.columns:
            return None
        values = pd.to_numeric(frame[column], errors="coerce")
        values = values[np.isfinite(values.to_numpy(dtype=float))]
        if values.empty:
            return None
        return float(values.mean())

    def _build_slice_metrics(self, results: pd.DataFrame) -> Dict[str, List[Dict[str, object]]]:
        if results.empty:
            return {"by_week": [], "by_team": []}

        frame = results.copy()
        frame["actual_home_win"] = (frame["actual_spread"] > 0).astype(float)
        frame["sim_brier_score"] = (frame["sim_home_win_prob"] - frame["actual_home_win"]) ** 2
        frame["total_bias"] = pd.to_numeric(frame["sim_total"], errors="coerce") - pd.to_numeric(frame["actual_total"], errors="coerce")
        frame["total_bias_vegas"] = pd.to_numeric(frame["sim_total"], errors="coerce") - pd.to_numeric(frame["total_line"], errors="coerce")
        frame["spread_bias"] = pd.to_numeric(frame["sim_spread"], errors="coerce") - pd.to_numeric(frame["actual_spread"], errors="coerce")

        def _group_metrics(group: pd.DataFrame) -> Dict[str, object]:
            non_ot = group[group.get("is_ot", 0) != 1] if "is_ot" in group.columns else group
            return {
                "games": int(len(group)),
                "brier": float(group["sim_brier_score"].mean()),
                "spread_rmse": self._safe_group_rmse(group, "actual_spread", "sim_spread"),
                "total_rmse": self._safe_group_rmse(group, "actual_total", "sim_total"),
                "mean_total_bias": self._safe_group_mean(group, "total_bias"),
                "mean_total_bias_vegas": self._safe_group_mean(group, "total_bias_vegas"),
                "mean_spread_bias": self._safe_group_mean(group, "spread_bias"),
                "ot_games": int(len(group) - len(non_ot)),
                "mean_total_bias_ex_ot": self._safe_group_mean(non_ot, "total_bias"),
                "mean_total_bias_vegas_ex_ot": self._safe_group_mean(non_ot, "total_bias_vegas"),
            }

        week_rows: List[Dict[str, object]] = []
        for week, group in frame.groupby("week", dropna=False):
            week_rows.append({"week": int(week), **_group_metrics(group)})

        team_rows: List[Dict[str, object]] = []
        for team, group in frame.groupby("home_team", dropna=False):
            team_rows.append({"team": str(team), **_group_metrics(group)})

        return {
            "by_week": sorted(week_rows, key=lambda row: row["week"]),
            "by_team": sorted(team_rows, key=lambda row: (row["brier"] if row["brier"] is not None else float("inf"), row["team"])),
        }

    def run(self, max_games: Optional[int] = None) -> None:
        attempted = 0
        for game_row in self.schedule.itertuples(index=False):
            if max_games is not None and attempted >= max_games:
                break
            game_id = game_row.game_id
            if self.resume and self._already_processed(game_id):
                continue
            attempted += 1
            try:
                record = self._simulate_completed_game(game_row)
            except SkipGameError as error:
                self._record_missing(game_row, str(error))
                continue
            except Exception as error:
                # Debug: print full traceback for specific errors
                if "could not convert string to float" in str(error):
                    import traceback
                    print(f"\n{'='*80}")
                    print(f"DEBUG: Float conversion error in {game_id}")
                    traceback.print_exc()
                    print(f"{'='*80}\n")
                self._record_missing(game_row, f"error: {error}")
                continue
            if not self.missing.empty and "game_id" in self.missing.columns:
                self.missing = self.missing[self.missing["game_id"].astype(str) != game_id].reset_index(drop=True)
                self.missing.to_csv(MISSING_PATH, index=False)
            self.results = pd.concat([self.results, pd.DataFrame([record])], ignore_index=True)
            self.results.to_csv(RESULTS_PATH, index=False)
            self._write_summary()

        self._write_summary()
        self._write_attempt_summary()
        self._write_playcall_bucket_summary()
        self._build_team_offense_calibration()

    def _write_nan_input_log(self, rows: List[Dict[str, object]]) -> None:
        expected_cols = [
            "game_id",
            "simulation",
            "model_key",
            "stage",
            "column",
            "column_nan_count",
            "calls_with_nan",
            "rows_observed",
            "rows_with_nan",
            "nan_cells",
        ]
        if not rows:
            if not NAN_INPUT_LOG_PATH.exists():
                pd.DataFrame(columns=expected_cols).to_csv(NAN_INPUT_LOG_PATH, index=False)
            return
        incoming = pd.DataFrame(rows)
        if incoming.empty:
            if not NAN_INPUT_LOG_PATH.exists():
                pd.DataFrame(columns=expected_cols).to_csv(NAN_INPUT_LOG_PATH, index=False)
            return
        for column in expected_cols:
            if column not in incoming.columns:
                incoming[column] = np.nan
        incoming = incoming[expected_cols]
        existing = self._load_table(NAN_INPUT_LOG_PATH)
        if not existing.empty:
            for column in expected_cols:
                if column not in existing.columns:
                    existing[column] = np.nan
            existing = existing[expected_cols]
            combined = pd.concat([existing, incoming], ignore_index=True)
        else:
            combined = incoming
        dedupe_cols = ["game_id", "simulation", "model_key", "stage", "column"]
        if set(dedupe_cols).issubset(combined.columns):
            combined = combined.drop_duplicates(subset=dedupe_cols, keep="last")
            combined = combined.sort_values(dedupe_cols).reset_index(drop=True)
        combined.to_csv(NAN_INPUT_LOG_PATH, index=False)

    def _write_attempt_diagnostics(self, rows: List[Dict[str, object]]) -> None:
        if not self.attempt_diagnostics or not rows:
            return
        incoming = pd.DataFrame(rows)
        existing = self._load_table(ATTEMPT_DIAGNOSTICS_PATH)
        if not existing.empty:
            combined = pd.concat([existing, incoming], ignore_index=True)
        else:
            combined = incoming
        if {"game_id", "simulation"}.issubset(combined.columns):
            combined = combined.drop_duplicates(subset=["game_id", "simulation"], keep="last")
            combined = combined.sort_values(["game_id", "simulation"]).reset_index(drop=True)
        combined.to_csv(ATTEMPT_DIAGNOSTICS_PATH, index=False)

    def _write_playcall_bucket_diagnostics(self, rows: List[Dict[str, object]]) -> None:
        if not self.attempt_diagnostics or not rows:
            return
        incoming = pd.DataFrame(rows)
        existing = self._load_table(PLAYCALL_BUCKET_DIAGNOSTICS_PATH)
        if not existing.empty:
            combined = pd.concat([existing, incoming], ignore_index=True)
        else:
            combined = incoming
        dedupe_cols = ["game_id", "simulation", "bucket_key"]
        if set(dedupe_cols).issubset(combined.columns):
            combined = combined.drop_duplicates(subset=dedupe_cols, keep="last")
            combined = combined.sort_values(dedupe_cols).reset_index(drop=True)
        combined.to_csv(PLAYCALL_BUCKET_DIAGNOSTICS_PATH, index=False)

    def _actual_playcall_bucket_rates(self, game_ids: List[str]) -> pd.DataFrame:
        if not game_ids:
            return pd.DataFrame()
        pbp = self._get_pbp_data().copy()
        if pbp.empty:
            return pd.DataFrame()
        pbp = pbp[pbp["game_id"].astype(str).isin(set(str(x) for x in game_ids))].copy()
        if pbp.empty:
            return pd.DataFrame()

        required = ["game_id", "down", "ydstogo", "score_differential", "qtr", "game_seconds_remaining", "pass_attempt", "rush_attempt"]
        for column in required:
            if column not in pbp.columns:
                return pd.DataFrame()

        frame = pbp[required].copy()
        frame["game_id"] = frame["game_id"].astype(str)
        frame["down"] = pd.to_numeric(frame["down"], errors="coerce")
        frame["ydstogo"] = pd.to_numeric(frame["ydstogo"], errors="coerce")
        frame["score_differential"] = pd.to_numeric(frame["score_differential"], errors="coerce").fillna(0.0)
        frame["qtr"] = pd.to_numeric(frame["qtr"], errors="coerce")
        frame["game_seconds_remaining"] = pd.to_numeric(frame["game_seconds_remaining"], errors="coerce")
        frame["pass_attempt"] = pd.to_numeric(frame["pass_attempt"], errors="coerce").fillna(0.0)
        frame["rush_attempt"] = pd.to_numeric(frame["rush_attempt"], errors="coerce").fillna(0.0)

        frame = frame.dropna(subset=["down", "ydstogo", "qtr", "game_seconds_remaining"])
        frame = frame[(frame["down"] >= 1) & (frame["down"] <= 4)].copy()
        frame = frame[(frame["pass_attempt"] > 0) | (frame["rush_attempt"] > 0)].copy()
        if frame.empty:
            return pd.DataFrame()

        def _clock_remaining(row: pd.Series) -> int:
            quarter = int(row["qtr"])
            game_seconds_remaining = int(row["game_seconds_remaining"])
            qclock = game_seconds_remaining - (4 - quarter) * 900
            return int(np.clip(qclock, 0, 900))

        frame["qclock"] = frame.apply(_clock_remaining, axis=1)
        frame["bucket_key"] = frame.apply(
            lambda row: playcall_bucket_key(
                down=int(row["down"]),
                distance=int(row["ydstogo"]),
                score_diff=float(row["score_differential"]),
                quarter=int(row["qtr"]),
                clock=int(row["qclock"]),
            ),
            axis=1,
        )

        grouped = (
            frame.groupby("bucket_key", as_index=False)[["pass_attempt", "rush_attempt"]]
            .sum(numeric_only=True)
            .rename(columns={"pass_attempt": "actual_pass_calls", "rush_attempt": "actual_run_calls"})
        )
        grouped["actual_off_snaps"] = grouped["actual_pass_calls"] + grouped["actual_run_calls"]
        grouped = grouped[grouped["actual_off_snaps"] > 0].copy()
        grouped["actual_pass_rate"] = grouped["actual_pass_calls"] / grouped["actual_off_snaps"]
        return grouped

    def _write_playcall_bucket_summary(self) -> None:
        if not self.attempt_diagnostics:
            return
        bucket_frame = self._load_table(PLAYCALL_BUCKET_DIAGNOSTICS_PATH)
        if bucket_frame.empty:
            PLAYCALL_BUCKET_SUMMARY_PATH.write_text(json.dumps({}, indent=2), encoding="utf-8")
            return

        numeric_cols = [
            "snaps",
            "off_snaps",
            "pass_calls",
            "run_calls",
            "punt_calls",
            "kick_calls",
            "pass_prob_sum",
            "run_prob_sum",
        ]
        for col in numeric_cols:
            if col in bucket_frame.columns:
                bucket_frame[col] = pd.to_numeric(bucket_frame[col], errors="coerce").fillna(0.0)

        sim_bucket = (
            bucket_frame.groupby("bucket_key", as_index=False)[numeric_cols]
            .sum(numeric_only=True)
        )
        sim_bucket = sim_bucket[sim_bucket["off_snaps"] > 0].copy()
        sim_bucket["sim_pass_rate"] = sim_bucket["pass_calls"] / sim_bucket["off_snaps"]
        sim_bucket["sim_pass_prob_mean"] = np.divide(
            sim_bucket["pass_prob_sum"],
            sim_bucket["snaps"],
            out=np.full(len(sim_bucket), np.nan),
            where=sim_bucket["snaps"] > 0,
        )

        game_ids = sorted(set(bucket_frame["game_id"].astype(str))) if "game_id" in bucket_frame.columns else []
        actual_bucket = self._actual_playcall_bucket_rates(game_ids)
        if actual_bucket.empty:
            summary = {
                "rows": int(len(bucket_frame)),
                "unique_buckets": int(sim_bucket["bucket_key"].nunique()) if "bucket_key" in sim_bucket.columns else 0,
                "games": int(len(game_ids)),
                "note": "actual bucket comparison unavailable",
            }
            PLAYCALL_BUCKET_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            return

        comparison = sim_bucket.merge(actual_bucket, on="bucket_key", how="inner")
        if comparison.empty:
            summary = {
                "rows": int(len(bucket_frame)),
                "unique_buckets": int(sim_bucket["bucket_key"].nunique()) if "bucket_key" in sim_bucket.columns else 0,
                "games": int(len(game_ids)),
                "note": "no overlapping buckets between simulation and actual",
            }
            PLAYCALL_BUCKET_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            return

        comparison["pass_rate_error"] = comparison["sim_pass_rate"] - comparison["actual_pass_rate"]
        comparison["pass_rate_abs_error"] = comparison["pass_rate_error"].abs()
        comparison["pass_prob_minus_actual"] = comparison["sim_pass_prob_mean"] - comparison["actual_pass_rate"]
        comparison = comparison.sort_values("actual_off_snaps", ascending=False).reset_index(drop=True)
        comparison.to_csv(PLAYCALL_BUCKET_COMPARISON_PATH, index=False)

        filtered = comparison[comparison["actual_off_snaps"] >= float(PLAYCALL_BUCKET_MIN_SUPPORT)].copy()
        metric_frame = filtered if not filtered.empty else comparison

        weight = metric_frame["actual_off_snaps"].to_numpy(dtype=float)
        weight_sum = float(np.nansum(weight))
        wmae = None
        if weight_sum > 0:
            wmae = float(np.nansum(metric_frame["pass_rate_abs_error"].to_numpy(dtype=float) * weight) / weight_sum)

        top_errors = []
        for row in metric_frame.sort_values("pass_rate_abs_error", ascending=False).head(15).itertuples(index=False):
            top_errors.append(
                {
                    "bucket_key": str(row.bucket_key),
                    "actual_off_snaps": float(row.actual_off_snaps),
                    "sim_pass_rate": float(row.sim_pass_rate),
                    "actual_pass_rate": float(row.actual_pass_rate),
                    "pass_rate_error": float(row.pass_rate_error),
                    "sim_pass_prob_mean": float(row.sim_pass_prob_mean) if np.isfinite(row.sim_pass_prob_mean) else None,
                }
            )

        summary = {
            "rows": int(len(bucket_frame)),
            "games": int(len(game_ids)),
            "unique_buckets_sim": int(sim_bucket["bucket_key"].nunique()),
            "unique_buckets_actual": int(actual_bucket["bucket_key"].nunique()),
            "overlap_buckets": int(comparison["bucket_key"].nunique()),
            "min_support_actual_off_snaps": int(PLAYCALL_BUCKET_MIN_SUPPORT),
            "overlap_buckets_after_min_support": int(metric_frame["bucket_key"].nunique()),
            "weighted_pass_rate_mae": wmae,
            "mean_pass_rate_error": float(metric_frame["pass_rate_error"].mean()),
            "mean_sim_pass_rate": float(metric_frame["sim_pass_rate"].mean()),
            "mean_actual_pass_rate": float(metric_frame["actual_pass_rate"].mean()),
            "top_bucket_errors": top_errors,
        }
        PLAYCALL_BUCKET_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def _write_attempt_summary(self) -> None:
        if not self.attempt_diagnostics:
            return
        attempt_frame = self._load_table(ATTEMPT_DIAGNOSTICS_PATH)
        if attempt_frame.empty:
            ATTEMPT_SUMMARY_PATH.write_text(json.dumps({}, indent=2), encoding="utf-8")
            return

        for col in [
            "play_call_pass",
            "play_call_run",
            "play_call_punt",
            "play_call_kick",
            "pass_outcomes_total",
            "sim_pass_attempts",
            "sim_rushing_attempts",
            "pass_attempt_delta",
        ]:
            if col in attempt_frame.columns:
                attempt_frame[col] = pd.to_numeric(attempt_frame[col], errors="coerce")

        pass_play_calls = attempt_frame.get("play_call_pass", pd.Series(dtype=float)).mean()
        run_play_calls = attempt_frame.get("play_call_run", pd.Series(dtype=float)).mean()
        total_play_calls = (attempt_frame.get("play_call_pass", 0) + attempt_frame.get("play_call_run", 0))
        pass_play_rate = None
        if isinstance(total_play_calls, pd.Series) and total_play_calls.notna().any():
            denom = total_play_calls.replace(0, np.nan)
            pass_play_rate = float((attempt_frame.get("play_call_pass", 0) / denom).mean())

        summary = {
            "rows": int(len(attempt_frame)),
            "games": int(attempt_frame["game_id"].nunique()) if "game_id" in attempt_frame.columns else 0,
            "mean_play_call_pass": float(pass_play_calls) if np.isfinite(pass_play_calls) else None,
            "mean_play_call_run": float(run_play_calls) if np.isfinite(run_play_calls) else None,
            "mean_pass_play_rate": pass_play_rate,
            "mean_sim_pass_attempts": float(attempt_frame["sim_pass_attempts"].mean()) if "sim_pass_attempts" in attempt_frame.columns else None,
            "mean_sim_rushing_attempts": float(attempt_frame["sim_rushing_attempts"].mean()) if "sim_rushing_attempts" in attempt_frame.columns else None,
            "mean_pass_attempt_delta": float(attempt_frame["pass_attempt_delta"].mean()) if "pass_attempt_delta" in attempt_frame.columns else None,
            # True invariant: pass_attempt_delta should equal -pass_outcome_scramble (scrambles
            # are QB rushes, correctly excluded from pass_attempts) now that sacks are counted
            # into qb.stats["pass_attempts"]. A nonzero residual after removing the expected
            # scramble offset indicates a real accounting bug.
            "invariant_violations": (
                int(((attempt_frame["pass_attempt_delta"] + attempt_frame.get("pass_outcome_scramble", 0)).abs() > 0).sum())
                if "pass_attempt_delta" in attempt_frame.columns
                else None
            ),
        }

        weekly = []
        if "game_id" in attempt_frame.columns and not self.results.empty and {"game_id", "week"}.issubset(self.results.columns):
            map_frame = self.results[["game_id", "week"]].drop_duplicates().copy()
            merged = attempt_frame.merge(map_frame, on="game_id", how="left")
            for week, week_df in merged.groupby("week", dropna=True):
                weekly.append(
                    {
                        "week": int(week),
                        "rows": int(len(week_df)),
                        "mean_pass_play_rate": float((week_df["play_call_pass"] / (week_df["play_call_pass"] + week_df["play_call_run"]).replace(0, np.nan)).mean()),
                        "mean_sim_pass_attempts": float(week_df["sim_pass_attempts"].mean()),
                        "mean_sim_rushing_attempts": float(week_df["sim_rushing_attempts"].mean()),
                        "mean_pass_attempt_delta": float(week_df["pass_attempt_delta"].mean()),
                    }
                )
        summary["by_week"] = sorted(weekly, key=lambda x: x["week"])

        actual_stats = self._get_actual_player_game_stats()
        game_attempt_comparison: List[Dict[str, object]] = []
        if "game_id" in attempt_frame.columns and not actual_stats.empty and "game_id" in actual_stats.columns:
            sim_game = (
                attempt_frame.groupby("game_id", as_index=False)[["sim_pass_attempts", "sim_rushing_attempts"]]
                .mean(numeric_only=True)
                .rename(
                    columns={
                        "sim_pass_attempts": "sim_pass_attempts_mean",
                        "sim_rushing_attempts": "sim_rushing_attempts_mean",
                    }
                )
            )
            actual_game = (
                actual_stats.groupby("game_id", as_index=False)[["attempts", "rushing_attempts"]]
                .sum(numeric_only=True)
                .rename(
                    columns={
                        "attempts": "actual_pass_attempts",
                        "rushing_attempts": "actual_rushing_attempts",
                    }
                )
            )
            compare = sim_game.merge(actual_game, on="game_id", how="inner")
            if not compare.empty:
                compare["pass_attempt_delta"] = compare["sim_pass_attempts_mean"] - compare["actual_pass_attempts"]
                compare["rush_attempt_delta"] = compare["sim_rushing_attempts_mean"] - compare["actual_rushing_attempts"]
                for row in compare.itertuples(index=False):
                    game_attempt_comparison.append(
                        {
                            "game_id": str(row.game_id),
                            "sim_pass_attempts_mean": float(row.sim_pass_attempts_mean),
                            "actual_pass_attempts": float(row.actual_pass_attempts),
                            "pass_attempt_delta": float(row.pass_attempt_delta),
                            "sim_rushing_attempts_mean": float(row.sim_rushing_attempts_mean),
                            "actual_rushing_attempts": float(row.actual_rushing_attempts),
                            "rush_attempt_delta": float(row.rush_attempt_delta),
                        }
                    )

                summary["actual_vs_sim"] = {
                    "games_compared": int(len(compare)),
                    "mean_pass_attempt_delta": float(compare["pass_attempt_delta"].mean()),
                    "mean_rush_attempt_delta": float(compare["rush_attempt_delta"].mean()),
                }

                # Points-per-100-offensive-plays: normalizes the point-production gap by the
                # play-volume gap, so a lower sim total can be attributed to "fewer plays" vs
                # "less efficient scoring per play".
                compare["sim_plays"] = compare["sim_pass_attempts_mean"] + compare["sim_rushing_attempts_mean"]
                compare["actual_plays"] = compare["actual_pass_attempts"] + compare["actual_rushing_attempts"]
                totals_lookup = self.results[["game_id", "sim_total", "actual_total"]].copy() if not self.results.empty else pd.DataFrame()
                if not totals_lookup.empty:
                    totals_lookup["game_id"] = totals_lookup["game_id"].astype(str)
                    plays_points = compare.merge(totals_lookup, on="game_id", how="inner")
                    plays_points = plays_points[(plays_points["sim_plays"] > 0) & (plays_points["actual_plays"] > 0)]
                    if not plays_points.empty:
                        sim_plays_total = float(plays_points["sim_plays"].sum())
                        actual_plays_total = float(plays_points["actual_plays"].sum())
                        sim_points_total = float(plays_points["sim_total"].sum())
                        actual_points_total = float(plays_points["actual_total"].sum())
                        summary["play_volume_vs_scoring"] = {
                            "games_compared": int(len(plays_points)),
                            "mean_sim_plays_per_game": float(plays_points["sim_plays"].mean()),
                            "mean_actual_plays_per_game": float(plays_points["actual_plays"].mean()),
                            "mean_plays_delta": float((plays_points["sim_plays"] - plays_points["actual_plays"]).mean()),
                            "sim_points_per_100_plays": float(sim_points_total / sim_plays_total * 100.0) if sim_plays_total > 0 else None,
                            "actual_points_per_100_plays": float(actual_points_total / actual_plays_total * 100.0) if actual_plays_total > 0 else None,
                        }

        # Penalty effect comparison: sim penalty-driven yardage/first-downs vs actual penalty
        # yardage for the same games, to test whether the penalty model meaningfully closes
        # (or fails to close) the totals gap.
        #
        # IMPORTANT: `penalty_yards_sum` is a NET, offense-relative signed value (positive when
        # a penalty benefits whichever team currently has the ball, negative when it hurts
        # them). Across many penalty events in a game this naturally trends toward ~0 even for
        # a well-calibrated model, since roughly half of penalties are called on the offense
        # and half on the defense. It is NOT comparable to `actual_penalty_yards`, which is a
        # GROSS/unsigned total (nflverse's `penalty_yards` column is always positive, summed
        # per team regardless of who benefited — this is what the HTML report's "Penalty yds
        # called: TEAM X, TEAM Y" line shows). The valid apples-to-apples comparison uses
        # `penalty_yards_abs_sum` (sum of |penalty_yards| per event) vs `actual_penalty_yards`.
        if "game_id" in attempt_frame.columns and {
            "penalty_no_play_count", "penalty_on_play_count", "penalty_first_down_count",
            "penalty_yards_sum", "penalty_yards_abs_sum",
        }.issubset(attempt_frame.columns):
            sim_penalty_game = attempt_frame.groupby("game_id", as_index=False)[
                ["penalty_no_play_count", "penalty_on_play_count", "penalty_first_down_count",
                 "penalty_yards_sum", "penalty_yards_abs_sum"]
            ].mean(numeric_only=True)
            game_ids = sorted(set(attempt_frame["game_id"].astype(str)))
            actual_penalty = self._get_actual_team_penalty_stats(game_ids)
            if not actual_penalty.empty:
                actual_penalty_game = actual_penalty.groupby("game_id", as_index=False)["actual_penalty_yards"].sum(numeric_only=True)
                penalty_compare = sim_penalty_game.merge(actual_penalty_game, on="game_id", how="inner")
                if not penalty_compare.empty:
                    summary["penalty_comparison"] = {
                        "games_compared": int(len(penalty_compare)),
                        "mean_sim_penalty_no_play_count": float(penalty_compare["penalty_no_play_count"].mean()),
                        "mean_sim_penalty_on_play_count": float(penalty_compare["penalty_on_play_count"].mean()),
                        "mean_sim_penalty_first_down_count": float(penalty_compare["penalty_first_down_count"].mean()),
                        "mean_sim_penalty_yards_net": float(penalty_compare["penalty_yards_sum"].mean()),
                        "mean_sim_penalty_yards_gross": float(penalty_compare["penalty_yards_abs_sum"].mean()),
                        "mean_actual_penalty_yards_gross": float(penalty_compare["actual_penalty_yards"].mean()),
                        "mean_penalty_yards_gross_delta": float((penalty_compare["penalty_yards_abs_sum"] - penalty_compare["actual_penalty_yards"]).mean()),
                    }


        # Kick (FG/PAT) make-rate comparison: sim vs actual for the same games.
        if "game_id" in attempt_frame.columns and {"fg_attempts", "fg_made", "pat_attempts", "pat_made"}.issubset(attempt_frame.columns):
            sim_fg_attempts = float(attempt_frame["fg_attempts"].sum())
            sim_fg_made = float(attempt_frame["fg_made"].sum())
            sim_pat_attempts = float(attempt_frame["pat_attempts"].sum())
            sim_pat_made = float(attempt_frame["pat_made"].sum())
            game_ids = sorted(set(attempt_frame["game_id"].astype(str)))
            pbp = self._get_pbp_data()
            kick_summary: Dict[str, object] = {
                "sim_fg_attempts": sim_fg_attempts,
                "sim_fg_made": sim_fg_made,
                "sim_fg_pct": float(sim_fg_made / sim_fg_attempts) if sim_fg_attempts > 0 else None,
                "sim_pat_attempts": sim_pat_attempts,
                "sim_pat_made": sim_pat_made,
                "sim_pat_pct": float(sim_pat_made / sim_pat_attempts) if sim_pat_attempts > 0 else None,
            }
            if not pbp.empty and "game_id" in pbp.columns:
                game_pbp = pbp[pbp["game_id"].astype(str).isin(set(game_ids))]
                if "field_goal_attempt" in game_pbp.columns and "field_goal_result" in game_pbp.columns:
                    actual_fg_attempts = float(pd.to_numeric(game_pbp["field_goal_attempt"], errors="coerce").fillna(0.0).sum())
                    actual_fg_made = float((game_pbp["field_goal_result"] == "made").sum())
                    kick_summary["actual_fg_attempts"] = actual_fg_attempts
                    kick_summary["actual_fg_made"] = actual_fg_made
                    kick_summary["actual_fg_pct"] = float(actual_fg_made / actual_fg_attempts) if actual_fg_attempts > 0 else None
                if "extra_point_attempt" in game_pbp.columns and "extra_point_result" in game_pbp.columns:
                    actual_pat_attempts = float(pd.to_numeric(game_pbp["extra_point_attempt"], errors="coerce").fillna(0.0).sum())
                    actual_pat_made = float((game_pbp["extra_point_result"] == "good").sum())
                    kick_summary["actual_pat_attempts"] = actual_pat_attempts
                    kick_summary["actual_pat_made"] = actual_pat_made
                    kick_summary["actual_pat_pct"] = float(actual_pat_made / actual_pat_attempts) if actual_pat_attempts > 0 else None
            summary["kick_comparison"] = kick_summary

        # Pass-outcome rate comparison: sim sack/completion/INT/incomplete rates vs actual
        # league rates for the same games, plus mean realized yards-per-play for rush/complete
        # pass/scramble to look for a systematic negative shift in the yardage-bin models.
        pass_outcome_cols = [
            "pass_outcome_sack", "pass_outcome_complete", "pass_outcome_interception",
            "pass_outcome_scramble", "pass_outcome_incomplete", "pass_outcomes_total",
        ]
        if "game_id" in attempt_frame.columns and set(pass_outcome_cols).issubset(attempt_frame.columns):
            totals = attempt_frame[pass_outcome_cols].sum(numeric_only=True)
            denom = float(totals["pass_outcomes_total"])
            sim_rates = {
                "sim_sack_rate": float(totals["pass_outcome_sack"] / denom) if denom > 0 else None,
                "sim_completion_rate": float(totals["pass_outcome_complete"] / denom) if denom > 0 else None,
                "sim_interception_rate": float(totals["pass_outcome_interception"] / denom) if denom > 0 else None,
                "sim_scramble_rate": float(totals["pass_outcome_scramble"] / denom) if denom > 0 else None,
                "sim_incomplete_rate": float(totals["pass_outcome_incomplete"] / denom) if denom > 0 else None,
            }
            game_ids = sorted(set(attempt_frame["game_id"].astype(str)))
            pbp = self._get_pbp_data()
            actual_rates: Dict[str, object] = {}
            if not pbp.empty and "game_id" in pbp.columns and "pass_attempt" in pbp.columns:
                game_pbp = pbp[pbp["game_id"].astype(str).isin(set(game_ids))]
                pass_attempts = pd.to_numeric(game_pbp["pass_attempt"], errors="coerce").fillna(0.0)
                actual_denom = float(pass_attempts.sum())
                if actual_denom > 0:
                    def _actual_rate(column: str) -> Optional[float]:
                        if column not in game_pbp.columns:
                            return None
                        return float(pd.to_numeric(game_pbp[column], errors="coerce").fillna(0.0).sum() / actual_denom)

                    actual_rates = {
                        "actual_sack_rate": _actual_rate("sack"),
                        "actual_completion_rate": _actual_rate("complete_pass"),
                        "actual_interception_rate": _actual_rate("interception"),
                        "actual_incomplete_rate": _actual_rate("incomplete_pass"),
                    }
            summary["pass_outcome_comparison"] = {**sim_rates, **actual_rates}

        # Mean realized yards-per-play for the ordinal/CORN yardage-bin models.
        yardage_pairs = [
            ("rush_plays", "rush_yards_sum", "mean_sim_rush_yards_per_play"),
            ("complete_pass_plays", "complete_pass_yards_sum", "mean_sim_complete_pass_yards_per_play"),
            ("scramble_plays", "scramble_yards_sum", "mean_sim_scramble_yards_per_play"),
            ("air_plays", "air_yards_sum", "mean_sim_air_yards_per_attempt"),
            ("yac_plays", "yac_yards_sum", "mean_sim_yac_yards_per_completion"),
            ("punt_plays", "punt_yards_sum", "mean_sim_punt_yards_per_punt"),
            ("time_plays", "time_elapsed_sum", "mean_sim_time_elapsed_per_timed_play"),
        ]
        yardage_summary: Dict[str, object] = {}
        for plays_col, yards_col, out_key in yardage_pairs:
            if plays_col in attempt_frame.columns and yards_col in attempt_frame.columns:
                total_plays = float(attempt_frame[plays_col].sum())
                total_yards = float(attempt_frame[yards_col].sum())
                yardage_summary[out_key] = float(total_yards / total_plays) if total_plays > 0 else None
        if yardage_summary:
            summary["yardage_per_play"] = yardage_summary

        summary["game_attempt_comparison"] = game_attempt_comparison
        ATTEMPT_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def _already_processed(self, game_id: str) -> bool:
        if not self.results.empty and "game_id" in self.results.columns:
            return game_id in set(self.results["game_id"].astype(str))
        return False

    def _build_live_tracker_line(
        self,
        game_id: str,
        home_team: str,
        away_team: str,
        completed: int,
        home_scores: List[int],
        away_scores: List[int],
        spread_line: float,
    ) -> str:
        spreads = np.asarray(home_scores, dtype=float) - np.asarray(away_scores, dtype=float)
        totals = np.asarray(home_scores, dtype=float) + np.asarray(away_scores, dtype=float)
        home_wins = int(np.sum(spreads > 0))
        away_wins = int(np.sum(spreads < 0))
        ties = int(np.sum(spreads == 0))
        decisions = max(home_wins + away_wins, 1)
        home_win_pct = home_wins / decisions

        spread_mean = float(np.mean(spreads))
        spread_median = float(np.quantile(spreads, 0.5))
        total_median = float(np.quantile(totals, 0.5))
        tracker = (
            f"[{game_id}] sims {completed}/{self.simulations} | "
            f"{home_team} win% {home_win_pct:.1%} ({home_wins}-{away_wins}"
            f"{'-' + str(ties) if ties else ''}) | "
            f"spread {home_team}-{away_team} mean {spread_mean:+.2f}, median {spread_median:+.1f} | "
            f"total median {total_median:.1f}"
        )
        if np.isfinite(spread_line):
            home_cover_pct = float(np.mean(spreads > spread_line))
            tracker += f" | cover vs {spread_line:+.1f}: {home_cover_pct:.1%}"
        return tracker

    def _print_live_tracker(
        self,
        game_id: str,
        home_team: str,
        away_team: str,
        completed: int,
        home_scores: List[int],
        away_scores: List[int],
        spread_line: float,
        final: bool = False,
    ) -> None:
        if completed <= 0:
            return
        line = self._build_live_tracker_line(
            game_id=game_id,
            home_team=home_team,
            away_team=away_team,
            completed=completed,
            home_scores=home_scores,
            away_scores=away_scores,
            spread_line=spread_line,
        )
        if self.print_game_summaries:
            if final:
                return
            print(line, flush=True)
            return
        print(line, end="\n" if final else "\r", flush=True)

    def _simulate_completed_game(self, game_row) -> Dict[str, object]:
        game_id = game_row.game_id
        package = self._build_game_package(game_row)
        sim_rows: List[Dict[str, object]] = []
        player_rows: List[Dict[str, object]] = []
        attempt_rows: List[Dict[str, object]] = []
        playcall_bucket_rows: List[Dict[str, object]] = []
        nan_input_log_rows: List[Dict[str, object]] = []
        home_scores: List[int] = []
        away_scores: List[int] = []
        spread_line = float(game_row.spread_line)

        serializable_package = self._serialize_game_package(package)

        if self.workers > 1 and self.simulations > 1:
            with ProcessPoolExecutor(
                max_workers=self.workers,
                initializer=_init_sim_worker,
                initargs=(str(self.runtime.data_dir), self.strict_schema),
            ) as executor:
                futures = [
                    executor.submit(
                        _simulate_single_game_run,
                        {
                            "data_dir": str(self.runtime.data_dir),
                            "game_id": game_id,
                            "simulation": index + 1,
                            "simulations": self.simulations,
                            "home_team": package["home_team"],
                            "away_team": package["away_team"],
                            "roof": package["roof"],
                            "home_players": serializable_package["home_players"],
                            "away_players": serializable_package["away_players"],
                            "team_features": serializable_package["team_features"],
                            "print_game_summaries": self.print_game_summaries,
                            "strict_schema": self.strict_schema,
                            "attempt_diagnostics": self.attempt_diagnostics,
                            "live_pbp": self.live_pbp,
                        },
                    )
                    for index in range(self.simulations)
                ]

                for future in as_completed(futures):
                    result = future.result()
                    home_scores.append(int(result["home_score"]))
                    away_scores.append(int(result["away_score"]))
                    sim_rows.append(result["sim_row"])
                    player_rows.extend(result["player_rows"])
                    attempt_row = result.get("attempt_diagnostics")
                    if isinstance(attempt_row, dict):
                        attempt_rows.append(attempt_row)
                    bucket_rows = result.get("playcall_bucket_rows")
                    if isinstance(bucket_rows, list):
                        playcall_bucket_rows.extend([row for row in bucket_rows if isinstance(row, dict)])
                    nan_rows = result.get("nan_input_log_rows")
                    if isinstance(nan_rows, list):
                        for row in nan_rows:
                            if isinstance(row, dict):
                                enriched = dict(row)
                                enriched["game_id"] = str(game_id)
                                enriched["simulation"] = int(result["sim_row"]["simulation"])
                                nan_input_log_rows.append(enriched)
                    self._print_live_tracker(
                        game_id=game_id,
                        home_team=package["home_team"],
                        away_team=package["away_team"],
                        completed=len(home_scores),
                        home_scores=home_scores,
                        away_scores=away_scores,
                        spread_line=spread_line,
                    )

            sim_rows.sort(key=lambda row: int(row["simulation"]))
            player_rows.sort(key=lambda row: (int(row["simulation"]), str(row["team"]), str(row["player_name"])))
        else:
            for index in range(self.simulations):
                self.runtime.reset_nan_input_log()
                away = Team(package["away_team"], package["away_players"], "away", package["team_features"].loc[package["away_team"]], self.runtime)
                home = Team(package["home_team"], package["home_players"], "home", package["team_features"].loc[package["home_team"]], self.runtime)
                game = Game(away, home, package["roof"], self.runtime)
                summary = game.simulate_game(
                    print_summary=self.print_game_summaries,
                    summary_prefix=f"[{game_id} sim {index + 1}/{self.simulations}]",
                    live_pbp=self.live_pbp,
                )
                if self.attempt_diagnostics:
                    attempt_row = game.diagnostics_row()
                    attempt_row.update(
                        {
                            "game_id": game_id,
                            "simulation": index + 1,
                            "home_team": package["home_team"],
                            "away_team": package["away_team"],
                        }
                    )
                    attempt_rows.append(attempt_row)
                    bucket_rows = game.diagnostics_bucket_rows()
                    for row in bucket_rows:
                        row.update(
                            {
                                "game_id": game_id,
                                "simulation": index + 1,
                                "home_team": package["home_team"],
                                "away_team": package["away_team"],
                            }
                        )
                        playcall_bucket_rows.append(row)
                home_scores.append(home.score)
                away_scores.append(away.score)
                sim_rows.append(
                    {
                        "simulation": index + 1,
                        "home": home.score,
                        "away": away.score,
                        "qb": summary["qb"].name if summary.get("qb") else "",
                        "receiver": summary["receiver"].name if summary.get("receiver") else "",
                        "rusher": summary["rusher"].name if summary.get("rusher") else "",
                    }
                )

                for team in [home, away]:
                    for player in team.players:
                        row: Dict[str, object] = {
                            "simulation": index + 1,
                            "team": team.name,
                            "player_name": player.name,
                            "player_id": player.player_id,
                            "position": player.position,
                        }
                        row.update(player.stats)
                        player_rows.append(row)

                for row in self.runtime.get_nan_input_log_rows():
                    enriched = dict(row)
                    enriched["game_id"] = str(game_id)
                    enriched["simulation"] = int(index + 1)
                    nan_input_log_rows.append(enriched)

                self._print_live_tracker(
                    game_id=game_id,
                    home_team=package["home_team"],
                    away_team=package["away_team"],
                    completed=index + 1,
                    home_scores=home_scores,
                    away_scores=away_scores,
                    spread_line=spread_line,
                )

        self._print_live_tracker(
            game_id=game_id,
            home_team=package["home_team"],
            away_team=package["away_team"],
            completed=len(home_scores),
            home_scores=home_scores,
            away_scores=away_scores,
            spread_line=spread_line,
            final=True,
        )

        raw_frame = pd.DataFrame(sim_rows)
        raw_frame.to_csv(self.raw_results_dir / f"{game_id}.csv", index=False)
        player_frame = pd.DataFrame(player_rows)
        player_frame.to_csv(self.raw_results_dir / f"{game_id}_players.csv", index=False)
        self._write_attempt_diagnostics(attempt_rows)
        self._write_playcall_bucket_diagnostics(playcall_bucket_rows)
        self._write_nan_input_log(nan_input_log_rows)

        home_scores_arr = np.asarray(home_scores, dtype=float)
        away_scores_arr = np.asarray(away_scores, dtype=float)
        home_prob = float(np.mean(home_scores_arr > away_scores_arr) / max(np.mean(home_scores_arr != away_scores_arr), 1e-9))
        overtime_flag = getattr(game_row, "overtime", np.nan)
        try:
            is_ot = float(overtime_flag) if pd.notna(overtime_flag) else np.nan
        except (TypeError, ValueError):
            is_ot = np.nan
        return {
            "game_id": game_id,
            "week": int(game_row.week),
            "home_team": package["home_team"],
            "away_team": package["away_team"],
            "actual_home_score": float(game_row.home_score),
            "actual_away_score": float(game_row.away_score),
            "actual_spread": float(game_row.home_score - game_row.away_score),
            "actual_total": float(game_row.home_score + game_row.away_score),
            "sim_home_median": float(np.quantile(home_scores_arr, 0.5)),
            "sim_away_median": float(np.quantile(away_scores_arr, 0.5)),
            "sim_spread": float(np.quantile(home_scores_arr - away_scores_arr, 0.5)),
            "sim_total": float(np.quantile(home_scores_arr + away_scores_arr, 0.5)),
            "sim_home_win_prob": home_prob,
            "spread_line": float(game_row.spread_line),
            "total_line": float(game_row.total_line),
            "home_moneyline": float(game_row.home_moneyline),
            "away_moneyline": float(game_row.away_moneyline),
            "simulations_run": int(self.simulations),
            "is_ot": is_ot,
        }

    def _serialize_game_package(self, package: Dict[str, object]) -> Dict[str, object]:
        player_feature_cols = self.runtime.required_player_feature_columns
        team_feature_cols = self.runtime.required_team_feature_columns

        def serialize_players(players: List[Player]) -> List[Dict[str, object]]:
            snapshots: List[Dict[str, object]] = []
            for player in players:
                values = {
                    column: float(player.attributes.iloc[0].get(column, 0.0))
                    for column in player_feature_cols
                }
                snapshots.append(
                    {
                        "name": player.name,
                        "position": player.position,
                        "player_id": player.player_id,
                        "attributes": values,
                    }
                )
            return snapshots

        team_features: Dict[str, Dict[str, float]] = {}
        for team_name in [package["home_team"], package["away_team"]]:
            team_features[str(team_name)] = {
                column: float(package["team_features"].loc[team_name].get(column, 0.0))
                for column in team_feature_cols
            }

        return {
            "home_players": serialize_players(package["home_players"]),
            "away_players": serialize_players(package["away_players"]),
            "team_features": team_features,
        }

    def _derive_team_rest_days(self, team: str, gameday: pd.Timestamp) -> float:
        prior_games = self.team_stats[
            (self.team_stats["team"].astype(str) == str(team))
            & (self.team_stats["game_date"] < gameday)
        ].copy()
        if prior_games.empty:
            return float("nan")
        last_game_date = pd.to_datetime(prior_games["game_date"], errors="coerce").max()
        if pd.isna(last_game_date):
            return float("nan")
        rest_days = float((pd.Timestamp(gameday).normalize() - pd.Timestamp(last_game_date).normalize()).days)
        return cap_rest_days(rest_days) if rest_days > 0 else float("nan")

    def _build_game_package(self, game_row) -> Dict[str, object]:
        game_id = game_row.game_id
        gameday = pd.Timestamp(game_row.gameday)
        home_team = str(game_row.home_team)
        away_team = str(game_row.away_team)
        team_features = self.team_stats[self.team_stats["game_id"].astype(str) == game_id].copy()
        if team_features["team"].nunique() != 2:
            raise SkipGameError(f"missing team features for {game_id}")
        team_features = team_features.set_index("team").reindex([home_team, away_team])
        if team_features.isna().all(axis=1).any():
            raise SkipGameError(f"missing one side of team features for {game_id}")
        team_features = team_features.reindex(columns=self.runtime.required_team_feature_columns)

        home_rest_schedule = pd.to_numeric(pd.Series([getattr(game_row, "home_rest", np.nan)]), errors="coerce").iloc[0]
        away_rest_schedule = pd.to_numeric(pd.Series([getattr(game_row, "away_rest", np.nan)]), errors="coerce").iloc[0]
        home_rest_history = self._derive_team_rest_days(home_team, gameday)
        away_rest_history = self._derive_team_rest_days(away_team, gameday)

        home_rest = (
            cap_rest_days(float(home_rest_history))
            if pd.notna(home_rest_history)
            else (cap_rest_days(float(home_rest_schedule)) if pd.notna(home_rest_schedule) else 7.0)
        )
        away_rest = (
            cap_rest_days(float(away_rest_history))
            if pd.notna(away_rest_history)
            else (cap_rest_days(float(away_rest_schedule)) if pd.notna(away_rest_schedule) else 7.0)
        )
        if "days_rest" in team_features.columns:
            team_features.loc[home_team, "days_rest"] = home_rest
            team_features.loc[away_team, "days_rest"] = away_rest
        if "opp_days_rest" in team_features.columns:
            team_features.loc[home_team, "opp_days_rest"] = away_rest
            team_features.loc[away_team, "opp_days_rest"] = home_rest

        home_slots = self._build_team_slots(home_team, int(game_row.week), gameday)
        away_slots = self._build_team_slots(away_team, int(game_row.week), gameday)
        home_players = self._build_player_objects(home_slots, home_team, int(game_row.week), gameday)
        away_players = self._build_player_objects(away_slots, away_team, int(game_row.week), gameday)

        return {
            "game_id": game_id,
            "gameday": gameday,
            "home_team": home_team,
            "away_team": away_team,
            "home_players": home_players,
            "away_players": away_players,
            "team_features": team_features,
            "roof": ROOF_TYPE.get(home_team, "outdoors"),
        }

    def _build_team_slots(self, team: str, week: int, gameday: pd.Timestamp) -> Dict[str, Dict[str, str]]:
        roster_week = self.rosters[(self.rosters["team"] == team) & (self.rosters["week"] == week)].copy()
        roster_week = roster_week[roster_week["player_id"].str.len() > 0]
        roster_week["player_id"] = roster_week["player_id"].astype(str)
        active_ids = set(roster_week.loc[roster_week["status"].isin(ACTIVE_STATUSES), "player_id"].astype(str))
        if not active_ids:
            active_ids = set(roster_week["player_id"].astype(str))

        depth = self.depth_charts[
            (self.depth_charts["team"] == team)
            & (self.depth_charts["dt"] <= gameday + pd.Timedelta(days=1))
        ].copy()
        if depth.empty:
            raise SkipGameError(f"missing depth chart snapshot for {team} before {gameday.date()}")
        latest_dt = depth["dt"].max()
        depth = depth[depth["dt"] == latest_dt].copy()

        slots: Dict[str, Dict[str, str]] = {}
        used_ids: set[str] = set()
        for position, limit in SLOT_LIMITS.items():
            position_depth = depth[depth["pos_abb"] == position].copy()
            selected_rows: List[Dict[str, str]] = []

            if not position_depth.empty:
                position_depth["active_priority"] = ~position_depth["gsis_id"].isin(active_ids)
                position_depth = position_depth.sort_values(["active_priority", "pos_rank", "player_name"])
                for row in position_depth.itertuples(index=False):
                    player_id = str(row.gsis_id)
                    if player_id in used_ids:
                        continue
                    selected_rows.append(
                        {
                            "player_id": player_id,
                            "player_name": str(row.player_name),
                            "position": position,
                        }
                    )
                    used_ids.add(player_id)
                    if len(selected_rows) >= limit:
                        break

            if len(selected_rows) < limit:
                fallback = roster_week[roster_week["position"] == position].copy()
                if not fallback.empty:
                    fallback["active_priority"] = ~fallback["player_id"].isin(active_ids)
                    fallback = fallback.sort_values(["active_priority", "player_name"])
                    for row in fallback.itertuples(index=False):
                        player_id = str(row.player_id)
                        if player_id in used_ids:
                            continue
                        selected_rows.append(
                            {
                                "player_id": player_id,
                                "player_name": str(row.player_name),
                                "position": position,
                            }
                        )
                        used_ids.add(player_id)
                        if len(selected_rows) >= limit:
                            break

            while len(selected_rows) < limit:
                slot_number = len(selected_rows) + 1
                selected_rows.append(
                    {
                        "player_id": f"placeholder_{team}_{position}_{week}_{slot_number}",
                        "player_name": f"{team} {position} Placeholder {slot_number}",
                        "position": position,
                    }
                )

            for slot_index, row in enumerate(selected_rows, start=1):
                slots[f"{position}{slot_index}"] = row
        return slots

    def _build_player_objects(
        self,
        slot_map: Dict[str, Dict[str, str]],
        team: str,
        week: int,
        gameday: pd.Timestamp,
    ) -> List[Player]:
        player_ids = [entry["player_id"] for entry in slot_map.values()]
        prior_player_rows = self.player_stats[
            (self.player_stats["player_id"].astype(str).isin(player_ids))
            & (self.player_stats["game_date"] <= gameday)
        ].copy()
        latest_prior = (
            prior_player_rows.sort_values(["player_id", "game_date"], ascending=[True, False])
            .groupby("player_id", as_index=False)
            .first()
        )

        roster_week = self.rosters[(self.rosters["team"] == team) & (self.rosters["week"] == week)].copy()
        prior_pool = self.player_stats[self.player_stats["game_date"] <= gameday].copy()
        latest_pool = (
            prior_pool.sort_values(["player_id", "game_date"], ascending=[True, False])
            .groupby("player_id", as_index=False)
            .first()
        )
        pool_with_position = latest_pool.merge(
            roster_week[["player_id", "position"]].drop_duplicates(),
            on="player_id",
            how="inner",
        )
        position_means = pool_with_position.groupby("position")[self.runtime.required_player_feature_columns].mean(numeric_only=True)
        global_means = latest_pool[self.runtime.required_player_feature_columns].mean(numeric_only=True)

        players: List[Player] = []
        for slot in SKILL_SLOT_ORDER:
            slot_entry = slot_map.get(slot)
            if slot_entry is None:
                raise SkipGameError(f"missing slot {slot} for {team} week {week}")
            player_row = latest_prior[latest_prior["player_id"].astype(str) == slot_entry["player_id"]].copy()
            has_prior_player_history = not player_row.empty
            if player_row.empty:
                player_row = pd.DataFrame([{"player_id": slot_entry["player_id"], "player_name": slot_entry["player_name"]}])
            if "player_name" not in player_row.columns or pd.isna(player_row.iloc[0].get("player_name")):
                player_row.loc[player_row.index[0], "player_name"] = slot_entry["player_name"]
            player_row = player_row.reindex(columns=["player_id", "player_name", *self.runtime.required_player_feature_columns], fill_value=np.nan)
            fill_source = position_means.loc[slot_entry["position"]] if slot_entry["position"] in position_means.index else global_means
            for column in self.runtime.required_player_feature_columns:
                value = player_row.iloc[0][column]
                if pd.isna(value):
                    if not has_prior_player_history and column in {"qb_team_games_played", "qb_dome_rate_bayes"}:
                        player_row.loc[player_row.index[0], column] = 0.0
                        continue
                    fallback_value = fill_source.get(column, np.nan)
                    if pd.isna(fallback_value):
                        fallback_value = global_means.get(column, np.nan)
                    if pd.isna(fallback_value):
                        fallback_value = 0.0
                    player_row.loc[player_row.index[0], column] = fallback_value
            if is_empty_slot(slot_entry["player_id"]):
                for column in set(USAGE_COLUMNS) & set(self.runtime.required_player_feature_columns):
                    player_row.loc[player_row.index[0], column] = 0.0
            players.append(
                Player(
                    name=str(player_row.iloc[0]["player_name"]),
                    position=slot,
                    player_id=str(player_row.iloc[0]["player_id"]),
                    attributes=player_row,
                )
            )
        return players

    def _record_missing(self, game_row, reason: str) -> None:
        missing_row = {
            "game_id": game_row.game_id,
            "week": int(game_row.week),
            "home_team": str(game_row.home_team),
            "away_team": str(game_row.away_team),
            "reason": reason,
        }
        self.missing = pd.concat([self.missing, pd.DataFrame([missing_row])], ignore_index=True)
        self.missing = self.missing.drop_duplicates(subset=["game_id"], keep="last")
        self.missing.to_csv(MISSING_PATH, index=False)
        self._write_summary()

    def _write_summary(self) -> None:
        schedule_games = self.schedule["game_id"].nunique()
        results = self.results.copy()
        slices = self._build_slice_metrics(results)
        schema_issues = list(getattr(self.runtime, "schema_issues", []))
        summary = {
            "games_scored": float(len(results)),
            "brier": None,
            "spread_rmse": None,
            "total_rmse": None,
            "vegas_brier": None,
            "mean_total_bias": None,
            "mean_total_bias_vegas": None,
            "mean_spread_bias": None,
            "ot_games": None,
            "mean_total_bias_ex_ot": None,
            "mean_total_bias_vegas_ex_ot": None,
            "total_rmse_ex_ot": None,
            "missing_games": float(max(schedule_games - len(results), 0)),
            "schema_issues": schema_issues,
            "schema_issue_count": float(len(schema_issues)),
            "by_week": slices["by_week"],
            "by_team": slices["by_team"],
        }
        if not results.empty:
            results["actual_home_win"] = (results["actual_spread"] > 0).astype(float)
            results["sim_brier_score"] = (results["sim_home_win_prob"] - results["actual_home_win"]) ** 2
            results["home_odds"] = results["home_moneyline"].apply(implied_prob_from_moneyline)
            results["away_odds"] = results["away_moneyline"].apply(implied_prob_from_moneyline)
            normalizer = results["home_odds"] + results["away_odds"]
            results["vegas_home_prob"] = np.divide(results["home_odds"], normalizer, out=np.full(len(results), np.nan), where=normalizer > 0)
            results["vegas_brier_score"] = (results["vegas_home_prob"] - results["actual_home_win"]) ** 2
            results["total_bias"] = pd.to_numeric(results["sim_total"], errors="coerce") - pd.to_numeric(results["actual_total"], errors="coerce")
            results["total_bias_vegas"] = pd.to_numeric(results["sim_total"], errors="coerce") - pd.to_numeric(results["total_line"], errors="coerce")
            results["spread_bias"] = pd.to_numeric(results["sim_spread"], errors="coerce") - pd.to_numeric(results["actual_spread"], errors="coerce")
            summary["brier"] = float(results["sim_brier_score"].mean())
            summary["spread_rmse"] = float(np.sqrt(((results["sim_spread"] - results["actual_spread"]) ** 2).mean()))
            summary["total_rmse"] = float(np.sqrt(((results["sim_total"] - results["actual_total"]) ** 2).mean()))
            summary["vegas_brier"] = float(results["vegas_brier_score"].dropna().mean()) if results["vegas_brier_score"].notna().any() else None
            summary["mean_total_bias"] = self._safe_group_mean(results, "total_bias")
            summary["mean_total_bias_vegas"] = self._safe_group_mean(results, "total_bias_vegas")
            summary["mean_spread_bias"] = self._safe_group_mean(results, "spread_bias")
            non_ot = results[results.get("is_ot", 0) != 1] if "is_ot" in results.columns else results
            summary["ot_games"] = int(len(results) - len(non_ot))
            summary["mean_total_bias_ex_ot"] = self._safe_group_mean(non_ot, "total_bias")
            summary["mean_total_bias_vegas_ex_ot"] = self._safe_group_mean(non_ot, "total_bias_vegas")
            summary["total_rmse_ex_ot"] = self._safe_group_rmse(non_ot, "actual_total", "sim_total")
        SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self._write_html_report(results, summary)

    def _build_sim_team_game_from_raw_players(self) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        if not self.raw_results_dir.exists():
            return pd.DataFrame()
        for f in sorted(self.raw_results_dir.glob("*_players.csv")):
            try:
                gid = f.stem.replace("_players", "")
                df = pd.read_csv(f)
                df["game_id"] = str(gid)
                frames.append(df)
            except Exception:
                continue
        if not frames:
            return pd.DataFrame()

        sim_players = pd.concat(frames, ignore_index=True)
        sim_players["game_id"] = sim_players["game_id"].astype(str)
        sim_players["team"] = sim_players["team"].astype(str)
        for col in [
            "pass_attempts",
            "rushing_attempts",
            "passing_yards",
            "rushing_yards",
            "passing_touchdowns",
            "rushing_touchdowns",
            "fumble_lost",
        ]:
            if col not in sim_players.columns:
                sim_players[col] = 0.0
            sim_players[col] = pd.to_numeric(sim_players[col], errors="coerce").fillna(0.0)

        sim_team_game = (
            sim_players.groupby(["game_id", "simulation", "team"], as_index=False)[
                [
                    "pass_attempts",
                    "rushing_attempts",
                    "passing_yards",
                    "rushing_yards",
                    "passing_touchdowns",
                    "rushing_touchdowns",
                    "fumble_lost",
                ]
            ]
            .sum(numeric_only=True)
            .rename(
                columns={
                    "pass_attempts": "sim_pass_attempts",
                    "rushing_attempts": "sim_rushing_attempts",
                    "passing_yards": "sim_passing_yards",
                    "rushing_yards": "sim_rushing_yards",
                    "passing_touchdowns": "sim_passing_tds",
                    "rushing_touchdowns": "sim_rushing_tds",
                    "fumble_lost": "sim_fumbles",
                }
            )
        )
        sim_team_game["sim_plays"] = sim_team_game["sim_pass_attempts"] + sim_team_game["sim_rushing_attempts"]
        return sim_team_game

    def _build_team_offense_calibration(self) -> Dict[str, object]:
        sim_team_game = self._build_sim_team_game_from_raw_players()
        actual_team = self._get_actual_team_game_stats()
        if sim_team_game.empty or actual_team.empty:
            TEAM_OFFENSE_CALIBRATION_PATH.write_text("", encoding="utf-8")
            summary = {"rows": 0, "games": 0, "teams": 0, "metrics": {}}
            TEAM_OFFENSE_CALIBRATION_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            return summary

        sim_team_summary = (
            sim_team_game.groupby(["game_id", "team"], as_index=False)[
                [
                    "sim_pass_attempts",
                    "sim_rushing_attempts",
                    "sim_plays",
                    "sim_passing_yards",
                    "sim_rushing_yards",
                    "sim_passing_tds",
                    "sim_rushing_tds",
                    "sim_fumbles",
                ]
            ]
            .mean(numeric_only=True)
        )
        team_compare = sim_team_summary.merge(actual_team, on=["game_id", "team"], how="inner")
        if team_compare.empty:
            TEAM_OFFENSE_CALIBRATION_PATH.write_text("", encoding="utf-8")
            summary = {"rows": 0, "games": 0, "teams": 0, "metrics": {}}
            TEAM_OFFENSE_CALIBRATION_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            return summary

        metric_pairs = {
            "pass_attempts": ("sim_pass_attempts", "actual_pass_attempts"),
            "rush_attempts": ("sim_rushing_attempts", "actual_rushing_attempts"),
            "passing_yards": ("sim_passing_yards", "actual_passing_yards"),
            "rushing_yards": ("sim_rushing_yards", "actual_rushing_yards"),
            "passing_tds": ("sim_passing_tds", "actual_passing_tds"),
            "rushing_tds": ("sim_rushing_tds", "actual_rushing_tds"),
            "plays": ("sim_plays", "actual_plays"),
        }

        metrics_summary: Dict[str, object] = {}
        for metric_name, (sim_col, act_col) in metric_pairs.items():
            if sim_col not in team_compare.columns or act_col not in team_compare.columns:
                continue
            sim_vals = pd.to_numeric(team_compare[sim_col], errors="coerce")
            act_vals = pd.to_numeric(team_compare[act_col], errors="coerce")
            mask = np.isfinite(sim_vals.to_numpy(dtype=float)) & np.isfinite(act_vals.to_numpy(dtype=float))
            if not mask.any():
                continue
            sim_arr = sim_vals.to_numpy(dtype=float)[mask]
            act_arr = act_vals.to_numpy(dtype=float)[mask]
            diff = sim_arr - act_arr
            metrics_summary[metric_name] = {
                "rows": int(mask.sum()),
                "mean_sim": float(np.mean(sim_arr)),
                "mean_actual": float(np.mean(act_arr)),
                "mean_bias": float(np.mean(diff)),
                "mae": float(np.mean(np.abs(diff))),
                "rmse": float(np.sqrt(np.mean(diff ** 2))),
            }
            team_compare[f"{metric_name}_delta"] = team_compare[sim_col] - team_compare[act_col]

        team_compare.to_csv(TEAM_OFFENSE_CALIBRATION_PATH, index=False)
        summary = {
            "rows": int(len(team_compare)),
            "games": int(team_compare["game_id"].nunique()) if "game_id" in team_compare.columns else 0,
            "teams": int(team_compare["team"].nunique()) if "team" in team_compare.columns else 0,
            "metrics": metrics_summary,
        }
        TEAM_OFFENSE_CALIBRATION_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def _report_player_names(self, players: pd.DataFrame) -> pd.DataFrame:
        from nflsim.engine.player_names import player_names

        players = players.copy()
        if getattr(self, "_report_name_lookup", None) is None:
            self._report_name_lookup = player_names(getattr(self, "player_stats", pd.DataFrame()), getattr(self, "rosters", pd.DataFrame()))
        names = self._report_name_lookup
        identities = players.player_id.astype(str)
        labels = players.get("player_name", identities).astype("string").str.strip()
        missing = labels.isna() | labels.eq(identities) | labels.str.lower().isin(["", "nan", "none", "<na>"])
        players["player_name"] = labels.mask(missing, identities.map(names)).fillna(identities)
        return players

    def _write_html_report(self, results: pd.DataFrame, summary: Dict[str, object]) -> None:
        self._report_name_lookup = None
        import base64, io
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        def fig_to_b64(fig) -> str:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
            plt.close(fig)
            buf.seek(0)
            return base64.b64encode(buf.read()).decode()

        def img_tag(b64: str) -> str:
            return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;margin:8px 0">'

        season_imgs: List[str] = []
        spread_mae_val: Optional[float] = None
        vegas_spread_mae_val: Optional[float] = None

        def add_identity_scatter(
            frame: pd.DataFrame,
            x_col: str,
            y_col: str,
            title: str,
            x_label: str,
            y_label: str,
            color: str,
        ) -> None:
            chart_frame = frame[[x_col, y_col]].copy()
            chart_frame[x_col] = pd.to_numeric(chart_frame[x_col], errors="coerce")
            chart_frame[y_col] = pd.to_numeric(chart_frame[y_col], errors="coerce")
            chart_frame = chart_frame.replace([np.inf, -np.inf], np.nan).dropna()
            if chart_frame.empty:
                return

            lim_min = float(min(chart_frame[x_col].min(), chart_frame[y_col].min()))
            lim_max = float(max(chart_frame[x_col].max(), chart_frame[y_col].max()))
            if not np.isfinite(lim_min) or not np.isfinite(lim_max):
                return
            if lim_min == lim_max:
                lim_min -= 1.0
                lim_max += 1.0

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(chart_frame[x_col], chart_frame[y_col], alpha=0.6, s=24, color=color)
            ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", linewidth=1)
            ax.set_xlim(lim_min, lim_max)
            ax.set_ylim(lim_min, lim_max)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(title)
            season_imgs.append(img_tag(fig_to_b64(fig)))

        if not results.empty:
            r = results.copy()
            r["actual_home_win"] = (r["actual_spread"] > 0).astype(float)
            r["sim_brier_score"] = (r["sim_home_win_prob"] - r["actual_home_win"]) ** 2
            r["home_odds"] = r["home_moneyline"].apply(implied_prob_from_moneyline)
            r["away_odds"] = r["away_moneyline"].apply(implied_prob_from_moneyline)
            normalizer = r["home_odds"] + r["away_odds"]
            r["vegas_home_prob"] = np.divide(r["home_odds"], normalizer, out=np.full(len(r), np.nan), where=normalizer > 0)
            r["vegas_brier_score"] = (r["vegas_home_prob"] - r["actual_home_win"]) ** 2

            # Sim spread vs actual
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(r["actual_spread"], r["sim_spread"], alpha=0.55, s=22, color="#2563eb")
            lim = max(abs(r["actual_spread"].max()), abs(r["actual_spread"].min()), 30)
            ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=1)
            ax.set_xlabel("Actual Spread (home - away)")
            ax.set_ylabel("Simulated Spread (median)")
            ax.set_title("Simulated vs Actual Spread")
            season_imgs.append(img_tag(fig_to_b64(fig)))

            # Sim spread vs Vegas spread
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(r["spread_line"], r["sim_spread"], alpha=0.55, s=22, color="#16a34a")
            lim = max(abs(r["spread_line"].dropna()).max(), 30)
            ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=1)
            ax.set_xlabel("Vegas Spread Line")
            ax.set_ylabel("Simulated Spread (median)")
            ax.set_title("Simulated Spread vs Vegas Line")
            season_imgs.append(img_tag(fig_to_b64(fig)))

            # RMSE comparison bar
            fig, ax = plt.subplots(figsize=(5, 4))
            spread_rmse = float(np.sqrt(((r["sim_spread"] - r["actual_spread"]) ** 2).mean()))
            total_rmse = float(np.sqrt(((r["sim_total"] - r["actual_total"]) ** 2).mean()))
            vegas_spread_rmse = float(np.sqrt(((r["spread_line"] - r["actual_spread"]) ** 2).dropna().mean()))
            vegas_total_rmse = float(np.sqrt(((r["total_line"] - r["actual_total"]) ** 2).dropna().mean()))
            labels = ["Spread\n(sim)", "Spread\n(Vegas)", "Total\n(sim)", "Total\n(Vegas)"]
            values = [spread_rmse, vegas_spread_rmse, total_rmse, vegas_total_rmse]
            colors = ["#2563eb", "#94a3b8", "#16a34a", "#94a3b8"]
            ax.bar(labels, values, color=colors)
            ax.set_ylabel("RMSE (points)")
            ax.set_title("RMSE Comparison")
            for i, v in enumerate(values):
                ax.text(i, v + 0.2, f"{v:.2f}", ha="center", fontsize=9)
            season_imgs.append(img_tag(fig_to_b64(fig)))

            # Brier score comparison
            fig, ax = plt.subplots(figsize=(4, 4))
            brier_sim = float(r["sim_brier_score"].mean())
            brier_vegas = float(r["vegas_brier_score"].dropna().mean()) if r["vegas_brier_score"].notna().any() else float("nan")
            b_labels = ["Model", "Vegas"]
            b_values = [brier_sim, brier_vegas]
            b_colors = ["#2563eb", "#94a3b8"]
            ax.bar([l for l, v in zip(b_labels, b_values) if np.isfinite(v)],
                   [v for v in b_values if np.isfinite(v)], color=[c for c, v in zip(b_colors, b_values) if np.isfinite(v)])
            ax.set_ylabel("Brier Score (lower = better)")
            ax.set_title("Brier Score Comparison")
            for i, (l, v) in enumerate(zip(b_labels, b_values)):
                if np.isfinite(v):
                    ax.text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=9)
            season_imgs.append(img_tag(fig_to_b64(fig)))

            # Equal-count probability buckets make the model and Vegas curves comparable.
            calib = r[["sim_home_win_prob", "vegas_home_prob", "actual_home_win"]].copy()
            for column in calib.columns:
                calib[column] = pd.to_numeric(calib[column], errors="coerce")
            calib = calib.replace([np.inf, -np.inf], np.nan).dropna()
            if not calib.empty:
                calibration_curves: Dict[str, pd.DataFrame] = {}
                for label, probability_col in [("Model", "sim_home_win_prob"), ("Vegas", "vegas_home_prob")]:
                    curve = calib[[probability_col, "actual_home_win"]].copy()
                    try:
                        curve["bucket"] = pd.qcut(curve[probability_col], q=10, duplicates="drop")
                    except ValueError:
                        continue
                    calibration_curves[label] = curve.groupby("bucket", observed=True).agg(
                        predicted=(probability_col, "mean"),
                        observed=("actual_home_win", "mean"),
                        games=("actual_home_win", "size"),
                    ).reset_index(drop=True)

                if calibration_curves:
                    fig, ax = plt.subplots(figsize=(6, 5))
                    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
                    curve_styles = {"Model": ("#2563eb", "o"), "Vegas": ("#64748b", "s")}
                    for label, grouped in calibration_curves.items():
                        if grouped.empty:
                            continue
                        color, marker = curve_styles[label]
                        ax.plot(
                            grouped["predicted"],
                            grouped["observed"],
                            marker=marker,
                            color=color,
                            label=label,
                        )
                        for _, row in grouped.iterrows():
                            ax.annotate(
                                f"n={int(row['games'])}",
                                (row["predicted"], row["observed"]),
                                textcoords="offset points",
                                xytext=(0, 8 if label == "Model" else -12),
                                ha="center",
                                fontsize=7,
                                color=color,
                            )
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.set_xlabel("Predicted Home Win Probability (bucket mean)")
                    ax.set_ylabel("Observed Home Win Rate")
                    ax.set_title("Win Probability Calibration (Equal-Count Buckets)")
                    ax.legend(fontsize=8)
                    season_imgs.append(img_tag(fig_to_b64(fig)))

            # Spread MAE: overall (model vs Vegas) and by actual-spread magnitude bucket
            spread_frame = r[["actual_spread", "sim_spread", "spread_line"]].copy()
            for column in spread_frame.columns:
                spread_frame[column] = pd.to_numeric(spread_frame[column], errors="coerce")
            spread_frame = spread_frame.replace([np.inf, -np.inf], np.nan)
            model_err = (spread_frame["sim_spread"] - spread_frame["actual_spread"]).abs().dropna()
            vegas_err = (spread_frame["spread_line"] - spread_frame["actual_spread"]).abs().dropna()
            if not model_err.empty:
                spread_mae_val = float(model_err.mean())
                vegas_spread_mae_val = float(vegas_err.mean()) if not vegas_err.empty else None

                fig, ax = plt.subplots(figsize=(5, 4))
                m_labels = ["Model", "Vegas"]
                m_values = [spread_mae_val, vegas_spread_mae_val if vegas_spread_mae_val is not None else float("nan")]
                keep = [np.isfinite(v) for v in m_values]
                ax.bar(
                    [l for l, k in zip(m_labels, keep) if k],
                    [v for v, k in zip(m_values, keep) if k],
                    color=[c for c, k in zip(["#2563eb", "#94a3b8"], keep) if k],
                )
                ax.set_ylabel("Spread MAE (points)")
                ax.set_title("Spread MAE vs Actual")
                idx = 0
                for v, k in zip(m_values, keep):
                    if k:
                        ax.text(idx, v + 0.15, f"{v:.2f}", ha="center", fontsize=9)
                        idx += 1
                season_imgs.append(img_tag(fig_to_b64(fig)))

                bucket_frame = spread_frame.dropna(subset=["actual_spread", "sim_spread"]).copy()
                bucket_frame["abs_error"] = (bucket_frame["sim_spread"] - bucket_frame["actual_spread"]).abs()
                bucket_frame["bucket"] = pd.cut(
                    bucket_frame["actual_spread"],
                    bins=[-np.inf, -21, -14, -7, -3, 0, 3, 7, 14, 21, np.inf],
                    labels=["<-21", "-21..-14", "-14..-7", "-7..-3", "-3..0", "0..3", "3..7", "7..14", "14..21", ">21"],
                )
                bucket_stats = bucket_frame.groupby("bucket", observed=True).agg(
                    mae=("abs_error", "mean"),
                    games=("abs_error", "size"),
                ).reset_index()
                bucket_stats = bucket_stats[bucket_stats["games"] > 0]
                if not bucket_stats.empty:
                    fig, ax = plt.subplots(figsize=(8, 4.5))
                    ax.bar(bucket_stats["bucket"].astype(str), bucket_stats["mae"], color="#0ea5e9")
                    ax.set_xlabel("Actual Spread (home - away)")
                    ax.set_ylabel("Spread MAE (points)")
                    ax.set_title("Spread MAE by Actual Spread Bucket")
                    ax.tick_params(axis="x", labelrotation=45)
                    for i, row in enumerate(bucket_stats.itertuples()):
                        ax.text(i, row.mae + 0.2, f"{row.mae:.1f}\nn={int(row.games)}", ha="center", fontsize=8)
                    season_imgs.append(img_tag(fig_to_b64(fig)))

            all_player_frames: List[pd.DataFrame] = []
            if self.raw_results_dir.exists():
                for f in sorted(self.raw_results_dir.glob("*_players.csv")):
                    try:
                        gid = f.stem.replace("_players", "")
                        df = pd.read_csv(f)
                        df["game_id"] = gid
                        all_player_frames.append(df)
                    except Exception:
                        pass

            if all_player_frames:
                sim_players = pd.concat(all_player_frames, ignore_index=True)
                sim_players["game_id"] = sim_players["game_id"].astype(str)
                sim_players["team"] = sim_players["team"].astype(str)
                for column in [
                    "pass_attempts",
                    "rushing_attempts",
                    "passing_yards",
                    "rushing_yards",
                    "passing_touchdowns",
                    "rushing_touchdowns",
                    "fumble_lost",
                ]:
                    if column not in sim_players.columns:
                        sim_players[column] = 0.0
                    sim_players[column] = pd.to_numeric(sim_players[column], errors="coerce").fillna(0.0)

                sim_team_game = (
                    sim_players.groupby(["game_id", "simulation", "team"], as_index=False)[
                        [
                            "pass_attempts",
                            "rushing_attempts",
                            "passing_yards",
                            "rushing_yards",
                            "passing_touchdowns",
                            "rushing_touchdowns",
                            "fumble_lost",
                        ]
                    ]
                    .sum(numeric_only=True)
                    .rename(
                        columns={
                            "pass_attempts": "sim_pass_attempts",
                            "rushing_attempts": "sim_rushing_attempts",
                            "passing_yards": "sim_passing_yards",
                            "rushing_yards": "sim_rushing_yards",
                            "passing_touchdowns": "sim_passing_tds",
                            "rushing_touchdowns": "sim_rushing_tds",
                            "fumble_lost": "sim_fumbles",
                        }
                    )
                )
                sim_team_game["sim_plays"] = sim_team_game["sim_pass_attempts"] + sim_team_game["sim_rushing_attempts"]
                sim_team_summary = (
                    sim_team_game.groupby(["game_id", "team"], as_index=False)[
                        [
                            "sim_pass_attempts",
                            "sim_rushing_attempts",
                            "sim_plays",
                            "sim_passing_yards",
                            "sim_rushing_yards",
                            "sim_passing_tds",
                            "sim_rushing_tds",
                        ]
                    ]
                    .mean(numeric_only=True)
                )
                actual_team_summary = self._get_actual_team_game_stats()
                if not actual_team_summary.empty:
                    team_compare = sim_team_summary.merge(actual_team_summary, on=["game_id", "team"], how="inner")
                    add_identity_scatter(
                        frame=team_compare,
                        x_col="actual_rushing_attempts",
                        y_col="sim_rushing_attempts",
                        title="Simulated vs Actual Team Rushing Attempts",
                        x_label="Actual Team Rushing Attempts",
                        y_label="Simulated Team Rushing Attempts (mean)",
                        color="#7c3aed",
                    )

                    add_identity_scatter(
                        frame=team_compare,
                        x_col="actual_passing_yards",
                        y_col="sim_passing_yards",
                        title="Simulated vs Actual Team Passing Yards",
                        x_label="Actual Team Passing Yards",
                        y_label="Simulated Team Passing Yards (mean)",
                        color="#0369a1",
                    )

                    add_identity_scatter(
                        frame=team_compare,
                        x_col="actual_rushing_yards",
                        y_col="sim_rushing_yards",
                        title="Simulated vs Actual Team Rushing Yards",
                        x_label="Actual Team Rushing Yards",
                        y_label="Simulated Team Rushing Yards (mean)",
                        color="#16a34a",
                    )

                    add_identity_scatter(
                        frame=team_compare,
                        x_col="actual_passing_tds",
                        y_col="sim_passing_tds",
                        title="Simulated vs Actual Team Passing TDs",
                        x_label="Actual Team Passing TDs",
                        y_label="Simulated Team Passing TDs (mean)",
                        color="#9333ea",
                    )

                    add_identity_scatter(
                        frame=team_compare,
                        x_col="actual_rushing_tds",
                        y_col="sim_rushing_tds",
                        title="Simulated vs Actual Team Rushing TDs",
                        x_label="Actual Team Rushing TDs",
                        y_label="Simulated Team Rushing TDs (mean)",
                        color="#b45309",
                    )

                    game_play_compare = (
                        team_compare.groupby("game_id", as_index=False)[["sim_plays", "actual_plays"]]
                        .sum(numeric_only=True)
                    )
                    add_identity_scatter(
                        frame=game_play_compare,
                        x_col="actual_plays",
                        y_col="sim_plays",
                        title="Simulated vs Actual Plays per Game",
                        x_label="Actual Offensive Plays per Game",
                        y_label="Simulated Offensive Plays per Game (mean)",
                        color="#ea580c",
                    )

                sim_game_fumbles = (
                    sim_team_game.groupby(["game_id", "simulation"], as_index=False)["sim_fumbles"]
                    .sum(numeric_only=True)
                    .groupby("game_id", as_index=False)["sim_fumbles"]
                    .mean(numeric_only=True)
                )
                actual_game_fumbles = self._get_actual_game_fumble_stats(r["game_id"].astype(str).tolist())
                if not actual_game_fumbles.empty:
                    fumble_compare = sim_game_fumbles.merge(actual_game_fumbles, on="game_id", how="inner")
                    add_identity_scatter(
                        frame=fumble_compare,
                        x_col="actual_fumbles",
                        y_col="sim_fumbles",
                        title="Simulated vs Actual Fumbles per Game",
                        x_label="Actual Fumbles per Game",
                        y_label="Simulated Fumbles per Game (mean)",
                        color="#dc2626",
                    )

            actual_totals = pd.to_numeric(r.get("actual_total", pd.Series(dtype=float)), errors="coerce")
            sim_totals = pd.to_numeric(r.get("sim_total", pd.Series(dtype=float)), errors="coerce")
            vegas_totals = pd.to_numeric(r.get("total_line", pd.Series(dtype=float)), errors="coerce")
            total_series = [
                actual_totals[np.isfinite(actual_totals.to_numpy(dtype=float))],
                sim_totals[np.isfinite(sim_totals.to_numpy(dtype=float))],
                vegas_totals[np.isfinite(vegas_totals.to_numpy(dtype=float))],
            ]
            if any(len(series) > 0 for series in total_series):
                total_min = min(float(series.min()) for series in total_series if len(series) > 0)
                total_max = max(float(series.max()) for series in total_series if len(series) > 0)
                if not np.isfinite(total_min) or not np.isfinite(total_max):
                    total_min, total_max = 0.0, 80.0
                if total_min == total_max:
                    total_min -= 2.0
                    total_max += 2.0
                bins = np.linspace(total_min, total_max, 18)

                fig, ax = plt.subplots(figsize=(10, 5))
                hist_data = [
                    actual_totals[np.isfinite(actual_totals.to_numpy(dtype=float))].to_numpy(dtype=float),
                    sim_totals[np.isfinite(sim_totals.to_numpy(dtype=float))].to_numpy(dtype=float),
                    vegas_totals[np.isfinite(vegas_totals.to_numpy(dtype=float))].to_numpy(dtype=float),
                ]
                hist_labels = ["Actual Totals", "Sim Totals (median)", "Vegas Total Line"]
                hist_colors = ["#0f766e", "#0891b2", "#7c3aed"]
                ax.hist(
                    hist_data,
                    bins=bins,
                    stacked=True,
                    label=hist_labels,
                    color=hist_colors,
                    alpha=0.78,
                    edgecolor="white",
                )
                for values, color in zip(hist_data, hist_colors):
                    if values.size > 0:
                        ax.axvline(float(np.mean(values)), color=color, linestyle="--", linewidth=1.2)
                ax.set_title("Total Points Distribution Comparison (Stacked)")
                ax.set_xlabel("Points")
                ax.set_ylabel("Games")
                ax.legend(fontsize=9)
                season_imgs.append(img_tag(fig_to_b64(fig)))

        # Season-long player CDF plots (percentile histograms)
        player_stat_cols_to_plot = [
            ("passing_yards", "QB"),
            ("pass_attempts", "QB"),
            ("completion_pct", "QB"),
            ("rushing_yards", None),
            ("receiving_yards", None),
            ("passing_touchdowns", "QB"),
            ("receptions", None),
            ("interceptions", "QB"),
        ]
        all_player_frames = []
        if self.raw_results_dir.exists():
            for f in sorted(self.raw_results_dir.glob("*_players.csv")):
                try:
                    gid = f.stem.replace("_players", "")
                    df = pd.read_csv(f)
                    df["game_id"] = gid
                    all_player_frames.append(df)
                except Exception:
                    pass

        player_cdf_imgs: List[str] = []
        if all_player_frames and not results.empty:
            all_players = pd.concat(all_player_frames, ignore_index=True)
            all_players["player_id"] = all_players["player_id"].astype(str)

            actual_stats = self._get_actual_player_game_stats()
            if not actual_stats.empty:
                game_meta = results[["game_id", "week", "home_team", "away_team"]].drop_duplicates().copy()
                home_map = game_meta[["game_id", "week", "home_team"]].rename(columns={"home_team": "team"})
                away_map = game_meta[["game_id", "week", "away_team"]].rename(columns={"away_team": "team"})
                game_team_map = pd.concat([home_map, away_map], ignore_index=True).drop_duplicates()
                if "game_id" in actual_stats.columns:
                    actual_with_game = actual_stats.copy()
                else:
                    actual_with_game = actual_stats.merge(
                        game_team_map,
                        left_on=["week", "recent_team"],
                        right_on=["week", "team"],
                        how="inner",
                    )

                actual_col_map = {
                    "passing_yards": "passing_yards",
                    "pass_attempts": "attempts",
                    "rushing_yards": "rushing_yards",
                    "receiving_yards": "receiving_yards",
                    "passing_touchdowns": "passing_tds",
                    "receptions": "receptions",
                    "interceptions": "interceptions",
                }

                # For each stat, histogram the percentile of the actual outcome in each player-game simulation distribution.
                for stat_col, pos_filter in player_stat_cols_to_plot:
                    sub = all_players.copy()
                    if pos_filter:
                        sub = sub[sub["position"].astype(str).str.startswith(pos_filter)]
                    sub = sub[sub["player_id"].notna()].copy()
                    if sub.empty:
                        continue

                    percentiles: List[float] = []
                    if stat_col == "completion_pct":
                        if not {"completions", "pass_attempts"}.issubset(sub.columns):
                            continue
                        if not {"completions", "attempts"}.issubset(actual_with_game.columns):
                            continue

                        actual_sub = actual_with_game[["game_id", "player_id", "completions", "attempts"]].copy()
                        actual_sub["player_id"] = actual_sub["player_id"].astype(str)
                        actual_sub["completion_pct"] = np.divide(
                            pd.to_numeric(actual_sub["completions"], errors="coerce"),
                            pd.to_numeric(actual_sub["attempts"], errors="coerce"),
                            out=np.full(len(actual_sub), np.nan),
                            where=pd.to_numeric(actual_sub["attempts"], errors="coerce") > 0,
                        )
                        actual_sub = actual_sub.dropna(subset=["completion_pct"])
                        if actual_sub.empty:
                            continue
                        actual_sub = (
                            actual_sub.groupby(["game_id", "player_id"], as_index=False)["completion_pct"]
                            .mean(numeric_only=True)
                        )
                        actual_lookup = actual_sub.set_index(["game_id", "player_id"])["completion_pct"]

                        for (gid, pid), group in sub.groupby(["game_id", "player_id"], as_index=False):
                            key = (str(gid), str(pid))
                            if key not in actual_lookup.index:
                                continue
                            actual_value = actual_lookup.loc[key]
                            if pd.isna(actual_value):
                                continue
                            sim_values = np.divide(
                                pd.to_numeric(group["completions"], errors="coerce").to_numpy(dtype=float),
                                pd.to_numeric(group["pass_attempts"], errors="coerce").to_numpy(dtype=float),
                                out=np.full(len(group), np.nan),
                                where=pd.to_numeric(group["pass_attempts"], errors="coerce").to_numpy(dtype=float) > 0,
                            )
                            sim_values = sim_values[np.isfinite(sim_values)]
                            if sim_values.size == 0:
                                continue
                            percentiles.append(float(np.mean(sim_values <= float(actual_value))))
                    else:
                        if stat_col not in sub.columns:
                            continue
                        actual_col = actual_col_map.get(stat_col)
                        if actual_col is None or actual_col not in actual_with_game.columns:
                            continue

                        actual_sub = actual_with_game[["game_id", "player_id", actual_col]].copy()
                        actual_sub["player_id"] = actual_sub["player_id"].astype(str)
                        actual_sub = actual_sub.dropna(subset=[actual_col])
                        if actual_sub.empty:
                            continue
                        actual_sub = (
                            actual_sub.groupby(["game_id", "player_id"], as_index=False)[actual_col]
                            .mean(numeric_only=True)
                        )
                        actual_lookup = actual_sub.set_index(["game_id", "player_id"])[actual_col]

                        for (gid, pid), group in sub.groupby(["game_id", "player_id"])[stat_col]:
                            key = (str(gid), str(pid))
                            if key not in actual_lookup.index:
                                continue
                            actual_value = actual_lookup.loc[key]
                            if pd.isna(actual_value):
                                continue
                            sim_values = pd.to_numeric(group, errors="coerce").dropna().to_numpy(dtype=float)
                            if sim_values.size == 0:
                                continue
                            percentiles.append(float(np.mean(sim_values <= float(actual_value))))

                    if len(percentiles) < 5:
                        continue

                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.hist(
                        percentiles,
                        bins=np.linspace(0.0, 1.0, 21),
                        density=True,
                        color="#2563eb",
                        alpha=0.75,
                        edgecolor="white",
                    )
                    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="Uniform baseline")
                    ax.set_xlim(0.0, 1.0)
                    ax.set_xlabel("Percentile of actual outcome in simulated distribution")
                    ax.set_ylabel("Density")
                    title_pos = f" ({pos_filter})" if pos_filter else ""
                    title_label = "Completion Pct" if stat_col == "completion_pct" else stat_col.replace('_', ' ').title()
                    ax.set_title(f"{title_label}{title_pos} — Simulation Coverage")
                    ax.legend(fontsize=8)
                    player_cdf_imgs.append(img_tag(fig_to_b64(fig)))

        # Per-game sections
        game_sections_html = ""
        if not results.empty:
            penalty_lookup: Dict[tuple[str, str], float] = {}
            penalty_stats = self._get_actual_team_penalty_stats(results["game_id"].astype(str).tolist())
            if not penalty_stats.empty and {"game_id", "team", "actual_penalty_yards"}.issubset(penalty_stats.columns):
                for row in penalty_stats.itertuples(index=False):
                    penalty_lookup[(str(row.game_id), str(row.team))] = float(row.actual_penalty_yards)

            actual_player_lookup = pd.DataFrame()
            actual_player_game = self._get_actual_player_game_stats()
            if not actual_player_game.empty and {"game_id", "player_id"}.issubset(actual_player_game.columns):
                actual_stat_map = {
                    "passing_yards": "Actual PassYds",
                    "attempts": "Actual Att",
                    "completions": "Actual Cmp",
                    "passing_tds": "Actual PassTD",
                    "interceptions": "Actual INT",
                    "rushing_yards": "Actual RushYds",
                    "rushing_attempts": "Actual RushAtt",
                    "receiving_yards": "Actual RecYds",
                    "receptions": "Actual Rec",
                }
                available_actual_cols = [c for c in actual_stat_map if c in actual_player_game.columns]
                if available_actual_cols:
                    actual_player_lookup = actual_player_game[["game_id", "player_id", *available_actual_cols]].copy()
                    actual_player_lookup["game_id"] = actual_player_lookup["game_id"].astype(str)
                    actual_player_lookup["player_id"] = actual_player_lookup["player_id"].astype(str)
                    for col in available_actual_cols:
                        actual_player_lookup[col] = pd.to_numeric(actual_player_lookup[col], errors="coerce")
                    actual_player_lookup = (
                        actual_player_lookup.groupby(["game_id", "player_id"], as_index=False)[available_actual_cols]
                        .mean(numeric_only=True)
                        .rename(columns=actual_stat_map)
                    )

            def _fmt_penalty(value: Optional[float]) -> str:
                if value is None or not np.isfinite(value):
                    return "n/a"
                return f"{value:.0f}"

            for _, row in results.sort_values(["week", "game_id"]).iterrows():
                gid = str(row["game_id"])
                raw_path = self.raw_results_dir / f"{gid}.csv"
                players_path = self.raw_results_dir / f"{gid}_players.csv"
                away_team = str(row["away_team"])
                home_team = str(row["home_team"])

                away_penalty = penalty_lookup.get((gid, away_team))
                home_penalty = penalty_lookup.get((gid, home_team))

                score_table = (
                    f"<b>{away_team}</b> @ <b>{home_team}</b> &nbsp;|&nbsp; "
                    f"Actual: {int(row['actual_away_score'])}–{int(row['actual_home_score'])} &nbsp;|&nbsp; "
                    f"Sim median: {row['sim_away_median']:.0f}–{row['sim_home_median']:.0f} &nbsp;|&nbsp; "
                    f"Spread: actual {row['actual_spread']:+.1f} / sim {row['sim_spread']:+.1f} / Vegas {row['spread_line']:+.1f} &nbsp;|&nbsp; "
                    f"Total: actual {row['actual_total']:.0f} / sim {row['sim_total']:.0f} / Vegas {row['total_line']:.0f} &nbsp;|&nbsp; "
                    f"Penalty yds called: {away_team} {_fmt_penalty(away_penalty)}, {home_team} {_fmt_penalty(home_penalty)} &nbsp;|&nbsp; "
                    f"Win prob (home): {row['sim_home_win_prob']:.1%}"
                )

                game_chart_html = ""
                if raw_path.exists():
                    try:
                        sim_df = pd.read_csv(raw_path)
                        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                        for ax, col, label in [
                            (axes[0], "home", f"{row['home_team']} score"),
                            (axes[1], "away", f"{row['away_team']} score"),
                        ]:
                            if col in sim_df.columns:
                                ax.hist(sim_df[col], bins=20, color="#2563eb", alpha=0.75, edgecolor="white", density=True)
                                ax.set_title(label)
                                ax.set_xlabel("Score")
                        fig.suptitle(f"{row['away_team']} @ {row['home_team']}")
                        game_chart_html = img_tag(fig_to_b64(fig))
                    except Exception:
                        pass

                player_table_html = ""
                if players_path.exists():
                    try:
                        pf = self._report_player_names(pd.read_csv(players_path))
                        stat_specs = [
                            ("passing_yards", "PassYds"),
                            ("pass_attempts", "Att"),
                            ("completions", "Cmp"),
                            ("passing_touchdowns", "PassTD"),
                            ("interceptions", "INT"),
                            ("rushing_yards", "RushYds"),
                            ("rushing_attempts", "RushAtt"),
                            ("rushing_touchdowns", "RushTD"),
                            ("receiving_yards", "RecYds"),
                            ("receptions", "Rec"),
                            ("receiving_touchdowns", "RecTD"),
                            ("air_yards", "AirYds"),
                        ]
                        stat_cols_show = [column for column, _ in stat_specs if column in pf.columns]
                        if stat_cols_show:
                            pf = pf.copy()
                            pf["game_id"] = gid
                            pf["player_id"] = pf["player_id"].astype(str)
                            for column in stat_cols_show:
                                pf[column] = pd.to_numeric(pf[column], errors="coerce").fillna(0.0)

                            p10 = lambda q: np.percentile(q, 10)
                            p90 = lambda q: np.percentile(q, 90)
                            p10.__name__ = "p10"
                            p90.__name__ = "p90"

                            grouped = (
                                pf.groupby(["game_id", "team", "player_name", "position", "player_id"], as_index=False)[stat_cols_show]
                                .agg(["mean", "median", p10, p90])
                            )
                            grouped.columns = ["_".join(str(part) for part in col if part) for col in grouped.columns.to_flat_index()]
                            grouped.columns = [column[:-1] if column.endswith("_") else column for column in grouped.columns]
                            grouped = grouped.reset_index()

                            if not actual_player_lookup.empty:
                                grouped = grouped.merge(
                                    actual_player_lookup,
                                    left_on=["game_id", "player_id"],
                                    right_on=["game_id", "player_id"],
                                    how="left",
                                )

                            rename_map = {
                                "team": "Team",
                                "team_": "Team",
                                "player_name": "Player",
                                "player_name_": "Player",
                                "position": "Pos",
                                "position_": "Pos",
                            }
                            for raw_col, label in stat_specs:
                                if raw_col not in stat_cols_show:
                                    continue
                                rename_map[f"{raw_col}_mean"] = f"{label} Avg"
                                rename_map[f"{raw_col}_median"] = f"{label} Med"
                                rename_map[f"{raw_col}_p10"] = f"{label} P10"
                                rename_map[f"{raw_col}_p90"] = f"{label} P90"
                            grouped = grouped.rename(columns=rename_map)

                            actual_cols = [
                                "Actual PassYds",
                                "Actual Att",
                                "Actual Cmp",
                                "Actual PassTD",
                                "Actual INT",
                                "Actual RushYds",
                                "Actual RushAtt",
                                "Actual RecYds",
                                "Actual Rec",
                            ]
                            actual_cols = [c for c in actual_cols if c in grouped.columns]

                            base_cols = ["Team", "Player", "Pos"]
                            if not set(base_cols).issubset(grouped.columns):
                                continue
                            sim_cols = [c for c in grouped.columns if c not in {"game_id", "player_id", *base_cols, *actual_cols}]
                            grouped = grouped[[*base_cols, *actual_cols, *sim_cols]]

                            position_rank = {"QB": 0, "RB": 1, "WR": 2, "TE": 3}
                            if "Pos" in grouped.columns:
                                grouped["_pos_root"] = grouped["Pos"].astype(str).str.extract(r"([A-Za-z]+)", expand=False).fillna("ZZ")
                                grouped["_pos_rank"] = grouped["_pos_root"].map(position_rank).fillna(99)
                                grouped = grouped.sort_values(["Team", "_pos_rank", "Pos", "Player"]).drop(columns=["_pos_root", "_pos_rank"])

                            for column in grouped.columns:
                                if column in {"Team", "Player", "Pos"}:
                                    continue
                                grouped[column] = pd.to_numeric(grouped[column], errors="coerce").round(1)

                            table_html = grouped.to_html(index=False, border=0, classes="ptable")
                            player_table_html = f'<div class="ptable-wrap">{table_html}</div>'
                    except Exception as error:
                        player_table_html = (
                            f'<div class="ptable-wrap"><div style="padding:10px;color:#7f1d1d;">'
                            f'Player stats unavailable: {error}'
                            f"</div></div>"
                        )

                safe_gid = gid.replace("-", "_").replace(".", "_")
                search_blob = f"{row['away_team']} {row['home_team']}".lower()
                game_sections_html += f"""
<div class="game-row" data-search="{search_blob}" onclick="toggleGame('{safe_gid}')">
  <span class="toggle-arrow" id="arrow_{safe_gid}">&#9654;</span> Week {int(row['week'])}: {score_table}
</div>
<div class="game-detail" id="detail_{safe_gid}" data-search="{search_blob}">
  {game_chart_html}
  {player_table_html}
</div>
"""

        brier_val = summary.get("brier")
        vegas_brier_val = summary.get("vegas_brier")
        spread_rmse_val = summary.get("spread_rmse")
        total_rmse_val = summary.get("total_rmse")
        games_scored = int(summary.get("games_scored", 0))
        missing_games = int(summary.get("missing_games", 0))
        schema_issue_count = int(summary.get("schema_issue_count", 0))
        weekly_rows = summary.get("by_week", []) or []
        team_rows = summary.get("by_team", []) or []

        weekly_table_html = ""
        if weekly_rows:
            weekly_frame = pd.DataFrame(weekly_rows)
            weekly_table_html = weekly_frame.to_html(index=False, border=0, classes="slice-table", float_format=lambda x: f"{x:.3f}")

        team_table_html = ""
        if team_rows:
            team_frame = pd.DataFrame(team_rows).sort_values(["brier", "team"], ascending=[False, True])
            team_table_html = team_frame.to_html(index=False, border=0, classes="slice-table", float_format=lambda x: f"{x:.3f}")

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Backtest Report {self.season}</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 0; padding: 20px; background: #f8fafc; color: #1e293b; }}
  h1 {{ margin-bottom: 4px; }}
  .summary-grid {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 16px 0 24px; }}
  .kpi {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px 20px; min-width: 140px; }}
  .kpi .label {{ font-size: 12px; color: #64748b; text-transform: uppercase; letter-spacing: .05em; }}
  .kpi .value {{ font-size: 22px; font-weight: 700; margin-top: 4px; }}
  .charts {{ display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 28px; }}
  .charts img {{ border-radius: 6px; border: 1px solid #e2e8f0; }}
  h2 {{ margin-top: 28px; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; }}
    .search-wrap {{ margin: 10px 0 14px; }}
    .search-input {{
        width: min(460px, 100%);
        padding: 9px 12px;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        font-size: 14px;
        background: #fff;
        color: #0f172a;
    }}
    .search-input:focus {{ outline: 2px solid #93c5fd; outline-offset: 1px; }}
  .game-row {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 10px 14px;
               margin: 6px 0; cursor: pointer; font-size: 13px; user-select: none; }}
  .game-row:hover {{ background: #f1f5f9; }}
  .toggle-arrow {{ display: inline-block; margin-right: 8px; transition: transform .15s; }}
  .game-detail {{ display: none; background: #fff; border: 1px solid #e2e8f0; border-top: none;
                  border-radius: 0 0 6px 6px; padding: 12px 16px; margin-bottom: 6px; }}
  .game-detail.open {{ display: block; }}
    .ptable-wrap {{
        max-height: 540px;
        overflow: auto;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        margin-top: 10px;
    }}
        .ptable {{ border-collapse: separate; border-spacing: 0; font-size: 12px; width: max-content; min-width: 100%; margin-top: 0; }}
    .ptable th, .ptable td {{ border: 1px solid #e2e8f0; padding: 4px 8px; text-align: right; white-space: nowrap; }}
        .ptable th {{ background: #f1f5f9; text-align: center; position: sticky; top: 0; z-index: 4; }}
    .ptable td:nth-child(1), .ptable td:nth-child(2), .ptable td:nth-child(3) {{ text-align: left; }}

        /* Fixed widths keep sticky offsets stable across all rows */
        .ptable th:nth-child(1), .ptable td:nth-child(1) {{
            min-width: 88px;
            width: 88px;
            max-width: 88px;
            position: sticky;
            left: 0;
            z-index: 3;
            background: #fff;
            box-shadow: 1px 0 0 #e2e8f0;
        }}
        .ptable th:nth-child(2), .ptable td:nth-child(2) {{
            min-width: 196px;
            width: 196px;
            max-width: 196px;
            position: sticky;
            left: 88px;
            z-index: 3;
            background: #fff;
            box-shadow: 1px 0 0 #e2e8f0;
        }}
        .ptable th:nth-child(3), .ptable td:nth-child(3) {{
            min-width: 74px;
            width: 74px;
            max-width: 74px;
            position: sticky;
            left: 284px;
            z-index: 3;
            background: #fff;
            box-shadow: 1px 0 0 #e2e8f0;
        }}
        .ptable th:nth-child(1), .ptable th:nth-child(2), .ptable th:nth-child(3) {{ background: #f1f5f9; z-index: 6; }}
    .slice-table {{ border-collapse: collapse; font-size: 12px; width: 100%; margin: 10px 0 18px; }}
    .slice-table th, .slice-table td {{ border: 1px solid #e2e8f0; padding: 4px 8px; text-align: right; }}
    .slice-table th {{ background: #f8fafc; text-align: center; }}
</style>
</head>
<body>
<h1>NFL {self.season} Simulation Backtest Report</h1>
<div class="summary-grid">
  <div class="kpi"><div class="label">Games Scored</div><div class="value">{games_scored}</div></div>
  <div class="kpi"><div class="label">Missing</div><div class="value">{missing_games}</div></div>
    <div class="kpi"><div class="label">Schema Issues</div><div class="value">{schema_issue_count}</div></div>
  <div class="kpi"><div class="label">Model Brier</div><div class="value">{f"{brier_val:.4f}" if brier_val is not None else "—"}</div></div>
  <div class="kpi"><div class="label">Vegas Brier</div><div class="value">{f"{vegas_brier_val:.4f}" if vegas_brier_val is not None else "—"}</div></div>
  <div class="kpi"><div class="label">Spread RMSE</div><div class="value">{f"{spread_rmse_val:.2f}" if spread_rmse_val is not None else "—"}</div></div>
  <div class="kpi"><div class="label">Spread MAE</div><div class="value">{f"{spread_mae_val:.2f}" if spread_mae_val is not None else "—"}</div></div>
  <div class="kpi"><div class="label">Vegas Spread MAE</div><div class="value">{f"{vegas_spread_mae_val:.2f}" if vegas_spread_mae_val is not None else "—"}</div></div>
  <div class="kpi"><div class="label">Total RMSE</div><div class="value">{f"{total_rmse_val:.2f}" if total_rmse_val is not None else "—"}</div></div>
</div>
<h2>Season-Level Charts</h2>
<div class="charts">{''.join(season_imgs)}</div>
<h2>Weekly Diagnostics</h2>
{weekly_table_html}
<h2>Team Diagnostics</h2>
{team_table_html}
<h2>Player Outcome Coverage (CDF Percentile Histograms)</h2>
<div class="charts">{''.join(player_cdf_imgs)}</div>
<h2>Game-by-Game Results</h2>
<div class="search-wrap">
    <input id="teamSearch" class="search-input" type="text" placeholder="Search by team (e.g., BUF, Bills, KC)" oninput="filterGames()" />
</div>
{game_sections_html}
<script>
function toggleGame(id) {{
  var detail = document.getElementById('detail_' + id);
  var arrow = document.getElementById('arrow_' + id);
  if (detail.classList.contains('open')) {{
    detail.classList.remove('open');
    arrow.style.transform = '';
  }} else {{
    detail.classList.add('open');
    arrow.style.transform = 'rotate(90deg)';
  }}
}}

function filterGames() {{
    var input = document.getElementById('teamSearch');
    var query = (input && input.value ? input.value : '').trim().toLowerCase();
    var rows = document.querySelectorAll('.game-row');

    rows.forEach(function(row) {{
        var detailId = row.getAttribute('id');
        var safeId = '';
        if (row.getAttribute('onclick')) {{
            var onClick = row.getAttribute('onclick');
            var match = onClick.match(/toggleGame\('([^']+)'\)/);
            safeId = match ? match[1] : '';
        }}
        var detail = safeId ? document.getElementById('detail_' + safeId) : null;
        var haystack = (row.getAttribute('data-search') || row.textContent || '').toLowerCase();
        var visible = query.length === 0 || haystack.indexOf(query) !== -1;

        row.style.display = visible ? '' : 'none';
        if (detail) {{
            detail.style.display = visible && detail.classList.contains('open') ? 'block' : 'none';
            if (!visible) {{
                detail.classList.remove('open');
                var arrow = row.querySelector('.toggle-arrow');
                if (arrow) {{
                    arrow.style.transform = '';
                }}
            }}
        }}
    }});
}}
</script>
</body>
</html>"""
        HTML_REPORT_PATH.write_text(html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest the torch game simulation pipeline on the 2025 regular season.")
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--workers", type=int, default=1, help="Worker processes for simulations (capped at 3).")
    parser.add_argument("--week-start", type=int)
    parser.add_argument("--week-end", type=int)
    parser.add_argument("--game-id", type=str, help="Only simulate the specified game_id from the schedule.")
    parser.add_argument("--max-games", type=int)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--reset-artifacts", action="store_true")
    parser.add_argument("--quiet-game-summaries", action="store_true")
    parser.add_argument("--strict-schema", action="store_true", help="Fail fast on runtime schema mismatches.")
    parser.add_argument(
        "--attempt-diagnostics",
        action="store_true",
        help="Write per-simulation pass/rush attempt diagnostics and accounting invariants.",
    )
    parser.add_argument(
        "--live-pbp",
        action="store_true",
        help="Print each play as it is simulated (implies --workers 1 --simulations 1). Requires --game-id.",
    )
    return parser.parse_args()


def main() -> None:
    import traceback
    try:
        args = parse_args()
        backtester = GameBacktester(
            season=args.season,
            simulations=args.simulations,
            workers=args.workers,
            week_start=args.week_start,
            week_end=args.week_end,
            resume=not args.no_resume,
            reset_artifacts=args.reset_artifacts,
            print_game_summaries=not args.quiet_game_summaries,
            strict_schema=args.strict_schema,
            attempt_diagnostics=args.attempt_diagnostics,
            game_id=args.game_id,
            live_pbp=args.live_pbp,
        )
        backtester.run(max_games=args.max_games)
    except Exception as e:
        print(f"\nERROR in main: {e}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()