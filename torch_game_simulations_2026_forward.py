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

from torch_game_simulation_runtime import Game, Player, ROOF_TYPE, SKILL_SLOT_ORDER, SimulationRuntime, Team


WORKSPACE_DIR = Path(__file__).resolve().parent
RESULTS_PATH = WORKSPACE_DIR / "forward_2026_predictions.csv"
MISSING_PATH = WORKSPACE_DIR / "forward_2026_missing_games.csv"
SUMMARY_PATH = WORKSPACE_DIR / "forward_2026_summary.json"
RAW_RESULTS_DIR = WORKSPACE_DIR / "forward_2026_game_results"
HTML_REPORT_PATH = WORKSPACE_DIR / "forward_2026_report.html"
ATTEMPT_SUMMARY_PATH = WORKSPACE_DIR / "forward_2026_attempt_summary.json"
PLAYER_SIMULATIONS_PATH = WORKSPACE_DIR / "forward_2026_player_simulations.csv"
ACTIVE_STATUSES = {"ACT"}
SLOT_LIMITS = {"QB": 1, "WR": 4, "RB": 3, "TE": 3}
SKILL_POSITIONS = tuple(SLOT_LIMITS.keys())
TEAM_CODE_ALIASES = {
    "ARI": ["AZ"],
}


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
        progress=not bool(task["print_game_summaries"]),
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
        "nan_input_log_rows": runtime.get_nan_input_log_rows(),
    }


class GameSimulator2026:
    def __init__(
        self,
        season: int = 2026,
        simulations: int = 200,
        workers: int = 1,
        week_start: Optional[int] = None,
        week_end: Optional[int] = None,
        reset_artifacts: bool = False,
        print_game_summaries: bool = True,
        strict_schema: bool = False,
        raw_results_dir: Path = RAW_RESULTS_DIR,
        game_id: Optional[str] = None,
        live_pbp: bool = False,
        data_cutoff_date: Optional[str] = None,
    ):
        self.season = season
        self.simulations = simulations
        self.workers = max(1, min(int(workers), 6))
        self.week_start = week_start
        self.week_end = week_end
        self.reset_artifacts = reset_artifacts
        self.print_game_summaries = print_game_summaries
        self.strict_schema = bool(strict_schema)
        self.live_pbp = bool(live_pbp)
        self.data_cutoff_date = pd.to_datetime(data_cutoff_date) if data_cutoff_date else None
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
        # Restrict player stats to data_cutoff_date if provided
        if self.data_cutoff_date is not None:
            self.player_stats = self.player_stats[self.player_stats["game_date"] <= self.data_cutoff_date].copy()
        self.team_stats = self.runtime.team_stats_wide.copy()
        self.team_stats["game_date"] = pd.to_datetime(self.team_stats["game_date"])
        # Restrict team stats to data_cutoff_date if provided
        if self.data_cutoff_date is not None:
            self.team_stats = self.team_stats[self.team_stats["game_date"] <= self.data_cutoff_date].copy()
        self._roster_week_fallback_logged: set[tuple[str, int, int]] = set()
        self._roster_status_fallback_logged: set[tuple[str, int]] = set()
        self.results = self._load_table(RESULTS_PATH)
        self.missing = self._load_table(MISSING_PATH)

    def _reset_outputs(self) -> None:
        for path in [
            RESULTS_PATH,
            MISSING_PATH,
            SUMMARY_PATH,
            ATTEMPT_SUMMARY_PATH,
            PLAYER_SIMULATIONS_PATH,
            HTML_REPORT_PATH,
        ]:
            if path.exists():
                path.unlink()
        if self.raw_results_dir.exists():
            shutil.rmtree(self.raw_results_dir)

    def _load_schedule(self) -> pd.DataFrame:
        """Load schedule for forward simulation - no completed game filter."""
        schedule = nfl.import_schedules([self.season]).copy()
        schedule = schedule[(schedule["game_type"] == "REG") & schedule["week"].between(1, 18)]
        # Forward simulation: no filter on completed games (home_score/away_score may be NaN)
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
    def _safe_group_mean(frame: pd.DataFrame, column: str) -> float | None:
        if frame.empty or column not in frame.columns:
            return None
        values = pd.to_numeric(frame[column], errors="coerce")
        values = values[np.isfinite(values.to_numpy(dtype=float))]
        if values.empty:
            return None
        return float(values.mean())

    def _build_slice_metrics(self, results: pd.DataFrame) -> Dict[str, List[Dict[str, object]]]:
        """Build metrics grouped by week and team for forward simulation (no actual data)."""
        if results.empty:
            return {"by_week": [], "by_team": []}

        frame = results.copy()
        # Forward-looking metrics: compare model projections to Vegas lines
        frame["spread_vs_line"] = pd.to_numeric(frame["sim_spread"], errors="coerce") - pd.to_numeric(frame["spread_line"], errors="coerce")
        frame["total_vs_line"] = pd.to_numeric(frame["sim_total"], errors="coerce") - pd.to_numeric(frame["total_line"], errors="coerce")

        def _group_metrics(group: pd.DataFrame) -> Dict[str, object]:
            return {
                "games": int(len(group)),
                "median_sim_spread": float(pd.to_numeric(group["sim_spread"], errors="coerce").median()),
                "median_sim_total": float(pd.to_numeric(group["sim_total"], errors="coerce").median()),
                "mean_spread_vs_line": self._safe_group_mean(group, "spread_vs_line"),
                "mean_total_vs_line": self._safe_group_mean(group, "total_vs_line"),
                "sim_spread_stdev": float(pd.to_numeric(group["sim_spread"], errors="coerce").std()),
                "sim_total_stdev": float(pd.to_numeric(group["sim_total"], errors="coerce").std()),
                "mean_sim_home_win_prob": float(pd.to_numeric(group["sim_home_win_prob"], errors="coerce").mean()),
            }

        week_rows: List[Dict[str, object]] = []
        for week, group in frame.groupby("week", dropna=False):
            week_rows.append({"week": int(week), **_group_metrics(group)})

        team_rows: List[Dict[str, object]] = []
        for team, group in frame.groupby("home_team", dropna=False):
            team_rows.append({"team": str(team), **_group_metrics(group)})

        return {
            "by_week": sorted(week_rows, key=lambda row: row["week"]),
            "by_team": sorted(team_rows, key=lambda row: (row["mean_spread_vs_line"] if row["mean_spread_vs_line"] is not None else 0, row["team"])),
        }

    def run(self, max_games: Optional[int] = None) -> None:
        print(
            f"Starting forward 2026 simulation: season={self.season}, games={len(self.schedule)}, "
            f"simulations={self.simulations}, workers={self.workers}",
            flush=True,
        )
        attempted = 0
        for game_row in self.schedule.itertuples(index=False):
            if max_games is not None and attempted >= max_games:
                break
            game_id = game_row.game_id
            attempted += 1
            try:
                record = self._simulate_game(game_row)
            except SkipGameError as error:
                self._record_missing(game_row, str(error))
                continue
            except Exception as error:
                import traceback
                if "could not convert string to float" in str(error):
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

    def _simulate_game(self, game_row) -> Dict[str, object]:
        game_id = game_row.game_id
        package = self._build_game_package(game_row)
        sim_rows: List[Dict[str, object]] = []
        player_rows: List[Dict[str, object]] = []
        home_scores: List[int] = []
        away_scores: List[int] = []
        spread_line = float(game_row.spread_line) if pd.notna(game_row.spread_line) else np.nan
        total_line = float(game_row.total_line) if pd.notna(game_row.total_line) else np.nan

        print(
            f"Simulating {game_id} ({package['home_team']} vs {package['away_team']}) with {self.simulations} sims "
            f"and {self.workers} worker(s)",
            flush=True,
        )

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
                    progress=not self.print_game_summaries,
                )
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
        if not player_frame.empty:
            player_frame.insert(0, "game_id", str(game_id))
            player_frame.insert(1, "week", int(game_row.week))
            player_frame.insert(2, "home_team", str(package["home_team"]))
            player_frame.insert(3, "away_team", str(package["away_team"]))
        player_frame.to_csv(self.raw_results_dir / f"{game_id}_players.csv", index=False)
        self._append_player_simulations(player_frame)

        home_scores_arr = np.asarray(home_scores, dtype=float)
        away_scores_arr = np.asarray(away_scores, dtype=float)
        spreads = home_scores_arr - away_scores_arr
        totals = home_scores_arr + away_scores_arr
        home_prob = float(np.mean(home_scores_arr > away_scores_arr) / max(np.mean(home_scores_arr != away_scores_arr), 1e-9))
        
        # Build output record with projection-based columns (no actual_* columns)
        home_moneyline = float(game_row.home_moneyline) if pd.notna(game_row.home_moneyline) else np.nan
        away_moneyline = float(game_row.away_moneyline) if pd.notna(game_row.away_moneyline) else np.nan
        moneyline_home_prob = implied_prob_from_moneyline(home_moneyline)

        return {
            "game_id": game_id,
            "week": int(game_row.week),
            "home_team": package["home_team"],
            "away_team": package["away_team"],
            # Projection outputs (no actual_* fields)
            "sim_home_median": float(np.quantile(home_scores_arr, 0.5)),
            "sim_away_median": float(np.quantile(away_scores_arr, 0.5)),
            "sim_spread": float(np.quantile(spreads, 0.5)),
            "sim_total": float(np.quantile(totals, 0.5)),
            "sim_home_win_prob": home_prob,
            # Percentiles for distribution
            "sim_spread_p10": float(np.quantile(spreads, 0.10)),
            "sim_spread_p25": float(np.quantile(spreads, 0.25)),
            "sim_spread_p75": float(np.quantile(spreads, 0.75)),
            "sim_spread_p90": float(np.quantile(spreads, 0.90)),
            "sim_total_p10": float(np.quantile(totals, 0.10)),
            "sim_total_p25": float(np.quantile(totals, 0.25)),
            "sim_total_p75": float(np.quantile(totals, 0.75)),
            "sim_total_p90": float(np.quantile(totals, 0.90)),
            "sim_spread_stdev": float(np.std(spreads)),
            "sim_total_stdev": float(np.std(totals)),
            # Vegas lines and implied probabilities
            "spread_line": spread_line,
            "total_line": total_line,
            "home_moneyline": home_moneyline,
            "away_moneyline": away_moneyline,
            "moneyline_home_prob": moneyline_home_prob,
            # Model vs Vegas divergence
            "spread_vs_line": float(np.quantile(spreads, 0.5)) - spread_line if np.isfinite(spread_line) else np.nan,
            "total_vs_line": float(np.quantile(totals, 0.5)) - total_line if np.isfinite(total_line) else np.nan,
            "home_win_prob_vs_moneyline": home_prob - moneyline_home_prob if np.isfinite(moneyline_home_prob) else np.nan,
            "simulations_run": int(self.simulations),
        }

    def _serialize_game_package(self, package: Dict[str, object]) -> Dict[str, object]:
        player_feature_cols = self.runtime.required_player_feature_columns
        team_feature_cols = self.runtime.required_team_feature_columns

        def serialize_players(players: List[Player]) -> List[Dict[str, object]]:
            snapshots: List[Dict[str, object]] = []
            for player in players:
                snapshots.append(
                    {
                        "name": player.name,
                        "position": player.position,
                        "player_id": player.player_id,
                        "attributes": {column: float(player.attributes[column].iloc[0]) for column in player_feature_cols},
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

    def _get_team_feature_snapshot(self, team: str, gameday: pd.Timestamp) -> pd.Series:
        team_frame = self.team_stats[self.team_stats["team"].astype(str) == str(team)].copy()
        if team_frame.empty:
            raise SkipGameError(f"missing historical team features for {team}")
        team_frame["game_date"] = pd.to_datetime(team_frame["game_date"], errors="coerce")
        team_frame = team_frame[team_frame["game_date"] < gameday].copy()
        if team_frame.empty:
            raise SkipGameError(f"missing prior team features for {team} before {gameday.date()}")
        team_frame = team_frame.sort_values(["game_date", "game_id"]).reset_index(drop=True)
        return team_frame.iloc[-1].copy()

    def _build_game_package(self, game_row) -> Dict[str, object]:
        game_id = game_row.game_id
        gameday = pd.Timestamp(game_row.gameday)
        home_team = str(game_row.home_team)
        away_team = str(game_row.away_team)
        team_feature_rows: List[pd.Series] = []
        for team in [home_team, away_team]:
            team_feature_rows.append(self._get_team_feature_snapshot(team, gameday))

        team_features = pd.DataFrame(team_feature_rows).set_index("team")
        team_features = team_features.reindex([home_team, away_team])
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
        team_candidates = [str(team)] + TEAM_CODE_ALIASES.get(str(team), [])
        roster_team = self.rosters[self.rosters["team"].isin(team_candidates)].copy()
        if roster_team.empty:
            raise SkipGameError(f"No roster rows for {team}")

        requested_week = int(week)
        available_weeks = sorted(pd.to_numeric(roster_team["week"], errors="coerce").dropna().astype(int).unique().tolist())
        if not available_weeks:
            raise SkipGameError(f"No roster weeks available for {team}")
        if requested_week in available_weeks:
            roster_week_num = requested_week
        else:
            prior_or_equal = [w for w in available_weeks if w <= requested_week]
            roster_week_num = prior_or_equal[-1] if prior_or_equal else available_weeks[-1]
            key = (str(team), requested_week, roster_week_num)
            if key not in self._roster_week_fallback_logged:
                print(
                    f"Roster fallback for {team}: requested week {requested_week}, using week {roster_week_num}",
                    flush=True,
                )
                self._roster_week_fallback_logged.add(key)

        roster_week = roster_team[roster_team["week"] == roster_week_num].copy()
        roster_week = roster_week[roster_week["player_id"].str.len() > 0]
        roster_week["player_id"] = roster_week["player_id"].astype(str)
        active_ids = set(roster_week.loc[roster_week["status"].isin(ACTIVE_STATUSES), "player_id"].astype(str))
        if not active_ids:
            key = (str(team), roster_week_num)
            if key not in self._roster_status_fallback_logged:
                print(
                    f"Roster status fallback for {team} week {roster_week_num}: no ACT statuses, using all listed skill players",
                    flush=True,
                )
                self._roster_status_fallback_logged.add(key)
            active_ids = set(roster_week["player_id"].astype(str))
        if not active_ids:
            raise SkipGameError(f"No usable players for {team} week {week}")

        depth = self.depth_charts[
            (self.depth_charts["team"] == team)
            & (self.depth_charts["dt"] <= gameday + pd.Timedelta(days=1))
        ].copy()
        if depth.empty:
            raise SkipGameError(f"No depth chart found for {team} before {gameday}")
        latest_dt = depth["dt"].max()
        depth = depth[depth["dt"] == latest_dt].copy()

        slots: Dict[str, Dict[str, str]] = {}
        used_ids: set[str] = set()
        for position, limit in SLOT_LIMITS.items():
            slots[position] = {}
            position_ranks = depth[depth["pos_abb"] == position].copy()
            sort_col = "depth_chart_position" if "depth_chart_position" in position_ranks.columns else ("pos_rank" if "pos_rank" in position_ranks.columns else None)
            if sort_col is not None:
                position_ranks = position_ranks.sort_values(sort_col).reset_index(drop=True)
            else:
                position_ranks = position_ranks.reset_index(drop=True)
            for rank_idx, rank_row in position_ranks.iterrows():
                gsis = str(rank_row["gsis_id"]).strip()
                if gsis in used_ids or gsis not in active_ids or len(slots[position]) >= limit:
                    continue
                player_name = rank_row["full_name"] if "full_name" in position_ranks.columns else rank_row.get("player_name", "")
                slots[position][str(gsis)] = str(player_name)
                used_ids.add(gsis)
        return slots

    def _build_player_objects(
        self,
        slot_map: Dict[str, Dict[str, str]],
        team: str,
        week: int,
        gameday: pd.Timestamp,
    ) -> List[Player]:
        players: List[Player] = []
        for position, slot_dict in slot_map.items():
            for slot_index, (player_id, player_name) in enumerate(slot_dict.items(), start=1):
                player_id_str = str(player_id).strip()
                if not player_id_str:
                    continue
                player_frame = self.player_stats[self.player_stats["player_id"].astype(str) == player_id_str].copy()
                if player_frame.empty:
                    player_frame = pd.DataFrame(
                        [{
                            "player_id": player_id_str,
                            "player_name": player_name,
                            "position": position,
                            "team": team,
                        }]
                    )
                    for col in self.runtime.required_player_feature_columns:
                        if col not in player_frame.columns:
                            player_frame[col] = 0.0
                else:
                    for col in self.runtime.required_player_feature_columns:
                        if col not in player_frame.columns:
                            player_frame[col] = 0.0
                slot_name = f"{position}{slot_index}"
                player_obj = Player(
                    name=player_name,
                    position=slot_name,
                    player_id=player_id_str,
                    attributes=player_frame,
                )
                players.append(player_obj)
        return players

    def _record_missing(self, game_row, reason: str) -> None:
        record = {
            "game_id": game_row.game_id,
            "week": int(game_row.week),
            "home_team": game_row.home_team,
            "away_team": game_row.away_team,
            "reason": reason,
        }
        self.missing = pd.concat([self.missing, pd.DataFrame([record])], ignore_index=True)
        self.missing.to_csv(MISSING_PATH, index=False)

    def _append_player_simulations(self, player_frame: pd.DataFrame) -> None:
        if player_frame.empty:
            return
        write_header = not PLAYER_SIMULATIONS_PATH.exists()
        player_frame.to_csv(PLAYER_SIMULATIONS_PATH, mode="a", header=write_header, index=False)

    def _write_summary(self) -> None:
        if self.results.empty:
            SUMMARY_PATH.write_text(json.dumps({}, indent=2), encoding="utf-8")
            return

        metrics = self._build_slice_metrics(self.results)
        summary = {
            "season": self.season,
            "games_completed": int(len(self.results)),
            "games_missing": int(len(self.missing)) if not self.missing.empty else 0,
            "simulations_per_game": self.simulations,
            "by_week": metrics["by_week"],
            "by_team": metrics["by_team"],
        }
        SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def _write_attempt_summary(self) -> None:
        """Write summary of projected play volumes and outcomes (forward-looking, no actual comparisons)."""
        summary = {
            "season": self.season,
            "games_projected": int(len(self.results)),
            "note": "Forward-looking projections; no actual game data for comparison.",
            "projection_metadata": {
                "data_cutoff_date": str(self.data_cutoff_date) if self.data_cutoff_date else "latest available",
                "simulations_per_game": self.simulations,
            }
        }
        ATTEMPT_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate the 2026 regular season with forward-looking projections.")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--workers", type=int, default=1, help="Worker processes for simulations (capped at 6).")
    parser.add_argument("--week-start", type=int)
    parser.add_argument("--week-end", type=int)
    parser.add_argument("--game-id", type=str, help="Only simulate the specified game_id from the schedule.")
    parser.add_argument("--max-games", type=int)
    parser.add_argument("--reset-artifacts", action="store_true")
    parser.add_argument("--no-game-summaries", action="store_true", help="Suppress game-by-game simulation output.")
    parser.add_argument("--strict-schema", action="store_true", help="Fail fast on runtime schema mismatches.")
    parser.add_argument(
        "--data-cutoff-date",
        type=str,
        help="Restrict player/team stats to this date (format: YYYY-MM-DD). Allows in-season updates.",
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
        simulator = GameSimulator2026(
            season=args.season,
            simulations=args.simulations,
            workers=args.workers,
            week_start=args.week_start,
            week_end=args.week_end,
            reset_artifacts=args.reset_artifacts,
            print_game_summaries=not args.no_game_summaries,
            strict_schema=args.strict_schema,
            game_id=args.game_id,
            live_pbp=args.live_pbp,
            data_cutoff_date=args.data_cutoff_date,
        )
        simulator.run(max_games=args.max_games)
    except Exception as e:
        print(f"\nERROR in main: {e}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
