from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import json
import re

import joblib
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)
except Exception:
    pass

from nflsim.engine.model_architecture import (
    DEFAULT_FLAT_ARCH,
    HierarchicalLatentGameModel,
    arch_sidecar_path,
    build_network,
    infer_arch_from_state_dict,
    infer_hidden_dims_from_state_dict,
    read_arch_sidecar,
)
from nflsim.engine.ngmm_model import NGMMModel
from nflsim.engine.defaults import artifact_dir
from nflsim.engine.game_effects import GameEffects
from nflsim.features.goal_line import GOAL_CAPPED_MODELS, GOAL_LINE_COLUMNS, goal_line_inputs


class CascadeModelWrapper:
    """Wraps cascade stage-2 (residual distribution) and stage-3 (inference) models.

    At game start, the stage-2 model produces a 32-bin probability distribution
    from context features; a residual value is drawn via inverse CDF sampling.
    That sampled residual is then used as the ``residual_com`` feature for every
    play-level forward pass through the stage-3 model.
    """

    def __init__(
        self,
        stage3_model: nn.Module,
        stage2_model: nn.Module,
        stage2_preprocessor: Any,
        bin_edges: np.ndarray,
        bin_centers: np.ndarray,
        pca_components: np.ndarray,
        pca_mean: np.ndarray,
        context_columns: List[str],
        stage3_preprocessor: Any,
        latent_dim: int = 1,
    ):
        self.stage3_model = stage3_model
        self.stage2_model = stage2_model
        self.stage2_preprocessor = stage2_preprocessor
        self.bin_edges = bin_edges
        self.bin_centers = bin_centers
        self.pca_components = pca_components
        self.pca_mean = pca_mean
        self.context_columns = context_columns
        self.stage3_preprocessor = stage3_preprocessor
        self.latent_dim = latent_dim

    def sample_residual(self, context_values: Dict[str, float], rng: np.random.Generator) -> float:
        """Sample a scalar residual from the stage-2 discretized distribution."""
        # Build array in the same column order used during training
        ctx_row = np.array(
            [float(pd.to_numeric(context_values.get(col, 0.0), errors="coerce")) for col in self.context_columns],
            dtype=np.float32,
        ).reshape(1, -1)
        ctx_arr = self.stage2_preprocessor.transform(ctx_row).astype(np.float32)
        ctx_arr = np.clip(np.nan_to_num(ctx_arr, nan=0.0), -10.0, 10.0)
        with torch.no_grad():
            dist = torch.softmax(self.stage2_model(torch.from_numpy(ctx_arr)), dim=-1).numpy()[0]
        # Inverse CDF sampling
        cdf = np.cumsum(dist)
        cdf = cdf / cdf[-1]
        u = rng.uniform()
        bin_idx = int(np.searchsorted(cdf, u, side="right"))
        bin_idx = min(bin_idx, len(dist) - 1)
        bin_width = self.bin_edges[bin_idx + 1] - self.bin_edges[bin_idx]
        prev_cdf = cdf[bin_idx - 1] if bin_idx > 0 else 0.0
        bin_mass = float(dist[bin_idx])
        frac = np.clip((u - prev_cdf) / bin_mass, 0.0, 1.0) if bin_mass > 0 else 0.5
        return float(self.bin_edges[bin_idx] + frac * bin_width)


SKILL_SLOT_ORDER = ["QB1", "WR1", "WR2", "WR3", "WR4", "RB1", "RB2", "RB3", "TE1", "TE2", "TE3"]
ROOF_TYPE = {
    "ARI": "dome",
    "ATL": "dome",
    "BAL": "outdoors",
    "BUF": "outdoors",
    "CAR": "outdoors",
    "CHI": "outdoors",
    "CIN": "outdoors",
    "CLE": "outdoors",
    "DAL": "dome",
    "DEN": "outdoors",
    "DET": "dome",
    "GB": "outdoors",
    "HOU": "dome",
    "IND": "dome",
    "JAX": "outdoors",
    "KC": "outdoors",
    "LV": "dome",
    "LAC": "dome",
    "LAR": "dome",
    "LA": "dome",
    "OAK": "outdoors",
    "MIA": "outdoors",
    "MIN": "dome",
    "NE": "outdoors",
    "NO": "dome",
    "NYG": "outdoors",
    "NYJ": "outdoors",
    "PHI": "outdoors",
    "PIT": "outdoors",
    "SF": "outdoors",
    "SEA": "outdoors",
    "TB": "outdoors",
    "TEN": "outdoors",
    "WAS": "outdoors",
}


def get_probs(y_pred_logits: np.ndarray) -> np.ndarray:
    """CORN class probabilities: P(0) = 1 - P(>0), P(k) = P(>k-1) - P(>k), P(last) = P(>last-1)."""
    probas = torch.cumprod(torch.sigmoid(torch.from_numpy(y_pred_logits)), dim=1).float().numpy()
    return np.concatenate([1.0 - probas[:, :1], probas[:, :-1] - probas[:, 1:], probas[:, -1:]], axis=1)


class PlayRow(dict):
    """One play's model inputs. forward_model_logits accepts it in place of a one-row DataFrame, which costs
    far more to build than the model call itself; ``frame()`` gives that DataFrame where one is needed."""

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([self], columns=list(self))


def build_frame_from_values(values: List[object], columns: List[str]) -> PlayRow:
    row = PlayRow((column, np.nan) for column in columns)
    for index, value in enumerate(values):
        if index >= len(columns):
            break
        row[columns[index]] = value
    return row


class SimulationRuntime:
    pass_outcomes_given_attempt = False
    goal_line_models = frozenset()
    passer_columns = [
        "cpoe_bayes",
        "turnover_bayes",
        "sack_bayes",
        "cpoe_short",
        "cpoe_medium",
        "cpoe_deep",
        "deep_pass_rate",
        "medium_pass_rate",
        "short_pass_rate",
    ]
    def_cols = [
        "deep_cpoe_allowed",
        "int_rate",
        "medium_cpoe_allowed",
        "short_cpoe_allowed",
        "yac_oe_allowed",
        "sack_rate",
        "rush_yards_per_carry",
        "def_rush_epa_rate",
        "def_pass_epa_rate",
        "def_winning_rate",
        "def_losing_rate",
        "def_tied_rate",
        "passing_massey_defense",
        "rushing_massey_defense",
    ]
    team_stats_cols = [
        "losing_rate",
        "tied_rate",
        "team_4th_rate",
        "winning_rate",
        "rushing_epa_rate",
        "epa_rate",
        "passing_massey_rating",
        "passing_massey_offense",
        "rushing_massey_rating",
        "rushing_massey_offense",
        "opp_def_pass_epa_rate",
        "opp_def_rush_epa_rate",
        "opp_pass_epa_rate",
        "opp_rushing_epa_rate",
    ]
    team_stats_columns = team_stats_cols
    team_tendency_columns = [
        "pass_oe",
        "pass_oe_weekly_variation",
        "target_gini_roll4",
        "rb_carry_split_gini_roll4",
        "rb_top_carry_share_roll4",
        "pace_of_play_bayes",
    ]
    qb_context_columns = ["qb_team_games_played", "qb_dome_rate_bayes"]
    pregame_context_columns = [
        "pass_oe",
        "pass_oe_weekly_variation",
        "target_gini_roll4",
        "rb_carry_split_gini_roll4",
        "rb_top_carry_share_roll4",
        "pace_of_play_bayes",
        "days_rest",
        "opp_days_rest",
    ]
    offense_context_columns = [
        "pass_oe",
        "pass_oe_weekly_variation",
        "target_gini_roll4",
    ]
    backfield_context_columns = [
        "rb_carry_split_gini_roll4",
        "rb_top_carry_share_roll4",
    ]
    game_state_cols = [
        "defteam_score",
        "down",
        "game_seconds_remaining",
        "posteam_score",
        "posteam_type",
        "yardline_100",
        "ydstogo",
        "roof",
        "score_differential",
    ]
    pass_columns = ["air_yards"]
    rusher_columns = ["close_bayes", "explosive_bayes", "ypc_bayes", "rush_rate", "scramble_rate"]
    position_cols = ["position"]
    receiver_columns = ["deep", "medium", "short", "fd_bayes", "rz_bayes", "yac_bayes", "adot", "ypt"]
    target_embed_columns = [f"target_embed_{index}" for index in range(1, 3)]
    running_embed_columns = [f"rb_target_embed_{index}" for index in range(1, 3)]
    play_cols = ["play_type_cat"]
    play_outcome = ["play_outcome"]
    play_outcome_cols = play_outcome
    penalty_context_cols = ["penalty_context"]
    # Special-teams models (punt_distance, punt_return, kickoff): the kicking team's punting and
    # coverage, the receiving team's returns (team columns st_* from nflsim.features.special_teams).
    special_teams_team_columns = [
        "st_punt_gross", "st_punt_return_allowed", "st_punt_return", "st_ko_start_allowed", "st_ko_start",
        "st_ko_rule_2024", "st_ko_rule_2025",
    ]
    special_teams_cols = [
        "kick_st_punt_gross", "kick_st_punt_return_allowed", "kick_st_ko_start_allowed",
        "ret_st_punt_return", "ret_st_ko_start", "st_ko_rule_2024",
    ]
    # Kickoff models trained on 2025 kickoffs also see the 2025 rule (touchbacks to the 35).
    kickoff_rule_cols = ["st_ko_rule_2025"]
    punt_landing_cols = ["kick_distance", "punt_landing_100"]
    slot_stat_cols: List[str] = []
    models = [
        "kick", "ay", "target", "comp", "int", "yac", "sack", "scramble", "rush",
        "scramble_yards", "punt", "time", "rusher", "play", "fumble", "def_ret",
        "penalty_no_play", "penalty_on_play", "penalty_yards", "penalty_first_down",
    ]
    # Loaded when the bundle has them; older bundles fall back to the previous behavior.
    optional_models = ["sack_yards", "punt_distance", "punt_return", "kickoff"]
    # Bound scaled inputs so a near-zero-variance training column (StandardScaler
    # scale_ ~1e-11) can't turn a tiny real-world deviation into a z-score in the
    # hundreds of millions and saturate/degenerate the network's output.
    MODEL_INPUT_CLIP = 10.0

    def __init__(self, data_dir: Optional[Path] = None, strict_schema: bool = False, debug_print_outliers: bool = False):
        self.data_dir = Path(data_dir) if data_dir is not None else artifact_dir()
        self.model_dict: Dict[str, nn.Module] = {}
        self.strict_schema = bool(strict_schema)
        # Opt-in only: printing every |z|>3 input is extremely verbose (thousands of lines per
        # game) and provides no value once the underlying feature-construction bug it's meant to
        # surface has been fixed; keep it available for future debugging but off by default.
        self.debug_print_outliers = bool(debug_print_outliers)
        self.column_dict: Dict[str, Iterable[str]] = {}
        self.bin_dict: Dict[str, pd.DataFrame] = {}
        self.processor: Dict[str, object] = {}
        self.schema_issues: List[str] = []
        self.nan_input_log: Dict[str, Dict[str, Any]] = {}
        self._nan_issue_labels: set[str] = set()

        self.player_stats_wide = pd.read_csv(self.data_dir / "player_stats_wide.csv")
        self.team_stats_wide = pd.read_csv(self.data_dir / "team_stats_wide.csv")
        self.required_player_feature_columns = sorted(
            set(self.passer_columns + self.receiver_columns + self.rusher_columns + self.qb_context_columns)
        )
        self.required_team_feature_columns = sorted(
            set(self.team_stats_cols + self.def_cols + self.team_tendency_columns + self.pregame_context_columns
                + self.special_teams_team_columns)
        )
        # Prior feature columns for hierarchical latent models
        self.target_prior_columns = self.team_stats_cols + self.team_tendency_columns + self.qb_context_columns
        self.rusher_prior_columns = self.team_stats_cols + self.team_tendency_columns
        self._load_models()
        self.pass_outcomes_given_attempt = self._pass_outcomes_given_attempt()
        self.goal_line_models = frozenset(
            model for model in GOAL_CAPPED_MODELS if GOAL_LINE_COLUMNS[0] in getattr(self.processor.get(model), "feature_names_in_", ())
        )
        self._sync_hierarchical_runtime_schema()
        self.game_effects = GameEffects.load(self.data_dir)

    def _record_schema_issue(self, message: str) -> None:
        if message not in self.schema_issues:
            self.schema_issues.append(message)

    def reset_nan_input_log(self) -> None:
        self.nan_input_log = {}
        self._nan_issue_labels = set()

    def _record_nan_detected_message(self, model_key: str, stage: str) -> None:
        label = f"{model_key}:{stage}"
        if label in self._nan_issue_labels:
            return
        self._nan_issue_labels.add(label)
        self._record_schema_issue(f"{label}: detected NaN inputs (see backtest_2025_nan_input_log.csv)")

    def _accumulate_nan_log(
        self,
        model_key: str,
        stage: str,
        row_count: int,
        nan_cells: int,
        rows_with_nan: int,
        column_nan_counts: Dict[str, int],
    ) -> None:
        if nan_cells <= 0:
            return
        key = f"{model_key}|{stage}"
        entry = self.nan_input_log.get(key)
        if entry is None:
            entry = {
                "model_key": model_key,
                "stage": stage,
                "calls_with_nan": 0,
                "rows_observed": 0,
                "rows_with_nan": 0,
                "nan_cells": 0,
                "column_nan_counts": {},
            }
            self.nan_input_log[key] = entry
        entry["calls_with_nan"] = int(entry["calls_with_nan"]) + 1
        entry["rows_observed"] = int(entry["rows_observed"]) + int(row_count)
        entry["rows_with_nan"] = int(entry["rows_with_nan"]) + int(rows_with_nan)
        entry["nan_cells"] = int(entry["nan_cells"]) + int(nan_cells)
        col_counts = entry["column_nan_counts"]
        for column, count in column_nan_counts.items():
            col_counts[str(column)] = int(col_counts.get(str(column), 0)) + int(count)
        self._record_nan_detected_message(model_key, stage)

    def _record_nan_columns(self, model_key: str, stage: str, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        nan_mask = frame.isna()
        nan_cells = int(nan_mask.to_numpy().sum())
        if nan_cells <= 0:
            return
        rows_with_nan = int(nan_mask.any(axis=1).sum())
        col_counts: Dict[str, int] = {}
        for column in frame.columns:
            count = int(nan_mask[column].sum())
            if count > 0:
                col_counts[str(column)] = count
        self._accumulate_nan_log(
            model_key=model_key,
            stage=stage,
            row_count=len(frame),
            nan_cells=nan_cells,
            rows_with_nan=rows_with_nan,
            column_nan_counts=col_counts,
        )

    def _record_nan_array(
        self,
        model_key: str,
        stage: str,
        values: np.ndarray,
        columns: Optional[List[str]] = None,
    ) -> None:
        values_arr = np.asarray(values, dtype=float)
        if values_arr.ndim != 2 or values_arr.size == 0:
            return
        nan_mask = np.isnan(values_arr)
        nan_cells = int(nan_mask.sum())
        if nan_cells <= 0:
            return
        rows_with_nan = int(np.any(nan_mask, axis=1).sum())
        if columns is not None and len(columns) == values_arr.shape[1]:
            names = [str(column) for column in columns]
        else:
            names = [f"col_{idx}" for idx in range(values_arr.shape[1])]
        col_counts: Dict[str, int] = {}
        for idx, name in enumerate(names):
            count = int(nan_mask[:, idx].sum())
            if count > 0:
                col_counts[name] = count
        self._accumulate_nan_log(
            model_key=model_key,
            stage=stage,
            row_count=int(values_arr.shape[0]),
            nan_cells=nan_cells,
            rows_with_nan=rows_with_nan,
            column_nan_counts=col_counts,
        )

    def get_nan_input_log_rows(self) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        for entry in self.nan_input_log.values():
            base = {
                "model_key": str(entry["model_key"]),
                "stage": str(entry["stage"]),
                "calls_with_nan": int(entry["calls_with_nan"]),
                "rows_observed": int(entry["rows_observed"]),
                "rows_with_nan": int(entry["rows_with_nan"]),
                "nan_cells": int(entry["nan_cells"]),
            }
            column_counts = entry.get("column_nan_counts", {})
            if not column_counts:
                rows.append({**base, "column": "(unknown)", "column_nan_count": 0})
                continue
            for column, count in sorted(column_counts.items()):
                rows.append({**base, "column": str(column), "column_nan_count": int(count)})
        return rows

    @staticmethod
    def _sort_embed_columns(columns: List[str]) -> List[str]:
        def _embed_index(column: str) -> int:
            match = re.search(r"(\d+)$", str(column))
            return int(match.group(1)) if match else 10**9

        return sorted(columns, key=_embed_index)

    def _sync_hierarchical_runtime_schema(self) -> None:
        # Use fitted preprocessors as the source of truth for prior and embed columns.
        target_prior_proc = self.processor.get("target_prior")
        if target_prior_proc is not None and hasattr(target_prior_proc, "feature_names_in_"):
            self.target_prior_columns = [str(column) for column in target_prior_proc.feature_names_in_]

        rusher_prior_proc = self.processor.get("rusher_prior")
        if rusher_prior_proc is not None and hasattr(rusher_prior_proc, "feature_names_in_"):
            self.rusher_prior_columns = [str(column) for column in rusher_prior_proc.feature_names_in_]

        target_play_proc = self.processor.get("target")
        if target_play_proc is not None and hasattr(target_play_proc, "feature_names_in_"):
            target_embed = [
                str(column)
                for column in target_play_proc.feature_names_in_
                if str(column).startswith("target_embed_")
            ]
            if target_embed:
                self.target_embed_columns = self._sort_embed_columns(target_embed)

        rusher_play_proc = self.processor.get("rusher")
        if rusher_play_proc is not None and hasattr(rusher_play_proc, "feature_names_in_"):
            rusher_embed = [
                str(column)
                for column in rusher_play_proc.feature_names_in_
                if str(column).startswith("rb_target_embed_")
            ]
            if rusher_embed:
                self.running_embed_columns = self._sort_embed_columns(rusher_embed)

        target_model = self.model_dict.get("target")
        if isinstance(target_model, HierarchicalLatentGameModel):
            expected = int(target_model.latent_dim)
            if len(self.target_embed_columns) != expected:
                if target_play_proc is not None and hasattr(target_play_proc, "feature_names_in_"):
                    aligned = [
                        str(column)
                        for column in target_play_proc.feature_names_in_
                        if str(column).startswith("target_embed_")
                    ]
                    aligned = self._sort_embed_columns(aligned)[:expected]
                    if aligned:
                        self.target_embed_columns = aligned
                if len(self.target_embed_columns) != expected:
                    self.target_embed_columns = [f"target_embed_{index}" for index in range(1, expected + 1)]
                message = (
                    "target: adjusted runtime embed columns to latent_dim "
                    f"{expected}: {self.target_embed_columns}"
                )
                self._record_schema_issue(message)

        rusher_model = self.model_dict.get("rusher")
        if isinstance(rusher_model, HierarchicalLatentGameModel):
            expected = int(rusher_model.latent_dim)
            if len(self.running_embed_columns) != expected:
                if rusher_play_proc is not None and hasattr(rusher_play_proc, "feature_names_in_"):
                    aligned = [
                        str(column)
                        for column in rusher_play_proc.feature_names_in_
                        if str(column).startswith("rb_target_embed_")
                    ]
                    aligned = self._sort_embed_columns(aligned)[:expected]
                    if aligned:
                        self.running_embed_columns = aligned
                if len(self.running_embed_columns) != expected:
                    self.running_embed_columns = [f"rb_target_embed_{index}" for index in range(1, expected + 1)]
                message = (
                    "rusher: adjusted runtime embed columns to latent_dim "
                    f"{expected}: {self.running_embed_columns}"
                )
                self._record_schema_issue(message)

    def with_goal_line_inputs(self, model: str, row: PlayRow, yards: float) -> PlayRow:
        """Add the goal-line inputs on bundles whose yardage models were trained with them."""
        if model in self.goal_line_models:
            row.update(goal_line_inputs(float(yards)))
        return row

    def _pass_outcomes_given_attempt(self) -> bool:
        """True when comp and int were trained on throws only; older bundles trained them on all dropbacks."""
        scopes = set()
        for model in ("comp", "int"):
            path = self.data_dir / f"{model}_arch.json"
            if path.exists():
                scopes.add(json.loads(path.read_text()).get("row_scope"))
        if len(scopes) > 1:
            raise ValueError(f"comp and int disagree on training rows: {sorted(map(str, scopes))}")
        return "attempts" in scopes

    def _load_models(self) -> None:
        for model in [*self.models, *self.optional_models]:
            bins = False
            bin_path = self.data_dir / f"{model}_bins.csv"
            if bin_path.exists():
                self.bin_dict[model] = pd.read_csv(bin_path)
                bins = True

            columns_path = self.data_dir / f"{model}_columns.pkl"
            if model == "scramble":
                scramble_classifier_columns = self.data_dir / "scramble_classifier_columns.pkl"
                if scramble_classifier_columns.exists():
                    columns_path = scramble_classifier_columns
            # Also check for cascade columns file
            cascade_columns_path = self.data_dir / f"{model}_cascade_columns.pkl"
            cascade_arch_check = self.data_dir / f"{model}_cascade_arch.json"
            if columns_path.exists():
                self.column_dict[model] = joblib.load(columns_path)
            elif cascade_columns_path.exists() and cascade_arch_check.exists():
                self.column_dict[model] = joblib.load(cascade_columns_path)
            else:
                continue

            processor_path = self.data_dir / f"{model}_preprocessor.pkl"
            if model == "scramble":
                scramble_classifier_processor = self.data_dir / "scramble_classifier_preprocessor.pkl"
                if scramble_classifier_processor.exists():
                    processor_path = scramble_classifier_processor
            if processor_path.exists():
                self.processor[model] = joblib.load(processor_path)

            # Check for cascade artifacts first (they replace the standard model entirely)
            cascade_arch_path = self.data_dir / f"{model}_cascade_arch.json"
            if cascade_arch_path.exists():
                try:
                    import json as _json
                    with open(cascade_arch_path) as _f:
                        cascade_meta = _json.load(_f)
                    if cascade_meta.get("kind") == "cascade":
                        # Load stage 3 model (inference)
                        s3_state = torch.load(self.data_dir / f"{model}_cascade_stage3_model.pt", map_location="cpu", weights_only=True)
                        s3_input_dim = cascade_meta["dims"]["stage3_input_dim"]
                        s3_output_dim = cascade_meta["dims"]["output_dim"]
                        s3_arch = cascade_meta.get("arch", {})
                        s3_hidden = s3_arch.get("stage3_hidden_dims", [64, 32, 16])
                        s3_dropout = s3_arch.get("stage3_dropout", 0.1)
                        from nflsim.engine.cascade_trainer import CascadeMLP, CascadeConfig as _CC
                        s3_config = _CC(hidden_dims=tuple(s3_hidden), dropout=s3_dropout)
                        stage3_net = CascadeMLP(s3_input_dim, s3_output_dim, s3_config)
                        stage3_net.load_state_dict(s3_state, strict=True)
                        stage3_net.eval()

                        # Load stage 2 model (residual distribution)
                        s2_state = torch.load(self.data_dir / f"{model}_cascade_stage2_model.pt", map_location="cpu", weights_only=True)
                        s2_context_dim = cascade_meta["dims"]["stage2_context_dim"]
                        s2_n_bins = cascade_meta["n_bins"]
                        s2_hidden = s3_arch.get("stage2_hidden_dims", [32, 64, 32])
                        s2_dropout = s3_arch.get("stage2_dropout", 0.1)
                        from nflsim.engine.cascade_trainer import ResidualGridNet
                        s2_config = _CC(stage2_hidden_dims=tuple(s2_hidden), stage2_dropout=s2_dropout)
                        stage2_net = ResidualGridNet(s2_context_dim, s2_n_bins, s2_config)
                        stage2_net.load_state_dict(s2_state, strict=True)
                        stage2_net.eval()

                        # Load preprocessors and metadata
                        s3_preprocessor = joblib.load(self.data_dir / f"{model}_cascade_stage3_preprocessor.pkl")
                        s2_preprocessor = joblib.load(self.data_dir / f"{model}_cascade_stage2_preprocessor.pkl")
                        bin_edges = np.array(cascade_meta["bin_edges"])
                        bin_centers = np.array(cascade_meta["bin_centers"])
                        pca_components = np.array(cascade_meta["pca_components"])
                        pca_mean = np.array(cascade_meta["pca_mean"])
                        context_columns = cascade_meta.get("context_columns", [])

                        wrapper = CascadeModelWrapper(
                            stage3_model=stage3_net,
                            stage2_model=stage2_net,
                            stage2_preprocessor=s2_preprocessor,
                            bin_edges=bin_edges,
                            bin_centers=bin_centers,
                            pca_components=pca_components,
                            pca_mean=pca_mean,
                            context_columns=context_columns,
                            stage3_preprocessor=s3_preprocessor,
                            latent_dim=1,
                        )
                        self.model_dict[model] = wrapper
                        self.processor[model] = s3_preprocessor
                        self.processor[f"{model}_cascade_stage2"] = s2_preprocessor
                        continue
                except FileNotFoundError as exc:
                    self._record_schema_issue(f"{model}: cascade artifact missing: {exc}")
                    continue

            try:
                model_path = self.data_dir / f"{model}_model.pt"
                if model == "scramble":
                    scramble_classifier_model = self.data_dir / "scramble_classifier_model.pt"
                    if scramble_classifier_model.exists():
                        model_path = scramble_classifier_model
                state_dict = torch.load(model_path, map_location="cpu")
                output_key = "out.weight" if "out.weight" in state_dict else "fixed_head.weight"
                checkpoint_out_width = int(state_dict[output_key].shape[0])
                # An arch sidecar (written by Torch Modeling.py) carries the tuned layer
                # widths/dropout; without one, fall back to inferring shapes from the weights.
                sidecar = read_arch_sidecar(arch_sidecar_path(self.data_dir, model_path.stem[: -len("_model")]))
                if sidecar is None and infer_hidden_dims_from_state_dict(state_dict) != DEFAULT_FLAT_ARCH["hidden_dims"]:
                    # Activation and dropout cannot be recovered from weights alone, so a
                    # non-legacy trunk without a sidecar may silently load the wrong nonlinearity.
                    self._record_schema_issue(
                        f"{model}: non-default architecture with no arch sidecar; "
                        f"assuming {DEFAULT_FLAT_ARCH['activation']} activation"
                    )
                network: nn.Module
                if (sidecar or {}).get("kind") == "ngmm":
                    dims = (sidecar or {}).get("dims", {})
                    network = build_network("ngmm", dims, (sidecar or {}).get("arch"))
                    prior_proc_path = self.data_dir / f"{model}_prior_preprocessor.pkl"
                    if prior_proc_path.exists():
                        self.processor[f"{model}_prior"] = joblib.load(prior_proc_path)
                    else:
                        raise FileNotFoundError(f"NGMM model '{model}' is missing group-context preprocessor")
                elif "prior_encoder.hidden.0.weight" in state_dict:
                    prior_input_dim = int(state_dict["prior_encoder.hidden.0.weight"].shape[1])
                    play_input_dim = int(state_dict["fc1.weight"].shape[1]) - int(state_dict["prior_encoder.mu.weight"].shape[0])
                    latent_dim = int(state_dict["prior_encoder.mu.weight"].shape[0])
                    arch = (sidecar or {}).get("arch") if (sidecar or {}).get("kind") == "hierarchical" else None
                    if arch is None:
                        arch = infer_arch_from_state_dict(state_dict, "hierarchical")
                    network = build_network(
                        "hierarchical",
                        {
                            "play_input_dim": play_input_dim,
                            "prior_input_dim": prior_input_dim,
                            "latent_dim": latent_dim,
                            # Hierarchical heads store the exact class count in out.weight rows.
                            "output_dim": checkpoint_out_width,
                            "ordinal_output": False,
                        },
                        arch,
                    )
                    # Load the prior preprocessor for this model
                    prior_proc_path = self.data_dir / f"{model}_prior_preprocessor.pkl"
                    if model == "scramble":
                        scramble_classifier_prior = self.data_dir / "scramble_classifier_prior_preprocessor.pkl"
                        if scramble_classifier_prior.exists():
                            prior_proc_path = scramble_classifier_prior
                    if prior_proc_path.exists():
                        self.processor[f"{model}_prior"] = joblib.load(prior_proc_path)
                    else:
                        message = (
                            f"{model}: hierarchical checkpoint is missing prior preprocessor "
                            f"'{prior_proc_path.name}'"
                        )
                        self._record_schema_issue(message)
                        if self.strict_schema:
                            raise ValueError(message)
                    play_processor = self.processor.get(model)
                    if play_processor is not None and hasattr(play_processor, "get_feature_names_out"):
                        play_width = int(len(play_processor.get_feature_names_out()))
                        if play_width != play_input_dim:
                            message = f"{model}: play preprocessor width {play_width} != checkpoint width {play_input_dim}"
                            self._record_schema_issue(message)
                            if self.strict_schema:
                                raise ValueError(message)
                    prior_processor = self.processor.get(f"{model}_prior")
                    if prior_processor is not None and hasattr(prior_processor, "get_feature_names_out"):
                        prior_width = int(len(prior_processor.get_feature_names_out()))
                        if prior_width != prior_input_dim:
                            message = f"{model}: prior preprocessor width {prior_width} != checkpoint width {prior_input_dim}"
                            self._record_schema_issue(message)
                            if self.strict_schema:
                                raise ValueError(message)
                elif bins:
                    input_dim = int(state_dict["fc1.weight"].shape[1])
                    arch = (sidecar or {}).get("arch") if (sidecar or {}).get("kind") == "corn" else None
                    if arch is None:
                        arch = infer_arch_from_state_dict(state_dict, "flat")
                    # CORN models output (num_classes - 1) logits.
                    network = build_network(
                        "corn", {"input_dim": input_dim, "output_dim": checkpoint_out_width + 1}, arch
                    )
                else:
                    input_dim = int(state_dict["fc1.weight"].shape[1])
                    arch = (sidecar or {}).get("arch") if (sidecar or {}).get("kind") == "flat" else None
                    if arch is None:
                        arch = infer_arch_from_state_dict(state_dict, "flat")
                    network = build_network(
                        "flat", {"input_dim": input_dim, "output_dim": checkpoint_out_width}, arch
                    )
                processor = self.processor.get(model)
                if processor is not None and hasattr(processor, "get_feature_names_out") and not isinstance(network, (HierarchicalLatentGameModel, NGMMModel)):
                    processor_width = int(len(processor.get_feature_names_out()))
                    if processor_width != int(state_dict["fc1.weight"].shape[1]):
                        message = f"{model}: preprocessor width {processor_width} != checkpoint width {int(state_dict['fc1.weight'].shape[1])}"
                        self._record_schema_issue(message)
                        if self.strict_schema:
                            raise ValueError(message)
                network.load_state_dict(state_dict, strict=True)
                network.eval()
                self.model_dict[model] = network
            except FileNotFoundError:
                continue

    def as_model_input(self, model: str, values: np.ndarray) -> torch.Tensor:
        network = self.model_dict[model]
        if isinstance(network, CascadeModelWrapper):
            expected_width = network.stage3_model.fc1.in_features
        elif isinstance(network, HierarchicalLatentGameModel):
            expected_width = network.fc1.in_features - network.latent_dim
        else:
            expected_width = network.fc1.in_features if hasattr(network, "fc1") else network.fixed[0].in_features
        values = values.astype(np.float32)
        if values.shape[1] < expected_width:
            message = f"{model}: input width {values.shape[1]} below expected width {expected_width}"
            self._record_schema_issue(message)
            if self.strict_schema:
                raise ValueError(message)
            pad_width = expected_width - values.shape[1]
            values = np.pad(values, ((0, 0), (0, pad_width)), mode="constant")
        elif values.shape[1] > expected_width:
            message = f"{model}: input width {values.shape[1]} exceeds expected width {expected_width}"
            self._record_schema_issue(message)
            if self.strict_schema:
                raise ValueError(message)
            values = values[:, :expected_width]
        return torch.as_tensor(values)

    def _post_transform_columns(self, processor: object) -> Optional[List[str]]:
        """Return column labels aligned with a processor's transform() output order.

        A ColumnTransformer reorders its output into per-transformer blocks (e.g. all
        numeric columns from the 'num' sub-transformer, followed by the one-hot expanded
        'cat' columns) which does NOT match `feature_names_in_` (the pre-transform input
        order) once any transformer earlier in the list drops/expands columns. Using
        `feature_names_in_` to index the transformed array silently pairs each column name
        with the wrong transformed value. Prefer `get_feature_names_out()`, which reflects
        the true output order, and strip the `<transformer>__` prefix to recover the
        original column name where possible.
        """
        if hasattr(processor, "get_feature_names_out"):
            try:
                raw_names = list(processor.get_feature_names_out())
            except Exception:
                raw_names = None
            if raw_names:
                cleaned = []
                for name in raw_names:
                    name = str(name)
                    cleaned.append(name.split("__", 1)[1] if "__" in name else name)
                return cleaned
        expected_columns = getattr(processor, "feature_names_in_", None)
        if expected_columns is not None:
            return [str(column) for column in expected_columns]
        return None

    def transform_with_processor(self, processor_key: str, frame: pd.DataFrame) -> np.ndarray:
        processor = self.processor[processor_key]
        transformed_frame = frame.copy()
        expected_columns = getattr(processor, "feature_names_in_", None)
        if expected_columns is not None:
            expected = [str(column) for column in expected_columns]
            missing = [column for column in expected if column not in transformed_frame.columns]
            if missing:
                message = f"{processor_key}: auto-filled missing columns: {missing}"
                self._record_schema_issue(message)
                for column in missing:
                    transformed_frame[column] = 0.0
            transformed_frame = transformed_frame.reindex(columns=expected, fill_value=0.0)
            self._record_nan_columns(processor_key, "processor_input", transformed_frame)
        return processor.transform(transformed_frame)

    def _validate_selector_required_columns(self, model_key: str, frame: pd.DataFrame) -> None:
        if model_key not in {"target", "rusher"}:
            return
        processor = self.processor.get(model_key)
        if processor is None:
            return
        expected_columns = getattr(processor, "feature_names_in_", None)
        if expected_columns is None:
            return
        expected = [str(column) for column in expected_columns]
        # residual_com is injected by forward_model_logits for cascade models
        # after this validation but before preprocessing, so exclude it here.
        model = self.model_dict.get(model_key)
        if isinstance(model, CascadeModelWrapper):
            expected = [c for c in expected if c != "residual_com"]
        missing = [column for column in expected if column not in frame.columns]
        if missing:
            message = (
                f"{model_key}: missing required selector columns (fail-fast): {missing}"
            )
            self._record_schema_issue(message)
            raise ValueError(message)

    def forward_model_logits(
        self,
        model_key: str,
        play_frame: pd.DataFrame,
        base_values: Optional[Dict[str, object]] = None,
        latent_draw: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        if isinstance(play_frame, PlayRow):
            play_frame = play_frame.frame()
        self._validate_selector_required_columns(model_key, play_frame)
        # For cascade models, inject the residual_com feature before preprocessing
        model = self.model_dict[model_key]
        if isinstance(model, CascadeModelWrapper):
            residual_value = 0.0
            if latent_draw is not None:
                residual_value = float(np.asarray(latent_draw, dtype=np.float64).ravel()[0])
            play_frame = play_frame.copy()
            play_frame["residual_com"] = residual_value
        play_inp = self.transform_with_processor(model_key, play_frame)
        processor = self.processor.get(model_key)
        play_columns = self._post_transform_columns(processor)
        self._record_nan_array(model_key, "post_transform", play_inp, play_columns)
        # Near-zero-variance training columns (e.g. a QB's receiver-slot features) give a
        # StandardScaler `scale_` of ~1e-11; any inference value that differs even slightly
        # from the fitted mean then explodes into a z-score in the hundreds of millions,
        # saturating the network and collapsing its output to a single degenerate class.
        play_inp = np.clip(np.nan_to_num(play_inp, nan=0.0), -self.MODEL_INPUT_CLIP, self.MODEL_INPUT_CLIP)
        
        if self.debug_print_outliers and play_columns is not None:
            for i, col in enumerate(play_columns):
                if i >= play_inp.shape[1]:
                    break
                if np.any(play_inp[:, i] > 3) or np.any(play_inp[:, i] < -3):
                    original_value = play_frame[col].values if col in play_frame.columns else None
                    print(f"{col} (original): {original_value}")
                    print(f"{col}: {play_inp[:, i]}")
        
        play_tensor = self.as_model_input(model_key, play_inp)
        model = self.model_dict[model_key]
        if model_key == "comp":
            if play_columns is not None:
                for i, col in enumerate(play_columns):
                    if i >= play_inp.shape[1]:
                        break
                    #print(f"{col}: {play_inp[:, i]}")
        if isinstance(model, CascadeModelWrapper):
            # The residual_com feature is already baked into the preprocessor's
            # expected columns; the latent_draw carries the sampled residual value.
            with torch.no_grad():
                return model.stage3_model(play_tensor)
        if isinstance(model, NGMMModel):
            prior_processor_key = f"{model_key}_prior"
            prior_values: Dict[str, object] = {}
            if base_values:
                prior_values.update(base_values)
            if not play_frame.empty:
                prior_values.update(play_frame.iloc[0].to_dict())

            
            prior_frame = pd.DataFrame([prior_values])
            prior_inp = self.transform_with_processor(prior_processor_key, prior_frame)
            prior_inp = np.clip(np.nan_to_num(prior_inp, nan=0.0), -self.MODEL_INPUT_CLIP, self.MODEL_INPUT_CLIP).astype(np.float32)
            prior_tensor = torch.as_tensor(prior_inp)
            if latent_draw is None:
                latent = torch.randn(1, model.latent_dim)
            else:
                latent_values = np.asarray(latent_draw, dtype=np.float32).reshape(1, -1)
                latent_values = latent_values[:, :model.latent_dim]
                if latent_values.shape[1] < model.latent_dim:
                    latent_values = np.pad(latent_values, ((0, 0), (0, model.latent_dim - latent_values.shape[1])))
                latent = torch.as_tensor(latent_values)

            with torch.no_grad():
                raw_logits = model.logits(play_tensor, prior_tensor, latent).squeeze(0)
                if model.target_mode == "binary":
                    # Legacy binary heads emit 2 raw class logits consumed as
                    # softmax(...)[..., 1]; NGMM's binary head emits a single logit,
                    # so pad a zero column to keep every existing call site compatible.
                    raw_logits = torch.cat([torch.zeros_like(raw_logits), raw_logits], dim=-1)
                return raw_logits

        if isinstance(model, HierarchicalLatentGameModel):
            if latent_draw is not None:
                latent = np.asarray(latent_draw, dtype=np.float32).reshape(1, -1)
                if latent.shape[1] < model.latent_dim:
                    pad_width = model.latent_dim - latent.shape[1]
                    latent = np.pad(latent, ((0, 0), (0, pad_width)), mode="constant")
                elif latent.shape[1] > model.latent_dim:
                    latent = latent[:, : model.latent_dim]
                latent_tensor = torch.as_tensor(latent)
                if play_tensor.shape[0] > 1 and latent_tensor.shape[0] == 1:
                    latent_tensor = latent_tensor.repeat(play_tensor.shape[0], 1)
                with torch.no_grad():
                    return model.forward_with_z(play_tensor, latent_tensor)

            prior_processor_key = f"{model_key}_prior"
            if prior_processor_key not in self.processor:
                raise ValueError(f"Hierarchical model '{model_key}' is missing prior preprocessor '{prior_processor_key}'.")

            prior_values: Dict[str, object] = {}
            if base_values:
                prior_values.update(base_values)
            if not play_frame.empty:
                prior_values.update(play_frame.iloc[0].to_dict())
            prior_frame = pd.DataFrame([prior_values])
            prior_inp = self.transform_with_processor(prior_processor_key, prior_frame)
            prior_processor = self.processor.get(prior_processor_key)
            prior_columns = getattr(prior_processor, "feature_names_in_", None)
            if prior_columns is not None:
                prior_columns = [str(column) for column in prior_columns]
            self._record_nan_array(prior_processor_key, "post_transform", prior_inp, prior_columns)
            prior_inp = np.clip(np.nan_to_num(prior_inp, nan=0.0), -self.MODEL_INPUT_CLIP, self.MODEL_INPUT_CLIP).astype(np.float32)
            #print(prior_inp)
            prior_tensor = torch.as_tensor(prior_inp)
            with torch.no_grad():
                z = model.sample_z(prior_tensor)
                return model.forward_with_z(play_tensor, z)

        with torch.no_grad():
            logits = model(play_tensor)
        if latent_draw is not None and self.game_effects is not None and model_key in self.game_effects.models:
            # Plain flat/CORN heads: the latent draw is this team-game's logit shift.
            logits = logits + torch.as_tensor(np.asarray(latent_draw, dtype=np.float32))
        return logits

    def _get_model_shape(self, model: str) -> tuple[int, int]:
        input_dim = 0
        for group_name in self.column_dict[model]:
            if model == "target" and group_name in {"receiver_columns", "slot_stat_cols"}:
                input_dim += len(self.receiver_columns) * 11
            elif model == "rusher" and group_name in {"rusher_columns", "slot_stat_cols"}:
                input_dim += len(self.rusher_columns) * 11
            elif model == "time" and group_name == "play_cols":
                input_dim += len(getattr(self, group_name)) * 8
            elif model == "fumble" and group_name == "play_outcome":
                input_dim += len(getattr(self, group_name)) * 3
            elif group_name == "game_state_cols":
                input_dim += len(getattr(self, group_name)) + 4
            elif group_name == "position_cols":
                input_dim += len(getattr(self, group_name)) + 4
            else:
                input_dim += len(getattr(self, group_name))

        if model in {"target", "rusher"}:
            output_dim = 11
        elif model == "play":
            output_dim = 4
        elif model == "time":
            output_dim = len(self.bin_dict[model])
        elif model == "fumble":
            output_dim = 2
        else:
            output_dim = len(self.bin_dict[model]) if model in self.bin_dict else 2
        return input_dim, output_dim


class Player:
    def __init__(self, name: str, position: str, player_id: str, attributes: pd.DataFrame):
        self.name = name
        self.position = position
        self.player_id = player_id
        self.attributes = attributes
        self.stats = {
            "passing_yards": 0,
            "rushing_yards": 0,
            "rushing_attempts": 0,
            "receiving_yards": 0,
            "passing_touchdowns": 0,
            "receiving_touchdowns": 0,
            "rushing_touchdowns": 0,
            "interceptions": 0,
            "receptions": 0,
            "sacks": 0,
            "punts": 0,
            "punt_yards": 0,
            "pat_attempts": 0,
            "pat_made": 0,
            "pass_attempts": 0,
            "completions": 0,
            "air_yards": 0,
            "fumble_lost": 0,
        }

    def value(self, column: str, *default):
        """``attributes[column].values[0]`` (or the default when absent) from a cache built on first use;
        attributes are fixed once a game starts."""
        values = self.__dict__.get("_values")
        if values is None:
            values = self._values = {column: self.attributes[column].values[0] for column in self.attributes.columns}
        if default:
            return values.get(column, default[0])
        return values[column]


class Team:
    def __init__(self, name: str, players: List[Player], home: str, attributes: pd.Series, runtime: SimulationRuntime):
        self.name = name
        self.players = players
        self.score = 0
        self.home = home
        self.attributes = attributes
        self.runtime = runtime
        # z will be sampled fresh at the start of each Game.simulate_game()
        self.wr_embed: np.ndarray = np.zeros(len(runtime.target_embed_columns), dtype=np.float32)
        self.rb_embed: np.ndarray = np.zeros(len(runtime.running_embed_columns), dtype=np.float32)
        self.model_latents: Dict[str, np.ndarray] = {}
        self.target_latent: Optional[np.ndarray] = None
        self.rusher_latent: Optional[np.ndarray] = None
        self.stats = {
            "sack": 0,
            "pat_made": 0,
            "kick_made": 0,
            "fg 0-39": 0,
            "fg 40-49": 0,
            "fg 50-59": 0,
            "interception": 0,
            "pa": 0,
            "fumble_recovered": 0,
            "defensive_touchdown": 0,
        }

    def attribute_list(self, columns: List[str], strict: bool = False) -> List[object]:
        """``[attributes[c] for c in columns]`` (strict) or ``[attributes.get(c, nan) ...]``, cached per columns
        list while ``attributes`` is the same object; a fresh list each call."""
        cache = self.__dict__.get("_attribute_cache")
        if cache is None or cache[0] is not self.attributes:
            cache = self._attribute_cache = (self.attributes, {})
        key = (id(columns), strict)
        if key not in cache[1]:
            cache[1][key] = (columns, [self.attributes[c] for c in columns] if strict
                             else [self.attributes.get(c, np.nan) for c in columns])
        return list(cache[1][key][1])

    def get_random_player(self, position: str) -> Optional[Player]:
        filtered = [player for player in self.players if position in player.position]
        return random.choice(filtered) if filtered else None


class Play:
    def __init__(self, down: int, distance: int, yardline: int, game: "Game"):
        self.down = down
        self.distance = distance
        self.yardline = yardline
        self.description = ""
        self.yards_gained = 0
        self.turnover = False
        self.defensive_touchdown = False
        self.touchdown = False
        self.pat_attempt = False
        self.punt = False
        self.game = game
        self.field_goal_attempt = False
        self.type = "other"

    def simulate(self, offense: Team, defense: Team, is_pat: bool = False) -> None:
        runtime = self.game.runtime

        def _latent_for(model_key: str) -> Optional[np.ndarray]:
            return offense.model_latents.get(model_key)

        defense.stats["pa"] = offense.score - 7 * defense.stats["defensive_touchdown"]
        offense.stats["pa"] = defense.score - 7 * offense.stats["defensive_touchdown"]
        gc = [
            defense.score,
            self.game.down,
            3600 - (self.game.quarter - 1) * 900 - (900 - self.game.clock),
            offense.score,
            offense.home,
            100 - self.game.yardline,
            self.game.distance,
            self.game.dome,
            offense.score - defense.score,
        ]
        dc = defense.attribute_list(runtime.def_cols, strict=True)
        tc = offense.attribute_list(runtime.team_stats_cols, strict=True)
        pregame_c = offense.attribute_list(runtime.pregame_context_columns)
        qbc = offense.attribute_list(runtime.qb_context_columns)
        base_row = dict(zip(runtime.game_state_cols, gc))
        base_row.update(zip(runtime.team_stats_cols, offense.attribute_list(runtime.team_stats_cols)))
        base_row.update(zip(runtime.team_tendency_columns, offense.attribute_list(runtime.team_tendency_columns)))
        base_row.update(zip(runtime.qb_context_columns, offense.attribute_list(runtime.qb_context_columns)))
        base_row.update(zip(runtime.pregame_context_columns, offense.attribute_list(runtime.pregame_context_columns)))
        base_row.update(zip(runtime.def_cols, defense.attribute_list(runtime.def_cols)))
        play_pred_columns = gc.copy()
        play_pred_columns.extend(tc)
        play_pred_columns.extend(dc)
        play_pred_columns.extend(pregame_c)
        if "passer_columns" in runtime.column_dict.get("play", []):
            qb1 = next((player for player in offense.players if player.position == "QB1"), None)
            if qb1 is None:
                raise ValueError(f"No QB1 available for play selection on {offense.name}")
            play_pred_columns.extend(qb1.value(col) for col in runtime.passer_columns)
        cols: List[str] = []
        for collist in runtime.column_dict["play"]:
            cols.extend(getattr(runtime, collist))
        play = build_frame_from_values(play_pred_columns, cols)
        play_selection = F.softmax(
            runtime.forward_model_logits("play", play, base_row, latent_draw=_latent_for("play")),
            dim=1,
        ).detach().numpy()
        # Start from model probabilities, then apply lightweight game-rule overrides for
        # special-teams decisions. This keeps core model behavior while preventing
        # unrealistic early-down punts/field goals and improving 4th-down realism.
        play_type = np.random.choice(["kick", "pass", "punt", "run"], p=play_selection[0, :])
        if not is_pat:
            if self.down < 4 and play_type in {"kick", "punt"}:
                pass_prob = float(play_selection[0, 1])
                run_prob = float(play_selection[0, 3])
                denom = max(pass_prob + run_prob, 1e-9)
                play_type = np.random.choice(["pass", "run"], p=[pass_prob / denom, run_prob / denom])
            elif self.down == 4:
                score_diff = float(offense.score - defense.score)
                trailing_late = (self.game.quarter >= 4 and self.game.clock <= 300 and score_diff <= -4)
                fg_distance = float((100 - self.yardline) + 17)

                # Prefer FG tries in clear makeable range on 4th-and-not-short, except when
                # trailing late where offenses should stay aggressive.
                if fg_distance <= 54 and self.distance >= 2 and not trailing_late:
                    play_type = "kick"
                # Avoid unrealistic very long FG tries from poor field position.
                elif fg_distance >= 62 and self.yardline < 70 and self.distance > 1 and not trailing_late:
                    play_type = "punt"

        if not is_pat:
            self.game.diagnostics["play_calls"][play_type] += 1
            self.game._record_playcall_bucket(
                play_type=play_type,
                play_selection=play_selection,
                down=int(self.down),
                distance=int(self.distance),
                score_diff=float(offense.score - defense.score),
                quarter=int(self.game.quarter),
                clock=int(self.game.clock),
            )

        if is_pat or play_type == "kick":
            if is_pat:
                # Real nflverse PBP has `down = NaN` and `ydstogo = 0` on every
                # extra_point_attempt row (a PAT isn't a "down"), and the kick model's
                # down/ydstogo StandardScaler was fit on that real distribution (mean/scale
                # dominated by 4th-down field goals, e.g. down mean ~3.86 / scale ~0.54).
                # Feeding a fabricated down=1 here produces a z-score around -5.3 for every
                # single PAT in every simulation (a massive, permanent out-of-distribution
                # input the network never saw in that form during training, since NaN "down"
                # gets zero-filled AFTER scaling in training via fill_nans_with_zero, not
                # scaled from a raw value of 1). Use NaN for down (matches training exactly:
                # NaN survives the ColumnTransformer and is zero-filled post-transform by
                # forward_model_logits' np.nan_to_num, mirroring fill_nans_with_zero) and 0
                # for ydstogo (the real recorded value on PAT rows, not a fabricated 10).
                gc = [
                    defense.score,
                    np.nan,
                    3600 - (self.game.quarter - 1) * 900 - (900 - self.game.clock),
                    offense.score,
                    offense.home,
                    15,
                    0,
                    self.game.dome,
                    offense.score - defense.score,
                ]
            kick_columns = gc.copy()
            cols = []
            for collist in runtime.column_dict["kick"]:
                cols.extend(getattr(runtime, collist))
            kick = build_frame_from_values(kick_columns, cols)
            kick_prob = F.softmax(
                runtime.forward_model_logits("kick", kick, base_row, latent_draw=_latent_for("kick")),
                dim=1,
            ).detach().numpy()[0][1]
            good = random.random() < kick_prob
            if is_pat:
                self.game.diagnostics["kick_attempts"]["pat_attempt"] += 1
                if good:
                    self.game.diagnostics["kick_attempts"]["pat_made"] += 1
                    offense.stats["pat_made"] += 1
                    offense.score += 1
                    self.description = f"{offense.name} made the PAT"
                else:
                    self.description = f"{offense.name} missed the PAT"
                self.pat_attempt = True
            else:
                self.type = "field_goal_attempt"
                self.game.diagnostics["kick_attempts"]["fg_attempt"] += 1
                if good:
                    self.game.diagnostics["kick_attempts"]["fg_made"] += 1
                    offense.stats["kick_made"] += 1
                    if self.game.yardline > 79:
                        offense.stats["fg 0-39"] += 1
                    elif self.game.yardline > 69:
                        offense.stats["fg 40-49"] += 1
                    else:
                        offense.stats["fg 50-59"] += 1
                    offense.score += 3
                    self.description = f"{offense.name} made the kick"
                else:
                    self.description = f"{offense.name} missed the kick"
                    self.yards_gained -= 8
                self.field_goal_attempt = True
            return

        if play_type == "pass":
            qb = offense.get_random_player("QB")
            if qb is None:
                raise ValueError(f"No QB available for {offense.name}")
            passer_c = [qb.value(col) for col in runtime.passer_columns]
            qbc = [qb.value(col, np.nan) for col in runtime.qb_context_columns]
            for column, value in zip(runtime.qb_context_columns, qbc):
                if pd.isna(value):
                    value = offense.attributes.get(column, np.nan)
                base_row[column] = value
            ay_pred_columns = gc.copy()
            ay_pred_columns.extend(dc)
            ay_pred_columns.extend(passer_c)
            ay_pred_columns.extend(tc)
            ay_pred_columns.extend(pregame_c)
            ay_pred_columns.extend(qbc)
            cols = []
            for collist in runtime.column_dict["ay"]:
                cols.extend(getattr(runtime, collist))
            ay = runtime.with_goal_line_inputs("ay", build_frame_from_values(ay_pred_columns, cols), 100 - self.yardline)
            air_yards_prob = get_probs(
                runtime.forward_model_logits("ay", ay, base_row, latent_draw=_latent_for("ay")).detach().numpy()
            )
            air_yards = min(np.random.choice(runtime.bin_dict["ay"]["0"], p=air_yards_prob[0, :]), 100 - self.yardline)

            receiver_selection_columns = gc.copy()
            receiver_selection_columns.append(air_yards)
            receiver_selection_columns.extend(list(offense.wr_embed))
            target_row: Dict[str, object] = dict(zip(runtime.game_state_cols, gc))
            target_row["air_yards"] = air_yards
            for index, column in enumerate(runtime.target_embed_columns):
                target_row[column] = float(offense.wr_embed[index]) if index < len(offense.wr_embed) else 0.0
            filled_positions: List[str] = []
            for player in offense.players:
                for feature in runtime.receiver_columns:
                    target_row[f"{feature}_{player.position}"] = player.value(feature)
                filled_positions.append(player.position)
            for position in [slot for slot in SKILL_SLOT_ORDER if slot not in filled_positions]:
                for feature in runtime.receiver_columns:
                    target_row[f"{feature}_{position}"] = np.nan
            for column, value in zip(runtime.pregame_context_columns, pregame_c):
                target_row[column] = value
            for column, value in zip(runtime.qb_context_columns, qbc):
                target_row[column] = value
            target = PlayRow(target_row)
            target_logits = runtime.forward_model_logits("target", target, base_row, latent_draw=offense.target_latent)
            target = F.softmax(target_logits, dim=1).detach().numpy()
            wr = None
            while wr is None:
                target_slot = np.random.choice(SKILL_SLOT_ORDER, p=target[0, :])
                for player in offense.players:
                    if player.position == target_slot:
                        wr = player
                        break
            receiver_c = [wr.value(col) for col in runtime.receiver_columns]

            sack_pred_columns = gc.copy()
            sack_pred_columns.extend(dc)
            sack_pred_columns.extend(passer_c)
            sack_pred_columns.extend(tc)
            sack_pred_columns.extend(pregame_c)
            sack_pred_columns.extend(qbc)
            cols = []
            for collist in runtime.column_dict["sack"]:
                cols.extend(getattr(runtime, collist))
            sack = build_frame_from_values(sack_pred_columns, cols)
            sack_prob = F.softmax(
                runtime.forward_model_logits("sack", sack, base_row, latent_draw=_latent_for("sack")),
                dim=1,
            ).detach().numpy()[0][1]

            comp_columns = gc.copy()
            comp_columns.extend(dc)
            comp_columns.extend(passer_c)
            comp_columns.append(air_yards)
            comp_columns.extend(tc)
            comp_columns.extend(pregame_c)
            comp_columns.extend(qbc)
            cols = []
            for collist in runtime.column_dict["comp"]:
                cols.extend(getattr(runtime, collist))
            comp = build_frame_from_values(comp_columns, cols)
            #for col, val in comp.iloc[0].items():
                #print(offense.name, col, val)
            completion_prob = F.softmax(
                runtime.forward_model_logits("comp", comp, base_row, latent_draw=_latent_for("comp")),
                dim=1,
            ).detach().numpy()[0][1]
            interception_prob = F.softmax(
                runtime.forward_model_logits("int", comp, base_row, latent_draw=_latent_for("int")),
                dim=1,
            ).detach().numpy()[0][1]
            scramble_prob = 0.0
            if "scramble" in runtime.model_dict:
                scramble_prob = float(
                    F.softmax(
                        runtime.forward_model_logits("scramble", sack, base_row, latent_draw=_latent_for("scramble")),
                        dim=1,
                    ).detach().numpy()[0][1]
                )

            # Context-aware pass outcome rewrite: the model outputs are still primary, but we
            # reweight classes to better match observed completion/rush/TD environment without
            # post-hoc score calibration.
#            comp_mult = 1.00
##            sack_mult = 0.96
##            int_mult = 0.95
##            scramble_mult = 1.02
##            if self.yardline >= 80:
##                comp_mult += 0.03
##                sack_mult *= 0.97
##            if self.yardline >= 90:
##                comp_mult += 0.03
##                int_mult *= 0.95

##            sack_prob = max(sack_prob * sack_mult, 0.0)
##            completion_prob = max(completion_prob * comp_mult, 0.0)
##            interception_prob = max(interception_prob * int_mult, 0.0)
##            scramble_prob = max(scramble_prob * scramble_mult, 0.0)
            if runtime.pass_outcomes_given_attempt:
                throw_prob = max(1 - sack_prob - scramble_prob, 0)
                completion_prob *= throw_prob
                interception_prob *= throw_prob
            incomplete_prob = max(1 - sack_prob - completion_prob - interception_prob - scramble_prob, 0)
            probs = np.array([sack_prob, completion_prob, interception_prob, scramble_prob, incomplete_prob])
            probs = probs / probs.sum()
            outcome = np.random.choice([0, 1, 2, 3, 4], p=probs)

            if outcome == 1:
                self.type = "complete_pass"
                self.game.diagnostics["pass_outcomes"]["complete_pass"] += 1
                self.game.diagnostics["yardage"]["air"]["plays"] += 1
                self.game.diagnostics["yardage"]["air"]["yards"] += float(air_yards)
                yac_columns = gc.copy()
                yac_columns.extend(dc)
                yac_columns.extend(receiver_c)
                yac_columns.append(air_yards)
                yac_columns.extend(tc)
                yac_columns.extend(pregame_c)
                yac_columns.extend(qbc)
                cols = []
                for collist in runtime.column_dict["yac"]:
                    cols.extend(getattr(runtime, collist))
                yac_prob = runtime.with_goal_line_inputs("yac", build_frame_from_values(yac_columns, cols), 100 - self.yardline - air_yards)
                yac_prob = get_probs(
                    runtime.forward_model_logits("yac", yac_prob, base_row, latent_draw=_latent_for("yac")).detach().numpy()
                )
                yac_yards = np.random.choice(runtime.bin_dict["yac"]["0"], p=yac_prob[0, :])
                self.game.diagnostics["yardage"]["yac"]["plays"] += 1
                self.game.diagnostics["yardage"]["yac"]["yards"] += float(yac_yards)
                qb.stats["pass_attempts"] += 1
                qb.stats["completions"] += 1
                yards = air_yards + yac_yards
                if yards + self.yardline >= 100:
                    yards = 100 - self.yardline
                self.yards_gained = yards
                wr.stats["receiving_yards"] += yards
                wr.stats["receptions"] += 1
                qb.stats["passing_yards"] += yards
                qb.stats["air_yards"] += air_yards
                self.game.diagnostics["yardage"]["complete_pass"]["plays"] += 1
                self.game.diagnostics["yardage"]["complete_pass"]["yards"] += float(yards)
                self.description = f"{qb.name} completed a pass to {wr.name} for {yards} yards. Now at the {self.yardline + yards} yardline"

                fumble_columns = gc.copy()
                fumble_columns.extend(tc)
                fumble_columns.extend(dc)
                fumble_columns.append("completion")
                fumble_columns.extend(pregame_c)
                cols = []
                for collist in runtime.column_dict["fumble"]:
                    cols.extend(getattr(runtime, collist))
                if yards + self.yardline < 100:
                    fumble = build_frame_from_values(fumble_columns, cols)
                    fumble_prob = (
                        F.softmax(
                            runtime.forward_model_logits("fumble", fumble, base_row, latent_draw=_latent_for("fumble")),
                            dim=1,
                        ).detach().numpy()[0][1]
                        / 2
                    )
                    if random.random() < fumble_prob:
                        self.turnover = True
                        self.description = (
                            f"{qb.name} completed a pass to {wr.name} for {yards} yards. Now at the {self.yardline + yards} yardline. "
                            f"{wr.name} fumbled and it was recovered by the defense"
                        )
                        wr.stats["fumble_lost"] += 1
                        defense.stats["fumble_recovered"] += 1
                        self._apply_return_yards(gc, defense)

            elif outcome == 2:
                self.type = "interception"
                self.game.diagnostics["pass_outcomes"]["interception"] += 1
                qb.stats["pass_attempts"] += 1
                defense.stats["interception"] += 1
                self.game.diagnostics["yardage"]["air"]["plays"] += 1
                self.game.diagnostics["yardage"]["air"]["yards"] += float(air_yards)
                if air_yards + self.yardline >= 100:
                    air_yards = 80 - (air_yards + self.yardline)
                self.yards_gained = air_yards
                self.turnover = True
                qb.stats["interceptions"] += 1
                qb.stats["air_yards"] += air_yards
                self.description = f"{qb.name} threw an interception!"
                self._apply_return_yards(gc, defense)

            elif outcome == 0:
                self.type = "sack"
                self.game.diagnostics["pass_outcomes"]["sack"] += 1
                qb.stats["pass_attempts"] += 1
                qb.stats["sacks"] += 1
                defense.stats["sack"] += 1
                if "sack_yards" in runtime.model_dict:
                    sack_yards_prob = get_probs(
                        runtime.forward_model_logits("sack_yards", sack, base_row, latent_draw=_latent_for("sack_yards")).detach().numpy()
                    )
                    self.yards_gained = int(np.random.choice(runtime.bin_dict["sack_yards"]["0"], p=sack_yards_prob[0, :]))
                else:
                    self.yards_gained = np.random.randint(-10, -1)
                self.description = f"{qb.name} was sacked for {self.yards_gained} yards"
                fumble_columns = gc.copy()
                fumble_columns.extend(tc)
                fumble_columns.extend(dc)
                fumble_columns.append("sack")
                fumble_columns.extend(pregame_c)
                cols = []
                for collist in runtime.column_dict["fumble"]:
                    cols.extend(getattr(runtime, collist))
                fumble = build_frame_from_values(fumble_columns, cols)
                fumble_prob = (
                    F.softmax(
                        runtime.forward_model_logits("fumble", fumble, base_row, latent_draw=_latent_for("fumble")),
                        dim=1,
                    ).detach().numpy()[0][1]
                    / 2
                )
                if random.random() < fumble_prob:
                    self.turnover = True
                    self.description = f"{qb.name} was sacked for {self.yards_gained} yards and fumbled. It was recovered by the defense"
                    qb.stats["fumble_lost"] += 1
                    defense.stats["fumble_recovered"] += 1
                    self._apply_return_yards(gc, defense)

            elif outcome == 3:
                self.type = "rush"
                self.game.diagnostics["pass_outcomes"]["scramble"] += 1
                qb.stats["rushing_attempts"] += 1
                rb = qb
                scramble_columns = gc.copy()
                scramble_columns.extend(dc)
                scramble_columns.extend([qb.value(col) for col in runtime.rusher_columns])
                scramble_columns.extend(tc)
                scramble_columns.extend(pregame_c)
                cols = []
                for collist in runtime.column_dict["scramble_yards"]:
                    resolved = "rusher_columns" if collist == "rusher_cols" else collist
                    cols.extend(getattr(runtime, resolved))
                scramble_frame = runtime.with_goal_line_inputs("scramble_yards", build_frame_from_values(scramble_columns, cols), 100 - self.yardline)
                scramble_yards_prob = get_probs(
                    runtime.forward_model_logits("scramble_yards", scramble_frame, base_row, latent_draw=_latent_for("scramble_yards")).detach().numpy()
                )
                scramble_yards = min(np.random.choice(runtime.bin_dict["scramble_yards"]["0"], p=scramble_yards_prob[0, :]), 100 - self.yardline + 1)
                self.yards_gained = scramble_yards
                qb.stats["rushing_yards"] += scramble_yards
                self.game.diagnostics["yardage"]["scramble"]["plays"] += 1
                self.game.diagnostics["yardage"]["scramble"]["yards"] += float(scramble_yards)
                self.description = f"{qb.name} scrambled for {scramble_yards} yards. Now at the {self.yardline + scramble_yards} yardline"

            else:
                self.type = "incomplete_pass"
                self.game.diagnostics["pass_outcomes"]["incomplete_pass"] += 1
                qb.stats["pass_attempts"] += 1
                self.game.diagnostics["yardage"]["air"]["plays"] += 1
                self.game.diagnostics["yardage"]["air"]["yards"] += float(air_yards)
                self.description = f"{qb.name} threw an incomplete pass"

        elif play_type == "run":
            self.type = "rush"
            rusher_row: Dict[str, object] = dict(zip(runtime.game_state_cols, gc))
            for index, column in enumerate(runtime.running_embed_columns):
                rusher_row[column] = float(offense.rb_embed[index]) if index < len(offense.rb_embed) else 0.0
            filled_positions = []
            for player in offense.players:
                for feature in runtime.rusher_columns:
                    rusher_row[f"{feature}_{player.position}"] = player.value(feature)
                filled_positions.append(player.position)
            for position in [slot for slot in SKILL_SLOT_ORDER if slot not in filled_positions]:
                for feature in runtime.rusher_columns:
                    rusher_row[f"{feature}_{position}"] = np.nan
            for column, value in zip(runtime.pregame_context_columns, pregame_c):
                rusher_row[column] = value
            rusher = PlayRow(rusher_row)
            rusher_logits = runtime.forward_model_logits("rusher", rusher, base_row, latent_draw=offense.rusher_latent)
            rusher = F.softmax(rusher_logits, dim=1).detach().numpy()
            rb = None
            while rb is None:
                target_slot = np.random.choice(SKILL_SLOT_ORDER, p=rusher[0, :])
                for player in offense.players:
                    if player.position == target_slot:
                        rb = player
                        break
            rusher_c = [rb.value(col) for col in runtime.rusher_columns]
            rush_columns = gc.copy()
            rush_columns.extend(dc)
            rush_columns.extend(rusher_c)
            rush_columns.extend(tc)
            ##rush_columns.append(rb.position[0:2])
            rush_columns.extend(pregame_c)
            cols = []
            for collist in runtime.column_dict["rush"]:
                resolved = "rusher_columns" if collist == "rusher_cols" else collist
                cols.extend(getattr(runtime, resolved))
            rb.stats["rushing_attempts"] += 1
            rush = runtime.with_goal_line_inputs("rush", build_frame_from_values(rush_columns, cols), 100 - self.yardline)
            rush_prob = get_probs(
                runtime.forward_model_logits("rush", rush, base_row, latent_draw=_latent_for("rush")).detach().numpy()
            )
            rush_yards = min(np.random.choice(runtime.bin_dict["rush"]["0"], p=rush_prob[0, :]), 100 - self.yardline + 1)
  ##          if rush_yards >= 0:
  ##              rush_boost = 1.04 if self.yardline < 80 else 1.02
  ##              rush_yards = int(np.round(float(rush_yards) * rush_boost))
##                if rush_yards == 0 and random.random() < 0.04:
##                    rush_yards = 1
            rush_yards = int(np.clip(rush_yards, -10, 100 - self.yardline + 1))
            self.yards_gained = rush_yards
            rb.stats["rushing_yards"] += rush_yards
            self.game.diagnostics["yardage"]["rush"]["plays"] += 1
            self.game.diagnostics["yardage"]["rush"]["yards"] += float(rush_yards)
            self.description = f"{rb.name} ran for {rush_yards} yards. Now at the {self.yardline + rush_yards} yardline"

            if rush_yards + self.yardline < 100:
                fumble_columns = gc.copy()
                fumble_columns.extend(tc)
                fumble_columns.extend(dc)
                fumble_columns.append("run")
                fumble_columns.extend(pregame_c)
                cols = []
                for collist in runtime.column_dict["fumble"]:
                    cols.extend(getattr(runtime, collist))
                fumble = build_frame_from_values(fumble_columns, cols)
                fumble_prob = (
                    F.softmax(
                        runtime.forward_model_logits("fumble", fumble, base_row, latent_draw=_latent_for("fumble")),
                        dim=1,
                    ).detach().numpy()[0][1]
                    / 2
                )
                if random.random() < fumble_prob:
                    self.turnover = True
                    self.description = f"{rb.name} ran for {rush_yards} yards. Now at the {self.yardline + rush_yards} yardline. He fumbled and the defense recovered"
                    rb.stats["fumble_lost"] += 1
                    defense.stats["fumble_recovered"] += 1
                    self._apply_return_yards(gc, defense)

        if self.yardline + self.yards_gained >= 100 and not self.turnover:
            self.touchdown = True
            self.yards_gained = 100 - self.yardline
            self.description += " — TOUCHDOWN!"
            offense.score += 6
            if self.type == "complete_pass":
                qb.stats["passing_touchdowns"] += 1
                wr.stats["receiving_touchdowns"] += 1
            elif self.type == "rush":
                rb.stats["rushing_touchdowns"] += 1

        if play_type == "punt" and self.game.split_punts:
            self.type = "punt"
            self.punt = True
            punter = offense.get_random_player("P")
            values = self.game.special_teams_values(offense, defense)
            gross = self.game.sample_special_teams("punt_distance", gc, values, offense)
            landing = self.yardline + gross
            if landing >= 100:
                # Touchback: the receiving team starts at its 20.
                self.yards_gained = 80 - self.yardline
            else:
                landing_100 = 100 - landing
                returned = self.game.sample_special_teams(
                    "punt_return", gc, {**values, "kick_distance": gross, "punt_landing_100": landing_100}, offense)
                returned = max(returned, 1 - landing_100)
                self.yards_gained = gross - returned
                if landing_100 + returned >= 100:
                    self.defensive_touchdown = True
                    defense.stats["defensive_touchdown"] += 1
                    self.game.diagnostics["special_teams"]["punt_return_tds"] += 1
            self.game.diagnostics["yardage"]["punt"]["plays"] += 1
            self.game.diagnostics["yardage"]["punt"]["yards"] += float(self.yards_gained)
            if punter:
                punter.stats["punts"] += 1
                punter.stats["punt_yards"] += gross
            self.description = f"{punter.name if punter else 'Punter'} punts {gross:.0f} yards, net {self.yards_gained:.0f}"
        elif play_type == "punt":
            self.type = "punt"
            punter = offense.get_random_player("P")
            punt_columns = gc.copy()
            cols = []
            for collist in runtime.column_dict["punt"]:
                cols.extend(getattr(runtime, collist))
            punt = build_frame_from_values(punt_columns, cols)
            punt_prob = get_probs(
                runtime.forward_model_logits("punt", punt, base_row, latent_draw=_latent_for("punt")).detach().numpy()
            )
            punt_yards = np.random.choice(runtime.bin_dict["punt"]["0"], p=punt_prob[0, :])
            self.yards_gained = punt_yards
            if punt_yards + self.yardline >= 100:
                self.yards_gained = 80 - self.yardline
            if punt_yards + self.yardline <= 0:
                self.defensive_touchdown = True
            self.punt = True
            self.game.diagnostics["yardage"]["punt"]["plays"] += 1
            self.game.diagnostics["yardage"]["punt"]["yards"] += float(self.yards_gained)
            if punter:
                punter.stats["punts"] += 1
                punter.stats["punt_yards"] += punt_yards
            self.description = f"{punter.name if punter else 'Punter'} punts for {self.yards_gained} yards"

    def _apply_return_yards(self, game_context: List[object], defense: Team) -> None:
        runtime = self.game.runtime
        cols: List[str] = []
        for collist in runtime.column_dict["def_ret"]:
            cols.extend(getattr(runtime, collist))
        ret_values = game_context.copy()
        ret_values.extend(self.game.possession.attribute_list(runtime.pregame_context_columns))
        ret_prob = build_frame_from_values(ret_values, cols)
        ret_latent = self.game.possession.model_latents.get("def_ret")
        ret_prob = get_probs(
            runtime.forward_model_logits(
                "def_ret",
                ret_prob,
                dict(zip(runtime.game_state_cols, game_context)),
                latent_draw=ret_latent,
            ).detach().numpy()
        )
        ret_yards = np.random.choice(runtime.bin_dict["def_ret"]["0"], p=ret_prob[0, :])
        self.yards_gained -= ret_yards
        if self.yards_gained + self.yardline <= 0:
            self.defensive_touchdown = True
            defense.stats["defensive_touchdown"] += 1


class Game:
    def __init__(self, team1: Team, team2: Team, dome: str, runtime: SimulationRuntime):
        self.team1 = team1
        self.team2 = team2
        self.runtime = runtime
        for player in team1.players:
            for stat in player.stats:
                player.stats[stat] = 0
        for player in team2.players:
            for stat in player.stats:
                player.stats[stat] = 0
        if random.random() < 0.5:
            self.possession = team1
            self.defense = team2
        else:
            self.possession = team2
            self.defense = team1
        self.yardline = 30
        self.down = 1
        self.distance = 10
        self.plays: List[Play] = []
        self.play_number = 0
        self.quarter = 1
        self.clock = 900
        self.dome = dome
        # Bundles with the special-teams models simulate kickoffs (opening, after scores, halftime)
        # and split punts into distance and return; older bundles keep the fixed rules.
        self.kickoff_model = "kickoff" in runtime.model_dict
        self.split_punts = {"punt_distance", "punt_return"} <= set(runtime.model_dict)
        self.halftime_pending = False
        self.opening_receiver: Optional[Team] = None
        self.diagnostics = {
            "play_calls": {"pass": 0, "run": 0, "punt": 0, "kick": 0},
            "pass_outcomes": {"sack": 0, "complete_pass": 0, "interception": 0, "scramble": 0, "incomplete_pass": 0},
            "play_call_buckets": {},
            "penalty_no_play_count": 0,
            "penalty_on_play_count": 0,
            "penalty_first_down_count": 0,
            "penalty_yards_sum": 0.0,
            "penalty_yards_abs_sum": 0.0,
            "kick_attempts": {"fg_attempt": 0, "fg_made": 0, "pat_attempt": 0, "pat_made": 0},
            "special_teams": {"kickoffs": 0, "kickoff_start_sum": 0.0, "kickoff_return_tds": 0, "punt_return_tds": 0},
            "yardage": {
                "rush": {"plays": 0, "yards": 0.0},
                "complete_pass": {"plays": 0, "yards": 0.0},
                "scramble": {"plays": 0, "yards": 0.0},
                "air": {"plays": 0, "yards": 0.0},
                "yac": {"plays": 0, "yards": 0.0},
                "punt": {"plays": 0, "yards": 0.0},
                "time": {"plays": 0, "seconds": 0.0},
            },
        }

    @staticmethod
    def _distance_bucket(distance: int) -> str:
        if distance <= 3:
            return "short"
        if distance <= 7:
            return "medium"
        return "long"

    @staticmethod
    def _score_bucket(score_diff: float) -> str:
        if score_diff <= -9:
            return "trailing_big"
        if score_diff <= -1:
            return "trailing"
        if score_diff >= 9:
            return "leading_big"
        if score_diff >= 1:
            return "leading"
        return "tied"

    @staticmethod
    def _clock_bucket(quarter: int, clock: int) -> str:
        if quarter >= 4 and clock <= 300:
            return "q4_late"
        if quarter >= 3:
            return "second_half"
        if quarter == 2 and clock <= 300:
            return "q2_late"
        return "first_half"

    def _playcall_bucket_key(self, down: int, distance: int, score_diff: float, quarter: int, clock: int) -> str:
        down_int = int(np.clip(int(down), 1, 4))
        return (
            f"down_{down_int}|dist_{self._distance_bucket(int(distance))}"
            f"|score_{self._score_bucket(float(score_diff))}|clock_{self._clock_bucket(int(quarter), int(clock))}"
        )

    def _record_playcall_bucket(
        self,
        play_type: str,
        play_selection: np.ndarray,
        down: int,
        distance: int,
        score_diff: float,
        quarter: int,
        clock: int,
    ) -> None:
        key = self._playcall_bucket_key(down, distance, score_diff, quarter, clock)
        bucket = self.diagnostics["play_call_buckets"].setdefault(
            key,
            {
                "snaps": 0.0,
                "pass_calls": 0.0,
                "run_calls": 0.0,
                "punt_calls": 0.0,
                "kick_calls": 0.0,
                "pass_prob_sum": 0.0,
                "run_prob_sum": 0.0,
            },
        )
        bucket["snaps"] += 1.0
        bucket["pass_prob_sum"] += float(play_selection[0, 1])
        bucket["run_prob_sum"] += float(play_selection[0, 3])
        if play_type == "pass":
            bucket["pass_calls"] += 1.0
        elif play_type == "run":
            bucket["run_calls"] += 1.0
        elif play_type == "punt":
            bucket["punt_calls"] += 1.0
        elif play_type == "kick":
            bucket["kick_calls"] += 1.0

    def diagnostics_row(self) -> Dict[str, object]:
        all_players = self.team1.players + self.team2.players
        pass_attempts = float(
            sum(
                player.stats.get("pass_attempts", 0)
                for player in all_players
                if str(player.position).startswith("QB")
            )
        )
        rushing_attempts = float(sum(player.stats.get("rushing_attempts", 0) for player in all_players))
        pass_outcomes_total = float(sum(self.diagnostics["pass_outcomes"].values()))
        # By design this is NOT expected to be 0: `pass_outcomes_total` counts every play the
        # selector routed to "pass" (sack/complete/interception/scramble/incomplete), but real
        # NFL stat convention excludes scrambles from `pass_attempts` (a scramble is a QB run,
        # not a passing attempt). So pass_attempt_delta should equal -scramble_count once sacks
        # are correctly counted (see qb.stats["pass_attempts"] += 1 in the sack branch above).
        pass_attempt_delta = float(pass_attempts - pass_outcomes_total)
        return {
            "play_call_pass": float(self.diagnostics["play_calls"]["pass"]),
            "play_call_run": float(self.diagnostics["play_calls"]["run"]),
            "play_call_punt": float(self.diagnostics["play_calls"]["punt"]),
            "play_call_kick": float(self.diagnostics["play_calls"]["kick"]),
            "pass_outcome_sack": float(self.diagnostics["pass_outcomes"]["sack"]),
            "pass_outcome_complete": float(self.diagnostics["pass_outcomes"]["complete_pass"]),
            "pass_outcome_interception": float(self.diagnostics["pass_outcomes"]["interception"]),
            "pass_outcome_scramble": float(self.diagnostics["pass_outcomes"]["scramble"]),
            "pass_outcome_incomplete": float(self.diagnostics["pass_outcomes"]["incomplete_pass"]),
            "pass_outcomes_total": pass_outcomes_total,
            "sim_pass_attempts": pass_attempts,
            "sim_rushing_attempts": rushing_attempts,
            "pass_attempt_delta": pass_attempt_delta,
            "penalty_no_play_count": float(self.diagnostics["penalty_no_play_count"]),
            "penalty_on_play_count": float(self.diagnostics["penalty_on_play_count"]),
            "penalty_first_down_count": float(self.diagnostics["penalty_first_down_count"]),
            "penalty_yards_sum": float(self.diagnostics["penalty_yards_sum"]),
            "penalty_yards_abs_sum": float(self.diagnostics["penalty_yards_abs_sum"]),
            "fg_attempts": float(self.diagnostics["kick_attempts"]["fg_attempt"]),
            "fg_made": float(self.diagnostics["kick_attempts"]["fg_made"]),
            "pat_attempts": float(self.diagnostics["kick_attempts"]["pat_attempt"]),
            "pat_made": float(self.diagnostics["kick_attempts"]["pat_made"]),
            "rush_plays": float(self.diagnostics["yardage"]["rush"]["plays"]),
            "rush_yards_sum": float(self.diagnostics["yardage"]["rush"]["yards"]),
            "complete_pass_plays": float(self.diagnostics["yardage"]["complete_pass"]["plays"]),
            "complete_pass_yards_sum": float(self.diagnostics["yardage"]["complete_pass"]["yards"]),
            "scramble_plays": float(self.diagnostics["yardage"]["scramble"]["plays"]),
            "scramble_yards_sum": float(self.diagnostics["yardage"]["scramble"]["yards"]),
            "air_plays": float(self.diagnostics["yardage"]["air"]["plays"]),
            "air_yards_sum": float(self.diagnostics["yardage"]["air"]["yards"]),
            "yac_plays": float(self.diagnostics["yardage"]["yac"]["plays"]),
            "yac_yards_sum": float(self.diagnostics["yardage"]["yac"]["yards"]),
            "punt_plays": float(self.diagnostics["yardage"]["punt"]["plays"]),
            "punt_yards_sum": float(self.diagnostics["yardage"]["punt"]["yards"]),
            "time_plays": float(self.diagnostics["yardage"]["time"]["plays"]),
            "time_elapsed_sum": float(self.diagnostics["yardage"]["time"]["seconds"]),
            **{key: float(value) for key, value in self.diagnostics["special_teams"].items()},
        }

    def diagnostics_bucket_rows(self) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        for bucket_key, values in self.diagnostics["play_call_buckets"].items():
            snaps = float(values.get("snaps", 0.0))
            pass_calls = float(values.get("pass_calls", 0.0))
            run_calls = float(values.get("run_calls", 0.0))
            off_snaps = pass_calls + run_calls
            rows.append(
                {
                    "bucket_key": str(bucket_key),
                    "snaps": snaps,
                    "off_snaps": off_snaps,
                    "pass_calls": pass_calls,
                    "run_calls": run_calls,
                    "punt_calls": float(values.get("punt_calls", 0.0)),
                    "kick_calls": float(values.get("kick_calls", 0.0)),
                    "pass_rate": float(pass_calls / off_snaps) if off_snaps > 0 else np.nan,
                    "pass_prob_mean": float(values.get("pass_prob_sum", 0.0) / snaps) if snaps > 0 else np.nan,
                    "run_prob_mean": float(values.get("run_prob_sum", 0.0) / snaps) if snaps > 0 else np.nan,
                    "pass_prob_sum": float(values.get("pass_prob_sum", 0.0)),
                    "run_prob_sum": float(values.get("run_prob_sum", 0.0)),
                }
            )
        return rows

    def _normalize_down_distance(self) -> None:
        """Keep down/distance physically valid for current field position.

        Distance-to-go cannot exceed the yards remaining to the goal line.
        Example: at yardline 89 (opponent 11), max distance is 11.
        """
        # Scrimmage snaps should not start at or beyond the goal line in this coordinate
        # system (100 == score boundary during a play), so cap pre-snap state at 99.
        self.yardline = float(np.clip(self.yardline, 0.0, 99.0))
        self.down = int(np.clip(int(self.down), 1, 4))
        max_to_goal = max(1.0, 100.0 - self.yardline)
        self.distance = float(np.clip(float(self.distance), 1.0, max_to_goal))

    def switch_possession(self, yardline: int = 70) -> None:
        self.possession, self.defense = self.defense, self.possession
        self.yardline = 100 - yardline
        self.down = 1
        self.distance = 10
        self._normalize_down_distance()

    def special_teams_values(self, kicking: Team, receiving: Team) -> Dict[str, float]:
        """kick_/ret_ special-teams features for a punt or kickoff (missing team columns are league average, 0)."""
        values: Dict[str, float] = {}
        for column in self.runtime.special_teams_team_columns:
            for role, team in (("kick", kicking), ("ret", receiving)):
                value = pd.to_numeric(team.attributes.get(column, 0.0), errors="coerce")
                values[f"{role}_{column}"] = float(value) if np.isfinite(value) else 0.0
        values["st_ko_rule_2024"] = values["kick_st_ko_rule_2024"]
        values["st_ko_rule_2025"] = values["kick_st_ko_rule_2025"]
        return values

    def sample_special_teams(self, model: str, state: List[object], values: Dict[str, float], team: Team) -> float:
        """Draw one yardage from a special-teams CORN model; ``team`` is the posteam whose game effects apply."""
        runtime = self.runtime
        base_row = dict(zip(runtime.game_state_cols, state))
        row = {**base_row, **values}
        columns = [column for group in runtime.column_dict[model] for column in getattr(runtime, group)]
        frame = PlayRow((column, row.get(column, np.nan)) for column in columns)
        logits = runtime.forward_model_logits(model, frame, base_row, latent_draw=team.model_latents.get(model))
        prob = get_probs(logits.detach().numpy())
        return float(np.random.choice(runtime.bin_dict[model]["0"], p=prob[0, :]))

    def kickoff(self, kicking: Team) -> None:
        """Kick off to the other team, which starts where the kickoff model puts it; a return TD scores 7 and it kicks back."""
        while True:
            receiving = self.team2 if kicking is self.team1 else self.team1
            state = [kicking.score, 1, 3600 - (self.quarter - 1) * 900 - (900 - self.clock), receiving.score,
                     receiving.home, 35, 10, self.dome, receiving.score - kicking.score]
            start = self.sample_special_teams("kickoff", state, self.special_teams_values(kicking, receiving), receiving)
            self.diagnostics["special_teams"]["kickoffs"] += 1
            self.diagnostics["special_teams"]["kickoff_start_sum"] += min(start, 100.0)
            self.possession, self.defense = receiving, kicking
            if start >= 100:
                receiving.score += 7
                receiving.stats["defensive_touchdown"] += 1
                self.diagnostics["special_teams"]["kickoff_return_tds"] += 1
                kicking = receiving
                continue
            self.yardline = float(np.clip(start, 1, 99))
            self.down = 1
            self.distance = 10
            self._normalize_down_distance()
            return

    def _sample_post_score_switch_yardline(self) -> int:
        """Sample receiving field position after kickoff (expressed in the current
        possession coordinate system expected by switch_possession).

        Historically we always started drives at the 30 (switch_yardline=70), which
        removes return variance and can bias drive outcomes. This introduces a simple,
        rule-based kickoff distribution while keeping touchbacks as the dominant mode.
        """
        # Touchback-heavy environment: most kickoffs start at the 30.
        if random.random() < 0.68:
            receiving_start = 30
        else:
            # Returned kicks: broad but realistic own-yardline range.
            receiving_start = int(np.clip(np.round(np.random.normal(loc=28.0, scale=5.5)), 15, 45))
        return int(100 - receiving_start)

    def tick_clock(self, play_type: str) -> None:
        # Dead-ball pre-snap penalties should burn less clock than full snaps.
#        if play_type == "no_play_penalty":
#            elapsed = 0
#            self.clock -= elapsed
#            if self.clock <= 0:
#                self.advance_quarter()
#            return

        gc = [
            self.defense.score,
            self.down,
            3600 - (self.quarter - 1) * 900 - (900 - self.clock),
            self.possession.score,
            self.possession.home,
            100 - self.yardline,
            self.distance,
            self.dome,
            self.possession.score - self.defense.score,
        ]
        tick_cols = gc.copy()
        tick_cols.append(play_type)
        tick_cols.extend(self.possession.attribute_list(self.runtime.pregame_context_columns))
        cols: List[str] = []
        for collist in self.runtime.column_dict["time"]:
            cols.extend(getattr(self.runtime, collist))
        time_frame = build_frame_from_values(tick_cols, cols)
        base_row = dict(zip(self.runtime.game_state_cols, gc))
        base_row.update(zip(self.runtime.team_stats_cols, self.possession.attribute_list(self.runtime.team_stats_cols)))
        base_row.update(zip(self.runtime.team_tendency_columns, self.possession.attribute_list(self.runtime.team_tendency_columns)))
        base_row.update(zip(self.runtime.qb_context_columns, self.possession.attribute_list(self.runtime.qb_context_columns)))
        base_row.update(zip(self.runtime.pregame_context_columns, self.possession.attribute_list(self.runtime.pregame_context_columns)))
        base_row.update(zip(self.runtime.def_cols, self.defense.attribute_list(self.runtime.def_cols)))
        time_latent = self.possession.model_latents.get("time")
        time_prob = get_probs(
            self.runtime.forward_model_logits("time", time_frame, base_row, latent_draw=time_latent).detach().numpy()
        )
        time_elapsed = int(np.random.choice(self.runtime.bin_dict["time"]["0"], p=time_prob[0, :]))
        if time_elapsed < 1:
            time_elapsed = 0
        self.diagnostics["yardage"]["time"]["plays"] += 1
        self.diagnostics["yardage"]["time"]["seconds"] += float(time_elapsed)
        self.clock -= time_elapsed
        if self.clock <= 0:
            self.advance_quarter()

    def advance_quarter(self) -> None:
        self.quarter += 1
        if self.quarter == 3 and self.kickoff_model:
            self.halftime_pending = True
        if self.quarter <= 4:
            self.clock = 900
        else:
            self.clock = 0

    def _penalty_game_state_row(self, extra: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        gc = [
            self.defense.score,
            self.down,
            3600 - (self.quarter - 1) * 900 - (900 - self.clock),
            self.possession.score,
            self.possession.home,
            100 - self.yardline,
            self.distance,
            self.dome,
            self.possession.score - self.defense.score,
        ]
        row: Dict[str, object] = dict(zip(self.runtime.game_state_cols, gc))
        row.update(zip(self.runtime.team_stats_cols, self.possession.attribute_list(self.runtime.team_stats_cols)))
        row.update(zip(self.runtime.def_cols, self.defense.attribute_list(self.runtime.def_cols)))
        row.update(zip(self.runtime.pregame_context_columns, self.possession.attribute_list(self.runtime.pregame_context_columns)))
        if extra:
            row.update(extra)
        return row

    def _sample_penalty_no_play(self) -> bool:
        """Probability that this down opportunity is a standalone penalty instead of a play."""
        if "penalty_no_play" not in self.runtime.model_dict:
            return False
        row = self._penalty_game_state_row()
        frame = PlayRow(row)
        prob = F.softmax(
            self.runtime.forward_model_logits("penalty_no_play", frame, row),
            dim=1,
        ).detach().numpy()[0][1]
        return random.random() < float(prob)

    def _sample_penalty_on_play(self, play_type_cat: str) -> bool:
        """Probability that an accepted penalty also occurs on top of the play just simulated."""
        if "penalty_on_play" not in self.runtime.model_dict:
            return False
        row = self._penalty_game_state_row({"play_type_cat": play_type_cat})
        frame = PlayRow(row)
        prob = F.softmax(
            self.runtime.forward_model_logits("penalty_on_play", frame, row),
            dim=1,
        ).detach().numpy()[0][1]
        return random.random() < float(prob)

    def _sample_penalty_outcome(self, penalty_context: str, play_type_cat: str) -> "tuple[float, bool]":
        """Shared outcome sampling for both the no-play and on-play penalty cases: returns
        (signed offense-relative yards, whether the offense gets an automatic first down)."""
        row = self._penalty_game_state_row({
            "penalty_context": penalty_context,
            "play_type_cat": play_type_cat,
        })
        frame = PlayRow(row)
        signed_yards = 0.0
        if "penalty_yards" in self.runtime.model_dict and "penalty_yards" in self.runtime.bin_dict:
            yards_prob = get_probs(
                self.runtime.forward_model_logits("penalty_yards", frame, row).detach().numpy()
            )
            signed_yards = float(np.random.choice(self.runtime.bin_dict["penalty_yards"]["0"], p=yards_prob[0, :]))
        auto_first_down = False
        if "penalty_first_down" in self.runtime.model_dict:
            fd_prob = F.softmax(
                self.runtime.forward_model_logits("penalty_first_down", frame, row),
                dim=1,
            ).detach().numpy()[0][1]
            auto_first_down = random.random() < float(fd_prob)
        # Automatic first downs should only be possible when the penalty benefits
        # the offense (defensive foul in this signed-yard convention).
        if auto_first_down and signed_yards <= 0:
            auto_first_down = False
        return signed_yards, auto_first_down

    def _format_play_line(self, play: "Play", quarter: int, clock: float) -> str:
        minutes, seconds = divmod(max(int(clock), 0), 60)
        down_labels = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}
        down_label = down_labels.get(play.down, f"{play.down}th")
        score_line = f"{self.team1.name} {self.team1.score} - {self.team2.name} {self.team2.score}"
        return (
            f"Q{quarter} {minutes:02d}:{seconds:02d} | {self.possession.name} ball | "
            f"{down_label} & {play.distance} at {play.yardline:.0f} | {play.description} | {score_line}"
        )

    @staticmethod
    def _down_label(down: int, distance: float) -> str:
        down_labels = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}
        return f"{down_labels.get(down, f'{down}th')} & {distance:.0f}"

    def _format_penalty_line(
        self,
        quarter: int,
        clock: float,
        penalty_yards: float,
        auto_first_down: bool,
        pre_down: int,
        pre_distance: float,
        pre_yardline: float,
        no_play: bool,
    ) -> str:
        minutes, seconds = divmod(max(int(clock), 0), 60)
        if penalty_yards > 0:
            offender = f"{self.defense.name} (defense)"
        elif penalty_yards < 0:
            offender = f"{self.possession.name} (offense)"
        else:
            offender = "offsetting penalties"
        context = "no play run" if no_play else "accepted on the play"
        first_down_note = ", automatic 1st down" if auto_first_down else ""
        pre_label = f"{self._down_label(pre_down, pre_distance)} at {pre_yardline:.0f}"
        post_label = f"{self._down_label(self.down, self.distance)} at {self.yardline:.0f}"
        return (
            f"Q{quarter} {minutes:02d}:{seconds:02d} | {self.possession.name} ball | "
            f"PENALTY on {offender}: {abs(penalty_yards):.0f} yds ({context}{first_down_note}) | "
            f"{pre_label} -> {post_label}"
        )

    def simulate_drive(self, live_pbp: bool = False, progress: bool = False, progress_interval: int = 25) -> None:
        play_counter = 0
        while self.clock > 0:
            if self.halftime_pending:
                # The drive in progress ends at the half; the opening kickoff's receiver kicks off.
                self.halftime_pending = False
                self.kickoff(self.opening_receiver)
            self._normalize_down_distance()
            if self.down <= 4:
                if self._sample_penalty_no_play():
                    self.diagnostics["penalty_no_play_count"] += 1
                    pre_down, pre_distance, pre_yardline = self.down, self.distance, self.yardline
                    penalty_yards, auto_first_down = self._sample_penalty_outcome("no_play", "not_applicable")
                    self.diagnostics["penalty_yards_sum"] += penalty_yards
                    self.diagnostics["penalty_yards_abs_sum"] += abs(penalty_yards)
                    self.yardline = float(np.clip(self.yardline + penalty_yards, 0, 100))
                    # The down is replayed either way since no play was actually run, but the
                    # distance still needs to reflect the yardage the penalty moved the ball
                    # (positive penalty_yards shortens it, negative lengthens it). Only reset
                    # to a fresh set of downs if the penalty explicitly grants an automatic
                    # first down or the yardage alone reaches the line to gain.
                    if auto_first_down or penalty_yards >= self.distance:
                        if auto_first_down:
                            self.diagnostics["penalty_first_down_count"] += 1
                        self.down = 1
                        self.distance = 10
                    else:
                        self.distance -= penalty_yards
                    self._normalize_down_distance()
                    self.tick_clock("no_play_penalty")
                    if live_pbp:
                        print(
                            self._format_penalty_line(
                                self.quarter,
                                self.clock,
                                penalty_yards,
                                auto_first_down,
                                pre_down,
                                pre_distance,
                                pre_yardline,
                                no_play=True,
                            )
                        )
                    continue

                quarter, clock = self.quarter, self.clock
                play = Play(self.down, self.distance, self.yardline, game=self)
                play.simulate(self.possession, self.defense)
                self.plays.append(play)
                self.play_number += 1
                self.tick_clock(play.type)
                if live_pbp:
                    print(self._format_play_line(play, quarter, clock))
                if play.field_goal_attempt:
                    if play.yards_gained != 0:
                        self.switch_possession(self.yardline + play.yards_gained)
                        continue
                    if self.kickoff_model:
                        self.kickoff(self.possession)
                    else:
                        self.switch_possession(70)
                    continue
                if play.touchdown:
                    pat = Play(0, 0, 0, game=self)
                    pat.simulate(self.possession, self.defense, is_pat=True)
                    self.plays.append(pat)
                    self.play_number += 1
                    if live_pbp:
                        print(self._format_play_line(pat, self.quarter, self.clock))
                    if self.kickoff_model:
                        self.kickoff(self.possession)
                    else:
                        self.switch_possession(self._sample_post_score_switch_yardline())
                    break
                if play.defensive_touchdown:
                    self.defense.score += 7
                    if self.kickoff_model:
                        self.kickoff(self.defense)
                        continue
                    self.down = 1
                    self.yardline = 30
                    self.distance = 10
                    self._normalize_down_distance()
                    continue

                net_yards = play.yards_gained
                auto_first_down = False
                penalty_yards = 0.0
                penalty_on_play = False
                raw_yardline = float(np.clip(self.yardline + play.yards_gained, 0, 100))
                if play.yards_gained >= self.distance:
                    raw_down, raw_distance = 1, 10
                else:
                    raw_down, raw_distance = self.down + 1, self.distance - play.yards_gained
                if not (play.punt or play.turnover or play.field_goal_attempt):
                    if self._sample_penalty_on_play(play.type):
                        penalty_on_play = True
                        self.diagnostics["penalty_on_play_count"] += 1
                        penalty_yards, auto_first_down = self._sample_penalty_outcome("on_play", play.type)
                        self.diagnostics["penalty_yards_sum"] += penalty_yards
                        self.diagnostics["penalty_yards_abs_sum"] += abs(penalty_yards)
                        if auto_first_down:
                            self.diagnostics["penalty_first_down_count"] += 1
                        # Only affects drive-level yardline/down/distance bookkeeping below;
                        # player stat credit was already finalized inside play.simulate() above
                        # and is intentionally left untouched.
                        net_yards = play.yards_gained + penalty_yards

                self.yardline += play.yards_gained
                if net_yards != play.yards_gained:
                    self.yardline = float(np.clip(self.yardline + (net_yards - play.yards_gained), 0, 100))
                if net_yards >= self.distance or auto_first_down:
                    self.down = 1
                    self.distance = 10
                else:
                    self.down += 1
                    self.distance -= net_yards
                self._normalize_down_distance()
                if live_pbp and penalty_on_play:
                    print(
                        self._format_penalty_line(
                            self.quarter,
                            self.clock,
                            penalty_yards,
                            auto_first_down,
                            raw_down,
                            raw_distance,
                            raw_yardline,
                            no_play=False,
                        )
                    )
                if play.punt or play.turnover:
                    self.switch_possession(self.yardline)
                    continue
            else:
                self.switch_possession(self.yardline)
            play_counter += 1
            if progress and play_counter % max(1, int(progress_interval)) == 0:
                print(
                    f"drive progress: quarter {self.quarter}, clock {self.clock}, down {self.down}, distance {self.distance}",
                    flush=True,
                )

    def simulate_game(
        self,
        print_summary: bool = False,
        summary_prefix: str = "",
        live_pbp: bool = False,
        progress: bool = False,
        progress_interval: int = 25,
    ) -> Dict[str, object]:
        self._resample_prior_embeddings()
        if self.kickoff_model:
            self.opening_receiver = self.possession
            self.kickoff(self.defense)
        while self.quarter <= 4 and self.clock > 0:
            self.simulate_drive(live_pbp=live_pbp, progress=progress, progress_interval=progress_interval)
            if progress and not live_pbp and self.play_number > 0 and self.play_number % max(1, int(progress_interval)) == 0:
                print(
                    f"[{summary_prefix}] progress: quarter {self.quarter}, clock {self.clock}, plays {self.play_number}",
                    flush=True,
                )
        summary = self.build_game_summary()
        if print_summary:
            self.print_game_summary(summary, prefix=summary_prefix)
        return summary

    @staticmethod
    def _starting_qb(team: Team) -> Optional[Player]:
        qb1 = next((player for player in team.players if str(player.position) == "QB1"), None)
        if qb1 is not None:
            return qb1
        return next((player for player in team.players if str(player.position).startswith("QB")), None)

    @staticmethod
    def _build_prior_source_values(team: Team, prior_cols: List[str]) -> Dict[str, float]:
        source: Dict[str, object] = {}
        if isinstance(team.attributes, pd.Series):
            source.update(team.attributes.to_dict())
        else:
            source.update(dict(team.attributes))

        qb = Game._starting_qb(team)
        if qb is not None and isinstance(qb.attributes, pd.DataFrame) and not qb.attributes.empty:
            qb_row = qb.attributes.iloc[0].to_dict()
            # Only inject fields that are part of this model's prior schema.
            for col in prior_cols:
                if col in qb_row:
                    source[col] = qb_row[col]

        out: Dict[str, float] = {}
        for col in prior_cols:
            out[col] = float(pd.to_numeric(source.get(col, 0.0), errors="coerce"))
        return out

    def _resample_prior_embeddings(self) -> None:
        runtime = self.runtime
        self.team1.model_latents = {}
        self.team2.model_latents = {}
        self.team1.target_latent = None
        self.team2.target_latent = None
        self.team1.rusher_latent = None
        self.team2.rusher_latent = None

        for model_key, model in runtime.model_dict.items():
            is_cascade = isinstance(model, CascadeModelWrapper)
            if not is_cascade and not isinstance(model, (HierarchicalLatentGameModel, NGMMModel)):
                continue

            if is_cascade:
                # Cascade models: sample from stage-2 residual distribution
                context_cols = model.context_columns
                for team in (self.team1, self.team2):
                    # Mirror training semantics: def_cols describe the opposing defense
                    # (merged on defteam+game_id), everything else belongs to this team.
                    opponent = self.team2 if team is self.team1 else self.team1
                    context_values = {}
                    for col in context_cols:
                        source = opponent.attributes if col in runtime.def_cols else team.attributes
                        val = source.get(col, 0.0)
                        context_values[col] = float(pd.to_numeric(val, errors="coerce"))
                    rng = np.random.default_rng()
                    residual = model.sample_residual(context_values, rng)
                    z = np.array([residual], dtype=np.float32)
                    team.model_latents[model_key] = z

                    if model_key == "target":
                        team.target_latent = z
                        embed = np.full(len(runtime.target_embed_columns), residual, dtype=np.float32)
                        team.wr_embed = embed
                    elif model_key == "rusher":
                        team.rusher_latent = z
                        embed = np.full(len(runtime.running_embed_columns), residual, dtype=np.float32)
                        team.rb_embed = embed
                continue

            prior_proc_key = f"{model_key}_prior"
            prior_proc = runtime.processor.get(prior_proc_key)
            if prior_proc is None:
                message = f"{model_key}: missing prior preprocessor; latent sampling skipped"
                runtime._record_schema_issue(message)
                if runtime.strict_schema:
                    raise ValueError(message)
                continue
            prior_cols = [str(column) for column in getattr(prior_proc, "feature_names_in_", [])]
            if not prior_cols:
                message = f"{model_key}: prior preprocessor has no feature_names_in_; latent sampling skipped"
                runtime._record_schema_issue(message)
                if runtime.strict_schema:
                    raise ValueError(message)
                continue

            for team in (self.team1, self.team2):
                prior_df = pd.DataFrame([self._build_prior_source_values(team, prior_cols)]).fillna(0.0)
                prior_scaled = runtime.transform_with_processor(prior_proc_key, prior_df).astype(np.float32)
                prior_scaled = np.clip(np.nan_to_num(prior_scaled, nan=0.0), -runtime.MODEL_INPUT_CLIP, runtime.MODEL_INPUT_CLIP)
                if isinstance(model, NGMMModel):
                    z = torch.randn(model.latent_dim).numpy().astype(np.float32)
                else:
                    with torch.no_grad():
                        z = model.sample_z(torch.as_tensor(prior_scaled)).numpy()[0].astype(np.float32)
                team.model_latents[model_key] = z

                if model_key == "target":
                    team.target_latent = z
                    embed = z[: len(runtime.target_embed_columns)]
                    if embed.shape[0] < len(runtime.target_embed_columns):
                        embed = np.pad(embed, (0, len(runtime.target_embed_columns) - embed.shape[0]), mode="constant")
                    team.wr_embed = embed.astype(np.float32)
                elif model_key == "rusher":
                    team.rusher_latent = z
                    embed = z[: len(runtime.running_embed_columns)]
                    if embed.shape[0] < len(runtime.running_embed_columns):
                        embed = np.pad(embed, (0, len(runtime.running_embed_columns) - embed.shape[0]), mode="constant")
                    team.rb_embed = embed.astype(np.float32)

        effects = runtime.game_effects
        if effects is not None:
            # One game draw shared by both teams, one team-game draw shared by all of a team's models.
            rng = getattr(runtime, "rng", None)
            if not isinstance(rng, np.random.Generator):
                rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))
            game_draw = effects.draw_game(rng)
            for team in (self.team1, self.team2):
                shifts = effects.team_shifts(game_draw, rng)
                team.model_latents.update(shifts)
                team.target_latent = shifts.get("target", team.target_latent)
                team.rusher_latent = shifts.get("rusher", team.rusher_latent)

    def build_game_summary(self) -> Dict[str, object]:
        all_players = self.team1.players + self.team2.players
        qb = max(
            [player for player in all_players if player.position.startswith("QB")],
            key=lambda player: (player.stats["passing_yards"], player.stats["passing_touchdowns"], -player.stats["interceptions"]),
            default=None,
        )
        receiver = max(
            all_players,
            key=lambda player: (player.stats["receiving_yards"], player.stats["receptions"], player.stats["receiving_touchdowns"]),
            default=None,
        )
        rusher = max(
            all_players,
            key=lambda player: (player.stats["rushing_yards"], player.stats["rushing_attempts"], player.stats["rushing_touchdowns"]),
            default=None,
        )
        home_team = self.team2.name if self.team2.home == "home" else self.team1.name
        home_score = self.team2.score if self.team2.home == "home" else self.team1.score
        away_score = self.team1.score if self.team2.home == "home" else self.team2.score
        return {
            "home_team": home_team,
            "away_team": self.team1.name if self.team2.home == "home" else self.team2.name,
            "team1": self.team1.name,
            "team2": self.team2.name,
            "team1_score": self.team1.score,
            "team2_score": self.team2.score,
            "total": self.team1.score + self.team2.score,
            "spread": home_score - away_score,
            "plays": len(self.plays),
            "qb": qb,
            "receiver": receiver,
            "rusher": rusher,
        }

    def print_game_summary(self, summary: Dict[str, object], prefix: str = "") -> None:
        label = f"{prefix} " if prefix else ""
        print(
            f"{label}Final: {summary['team1']} {summary['team1_score']}, {summary['team2']} {summary['team2_score']} "
            f"(Total {summary['total']}, Spread {summary['home_team']} {summary['spread']:+g})"
        )
        print(self._format_leader_line("QB", summary.get("qb"), ["completions", "pass_attempts", "passing_yards", "passing_touchdowns", "interceptions"]))
        print(self._format_leader_line("Receiver", summary.get("receiver"), ["receptions", "receiving_yards", "receiving_touchdowns"]))
        print(self._format_leader_line("Rusher", summary.get("rusher"), ["rushing_attempts", "rushing_yards", "rushing_touchdowns"]))

    @staticmethod
    def _format_leader_line(label: str, player: Optional[Player], stats: List[str]) -> str:
        if player is None:
            return f"{label}: unavailable"
        if label == "QB":
            return (
                f"QB: {player.name} {player.stats['completions']}/{player.stats['pass_attempts']}, "
                f"{player.stats['passing_yards']} pass yds, {player.stats['passing_touchdowns']} pass TD, "
                f"{player.stats['interceptions']} INT"
            )
        if label == "Receiver":
            return (
                f"Receiver: {player.name} {player.stats['receptions']} rec, {player.stats['receiving_yards']} rec yds, "
                f"{player.stats['receiving_touchdowns']} rec TD"
            )
        return (
            f"Rusher: {player.name} {player.stats['rushing_attempts']} att, {player.stats['rushing_yards']} rush yds, "
            f"{player.stats['rushing_touchdowns']} rush TD"
        )