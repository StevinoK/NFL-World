from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

SCHEMA_VERSION = 1
CONFIG_FILENAME = "model_feature_config.json"

FEATURE_NAME_ALIASES = {
    "qb_games_played": "qb_team_games_played",
    "rb_carry_split_roll4": "rb_carry_split_gini_roll4",
}

BASE_FEATURE_GROUPS: Dict[str, List[str]] = {
    "game_state_cols": [
        "defteam_score",
        "down",
        "game_seconds_remaining",
        "posteam_score",
        "posteam_type",
        "yardline_100",
        "ydstogo",
        "roof",
        "score_differential",
    ],
    "position_cols": ["position"],
    "pass_columns": ["air_yards"],
    "rusher_columns": ["close_bayes", "explosive_bayes", "ypc_bayes", "rush_rate", "scramble_rate"],
    "def_cols": [
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

    ],
    "team_stats_columns": [
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
    ],
    "team_tendency_columns": [
        "pass_oe",
        "pass_oe_weekly_variation",
        "target_gini_roll4",
        "rb_carry_split_gini_roll4",
        "rb_top_carry_share_roll4",
        "pace_of_play_bayes",
    ],
    "offense_context_columns": [
        "pass_oe",
        "pass_oe_weekly_variation",
        "target_gini_roll4",
    ],
    "backfield_context_columns": [
        "rb_carry_split_gini_roll4",
        "rb_top_carry_share_roll4",
    ],
    "pregame_context_columns": [
        "pass_oe",
        "pass_oe_weekly_variation",
        "target_gini_roll4",
        "rb_carry_split_gini_roll4",
        "rb_top_carry_share_roll4",
        "pace_of_play_bayes",
        "days_rest",
        "opp_days_rest",
    ],
    "receiver_columns": ["deep", "medium", "short", "fd_bayes", "rz_bayes", "yac_bayes", "adot", "ypt"],
    "passer_columns": [
        "cpoe_bayes",
        "turnover_bayes",
        "sack_bayes",
        "cpoe_short",
        "cpoe_medium",
        "cpoe_deep",
        "deep_pass_rate",
        "medium_pass_rate",
        "short_pass_rate",
    ],
    "qb_context_columns": ["qb_team_games_played", "qb_dome_rate_bayes"],
    "play_cols": ["play_type_cat"],
    "play_outcome_cols": ["play_outcome"],
    "penalty_context_cols": ["penalty_context"],
    "slot_stat_cols": [],
    "target_embed_columns": [],
    "running_embed_columns": [],
}

DEFAULT_MODEL_CONFIG: Dict[str, Dict[str, List[str]]] = {
    "yac": {
        "feature_groups": [
            "game_state_cols",
            "def_cols",
            "receiver_columns",
            "pass_columns",
            "team_stats_columns",
            "pregame_context_columns",
            "qb_context_columns",
            "position_cols",
        ],
        "categorical_cols": ["roof", "posteam_type", "position"],
    },
    "air_yards": {
        "feature_groups": ["game_state_cols", "def_cols", "passer_columns", "team_stats_columns", "pregame_context_columns", "qb_context_columns"],
        "categorical_cols": ["roof", "posteam_type"],
    },
    "sack": {
        "feature_groups": ["game_state_cols", "def_cols", "passer_columns", "team_stats_columns", "pregame_context_columns", "qb_context_columns"],
        "categorical_cols": ["roof", "posteam_type"],
    },
    "scramble": {
        "feature_groups": ["game_state_cols", "def_cols", "passer_columns", "team_stats_columns", "pregame_context_columns", "qb_context_columns"],
        "categorical_cols": ["roof", "posteam_type"],
    },
    "interception": {
        "feature_groups": ["game_state_cols", "def_cols", "passer_columns", "pass_columns", "team_stats_columns", "pregame_context_columns", "qb_context_columns"],
        "categorical_cols": ["roof", "posteam_type"],
    },
    "completion": {
        "feature_groups": ["game_state_cols", "def_cols", "passer_columns", "pass_columns", "team_stats_columns", "pregame_context_columns", "qb_context_columns"],
        "categorical_cols": ["roof", "posteam_type"],
    },
    "punt": {
        "feature_groups": ["game_state_cols"],
        "categorical_cols": ["roof", "posteam_type"],
    },
    "time_elapsed": {
        "feature_groups": ["game_state_cols", "play_cols", "pregame_context_columns"],
        "categorical_cols": ["roof", "posteam_type", "play_type_cat"],
    },
    "kick": {
        "feature_groups": ["game_state_cols"],
        "categorical_cols": ["roof", "posteam_type"],
    },
    "rush_yards": {
        "feature_groups": ["game_state_cols", "def_cols", "rusher_columns", "team_stats_columns", "pregame_context_columns", "position_cols"],
        "categorical_cols": ["roof", "posteam_type", "position"],
    },
    "scramble_yards": {
        "feature_groups": ["game_state_cols", "def_cols", "rusher_columns", "team_stats_columns", "pregame_context_columns", "position_cols"],
        "categorical_cols": ["roof", "posteam_type", "position"],
    },
    "play_type": {
        "feature_groups": ["game_state_cols", "team_stats_columns", "pregame_context_columns", "def_cols"],
        "categorical_cols": ["roof", "posteam_type"],
    },
    "target_selection": {
        "feature_groups": ["game_state_cols", "slot_stat_cols", "pass_columns", "target_embed_columns", "pregame_context_columns", "qb_context_columns"],
        "categorical_cols": ["roof", "posteam_type"],
        "use_prior_game_state": True,
        "prior_feature_groups": ["team_stats_columns", "pregame_context_columns", "qb_context_columns"],
        "latent_dim": 8,
        "kl_weight": 0.05,
    },
    "rusher_selection": {
        "feature_groups": ["game_state_cols", "slot_stat_cols", "running_embed_columns", "pregame_context_columns"],
        "categorical_cols": ["roof", "posteam_type"],
        "use_prior_game_state": True,
        "prior_feature_groups": ["team_stats_columns", "pregame_context_columns"],
        "latent_dim": 8,
        "kl_weight": 0.05,
    },
    "fumble": {
        "feature_groups": ["game_state_cols", "team_stats_columns", "pregame_context_columns", "def_cols", "play_outcome_cols"],
        "categorical_cols": ["roof", "posteam_type", "play_outcome"],
    },
    "defensive_return_yards": {
        "feature_groups": ["game_state_cols", "pregame_context_columns"],
        "categorical_cols": ["roof", "posteam_type"],
    },
    "penalty_no_play": {
        "feature_groups": ["game_state_cols", "team_stats_columns", "def_cols", "pregame_context_columns"],
        "categorical_cols": ["roof", "posteam_type"],
    },
    "penalty_on_play": {
        "feature_groups": ["game_state_cols", "team_stats_columns", "def_cols", "pregame_context_columns", "play_cols"],
        "categorical_cols": ["roof", "posteam_type", "play_type_cat"],
    },
    "penalty_yards": {
        "feature_groups": ["game_state_cols", "team_stats_columns", "def_cols", "pregame_context_columns", "play_cols", "penalty_context_cols"],
        "categorical_cols": ["roof", "posteam_type", "play_type_cat", "penalty_context"],
    },
    "penalty_first_down": {
        "feature_groups": ["game_state_cols", "team_stats_columns", "def_cols", "pregame_context_columns", "play_cols", "penalty_context_cols"],
        "categorical_cols": ["roof", "posteam_type", "play_type_cat", "penalty_context"],
    },
}


def _unique_ordered(items: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def build_feature_catalog(runtime_groups: Mapping[str, List[str]] | None = None) -> Dict[str, List[str]]:
    catalog = {name: list(cols) for name, cols in BASE_FEATURE_GROUPS.items()}
    if runtime_groups:
        for name, cols in runtime_groups.items():
            catalog[name] = list(cols)
    return catalog


def expand_feature_groups(feature_groups: List[str], feature_catalog: Mapping[str, List[str]]) -> List[str]:
    missing = [group for group in feature_groups if group not in feature_catalog]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Unknown feature group(s): {missing_text}")

    expanded: List[str] = []
    for group in feature_groups:
        expanded.extend(feature_catalog[group])
    return _unique_ordered(expanded)


def build_default_config(feature_catalog: Mapping[str, List[str]]) -> Dict[str, object]:
    models: Dict[str, Dict[str, object]] = {}
    for model_name, model_cfg in DEFAULT_MODEL_CONFIG.items():
        groups = list(model_cfg["feature_groups"])
        prior_groups = list(model_cfg.get("prior_feature_groups", []))
        models[model_name] = {
            "feature_groups": groups,
            "expanded_columns": expand_feature_groups(groups, feature_catalog),
            "categorical_cols": list(model_cfg.get("categorical_cols", [])),
            "use_prior_game_state": bool(model_cfg.get("use_prior_game_state", False)),
            "prior_feature_groups": prior_groups,
            "prior_expanded_columns": expand_feature_groups(prior_groups, feature_catalog) if prior_groups else [],
            "latent_dim": int(model_cfg.get("latent_dim", 0)),
            "kl_weight": float(model_cfg.get("kl_weight", 0.0)),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "models": models,
    }


def load_feature_config(config_path: Path | str, feature_catalog: Mapping[str, List[str]]) -> Dict[str, object]:
    path = Path(config_path)
    if not path.exists():
        config = build_default_config(feature_catalog)
        save_feature_config(config, path)
        return config

    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    validate_feature_config(config, feature_catalog)
    migrated = migrate_feature_config(config)
    if migrated:
        save_feature_config(config, path)
    return config


def save_feature_config(config: Dict[str, object], config_path: Path | str) -> None:
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def validate_feature_config(config: Mapping[str, object], feature_catalog: Mapping[str, List[str]]) -> None:
    if "models" not in config or not isinstance(config["models"], dict):
        raise ValueError("Config must include a 'models' object.")

    models = config["models"]
    for model_name, model_cfg in models.items():
        if not isinstance(model_cfg, dict):
            raise ValueError(f"Model '{model_name}' config must be an object.")

        groups = model_cfg.get("feature_groups")
        if not isinstance(groups, list) or not all(isinstance(g, str) for g in groups):
            raise ValueError(f"Model '{model_name}' must define 'feature_groups' as a string list.")

        _ = expand_feature_groups(groups, feature_catalog)

        if model_cfg.get("use_prior_game_state", False):
            prior_groups = model_cfg.get("prior_feature_groups", [])
            if not isinstance(prior_groups, list) or not all(isinstance(g, str) for g in prior_groups):
                raise ValueError(f"Model '{model_name}' must define 'prior_feature_groups' as a string list when prior modeling is enabled.")
            _ = expand_feature_groups(prior_groups, feature_catalog)

        latent_dim = model_cfg.get("latent_dim", 0)
        if not isinstance(latent_dim, int) or latent_dim < 0:
            raise ValueError(f"Model '{model_name}' must define a non-negative integer 'latent_dim'.")

        kl_weight = model_cfg.get("kl_weight", 0.0)
        if not isinstance(kl_weight, (int, float)) or kl_weight < 0:
            raise ValueError(f"Model '{model_name}' must define a non-negative 'kl_weight'.")


def migrate_feature_config(config: Dict[str, object]) -> bool:
    models = config.get("models", {})
    changed = False

    feature_group_additions = {
        "yac": ["pregame_context_columns"],
        "air_yards": ["pregame_context_columns"],
        "sack": ["pregame_context_columns"],
        "interception": ["pregame_context_columns"],
        "completion": ["pregame_context_columns"],
        "rush_yards": ["pregame_context_columns"],
        "play_type": ["pregame_context_columns"],
        "target_selection": ["pregame_context_columns"],
        "rusher_selection": ["pregame_context_columns"],
        "fumble": ["pregame_context_columns"],
        "defensive_return_yards": ["pregame_context_columns"],
    }

    prior_group_additions = {
        "target_selection": ["pregame_context_columns", "qb_context_columns"],
        "rusher_selection": ["pregame_context_columns"],
    }

    qb_group_additions = {
        "yac": ["qb_context_columns"],
        "air_yards": ["qb_context_columns"],
        "sack": ["qb_context_columns"],
        "interception": ["qb_context_columns"],
        "completion": ["qb_context_columns"],
        "target_selection": ["qb_context_columns"],
    }

    for model_name, additions in feature_group_additions.items():
        model_cfg = models.get(model_name)
        if not isinstance(model_cfg, dict):
            continue
        groups = model_cfg.get("feature_groups")
        if not isinstance(groups, list):
            continue
        for group_name in additions:
            if group_name not in groups:
                groups.append(group_name)
                changed = True

    for model_name, additions in qb_group_additions.items():
        model_cfg = models.get(model_name)
        if not isinstance(model_cfg, dict):
            continue
        groups = model_cfg.get("feature_groups")
        if not isinstance(groups, list):
            continue
        for group_name in additions:
            if group_name not in groups:
                groups.append(group_name)
                changed = True

    for model_name, additions in prior_group_additions.items():
        model_cfg = models.get(model_name)
        if not isinstance(model_cfg, dict):
            continue
        groups = model_cfg.get("prior_feature_groups")
        if not isinstance(groups, list):
            continue
        for group_name in additions:
            if group_name not in groups:
                groups.append(group_name)
                changed = True
    return changed


def resolve_model_features(
    config: Mapping[str, object],
    model_name: str,
    feature_catalog: Mapping[str, List[str]],
) -> List[str]:
    models = config.get("models", {})
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not found in config.")

    model_cfg = models[model_name]
    groups = model_cfg.get("feature_groups", [])
    return expand_feature_groups(groups, feature_catalog)


def resolve_model_expanded_columns(config: Mapping[str, object], model_name: str) -> List[str]:
    models = config.get("models", {})
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not found in config.")
    model_cfg = models[model_name]
    return list(model_cfg.get("expanded_columns", []))


def resolve_model_categorical_cols(config: Mapping[str, object], model_name: str) -> List[str]:
    models = config.get("models", {})
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not found in config.")
    model_cfg = models[model_name]
    return list(model_cfg.get("categorical_cols", []))


def resolve_model_prior_groups(config: Mapping[str, object], model_name: str) -> List[str]:
    models = config.get("models", {})
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not found in config.")
    model_cfg = models[model_name]
    return list(model_cfg.get("prior_feature_groups", []))


def resolve_model_prior_features(
    config: Mapping[str, object],
    model_name: str,
    feature_catalog: Mapping[str, List[str]],
) -> List[str]:
    models = config.get("models", {})
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not found in config.")

    model_cfg = models[model_name]
    prior_groups = model_cfg.get("prior_feature_groups", [])
    return expand_feature_groups(prior_groups, feature_catalog) if prior_groups else []


def resolve_model_prior_expanded_columns(config: Mapping[str, object], model_name: str) -> List[str]:
    models = config.get("models", {})
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not found in config.")
    model_cfg = models[model_name]
    return list(model_cfg.get("prior_expanded_columns", []))


def resolve_model_prior_enabled(config: Mapping[str, object], model_name: str) -> bool:
    models = config.get("models", {})
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not found in config.")
    model_cfg = models[model_name]
    return bool(model_cfg.get("use_prior_game_state", False))


def resolve_model_latent_dim(config: Mapping[str, object], model_name: str, default: int = 0) -> int:
    models = config.get("models", {})
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not found in config.")
    model_cfg = models[model_name]
    latent_dim = model_cfg.get("latent_dim", default)
    return int(latent_dim)


def resolve_model_kl_weight(config: Mapping[str, object], model_name: str, default: float = 0.0) -> float:
    models = config.get("models", {})
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not found in config.")
    model_cfg = models[model_name]
    kl_weight = model_cfg.get("kl_weight", default)
    return float(kl_weight)


def ensure_expanded_columns(
    config: Dict[str, object],
    feature_catalog: Mapping[str, List[str]],
) -> Dict[str, object]:
    models = config.get("models", {})
    for model_name, model_cfg in models.items():
        groups = model_cfg.get("feature_groups", [])
        model_cfg["expanded_columns"] = expand_feature_groups(groups, feature_catalog)
        prior_groups = model_cfg.get("prior_feature_groups", [])
        model_cfg["prior_expanded_columns"] = expand_feature_groups(prior_groups, feature_catalog) if prior_groups else []
    return config
