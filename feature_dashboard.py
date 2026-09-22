from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

try:
    from feature_config_utils import (
        CONFIG_FILENAME,
        FEATURE_NAME_ALIASES,
        build_default_config,
        build_feature_catalog,
        ensure_expanded_columns,
        load_feature_config,
        resolve_model_kl_weight,
        resolve_model_latent_dim,
        resolve_model_prior_enabled,
        resolve_model_prior_groups,
        save_feature_config,
    )
except ImportError:
    from feature_config_utils import (
        CONFIG_FILENAME,
        build_default_config,
        build_feature_catalog,
        ensure_expanded_columns,
        load_feature_config,
        save_feature_config,
    )

    FEATURE_NAME_ALIASES = {}

    def resolve_model_kl_weight(config: dict, model_name: str, default: float = 0.0) -> float:
        return float(config.get("models", {}).get(model_name, {}).get("kl_weight", default))


    def resolve_model_latent_dim(config: dict, model_name: str, default: int = 0) -> int:
        return int(config.get("models", {}).get(model_name, {}).get("latent_dim", default))


    def resolve_model_prior_enabled(config: dict, model_name: str) -> bool:
        return bool(config.get("models", {}).get(model_name, {}).get("use_prior_game_state", False))


    def resolve_model_prior_groups(config: dict, model_name: str) -> list[str]:
        return list(config.get("models", {}).get(model_name, {}).get("prior_feature_groups", []))

try:
    from streamlit_sortables import sort_items
except ImportError:  # pragma: no cover
    sort_items = None

try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
except Exception:  # pragma: no cover
    get_script_run_ctx = None

APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / CONFIG_FILENAME
SNAPSHOT_DIR = APP_DIR / "feature_config_snapshots"

REQUIRED_GROUPS = {
    "team_tendency_columns": [
        "pass_oe",
        "target_gini_roll4",
        "rb_carry_split_gini_roll4",
        "rb_top_carry_share_roll4",
    ],
    "qb_context_columns": ["qb_team_games_played"],
}


def _init_state(config: dict) -> None:
    if "model_config" not in st.session_state:
        st.session_state.model_config = config


def _sortable_for_model(model_name: str, groups: list[str], catalog_keys: list[str]) -> list[str]:
    assigned = list(groups)
    available = [g for g in catalog_keys if g not in assigned]

    if sort_items is None:
        st.info("Install streamlit-sortables for drag-and-drop: pip install streamlit-sortables")
        selected = st.multiselect(
            f"Feature groups for {model_name}",
            options=catalog_keys,
            default=assigned,
            key=f"multiselect_{model_name}",
        )
        return selected

    st.caption(f"Drag groups into {model_name} and reorder them.")
    result = sort_items(
        [assigned, available],
        direction="horizontal",
        custom_style="""
            .sortable-component {
                border: 1px solid #ccc;
                border-radius: 8px;
                min-height: 250px;
                padding: 8px;
                background-color: #f9f9f9;
            }
        """,
        key=f"sort_{model_name}",
    )
    return result[0] if result and isinstance(result, list) and len(result) > 0 else assigned


def _list_select_for_groups(label: str, model_name: str, current_groups: list[str], catalog_keys: list[str]) -> list[str]:
    if sort_items is None:
        return st.multiselect(
            label,
            options=catalog_keys,
            default=current_groups,
            key=f"{label}_{model_name}",
        )

    st.caption(f"Drag groups into {label.lower()} for {model_name}.")
    result = sort_items(
        [list(current_groups), [g for g in catalog_keys if g not in current_groups]],
        direction="horizontal",
        custom_style="""
            .sortable-component {
                border: 1px solid #ccc;
                border-radius: 8px;
                min-height: 220px;
                padding: 8px;
                background-color: #f9f9f9;
            }
        """,
        key=f"{label}_{model_name}",
    )
    return result[0] if result and isinstance(result, list) and len(result) > 0 else list(current_groups)


def _render_model_summary(config: dict) -> None:
    models = config.get("models", {})
    rows = []
    for model_name, model_cfg in models.items():
        rows.append(
            {
                "model": model_name,
                "groups": len(model_cfg.get("feature_groups", [])),
                "expanded_columns": len(model_cfg.get("expanded_columns", [])),
                "group_list": ", ".join(model_cfg.get("feature_groups", [])),
            }
        )

    st.subheader("Current Mapping")
    st.dataframe(rows, width="stretch")


def main() -> None:
    st.set_page_config(page_title="Torch Model Feature Mapper", layout="wide")
    st.title("Torch Feature Mapper")
    st.write("Assign feature groups to each model and save a config used by Torch Modeling.py.")

    feature_catalog = build_feature_catalog()
    for group_name, cols in REQUIRED_GROUPS.items():
        feature_catalog.setdefault(group_name, cols)

    config = load_feature_config(CONFIG_PATH, feature_catalog)
    config = ensure_expanded_columns(config, feature_catalog)
    _init_state(config)

    active = st.session_state.model_config
    models = active.get("models", {})

    st.info(
        "New feature groups added: team_tendency_columns (pass_oe, target_gini_roll4, "
        "rb_carry_split_gini_roll4, rb_top_carry_share_roll4) and qb_context_columns (qb_team_games_played)."
    )
    if FEATURE_NAME_ALIASES:
        alias_text = ", ".join(f"{k} -> {v}" for k, v in FEATURE_NAME_ALIASES.items())
        st.caption(f"Alias mapping: {alias_text}")

    st.sidebar.header("Config Actions")
    if st.sidebar.button("Reload from Disk"):
        st.session_state.model_config = load_feature_config(CONFIG_PATH, feature_catalog)
        st.rerun()

    if st.sidebar.button("Reset to Baseline"):
        baseline = build_default_config(feature_catalog)
        st.session_state.model_config = baseline
        st.rerun()

    if st.sidebar.button("Save Config"):
        to_save = ensure_expanded_columns(st.session_state.model_config, feature_catalog)
        save_feature_config(to_save, CONFIG_PATH)
        st.sidebar.success(f"Saved {CONFIG_FILENAME}")

    if st.sidebar.button("Save Snapshot"):
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snap_path = SNAPSHOT_DIR / f"model_feature_config_{stamp}.json"
        to_save = ensure_expanded_columns(st.session_state.model_config, feature_catalog)
        save_feature_config(to_save, snap_path)
        st.sidebar.success(f"Saved snapshot: {snap_path.name}")

    st.subheader("Model Editor")
    all_groups = list(feature_catalog.keys())
    model_names = list(models.keys())
    selected_model = st.selectbox("Choose model", model_names)

    model_cfg = models[selected_model]
    current_groups = list(model_cfg.get("feature_groups", []))
    updated_groups = _sortable_for_model(selected_model, current_groups, all_groups)

    use_prior = st.toggle(
        "Use prior game state latent",
        value=bool(model_cfg.get("use_prior_game_state", False)),
        key=f"prior_toggle_{selected_model}",
    )

    prior_groups = list(model_cfg.get("prior_feature_groups", []))
    if use_prior:
        prior_groups = _list_select_for_groups("Prior feature groups", selected_model, prior_groups, all_groups)

    latent_dim = st.number_input(
        "Latent dimension",
        min_value=0,
        max_value=64,
        value=resolve_model_latent_dim(st.session_state.model_config, selected_model, default=8),
        step=1,
        key=f"latent_dim_{selected_model}",
    )

    kl_weight = st.number_input(
        "KL weight",
        min_value=0.0,
        max_value=10.0,
        value=resolve_model_kl_weight(st.session_state.model_config, selected_model, default=0.05),
        step=0.01,
        format="%.2f",
        key=f"kl_weight_{selected_model}",
    )

    selected_cats = st.multiselect(
        "Categorical columns",
        options=sorted({c for vals in feature_catalog.values() for c in vals}),
        default=model_cfg.get("categorical_cols", []),
        key=f"cats_{selected_model}",
    )

    settings_changed = (
        updated_groups != current_groups
        or selected_cats != model_cfg.get("categorical_cols", [])
        or use_prior != bool(model_cfg.get("use_prior_game_state", False))
        or prior_groups != list(model_cfg.get("prior_feature_groups", []))
        or int(latent_dim) != int(model_cfg.get("latent_dim", 0))
        or float(kl_weight) != float(model_cfg.get("kl_weight", 0.0))
    )

    if settings_changed:
        st.session_state.model_config["models"][selected_model]["feature_groups"] = updated_groups
        st.session_state.model_config["models"][selected_model]["categorical_cols"] = selected_cats
        st.session_state.model_config["models"][selected_model]["use_prior_game_state"] = use_prior
        st.session_state.model_config["models"][selected_model]["prior_feature_groups"] = prior_groups if use_prior else []
        st.session_state.model_config["models"][selected_model]["latent_dim"] = int(latent_dim)
        st.session_state.model_config["models"][selected_model]["kl_weight"] = float(kl_weight)
        st.session_state.model_config = ensure_expanded_columns(st.session_state.model_config, feature_catalog)

    st.subheader("Expanded Columns Preview")
    st.code("\n".join(st.session_state.model_config["models"][selected_model].get("expanded_columns", [])) or "(no columns)")

    _render_model_summary(st.session_state.model_config)

    st.markdown("### Group Catalog")
    for group_name in all_groups:
        with st.expander(f"{group_name} ({len(feature_catalog[group_name])})"):
            if feature_catalog[group_name]:
                st.write(feature_catalog[group_name])
            else:
                st.write("Dynamic group. This is populated at runtime in Torch Modeling.py.")


if __name__ == "__main__":
    if get_script_run_ctx is None or get_script_run_ctx() is None:
        os.execv(
            sys.executable,
            [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve())],
        )

    main()
