from __future__ import annotations

import numpy as np
import pandas as pd

from nflsim.engine.cascade_model_training import _filter_non_meaningful_games
from nflsim.configuration import MODEL_CONTEXT, V2_MODEL_CONTEXT, context_columns
from nflsim.features.builder import canonicalize_teams
from nflsim.features.slots import USAGE_COLUMNS, rank_slots
from nflsim.features.special_teams import COLUMNS as SPECIAL_TEAMS_COLUMNS, kickoff_starts
from nflsim.features.player_value import PV_DEFENSE_COLUMNS, PV_OFFENSE_COLUMNS
from nflsim.semantics import (
    ISOLATED_MU_CONTEXT_V2, context_available_column, opponent_defense_source,
)
from nflsim.features.weather import WEATHER_COLUMNS, weather_frame
from nflsim.engine.simulation_runtime import SKILL_SLOT_ORDER, SimulationRuntime


def context_eligibility(frame, model, semantics=None):
    columns = context_columns(model, semantics)
    original = frame.reindex(columns=columns).apply(pd.to_numeric, errors="coerce")
    unavailable_by_column = {
        column: int((~np.isfinite(original[column].to_numpy(dtype=float))).sum())
        for column in columns
    }
    values = original.copy()
    if semantics == ISOLATED_MU_CONTEXT_V2:
        values = _neutralize_v2_context(values)
    checked_columns = list(values.columns)
    available = pd.DataFrame(np.isfinite(values.to_numpy(dtype=float)), index=frame.index, columns=checked_columns)
    eligible = available.all(axis=1)
    diagnostics = {"required_context": checked_columns, "candidate_rows": len(frame), "eligible_rows": int(eligible.sum()), "excluded_rows": int((~eligible).sum()), "unavailable_by_column": unavailable_by_column, "reason": "unavailable_preweek_ridge_context"}
    return eligible, diagnostics


def _neutralize_v2_context(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.copy()
    optional = [column for column in values if column.startswith("ridge_") and not column.endswith("_available")]
    for column in dict.fromkeys(optional):
        availability = context_available_column(column)
        source = pd.to_numeric(values[column], errors="coerce")
        if availability in values:
            flag = pd.to_numeric(values[availability], errors="coerce").fillna(0.0).gt(0.0)
            flag &= np.isfinite(source)
        else:
            flag = np.isfinite(source)
            values[availability] = flag.astype(float)
        values[column] = source.where(flag, 0.0)
        values[availability] = flag.astype(float)
    for column in values:
        if column.startswith("ridge_") and column.endswith("_available"):
            flag = pd.to_numeric(values[column], errors="coerce")
            values[column] = (np.isfinite(flag) & flag.gt(0.0)).astype(float)
    return values


def normalize_training_season(pbp):
    frame = pbp.copy()
    if "game_id" not in frame:
        raise ValueError("Training PBP missing game_id")
    parsed = frame.game_id.astype("string").str.extract(r"^(\d{4})_(\d{2})_")
    if parsed.isna().any().any():
        raise ValueError("Training PBP contains invalid season-prefixed game IDs")
    for field, index in (("season", 0), ("week", 1)):
        values = parsed[index].astype(int)
        for column in (field, f"{field}_x", f"{field}_y"):
            if column not in frame:
                continue
            source = pd.to_numeric(frame[column], errors="raise")
            if (source.notna() & source.ne(values)).any():
                raise ValueError(f"Training PBP {column} disagrees with game_id {field}")
        frame[field] = values
    return frame


def prepare_frame(pbp, players, teams, semantics=None, seasons=(2017, 2024)):
    """Preserve unavailable warm-up context; model_frame filters only its own inputs.

    Eligibility counts are exposed in attrs['ridge_context_eligibility']; no
    future-derived or zero-imputed substitute is constructed for missing fits.
    """
    pbp = normalize_training_season(pbp)
    pbp, players, teams = [canonicalize_teams(frame) for frame in (pbp, players, teams)]
    pbp.loc[:, list(WEATHER_COLUMNS)] = weather_frame(pbp)
    required = {"season", "season_type", "aborted_play", "penalty", "game_id", "posteam", "defteam"}
    if not required <= set(pbp):
        raise ValueError(f"Training PBP missing required columns: {sorted(required-set(pbp))}")
    pbp = pbp[pbp.season.between(*seasons) & pbp.season_type.eq("REG") & pbp.aborted_play.eq(0)]
    pbp = _filter_non_meaningful_games(pbp)
    pbp = pbp.sort_values(["game_id", "play_id"])
    pbp["elapsed_time"] = pbp.groupby("game_id").game_seconds_remaining.diff(-1)
    pbp["net_yards"] = pbp.kick_distance-pbp.return_yards
    def attach(table, key, left, columns):
        nonlocal pbp
        missing = set(columns)-set(table)
        if missing:
            raise ValueError(f"Frozen {key} table missing: {sorted(missing)}")
        pbp = pbp.drop(columns=[column for column in columns if column in pbp], errors="ignore").merge(table.set_index([key, "game_id"])[columns], left_on=[left, "game_id"], right_index=True, how="left", validate="many_to_one")
    attach(players, "player_id", "receiver_player_id", SimulationRuntime.receiver_columns)
    # nflverse leaves passer_player_id empty on scrambles; without the scrambling QB's stats the
    # dropback models learn "passer stats missing => scramble", which never happens in simulation.
    scramble = pbp.qb_scramble.eq(1) if "qb_scramble" in pbp else pd.Series(False, index=pbp.index)
    pbp["dropback_qb_id"] = pbp.passer_player_id.where(pbp.passer_player_id.notna() | ~scramble, pbp.get("rusher_player_id"))
    attach(players, "player_id", "dropback_qb_id", SimulationRuntime.passer_columns + SimulationRuntime.qb_context_columns)
    attach(teams, "team", "defteam", SimulationRuntime.def_cols + [column for column in PV_DEFENSE_COLUMNS if column in teams])
    own = list(dict.fromkeys(SimulationRuntime.team_stats_cols + SimulationRuntime.pregame_context_columns
                             + [column for column in teams if column.startswith("ridge_") or column in PV_OFFENSE_COLUMNS]))
    attach(teams, "team", "posteam", own)
    # Special-teams features for both sides; the punt/kickoff models assign kicking and receiving roles.
    special = [column for column in SPECIAL_TEAMS_COLUMNS if column in teams]
    for side in ("posteam", "defteam") if special else ():
        values = teams.set_index(["team", "game_id"])[special].add_prefix(f"{side}_")
        pbp = pbp.drop(columns=[column for column in values if column in pbp]).merge(
            values, left_on=[side, "game_id"], right_index=True, how="left", validate="many_to_one")
    if semantics == ISOLATED_MU_CONTEXT_V2:
        fit_targets = sorted({target for targets in V2_MODEL_CONTEXT.values() for target in targets})
        opponent_values = {}
        for target in fit_targets:
            source = f"ridge_{target}_defense_mu"
            if source in teams:
                opponent_values[f"ridge_{target}_opponent_defense_effect"] = source
                opponent_values[f"ridge_{target}_opponent_defense_effect_available"] = f"{source}_available"
        opponent = teams.set_index(["team", "game_id"])
        keys = pd.MultiIndex.from_arrays([pbp["defteam"], pbp["game_id"]])
        for destination, source in opponent_values.items():
            if source in opponent:
                pbp[destination] = opponent[source].reindex(keys).to_numpy()
    models = list(dict.fromkeys([*MODEL_CONTEXT, *V2_MODEL_CONTEXT])) if semantics == ISOLATED_MU_CONTEXT_V2 else list(MODEL_CONTEXT)
    required_context = list(dict.fromkeys(column for model in models for column in context_columns(model, semantics)))
    for column in required_context:
        if column not in pbp:
            pbp[column] = np.nan
    pbp["game_team_id"] = pbp.game_id.astype(str) + "::" + pbp.posteam.astype(str)
    pbp.attrs["ridge_context_eligibility"] = {model: context_eligibility(pbp, model, semantics)[1] for model in models}
    if semantics is not None:
        pbp.attrs["feature_semantics"] = semantics
    return pbp


def slot_frame(frame, players, roster, role, stat_columns):
    frame, players, roster = [canonicalize_teams(table) for table in (frame, players, roster)]
    if not {"game_id", "team", "player_id", "position", "depth_team", "report_status"} <= set(roster):
        raise ValueError("Selection training requires pregame roster with depth_team and report_status")
    roster = roster.merge(players, on=["game_id", "player_id"], how="left", validate="many_to_one")
    roster = rank_slots(roster)
    slots = roster.pivot(index=["game_id", "team"], columns="slot", values="player_id").reindex(columns=SKILL_SLOT_ORDER)
    merged = frame.merge(slots, left_on=["game_id", "posteam"], right_index=True, how="inner", validate="many_to_one")
    matches = merged[SKILL_SLOT_ORDER].eq(merged[f"{role}_player_id"], axis=0)
    merged = merged[matches.any(axis=1)].copy()
    merged["selection_label"] = matches.loc[merged.index].to_numpy().argmax(axis=1)
    for column in stat_columns:
        values = roster.pivot(index=["game_id", "team"], columns="slot", values=column).reindex(index=slots.index, columns=SKILL_SLOT_ORDER)
        empty = slots.isna().to_numpy()
        values = values.mask(empty, 0.0) if column in USAGE_COLUMNS else values.mask(empty, values.median(), axis=1)
        values.columns = [f"{column}_{slot}" for slot in SKILL_SLOT_ORDER]
        merged = merged.merge(values, left_on=["game_id", "posteam"], right_index=True, how="left", validate="many_to_one")
    return merged


SPECIAL_TEAMS_MODELS = ("punt_distance", "punt_return", "kickoff")


def special_teams_rows(model, frame):
    """Punt distance, punt return (given a non-touchback punt) or kickoff receiving start, with kick_/ret_ role features.

    Punts: posteam kicks. Kickoffs: posteam receives (nflverse convention), and the target is the
    receiving team's own-yardline start, 100 for a return touchdown.
    """
    if model == "kickoff":
        starts = kickoff_starts(frame)
        # nflverse kickoffs have no down; fix the state the simulator will use at every kickoff.
        rows = frame.loc[starts.index].assign(down=1, ydstogo=10)
        target = starts.start
        kicking, receiving = "defteam", "posteam"
    else:
        punt = frame.punt_attempt.eq(1) & ~frame.punt_blocked.eq(1) & frame.kick_distance.notna()
        rows = frame[punt].copy()
        kicking, receiving = "posteam", "defteam"
        if model == "punt_return":
            rows = rows[~rows.touchback.eq(1)].copy()
            rows["punt_landing_100"] = rows.yardline_100 - rows.kick_distance
            target = rows.return_yards.fillna(0.0)
        else:
            target = rows.kick_distance
    for column in SPECIAL_TEAMS_COLUMNS:
        for role, side in (("kick", kicking), ("ret", receiving)):
            source = f"{side}_{column}"
            rows[f"{role}_{column}"] = rows[source] if source in rows else np.nan
    rows["st_ko_rule_2024"] = rows["kick_st_ko_rule_2024"]
    rows["st_ko_rule_2025"] = rows["kick_st_ko_rule_2025"]
    return rows, target


def classify(frame):
    return np.select([frame.incomplete_pass.eq(1), frame.complete_pass.eq(1), frame.rush_attempt.eq(1), frame.sack.eq(1), frame.punt_attempt.eq(1), frame.interception.eq(1), frame.field_goal_attempt.eq(1)], ["incomplete_pass", "complete_pass", "rush", "sack", "punt", "interception", "field_goal_attempt"], "other")


# Trained on throws only and read as P(outcome | throw): a sack or scramble has no air yards, and filling
# them in would teach the network that passes near the average depth are never caught.
ATTEMPT_MODELS = {"comp", "int"}


def model_frame(model, pbp, players, roster, semantics=None):
    """Exclude nonfinite required ridge context after model-specific play selection.

    Legacy standalone frames without any ridge columns retain their old contract.
    Prepared ridge frames always carry the configured context columns, even when
    a sparse target is unavailable throughout the supplied history.
    """
    frame, players, roster = [canonicalize_teams(table) for table in (pbp, players, roster)]
    missing_weather = set(WEATHER_COLUMNS) - set(frame)
    if missing_weather:
        weather = weather_frame(frame)
        for column in missing_weather:
            frame[column] = weather[column]
    if model not in {"penalty_no_play", "penalty_on_play", "penalty_yards", "penalty_first_down", "time", "kickoff"}:
        frame = frame[frame.penalty.eq(0)].copy()
    flag = lambda column: frame[column].eq(1)
    binary = {"sack": "sack", "scramble": "qb_scramble", "int": "interception", "comp": "complete_pass"}
    if model in binary:
        frame = frame[flag("qb_dropback")]
        if model in ATTEMPT_MODELS:
            frame = frame[frame.air_yards.notna() & ~flag("sack") & ~flag("qb_scramble")]
        target = frame[binary[model]].fillna(0).astype(int)
    elif model == "play":
        frame["label"] = np.select([flag("qb_dropback"), flag("rush"), flag("punt_attempt"), flag("field_goal_attempt")], ["pass", "rush", "punt", "kick"], "other")
        frame = frame[frame.label.ne("other")]
        target = frame.label
    elif model in {"target", "rusher"}:
        role = "receiver" if model == "target" else "rusher"
        selected = frame[flag("pass_attempt")] if model == "target" else frame[flag("rush_attempt") & ~flag("qb_scramble")]
        frame = slot_frame(selected, players, roster, role, SimulationRuntime.receiver_columns if model == "target" else SimulationRuntime.rusher_columns)
        target = frame.selection_label
    elif model == "kick":
        frame = frame[flag("field_goal_attempt") | flag("extra_point_attempt")]
        if "kick_attempt" in frame:
            frame = frame[frame.kick_attempt.eq(1)]
        target = (frame.field_goal_result.eq("made") | frame.extra_point_result.eq("good")).astype(int)
    elif model in {"yac", "ay", "punt", "rush", "scramble_yards", "sack_yards", "def_ret"}:
        masks = {"yac": ("pass_attempt", "yards_after_catch"), "ay": ("pass_attempt", "air_yards"), "punt": ("punt_attempt", "net_yards"), "rush": ("rush_attempt", "yards_gained"), "scramble_yards": ("qb_scramble", "yards_gained"), "sack_yards": ("sack", "yards_gained"), "def_ret": ("interception", "return_yards")}
        indicator, outcome = masks[model]
        mask = flag(indicator)
        if model == "rush":
            mask &= ~flag("qb_scramble")
        if model == "def_ret":
            mask |= flag("fumble_lost")
        if model == "yac":
            mask &= frame.air_yards.notna()
        frame = frame[mask & frame[outcome].notna()].copy()
        if model in {"yac", "rush", "scramble_yards"}:
            role = "receiver" if model == "yac" else "rusher"
            positions = roster[["game_id", "player_id", "position"]].drop_duplicates(["game_id", "player_id"])
            frame = frame.drop(columns="position", errors="ignore").merge(positions, left_on=["game_id", f"{role}_player_id"], right_on=["game_id", "player_id"], how="left", validate="many_to_one")
            frame.position = frame.position.replace({"FB": "RB"}).fillna("OTHER")
        if model in {"rush", "scramble_yards"}:
            cols = SimulationRuntime.rusher_columns
            frame = frame.drop(columns=[column for column in cols if column in frame]).merge(players.set_index(["player_id", "game_id"])[cols], left_on=["rusher_player_id", "game_id"], right_index=True, how="left", validate="many_to_one")
        target = frame[outcome]
    elif model in SPECIAL_TEAMS_MODELS:
        frame, target = special_teams_rows(model, frame)
    elif model == "time":
        frame["play_type_cat"] = classify(frame)
        no_play = frame.penalty.eq(1) & frame.play_type.eq("no_play")
        frame.loc[no_play, "play_type_cat"] = "no_play_penalty"
        if "elapsed_time" not in frame:
            raise ValueError("time training requires original prepared elapsed_time column")
        ordinary = frame.penalty.eq(0) & ~frame.play_type.isin(["no_play", "kickoff"])
        frame = frame[(ordinary | no_play) & frame.elapsed_time.between(1, 120)]
        target = frame.elapsed_time
    elif model == "fumble":
        frame["play_outcome"] = np.select([flag("pass_attempt") & flag("sack"), flag("pass_attempt") & flag("complete_pass"), flag("rush_attempt")], ["sack", "completion", "run"], "other")
        frame = frame[frame.play_outcome.ne("other")]
        target = frame.fumble.eq(1).astype(int)
    elif model.startswith("penalty_"):
        real = flag("qb_dropback") | flag("rush") | flag("punt_attempt") | flag("field_goal_attempt")
        no_play = frame.play_type.eq("no_play") & flag("penalty")
        frame["play_type_cat"] = classify(frame)
        if model == "penalty_no_play":
            frame = frame[real | no_play]
            target = no_play.loc[frame.index].astype(int)
        elif model == "penalty_on_play":
            frame = frame[real]
            target = frame.penalty.fillna(0).astype(int)
        else:
            frame = frame[flag("penalty") & frame.penalty_yards.notna() & frame.penalty_team.notna()].copy()
            frame["penalty_context"] = np.where(frame.play_type.eq("no_play"), "no_play", "on_play")
            frame.loc[frame.penalty_context.eq("no_play"), "play_type_cat"] = "not_applicable"
            target = pd.Series(np.where(frame.penalty_team.eq(frame.posteam), -frame.penalty_yards, frame.penalty_yards), index=frame.index) if model == "penalty_yards" else frame.first_down_penalty.fillna(0).astype(int)
    else:
        raise ValueError(f"Unknown model {model}")
    diagnostics = None
    if any(column.startswith("ridge_") for column in frame):
        if semantics == ISOLATED_MU_CONTEXT_V2:
            frame = _neutralize_v2_context(frame)
        eligible, diagnostics = context_eligibility(frame, model, semantics)
        target = target.iloc[np.flatnonzero(eligible.to_numpy())]
        frame = frame.loc[eligible]
    frame = frame.reset_index(drop=True)
    if diagnostics is not None:
        frame.attrs["ridge_context_eligibility"] = diagnostics
    return frame, pd.Series(target.to_numpy())