from __future__ import annotations

from dataclasses import fields
from typing import Iterable

import numpy as np
import pandas as pd

from nflsim.features.bayesian_history import TEAM_DIRECT_LAG_COLUMNS
from nflsim.configuration import v2_optional_context_columns
from nflsim.features.player_value import PV_COLUMNS, PV_DEFENSE_COLUMNS, PV_OFFENSE_COLUMNS
from nflsim.ridge.metrics import (
    TARGETS, RidgeConfig, RidgeTarget, build_design, build_game_context,
    build_target_rows, fit_coefficients, recency_weights,
)
from nflsim.features.weather import WEATHER_COLUMNS
from nflsim.semantics import (
    ISOLATED_MU_CONTEXT_V2, ISOLATED_MU_CONTEXT_V3, ISOLATED_MU_V1, ISOLATED_SEMANTICS, context_available_column, defense_mu_column, offense_column,
    opponent_defense_column, venue_column as ridge_venue_column,
)


TEAM_ALIASES = {"LA": "LAR", "STL": "LAR", "SD": "LAC", "OAK": "LV", "JAC": "JAX", "WSH": "WAS"}


def canonicalize_teams(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in ("team", "home_team", "away_team", "posteam", "defteam", "penalty_team", "club_code", "recent_team"):
        if column in frame:
            frame[column] = frame[column].replace(TEAM_ALIASES)
    if "game_id" in frame:
        frame["game_id"] = frame.game_id.astype("string")
    return frame


def skill_roster(roster: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    required = {"game_id", "team", "player_id", "position"}
    if not required <= set(roster):
        raise ValueError(f"Roster must have {sorted(required)}")
    roster = canonicalize_teams(roster)
    roster["position"] = roster.position.replace({"FB": "RB"})
    missing_location = roster[["game_id", "team"]].isna().any(axis=1)
    if missing_location.any():
        raise ValueError("Missing roster game/team keys")
    identified = roster.player_id.notna() & roster.player_id.astype("string").str.strip().ne("").fillna(False)
    relevant = roster.position.isin(["QB", "RB", "WR", "TE"])
    reasons = {
        "roster_missing_position": roster.position.isna(),
        "roster_non_skill_position": roster.position.notna() & ~relevant,
        "roster_missing_player_id": relevant & ~identified,
    }
    diagnostics = [{"reason": reason, "count": int(mask.sum())} for reason, mask in reasons.items() if mask.any()]
    return roster.loc[relevant & identified].copy(), diagnostics


def prepare_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    frame = canonicalize_teams(pbp)
    if "yac_oe" not in frame and {"yards_after_catch", "xyac_mean_yardage"} <= set(frame):
        frame["yac_oe"] = frame["yards_after_catch"] - frame["xyac_mean_yardage"]
    if "depth_bucket" not in frame and "air_yards" in frame:
        frame["depth_bucket"] = np.where(frame.air_yards >= 25, "deep", np.where(frame.air_yards >= 5, "medium", "short"))
    if "target" not in frame and {"pass_attempt", "receiver_player_id"} <= set(frame):
        frame["target"] = (frame.pass_attempt.eq(1) & frame.receiver_player_id.notna()).astype(int)
    if "explosive" not in frame and "rushing_yards" in frame:
        frame["explosive"] = (frame.rushing_yards.fillna(0) > 15).astype(int)
    return frame


def parse_config(value: dict) -> RidgeConfig:
    names = {field.name for field in fields(RidgeConfig)}
    return RidgeConfig(**{key: item for key, item in value.items() if key in names})


def build_estimates(
    pbp: pd.DataFrame, schedule: pd.DataFrame, roster: pd.DataFrame,
    parameters: dict, targets: Iterable[RidgeTarget] = TARGETS,
    predict_seasons: Iterable[int] | None = None,
    player_value: pd.DataFrame | None = None,
    feature_semantics: str = ISOLATED_MU_V1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Use strictly earlier fits; unavailable warm-up metrics are omitted and context is NaN.

    Published skill/defense metrics are isolated mus (``isolated_mu_v1``): the raw signed
    ridge subject/defense coefficient from the joint fit. Intercept, position, PV and venue/offense
    terms are never folded in. Zero is reserved for unseen levels of an available fit, never
    for targets (such as sparse FTN interception_worthy) with no weighted history.
    With ``player_value``, the joint fit includes the five PV columns and their weekly
    coefficients are emitted as ``player_value_coefficient`` diagnostics only.

    ``isolated_mu_context_v2`` publishes a paired defense output from the subject fit's own
    defense block (no separate defense fit), adds own ``ridge_<t>_defense_mu`` and this-match
    ``ridge_<t>_opponent_defense_effect`` for unpaired targets, and venue context for team fits.
    """
    if feature_semantics not in ISOLATED_SEMANTICS:
        raise ValueError(f"Unsupported feature semantics {feature_semantics}")
    # V3 tables are V2 feature tables; V3 only changes which columns Torch models consume.
    v2 = feature_semantics in {ISOLATED_MU_CONTEXT_V2, ISOLATED_MU_CONTEXT_V3}
    targets = tuple(targets)
    prediction_seasons = None if predict_seasons is None else {int(value) for value in predict_seasons}
    roster, roster_diagnostics = skill_roster(roster)
    if roster.duplicated(["game_id", "team", "player_id"]).any():
        raise ValueError("Duplicate roster game/player keys")
    schedule = canonicalize_teams(schedule)
    schedule["game_id"] = schedule.game_id.astype(str)
    venue_column = next((column for column in ("stadium_id", "game_stadium", "stadium") if column in schedule), None)
    if venue_column is None or schedule[venue_column].isna().any():
        raise ValueError("Explicit scheduled stadium identity required; home-team fallback is forbidden")
    games = build_game_context(schedule)
    team_keys = pd.concat([
        games.assign(team=games.home_team, opponent=games.away_team, is_home=(~games.neutral).astype(float), rest_diff=games.home_rest-games.away_rest),
        games.assign(team=games.away_team, opponent=games.home_team, is_home=0.0, rest_diff=games.away_rest-games.home_rest),
    ], ignore_index=True)
    roster_columns = [column for column in roster if column not in team_keys or column in {"game_id", "team"}]
    player_keys = roster[roster_columns].merge(team_keys, on=["game_id", "team"], how="left", validate="many_to_one", indicator=True)
    unmatched = player_keys.loc[player_keys._merge.ne("both"), ["game_id", "team", "player_id"]]
    if not unmatched.empty:
        raise ValueError(f"Unmatched roster keys: {unmatched.head(5).to_dict('records')}")
    player_keys = player_keys.drop(columns="_merge")
    pbp = prepare_pbp(pbp)
    pbp["game_id"] = pbp.game_id.astype(str)
    player_records, team_records, context_records, diagnostics = [], [], [], list(roster_diagnostics)
    for season, week in sorted(set(zip(games.season, games.week))):
        if prediction_seasons is not None and int(season) not in prediction_seasons:
            continue
        if int(week) == 1:
            print(f"[estimates] season {int(season)}", flush=True)
        current = team_keys[(team_keys.season == season) & (team_keys.week == week)]
        current_players = player_keys[(player_keys.season == season) & (player_keys.week == week)]
        earlier_games = games[(games.season < season) | ((games.season == season) & (games.week < week))]
        history = pbp[pbp.game_id.isin(earlier_games.game_id)]
        prior_roster = roster[roster.game_id.isin(earlier_games.game_id)]
        positions = prior_roster.drop_duplicates("player_id", keep="last").set_index("player_id").position.to_dict()
        context = { (row.game_id, row.team): {"game_id": row.game_id, "team": row.team, "venue": row.venue, "is_home": row.is_home, "rest_diff": row.rest_diff, "days_rest": row.home_rest if row.team == row.home_team else row.away_rest, "opp_days_rest": row.away_rest if row.team == row.home_team else row.home_rest, **{column: getattr(row, column) for column in WEATHER_COLUMNS}} for row in current.itertuples() }
        for record in context.values():
            record.update({f"ridge_{target.name}_{suffix}_effect": np.nan for target in targets if target.scope == "player" and target.player_output for suffix in ("venue", "offense")})
            if v2:
                for target in targets:
                    if not target.player_output and target.scope != "team":
                        continue
                    record[defense_mu_column(target.name)] = np.nan
                    record[context_available_column(defense_mu_column(target.name))] = 0.0
                    record[ridge_venue_column(target.name)] = np.nan
                    optional = [ridge_venue_column(target.name)]
                    if target.scope == "player":
                        optional.append(offense_column(target.name))
                    if target.defense_output is None or target.scope == "team":
                        optional.append(opponent_defense_column(target.name))
                        record[opponent_defense_column(target.name)] = np.nan
                    record.update({context_available_column(column): 0.0 for column in optional})
        for target in targets:
            outputs = [("subject", target.player_output), ("defense", target.defense_output)]
            for side, metric in outputs:
                if metric is None:
                    continue
                if v2 and side == "defense" and target.player_output is not None:
                    continue  # published from the subject fit's own defense block below
                if metric not in parameters:
                    raise ValueError(f"Missing configuration for {metric}")
                config = parse_config(parameters[metric])
                window = earlier_games[earlier_games.season >= season-config.lookback_seasons]
                rows = build_target_rows(history[history.game_id.isin(window.game_id)], target, window, positions,
                                         player_value=player_value)
                if rows.empty:
                    diagnostics.append({"season": int(season), "week": int(week), "metric": metric, "reason": "no_earlier_observations", "availability": "unavailable_warmup", "count": 1})
                    continue
                weights = recency_weights(rows.season.to_numpy(), rows.week.to_numpy(), rows.season_last_week.to_numpy(), int(season), int(week), config) * rows["count"].to_numpy()
                if not (weights > 0).any():
                    diagnostics.append({"season": int(season), "week": int(week), "metric": metric, "reason": "no_positive_history_weights", "availability": "unavailable_warmup", "count": 1})
                    continue
                coefficients = fit_coefficients(
                    rows, build_design(rows, target.scope), config, int(season), int(week))
                lookup = coefficients.set_index(["block", "label"]).coefficient.to_dict()
                def effect(block, label):
                    return float(lookup.get((block, str(label)), 0.0))
                pv_terms = PV_OFFENSE_COLUMNS if side == "subject" else PV_DEFENSE_COLUMNS
                pv_beta = {column: effect("player_value", column) for column in pv_terms} if player_value is not None else {}
                for column, beta in pv_beta.items():
                    diagnostics.append({"season": int(season), "week": int(week), "metric": metric, "side": side,
                                        "reason": "player_value_coefficient", "entity": column, "estimate": beta, "count": 1})
                diagnostics.append({"season": int(season), "week": int(week), "metric": metric, "side": side,
                                    "reason": "intercept_coefficient", "entity": "intercept", "estimate": effect("intercept", "intercept"), "count": 1})

                if side == "subject" and target.scope == "player":
                    for position in ("QB", "WR", "RB", "TE", "OTHER"):
                        diagnostics.append({"season": int(season), "week": int(week), "metric": metric, "reason": "position_prior", "entity": position, "estimate": 0.0, "count": 1})
                        diagnostics.append({"season": int(season), "week": int(week), "metric": metric, "side": side, "reason": "position_coefficient", "entity": position, "estimate": effect("position", position), "count": 1})
                keys = current_players if side == "subject" and target.scope == "player" else current
                for row in keys.itertuples():
                    entity = str(row.player_id) if keys is current_players else str(row.team)
                    block = "subject" if side == "subject" else "defense"
                    estimate = effect(block, entity)
                    record = {"game_id": row.game_id, "game_date": row.game_date, "source": metric, "average": estimate, "bayes_estimate_mean": estimate, "estimand": feature_semantics}
                    if keys is current_players:
                        record.update(player_id=entity, player_name=getattr(row, "player_name", entity))
                        player_records.append(record)
                    else:
                        record["team"] = entity
                        team_records.append(record)
                    if (block, entity) not in lookup:
                        diagnostics.append({"season": int(season), "week": int(week), "metric": metric, "reason": "unseen_entity_prior", "entity": entity, "count": 1})
                if side == "subject" and target.scope == "player":
                    for row in current.itertuples():
                        for block, suffix, label in (("venue", "venue", row.venue), ("offense_team", "offense", row.team)):
                            name = f"ridge_{target.name}_{suffix}_effect"
                            context[(row.game_id, row.team)][name] = effect(block, label)
                            if (block, str(label)) not in lookup:
                                diagnostics.append({"season": int(season), "week": int(week), "metric": metric, "reason": f"unseen_{block}_zero_effect", "entity": str(label), "count": 1})
                if v2 and side == "subject":
                    for row in current.itertuples():
                        record = context[(row.game_id, row.team)]
                        record[defense_mu_column(target.name)] = effect("defense", row.team)
                        record[context_available_column(defense_mu_column(target.name))] = 1.0
                        record[context_available_column(ridge_venue_column(target.name))] = 1.0
                        if target.scope == "team":
                            record[ridge_venue_column(target.name)] = effect("venue", row.venue)
                        else:
                            record[context_available_column(offense_column(target.name))] = 1.0
                            record[ridge_venue_column(target.name)] = effect("venue", row.venue)
                            record[offense_column(target.name)] = effect("offense_team", row.team)
                        if target.defense_output is None or target.scope == "team":
                            record[opponent_defense_column(target.name)] = effect("defense", row.opponent)
                            record[context_available_column(opponent_defense_column(target.name))] = 1.0
                        if target.defense_output is not None:
                            team_records.append({"game_id": row.game_id, "game_date": row.game_date, "source": target.defense_output,
                                                 "average": effect("defense", row.team), "bayes_estimate_mean": effect("defense", row.team),
                                                 "estimand": feature_semantics, "team": str(row.team)})
                    if target.defense_output is not None:
                        diagnostics.append({"season": int(season), "week": int(week), "metric": target.defense_output, "side": "defense",
                                            "reason": "same_subject_fit_defense_block", "entity": metric, "count": 1})
        context_records.extend(context.values())
    return tuple(pd.DataFrame(records) for records in (player_records, team_records, context_records, diagnostics))


def overlay(original: pd.DataFrame, replacements: pd.DataFrame, entity: str) -> pd.DataFrame:
    original = canonicalize_teams(original)
    replacements = canonicalize_teams(replacements)
    keys = [entity, "game_id", "source"]
    if replacements.empty:
        raise ValueError("No preweek ridge estimates; warm-up history is required")
    if replacements.duplicated(keys).any():
        raise ValueError("Duplicate ridge replacement keys")
    covered = set(replacements.source)
    kept = original[~original.source.isin(covered)].copy()
    fresh = replacements.reindex(columns=list(dict.fromkeys([*original.columns, *replacements.columns])))
    for column in fresh:
        if "std" in column or "variance" in column:
            fresh[column] = np.nan
    return pd.concat([kept, fresh], ignore_index=True)


def wide_tables(players: pd.DataFrame, teams: pd.DataFrame, context: pd.DataFrame, player_value: pd.DataFrame | None = None):
    players, teams, context = [canonicalize_teams(frame) for frame in (players, teams, context)]
    def pivot(frame, entity):
        frame = frame.copy()
        frame["game_date"] = pd.to_datetime(frame["game_date"], errors="raise", format="ISO8601")
        frame["average"] = frame.bayes_estimate_mean.combine_first(frame.average)
        keys = [entity, "game_id", "game_date"]
        if frame.duplicated([entity, "game_id", "source"]).any():
            raise ValueError("Duplicate long table keys")
        result = frame.pivot_table(index=keys, columns="source", values="average").reset_index()
        result = result.sort_values([entity, "game_date", "game_id"])
        value_columns = [column for column in result if column not in keys]
        result[value_columns] = result.groupby(entity)[value_columns].ffill()
        return result
    player_wide = pivot(players, "player_id")
    names = players.drop_duplicates("player_id", keep="last").set_index("player_id").get("player_name", pd.Series(dtype=str))
    player_wide["player_name"] = player_wide.player_id.map(names).fillna(player_wide.player_id)
    team_wide = pivot(teams, "team")
    for column in TEAM_DIRECT_LAG_COLUMNS:
        if column in team_wide:
            team_wide[column] = team_wide.groupby("team")[column].shift(1)
    context_columns = [column for column in context if column not in {"game_id", "team"}]
    team_wide = team_wide.drop(columns=[column for column in context_columns if column in team_wide]).merge(context, on=["game_id", "team"], how="left", validate="one_to_one")
    if player_value is not None:
        values = canonicalize_teams(player_value)[["game_id", "team", *PV_COLUMNS]].drop_duplicates(["game_id", "team"])
        values["game_id"] = values.game_id.astype(str)
        team_wide["game_id"] = team_wide.game_id.astype(str)
        team_wide = team_wide.drop(columns=[column for column in PV_COLUMNS if column in team_wide]).merge(
            values, on=["game_id", "team"], how="left", validate="one_to_one")
    aliases = {"opp_def_pass_epa_rate": "def_pass_epa_rate", "opp_def_rush_epa_rate": "def_rush_epa_rate", "opp_pass_epa_rate": "epa_rate", "opp_rushing_epa_rate": "rushing_epa_rate"}
    game_teams = team_wide.groupby("game_id").team.agg(list)
    opponent = [next((team for team in game_teams[row.game_id] if team != row.team), None) for row in team_wide.itertuples()]
    indexed = team_wide.set_index(["game_id", "team"])
    for alias, source in aliases.items():
        if source in indexed:
            team_wide[alias] = indexed[source].reindex(list(zip(team_wide.game_id, opponent))).to_numpy()
    numeric = pd.concat([player_wide.select_dtypes("number"), team_wide.select_dtypes("number")])
    if np.isinf(numeric.to_numpy()).any():
        raise ValueError("Infinite feature values")
    return player_wide, team_wide