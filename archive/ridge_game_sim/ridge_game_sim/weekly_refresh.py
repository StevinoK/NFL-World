from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from ridge_game_sim.build_features import build_feature_tables, selection_input_hashes
from ridge_game_sim.bundle import seal_bundle, validate_bundle, verify_bundle_hash
from ridge_game_sim.feature_builder import canonicalize_teams, skill_roster
from ridge_game_sim.paths import ROOT, digest, output_dir, write_json

FEATURE_FILES = {
    "player_stats.csv", "Team_Stats.csv", "player_stats_wide.csv", "team_stats_wide.csv",
    "ridge_context.csv", "ridge_fallbacks.csv", "feature_manifest.json",
    "retained_player_stats.csv", "retained_team_stats.csv", "retained_manifest.json",
    "retained_diagnostics.csv",
}


def normalize_depth_team(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "depth_team" in result:
        result["depth_team"] = pd.to_numeric(result["depth_team"], errors="coerce").astype("float64")
    return result


def attach_roster_player_names(roster: pd.DataFrame, ids: pd.DataFrame | None = None) -> pd.DataFrame:
    result = roster.copy()
    resolved = pd.Series(pd.NA, index=result.index, dtype="string")
    player_ids = result.player_id.astype("string")

    def merge_names(values: pd.Series) -> None:
        nonlocal resolved
        values = values.astype("string").str.strip()
        valid = values.notna() & values.ne("") & values.ne(player_ids)
        resolved = resolved.combine_first(values.where(valid))

    if "football_name" in result:
        merge_names(result.football_name)
    if "full_name" in result:
        merge_names(result.full_name)
    if {"first_name", "last_name"} <= set(result):
        first = result.first_name.astype("string").str.strip().fillna("")
        last = result.last_name.astype("string").str.strip().fillna("")
        merge_names((first + " " + last).str.strip())
    if ids is not None and {"gsis_id", "name"} <= set(ids):
        lookup = ids.dropna(subset=["gsis_id", "name"]).drop_duplicates("gsis_id", keep="last")
        names = result[["player_id"]].merge(lookup[["gsis_id", "name"]], left_on="player_id", right_on="gsis_id", how="left")["name"]
        names.index = result.index
        merge_names(names)
    if "player_name" in result:
        merge_names(result.player_name)
    result["player_name"] = resolved.fillna(player_ids).astype(str)
    return result


def add_scheduled_roster_slots(roster: pd.DataFrame, schedule: pd.DataFrame, season: int,
                               injuries: pd.DataFrame | None = None) -> pd.DataFrame:
    """Carry each team's latest available active roster into its unplayed scheduled games."""
    roster = canonicalize_teams(roster)
    schedule = canonicalize_teams(schedule)
    roster = roster.loc[roster.season.eq(season)].copy()
    scheduled = schedule.loc[
        schedule.season.eq(season) & schedule.game_type.eq("REG") & schedule.week.between(1, 18)
    ]
    team_slots = pd.concat([
        scheduled[["game_id", "season", "week", "game_date", "home_team"]].rename(columns={"home_team": "team"}),
        scheduled[["game_id", "season", "week", "game_date", "away_team"]].rename(columns={"away_team": "team"}),
    ], ignore_index=True)
    additions = []
    for team, games in team_slots.groupby("team", sort=False):
        team_rows = roster.loc[roster.team.astype(str).eq(str(team))]
        if team_rows.empty:
            continue
        latest_week = int(pd.to_numeric(team_rows.week, errors="coerce").max())
        latest = team_rows.loc[pd.to_numeric(team_rows.week, errors="coerce").eq(latest_week)]
        for game in games.loc[games.week.astype(int).gt(latest_week)].itertuples(index=False):
            exists = roster.game_id.astype(str).eq(str(game.game_id)) & roster.team.astype(str).eq(str(team))
            if exists.any():
                continue
            projected = latest.copy()
            projected["game_id"] = str(game.game_id)
            projected["season"] = int(season)
            projected["week"] = int(game.week)
            if "game_date" in projected:
                projected["game_date"] = game.game_date
            projected["roster_source_week"] = latest_week
            projected["roster_is_projected"] = True
            additions.append(projected)
    if not additions:
        return roster
    actual = roster.copy()
    actual["roster_is_projected"] = False
    combined = pd.concat([actual, *additions], ignore_index=True, sort=False)
    combined = combined.drop_duplicates(["game_id", "team", "player_id"], keep="last")
    if injuries is not None and {"season", "week", "gsis_id", "report_status"} <= set(injuries):
        updates = injuries.drop_duplicates(["season", "week", "gsis_id"], keep="last").rename(columns={"gsis_id": "player_id"})
        updates = updates[["season", "week", "player_id", "report_status"]]
        combined = combined.drop(columns=["report_status"], errors="ignore").merge(
            updates, on=["season", "week", "player_id"], how="left", validate="many_to_one")
    return combined


def attach_current_depth_order(roster: pd.DataFrame, depth: pd.DataFrame, season: int) -> pd.DataFrame:
    result = roster.copy()
    result["position"] = result.position.replace({"FB": "RB"})
    result = result.drop(columns=["depth_team"], errors="ignore")
    if {"season", "week", "gsis_id", "depth_team"} <= set(depth):
        lookup = depth.drop_duplicates(["season", "week", "gsis_id"], keep="first").rename(columns={"gsis_id": "player_id"})
        merged = result.merge(lookup[["season", "week", "player_id", "depth_team"]],
                              on=["season", "week", "player_id"], how="left", validate="many_to_one")
        return normalize_depth_team(merged)
    if not {"dt", "team", "gsis_id", "pos_abb", "pos_rank"} <= set(depth):
        raise ValueError(f"Depth chart schema unsupported; columns received: {sorted(depth.columns)}")
    lookup = depth.copy()
    lookup["dt"] = pd.to_datetime(lookup.dt, errors="coerce", utc=True).dt.tz_convert(None)
    lookup["pos_abb"] = lookup.pos_abb.replace({"FB": "RB"})
    lookup["pos_rank"] = pd.to_numeric(lookup.pos_rank, errors="coerce")
    lookup = lookup[lookup.pos_abb.isin(["QB", "RB", "WR", "TE"])]
    lookup = lookup.sort_values("dt").drop_duplicates(["gsis_id", "pos_abb"], keep="last")
    lookup = lookup.rename(columns={"gsis_id": "player_id", "pos_abb": "position",
                                    "pos_rank": "_depth_rank", "team": "_depth_team"})
    result = result.merge(lookup[["player_id", "position", "_depth_rank", "_depth_team"]],
                          on=["player_id", "position"], how="left", validate="many_to_one")
    same_team = result.team.astype(str).eq(result._depth_team.astype(str))
    result["depth_team"] = result._depth_rank.where(same_team)
    return normalize_depth_team(result.drop(columns=["_depth_rank", "_depth_team"]))


def prepare_current_season_inputs(season: int, destination: Path) -> dict[str, Path]:
    """Refresh only the active season and preserve prior frozen PBP/roster/schedule inputs."""
    import nfl_data_py as nfl
    from bayesian_feature_builder import load_pbp_data, load_snap_counts

    destination = output_dir(destination)
    old_dir = ROOT / "artifacts/inputs"
    required = [old_dir / name for name in ("pbp.parquet", "schedule.parquet", "roster.parquet", "snap_counts.parquet")]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Initial frozen inputs missing; run prepare_inputs first: {missing}")

    old_pbp, old_schedule, old_roster, old_snaps = [pd.read_parquet(path) for path in required]
    new_pbp = load_pbp_data([season], cache_path=ROOT / "artifacts/cache/pbp")
    new_schedule = canonicalize_teams(nfl.import_schedules([season]))
    new_schedule = new_schedule[new_schedule.game_type.eq("REG")].copy()
    new_schedule["game_date"] = new_schedule.gameday
    if "location" not in new_schedule:
        new_schedule["location"] = "Home"
    new_roster = canonicalize_teams(nfl.import_weekly_rosters([season]).copy())
    team_games = pd.concat([
        new_schedule[["game_id", "season", "week", "home_team"]].rename(columns={"home_team": "team"}),
        new_schedule[["game_id", "season", "week", "away_team"]].rename(columns={"away_team": "team"}),
    ], ignore_index=True)
    new_roster = new_roster.merge(team_games, on=["season", "week", "team"], how="inner", validate="many_to_one")
    injuries = nfl.import_injuries([season])
    injury_updates = None
    if {"season", "week", "gsis_id", "report_status"} <= set(injuries):
        injury_updates = injuries.drop_duplicates(["season", "week", "gsis_id"], keep="last").rename(columns={"gsis_id": "player_id"})
    new_roster = add_scheduled_roster_slots(new_roster, new_schedule, season, injury_updates)
    depth = nfl.import_depth_charts([season])
    new_roster = attach_current_depth_order(new_roster, depth, season)
    try:
        ids = nfl.import_ids()
    except Exception:
        ids = pd.DataFrame()
    new_roster = attach_roster_player_names(new_roster, ids)
    new_roster = new_roster.drop_duplicates(["game_id", "team", "player_id"])
    new_snaps = canonicalize_teams(load_snap_counts([season]))

    old_pbp = old_pbp.loc[pd.to_numeric(old_pbp.game_id.astype(str).str[:4], errors="coerce").__lt__(season)]
    old_schedule = old_schedule.loc[pd.to_numeric(old_schedule.game_id.astype(str).str[:4], errors="coerce").lt(season)]
    old_roster = old_roster.loc[pd.to_numeric(old_roster.game_id.astype(str).str[:4], errors="coerce").lt(season)]
    old_snaps = old_snaps.loc[pd.to_numeric(old_snaps.game_id.astype(str).str[:4], errors="coerce").lt(season)]
    frames = {
        "pbp.parquet": pd.concat([old_pbp, new_pbp], ignore_index=True, sort=False),
        "schedule.parquet": pd.concat([old_schedule, new_schedule], ignore_index=True, sort=False).drop_duplicates("game_id", keep="last"),
        "roster.parquet": normalize_depth_team(pd.concat([old_roster, new_roster], ignore_index=True, sort=False).drop_duplicates(["game_id", "team", "player_id"], keep="last")),
        "snap_counts.parquet": pd.concat([old_snaps, new_snaps], ignore_index=True, sort=False).drop_duplicates(["game_id", "gsis_id", "position"], keep="last"),
    }
    paths = {}
    for name, frame in frames.items():
        frame = canonicalize_teams(frame)
        path = destination / name
        frame.to_parquet(path, index=False)
        paths[name] = path
    write_json(destination / "input_manifest.json", {"active_season": season, "prior_seasons_frozen": True,
        "files": {name: digest(path) for name, path in paths.items()}, "source": "nfl_data_py current-season refresh"})
    return paths


def _attest_frozen_selection(parameters_path: Path, base_bundle: Path, inputs: dict[str, Path]):
    persisted_parameters = base_bundle / "ridge_selection_parameters.json"
    if persisted_parameters.exists():
        persisted_manifest = json.loads((base_bundle / "feature_manifest.json").read_text())
        if persisted_manifest.get("selection_sha256") == digest(persisted_parameters):
            selected = json.loads(persisted_parameters.read_text())
            if selected.get("input_hashes") == selection_input_hashes(*(pd.read_parquet(inputs[name]) for name in ("pbp.parquet", "schedule.parquet", "roster.parquet"))):
                return persisted_parameters
            return selected
    parameters_path = Path(parameters_path)
    if not parameters_path.exists():
        raise FileNotFoundError(f"Frozen 2024 ridge parameters not found: {parameters_path}")
    selected = json.loads(parameters_path.read_text())
    if selected.get("validation_season") != 2024 or selected.get("max_selection_season") != 2024:
        raise ValueError("Weekly refresh requires parameters selected and validated only on 2024")
    current = selection_input_hashes(*(pd.read_parquet(inputs[name]) for name in ("pbp.parquet", "schedule.parquet", "roster.parquet")))
    base_feature_manifest = base_bundle / "feature_manifest.json"
    if base_feature_manifest.exists():
        manifest = json.loads(base_feature_manifest.read_text())
        if manifest.get("selection_sha256") == digest(parameters_path):
            old_inputs_match = True
            for name, checksum in manifest.get("inputs", {}).items():
                candidate = Path(name)
                if not candidate.is_absolute():
                    candidate = ROOT.parent / candidate
                if not candidate.is_file() or digest(candidate) != checksum:
                    old_inputs_match = False
                    break
            if old_inputs_match:
                migrated = dict(selected)
                migrated["input_hashes"] = current
                migrated["migrated_from_parameter_sha256"] = digest(parameters_path)
                return migrated
    if selected.get("input_hashes") == current:
        return parameters_path
    if not selected.get("parameters"):
        raise ValueError("Parameter file has no frozen per-metric settings")
    return selected


def update_bundle_feature_lineage(candidate: Path) -> None:
    manifest_path = candidate / "training_manifest.json"
    feature_path = candidate / "feature_manifest.json"
    state = json.loads(manifest_path.read_text())
    feature_hash = digest(feature_path)
    state["lineage"]["features"] = feature_hash
    for record in state["trained"].values():
        record["lineage"] = dict(state["lineage"])
    write_json(manifest_path, state)


def refresh_weekly_bundle(season: int, base_bundle: Path = ROOT / "artifacts/runtime",
                          parameter_file: Path = ROOT / "config/ridge_parameters.json") -> Path:
    """Download current-season data, generate future pregame features, and publish a new immutable bundle."""
    from ridge_game_sim.build_features import build_feature_tables

    if int(season) != datetime.now().year:
        raise ValueError(f"Weekly refresh is limited to the active season ({datetime.now().year}); historical seasons stay frozen")
    base_bundle = Path(base_bundle).resolve()
    if not (base_bundle / "bundle_manifest.json").exists():
        raise ValueError("Base Ridge bundle must be sealed before weekly feature refresh")
    validate_bundle(base_bundle, require_ready=True)
    base_hash = verify_bundle_hash(base_bundle)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work = output_dir(ROOT / "artifacts/weekly_refresh" / f"{season}_{stamp}")
    inputs = prepare_current_season_inputs(season, work / "inputs")
    frozen_parameters = _attest_frozen_selection(parameter_file, base_bundle, inputs)
    if isinstance(frozen_parameters, dict):
        frozen_path = work / "ridge_parameters.json"
        write_json(frozen_path, frozen_parameters)
    else:
        frozen_path = frozen_parameters
    feature_dir = output_dir(work / "features")
    input_hashes = {name: digest(path) for name, path in inputs.items()}
    feature_manifest = build_feature_tables(
        pd.read_parquet(inputs["pbp.parquet"]), pd.read_parquet(inputs["schedule.parquet"]),
        pd.read_parquet(inputs["roster.parquet"]), pd.read_parquet(inputs["snap_counts.parquet"]),
        frozen_path, feature_dir, reselect_if_training_history_changed=True, source_hashes=input_hashes,
        input_paths=tuple(inputs.values()),
        prediction_seasons=[season], prior_bundle=base_bundle,
    )
    feature_manifest["weekly_refresh"] = {"season": season, "base_bundle_hash": base_hash,
        "timestamp": datetime.now().isoformat(), "frozen_selection_sha256": digest(frozen_path)}
    shutil.copy2(frozen_path, feature_dir / "ridge_selection_parameters.json")
    feature_manifest["selection_sha256"] = digest(feature_dir / "ridge_selection_parameters.json")
    write_json(feature_dir / "feature_manifest.json", feature_manifest)

    candidate = output_dir(work / "candidate")
    shutil.copytree(base_bundle, candidate, dirs_exist_ok=True)
    published_inputs = output_dir(candidate / "weekly_inputs")
    for name in ("schedule.parquet", "roster.parquet"):
        shutil.copy2(inputs[name], published_inputs / name)
    write_json(published_inputs / "input_manifest.json", {
        "active_season": season,
        "files": {name: digest(published_inputs / name) for name in ("schedule.parquet", "roster.parquet")},
        "source_manifest_sha256": digest(inputs["pbp.parquet"].parent / "input_manifest.json"),
    })
    for name in FEATURE_FILES | {"ridge_selection_parameters.json"}:
        source = feature_dir / name
        if source.exists():
            shutil.copy2(source, candidate / name)
    (candidate / "bundle_manifest.json").unlink(missing_ok=True)
    update_bundle_feature_lineage(candidate)
    validate_bundle(candidate, require_ready=True)
    bundle_hash = seal_bundle(candidate)
    final = ROOT / "artifacts/bundles" / bundle_hash
    if final.exists():
        if verify_bundle_hash(final) != bundle_hash:
            raise ValueError("Existing hash-addressed bundle is corrupt")
        shutil.rmtree(candidate)
    else:
        output_dir(final.parent)
        shutil.move(str(candidate), str(final))
    write_json(work / "refresh_manifest.json", {"status": "complete", "season": season,
        "base_bundle_hash": base_hash, "new_bundle_hash": bundle_hash, "bundle_path": str(final),
        "feature_manifest_sha256": digest(final / "feature_manifest.json"),
        "input_manifest_sha256": digest(inputs["pbp.parquet"].parent / "input_manifest.json")})
    return final
