from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

from nflsim.names import roster_display_names
from nflsim.paths import ROOT


RIDGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEAM_DESC_ALIASES = {"LA": "LAR", "LAR": "LA", "LV": "OAK", "OAK": "LV", "SD": "LAC", "STL": "LA"}
DEFENSIVE_METRICS = {
    "int_rate", "sack_rate", "rush_yards_per_carry",
    "passing_massey_defense", "rushing_massey_defense",
}
IDENTIFIER_COLUMNS = {
    "player_id", "player_name", "team", "venue", "stadium_id", "game_id", "next_game_id",
    "game_date", "season", "week", "position", "name", "label",
}


def numeric_metrics(table: pd.DataFrame) -> list[str]:
    return [column for column in table.select_dtypes(include="number")
            if column not in IDENTIFIER_COLUMNS and not column.startswith("Unnamed:")]


def series_color(series: str, index: int, block: str, color: str,
                 selected_team: str, team_colors: dict[str, str]) -> str:
    fallback = DEFAULT_COLORS[index % len(DEFAULT_COLORS)]
    if block not in {"offense_team", "defense", "lineup_value"}:
        return fallback
    team = series if color == "entity" else selected_team if color == "None" else ""
    return team_colors.get(team) or team_colors.get(TEAM_DESC_ALIASES.get(team, "")) or fallback


def observed_graph_data(frame: pd.DataFrame, today=None) -> pd.DataFrame:
    if "game_date" not in frame:
        return frame.copy()
    dates = pd.to_datetime(frame["game_date"], format="mixed", errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
    cutoff = pd.Timestamp(today if today is not None else pd.Timestamp.today()).normalize()
    return frame.loc[dates.notna() & dates.le(cutoff)].copy()


def current_artifacts(root: Path = RIDGE_ROOT) -> Path:
    pointer = root / "config" / "last_upload.json"
    if pointer.exists():
        bundle_hash = json.loads(pointer.read_text(encoding="utf-8"))["bundle_hash"]
        bundle = root / "artifacts" / "bundles" / bundle_hash
        if not bundle.is_dir():
            raise FileNotFoundError(f"Active bundle {bundle_hash} is not present locally: {bundle}")
        return bundle
    return root / "artifacts" / "runtime"


def artifact_signature(directory: Path) -> tuple:
    return tuple((path.name, path.stat().st_mtime_ns, path.stat().st_size)
                 for path in sorted(directory.iterdir()) if path.suffix in {".csv", ".parquet", ".json"})


def player_value_graph_frames(directory: Path, frames: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """Weekly PV coefficients (block player_value) and team lineup PV (block lineup_value)."""
    result = []
    dated = pd.concat([frame[["game_id", "game_date"]] for frame in frames if "game_date" in frame], ignore_index=True) if frames else pd.DataFrame(columns=["game_id", "game_date"])
    dated = dated.dropna().drop_duplicates("game_id")
    lineup_path = directory / "player_value.csv"
    if lineup_path.exists():
        values = pd.read_csv(lineup_path, dtype={"game_id": str, "team": str})
        columns = [column for column in values if column.startswith("pv_")]
        frame = values.melt(id_vars=["game_id", "team"], value_vars=columns, var_name="metric", value_name="estimate")
        frame = frame.rename(columns={"team": "label"}).merge(dated, on="game_id", how="left")
        frame["block"], frame["side"] = "lineup_value", np.where(frame.metric.eq("pv_def_dl"), "defense", "subject")
        frame["coefficient"], frame["weighted_exposure"] = np.nan, np.nan
        frame["value_kind"] = "Pregame lineup plus-minus"
        result.append(frame)
    coefficient_path = directory / "ridge_pv_coefficients.csv"
    if coefficient_path.exists() and not dated.empty:
        betas = pd.read_csv(coefficient_path)
        if not betas.empty:
            week_dates = dated.assign(season=dated.game_id.str[:4].astype(int), week=dated.game_id.str[5:7].astype(int))
            week_dates["game_date"] = pd.to_datetime(week_dates.game_date, errors="coerce", format="mixed")
            first = week_dates.groupby(["season", "week"], as_index=False).game_date.min()
            frame = betas.merge(first, on=["season", "week"], how="inner")
            frame["game_id"] = frame.season.astype(str) + "_" + frame.week.astype(int).astype(str).str.zfill(2) + "_PV"
            frame = frame.rename(columns={"term": "label", "beta": "coefficient"})
            frame["estimate"] = frame.coefficient
            frame["block"], frame["weighted_exposure"] = "player_value", np.nan
            frame["value_kind"] = "Weekly PV coefficient"
            result.append(frame[["game_id", "game_date", "label", "metric", "estimate", "coefficient", "block", "side",
                                 "weighted_exposure", "value_kind"]])
    return result


def roster_names(directory: Path) -> dict[str, str]:
    """Nickname + last name for every player in the frozen and the bundle's current-season rosters."""
    names = {}
    for path in (ROOT / "artifacts/inputs/roster.parquet", Path(directory) / "weekly_inputs/roster.parquet"):
        if path.exists():
            names.update(roster_display_names(pd.read_parquet(path)))
    return names


def read_graph_data(directory: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    if (directory / "weekly_coefficients.parquet").exists():
        history = pd.read_parquet(directory / "weekly_coefficients.parquet")
        names_path = directory / "player_names.csv"
        names = pd.read_csv(names_path) if names_path.exists() else pd.DataFrame(columns=["label", "name"])
        return history, {**dict(zip(names["label"].astype(str), names["name"].astype(str))), **roster_names(directory)}
    frames = []
    names = {}
    for filename, identity, block in (("player_stats_wide.csv", "player_id", "subject"),
                                       ("team_stats_wide.csv", "team", "offense_team"),
                                       ("venue_stats_wide.csv", "venue", "venue")):
        if not (directory / filename).exists():
            continue
        table = pd.read_csv(directory / filename)
        if "player_name" in table:
            known = table.dropna(subset=[identity, "player_name"])
            names.update(dict(zip(known[identity].astype(str), known["player_name"].astype(str))))
        columns = [column for column in numeric_metrics(table)
                   if (not column.startswith("ridge_") or not column.endswith(("_venue_effect", "_offense_effect")))
                   and not column.startswith("pv_")]
        identifiers = [column for column in (identity, "game_id", "game_date") if column in table]
        frame = table.melt(id_vars=identifiers, value_vars=columns, var_name="metric", value_name="estimate")
        frame = frame.rename(columns={identity: "label"})
        frame["block"] = block
        if block == "offense_team":
            defensive = (frame["metric"].isin(DEFENSIVE_METRICS)
                         | frame["metric"].str.startswith("def_")
                         | frame["metric"].str.endswith("_allowed"))
            frame.loc[defensive, "block"] = "defense"
        frame["side"] = np.where(frame["block"].eq("defense"), "defense", "subject")
        frame["coefficient"] = np.nan
        frame["weighted_exposure"] = np.nan
        frame["value_kind"] = "Production estimate"
        frames.append(frame)
    context_path = directory / "ridge_context.csv"
    context = pd.read_csv(context_path) if context_path.exists() else pd.DataFrame()
    for column in context:
        if not column.startswith("ridge_") or not column.endswith(("_venue_effect", "_offense_effect")):
            continue
        venue = column.endswith("_venue_effect")
        suffix = "_venue_effect" if venue else "_offense_effect"
        target = column[len("ridge_"):-len(suffix)]
        frame = context[["game_id", "venue" if venue else "team", column]].rename(
            columns={"venue" if venue else "team": "label", column: "coefficient"})
        frame["metric"] = target
        frame["block"] = "venue" if venue else "offense_team"
        frame["side"] = "subject"
        frame["estimate"] = frame["coefficient"]
        frame["weighted_exposure"] = np.nan
        frame["value_kind"] = "Marginal effect"
        frames.append(frame.drop_duplicates(["game_id", "label"]))
    frames.extend(player_value_graph_frames(directory, frames))
    history = pd.concat(frames, ignore_index=True).dropna(subset=["estimate", "label"])
    game_dates = history.dropna(subset=["game_date"]).drop_duplicates("game_id").set_index("game_id")["game_date"]
    history["game_date"] = history["game_date"].fillna(history["game_id"].map(game_dates))
    parsed = history["game_id"].str.split("_", expand=True)
    history["season"] = pd.to_numeric(parsed[0], errors="raise").astype(int)
    history["week"] = pd.to_numeric(parsed[1], errors="raise").astype(int)
    history["label"] = history["label"].astype(str)
    return history.reset_index(drop=True), {**names, **roster_names(directory)}


def build_figure(frame: pd.DataFrame, x: str, y: str, mark: str,
                 color: str | None = "entity", facet: str | None = None,
                 size: str | None = None, *, color_map: dict | None = None,
                 labels: dict | None = None, title: str = "", legend_title: str | None = None,
                 height: int = 550, font_size: int = 14, line_width: int = 2,
                 marker_size: int = 7, show_grid: bool = True,
                 background: str = "rgba(0,0,0,0)", text_color: str | None = None):
    ordered = frame.sort_values(["season", "week", "label"]).copy()
    ordered["row_id"] = ordered.index
    hover = [column for column in ("metric", "entity", "season", "week", "value_kind") if column in ordered]
    kwargs = dict(data_frame=ordered, x=x, y=y, color=color, facet_col=facet,
                  facet_col_wrap=3 if facet else 0, hover_data=hover,
                  color_discrete_sequence=list(color_map.values()) if color_map and not color else DEFAULT_COLORS,
                  color_discrete_map=color_map, labels=labels)
    if mark == "Line":
        figure = px.line(**kwargs, line_group="entity", markers=True, custom_data=["row_id"])
    elif mark == "Scatter":
        figure = px.scatter(**kwargs, size=size, custom_data=["row_id"])
    elif mark == "Bar":
        keys = list(dict.fromkeys([x] + ([color] if color else []) + ([facet] if facet else [])))
        kwargs["data_frame"] = ordered.drop_duplicates(keys) if y in keys else ordered.groupby(keys, as_index=False, dropna=False)[y].mean()
        kwargs["hover_data"] = None
        figure = px.bar(**kwargs, barmode="group")
    elif mark == "Box":
        figure = px.box(**kwargs, points="outliers")
    else:
        figure = px.histogram(**kwargs, histfunc="avg")
    figure.update_layout(height=height, margin=dict(l=20, r=20, t=60 if title else 40, b=30),
                         title_text=title, font=dict(size=font_size, color=text_color),
                         legend_title_text=legend_title if legend_title is not None else color or "",
                         dragmode="select", paper_bgcolor=background, plot_bgcolor=background)
    if mark in {"Line", "Scatter"} and size is None:
        figure.update_traces(marker_size=marker_size)
    if mark == "Line":
        figure.update_traces(line_width=line_width)
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(showgrid=show_grid, gridcolor="#e8ebee", zerolinecolor="#dde1e5")
    if labels:
        if x in labels:
            figure.update_xaxes(title_text=labels[x])
        if y in labels:
            figure.update_yaxes(title_text=labels[y])
    return figure