"""Report builder for league rating simulations.

Reads combined legacy `*_games.csv` or Ridge `games.csv` simulation rows and renders:
  1. Violin contours of point spread by team, team-colored, descending by median spread.
  2. Points-for vs points-against scatter with team logos as the markers.
  3. A grid of LightSource-shaded scorigami heatmaps, one per team vs. the league.
"""

from __future__ import annotations

import argparse
import base64
import io
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nfl_data_py as nfl
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, LightSource, LinearSegmentedColormap
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.ticker import MaxNLocator
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde

from nflsim import ui
from nflsim.dashboard.power_ratings import load_ratings, our_team, save_ratings
from nflsim.paths import ROOT
from nflsim.sim.synthetic_games import team_games

ui.style_matplotlib()
LOGO_CACHE_DIR = ROOT / "outputs/team_logo_cache"
RELIEF_EXAGGERATION = 15.0
SCORIGAMI_CMAP = LinearSegmentedColormap.from_list(
    "deep_soil_to_meadow",
    [
        (0.00, "#8a6a3c"),
        (0.15, "#9c7f45"),
        (0.28, "#948b40"),
        (0.38, "#74933a"),
        (0.55, "#539030"),
        (0.72, "#378026"),
        (0.87, "#20701f"),
        (1.00, "#0f5a1a"),
    ],
)
# team_stats_wide uses LA/LV; nflverse team_desc historically carries both spellings.
TEAM_DESC_ALIASES = {"LA": "LAR", "LAR": "LA", "LV": "OAK", "OAK": "LV"}


# ---------------------------------------------------------------- data loading


def load_games(folder: Path) -> pd.DataFrame:
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    ridge_games = next((path for path in (folder / "games.csv", folder / "synthetic_games.csv") if path.is_file()), None)
    if ridge_games:
        games = pd.read_csv(ridge_games)
        required = {"home_team", "away_team", "home_score", "away_score", "spread", "total"}
        if not required <= set(games.columns):
            raise ValueError(f"Ridge games.csv missing columns: {sorted(required - set(games.columns))}")
        return games
    combined = sorted(folder.glob("synthetic_home_week_*_games.csv"))
    if combined:
        return pd.read_csv(combined[-1])
    per_game = [p for p in folder.glob("*_games.csv") if "_SYN_" in p.name]
    if not per_game:
        raise FileNotFoundError(f"No Ridge games.csv or synthetic *_games.csv found in {folder}")
    return pd.concat([pd.read_csv(p) for p in per_game], ignore_index=True)


def team_style() -> pd.DataFrame:
    desc = nfl.import_team_desc()[["team_abbr", "team_color", "team_color2", "team_logo_espn"]]
    return desc.set_index("team_abbr")


def style_for(styles: pd.DataFrame, team: str) -> pd.Series:
    if team in styles.index:
        return styles.loc[team]
    alias = TEAM_DESC_ALIASES.get(team)
    if alias and alias in styles.index:
        return styles.loc[alias]
    return pd.Series({"team_color": "#666666", "team_color2": "#222222", "team_logo_espn": None})


def logo_image(url: Optional[str], team: str, target_px: int = 72) -> Optional[np.ndarray]:
    """Fetch (and disk-cache) a team logo. Returns None when unavailable offline."""
    if not url:
        return None
    LOGO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = LOGO_CACHE_DIR / f"{team}.png"
    if not cached.exists():
        try:
            with urllib.request.urlopen(url, timeout=15) as response:
                cached.write_bytes(response.read())
        except Exception:
            return None
    try:
        image = Image.open(cached).convert("RGBA")
    except Exception:
        return None

    # Source logos vary from 500px crests to 4096px wordmarks with huge transparent margins,
    # so trim to the visible pixels and rescale every team to one on-screen size.
    bbox = image.getchannel("A").getbbox()
    if bbox:
        image = image.crop(bbox)
    scale = target_px / max(image.size)
    return np.asarray(image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.LANCZOS))


# ------------------------------------------------------------------- violins


def render_spread_violins(
    games: pd.DataFrame, styles: pd.DataFrame, out_path: Path, trim: float = 0.02
) -> Path:
    neutral = "neutral_site" in games and games.neutral_site.fillna(False).astype(bool).all()
    order = (
        games.groupby("home_team")["spread"].median().sort_values(ascending=False).index.tolist()
    )
    data = [games.loc[games["home_team"] == team, "spread"].to_numpy(dtype=float) for team in order]

    fig, ax = plt.subplots(figsize=(max(12.0, 0.46 * len(order)), 7.5))
    half_width = 0.45
    # matplotlib's violinplot always runs the KDE past the data, so build each body from a
    # gaussian_kde evaluated only inside the team's central quantile band.
    for position, (values, team) in enumerate(zip(data, order)):
        low, high = np.quantile(values, [trim, 1.0 - trim])
        if high <= low:
            continue
        grid = np.linspace(low, high, 200)
        density = gaussian_kde(values)(grid)
        offsets = density / density.max() * half_width

        team_style_row = style_for(styles, team)
        ax.fill_betweenx(
            grid,
            position - offsets,
            position + offsets,
            facecolor=team_style_row["team_color"],
            edgecolor=team_style_row["team_color2"],
            linewidth=0.9,
            alpha=0.92,
            zorder=2,
        )

    medians = [float(np.median(values)) for values in data]
    ax.scatter(np.arange(len(order)), medians, color="white", edgecolor="black", zorder=3, s=26)
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)

    ax.set_xlim(-0.8, len(order) - 0.2)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, rotation=90, fontsize=9)
    site_label = "neutral site" if neutral else "home"
    ax.set_ylabel(f"Point margin vs. the league ({site_label})")
    ax.set_title(
        f"Simulated Point Margin vs. the League ({site_label}) — middle {100 * (1 - 2 * trim):.0f}%"
    )
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ------------------------------------------------------ points for / against


def render_points_grid(games: pd.DataFrame, styles: pd.DataFrame, out_path: Path) -> Path:
    summary = (
        games.groupby("home_team")
        .agg(points_for=("home_score", "mean"), points_against=("away_score", "mean"))
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(11, 9))
    for row in summary.itertuples():
        team_style_row = style_for(styles, row.home_team)
        image = logo_image(team_style_row["team_logo_espn"], row.home_team)
        if image is None:
            ax.scatter(
                row.points_for,
                row.points_against,
                color=team_style_row["team_color"],
                edgecolor="black",
                s=180,
                zorder=3,
            )
            ax.annotate(row.home_team, (row.points_for, row.points_against), fontsize=8, ha="center", va="center")
            continue
        ax.add_artist(
            AnnotationBbox(
                OffsetImage(image, zoom=0.62),
                (row.points_for, row.points_against),
                frameon=False,
                zorder=3,
            )
        )

    mean_for = float(summary["points_for"].mean())
    mean_against = float(summary["points_against"].mean())
    ax.axvline(mean_for, color="grey", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.axhline(mean_against, color="grey", linestyle="--", linewidth=1.0, alpha=0.8)

    pad = 1.5
    ax.set_xlim(summary["points_for"].min() - pad, summary["points_for"].max() + pad)
    ax.set_ylim(summary["points_against"].min() - pad, summary["points_against"].max() + pad)
    # Better defense is fewer points allowed, so the strongest teams land bottom-right.
    ax.invert_yaxis()
    ax.set_xlabel("Average points scored vs. the league")
    ax.set_ylabel("Average points allowed to the league")
    ax.set_title("Offense vs. Defense against the League")
    ax.grid(alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------- scorigami


def scorigami_rgb(
    home_scores: pd.Series,
    away_scores: pd.Series,
    axis_max: int,
    sigma: float = 2.4,
    vert_exag: float = RELIEF_EXAGGERATION,
) -> Tuple[np.ndarray, List[int]]:
    all_scores = list(range(0, axis_max + 1))
    counts = pd.crosstab(away_scores, home_scores).reindex(index=all_scores, columns=all_scores, fill_value=0)
    z = gaussian_filter(counts.to_numpy(dtype=float), sigma=sigma)

    levels = MaxNLocator(nbins=10).tick_values(0, max(float(z.max()), 1.0))
    norm = BoundaryNorm(levels, ncolors=SCORIGAMI_CMAP.N, clip=True)
    light = LightSource(azdeg=315, altdeg=45)
    elevation = z / max(float(z.max()), 1e-9)
    rgb = light.shade_rgb(
        SCORIGAMI_CMAP(norm(z))[:, :, :3],
        elevation,
        vert_exag=vert_exag,
        dx=1.0,
        dy=1.0,
        blend_mode="overlay",
    )
    return rgb, all_scores


def render_scorigami_grid(games: pd.DataFrame, out_path: Path, columns: int = 8) -> Path:
    teams = sorted(games["home_team"].dropna().unique().tolist())
    # A handful of blowout sims would otherwise stretch every panel to ~75 points and shrink
    # the region that actually holds the mass.
    axis_max = int(
        np.ceil(max(games["home_score"].quantile(0.999), games["away_score"].quantile(0.999)))
    )
    rows = int(np.ceil(len(teams) / columns))

    fig, axes = plt.subplots(rows, columns, figsize=(2.35 * columns, 2.5 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, team in zip(axes, teams):
        subset = games[games["home_team"] == team]
        rgb, all_scores = scorigami_rgb(
            subset["home_score"].round().astype(int).clip(0, axis_max),
            subset["away_score"].round().astype(int).clip(0, axis_max),
            axis_max,
            vert_exag=4.0,
        )
        extent = [all_scores[0] - 0.5, all_scores[-1] + 0.5, all_scores[0] - 0.5, all_scores[-1] + 0.5]
        ax.imshow(rgb, extent=extent, origin="lower", interpolation="nearest", aspect="equal")
        ax.plot([extent[0], extent[1]], [extent[2], extent[3]], color="black", linewidth=0.7, alpha=0.8)
        ax.set_title(team, fontsize=9)
        ax.set_xticks(np.arange(0, axis_max + 1, 10))
        ax.set_yticks(np.arange(0, axis_max + 1, 10))
        ax.tick_params(labelsize=6)

    for ax in axes[len(teams) :]:
        ax.axis("off")

    fig.supxlabel("Team points", fontsize=11)
    fig.supylabel("Opponent points", fontsize=11)
    fig.suptitle("Simulated Score Distributions vs. the League", fontsize=13)
    fig.tight_layout(rect=(0.01, 0.01, 1.0, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# -------------------------------------------------------------- html report


def embed(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def logo_data_uri(styles: pd.DataFrame, team: str) -> Optional[str]:
    url = style_for(styles, team).get("team_logo_espn")
    if logo_image(url, team) is None:
        return None
    cached = LOGO_CACHE_DIR / f"{team}.png"
    return f"data:image/png;base64,{embed(cached)}"


# Sim columns: key, label, format, the value that shades as neutral (None for the league average), distance to full shade.
SIM_COLUMNS = [
    ("mean_spread", "Spread", ui.signed, 0.0, ui.SPREAD_SCALE),
    ("win_prob", "Win %", ui.percent, 0.5, ui.WIN_SCALE),
    ("points_for", "Points for", ui.number, None, ui.POINTS_SCALE),
    ("points_against", "Points against", ui.number, None, -ui.POINTS_SCALE),
]
TABLE_STYLE = f"""
{ui.FONT_IMPORT}
body {{ font-family: {ui.FONT}; color: {ui.INK}; background: {ui.CANVAS}; }}
img.chart {{ max-width: 100%; border: 1px solid {ui.BORDER}; border-radius: 8px; }}
{ui.TABLE_CSS}
"""


def ratings_note(notes: Optional[dict]) -> str:
    text = ("Spread is average margin against the league, so positive is good. Blue is better than average and orange worse, so few points "
            "against is blue. ")
    if not notes:
        return text + "No outside ratings were available."
    parts = [f'<a href="{item["page"]}" target="_blank">{item["label"]}</a>' + (f' ({item["as_of"]})' if item["as_of"] else "") for item in notes["sources"]]
    text += ("Our gap is our spread minus the median outside rating: blue means we rate the team higher than outside sources, orange lower. "
             f'Outside ratings were pulled {notes["pulled"]}, in points against an average team on a neutral field and centered on each '
             f'source\'s league average: {", ".join(parts)}.')
    if notes["failed"]:
        text += f' Unavailable: {", ".join(item["label"] for item in notes["failed"])}.'
    return text


def summary_table_html(summary: pd.DataFrame, styles: pd.DataFrame, ratings: Optional[pd.DataFrame] = None, notes: Optional[dict] = None) -> str:
    ratings = ratings if ratings is not None else pd.DataFrame(index=summary.team.map(our_team))
    sources = list(ratings.columns)
    medians = ratings.median(axis=1) if sources else None
    neutral = {key: float(summary[key].mean()) if center is None else center for key, _, _, center, _ in SIM_COLUMNS}
    groups = f'<th class="group" colspan="2"></th><th class="group start" colspan="{len(SIM_COLUMNS)}">Our simulations</th>'
    labels = '<th>#</th><th class="text">Team</th>' + "".join(
        f'<th{" class=start" if position == 0 else ""}>{label}</th>' for position, (_, label, _, _, _) in enumerate(SIM_COLUMNS))
    if sources:
        groups += f'<th class="group start" colspan="2">Against outside</th><th class="group start" colspan="{len(sources)}">Outside ratings</th>'
        labels += '<th class="start">Median</th><th>Our gap</th>' + "".join(
            f'<th{" class=start" if position == 0 else ""}>{source}</th>' for position, source in enumerate(sources))

    rows = []
    for rank, row in enumerate(summary.itertuples(index=False), start=1):
        team = row.team
        logo = logo_data_uri(styles, team)
        badge = f'<img class="logo" src="{logo}" alt="" />' if logo else ""
        cells = "".join(
            ui.table_cell(text_format(float(getattr(row, key))), (float(getattr(row, key)) - neutral[key]) / scale, "lead start" if position == 0 else "")
            for position, (key, _, text_format, _, scale) in enumerate(SIM_COLUMNS)
        )
        if sources:
            rated = our_team(team) in ratings.index
            median = float(medians[our_team(team)]) if rated else float("nan")
            gap = float(row.mean_spread) - median
            cells += ui.table_cell(ui.signed(median), median / ui.SPREAD_SCALE, "start") + ui.table_cell(ui.signed(gap), gap / ui.GAP_SCALE, "shaded")
            for position, source in enumerate(sources):
                value = float(ratings.at[our_team(team), source]) if rated else float("nan")
                cells += ui.table_cell(ui.signed(value), value / ui.SPREAD_SCALE, "start" if position == 0 else "")
        rows.append(f'<tr><td class="rank">{rank}</td><td class="text team">{badge}{team}</td>{cells}</tr>')

    return (f'<table class="nfl-table"><thead><tr>{groups}</tr><tr>{labels}</tr></thead><tbody>' + "".join(rows)
            + f'</tbody></table><div class="nfl-note">{ratings_note(notes)}</div>')


def write_html(
    paths: Dict[str, Path], summary: pd.DataFrame, styles: pd.DataFrame, out_path: Path,
    site_label: str = "home", ratings: Optional[pd.DataFrame] = None, notes: Optional[dict] = None,
) -> Path:
    sections = "\n".join(
        f'<h2>{title}</h2><img class="chart" src="data:image/png;base64,{embed(path)}" />'
        for title, path in paths.items()
    )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>League Ratings Report</title>
<style>
{TABLE_STYLE}
body {{ margin: 28px; }}
</style></head>
<body>
<h1>League Ratings Report ({site_label})</h1>
<p>Team labels identify the ranked team; scores are not adjusted for home-field advantage.</p>
{sections}
<h2>Team Summary</h2>
{summary_table_html(summary, styles, ratings, notes)}
</body></html>"""
    out_path.write_text(html, encoding="utf-8")
    return out_path


def write_table_html(
    summary: pd.DataFrame,
    styles: pd.DataFrame,
    out_path: Path,
    title: str = "League Ratings",
    ratings: Optional[pd.DataFrame] = None,
    notes: Optional[dict] = None,
) -> Path:
    """Standalone, self-contained table page (logos inlined) for embedding elsewhere."""
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
{TABLE_STYLE}
body {{ margin: 0; padding: 20px; }}
h2.table-title {{ font-size: 18px; margin: 0 0 4px; }}
p.table-sub {{ font-size: 12px; color: {ui.MUTED}; margin: 0 0 12px; }}
</style></head>
<body>
<h2 class="table-title">{title}</h2>
<p class="table-sub">Each team simulated against every team in the league, itself included. Ranked by simulated spread.</p>
{summary_table_html(summary, styles, ratings, notes)}
</body></html>"""
    out_path.write_text(html, encoding="utf-8")
    return out_path


def export_table_png(html_path: Path, out_path: Path, scale: int = 2) -> Optional[Path]:
    """Screenshot the standalone table with Playwright; returns None when it is unavailable."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "Playwright not installed; skipping table PNG "
            "(pip install playwright && playwright install chromium)."
        )
        return None

    try:
        with sync_playwright() as play:
            browser = play.chromium.launch()
            page = browser.new_page(
                viewport={"width": 1400, "height": 1000}, device_scale_factor=scale
            )
            page.goto(html_path.resolve().as_uri())
            page.locator("body").screenshot(path=str(out_path))
            browser.close()
    except Exception as exc:  # missing browser binaries, render failure, etc.
        print(f"Table PNG export failed: {exc}")
        return None
    return out_path


def team_summary(games: pd.DataFrame) -> pd.DataFrame:
    frame = games.assign(_win=(games["home_score"] > games["away_score"]).astype(float))
    grouped = frame.groupby("home_team")
    summary = pd.DataFrame(
        {
            "team": grouped.size().index,
            "sims": grouped.size().to_numpy(),
            "points_for": grouped["home_score"].mean().to_numpy(),
            "points_against": grouped["away_score"].mean().to_numpy(),
            "median_spread": grouped["spread"].median().to_numpy(),
            "mean_spread": grouped["spread"].mean().to_numpy(),
            "spread_stdev": grouped["spread"].std().to_numpy(),
            "median_total": grouped["total"].median().to_numpy(),
            "win_prob": grouped["_win"].mean().to_numpy(),
        }
    )
    return summary.sort_values("mean_spread", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------- driver


CHART_TITLES = {
    "Point Spread Distributions": "synthetic_spread_violins.png",
    "Points For vs. Points Against": "synthetic_points_grid.png",
    "Score Heatmaps vs. the League": "synthetic_scorigami_grid.png",
}


def _write_tables(games: pd.DataFrame, styles: pd.DataFrame, target: Path, table_png: bool, ratings: pd.DataFrame, notes: dict) -> None:
    """Everything in the report that holds the team table; charts must already be in the folder."""
    neutral = "neutral_site" in games and games.neutral_site.fillna(False).astype(bool).all()
    summary = team_summary(games)
    summary.assign(rating_team=summary.team.map(our_team)).merge(ratings, left_on="rating_team", right_index=True, how="left").drop(columns="rating_team").to_csv(target / "synthetic_report_summary.csv", index=False)
    paths = {title: target / name for title, name in CHART_TITLES.items()}
    report = write_html(paths, summary, styles, target / "synthetic_report.html", "neutral site" if neutral else "home", ratings, notes)
    table_html = write_table_html(summary, styles, target / "synthetic_summary_table.html", ratings=ratings, notes=notes)
    print(f"Wrote {report}")
    print(f"Wrote {table_html}")
    if table_png:
        png = export_table_png(table_html, target / "synthetic_summary_table.png")
        if png:
            print(f"Wrote {png}")


def refresh_ratings(folder: Path, out_dir: Optional[Path] = None) -> dict:
    """Pull the outside ratings again and rewrite the team table, leaving the charts alone."""
    target = out_dir or folder
    ratings, notes = save_ratings(target)
    _write_tables(team_games(load_games(folder)), team_style(), target, False, ratings, notes)
    return notes


def rebuild_tables(folder: Path, out_dir: Optional[Path] = None) -> None:
    """Rewrite the team table from the saved ratings, for reports built before a table change."""
    target = out_dir or folder
    ratings, notes = load_ratings(target)
    _write_tables(team_games(load_games(folder)), team_style(), target, False, ratings, notes)


def build_report(
    folder: Path, out_dir: Optional[Path] = None, table_png: bool = True
) -> Path:
    games = team_games(load_games(folder))
    styles = team_style()
    target = out_dir or folder
    target.mkdir(parents=True, exist_ok=True)
    render_spread_violins(games, styles, target / CHART_TITLES["Point Spread Distributions"])
    render_points_grid(games, styles, target / CHART_TITLES["Points For vs. Points Against"])
    render_scorigami_grid(games, target / CHART_TITLES["Score Heatmaps vs. the League"])
    ratings, notes = save_ratings(target)
    _write_tables(games, styles, target, table_png, ratings, notes)
    return target / "synthetic_report.html"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a report from league rating simulations.")
    parser.add_argument("folder", type=Path, help="Run folder holding games.csv or synthetic_games.csv.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Where to write the report (defaults to the folder).")
    parser.add_argument("--no-table-png", action="store_true", help="Skip the Playwright screenshot of the summary table.")
    args = parser.parse_args()
    build_report(args.folder.expanduser(), args.out_dir.expanduser() if args.out_dir else None, table_png=not args.no_table_png)


if __name__ == "__main__":
    main()
