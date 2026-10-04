"""Report builder for the synthetic-opponent home simulations.

Reads the combined `*_games.csv` written by `simulate_synthetic_home_games.py` and renders:
  1. Violin contours of point spread by team, team-colored, descending by median spread.
  2. Points-for vs points-against scatter with team logos as the markers.
  3. A grid of LightSource-shaded scorigami heatmaps, one per team vs. the synthetic opponent.
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

from project_paths import DEFAULT_SIMULATION_OUTPUT_DIR

WORKSPACE_DIR = DEFAULT_SIMULATION_OUTPUT_DIR
LOGO_CACHE_DIR = Path(__file__).resolve().parent / ".team_logo_cache"
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
    combined = sorted(folder.glob("synthetic_home_week_*_games.csv"))
    if combined:
        return pd.read_csv(combined[-1])
    per_game = [p for p in folder.glob("*_games.csv") if "_SYN_" in p.name]
    if not per_game:
        raise FileNotFoundError(f"No synthetic *_games.csv found in {folder}")
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
    ax.set_ylabel("Point spread vs. synthetic opponent (home margin)")
    ax.set_title(
        f"Simulated Point Spread vs. Replacement-Level Opponent (home) — middle {100 * (1 - 2 * trim):.0f}%"
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
    ax.set_xlabel("Average points scored vs. synthetic opponent")
    ax.set_ylabel("Average points allowed to synthetic opponent")
    ax.set_title("Offense vs. Defense against a Replacement-Level Opponent")
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
        ax.set_title(f"{team} vs SYN", fontsize=9)
        ax.set_xticks(np.arange(0, axis_max + 1, 10))
        ax.set_yticks(np.arange(0, axis_max + 1, 10))
        ax.tick_params(labelsize=6)

    for ax in axes[len(teams) :]:
        ax.axis("off")

    fig.supxlabel("Team points", fontsize=11)
    fig.supylabel("Synthetic opponent points", fontsize=11)
    fig.suptitle("Simulated Score Distributions vs. a Replacement-Level Opponent", fontsize=13)
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


TABLE_CSS = """
table.summary { border-collapse: separate; border-spacing: 0; font-size: 13px; width: 100%;
  max-width: 1200px; background: #fff; border: 1px solid #e4e4e4; border-radius: 8px; overflow: hidden; }
table.summary th { font-size: 11px; letter-spacing: .04em; text-transform: uppercase; color: #5a5a5a;
  background: #f5f5f4; padding: 9px 10px; text-align: right; border-bottom: 1px solid #e0e0e0; white-space: nowrap; }
table.summary td { padding: 7px 10px; border-bottom: 1px solid #f0f0f0; vertical-align: middle; }
table.summary tbody tr:hover { background: color-mix(in srgb, var(--team) 9%, #fff); }
table.summary tbody tr:last-child td { border-bottom: none; }

td.rank, th.rank { text-align: right; color: #9a9a9a; font-variant-numeric: tabular-nums; width: 34px; }
td.team, th.team { text-align: left; }
td.team { border-left: 4px solid var(--team); font-weight: 600; white-space: nowrap; }
td.team .logo { width: 20px; height: 20px; object-fit: contain; border: none; border-radius: 0;
  vertical-align: middle; margin-right: 7px; }
td.team .logo.placeholder { display: inline-block; background: var(--team2); border-radius: 50%; }
td.team .abbr { vertical-align: middle; }

td.num { text-align: right; font-variant-numeric: tabular-nums; min-width: 92px; }
td.num.plain { color: #666; }
td.num .val { display: block; font-size: 12px; line-height: 1.1; margin-bottom: 3px; }
td.num .track { position: relative; display: block; height: 6px; border-radius: 3px; background: #efefee; }
td.num .bar { position: absolute; top: 0; height: 100%; border-radius: 3px; opacity: .85; }
td.num .zero { position: absolute; left: 50%; top: -1px; width: 1px; height: calc(100% + 2px); background: #c4c4c4; }
"""

# Columns rendered with an in-cell bar: label, format, and bar mode.
SUMMARY_COLUMNS: List[Tuple[str, str, str]] = [
    ("sims", "Sims", "plain"),
    ("points_for", "Points For", "positive"),
    ("points_against", "Points Against", "positive"),
    ("median_spread", "Median Spread", "diverging"),
    ("mean_spread", "Mean Spread", "diverging"),
    ("spread_stdev", "Spread SD", "positive"),
    ("median_total", "Median Total", "positive"),
    ("win_prob", "Win Prob", "percent"),
]


def _bar_cell(value: float, mode: str, lo: float, hi: float, color: str) -> str:
    """Datawrapper-style cell: numeric label above a proportional bar."""
    if mode == "percent":
        text = f"{value * 100:.1f}%"
    elif mode == "plain":
        return f'<td class="num plain">{value:,.0f}</td>'
    else:
        text = f"{value:.2f}"

    span = max(hi - lo, 1e-9)
    if mode == "diverging":
        extent = max(abs(lo), abs(hi), 1e-9)
        frac = min(abs(value) / extent, 1.0) * 50.0
        left, width = (50.0 - frac, frac) if value < 0 else (50.0, frac)
        bar = (
            f'<span class="bar" style="left:{left:.2f}%;width:{width:.2f}%;background:{color};"></span>'
            '<span class="zero"></span>'
        )
    else:
        frac = (value - lo) / span if mode != "percent" else value
        width = min(max(frac, 0.0), 1.0) * 100.0
        bar = f'<span class="bar" style="left:0;width:{width:.2f}%;background:{color};"></span>'
    return f'<td class="num"><span class="val">{text}</span><span class="track">{bar}</span></td>'


def summary_table_html(summary: pd.DataFrame, styles: pd.DataFrame) -> str:
    ranges = {
        key: (float(summary[key].min()), float(summary[key].max()))
        for key, _, _ in SUMMARY_COLUMNS
        if key in summary.columns
    }
    header = "".join(
        f"<th>{label}</th>" for key, label, _ in SUMMARY_COLUMNS if key in ranges
    )

    rows = []
    for rank, row in enumerate(summary.itertuples(index=False), start=1):
        team = getattr(row, "team")
        style = style_for(styles, team)
        primary = style.get("team_color") or "#666666"
        secondary = style.get("team_color2") or "#222222"
        logo = logo_data_uri(styles, team)
        badge = (
            f'<img class="logo" src="{logo}" alt="{team}" />'
            if logo
            else '<span class="logo placeholder"></span>'
        )
        cells = "".join(
            _bar_cell(float(getattr(row, key)), mode, *ranges[key], primary)
            for key, _, mode in SUMMARY_COLUMNS
            if key in ranges
        )
        rows.append(
            f'<tr style="--team:{primary};--team2:{secondary};">'
            f'<td class="rank">{rank}</td>'
            f'<td class="team">{badge}<span class="abbr">{team}</span></td>'
            f"{cells}</tr>"
        )

    return (
        '<table class="summary"><thead><tr><th class="rank">#</th><th class="team">Team</th>'
        f"{header}</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def write_html(
    paths: Dict[str, Path], summary: pd.DataFrame, styles: pd.DataFrame, out_path: Path
) -> Path:
    sections = "\n".join(
        f'<h2>{title}</h2><img src="data:image/png;base64,{embed(path)}" />'
        for title, path in paths.items()
    )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Synthetic Opponent Simulation Report</title>
<style>
body {{ font-family: Segoe UI, Arial, sans-serif; margin: 28px; color: #1c1c1c; background: #fbfbfa; }}
img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 6px; }}
{TABLE_CSS}
</style></head>
<body>
<h1>Synthetic Opponent Simulation Report</h1>
{sections}
<h2>Team Summary</h2>
{summary_table_html(summary, styles)}
</body></html>"""
    out_path.write_text(html, encoding="utf-8")
    return out_path


def write_table_html(
    summary: pd.DataFrame,
    styles: pd.DataFrame,
    out_path: Path,
    title: str = "Simulated Team Strength vs. a Replacement-Level Opponent",
) -> Path:
    """Standalone, self-contained table page (logos inlined) for embedding elsewhere."""
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
body {{ font-family: Segoe UI, Arial, sans-serif; margin: 0; padding: 20px; color: #1c1c1c;
  background: #fff; display: inline-block; }}
h2.table-title {{ font-size: 18px; margin: 0 0 4px; }}
p.table-sub {{ font-size: 12px; color: #777; margin: 0 0 12px; }}
{TABLE_CSS}
</style></head>
<body>
<h2 class="table-title">{title}</h2>
<p class="table-sub">Each team simulated at home against an identical synthetic opponent.</p>
{summary_table_html(summary, styles)}
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
    return summary.sort_values("median_spread", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------- driver


def build_report(
    folder: Path, out_dir: Optional[Path] = None, table_png: bool = True
) -> Path:
    games = load_games(folder)
    styles = team_style()
    target = out_dir or folder
    target.mkdir(parents=True, exist_ok=True)

    paths = {
        "Point Spread Distributions": render_spread_violins(
            games, styles, target / "synthetic_spread_violins.png"
        ),
        "Points For vs. Points Against": render_points_grid(
            games, styles, target / "synthetic_points_grid.png"
        ),
        "Score Heatmaps vs. Synthetic Opponent": render_scorigami_grid(
            games, target / "synthetic_scorigami_grid.png"
        ),
    }
    summary = team_summary(games)
    summary.to_csv(target / "synthetic_report_summary.csv", index=False)
    report = write_html(paths, summary, styles, target / "synthetic_report.html")
    table_html = write_table_html(summary, styles, target / "synthetic_summary_table.html")
    print(f"Wrote {report}")
    print(f"Wrote {table_html}")
    if table_png:
        png = export_table_png(table_html, target / "synthetic_summary_table.png")
        if png:
            print(f"Wrote {png}")
    for path in paths.values():
        print(f"Wrote {path}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a report from synthetic-opponent simulations.")
    parser.add_argument(
        "--folder",
        type=Path,
        default=WORKSPACE_DIR / "synthetic_home_week_04",
        help="Folder containing the synthetic simulation CSVs.",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="Where to write the report (defaults to --folder).")
    parser.add_argument(
        "--no-table-png",
        action="store_true",
        help="Skip the Playwright screenshot of the standalone summary table.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_report(
        args.folder.expanduser(),
        args.out_dir.expanduser() if args.out_dir else None,
        table_png=not args.no_table_png,
    )
