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

WORKSPACE_DIR = Path(r"C:\Users\steve\OneDrive\Desktop\Football\Modeling\Simulations")
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
        f"Simulated Point Spread vs. League-Average Opponent (home) — middle {100 * (1 - 2 * trim):.0f}%"
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
    ax.set_title("Offense vs. Defense against a League-Average Opponent")
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
    fig.suptitle("Simulated Score Distributions vs. a League-Average Opponent", fontsize=13)
    fig.tight_layout(rect=(0.01, 0.01, 1.0, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# -------------------------------------------------------------- html report


def embed(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def write_html(paths: Dict[str, Path], summary: pd.DataFrame, out_path: Path) -> Path:
    sections = "\n".join(
        f'<h2>{title}</h2><img src="data:image/png;base64,{embed(path)}" />'
        for title, path in paths.items()
    )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Synthetic Opponent Simulation Report</title>
<style>
body {{ font-family: Segoe UI, Arial, sans-serif; margin: 28px; color: #1c1c1c; }}
img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 6px; }}
table {{ border-collapse: collapse; font-size: 13px; }}
th, td {{ border: 1px solid #ddd; padding: 4px 8px; text-align: right; }}
th {{ background: #f3f3f3; }}
td:first-child, th:first-child {{ text-align: left; }}
</style></head>
<body>
<h1>Synthetic Opponent Simulation Report</h1>
{sections}
<h2>Team Summary</h2>
{summary.to_html(index=False, float_format=lambda v: f"{v:.2f}")}
</body></html>"""
    out_path.write_text(html, encoding="utf-8")
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


def build_report(folder: Path, out_dir: Optional[Path] = None) -> Path:
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
    report = write_html(paths, summary, target / "synthetic_report.html")
    print(f"Wrote {report}")
    for path in paths.values():
        print(f"Wrote {path}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a report from synthetic-opponent simulations.")
    parser.add_argument(
        "--folder",
        type=Path,
        default=WORKSPACE_DIR / "synthetic_home_week_01",
        help="Folder containing the synthetic simulation CSVs.",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="Where to write the report (defaults to --folder).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_report(args.folder.expanduser(), args.out_dir.expanduser() if args.out_dir else None)
