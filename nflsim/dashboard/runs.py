"""Find saved simulation runs for the results pages."""

from __future__ import annotations

from pathlib import Path

from nflsim.paths import ROOT

OUTPUTS = ROOT / "outputs"


def _newest_first(paths) -> list[Path]:
    return sorted(paths, key=lambda path: path.stat().st_mtime, reverse=True)


def _is_synthetic(folder: Path) -> bool:
    if "syn" in folder.name.lower().split("-") or folder.parent.name.startswith("synthetic"):
        return True
    with open(folder / "games.csv", encoding="utf-8") as games:
        games.readline()
        return "_SYN_" in games.readline()


def weekly_runs(outputs: Path = OUTPUTS) -> list[Path]:
    """Folders holding a real-matchup run, newest first."""
    folders = [*outputs.glob("*"), *outputs.glob("simulations/*")]
    finished = [folder for folder in folders if (folder / "games.csv").is_file() and (folder / "players.csv").is_file()]
    return _newest_first(folder for folder in finished if not _is_synthetic(folder))


def synthetic_runs(outputs: Path = OUTPUTS) -> list[Path]:
    """Folders holding a run against the synthetic opponent, newest first."""
    command_line = [folder for folder in outputs.glob("synthetic/*") if (folder / "synthetic_games.csv").is_file()]
    dashboard = [folder for folder in outputs.glob("simulations/synthetic_home_week_*/*") if (folder / "games.csv").is_file()]
    return _newest_first([*command_line, *dashboard])
