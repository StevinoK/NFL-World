"""Readable names for model bundles, so nobody has to remember what a hash or folder name means.

Names live in config/bundle_notes.json, keyed by bundle folder. A weekly refresh inherits the name of the bundle it came from.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from nflsim.paths import ROOT

NOTES = ROOT / "config/bundle_notes.json"
BUNDLES = ROOT / "artifacts/bundles"
BASE = ROOT / "artifacts/runtime"


@dataclass(frozen=True)
class BundleInfo:
    path: Path
    name: str
    detail: str
    built: datetime
    through: str | None  # "2026 week 8" for a refreshed bundle; None when it cannot simulate upcoming games.

    @property
    def can_simulate_upcoming(self) -> bool:
        return self.through is not None


def read_notes(path: Path = NOTES) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def note_bundle(bundle: Path, name: str, detail: str = "", path: Path = NOTES) -> None:
    notes = read_notes(path)
    notes[Path(bundle).name] = {"name": name, "detail": detail}
    path.write_text(json.dumps(notes, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def data_through(bundle: Path) -> str | None:
    """Last completed week in a refreshed bundle's schedule."""
    schedule_path = Path(bundle) / "weekly_inputs/schedule.parquet"
    if not schedule_path.is_file():
        return None
    schedule = pd.read_parquet(schedule_path, columns=["season", "week", "home_score"])
    played = schedule[pd.to_numeric(schedule.home_score, errors="coerce").notna()]
    if played.empty:
        return "no games yet"
    season = int(played.season.max())
    return f"{season} week {int(played[played.season.eq(season)].week.max())}"


def bundle_info(bundle: Path, notes: dict | None = None) -> BundleInfo:
    bundle = Path(bundle)
    note = (read_notes() if notes is None else notes).get(bundle.name, {})
    fallback = "Original training bundle" if bundle.resolve() == BASE.resolve() else f"Unnamed bundle {bundle.name[:12]}"
    return BundleInfo(bundle, note.get("name", fallback), note.get("detail", ""), datetime.fromtimestamp(bundle.stat().st_mtime), data_through(bundle))


def bundle_label(info: BundleInfo, current: Path | None = None) -> str:
    """"Seed-averaged models · built Nov 5 · through 2026 week 8 · current"."""
    parts = [info.name, f"built {info.built:%b} {info.built.day}"]
    parts.append(f"through {info.through}" if info.through else "training data only")
    if current is not None and info.path.resolve() == Path(current).resolve():
        parts.append("current")
    return " · ".join(parts)


def all_bundles() -> list[BundleInfo]:
    """Sealed bundles, newest first: those able to simulate upcoming games, then the rest."""
    notes = read_notes()
    folders = [path for path in BUNDLES.glob("*") if (path / "bundle_manifest.json").is_file()]
    infos = [bundle_info(path, notes) for path in folders]
    if (BASE / "bundle_manifest.json").is_file():
        infos.append(bundle_info(BASE, notes))
    return sorted(infos, key=lambda info: (info.can_simulate_upcoming, info.built), reverse=True)
