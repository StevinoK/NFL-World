"""Posterior SDs of the published ridge ratings, for drawing team and player strengths once per simulated game.

Refits the production ridge targets for the requested weeks exactly as the feature build does (same parameters,
over-expectation targets, player value and semantics), with each fit's Gaussian posterior
(``nflsim.ridge.metrics.ridge_posterior``). Writes one row per published rating column: game_id, kind (team or
player), entity, column, the coefficient it reads (fit, block, label) and its SD. Opponent copies
(``opp_def_pass_epa_rate`` ...) get rows too. The simulator reads the table with NFLSIM_STRENGTH_DRAWS
(``nflsim.sim.strength_draws``); bundles are not changed.

    python -m nflsim.ridge.uncertainty --pbp <pbp.parquet> --schedule <schedule.parquet> --roster <roster.parquet> \\
        --player-value <bundle>/player_value.csv --feature-semantics <bundle semantics> --season 2026 --weeks 1-7 \\
        --output nflsim/outputs/strength_sd/2026_w1-7.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from nflsim.features.builder import OPPONENT_ALIASES, build_estimates, canonicalize_teams, prepare_pbp
from nflsim.paths import ROOT, output_dir
from nflsim.ridge.expectations import production_targets, with_production_expectations
from nflsim.ridge.metrics import TARGETS
from nflsim.semantics import ISOLATED_MU_V1, ISOLATED_SEMANTICS


def with_opponent_copies(sds: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Rows for the team-table columns that copy the opponent's rating, reading the opponent's coefficient."""
    games = canonicalize_teams(schedule)[["game_id", "home_team", "away_team"]].astype(str)
    opponent = {**{(row.game_id, row.home_team): row.away_team for row in games.itertuples()},
                **{(row.game_id, row.away_team): row.home_team for row in games.itertuples()}}
    copies = []
    for alias, source in OPPONENT_ALIASES.items():
        own = sds[sds.kind.eq("team") & sds.column.eq(source)]
        for row in own.itertuples(index=False):
            other = opponent.get((row.game_id, row.entity))
            if other is not None:
                copies.append({**row._asdict(), "entity": other, "column": alias})
    return pd.concat([sds, pd.DataFrame(copies, columns=sds.columns)], ignore_index=True)


def coefficient_sd_table(pbp, schedule, roster, parameters, season, weeks, player_value=None,
                         feature_semantics=ISOLATED_MU_V1) -> pd.DataFrame:
    sds: list[dict] = []
    build_estimates(with_production_expectations(prepare_pbp(pbp)), schedule, roster, parameters,
                    targets=production_targets(TARGETS), predict_seasons=[season],
                    predict_weeks={(int(season), int(week)) for week in weeks}, player_value=player_value,
                    feature_semantics=feature_semantics, coefficient_sds=sds)
    columns = ["game_id", "kind", "entity", "column", "fit", "block", "label", "sd"]
    return with_opponent_copies(pd.DataFrame(sds, columns=columns), schedule)


def parse_weeks(text: str) -> list[int]:
    weeks = []
    for part in text.split(","):
        first, _, last = part.partition("-")
        weeks.extend(range(int(first), int(last or first) + 1))
    return weeks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pbp", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--player-value", type=Path, help="The bundle's player_value.csv (omit for fits without PV)")
    parser.add_argument("--parameters", type=Path, default=ROOT / "config/ridge_parameters.json")
    parser.add_argument("--feature-semantics", choices=ISOLATED_SEMANTICS, default=ISOLATED_MU_V1)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--weeks", required=True, help="e.g. 1-7 or 3,5,8-10")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    read = lambda path: pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, low_memory=False)
    player_value = pd.read_csv(args.player_value, dtype={"game_id": str}) if args.player_value else None
    parameters = json.loads(args.parameters.read_text())["parameters"]
    table = coefficient_sd_table(read(args.pbp), read(args.schedule), read(args.roster), parameters, args.season,
                                 parse_weeks(args.weeks), player_value, args.feature_semantics)
    output_dir(args.output.parent)
    table.to_csv(args.output, index=False)
    summary = table.groupby(["kind", "column"]).sd.mean().sort_values(ascending=False)
    print(f"{len(table)} rows for {table.game_id.nunique()} games; mean SD by column:")
    print(summary.head(30).to_string())


if __name__ == "__main__":
    main()
