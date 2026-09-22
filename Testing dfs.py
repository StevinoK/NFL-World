"""Command-line DK Showdown DFS optimizer (see `dfs_dashboard.py` for the interactive
version with file pickers, kicker-ambiguity prompts, and public-projection blending)."""

from __future__ import annotations

import argparse

import pandas as pd

from dfs_optimizer import (
    align_simulation_columns,
    apply_kicker_selection,
    build_field_mean,
    build_field_simulations,
    build_player_map,
    compute_lineup_score_distributions,
    draft_our_entries,
    find_ambiguous_kickers,
    generate_valid_lineups,
    load_entries,
    load_prices,
    load_simulations,
    merge_prices_and_sims,
    run_field_simulation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and export DK Showdown GPP lineups.")
    parser.add_argument("--simulations", required=True, help="Path to the player simulations CSV.")
    parser.add_argument("--prices", required=True, help="Path to the DraftKings salaries CSV.")
    parser.add_argument("--entries", required=True, help="Path to the DraftKings entries CSV.")
    parser.add_argument("--output", required=True, help="Path to write the filled entries CSV.")
    parser.add_argument("--salary-min", type=int, default=48000)
    parser.add_argument("--salary-max", type=int, default=50000)
    parser.add_argument("--price-floor", type=int, default=600)
    parser.add_argument("--num-agents", type=int, default=500)
    parser.add_argument("--team-size", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--subsample", type=float, default=0.5)
    return parser.parse_args()


def resolve_kickers_interactively(prices: pd.DataFrame) -> pd.DataFrame:
    ambiguous = find_ambiguous_kickers(prices)
    if not ambiguous:
        return prices

    selections = {}
    for team, group in ambiguous.items():
        options = group.drop_duplicates("Name").sort_values("Salary", ascending=False).reset_index(drop=True)
        print(f"Multiple kickers priced for {team}:")
        for i, row in options.iterrows():
            print(f"  [{i}] {row['Name']} (${row['Salary']})")
        choice = input(f"Pick starting kicker index for {team} [0]: ").strip()
        index = int(choice) if choice else 0
        selections[team] = options.loc[index, "Name"]
    return apply_kicker_selection(prices, selections)


def main() -> None:
    args = parse_args()

    prices = load_prices(args.prices)
    prices = resolve_kickers_interactively(prices)

    simulations_raw = load_simulations(args.simulations)
    simulations, means, covs = align_simulation_columns(simulations_raw)
    merged_prices = merge_prices_and_sims(prices, means, price_floor=args.price_floor)

    lineups_df = generate_valid_lineups(merged_prices, args.salary_min, args.salary_max)
    print(f"Found {len(lineups_df)} valid lineups")

    player_map = build_player_map(merged_prices)
    lineup_sims_df, lu_scores_array = compute_lineup_score_distributions(lineups_df, simulations, player_map)

    # No network calls from the CLI path -- field model reuses our own simulation means.
    field_mean = build_field_mean(pd.Series(dtype=float), means)
    field_simulations = build_field_simulations(covs, field_mean, len(simulations))
    _, market_scores_array = compute_lineup_score_distributions(lineups_df, field_simulations, player_map)

    def log_progress(done: int, total: int, drift: float) -> None:
        print(f"Field iteration {done}/{total}, ownership drift {drift:.4f}")

    _, _, field_indices = run_field_simulation(
        lu_scores_array,
        market_scores_array,
        lineup_sims_df,
        num_agents=args.num_agents,
        team_size=args.team_size,
        iterations=args.iterations,
        subsample=args.subsample,
        progress_callback=log_progress,
    )

    export_df = load_entries(args.entries)
    filled_export, drafted_lineups_df, _ = draft_our_entries(
        export_df, lu_scores_array, field_indices, lineup_sims_df, player_map
    )
    print(drafted_lineups_df)
    filled_export.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

