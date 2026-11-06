"""Runner script to sequentially refit all positions under the updated box framework."""

import sys
import time
from pathlib import Path
import pandas as pd

import player_value.plus_minus as plus_minus
POSITIONS = ['RB', 'WR', 'TE', 'OL', 'DL', 'LB', 'DB', 'QB']
YEARS = list(range(2016, 2026))

def main():
    total_start = time.perf_counter()
    results_summary = []
    for pos in POSITIONS:
        print(f"\n{'='*70}\nSTARTING REFIT: {pos}\n{'='*70}", flush=True)
        t0 = time.perf_counter()
        model = plus_minus.PlayerPlusMinus(pos, YEARS, method='box')
        model.run_full_pipeline(incremental=True)
        elapsed = time.perf_counter() - t0
        df = model.splits_df
        n_rows = len(df)
        n_weeks = len(df[['season', 'week']].drop_duplicates()) if not df.empty else 0
        results_summary.append({
            'position': pos,
            'elapsed_seconds': round(elapsed, 1),
            'rows': n_rows,
            'weeks': n_weeks,
        })
        print(f"\nCOMPLETED {pos} in {elapsed:.1f}s ({elapsed/60:.1f}m)", flush=True)
        print(f"Total rows: {n_rows:,}, Unique Season/Weeks: {n_weeks}", flush=True)
        top = model.get_results(10)
        if not top.empty:
            print(f"\nTop 10 {pos} (Latest Season/Week):", flush=True)
            cols = [c for c in ['player_name', 'team', 'season', 'week', 'box_plus_minus', 'n_on'] if c in top.columns]
            print(top[cols].to_string(index=False), flush=True)

    total_elapsed = time.perf_counter() - total_start
    print(f"\n{'='*70}\nALL POSITIONS REFIT COMPLETE in {total_elapsed/60:.1f}m\n{'='*70}", flush=True)
    print(pd.DataFrame(results_summary).to_string(index=False), flush=True)

if __name__ == '__main__':
    main()
