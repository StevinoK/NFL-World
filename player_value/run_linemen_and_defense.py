"""Runner script for remaining defensive and offensive line positions."""

import sys
import time
from pathlib import Path
import pandas as pd

import player_value.plus_minus as plus_minus
POSITIONS = ['OL', 'DL', 'LB', 'DB']
YEARS = list(range(2016, 2026))

def main():
    total_start = time.perf_counter()
    summary = []
    for pos in POSITIONS:
        print(f"\n{'='*70}\nSTARTING POSITION: {pos}\n{'='*70}", flush=True)
        t0 = time.perf_counter()
        model = plus_minus.PlayerPlusMinus(pos, YEARS, method='box')
        model.run_full_pipeline(incremental=True)
        elapsed = time.perf_counter() - t0
        df = model.splits_df
        n_rows = len(df)
        n_weeks = len(df[['season', 'week']].drop_duplicates()) if not df.empty else 0
        summary.append({
            'position': pos,
            'elapsed_min': round(elapsed / 60, 2),
            'rows': n_rows,
            'weeks': n_weeks,
        })
        print(f"\nCOMPLETED {pos} in {elapsed/60:.1f}m ({n_rows:,} rows across {n_weeks} weeks)", flush=True)
        top = model.get_results(10)
        if not top.empty:
            print(f"\nTop 10 {pos} (Latest Season/Week):", flush=True)
            cols = [c for c in ['player_name', 'team', 'season', 'week', 'box_plus_minus', 'n_on'] if c in top.columns]
            print(top[cols].to_string(index=False), flush=True)

    total_elapsed = time.perf_counter() - total_start
    print(f"\n{'='*70}\nALL REMAINING POSITIONS COMPLETE in {total_elapsed/60:.1f}m\n{'='*70}", flush=True)
    print(pd.DataFrame(summary).to_string(index=False), flush=True)

if __name__ == '__main__':
    main()
