#!/usr/bin/env python
"""
Entry point for running 2026 forward-looking game simulations.
Usage:
  python run_2026_forward_simulator.py --simulations 200 --workers 4
  python run_2026_forward_simulator.py --week-start 1 --week-end 8
  python run_2026_forward_simulator.py --game-id 2026_01_BUF_NYJ
  python run_2026_forward_simulator.py --data-cutoff-date 2025-12-15  (for in-season updates)
"""

from torch_game_simulations_2026_forward import main

if __name__ == "__main__":
    main()
