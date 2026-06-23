"""
run_backtest.py — True out-of-sample World Cup backtest runner.

Execution order
───────────────
1. Run the true OOS backtest (trains one model per tournament on pre-cutoff data).
2. (Optional) Print a side-by-side comparison table to stdout and save to
   reports/backtest_comparison.txt.

Usage
─────
    python scripts/run_backtest.py [--sims N] [--debug-compare]

Options
    --sims N              Number of Monte Carlo simulations per tournament (default 10000)
    --debug-compare       Run the contaminated (production) backtest and compare against OOS
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import N_SIMULATIONS, REPORT_DIR
from core.logger import get_logger
from backtesting.world_cup_backtest import (
    run_world_cup_backtest,
    run_oos_world_cup_backtest,
)

log = get_logger("run_backtest")
_CONTAMINATED_JSON = REPORT_DIR / "historical_backtest.json"
_DIV = "=" * 82


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the true out-of-sample World Cup backtest."
    )
    parser.add_argument(
        "--sims",
        type=int,
        default=N_SIMULATIONS,
        help=f"Monte Carlo simulations per tournament (default: {N_SIMULATIONS})",
    )
    parser.add_argument(
        "--debug-compare",
        action="store_true",
        help="Run the contaminated backtest and generate a comparison report",
    )
    args = parser.parse_args()

    if args.debug_compare:
        print()
        print(_DIV)
        print("  STEP 1 — Running contaminated (production-model) backtest")
        print("  This is required so the comparison table has a baseline.")
        print(_DIV)
        run_world_cup_backtest(n_sims=args.sims)

    print()
    print(_DIV)
    print("  TRUE OUT-OF-SAMPLE BACKTEST")
    print("  Training one model per tournament on pre-cutoff data only.")
    print(_DIV)
    oos_summary = run_oos_world_cup_backtest(n_sims=args.sims, compare=args.debug_compare)

    print()
    print(_DIV)
    print("  DONE — Files written to reports/")
    print(f"    OOS results   : {REPORT_DIR / 'oos_backtest.json'}")
    print(f"    OOS report    : {REPORT_DIR / 'oos_backtest.txt'}")
    if args.debug_compare:
        print(f"    Comparison    : {REPORT_DIR / 'backtest_comparison.txt'}")
    print(_DIV)


if __name__ == "__main__":
    main()
