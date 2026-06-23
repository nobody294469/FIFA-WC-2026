import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from core.config import RESULTS_CSV, TRAINING_START, TEAM_NAME_ALIASES
from data.ingestion import load_raw_results, build_match_records

def main():
    print("=== TRAINING ROW CONSTRUCTION AUDIT ===\n")
    
    # 1. Raw Dataset
    raw_df = pd.read_csv(RESULTS_CSV, parse_dates=["date"])
    total_raw = len(raw_df)
    print(f"Total raw matches in results.csv: {total_raw}")
    
    # 2. Incomplete Matches Filter
    clean_df = raw_df.dropna(subset=["home_score", "away_score"]).copy()
    completed_matches = len(clean_df)
    discarded_incomplete = total_raw - completed_matches
    print(f"-> Discarded {discarded_incomplete} matches due to missing scores (future/forfeits) in `load_raw_results()`.")
    print(f"-> Remaining completed matches: {completed_matches}")
    
    # 3. Era Filter
    cutoff = pd.Timestamp(TRAINING_START)
    clean_df['date'] = pd.to_datetime(clean_df['date'])
    modern_df = clean_df[clean_df['date'] >= cutoff]
    era_discarded = completed_matches - len(modern_df)
    print(f"-> Discarded {era_discarded} matches occurring before {TRAINING_START} in `_process_chronological()`.")
    print(f"-> Remaining modern era matches: {len(modern_df)}")
    
    # 4. Team Filters
    wc_teams = set(TEAM_NAME_ALIASES.values())
    
    # Current behavior: BOTH teams must be WC2026
    both_wc_mask = modern_df['home_team'].isin(wc_teams) & modern_df['away_team'].isin(wc_teams)
    both_wc = modern_df[both_wc_mask]
    current_training_rows = len(both_wc)
    
    print(f"-> Discarded {len(modern_df) - current_training_rows} modern matches where at least one team is NOT in the 48 WC2026 teams.")
    print(f"-> Current Training Rows: {current_training_rows}\n")
    
    print("=== CORPUS EXPANSION ESTIMATES ===\n")
    
    # Alternative 1: AT LEAST ONE team is WC2026
    one_wc_mask = modern_df['home_team'].isin(wc_teams) | modern_df['away_team'].isin(wc_teams)
    one_wc = modern_df[one_wc_mask]
    one_wc_rows = len(one_wc)
    print(f"Scenario 1: Keep matches where AT LEAST ONE team is WC2026.")
    print(f"   -> Rows: {one_wc_rows} (+{one_wc_rows - current_training_rows} additional rows)")
    
    # Alternative 2: ALL modern matches
    print(f"Scenario 2: Keep ALL modern international matches.")
    print(f"   -> Rows: {len(modern_df)} (+{len(modern_df) - current_training_rows} additional rows)")
    
    # Determine leakage risks
    print("\n=== LEAKAGE ANALYSIS ===")
    print("Adding matches where non-WC teams play WC teams (or each other) does NOT introduce leakage, provided:")
    print("1. Form and Elo are computed chronologically (which they already are for all 49k matches).")
    print("2. The matches occurred before the test period.")
    print("Currently, Elo and Form state are already updated for all matches. The restriction is only on whether we *capture* the match as an (X, y) row for XGBoost. Capturing more rows simply exposes XGBoost to more examples of Elo/Form differentials predicting match outcomes.")

if __name__ == "__main__":
    main()
