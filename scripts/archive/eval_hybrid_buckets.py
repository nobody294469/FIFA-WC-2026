import sys
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core.config
from core.config import WC2026_TEAMS
from data.ingestion import MatchDataPipeline, load_raw_results, build_match_records
from data.elo_tracker import EloTracker
from data.feature_engineering import build_training_features, build_inference_features, FEATURE_COLS
from models.ml_engine import MatchOutcomeModel

def build_datasets():
    print("Loading data pipeline...")
    pipeline = MatchDataPipeline()
    raw = load_raw_results()
    all_matches = build_match_records(raw)
    
    elo = EloTracker()
    elo.batch_update(all_matches)
    
    df_train_raw = pipeline.training_df[pipeline.training_df["date"] < pd.Timestamp("2024-01-01")].copy()
    X_full, y_full = build_training_features(pipeline.training_df, elo)
    
    cutoff_idx = len(df_train_raw)
    X_tr = X_full.iloc[:cutoff_idx].copy()
    y_tr = y_full.iloc[:cutoff_idx].copy()
    
    pairs = list(combinations(WC2026_TEAMS, 2))
    form_scores = pipeline.current_form_scores()
    of_scores = pipeline.current_opp_adjusted_form_scores()
    form_gd = pipeline.current_form_gd()
    f_details = pipeline.current_form_details()
    
    X_inf = build_inference_features(
        pairs = pairs,
        elo_tracker = elo,
        form_scores = form_scores,
        opp_adj_form_scores = of_scores,
        form_gd = form_gd,
        form_details = f_details,
        h2h_fn = pipeline.current_h2h,
        neutral = True,
        tournament = "FIFA World Cup",
    )
    
    return X_tr, y_tr, X_inf, pairs

def main():
    X_tr, y_tr, X_inf, pairs = build_datasets()
    
    core.config.HYBRID_CALIBRATION_RANK_THRESHOLD = 20
    model = MatchOutcomeModel()
    model.fit(X_tr, y_tr)
    
    is_elite_inf = (X_inf["h_rank"] <= 20) & (X_inf["a_rank"] <= 20)
    elite_X = X_inf[is_elite_inf].copy()
    elite_pairs = [pairs[i] for i in elite_X.index]
    
    print(f"\nAnalyzing {len(elite_X)} elite matches out of {len(X_inf)} total WC combinations.")
    
    hybrid_elite = model.predict_proba(elite_X)
    
    # We round slightly to prevent floating point noise from artificially inflating unique counts.
    # 4 decimal places = 0.01% precision
    rounded_probs = np.round(hybrid_elite, 4)
    
    # Create string tuples for counting
    str_probs = [f"{p[0]:.4f}/{p[1]:.4f}/{p[2]:.4f}" for p in rounded_probs]
    counts = Counter(str_probs)
    
    print(f"\nNumber of unique probability buckets: {len(counts)} (out of {len(elite_X)} matches)")
    
    print("\nMost common probability buckets (Top 10):")
    for prob_str, cnt in counts.most_common(10):
        print(f"  {prob_str}: {cnt} matches")
        
    print("\nMatches mapped to the most common bucket:")
    most_common_str = counts.most_common(1)[0][0]
    for i, p_str in enumerate(str_probs):
        if p_str == most_common_str:
            print(f"  {elite_pairs[i][0]} vs {elite_pairs[i][1]}")

if __name__ == "__main__":
    main()
