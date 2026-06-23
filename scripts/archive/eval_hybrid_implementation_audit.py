import sys
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations

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
    X_te = X_full.iloc[cutoff_idx:].copy()
    y_te = y_full.iloc[cutoff_idx:].copy()
    
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
    
    return X_tr, y_tr, X_te, y_te, X_inf, pairs

def main():
    X_tr, y_tr, X_te, y_te, X_inf, pairs = build_datasets()
    
    thresholds = [18, 20, 22]
    
    print("\n--- 1. ROUTED MATCH COUNTS ---")
    
    for t in thresholds:
        is_elite_tr = (X_tr["h_rank"] <= t) & (X_tr["a_rank"] <= t)
        is_elite_te = (X_te["h_rank"] <= t) & (X_te["a_rank"] <= t)
        is_elite_inf = (X_inf["h_rank"] <= t) & (X_inf["a_rank"] <= t)
        
        print(f"Threshold = {t}:")
        print(f"  Training routed: {is_elite_tr.sum()} / {len(X_tr)} ({(is_elite_tr.sum()/len(X_tr))*100:.1f}%)")
        print(f"  Holdout routed:  {is_elite_te.sum()} / {len(X_te)} ({(is_elite_te.sum()/len(X_te))*100:.1f}%)")
        print(f"  WC inference routed: {is_elite_inf.sum()} / {len(X_inf)} ({(is_elite_inf.sum()/len(X_inf))*100:.1f}%)")
    
    print("\nTraining models to inspect probabilities at T=20...")
    core.config.HYBRID_CALIBRATION_RANK_THRESHOLD = 20
    model = MatchOutcomeModel()
    model.fit(X_tr, y_tr)
    
    # We want Raw XGB, Global Platt, and Hybrid
    # Raw XGB = model.model.predict_proba
    # Global Platt = model.calibrated_model.predict_proba
    # Hybrid = model.predict_proba
    
    print("\n--- 2. PROBABILITY COMPARISONS (T=20) ---")
    elite_samples = [
        ("Spain", "Argentina"),
        ("France", "England"),
        ("Brazil", "Portugal"),
        ("Germany", "Netherlands"),
        ("Italy", "Belgium"),
        ("Spain", "France"),
        ("Argentina", "Brazil"),
        ("England", "Germany"),
        ("Portugal", "Italy"),
        ("Netherlands", "Belgium")
    ]
    
    # Print sample side by side
    print(f"{'Matchup':<25} | {'Raw XGB (W/D/L)':<22} | {'Global Platt (W/D/L)':<22} | {'Hybrid (W/D/L)':<22}")
    print("-" * 100)
    
    for h, a in elite_samples:
        if h not in WC2026_TEAMS or a not in WC2026_TEAMS:
            continue
        try:
            idx = pairs.index((h, a))
        except ValueError:
            idx = pairs.index((a, h))
            h, a = a, h # flip to match pair order
            
        row = X_inf.iloc[[idx]]
        
        raw_p = model.model.predict_proba(row[FEATURE_COLS])[0]
        platt_p = model.calibrated_model.predict_proba(row[FEATURE_COLS])[0]
        hybrid_p = model.predict_proba(row)[0]
        
        raw_s = f"{raw_p[0]:.2f}/{raw_p[1]:.2f}/{raw_p[2]:.2f}"
        platt_s = f"{platt_p[0]:.2f}/{platt_p[1]:.2f}/{platt_p[2]:.2f}"
        hybrid_s = f"{hybrid_p[0]:.2f}/{hybrid_p[1]:.2f}/{hybrid_p[2]:.2f}"
        
        name = f"{h} vs {a}"
        print(f"{name:<25} | {raw_s:<22} | {platt_s:<22} | {hybrid_s:<22}")

    print("\n--- 3. LARGEST DRAW SHIFTS (Platt vs Hybrid) ---")
    # Evaluate all elite matchups in inference set
    is_elite_inf = (X_inf["h_rank"] <= 20) & (X_inf["a_rank"] <= 20)
    elite_X = X_inf[is_elite_inf]
    elite_pairs = [pairs[i] for i in elite_X.index]
    
    platt_elite = model.calibrated_model.predict_proba(elite_X[FEATURE_COLS])
    hybrid_elite = model.predict_proba(elite_X)
    
    draw_diffs = hybrid_elite[:, 1] - platt_elite[:, 1]
    
    # Sort by absolute difference
    sorted_idx = np.argsort(np.abs(draw_diffs))[::-1]
    
    count = 0
    for i in sorted_idx:
        diff = draw_diffs[i] * 100
        if abs(diff) > 2.0:
            h, a = elite_pairs[i]
            p_platt = platt_elite[i, 1]
            p_hybrid = hybrid_elite[i, 1]
            print(f"{h} vs {a}: {p_platt:.1%} -> {p_hybrid:.1%} (Shift: {diff:+.1f}%)")
            count += 1
            if count >= 5:
                break
                
    if count == 0:
        print("No matchups found with >2% draw shift.")

if __name__ == "__main__":
    main()
