import sys
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core.config
from core.config import WC2026_TEAMS
from data.ingestion import MatchDataPipeline, load_raw_results, build_match_records
from data.elo_tracker import EloTracker
from data.feature_engineering import build_training_features, build_inference_features, FEATURE_COLS
from models.ml_engine import MatchOutcomeModel, ProbabilityMatrixBuilder
from simulation.monte_carlo import TournamentSimulator

def main():
    print("Loading data pipeline...")
    pipeline = MatchDataPipeline()
    raw = load_raw_results()
    all_matches = build_match_records(raw)
    
    elo = EloTracker()
    elo.batch_update(all_matches)
    
    X_full, y_full = build_training_features(pipeline.training_df, elo)
    
    # Train model
    core.config.HYBRID_CALIBRATION_RANK_THRESHOLD = 20
    model = MatchOutcomeModel()
    print("Fitting model...")
    model.fit(X_full, y_full)
    
    # Probability Matrix Builder
    form_scores = pipeline.current_form_scores()
    of_scores = pipeline.current_opp_adjusted_form_scores()
    form_gd = pipeline.current_form_gd()
    f_details = pipeline.current_form_details()
    
    pairs = [(WC2026_TEAMS[i], WC2026_TEAMS[j]) for i in range(len(WC2026_TEAMS)) for j in range(len(WC2026_TEAMS)) if i != j]
    
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
    
    print("\n--- 1. Hybrid Calibration Trigger inside Inference ---")
    is_elite_inf = (X_inf["h_rank"] <= 20) & (X_inf["a_rank"] <= 20)
    print(f"Total combinations in probability matrix: {len(X_inf)}")
    print(f"Matches strictly routed through local Isotonic Calibrator: {is_elite_inf.sum()} / {len(X_inf)} ({(is_elite_inf.sum()/len(X_inf))*100:.1f}%)")
    
    print("\n--- 2. Elite Matchup Probabilities (Top-20 vs Top-20) ---")
    elite_indices = np.where(is_elite_inf)[0]
    
    # Take 20 elite matchups
    selected_indices = elite_indices[:20] 
    
    # To get realistic matchups, we can pick some named ones instead of just the first 20 which might all be Argentina vs X.
    named_elites = [
        ("Spain", "Argentina"), ("France", "England"), ("Brazil", "Portugal"), 
        ("Germany", "Netherlands"), ("Italy", "Belgium"), ("Spain", "France"),
        ("Argentina", "Brazil"), ("England", "Germany"), ("Portugal", "Italy"),
        ("Netherlands", "Belgium"), ("Mexico", "USA"), ("Uruguay", "Colombia"),
        ("Croatia", "Switzerland"), ("Japan", "South Korea"), ("Morocco", "Senegal"),
        ("France", "Brazil"), ("Argentina", "England"), ("Spain", "Portugal"),
        ("Germany", "Italy"), ("Netherlands", "Croatia")
    ]
    
    print(f"{'Matchup':<25} | {'Raw XGB (W/D/L)':<22} | {'Global Platt (W/D/L)':<22} | {'Hybrid (W/D/L)':<22}")
    print("-" * 100)
    
    for h, a in named_elites:
        if h not in WC2026_TEAMS or a not in WC2026_TEAMS:
            continue
        try:
            idx = pairs.index((h, a))
        except ValueError:
            continue
            
        row = X_inf.iloc[[idx]]
        
        raw_p = model.model.predict_proba(row[FEATURE_COLS])[0]
        platt_p = model.calibrated_model.predict_proba(row[FEATURE_COLS])[0]
        hybrid_p = model.predict_proba(row)[0]
        
        raw_s = f"{raw_p[0]:.2f}/{raw_p[1]:.2f}/{raw_p[2]:.2f}"
        platt_s = f"{platt_p[0]:.2f}/{platt_p[1]:.2f}/{platt_p[2]:.2f}"
        hybrid_s = f"{hybrid_p[0]:.2f}/{hybrid_p[1]:.2f}/{hybrid_p[2]:.2f}"
        
        print(f"{h + ' vs ' + a:<25} | {raw_s:<22} | {platt_s:<22} | {hybrid_s:<22}")

    print("\n--- 3. Simulator Output Verification ---")
    pmb = ProbabilityMatrixBuilder(model)
    pmb.build(
        teams = WC2026_TEAMS,
        elo_tracker = elo,
        form_scores = form_scores,
        opp_adj_form_scores = of_scores,
        form_gd = form_gd,
        form_details = f_details,
        h2h_fn = pipeline.current_h2h,
    )
    
    # Verify that Hybrid is in the matrix for Spain vs Argentina
    idx_h = WC2026_TEAMS.index("Spain")
    idx_a = WC2026_TEAMS.index("Argentina")
    
    # Calculate the neutral averaged hybrid probability manually
    idx_pair1 = pairs.index(("Spain", "Argentina"))
    idx_pair2 = pairs.index(("Argentina", "Spain"))
    p1 = model.predict_proba(X_inf.iloc[[idx_pair1]])[0]
    p2 = model.predict_proba(X_inf.iloc[[idx_pair2]])[0]
    expected_draw = (p1[1] + p2[1]) / 2.0
    expected_win = (p1[0] + p2[2]) / 2.0
    total = expected_win + expected_draw + (p2[0] + p1[2]) / 2.0
    
    expected_draw_norm = expected_draw / total
    
    actual_matrix_draw = pmb.P_draw[idx_h, idx_a]
    print(f"Spain vs Argentina Expected Hybrid Draw in PMB: {expected_draw_norm:.4f}")
    print(f"Spain vs Argentina Actual PMB Draw Matrix: {actual_matrix_draw:.4f}")
    assert np.isclose(expected_draw_norm, actual_matrix_draw), "PMB is NOT using Hybrid probabilities!"
    print("VERIFIED: ProbabilityMatrixBuilder accurately receives Hybrid probabilities.")

    print("\n--- 4. Champion Probability Shift (100,000 sims) ---")
    # Simulate with Hybrid
    sim_hybrid = TournamentSimulator(pmb)
    core.config.RANDOM_SEED = 42
    res_hybrid = sim_hybrid.run(n_sims=100000).to_dataframe()
    
    # Create a Global Platt only model by stripping local isotonic models
    model_platt = MatchOutcomeModel()
    model_platt.model = model.model
    model_platt.calibrated_model = model.calibrated_model
    model_platt.local_isotonic_models = None 
    model_platt.feat_importance = model.feat_importance
    model_platt._explainer = model._explainer
    
    pmb_platt = ProbabilityMatrixBuilder(model_platt)
    pmb_platt.build(
        teams = WC2026_TEAMS,
        elo_tracker = elo,
        form_scores = form_scores,
        opp_adj_form_scores = of_scores,
        form_gd = form_gd,
        form_details = f_details,
        h2h_fn = pipeline.current_h2h,
    )
    
    sim_platt = TournamentSimulator(pmb_platt)
    core.config.RANDOM_SEED = 42
    res_platt = sim_platt.run(n_sims=100000).to_dataframe()
    
    targets = ["Spain", "Argentina", "France", "Brazil", "England"]
    print(f"{'Team':<12} | {'Global Platt %':<15} | {'Hybrid %':<10} | {'Shift':<10}")
    print("-" * 55)
    for t in targets:
        p_platt = res_platt.loc[res_platt["team"] == t, "champion_pct"].values[0]
        p_hybrid = res_hybrid.loc[res_hybrid["team"] == t, "champion_pct"].values[0]
        shift = p_hybrid - p_platt
        print(f"{t:<12} | {p_platt:>14.1f}% | {p_hybrid:>9.1f}% | {shift:>+9.1f}%")
        
if __name__ == "__main__":
    main()
