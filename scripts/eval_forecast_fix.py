import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core.config
from core.config import WC2026_TEAMS
from data.ingestion import MatchDataPipeline
from data.elo_tracker import EloTracker
from models.ml_engine import MatchOutcomeModel, ProbabilityMatrixBuilder
from simulation.monte_carlo import TournamentSimulator

def get_forecast_results(model: MatchOutcomeModel, pipeline, elo, num_sims=50000):
    form_scores = pipeline.current_form_scores()
    of_scores = pipeline.current_opp_adjusted_form_scores()
    form_gd = pipeline.current_form_gd()
    f_details = pipeline.current_form_details()
    h2h_fn = pipeline.current_h2h
    
    pmb = ProbabilityMatrixBuilder(model)
    pmb.build(
        teams = WC2026_TEAMS,
        elo_tracker = elo,
        form_scores = form_scores,
        opp_adj_form_scores = of_scores,
        form_gd = form_gd,
        form_details = f_details,
        h2h_fn = h2h_fn,
    )
    
    sim = TournamentSimulator(pmb)
    core.config.RANDOM_SEED = 42
    res = sim.run(n_sims=num_sims)
    
    # We also want to capture Spain vs France
    idx_h = WC2026_TEAMS.index("Spain")
    idx_a = WC2026_TEAMS.index("France")
    matrix_p = (pmb.P_win[idx_h, idx_a], pmb.P_draw[idx_h, idx_a], pmb.P_loss[idx_h, idx_a])
    
    return res.to_dataframe(), matrix_p

def main():
    print("Loading data pipeline...")
    pipeline = MatchDataPipeline()
    pipeline = MatchDataPipeline()
    elo = EloTracker()
    elo.batch_update(pipeline.matches)
    
    # Load model from disk
    print("Loading Hybrid model from disk...")
    model_hybrid = MatchOutcomeModel()
    model_hybrid.load()
    
    if model_hybrid.local_isotonic_models is None:
        print("ERROR: local_isotonic_models is still None! Persistence bug not fixed.")
        return
    else:
        print("VERIFIED: local_isotonic_models loaded successfully.")
    
    # Clone and strip for Platt
    model_platt = MatchOutcomeModel()
    model_platt.model = model_hybrid.model
    model_platt.calibrated_model = model_hybrid.calibrated_model
    model_platt.local_isotonic_models = None
    model_platt.feat_importance = model_hybrid.feat_importance
    model_platt._explainer = model_hybrid._explainer
    
    print("\nRunning Forced Global Platt Forecast...")
    df_platt, p_platt = get_forecast_results(model_platt, pipeline, elo, 50000)
    
    print("Running Forced Hybrid Forecast...")
    df_hybrid, p_hybrid = get_forecast_results(model_hybrid, pipeline, elo, 50000)
    
    print("\n--- Spain vs France Probability Vector (from PMB Matrix) ---")
    print(f"Global Platt PMB Matrix: {p_platt[0]:.2f} / {p_platt[1]:.2f} / {p_platt[2]:.2f}")
    print(f"Hybrid PMB Matrix:       {p_hybrid[0]:.2f} / {p_hybrid[1]:.2f} / {p_hybrid[2]:.2f}")
    
    targets = ["Spain", "Argentina", "France", "Brazil", "England"]
    print(f"\n{'Team':<12} | {'Global Platt %':<15} | {'Hybrid %':<10} | {'Shift':<10}")
    print("-" * 55)
    for t in targets:
        p_p = df_platt.loc[df_platt["team"] == t, "champion_pct"].values[0]
        p_h = df_hybrid.loc[df_hybrid["team"] == t, "champion_pct"].values[0]
        shift = p_h - p_p
        print(f"{t:<12} | {p_p:>14.1f}% | {p_h:>9.1f}% | {shift:>+9.1f}%")

if __name__ == "__main__":
    main()
