import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from itertools import combinations
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core.config
from core.config import WC2026_TEAMS
from data.ingestion import MatchDataPipeline, load_raw_results, build_match_records
from data.elo_tracker import EloTracker
from data.feature_engineering import build_training_features, build_inference_features, FEATURE_COLS
from models.ml_engine import MatchOutcomeModel, _multiclass_brier_score
from simulation.monte_carlo import TournamentSimulator
from sklearn.metrics import log_loss

def get_ece(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

def eval_metrics(model, df_te, X_te, y_te, top_thresh=20):
    proba = model.predict_proba(X_te)
    
    # Global metrics
    g_ll = log_loss(y_te, proba, labels=[0,1,2])
    g_brier = _multiclass_brier_score(y_te.values, proba)
    y_true_draw_g = (y_te == 1).astype(int)
    y_prob_draw_g = proba[:, 1]
    g_ece = get_ece(y_true_draw_g.values, y_prob_draw_g)
    
    # Subset metrics
    is_top = (df_te["h_rank"] <= top_thresh) & (df_te["a_rank"] <= top_thresh)
    if is_top.sum() == 0:
        return g_ll, g_brier, g_ece, 0, 0, 0
    
    proba_sub = proba[is_top]
    y_te_sub = y_te[is_top]
    s_ll = log_loss(y_te_sub, proba_sub, labels=[0,1,2])
    s_brier = _multiclass_brier_score(y_te_sub.values, proba_sub)
    y_true_draw_s = (y_te_sub == 1).astype(int)
    y_prob_draw_s = proba_sub[:, 1]
    s_ece = get_ece(y_true_draw_s.values, y_prob_draw_s)
    
    return g_ll, g_brier, g_ece, s_ll, s_brier, s_ece

def run_simulation(model, pipeline, elo_tracker, num_sims=100000, seeds=[1, 2, 3, 4, 5]):
    from models.ml_engine import ProbabilityMatrixBuilder
    
    form_scores         = pipeline.current_form_scores()
    of_scores           = pipeline.current_opp_adjusted_form_scores()
    form_gd             = pipeline.current_form_gd()
    f_details           = pipeline.current_form_details()
    
    pmb = ProbabilityMatrixBuilder(model)
    pmb.build(
        teams       = WC2026_TEAMS,
        elo_tracker = elo_tracker,
        form_scores = form_scores,
        opp_adj_form_scores = of_scores,
        form_gd     = form_gd,
        form_details= f_details,
        h2h_fn      = pipeline.current_h2h,
    )
        
    sim = TournamentSimulator(pmb)
    
    res_dict = defaultdict(list)
    targets = ["Spain", "Argentina", "France", "Brazil", "England", "Portugal"]
    
    for s in seeds:
        core.config.RANDOM_SEED = s
        results = sim.run(n_sims=num_sims)
        df = results.to_dataframe()
        for t in targets:
            pct = df.loc[df["team"] == t, "champion_pct"].values[0]
            res_dict[t].append(pct)
            
    means = {k: np.mean(v) for k, v in res_dict.items()}
    stds = {k: np.std(v) for k, v in res_dict.items()}
    return means, stds

def main():
    print("Loading data pipeline...")
    pipeline = MatchDataPipeline()
    
    # elo tracker needs to be fully built from the pipeline or from raw?
    # MatchDataPipeline already handles all parsing. We just need an EloTracker built from results.csv
    # Actually wait, elo tracker needs all_matches. 
    raw = load_raw_results()
    all_matches = build_match_records(raw)
    
    elo = EloTracker()
    elo.batch_update(all_matches)
    
    print("Building training/test sets...")
    df_train_raw = pipeline.training_df[pipeline.training_df["date"] < pd.Timestamp("2024-01-01")].copy()
    df_test_raw = pipeline.training_df[pipeline.training_df["date"] >= pd.Timestamp("2024-01-01")].copy()
    
    # No, let's just build proper features
    X_full, y_full = build_training_features(pipeline.training_df, elo)
    
    cutoff_idx = len(df_train_raw)
    X_tr = X_full.iloc[:cutoff_idx]
    y_tr = y_full.iloc[:cutoff_idx]
    X_te = X_full.iloc[cutoff_idx:]
    y_te = y_full.iloc[cutoff_idx:]
    df_te = pipeline.training_df.iloc[cutoff_idx:].copy()
    df_te["h_rank"] = X_te["h_rank"].values
    df_te["a_rank"] = X_te["a_rank"].values

    configs = [
        {"name": "Global Platt", "hybrid": False, "thresh": 20},
        {"name": "Hybrid (T=18)", "hybrid": True, "thresh": 18},
        {"name": "Hybrid (T=20)", "hybrid": True, "thresh": 20},
        {"name": "Hybrid (T=22)", "hybrid": True, "thresh": 22},
    ]
    
    results = []
    
    for c in configs:
        print(f"\\n=== Testing {c['name']} ===")
        core.config.HYBRID_CALIBRATION_RANK_THRESHOLD = c["thresh"]
        
        # Train
        model = MatchOutcomeModel()
        model.fit(X_tr, y_tr)
        
        if not c["hybrid"]:
            # Disable hybrid for inference
            model.local_isotonic_models = None
            
        print("Evaluating holdout metrics (2024-2025)...")
        gll, gbrier, gece, sll, sbrier, sece = eval_metrics(model, df_te, X_te, y_te, top_thresh=20)
        
        print("Running 5x 100k simulations...")
        means, stds = run_simulation(model, pipeline, elo, num_sims=100000, seeds=[101, 102, 103, 104, 105])
        
        results.append({
            "Method": c["name"],
            "Log Loss": gll,
            "Brier": gbrier,
            "ECE": gece,
            "Spain %": f"{means['Spain']:.1f} +/- {stds['Spain']:.2f}",
            "Argentina %": f"{means['Argentina']:.1f} +/- {stds['Argentina']:.2f}",
            "France %": f"{means['France']:.1f} +/- {stds['France']:.2f}",
            "Brazil %": f"{means['Brazil']:.1f} +/- {stds['Brazil']:.2f}",
            "England %": f"{means['England']:.1f} +/- {stds['England']:.2f}",
            "Portugal %": f"{means['Portugal']:.1f} +/- {stds['Portugal']:.2f}",
        })
        
    print("\\n\\n================ FINAL AUDIT TABLE ================")
    df_res = pd.DataFrame(results)
    df_res.to_csv("final_audit_results.csv", index=False)
    print(df_res.to_string(index=False))

if __name__ == "__main__":
    main()
