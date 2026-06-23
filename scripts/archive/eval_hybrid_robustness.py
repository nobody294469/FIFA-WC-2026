import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
import scipy.stats as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import XGB_PARAMS, WC2026_TEAMS
from data.feature_engineering import FEATURE_COLS
from scripts.eval_calibration_ablation import _build_corpora, get_ece
from models.ml_engine import _multiclass_brier_score, ProbabilityMatrixBuilder
from simulation.monte_carlo import TournamentSimulator
from data.ingestion import MatchDataPipeline
from data.elo_tracker import EloTracker

class MockModel:
    def __init__(self, pred_fn):
        self.pred_fn = pred_fn
    def predict_proba(self, X):
        if "home_team" in X.columns:
            X = X.drop(columns=["home_team", "away_team", "date"], errors="ignore")
        return self.pred_fn(X[FEATURE_COLS])

def build_hybrid_components(df_tr, y_tr):
    X_tr = df_tr[FEATURE_COLS]
    base_xgb = xgb.XGBClassifier(**XGB_PARAMS)
    base_xgb.fit(X_tr, y_tr)
    
    global_platt = CalibratedClassifierCV(estimator=xgb.XGBClassifier(**XGB_PARAMS), method='sigmoid', cv=3)
    global_platt.fit(X_tr, y_tr)
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    oof_proba = np.zeros((len(X_tr), 3))
    for train_idx, val_idx in skf.split(X_tr, y_tr):
        m = xgb.XGBClassifier(**XGB_PARAMS)
        m.fit(X_tr.iloc[train_idx], y_tr.iloc[train_idx])
        oof_proba[val_idx] = m.predict_proba(X_tr.iloc[val_idx])

    is_top20_tr = (df_tr["h_rank"] <= 20) & (df_tr["a_rank"] <= 20)
    oof_top20 = oof_proba[is_top20_tr]
    y_top20 = y_tr[is_top20_tr].values
    
    local_ir_models = []
    for c in range(3):
        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(oof_top20[:, c], (y_top20 == c).astype(float))
        local_ir_models.append(ir)
        
    return base_xgb, global_platt, local_ir_models, is_top20_tr.sum(), y_top20

def get_hybrid_pred_fn(base_xgb, global_platt, local_ir_models):
    def predict_hybrid(X, is_t20=None):
        global_proba = global_platt.predict_proba(X)
        raw_proba = base_xgb.predict_proba(X)
        
        final_proba = global_proba.copy()
        
        if is_t20 is not None and is_t20.any():
            iso_proba = np.zeros((is_t20.sum(), 3))
            for c in range(3):
                iso_proba[:, c] = local_ir_models[c].predict(raw_proba[is_t20, c])
            iso_sums = iso_proba.sum(axis=1, keepdims=True)
            iso_sums[iso_sums == 0] = 1.0
            iso_proba /= iso_sums
            final_proba[is_t20] = iso_proba
            
        return final_proba
    return predict_hybrid

def main():
    print("Building Scenario 1 dataset...")
    df_s1, _ = _build_corpora()

    print("\n" + "="*50)
    print("1. Elite Calibration Training Sample Size Audit")
    print("="*50)
    # Using the standard 2023 cutoff to inspect training data
    cutoff = pd.Timestamp("2023-01-01")
    train_s1 = df_s1[df_s1["date"] < cutoff].copy()
    
    is_top20_tr = (train_s1["h_rank"] <= 20) & (train_s1["a_rank"] <= 20)
    y_top20 = train_s1[is_top20_tr]["target"].astype(int)
    
    print(f"Total Top-20 vs Top-20 training matches: {len(y_top20)}")
    print("Class Balance:")
    print(f"  Home Win: {np.mean(y_top20 == 2):.1%}")
    print(f"  Draw:     {np.mean(y_top20 == 1):.1%}")
    print(f"  Away Win: {np.mean(y_top20 == 0):.1%}")
    
    if len(y_top20) < 300:
        print("WARNING: Sample size might be dangerously low for Isotonic Regression.")
    else:
        print("OK: Sample size is adequate for 3-class Isotonic Regression.")


    print("\n" + "="*50)
    print("2. 2024-2025 Holdout Evaluation")
    print("="*50)
    # Train on < 2024, Evaluate on >= 2024
    cutoff_24 = pd.Timestamp("2024-01-01")
    train_24 = df_s1[df_s1["date"] < cutoff_24].copy()
    test_24 = df_s1[df_s1["date"] >= cutoff_24].copy()
    
    y_tr24 = train_24["target"].astype(int)
    
    base_xgb24, global_platt24, local_ir_models24, n_t20_24, _ = build_hybrid_components(train_24, y_tr24)
    pred_hybrid24 = get_hybrid_pred_fn(base_xgb24, global_platt24, local_ir_models24)
    pred_platt24 = global_platt24.predict_proba
    
    X_te24 = test_24[FEATURE_COLS]
    y_te24 = test_24["target"].astype(int)
    is_top20_te24 = (test_24["h_rank"] <= 20) & (test_24["a_rank"] <= 20)
    
    print("OVERALL (2024+)")
    for name, fn in [("Global Platt", pred_platt24), ("Hybrid", lambda x: pred_hybrid24(x, is_top20_te24))]:
        p = fn(X_te24)
        ll = log_loss(y_te24, p, labels=[0,1,2])
        br = _multiclass_brier_score(y_te24.values, p)
        print(f"  {name:<15}: LogLoss={ll:.4f} | Brier={br:.4f}")
        
    print(f"\nTOP-20 vs TOP-20 ONLY (2024+, N={is_top20_te24.sum()})")
    X_t24 = X_te24[is_top20_te24]
    y_t24 = y_te24[is_top20_te24]
    all_true = np.ones(len(X_t24), dtype=bool)
    
    for name, fn in [("Global Platt", pred_platt24), ("Hybrid", lambda x: pred_hybrid24(x, all_true))]:
        p = fn(X_t24)
        ll = log_loss(y_t24, p, labels=[0,1,2])
        br = _multiclass_brier_score(y_t24.values, p)
        ad = p[:, 1].mean() * 100
        print(f"  {name:<15}: LogLoss={ll:.4f} | Brier={br:.4f} | AvgDraw={ad:.1f}%")


    print("\n" + "="*50)
    print("3 & 4. Monte Carlo Stability (5x 50,000 runs)")
    print("="*50)
    
    # Use full available data to train for forward-looking 2026 simulation
    # Let's train using <2023 for fairness to original benchmark, or just use all data? 
    # The benchmark prompt didn't specify. We'll use all data up to today to generate realistic PMB.
    y_full = df_s1["target"].astype(int)
    
    base_xgb, global_platt, local_ir_models, _, _ = build_hybrid_components(df_s1, y_full)
    pred_hybrid = get_hybrid_pred_fn(base_xgb, global_platt, local_ir_models)
    
    pipeline = MatchDataPipeline()
    elo = EloTracker()
    elo.batch_update(pipeline.matches)
    
    sorted_ratings = sorted(elo.ratings.values(), reverse=True)
    top20_teams = [t for t, e in elo.ratings.items() if e >= sorted_ratings[19]]
    
    mock_model = MockModel(global_platt.predict_proba)
    pmb = ProbabilityMatrixBuilder(mock_model)
    pmb.build(
        teams=WC2026_TEAMS, elo_tracker=elo, form_scores=pipeline.current_form_scores(),
        opp_adj_form_scores=pipeline.current_opp_adjusted_form_scores(), form_gd=pipeline.current_form_gd(),
        h2h_fn=pipeline.current_h2h, form_details=pipeline.current_form_details(), ko_resolution="coinflip"
    )
    
    top20_indices = [pmb.team_idx[t] for t in top20_teams if t in pmb.team_idx]
    
    mock_raw = MockModel(base_xgb.predict_proba)
    pmb_raw = ProbabilityMatrixBuilder(mock_raw)
    pmb_raw.build(
        teams=WC2026_TEAMS, elo_tracker=elo, form_scores=pipeline.current_form_scores(),
        opp_adj_form_scores=pipeline.current_opp_adjusted_form_scores(), form_gd=pipeline.current_form_gd(),
        h2h_fn=pipeline.current_h2h, form_details=pipeline.current_form_details(), ko_resolution="coinflip"
    )
    
    for i in range(pmb.P_win.shape[0]):
        for j in range(pmb.P_win.shape[1]):
            if i == j: continue
            if i in top20_indices and j in top20_indices:
                raw_w, raw_d, raw_l = pmb_raw.P_win[i,j], pmb_raw.P_draw[i,j], pmb_raw.P_loss[i,j]
                raw_arr = np.array([[raw_w, raw_d, raw_l]])
                iso_proba = np.zeros((1, 3))
                for c in range(3):
                    iso_proba[:, c] = local_ir_models[c].predict(raw_arr[:, c])
                s = iso_proba.sum()
                if s > 0: iso_proba /= s
                else: iso_proba = np.array([[1/3, 1/3, 1/3]])
                
                pmb.P_win[i,j] = iso_proba[0, 0]
                pmb.P_draw[i,j] = iso_proba[0, 1]
                pmb.P_loss[i,j] = iso_proba[0, 2]
                pmb.P_ko[i,j] = iso_proba[0, 0] + (iso_proba[0, 1] * 0.5)

    n_runs = 5
    n_sims = 50000
    champion_pcts = {t: [] for t in WC2026_TEAMS}
    
    for r in range(n_runs):
        print(f"  Run {r+1}/{n_runs} (50k sims)...")
        sim = TournamentSimulator(pmb)
        res = sim.run(n_sims)
        df = res.to_dataframe()
        for _, row in df.iterrows():
            champion_pcts[row["team"]].append(row["champion_pct"])

    print("\nTop 10 Teams (Champion Probability 95% CI):")
    print(f"{'Team':<15} | {'Mean%':>8} | {'95% CI':>15}")
    print("-" * 45)
    
    # Calculate means to sort
    means = {t: np.mean(champion_pcts[t]) for t in WC2026_TEAMS}
    sorted_teams = sorted(WC2026_TEAMS, key=lambda t: means[t], reverse=True)[:10]
    
    for t in sorted_teams:
        arr = np.array(champion_pcts[t])
        mean = np.mean(arr)
        # 95% CI using t-distribution
        ci = st.t.interval(0.95, df=len(arr)-1, loc=np.mean(arr), scale=st.sem(arr))
        ci_str = f"[{ci[0]:.2f}%, {ci[1]:.2f}%]"
        print(f"{t:<15} | {mean:>7.2f}% | {ci_str:>15}")

if __name__ == "__main__":
    main()
