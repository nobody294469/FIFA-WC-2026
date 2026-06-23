import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

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

def main():
    print("Building Scenario 1 dataset...")
    df_s1, _ = _build_corpora()

    cutoff = pd.Timestamp("2023-01-01")
    train_s1 = df_s1[df_s1["date"] < cutoff].copy()
    test_s1 = df_s1[df_s1["date"] >= cutoff].copy()
    
    X_tr = train_s1[FEATURE_COLS]
    y_tr = train_s1["target"].astype(int)
    
    X_te = test_s1[FEATURE_COLS]
    y_te = test_s1["target"].astype(int)

    is_top20_te = (test_s1["h_rank"] <= 20) & (test_s1["a_rank"] <= 20)
    
    print("\nTraining Base XGBoost...")
    base_xgb = xgb.XGBClassifier(**XGB_PARAMS)
    base_xgb.fit(X_tr, y_tr)
    
    print("Training Global Platt (CalibratedClassifierCV)...")
    global_platt = CalibratedClassifierCV(estimator=xgb.XGBClassifier(**XGB_PARAMS), method='sigmoid', cv=3)
    global_platt.fit(X_tr, y_tr)
    
    print("Training Global Isotonic (CalibratedClassifierCV)...")
    global_iso = CalibratedClassifierCV(estimator=xgb.XGBClassifier(**XGB_PARAMS), method='isotonic', cv=3)
    global_iso.fit(X_tr, y_tr)

    print("Generating OOF predictions for Local Isotonic...")
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    oof_proba = np.zeros((len(X_tr), 3))
    for train_idx, val_idx in skf.split(X_tr, y_tr):
        m = xgb.XGBClassifier(**XGB_PARAMS)
        m.fit(X_tr.iloc[train_idx], y_tr.iloc[train_idx])
        oof_proba[val_idx] = m.predict_proba(X_tr.iloc[val_idx])

    is_top20_tr = (train_s1["h_rank"] <= 20) & (train_s1["a_rank"] <= 20)
    oof_top20 = oof_proba[is_top20_tr]
    y_top20 = y_tr[is_top20_tr].values
    
    print(f"Fitting Local Isotonic on {is_top20_tr.sum()} Top-20 matches...")
    local_ir_models = []
    for c in range(3):
        ir = IsotonicRegression(out_of_bounds='clip')
        # Isotonic regression expects 1D arrays
        ir.fit(oof_top20[:, c], (y_top20 == c).astype(float))
        local_ir_models.append(ir)

    # Hybrid Prediction Function
    def predict_hybrid(X, is_t20=None):
        global_proba = global_platt.predict_proba(X)
        raw_proba = base_xgb.predict_proba(X)
        
        final_proba = global_proba.copy()
        
        if is_t20 is not None and is_t20.any():
            iso_proba = np.zeros((is_t20.sum(), 3))
            for c in range(3):
                iso_proba[:, c] = local_ir_models[c].predict(raw_proba[is_t20, c])
            # Normalize
            iso_sums = iso_proba.sum(axis=1, keepdims=True)
            iso_sums[iso_sums == 0] = 1.0
            iso_proba /= iso_sums
            final_proba[is_t20] = iso_proba
            
        return final_proba

    # Evaluation
    methods = {
        "Raw XGB": lambda X: base_xgb.predict_proba(X),
        "Global Platt": lambda X: global_platt.predict_proba(X),
        "Global Isotonic": lambda X: global_iso.predict_proba(X),
        "Hybrid Calibration": predict_hybrid
    }

    print("\n=== OVERALL TEMPORAL HOLDOUT METRICS ===")
    print(f"{'Method':<20} | {'Log Loss':>8} | {'Brier':>8} | {'Avg Draw':>9} | {'Draw ECE':>8}")
    print("-" * 65)
    for name, pred_fn in methods.items():
        if name == "Hybrid Calibration":
            p = pred_fn(X_te, is_t20=is_top20_te)
        else:
            p = pred_fn(X_te)
        ll = log_loss(y_te, p, labels=[0,1,2])
        br = _multiclass_brier_score(y_te.values, p)
        ad = p[:, 1].mean() * 100
        ece = get_ece((y_te == 1).values, p[:, 1])
        print(f"{name:<20} | {ll:>8.5f} | {br:>8.5f} | {ad:>8.2f}% | {ece:>8.5f}")

    print("\n=== TOP-20 VS TOP-20 METRICS (120 matches) ===")
    print(f"{'Method':<20} | {'Log Loss':>8} | {'Brier':>8} | {'Avg Draw':>9} | {'Draw ECE':>8}")
    print("-" * 65)
    X_t20 = X_te[is_top20_te]
    y_t20 = y_te[is_top20_te]
    # For the Top-20 only metric, all rows are Top-20
    all_true_mask = np.ones(len(X_t20), dtype=bool)
    for name, pred_fn in methods.items():
        if name == "Hybrid Calibration":
            p = pred_fn(X_t20, is_t20=all_true_mask)
        else:
            p = pred_fn(X_t20)
        ll = log_loss(y_t20, p, labels=[0,1,2])
        br = _multiclass_brier_score(y_t20.values, p)
        ad = p[:, 1].mean() * 100
        ece = get_ece((y_t20 == 1).values, p[:, 1])
        print(f"{name:<20} | {ll:>8.5f} | {br:>8.5f} | {ad:>8.2f}% | {ece:>8.5f}")

    print("\n=== 50k WC2026 SIMULATIONS ===")
    
    # We need to build a ProbabilityMatrixBuilder that uses predict_hybrid
    # We'll mock the MatchOutcomeModel so PMB can use it
    class MockModel:
        def __init__(self, pred_fn):
            self.pred_fn = pred_fn
        def predict_proba(self, X):
            if "home_team" in X.columns:
                X = X.drop(columns=["home_team", "away_team", "date"], errors="ignore")
            return self.pred_fn(X[FEATURE_COLS])
            
    pipeline = MatchDataPipeline()
    elo = EloTracker()
    elo.batch_update(pipeline.matches)
    
    sim_results = {}
    for name, pred_fn in methods.items():
        print(f"Simulating {name}...")
        
        # For simulation, we need a special handling for Hybrid
        if name == "Hybrid Calibration":
            mock_model = MockModel(methods["Global Platt"])
        else:
            mock_model = MockModel(pred_fn)
            
        pmb = ProbabilityMatrixBuilder(mock_model)
        pmb.build(
            teams=WC2026_TEAMS,
            elo_tracker=elo,
            form_scores=pipeline.current_form_scores(),
            opp_adj_form_scores=pipeline.current_opp_adjusted_form_scores(),
            form_gd=pipeline.current_form_gd(),
            h2h_fn=pipeline.current_h2h,
            form_details=pipeline.current_form_details(),
            ko_resolution="coinflip"
        )
        
        if name == "Hybrid Calibration":
            # Identify Top 20 teams
            sorted_ratings = sorted(elo.ratings.values(), reverse=True)
            top20_teams = [t for t, e in elo.ratings.items() if e >= sorted_ratings[19]]
            top20_indices = [pmb.team_idx[t] for t in top20_teams if t in pmb.team_idx]
            
            # We need the RAW probabilities for Top-20 vs Top-20
            # Let's rebuild a PMB with Raw XGB
            mock_raw = MockModel(methods["Raw XGB"])
            pmb_raw = ProbabilityMatrixBuilder(mock_raw)
            pmb_raw.build(
                teams=WC2026_TEAMS,
                elo_tracker=elo,
                form_scores=pipeline.current_form_scores(),
                opp_adj_form_scores=pipeline.current_opp_adjusted_form_scores(),
                form_gd=pipeline.current_form_gd(),
                h2h_fn=pipeline.current_h2h,
                form_details=pipeline.current_form_details(),
                ko_resolution="coinflip"
            )
            
            for i in range(pmb.P_win.shape[0]):
                for j in range(pmb.P_win.shape[1]):
                    if i == j: continue
                    if i in top20_indices and j in top20_indices:
                        raw_w = pmb_raw.P_win[i,j]
                        raw_d = pmb_raw.P_draw[i,j]
                        raw_l = pmb_raw.P_loss[i,j]
                        
                        raw_arr = np.array([[raw_w, raw_d, raw_l]])
                        iso_proba = np.zeros((1, 3))
                        for c in range(3):
                            iso_proba[:, c] = local_ir_models[c].predict(raw_arr[:, c])
                            
                        s = iso_proba.sum()
                        if s > 0:
                            iso_proba /= s
                        else:
                            iso_proba = np.array([[1/3, 1/3, 1/3]])
                            
                        pmb.P_win[i,j] = iso_proba[0, 0]
                        pmb.P_draw[i,j] = iso_proba[0, 1]
                        pmb.P_loss[i,j] = iso_proba[0, 2]
                        pmb.P_ko[i,j] = iso_proba[0, 0] + (iso_proba[0, 1] * 0.5)

        sim = TournamentSimulator(pmb)
        res = sim.run(50000)
        df_res = res.to_dataframe()
        sim_results[name] = df_res
        
    # Print Champion probabilities for Top teams
    print(f"\n{'Team':<15} | {'Raw XGB':>10} | {'Global Platt':>12} | {'Global Iso':>10} | {'Hybrid':>10}")
    print("-" * 70)
    top_teams = sim_results["Global Platt"].head(10)["team"].tolist()
    
    for t in top_teams:
        vals = []
        for name in methods.keys():
            df = sim_results[name]
            val = df.loc[df["team"] == t, "champion_pct"].values[0]
            vals.append(val)
        print(f"{t:<15} | {vals[0]:>9.2f}% | {vals[1]:>11.2f}% | {vals[2]:>9.2f}% | {vals[3]:>9.2f}%")
        
    print("\nTop-3 Title Concentration:")
    for name in methods.keys():
        t3 = sim_results[name].head(3)["champion_pct"].sum()
        print(f"  {name:<20}: {t3:.1f}%")

if __name__ == "__main__":
    main()
