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
from models.ml_engine import _multiclass_brier_score

def build_hybrid_components(df_tr, y_tr, threshold):
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

    is_elite_tr = (df_tr["h_rank"] <= threshold) & (df_tr["a_rank"] <= threshold)
    oof_elite = oof_proba[is_elite_tr]
    y_elite = y_tr[is_elite_tr].values
    
    local_ir_models = []
    for c in range(3):
        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(oof_elite[:, c], (y_elite == c).astype(float))
        local_ir_models.append(ir)
        
    return base_xgb, global_platt, local_ir_models, is_elite_tr.sum(), y_elite

def get_hybrid_pred_fn(base_xgb, global_platt, local_ir_models):
    def predict_hybrid(X, is_elite=None):
        global_proba = global_platt.predict_proba(X)
        raw_proba = base_xgb.predict_proba(X)
        final_proba = global_proba.copy()
        
        if is_elite is not None and is_elite.any():
            iso_proba = np.zeros((is_elite.sum(), 3))
            for c in range(3):
                iso_proba[:, c] = local_ir_models[c].predict(raw_proba[is_elite, c])
            iso_sums = iso_proba.sum(axis=1, keepdims=True)
            iso_sums[iso_sums == 0] = 1.0
            iso_proba /= iso_sums
            final_proba[is_elite] = iso_proba
            
        return final_proba
    return predict_hybrid

def main():
    print("Building Scenario 1 dataset...")
    df_s1, _ = _build_corpora()
    cutoff_24 = pd.Timestamp("2024-01-01")
    train_24 = df_s1[df_s1["date"] < cutoff_24].copy()
    test_24 = df_s1[df_s1["date"] >= cutoff_24].copy()
    
    y_tr24 = train_24["target"].astype(int)
    X_te24 = test_24[FEATURE_COLS]
    y_te24 = test_24["target"].astype(int)
    
    print("\nEvaluating Global Platt (Baseline):")
    # Quick global platt
    X_tr = train_24[FEATURE_COLS]
    platt = CalibratedClassifierCV(estimator=xgb.XGBClassifier(**XGB_PARAMS), method='sigmoid', cv=3)
    platt.fit(X_tr, y_tr24)
    base_p = platt.predict_proba(X_te24)
    overall_ll = log_loss(y_te24, base_p, labels=[0,1,2])
    overall_br = _multiclass_brier_score(y_te24.values, base_p)
    print(f"Overall LogLoss: {overall_ll:.4f} | Brier: {overall_br:.4f}")
    
    thresholds = [10, 15, 20, 25, 30]
    
    print("\n" + "="*80)
    print(f"{'Threshold':<10} | {'Train N':<8} | {'Ovr LL':<8} | {'Ovr Brier':<10} | {'Sub LL':<8} | {'Sub Brier':<10} | {'Avg Draw%':<10}")
    print("-" * 80)
    
    for t in thresholds:
        base_xgb, global_platt, local_ir_models, n_tr, _ = build_hybrid_components(train_24, y_tr24, t)
        pred_hybrid = get_hybrid_pred_fn(base_xgb, global_platt, local_ir_models)
        
        is_elite_te = (test_24["h_rank"] <= t) & (test_24["a_rank"] <= t)
        
        # Overall
        p_ovr = pred_hybrid(X_te24, is_elite_te)
        ll_ovr = log_loss(y_te24, p_ovr, labels=[0,1,2])
        br_ovr = _multiclass_brier_score(y_te24.values, p_ovr)
        
        # Subset
        p_sub = p_ovr[is_elite_te]
        y_sub = y_te24[is_elite_te]
        
        if len(y_sub) > 0:
            ll_sub = log_loss(y_sub, p_sub, labels=[0,1,2])
            br_sub = _multiclass_brier_score(y_sub.values, p_sub)
            ad_sub = p_sub[:, 1].mean() * 100
        else:
            ll_sub = 0; br_sub = 0; ad_sub = 0
            
        print(f"Top-{t:<6} | {n_tr:<8} | {ll_ovr:.4f}   | {br_ovr:.4f}     | {ll_sub:.4f}   | {br_sub:.4f}      | {ad_sub:.1f}%")

if __name__ == "__main__":
    main()
