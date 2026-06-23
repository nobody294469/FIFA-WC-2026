import os
import sys
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.inspection import permutation_importance
from sklearn.metrics import log_loss

from core.logger import get_logger
from data.ingestion import MatchDataPipeline
from data.elo_tracker import EloTracker
from data.feature_engineering import build_training_features, FEATURE_COLS
from models.ml_engine import MatchOutcomeModel

log = get_logger("audit_form_importance")

TARGET_FEATURES = [
    "form_diff",
    "opp_adj_form_diff",
    "home_form",
    "away_form"
]

def main():
    print("Initializing Pipeline...")
    pipeline = MatchDataPipeline()
    elo = EloTracker()
    elo.batch_update(pipeline.matches)

    print("Building training features (this may take a moment)...")
    X, y = build_training_features(pipeline.training_df, elo)

    print("Loading production model...")
    model = MatchOutcomeModel()
    model.load()

    print("\n=== GAIN IMPORTANCE ===")
    total_gain = 0.0
    for feat in TARGET_FEATURES:
        gain = model.feat_importance.get(feat, 0.0)
        total_gain += gain
        print(f"{feat:20}: {gain:.4f}")
    print(f"{'Combined Form Gain':<20}: {total_gain:.4f} ({(total_gain)*100:.1f}%)")

    print("\n=== SHAP IMPORTANCE ===")
    dmatrix = xgb.DMatrix(X)
    contribs = model.model.get_booster().predict(dmatrix, pred_contribs=True)
    abs_shap = np.abs(contribs[:, :, :-1])
    mean_abs_shap = np.mean(abs_shap, axis=(0, 1))
    shap_importance = dict(zip(FEATURE_COLS, mean_abs_shap))
    
    total_shap_all = sum(shap_importance.values())
    total_form_shap = 0.0
    for feat in TARGET_FEATURES:
        val = shap_importance.get(feat, 0.0)
        total_form_shap += val
        print(f"{feat:20}: {val:.4f}")
    
    pct_shap = (total_form_shap / total_shap_all) * 100 if total_shap_all > 0 else 0
    print(f"{'Combined Form SHAP':<20}: {total_form_shap:.4f} ({pct_shap:.1f}% of total SHAP)")

    print("\n=== PERMUTATION IMPORTANCE ===")
    class Wrapper:
        def __init__(self, m):
            self.m = m
        def predict_proba(self, X_data):
            return self.m.predict_proba(pd.DataFrame(X_data, columns=FEATURE_COLS))
        def get_params(self, deep=True):
            return {}

    def log_loss_scorer(estimator, X_data, y_data):
        probs = estimator.predict_proba(X_data)
        return -log_loss(y_data, probs)

    wrapper = Wrapper(model.calibrated_model)
    perm_res = permutation_importance(wrapper, X.values, y.values, scoring=log_loss_scorer, n_repeats=5, random_state=42, n_jobs=-1)
    
    perm_importance = dict(zip(FEATURE_COLS, perm_res.importances_mean))
    
    total_perm_all = sum([v for v in perm_importance.values() if v > 0]) # exclude negative noise
    total_form_perm = 0.0
    for feat in TARGET_FEATURES:
        val = perm_importance.get(feat, 0.0)
        # only add positive permutation importance to total
        if val > 0:
            total_form_perm += val
        print(f"{feat:20}: {val:.5f}")
        
    pct_perm = (total_form_perm / total_perm_all) * 100 if total_perm_all > 0 else 0
    print(f"{'Combined Form Perm':<20}: {total_form_perm:.5f} ({pct_perm:.1f}% of total positive Perm Importance)")

if __name__ == "__main__":
    main()
