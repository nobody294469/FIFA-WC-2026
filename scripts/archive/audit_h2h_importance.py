import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.inspection import permutation_importance
from sklearn.metrics import log_loss

from core.logger import get_logger
from data.ingestion import MatchDataPipeline
from data.elo_tracker import EloTracker
from data.feature_engineering import build_training_features, FEATURE_COLS
from models.ml_engine import MatchOutcomeModel

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

    # 1. Gain Importance
    print("\n=== GAIN IMPORTANCE ===")
    gain = model.feat_importance.get('h2h_diff', 0.0)
    print(f"h2h_diff Gain Importance: {gain:.4f}")
    print("\nTop 5 features by Gain:")
    print(model.importance_report().head(5).to_string(index=False))

    # 2. SHAP Importance
    print("\n=== SHAP IMPORTANCE ===")
    dmatrix = xgb.DMatrix(X)
    contribs = model.model.get_booster().predict(dmatrix, pred_contribs=True)
    # contribs shape: (n_samples, n_classes, n_features + 1)
    
    # Exclude the bias term (last element) and average absolute SHAP values across samples and classes
    abs_shap = np.abs(contribs[:, :, :-1]) # (n_samples, n_classes, n_features)
    mean_abs_shap = np.mean(abs_shap, axis=(0, 1)) # (n_features,)
    
    shap_importance = dict(zip(FEATURE_COLS, mean_abs_shap))
    shap_sorted = sorted(shap_importance.items(), key=lambda x: -x[1])
    h2h_shap = shap_importance.get('h2h_diff', 0.0)

    print(f"h2h_diff Mean |SHAP| (Log-Odds Contribution): {h2h_shap:.4f}")
    print("\nTop 5 features by Mean |SHAP|:")
    for feat, val in shap_sorted[:5]:
        print(f"{feat:25}: {val:.4f}")

    # 3. Permutation Importance
    print("\n=== PERMUTATION IMPORTANCE ===")
    print("Calculating permutation importance (this involves predicting 5 times per feature)...")
    
    class Wrapper:
        def __init__(self, m):
            self.m = m
        def predict_proba(self, X_data):
            # Calibrated model handles DataFrames natively
            return self.m.predict_proba(pd.DataFrame(X_data, columns=FEATURE_COLS))
        def get_params(self, deep=True):
            return {}

    def log_loss_scorer(estimator, X_data, y_data):
        probs = estimator.predict_proba(X_data)
        return -log_loss(y_data, probs)

    wrapper = Wrapper(model.calibrated_model)
    # Use negative log loss so higher is better. Permutation importance shows the DROP in this score.
    # Therefore, a positive importance means the score dropped (worsened) when the feature was permuted.
    perm_res = permutation_importance(wrapper, X.values, y.values, scoring=log_loss_scorer, n_repeats=5, random_state=42, n_jobs=-1)
    
    perm_importance = dict(zip(FEATURE_COLS, perm_res.importances_mean))
    perm_sorted = sorted(perm_importance.items(), key=lambda x: -x[1])
    h2h_perm = perm_importance.get('h2h_diff', 0.0)

    print(f"h2h_diff Permutation Importance (increase in log loss when shuffled): {h2h_perm:.4f}")
    print("\nTop 5 features by Permutation Importance:")
    for feat, val in perm_sorted[:5]:
        print(f"{feat:25}: {val:.5f}")

if __name__ == "__main__":
    main()
