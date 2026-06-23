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

from core.logger import get_logger
from data.ingestion import MatchDataPipeline
from data.elo_tracker import EloTracker
from data.feature_engineering import build_training_features, FEATURE_COLS
from models.ml_engine import MatchOutcomeModel

log = get_logger("eval_h2h_addition")

# We pulled these from the previously run Isotonic benchmark for the Platt (Sigmoid) model which represents our Baseline
BASELINE_METRICS = {
    "Accuracy": 0.5019,
    "Log Loss": 1.0238,
    "Brier": 0.6135,
    "Macro F1": 0.3584
}

def main():
    print("Initializing Pipeline...")
    pipeline = MatchDataPipeline()
    elo = EloTracker()
    elo.batch_update(pipeline.matches)

    print(f"Building training features (this may take a moment)...")
    print(f"Current FEATURE_COLS: {FEATURE_COLS}")
    
    if "h2h_diff" not in FEATURE_COLS:
        raise ValueError("h2h_diff is STILL not in FEATURE_COLS. The bug was not patched correctly.")

    X, y = build_training_features(pipeline.training_df, elo)

    print("\nTraining production model with h2h_diff included...")
    model = MatchOutcomeModel()
    model.fit(X, y)
    
    print("\nSaving new production model...")
    model.save()
    
    # Evaluate Platt Calibrated Model which is the production standard
    # ml_engine.py evaluates raw model in CV loop by default, but we can evaluate the calibrated model.
    # Actually, ml_engine cv_metrics stores raw xgboost metrics.
    # To get a true 1-to-1 with baseline, let's just look at eval_isotonic_calibration.py behavior 
    # Or just run 5-fold CV for Platt here like we did before.
    
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.calibration import CalibratedClassifierCV
    from core.config import XGB_PARAMS, MODEL_CV_FOLDS
    from sklearn.metrics import accuracy_score, f1_score, log_loss
    
    def _multiclass_brier_score(y_true: pd.Series, proba: np.ndarray) -> float:
        y_idx = np.asarray(y_true, dtype=np.int64)
        y_onehot = np.eye(proba.shape[1], dtype=np.float64)[y_idx]
        return float(np.mean(np.sum((proba - y_onehot) ** 2, axis=1)))

    tscv = TimeSeriesSplit(n_splits=MODEL_CV_FOLDS)
    
    oof_y = []
    oof_platt = []

    print(f"Starting {MODEL_CV_FOLDS}-fold Nested CV evaluation for Platt Calibrated Model...")
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        
        # 2. Platt (Sigmoid)
        clf_platt = CalibratedClassifierCV(estimator=xgb.XGBClassifier(**XGB_PARAMS), method='sigmoid', cv=3)
        clf_platt.fit(X_tr, y_tr)
        prob_platt = clf_platt.predict_proba(X_val)
        
        oof_y.extend(y_val)
        oof_platt.append(prob_platt)

    oof_y = np.array(oof_y)
    oof_platt = np.vstack(oof_platt)

    preds = oof_platt.argmax(axis=1)
    acc = accuracy_score(oof_y, preds)
    ll = log_loss(oof_y, oof_platt)
    brier = _multiclass_brier_score(oof_y, oof_platt)
    f1 = f1_score(oof_y, preds, average="macro")

    print("\n=== FULL 5-FOLD CV COMPARISON (PLATT CALIBRATED) ===")
    print(f"{'Method':<20} | {'Accuracy':<8} | {'Log Loss':<9} | {'Brier':<8} | {'Macro F1'}")
    print("-" * 65)
    print(f"{'Baseline':<20} | {BASELINE_METRICS['Accuracy']:.4f}   | {BASELINE_METRICS['Log Loss']:.4f}    | {BASELINE_METRICS['Brier']:.4f}   | {BASELINE_METRICS['Macro F1']:.4f}")
    print(f"{'Baseline + h2h_diff':<20} | {acc:.4f}   | {ll:.4f}    | {brier:.4f}   | {f1:.4f}")
    
    print("\n=== INCREMENTAL LIFT ===")
    print(f"Accuracy Diff: {(acc - BASELINE_METRICS['Accuracy']):+.4f}")
    print(f"Log Loss Diff: {(BASELINE_METRICS['Log Loss'] - ll):+.4f} (positive means log loss decreased/improved)")
    print(f"Brier Diff:    {(BASELINE_METRICS['Brier'] - brier):+.4f} (positive means brier decreased/improved)")
    print(f"Macro F1 Diff: {(f1 - BASELINE_METRICS['Macro F1']):+.4f}")
    
    # Gain and SHAP
    print("\n=== IMPORTANCE METRICS FOR h2h_diff ===")
    gain = model.feat_importance.get('h2h_diff', 0.0)
    print(f"Gain Importance: {gain:.4f}")
    
    dmatrix = xgb.DMatrix(X)
    contribs = model.model.get_booster().predict(dmatrix, pred_contribs=True)
    abs_shap = np.abs(contribs[:, :, :-1])
    mean_abs_shap = np.mean(abs_shap, axis=(0, 1))
    shap_importance = dict(zip(FEATURE_COLS, mean_abs_shap))
    h2h_shap = shap_importance.get('h2h_diff', 0.0)
    
    print(f"Mean |SHAP| (Log-Odds Contribution): {h2h_shap:.4f}")
    
    print("\nTop 5 features by Gain:")
    print(model.importance_report().head(5).to_string(index=False))

if __name__ == "__main__":
    main()
