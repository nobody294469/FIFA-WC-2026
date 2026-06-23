import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, log_loss

from core.config import XGB_PARAMS, MODEL_CV_FOLDS
from data.ingestion import MatchDataPipeline
from data.feature_engineering import build_training_features

def _multiclass_brier_score(y_true: pd.Series, proba: np.ndarray) -> float:
    y_idx = np.asarray(y_true, dtype=np.int64)
    y_onehot = np.eye(proba.shape[1], dtype=np.float64)[y_idx]
    return float(np.mean(np.sum((proba - y_onehot) ** 2, axis=1)))

print("Loading data...")
pipeline = MatchDataPipeline()
from data.elo_tracker import EloTracker
elo = EloTracker()
elo.batch_update(pipeline.matches)
X, y = build_training_features(pipeline.training_df, elo)

tscv = TimeSeriesSplit(n_splits=MODEL_CV_FOLDS)

oof_y = []
oof_raw = []
oof_platt = []
oof_iso = []

print(f"Starting {MODEL_CV_FOLDS}-fold Nested CV evaluation...")
for fold, (tr_idx, val_idx) in enumerate(tscv.split(X), 1):
    print(f"  Fold {fold}...")
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
    
    # 1. Raw
    clf_raw = xgb.XGBClassifier(**XGB_PARAMS)
    clf_raw.fit(X_tr, y_tr, verbose=False)
    prob_raw = clf_raw.predict_proba(X_val)
    
    # 2. Platt (Sigmoid)
    clf_platt = CalibratedClassifierCV(estimator=xgb.XGBClassifier(**XGB_PARAMS), method='sigmoid', cv=3)
    clf_platt.fit(X_tr, y_tr)
    prob_platt = clf_platt.predict_proba(X_val)
    
    # 3. Isotonic
    clf_iso = CalibratedClassifierCV(estimator=xgb.XGBClassifier(**XGB_PARAMS), method='isotonic', cv=3)
    clf_iso.fit(X_tr, y_tr)
    prob_iso = clf_iso.predict_proba(X_val)
    
    oof_y.extend(y_val)
    oof_raw.append(prob_raw)
    oof_platt.append(prob_platt)
    oof_iso.append(prob_iso)

print("Computing Metrics...")
oof_y = np.array(oof_y)
oof_raw = np.vstack(oof_raw)
oof_platt = np.vstack(oof_platt)
oof_iso = np.vstack(oof_iso)

def get_metrics(probs):
    preds = probs.argmax(axis=1)
    acc = accuracy_score(oof_y, preds)
    ll = log_loss(oof_y, probs)
    brier = _multiclass_brier_score(oof_y, probs)
    f1 = f1_score(oof_y, preds, average="macro")
    return acc, ll, brier, f1

metrics_raw = get_metrics(oof_raw)
metrics_platt = get_metrics(oof_platt)
metrics_iso = get_metrics(oof_iso)

print("\n=== EVALUATION RESULTS (OOF) ===")
print(f"{'Method':<12} | {'Accuracy':<8} | {'Log Loss':<9} | {'Brier':<8} | {'Macro F1'}")
print("-" * 55)
print(f"{'Raw XGBoost':<12} | {metrics_raw[0]:.4f}   | {metrics_raw[1]:.4f}    | {metrics_raw[2]:.4f}   | {metrics_raw[3]:.4f}")
print(f"{'Platt':<12} | {metrics_platt[0]:.4f}   | {metrics_platt[1]:.4f}    | {metrics_platt[2]:.4f}   | {metrics_platt[3]:.4f}")
print(f"{'Isotonic':<12} | {metrics_iso[0]:.4f}   | {metrics_iso[1]:.4f}    | {metrics_iso[2]:.4f}   | {metrics_iso[3]:.4f}")

# Relative improvements
def rel_imp(base, new):
    return (base - new) / base * 100

print("\n=== RELATIVE IMPROVEMENTS (vs Raw) ===")
print(f"Platt Log Loss Improvement:    {rel_imp(metrics_raw[1], metrics_platt[1]):+.2f}%")
print(f"Platt Brier Improvement:       {rel_imp(metrics_raw[2], metrics_platt[2]):+.2f}%")
print(f"Isotonic Log Loss Improvement: {rel_imp(metrics_raw[1], metrics_iso[1]):+.2f}%")
print(f"Isotonic Brier Improvement:    {rel_imp(metrics_raw[2], metrics_iso[2]):+.2f}%")

# Generate Reliability Diagrams
print("\nGenerating Reliability Diagrams...")
os.makedirs('reports/calibration', exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
classes = ['Home Win', 'Draw', 'Away Win']
methods = [('Raw', oof_raw), ('Platt', oof_platt), ('Isotonic', oof_iso)]
colors = ['blue', 'orange', 'green']

for c_idx, ax in enumerate(axes):
    ax.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
    for m_idx, (name, probs) in enumerate(methods):
        # We binarize the labels for the specific class
        y_true_bin = (oof_y == c_idx).astype(int)
        prob_pos = probs[:, c_idx]
        fraction_of_positives, mean_predicted_value = calibration_curve(y_true_bin, prob_pos, n_bins=10)
        ax.plot(mean_predicted_value, fraction_of_positives, 's-', label=name, color=colors[m_idx])
    
    ax.set_title(f'Reliability Diagram: {classes[c_idx]}')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.legend(loc='lower right')

plt.tight_layout()
plt.savefig('reports/calibration/reliability_diagrams.png')
print("Saved plots to reports/calibration/reliability_diagrams.png")
