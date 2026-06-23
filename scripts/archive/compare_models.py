"""
compare_models.py - Compares XGBoost, LightGBM, and CatBoost
using exact same leakage-free features and time-aware CV.
"""
from __future__ import annotations

import sys
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss, f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

from core.config import XGB_PARAMS, XGB_EARLY_STOPPING_ROUNDS, MODEL_CV_FOLDS, TRAINING_START
from data.ingestion import MatchDataPipeline
from data.elo_tracker import EloTracker
from data.feature_engineering import build_training_features, FEATURE_COLS

def _multiclass_brier_score(y_true: pd.Series, y_prob: np.ndarray) -> float:
    # 0, 1, 2
    n_samples = len(y_true)
    y_true_onehot = np.zeros((n_samples, 3))
    y_true_onehot[np.arange(n_samples), y_true.values] = 1.0
    return float(np.mean(np.sum((y_prob - y_true_onehot) ** 2, axis=1)))

def evaluate_model(model_name: str, get_model_fn, X, y):
    print(f"\n--- Evaluating {model_name} ---")
    tscv = TimeSeriesSplit(n_splits=MODEL_CV_FOLDS)
    fold_metrics = []
    
    t0 = time.perf_counter()
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        
        clf = get_model_fn()
        
        if model_name == "XGBoost":
            clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            proba = clf.predict_proba(X_val)
        elif model_name == "LightGBM":
            # lgb early stopping via callbacks in fit
            clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], 
                    callbacks=[lgb.early_stopping(XGB_EARLY_STOPPING_ROUNDS, verbose=False)])
            proba = clf.predict_proba(X_val)
        elif model_name == "CatBoost":
            clf.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False, early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS)
            proba = clf.predict_proba(X_val)
            
        preds = proba.argmax(axis=1)
        
        acc = accuracy_score(y_val, preds)
        ll = log_loss(y_val, proba)
        brier = _multiclass_brier_score(y_val, proba)
        f1 = f1_score(y_val, preds, average="macro")
        
        fold_metrics.append({
            "accuracy": acc,
            "logloss": ll,
            "brier": brier,
            "f1_macro": f1
        })
        print(f"  Fold {fold}: acc={acc:.4f}, logloss={ll:.4f}, brier={brier:.4f}")
        
    elapsed = time.perf_counter() - t0
    
    df = pd.DataFrame(fold_metrics)
    summary = {
        "model": model_name,
        "mean_acc": df["accuracy"].mean(),
        "std_acc": df["accuracy"].std(),
        "mean_logloss": df["logloss"].mean(),
        "mean_brier": df["brier"].mean(),
        "mean_f1": df["f1_macro"].mean(),
        "time_s": elapsed
    }
    print(f"  -> Mean Acc: {summary['mean_acc']:.4f} | LogLoss: {summary['mean_logloss']:.4f} | Brier: {summary['mean_brier']:.4f}")
    return summary

def main():
    print("Building dataset...")
    pipeline = MatchDataPipeline()
    elo = EloTracker()
    
    elo.batch_update(pipeline.matches)
    
    X_full, y_full = build_training_features(
        training_df=pipeline.training_df,
        elo_tracker=elo,
    )
    
    X = X_full[FEATURE_COLS]
    y = y_full
    
    print(f"Dataset shape: {X.shape}")
    
    def get_xgb():
        params = dict(XGB_PARAMS)
        params["early_stopping_rounds"] = XGB_EARLY_STOPPING_ROUNDS
        return xgb.XGBClassifier(**params)
        
    def get_lgbm():
        # equivalent params
        return lgb.LGBMClassifier(
            n_estimators=800,
            max_depth=2,
            learning_rate=0.1,
            subsample=0.6,
            colsample_bytree=0.9,
            min_child_samples=10,
            objective="multiclass",
            num_class=3,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
        
    def get_catboost():
        return CatBoostClassifier(
            iterations=800,
            depth=2,
            learning_rate=0.1,
            bootstrap_type="Bernoulli",
            subsample=0.6,
            rsm=0.9,
            min_data_in_leaf=10,
            loss_function="MultiClass",
            classes_count=3,
            random_seed=42,
            thread_count=-1,
            verbose=False,
            allow_writing_files=False
        )
        
    results = []
    results.append(evaluate_model("XGBoost", get_xgb, X, y))
    results.append(evaluate_model("LightGBM", get_lgbm, X, y))
    results.append(evaluate_model("CatBoost", get_catboost, X, y))
    
    print("\n" + "="*80)
    print(" MODEL COMPARISON SUMMARY")
    print("="*80)
    df_res = pd.DataFrame(results)
    df_res = df_res[["model", "mean_acc", "std_acc", "mean_logloss", "mean_brier", "mean_f1", "time_s"]]
    print(df_res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("="*80)

if __name__ == "__main__":
    main()
