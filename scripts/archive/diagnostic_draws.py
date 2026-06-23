import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import XGB_PARAMS
from data.feature_engineering import FEATURE_COLS
from scripts.eval_draw_adj import _build_holdout_corpora

def main():
    print("Building Scenario 1 dataset...")
    df = _build_holdout_corpora()
    
    cutoff = pd.Timestamp("2023-01-01")
    train = df[df["date"] < cutoff]
    
    print("\n=== BASE RATES IN TRAINING ===")
    print(f"Total training matches: {len(train)}")
    print(f"Global draw rate: {train['target'].value_counts(normalize=True).get(1, 0):.2%}")
    
    # Define "elite vs elite" loosely as high-ranked vs high-ranked
    # Using h_rank <= 20 and a_rank <= 20
    elite_vs_elite = train[(train["h_rank"] <= 20) & (train["a_rank"] <= 20)]
    print(f"\nElite vs Elite matches in training: {len(elite_vs_elite)}")
    print(f"Elite vs Elite draw rate: {elite_vs_elite['target'].value_counts(normalize=True).get(1, 0):.2%}")
    
    # Define "elite vs minnow" 
    elite_vs_minnow = train[((train["h_rank"] <= 20) & (train["a_rank"] > 100)) | 
                            ((train["h_rank"] > 100) & (train["a_rank"] <= 20))]
    print(f"\nElite vs Minnow matches in training: {len(elite_vs_minnow)}")
    print(f"Elite vs Minnow draw rate: {elite_vs_minnow['target'].value_counts(normalize=True).get(1, 0):.2%}")
    
    # Small Elo Diff (< 100)
    close_matches = train[train["elo_diff"].abs() < 100]
    print(f"\nClose matches (Elo Diff < 100) in training: {len(close_matches)}")
    print(f"Close matches draw rate: {close_matches['target'].value_counts(normalize=True).get(1, 0):.2%}")
    
    print("\n=== XGBoost NATIVE vs CALIBRATED PROBABILITIES ===")
    X_tr = train[FEATURE_COLS]
    y_tr = train["target"].astype(int)
    
    # 1. Native XGB
    xgb_model = xgb.XGBClassifier(**XGB_PARAMS)
    xgb_model.fit(X_tr, y_tr)
    native_proba = xgb_model.predict_proba(X_tr)
    
    # 2. Calibrated XGB
    cal_model = CalibratedClassifierCV(estimator=xgb.XGBClassifier(**XGB_PARAMS), method='sigmoid', cv=3)
    cal_model.fit(X_tr, y_tr)
    cal_proba = cal_model.predict_proba(X_tr)
    
    # Check predictions on Elite vs Elite
    elite_indices = elite_vs_elite.index
    native_draw_elite = native_proba[elite_indices, 1].mean()
    cal_draw_elite = cal_proba[elite_indices, 1].mean()
    
    print(f"Native XGB Average Predicted Draw Prob (Elite vs Elite): {native_draw_elite:.2%}")
    print(f"Calibrated Average Predicted Draw Prob (Elite vs Elite): {cal_draw_elite:.2%}")
    
    # Global vs Close
    close_indices = close_matches.index
    print(f"\nNative XGB Avg Predicted Draw Prob (Elo Diff < 100): {native_proba[close_indices, 1].mean():.2%}")
    print(f"Calibrated Avg Predicted Draw Prob (Elo Diff < 100): {cal_proba[close_indices, 1].mean():.2%}")

if __name__ == "__main__":
    main()
