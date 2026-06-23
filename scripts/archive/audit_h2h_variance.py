import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

from core.logger import get_logger
from data.ingestion import MatchDataPipeline
from data.elo_tracker import EloTracker
from data.feature_engineering import build_training_features

log = get_logger("audit_h2h_variance")

def main():
    print("Initializing Pipeline...")
    pipeline = MatchDataPipeline()
    elo = EloTracker()
    elo.batch_update(pipeline.matches)

    print("Building training features (this may take a moment)...")
    X, y = build_training_features(pipeline.training_df, elo)
    
    # h2h_diff is not in FEATURE_COLS, but it exists in the raw training_df
    h2h = pipeline.training_df['h2h_diff']
    
    zero_pct = (h2h == 0).mean() * 100
    
    print("\n=== H2H_DIFF STATISTICAL AUDIT ===")
    print(f"Total Rows: {len(h2h)}")
    print(f"Percentage of rows where h2h_diff == 0: {zero_pct:.2f}%")
    print(f"Min: {h2h.min():.4f}")
    print(f"Max: {h2h.max():.4f}")
    print(f"Mean: {h2h.mean():.4f}")
    print(f"Std: {h2h.std():.4f}")
    print(f"Number of unique values: {h2h.nunique()}")
    
    # Correlation with target
    # Target is categorical: 0 (home win), 1 (draw), 2 (away win)
    # Pearson correlation with categorical isn't perfect, but we can compute it.
    # A negative correlation is expected if high h2h_diff (favors home) corresponds to lower target value (0).
    corr = h2h.corr(y)
    print(f"Pearson Correlation with target (0=Home Win, 1=Draw, 2=Away Win): {corr:.4f}")
    
    print("\nTop 10 most common values:")
    top_values = h2h.value_counts().head(10)
    for val, count in top_values.items():
        print(f"Value: {val:8.4f} | Count: {count:5d} | Pct: {(count/len(h2h))*100:.2f}%")

if __name__ == "__main__":
    main()
