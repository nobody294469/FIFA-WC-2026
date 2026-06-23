import os
import shap
import matplotlib.pyplot as plt
import pandas as pd

from models.ml_engine import MatchOutcomeModel
from data.ingestion import MatchDataPipeline
from data.feature_engineering import build_training_features, FEATURE_COLS
from data.elo_tracker import EloTracker
from core.config import RESULTS_CSV

def main():
    print("Loading data...")
    pipeline = MatchDataPipeline()
    training_df = pipeline.training_df

    print("Building training features...")
    elo = EloTracker()
    X_full, y_full = build_training_features(training_df, elo)
    X = X_full[FEATURE_COLS]

    print("Loading model...")
    model = MatchOutcomeModel()
    model.load()

    print("Generating SHAP summary plot...")
    explainer = model._explainer
    shap_values = explainer(X)

    # SHAP for multi-class returns an array of shape (N, num_features, num_classes)
    # We plot the mean absolute SHAP value for each feature
    # The summary plot handles multi-class natively if passed appropriately.
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig("reports/shap_summary_bar.png", dpi=300)
    plt.close()

    # For individual class (e.g., class 0 - Home Win), we can plot the beeswarm
    plt.figure(figsize=(10, 8))
    # Check if shap_values is a list (old shap) or Explanation object (new shap)
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[0], X, show=False)
    else:
        # For Explanation object
        shap.summary_plot(shap_values[:, :, 0], X, show=False)
    plt.tight_layout()
    plt.savefig("reports/shap_summary_beeswarm_home_win.png", dpi=300)
    plt.close()
    
    print("Saved SHAP plots to reports/")

if __name__ == "__main__":
    main()
