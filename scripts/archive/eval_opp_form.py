import numpy as np
import pandas as pd
import shap
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
from data.feature_engineering import build_training_features, FEATURE_COLS
from data.elo_tracker import EloTracker
from models.ml_engine import MatchOutcomeModel
from data.ingestion import MatchDataPipeline
from core.config import RESULTS_CSV
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("eval_opp_form")

def run_evaluation():
    log.info("Running MatchDataPipeline...")
    p = MatchDataPipeline()
    training_df = p.training_df
    
    log.info("Loading training features...")
    elo = EloTracker()
    df, y = build_training_features(training_df, elo)
    df['target'] = y
    
    tdf = training_df.sort_values("date").reset_index(drop=True)
    home_own = df['home_form']
    home_opp_mean = tdf['home_opp_mean']
    away_own = df['away_form']
    away_opp_mean = tdf['away_opp_mean']

    print(f"\nStats for own_form (Home): Mean={home_own.mean():.4f}, Min={home_own.min():.4f}, Max={home_own.max():.4f}")
    print(f"Stats for opp_mean (Home): Mean={home_opp_mean.mean():.4f}, Min={home_opp_mean.min():.4f}, Max={home_opp_mean.max():.4f}")

    formulations = {
        "Current (inverted)": lambda own, opp: own - opp,
        "Proposed (additive)": lambda own, opp: own + opp - 1.0,
        "Multiplicative": lambda own, opp: own * opp,
        "Relative Overperformance": lambda own, opp: own / np.maximum(opp, 0.01),
        "Standardized Residual": lambda own, opp: (own - (1.0 - opp)) / np.std(own - (1.0 - opp)) if np.std(own - (1.0 - opp)) > 0 else (own - (1.0 - opp))
    }

    results = []

    for name, func in formulations.items():
        log.info(f"Evaluating: {name}")
        
        home_adj = func(home_own, home_opp_mean)
        away_adj = func(away_own, away_opp_mean)
        
        df['opp_adj_form_diff'] = home_adj - away_adj

        diff_mean = df['opp_adj_form_diff'].mean()
        diff_std = df['opp_adj_form_diff'].std()
        
        home_points = np.where(df['target'] == 2, 1.0, np.where(df['target'] == 1, 0.5, 0.0))
        corr = np.corrcoef(df['opp_adj_form_diff'], home_points)[0, 1]
            
        from sklearn.model_selection import TimeSeriesSplit
        
        X = df[FEATURE_COLS]
        target = df['target']
        tscv = TimeSeriesSplit(n_splits=5)
        log_losses, briers, accs = [], [], []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, y_train = X.iloc[train_idx], target.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], target.iloc[val_idx]
            
            model = MatchOutcomeModel()
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_val)
            preds = np.argmax(probs, axis=1)
            
            # Outcome probabilities true matrix
            y_true_mat = np.zeros((len(y_val), 3))
            y_true_mat[np.arange(len(y_val)), y_val] = 1.0
            
            log_losses.append(log_loss(y_val, probs))
            briers.append(brier_score_loss(y_true_mat.ravel(), probs.ravel()))
            accs.append(accuracy_score(y_val, preds))
            
        metrics = {
            'log_loss': np.mean(log_losses),
            'brier_score': np.mean(briers),
            'accuracy': np.mean(accs)
        }
        
        from xgboost import XGBClassifier
        xgb = XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=3,
            random_state=42
        )
        xgb.fit(X, target)
        
        feat_idx = list(X.columns).index('opp_adj_form_diff')
        importance = xgb.feature_importances_[feat_idx]
        
        results.append({
            "Formulation": name,
            "Mean": diff_mean,
            "Std": diff_std,
            "Correlation": corr,
            "Log Loss": metrics['log_loss'],
            "Brier Score": metrics['brier_score'],
            "Accuracy": metrics['accuracy'],
            "Feat_Imp": importance
        })

    res_df = pd.DataFrame(results)
    print("\n" + "="*90)
    print("Ablation Study Results")
    print("="*90)
    print(res_df.to_string(index=False))

if __name__ == "__main__":
    run_evaluation()
