import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import XGB_PARAMS, TRAINING_START, TEAM_NAME_ALIASES, ALIAS_REVERSE
from core.logger import get_logger
from data.elo_tracker import EloTracker
from data.feature_engineering import FEATURE_COLS, tournament_importance
from data.ingestion import FormTracker, H2HRegistry, build_match_records, load_raw_results
from models.ml_engine import _multiclass_brier_score

log = get_logger("eval_calibration")

def _build_holdout_corpora() -> Tuple[pd.DataFrame, pd.DataFrame]:
    training_start = pd.Timestamp(TRAINING_START)
    wc_csv_names = set(TEAM_NAME_ALIASES.values())

    raw = load_raw_results()
    all_matches = build_match_records(raw)
    
    raw_clean = raw.dropna(subset=["home_score", "away_score"]).copy()
    raw_clean["home_score"] = raw_clean["home_score"].astype(int)
    raw_clean["away_score"] = raw_clean["away_score"].astype(int)
    raw_clean = raw_clean.sort_values("date").reset_index(drop=True)

    all_matches = all_matches.copy()
    all_matches["home_score"] = raw_clean["home_score"].values
    all_matches["away_score"] = raw_clean["away_score"].values

    form_tracker = FormTracker(window=10)
    h2h = H2HRegistry()
    opp_history: Dict[str, List[str]] = {}
    
    base_rows: List[Dict] = []
    s1_rows: List[Dict] = []

    pre_training_hist = all_matches[all_matches["date"] < training_start]
    replay_elo = EloTracker()
    replay_elo.batch_update(pre_training_hist)

    for _, row in all_matches.iterrows():
        h = str(row["home_team"])
        a = str(row["away_team"])
        hs = int(row["home_score"])
        as_ = int(row["away_score"])
        date = row["date"]
        tournament = str(row["tournament"])
        neutral = bool(row["neutral"])

        if date >= training_start:
            h_can = ALIAS_REVERSE.get(h, h)
            a_can = ALIAS_REVERSE.get(a, a)

            rh = replay_elo._get(h)
            ra = replay_elo._get(a)
            elo_diff = rh - ra

            h_w, h_d, h_l, h_gd = form_tracker.get_form(h)
            a_w, a_d, a_l, a_gd = form_tracker.get_form(a)
            h_form = form_tracker.get_form_score(h)
            a_form = form_tracker.get_form_score(a)

            recent_opp_h = opp_history.get(h, [])[-form_tracker.window:]
            recent_opp_a = opp_history.get(a, [])[-form_tracker.window:]

            def _opp_adj(own_form, opponents):
                if not opponents:
                    return 0.0
                opp_mean = sum(form_tracker.get_form_score(o) for o in opponents) / len(opponents)
                return own_form + opp_mean - 1.0

            opp_adj_h = _opp_adj(h_form, recent_opp_h)
            opp_adj_a = _opp_adj(a_form, recent_opp_a)

            record = {
                "date": date,
                "home_team": h_can,
                "away_team": a_can,
                "tournament": tournament,
                "neutral": neutral,
                "target": int(row["target"]),
                "elo_diff": elo_diff,
                "h2h_diff": h2h.win_rate_diff(h, a),
                "home_form": h_form,
                "away_form": a_form,
                "opp_adj_form_diff": opp_adj_h - opp_adj_a,
                "gd_diff": h_gd - a_gd,
                "home_win_rate": h_w,
                "away_win_rate": a_w,
                "home_draw_rate": h_d,
                "away_draw_rate": a_d,
                "home_loss_rate": h_l,
            }

            if h in wc_csv_names or a in wc_csv_names:
                s1_rows.append(record)

            if h in wc_csv_names and a in wc_csv_names:
                base_rows.append(record)

        form_tracker.update(h, a, hs, as_)
        h2h.update(h, a, hs, as_)
        opp_history.setdefault(h, []).append(a)
        opp_history.setdefault(a, []).append(h)
        replay_elo.update(h, a, int(row["target"]), tournament, neutral)

    def _to_df(rows: List[Dict]) -> pd.DataFrame:
        tdf = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
        tdf["form_diff"] = tdf["home_form"].astype(float) - tdf["away_form"].astype(float)
        tdf["match_type_enc"] = tdf["tournament"].map(tournament_importance).fillna(0.40)
        tdf["neutral_venue"] = tdf["neutral"].astype(float)
        return tdf

    df_base = _to_df(base_rows)
    df_s1 = _to_df(s1_rows)

    return df_base, df_s1

def calculate_ece(y_true, y_prob, n_bins=10):
    ece = 0.0
    for class_idx in range(y_prob.shape[1]):
        prob_true, prob_pred = calibration_curve((y_true == class_idx).astype(int), y_prob[:, class_idx], n_bins=n_bins, strategy='uniform')
        # We need the fraction of samples in each bin
        bin_counts, _ = np.histogram(y_prob[:, class_idx], bins=n_bins, range=(0, 1))
        bin_weights = bin_counts[bin_counts > 0] / len(y_prob)
        ece += np.sum(bin_weights * np.abs(prob_true - prob_pred))
    return ece / y_prob.shape[1] # Macro ECE

def _eval_calibration(df_train: pd.DataFrame, df_test: pd.DataFrame, label: str) -> Dict:
    X_tr = df_train[FEATURE_COLS].copy()
    y_tr = df_train["target"].astype(int)
    
    X_te = df_test[FEATURE_COLS].copy()
    y_te = df_test["target"].astype(int)
    
    # 1. Feature Importance (Single uncalibrated model for SHAP/Gain)
    base_clf = xgb.XGBClassifier(**XGB_PARAMS)
    base_clf.fit(X_tr, y_tr)
    gain_importance = dict(zip(FEATURE_COLS, base_clf.feature_importances_))
    
    explainer = shap.TreeExplainer(base_clf)
    shap_values = explainer.shap_values(X_tr)
    if isinstance(shap_values, list): # For multiclass
        shap_abs = np.zeros(X_tr.shape[1])
        for sv in shap_values:
            shap_abs += np.abs(sv).mean(axis=0)
        shap_importance = dict(zip(FEATURE_COLS, shap_abs / len(shap_values)))
    else:
        # Some XGBoost multiclass shap_values returns shape (n_samples, n_features, n_classes)
        if len(shap_values.shape) == 3:
            shap_abs = np.abs(shap_values).mean(axis=(0, 2))
            shap_importance = dict(zip(FEATURE_COLS, shap_abs))
        else:
            shap_abs = np.abs(shap_values).mean(axis=0)
            shap_importance = dict(zip(FEATURE_COLS, shap_abs))

    # 2. Calibration / Prediction (Calibrated model)
    clf = CalibratedClassifierCV(estimator=xgb.XGBClassifier(**XGB_PARAMS), method='sigmoid', cv=3)
    clf.fit(X_tr, y_tr)
    
    proba = clf.predict_proba(X_te)
    ece = calculate_ece(y_te.values, proba, n_bins=10)
    
    # Save calibration curves for Win class (target=0)
    prob_true, prob_pred = calibration_curve((y_te == 0).astype(int), proba[:, 0], n_bins=10, strategy='uniform')
    
    return {
        "gain": gain_importance,
        "shap": shap_importance,
        "ece": ece,
        "prob_true": prob_true,
        "prob_pred": prob_pred
    }

def main():
    log.info("Building corpora chronologically to avoid leakage...")
    df_base, df_s1 = _build_holdout_corpora()

    cutoff = pd.Timestamp("2023-01-01")
    train_base = df_base[df_base["date"] < cutoff]
    train_s1 = df_s1[df_s1["date"] < cutoff]
    test_identical = df_s1[df_s1["date"] >= cutoff]

    log.info("Evaluating Baseline...")
    base_res = _eval_calibration(train_base, test_identical, "Baseline")
    
    log.info("Evaluating Scenario 1...")
    s1_res = _eval_calibration(train_s1, test_identical, "Scenario 1")

    print("\n=== CALIBRATION & FEATURE IMPORTANCE BENCHMARK ===")
    
    print("\n1. Expected Calibration Error (ECE)")
    print(f"   Baseline:   {base_res['ece']:.5f}")
    print(f"   Scenario 1: {s1_res['ece']:.5f}")
    
    print("\n2. Feature Importance (SHAP absolute mean)")
    keys = ["elo_diff", "h2h_diff", "form_diff", "opp_adj_form_diff", "gd_diff"]
    print(f"{'Feature':<20} | {'Baseline SHAP':<15} | {'Scenario 1 SHAP':<15} | {'Baseline Gain':<15} | {'Scenario 1 Gain':<15}")
    print("-" * 90)
    for k in keys:
        bs = base_res['shap'].get(k, 0)
        s1s = s1_res['shap'].get(k, 0)
        bg = base_res['gain'].get(k, 0)
        s1g = s1_res['gain'].get(k, 0)
        print(f"{k:<20} | {bs:.5f}         | {s1s:.5f}         | {bg:.5f}         | {s1g:.5f}")
        
    # Generate Reliability Diagram
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.plot(base_res["prob_pred"], base_res["prob_true"], "s-", label=f"Baseline (ECE={base_res['ece']:.3f})")
    plt.plot(s1_res["prob_pred"], s1_res["prob_true"], "s-", label=f"Scenario 1 (ECE={s1_res['ece']:.3f})")
    plt.xlabel("Mean Predicted Probability (Home Win)")
    plt.ylabel("Fraction of Positives")
    plt.title("Reliability Diagram (Home Win - 2023+ matches)")
    plt.legend(loc="lower right")
    plt.grid(True)
    out_path = str(ROOT / "reports" / "reliability_diagram.png")
    plt.savefig(out_path)
    print(f"\nReliability diagram saved to: {out_path}")

if __name__ == "__main__":
    main()
