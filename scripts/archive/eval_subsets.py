import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import XGB_PARAMS, TRAINING_START, TEAM_NAME_ALIASES, ALIAS_REVERSE
from core.logger import get_logger
from data.elo_tracker import EloTracker
from data.feature_engineering import FEATURE_COLS, tournament_importance
from data.ingestion import FormTracker, H2HRegistry, build_match_records, load_raw_results
from models.ml_engine import _multiclass_brier_score

log = get_logger("eval_subsets")

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

            # Approximate rank by sorting current elo ratings
            sorted_ratings = sorted(replay_elo.ratings.values(), reverse=True)
            h_rank = sorted_ratings.index(rh) + 1 if rh in sorted_ratings else 200
            a_rank = sorted_ratings.index(ra) + 1 if ra in sorted_ratings else 200

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
                "h_rank": h_rank,
                "a_rank": a_rank
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
    if len(y_true) == 0: return 0.0
    ece = 0.0
    for class_idx in range(y_prob.shape[1]):
        prob_true, prob_pred = calibration_curve((y_true == class_idx).astype(int), y_prob[:, class_idx], n_bins=n_bins, strategy='uniform')
        bin_counts, _ = np.histogram(y_prob[:, class_idx], bins=n_bins, range=(0, 1))
        bin_weights = bin_counts[bin_counts > 0] / len(y_prob)
        ece += np.sum(bin_weights * np.abs(prob_true - prob_pred))
    return ece / y_prob.shape[1]

def evaluate_subset(df_train: pd.DataFrame, df_test: pd.DataFrame, label: str):
    X_tr = df_train[FEATURE_COLS].copy()
    y_tr = df_train["target"].astype(int)
    
    clf = CalibratedClassifierCV(estimator=xgb.XGBClassifier(**XGB_PARAMS), method='sigmoid', cv=3)
    clf.fit(X_tr, y_tr)
    
    subsets = {
        "ALL": df_test,
        "Top-20 vs Top-20": df_test[(df_test["h_rank"] <= 20) & (df_test["a_rank"] <= 20)],
        "Top-20 vs Bottom-100": df_test[((df_test["h_rank"] <= 20) & (df_test["a_rank"] >= 100)) | ((df_test["a_rank"] <= 20) & (df_test["h_rank"] >= 100))],
        "Neutral-Site": df_test[df_test["neutral"] == True],
        "Tournament Only": df_test[df_test["tournament"] != "Friendly"],
    }

    res = {}
    for sub_name, sub_df in subsets.items():
        if len(sub_df) == 0:
            res[sub_name] = {"acc": np.nan, "ll": np.nan, "ece": np.nan, "n": 0}
            continue
            
        X_te = sub_df[FEATURE_COLS].copy()
        y_te = sub_df["target"].astype(int)
        
        proba = clf.predict_proba(X_te)
        preds = proba.argmax(axis=1)
        
        acc = accuracy_score(y_te, preds)
        ll = log_loss(y_te, proba, labels=[0,1,2])
        ece = calculate_ece(y_te.values, proba)
        
        res[sub_name] = {"acc": acc, "ll": ll, "ece": ece, "n": len(sub_df)}
        
    return res

def main():
    log.info("Building corpora chronologically to avoid leakage...")
    df_base, df_s1 = _build_holdout_corpora()

    cutoff = pd.Timestamp("2023-01-01")
    train_base = df_base[df_base["date"] < cutoff]
    train_s1 = df_s1[df_s1["date"] < cutoff]
    
    # We use identical test set!
    test_identical = df_s1[df_s1["date"] >= cutoff]

    log.info("Evaluating Baseline model on subsets...")
    base_metrics = evaluate_subset(train_base, test_identical, "Baseline")
    
    log.info("Evaluating Scenario 1 model on subsets...")
    s1_metrics = evaluate_subset(train_s1, test_identical, "Scenario 1")

    print("\n=== SUBSET PERFORMANCE AUDIT (Test: Identical 2023+ matches) ===")
    print(f"{'Subset':<22} | {'N':<5} | {'Base LL':<9} | {'S1 LL':<9} | {'Base ECE':<9} | {'S1 ECE':<9} | {'Base Acc':<9} | {'S1 Acc':<9}")
    print("-" * 105)
    
    for sub_name in base_metrics.keys():
        b = base_metrics[sub_name]
        s = s1_metrics[sub_name]
        print(f"{sub_name:<22} | {b['n']:<5} | {b['ll']:.5f}   | {s['ll']:.5f}   | {b['ece']:.5f}   | {s['ece']:.5f}   | {b['acc']:.5f}   | {s['acc']:.5f}")
        
if __name__ == "__main__":
    main()
