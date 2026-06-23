import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.calibration import CalibratedClassifierCV

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import XGB_PARAMS, TRAINING_START, TEAM_NAME_ALIASES, ALIAS_REVERSE
from core.logger import get_logger
from data.elo_tracker import EloTracker
from data.feature_engineering import FEATURE_COLS, tournament_importance
from data.ingestion import FormTracker, H2HRegistry, build_match_records, load_raw_results
from models.ml_engine import _multiclass_brier_score

log = get_logger("eval_temporal_holdout")

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

def _eval_holdout(df_train: pd.DataFrame, df_test: pd.DataFrame, label: str) -> Dict[str, float]:
    X_tr = df_train[FEATURE_COLS].copy()
    y_tr = df_train["target"].astype(int)
    
    X_te = df_test[FEATURE_COLS].copy()
    y_te = df_test["target"].astype(int)
    
    log.info("[%s] Train: %d rows (2010-2022) | Test: %d rows (2023+)", label, len(df_train), len(df_test))
    
    clf = CalibratedClassifierCV(estimator=xgb.XGBClassifier(**XGB_PARAMS), method='sigmoid', cv=3)
    clf.fit(X_tr, y_tr)
    
    proba = clf.predict_proba(X_te)
    preds = proba.argmax(axis=1)
    
    acc = accuracy_score(y_te, preds)
    ll = log_loss(y_te, proba)
    brier = _multiclass_brier_score(y_te, proba)
    f1 = f1_score(y_te, preds, average="macro")
    
    log.info("[%s] Holdout Summary - acc=%.4f logloss=%.4f brier=%.4f f1=%.4f", label, acc, ll, brier, f1)
    
    return {
        "accuracy": float(acc),
        "log_loss": float(ll),
        "brier_score": float(brier),
        "f1_macro": float(f1),
        "train_rows": len(df_train),
        "test_rows": len(df_test),
    }

def main():
    log.info("Building corpora chronologically to avoid leakage...")
    df_base, df_s1 = _build_holdout_corpora()

    cutoff = pd.Timestamp("2023-01-01")
    
    # Baseline Train: Both teams are WC participants, 2010-2022
    train_base = df_base[df_base["date"] < cutoff]
    
    # Scenario 1 Train: At least one team is WC participant, 2010-2022
    train_s1 = df_s1[df_s1["date"] < cutoff]
    
    # IDENTICAL TEST SET: At least one team is WC participant, 2023+
    test_identical = df_s1[df_s1["date"] >= cutoff]

    base_res = _eval_holdout(train_base, test_identical, "Baseline")
    s1_res = _eval_holdout(train_s1, test_identical, "Scenario 1")

    print("\n=== TEMPORAL HOLDOUT BENCHMARK (Train: 2010-2022 | Test: 2023+) ===")
    
    print("\nMetrics on 2023+ matches:")
    print(f"{'Metric':<15} | {'Baseline':<20} | {'Scenario 1':<20} | {'Delta':<10}")
    print("-" * 75)
    
    acc_d = s1_res["accuracy"] - base_res["accuracy"]
    ll_d = s1_res["log_loss"] - base_res["log_loss"]
    brier_d = s1_res["brier_score"] - base_res["brier_score"]
    f1_d = s1_res["f1_macro"] - base_res["f1_macro"]
    
    print(f"{'Accuracy':<15} | {base_res['accuracy']:.5f}             | {s1_res['accuracy']:.5f}             | {acc_d:+.5f}")
    print(f"{'Log Loss':<15} | {base_res['log_loss']:.5f}             | {s1_res['log_loss']:.5f}             | {ll_d:+.5f}")
    print(f"{'Brier Score':<15} | {base_res['brier_score']:.5f}             | {s1_res['brier_score']:.5f}             | {brier_d:+.5f}")
    print(f"{'Macro F1':<15} | {base_res['f1_macro']:.5f}             | {s1_res['f1_macro']:.5f}             | {f1_d:+.5f}")
    
    print(f"\n{'Train Rows':<15} | {base_res['train_rows']:<20} | {s1_res['train_rows']:<20}")
    print(f"{'Test Rows':<15} | {base_res['test_rows']:<20} | {s1_res['test_rows']:<20}")

if __name__ == "__main__":
    main()
