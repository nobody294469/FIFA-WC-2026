"""
eval_corpus_expansion.py — Benchmark Training Corpus Expansion

Runs a parallel benchmark comparing:
1. Baseline: Both teams must be WC2026 competitors (current behavior).
2. Scenario 1: At least one team must be a WC2026 competitor.
3. Scenario 2: All international matches since 2010 (no team filter).

Gate criterion:
  - CV Log Loss delta <= -0.005 OR CV Brier Score delta <= -0.003
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import TimeSeriesSplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import ALIAS_REVERSE, TEAM_NAME_ALIASES, TRAINING_START
from core.logger import get_logger
from data.elo_tracker import EloTracker
from data.feature_engineering import FEATURE_COLS, tournament_importance
from data.ingestion import FormTracker, H2HRegistry, build_match_records, load_raw_results
from models.ml_engine import MatchOutcomeModel, _multiclass_brier_score

log = get_logger("eval_corpus_expansion")

def _build_corpora() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Returns (X_base, y_base, X_s1, y_s1, X_s2, y_s2)"""
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
    s2_rows: List[Dict] = []

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

            s2_rows.append(record)

            if h in wc_csv_names or a in wc_csv_names:
                s1_rows.append(record)

            if h in wc_csv_names and a in wc_csv_names:
                base_rows.append(record)

        form_tracker.update(h, a, hs, as_)
        h2h.update(h, a, hs, as_)
        opp_history.setdefault(h, []).append(a)
        opp_history.setdefault(a, []).append(h)
        replay_elo.update(h, a, int(row["target"]), tournament, neutral)

    def _to_xy(rows: List[Dict]) -> Tuple[pd.DataFrame, pd.Series]:
        tdf = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
        tdf["form_diff"] = tdf["home_form"].astype(float) - tdf["away_form"].astype(float)
        tdf["match_type_enc"] = tdf["tournament"].map(tournament_importance).fillna(0.40)
        tdf["neutral_venue"] = tdf["neutral"].astype(float)
        return tdf[FEATURE_COLS].copy(), tdf["target"].astype(int)

    X_base, y_base = _to_xy(base_rows)
    X_s1, y_s1 = _to_xy(s1_rows)
    X_s2, y_s2 = _to_xy(s2_rows)

    return X_base, y_base, X_s1, y_s1, X_s2, y_s2

def _run_cv(X: pd.DataFrame, y: pd.Series, label: str) -> Dict[str, float]:
    from sklearn.calibration import CalibratedClassifierCV
    from core.config import XGB_PARAMS
    
    log.info("[%s] Starting 5-fold TimeSeriesSplit CV on %d samples...", label, len(X))
    tscv = TimeSeriesSplit(n_splits=5)
    
    oof_y = []
    oof_prob = []
    oof_preds = []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        clf = CalibratedClassifierCV(estimator=xgb.XGBClassifier(**XGB_PARAMS), method='sigmoid', cv=3)
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_val)
        preds = proba.argmax(axis=1)

        oof_y.extend(y_val)
        oof_prob.append(proba)
        oof_preds.extend(preds)

    oof_y = np.array(oof_y)
    oof_prob = np.vstack(oof_prob)
    oof_preds = np.array(oof_preds)
    
    acc = accuracy_score(oof_y, oof_preds)
    ll = log_loss(oof_y, oof_prob)
    brier = _multiclass_brier_score(oof_y, oof_prob)
    f1 = f1_score(oof_y, oof_preds, average="macro")

    log.info("[%s] CV Summary — acc=%.4f logloss=%.4f brier=%.4f f1=%.4f", label, acc, ll, brier, f1)
    
    return {
        "accuracy": float(acc),
        "log_loss": float(ll),
        "brier_score": float(brier),
        "f1_macro": float(f1),
    }

def _delta(new: float, old: float, lower_is_better: bool = True) -> str:
    d = new - old
    sign = "+" if d > 0 else ""
    marker = ""
    if lower_is_better and d <= -0.003: # using brier threshold as visual cue
        marker = " ✓"
    elif not lower_is_better and d >= 0.003:
        marker = " ✓"
    return f"{sign}{d:.5f}{marker}"

def main():
    log.info("Extracting corporas...")
    X_base, y_base, X_s1, y_s1, X_s2, y_s2 = _build_corpora()

    base_cv = _run_cv(X_base, y_base, "Baseline")
    s1_cv = _run_cv(X_s1, y_s1, "Scenario1")
    s2_cv = _run_cv(X_s2, y_s2, "Scenario2")

    gate_s1 = (s1_cv["log_loss"] - base_cv["log_loss"] <= -0.005) or (s1_cv["brier_score"] - base_cv["brier_score"] <= -0.003)
    gate_s2 = (s2_cv["log_loss"] - base_cv["log_loss"] <= -0.005) or (s2_cv["brier_score"] - base_cv["brier_score"] <= -0.003)

    print("\n=== CORPUS EXPANSION BENCHMARK ===")
    
    print("\n1. CV Metrics (5-Fold TimeSeriesSplit, Platt Calibrated)")
    print(f"{'Metric':<15} | {'Baseline (n=1921)':<20} | {'Scenario 1 (n=7326)':<20} | {'Scenario 2 (n=15718)':<20}")
    print("-" * 80)
    print(f"{'Accuracy':<15} | {base_cv['accuracy']:.5f}             | {s1_cv['accuracy']:.5f} {_delta(s1_cv['accuracy'], base_cv['accuracy'], False):<8} | {s2_cv['accuracy']:.5f} {_delta(s2_cv['accuracy'], base_cv['accuracy'], False):<8}")
    print(f"{'Log Loss':<15} | {base_cv['log_loss']:.5f}             | {s1_cv['log_loss']:.5f} {_delta(s1_cv['log_loss'], base_cv['log_loss'], True):<8} | {s2_cv['log_loss']:.5f} {_delta(s2_cv['log_loss'], base_cv['log_loss'], True):<8}")
    print(f"{'Brier Score':<15} | {base_cv['brier_score']:.5f}             | {s1_cv['brier_score']:.5f} {_delta(s1_cv['brier_score'], base_cv['brier_score'], True):<8} | {s2_cv['brier_score']:.5f} {_delta(s2_cv['brier_score'], base_cv['brier_score'], True):<8}")
    print(f"{'Macro F1':<15} | {base_cv['f1_macro']:.5f}             | {s1_cv['f1_macro']:.5f} {_delta(s1_cv['f1_macro'], base_cv['f1_macro'], False):<8} | {s2_cv['f1_macro']:.5f} {_delta(s2_cv['f1_macro'], base_cv['f1_macro'], False):<8}")

    print("\nFinal Decision:")
    best_candidate = None
    if gate_s1 and gate_s2:
        # Pick the one with the best log loss
        if s2_cv["log_loss"] < s1_cv["log_loss"]:
            best_candidate = "Scenario 2"
        else:
            best_candidate = "Scenario 1"
        print(f"-> ADOPT {best_candidate}. Both cleared the gate, {best_candidate} performed best.")
    elif gate_s2:
        best_candidate = "Scenario 2"
        print("-> ADOPT Scenario 2. It successfully cleared the gate.")
    elif gate_s1:
        best_candidate = "Scenario 1"
        print("-> ADOPT Scenario 1. It successfully cleared the gate.")
    else:
        print("-> REJECT both. Neither expansion provided enough lift.")
        
    if best_candidate:
        print(f"\nACTION REQUIRED: Update MatchDataPipeline to use {best_candidate} and run OOS backtest.")

if __name__ == "__main__":
    main()
