"""
eval_tournament_weighted_form.py — Benchmark Tournament-Weighted Form vs Current Form

Runs a parallel benchmark replacing the standard FormTracker with a
Tournament-Weighted FormTracker where matches are weighted by their tournament importance.

Gate criterion (ADOPT if any one is met, REJECT if none):
  - CV Log Loss delta <= -0.005
  - CV Brier Score delta <= -0.002

PRODUCTION CODE IS NOT MODIFIED BY THIS SCRIPT.
"""
from __future__ import annotations

import os
import sys
import time
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

log = get_logger("eval_tournament_weighted_form")

# ══════════════════════════════════════════════════════════════════════════════
# Tournament-Weighted Form Tracker
# ══════════════════════════════════════════════════════════════════════════════

class TournamentWeightedFormTracker(FormTracker):
    """
    Extends FormTracker to apply a weight based on tournament importance.
    Stores (win_flag, draw_flag, loss_flag, gd, weight)
    """
    def update_weighted(self, home: str, away: str, home_score: int, away_score: int, tournament: str) -> None:
        self._ensure(home)
        self._ensure(away)
        gd = home_score - away_score
        
        weight = tournament_importance(tournament)
        
        if home_score > away_score:
            self._records[home].append((1, 0, 0, gd, weight))
            self._records[away].append((0, 0, 1, -gd, weight))
        elif home_score == away_score:
            self._records[home].append((0, 1, 0, 0, weight))
            self._records[away].append((0, 1, 0, 0, weight))
        else:
            self._records[home].append((0, 0, 1, gd, weight))
            self._records[away].append((1, 0, 0, -gd, weight))

        if len(self._records[home]) > self.window * 3:
            self._records[home] = self._records[home][-self.window * 3:]
        if len(self._records[away]) > self.window * 3:
            self._records[away] = self._records[away][-self.window * 3:]

    def get_form(self, team: str) -> Tuple[float, float, float, float]:
        recs = self._records.get(team, [])
        recent = recs[-self.window:] if len(recs) >= self.window else recs
        if not recent:
            return 0.33, 0.33, 0.33, 0.0

        # Elements are (win, draw, loss, gd, weight)
        total_weight = sum(r[4] for r in recent)
        
        if total_weight == 0:
            return 0.33, 0.33, 0.33, 0.0
            
        wins = sum(r[0] * r[4] for r in recent) / total_weight
        draws = sum(r[1] * r[4] for r in recent) / total_weight
        losses = sum(r[2] * r[4] for r in recent) / total_weight
        avg_gd = sum(r[3] * r[4] for r in recent) / total_weight
        
        return wins, draws, losses, avg_gd

# ══════════════════════════════════════════════════════════════════════════════
# Feature dataset builder
# ══════════════════════════════════════════════════════════════════════════════

def _build_feature_dataset(use_weighted: bool) -> Tuple[pd.DataFrame, pd.Series]:
    """Build (X, y) using either baseline or weighted form tracker."""
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

    if use_weighted:
        form_tracker = TournamentWeightedFormTracker(window=10)
    else:
        form_tracker = FormTracker(window=10)
        
    h2h = H2HRegistry()
    opp_history: Dict[str, List[str]] = {}
    training_rows: List[Dict] = []

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

        if (date >= training_start and h in wc_csv_names and a in wc_csv_names):
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

            training_rows.append({
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
            })

        if use_weighted:
            form_tracker.update_weighted(h, a, hs, as_, tournament)
        else:
            form_tracker.update(h, a, hs, as_)
            
        h2h.update(h, a, hs, as_)
        opp_history.setdefault(h, []).append(a)
        opp_history.setdefault(a, []).append(h)
        replay_elo.update(h, a, int(row["target"]), tournament, neutral)

    tdf = pd.DataFrame(training_rows).sort_values("date").reset_index(drop=True)
    tdf["form_diff"] = tdf["home_form"].astype(float) - tdf["away_form"].astype(float)
    tdf["match_type_enc"] = tdf["tournament"].map(tournament_importance).fillna(0.40)
    tdf["neutral_venue"] = tdf["neutral"].astype(float)

    missing = [c for c in FEATURE_COLS if c not in tdf.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    return tdf[FEATURE_COLS].copy(), tdf["target"].astype(int)

# ══════════════════════════════════════════════════════════════════════════════
# Evaluation metrics
# ══════════════════════════════════════════════════════════════════════════════

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

def _get_importance(X: pd.DataFrame, y: pd.Series) -> Tuple[Dict[str, float], Dict[str, float]]:
    m = MatchOutcomeModel()
    m.fit(X, y)
    
    gain = m.feat_importance
    
    dmatrix = xgb.DMatrix(X)
    contribs = m.model.get_booster().predict(dmatrix, pred_contribs=True)
    abs_shap = np.abs(contribs[:, :, :-1])
    mean_abs_shap = np.mean(abs_shap, axis=(0, 1))
    shap = dict(zip(FEATURE_COLS, mean_abs_shap))
    
    return gain, shap

def _delta(new: float, old: float, lower_is_better: bool = True) -> str:
    d = new - old
    sign = "+" if d > 0 else ""
    marker = ""
    if lower_is_better and d < -0.001:
        marker = " ✓"
    elif not lower_is_better and d > 0.001:
        marker = " ✓"
    return f"{sign}{d:.5f}{marker}"

def main():
    log.info("Building baseline dataset...")
    X_base, y_base = _build_feature_dataset(use_weighted=False)
    
    log.info("Building tournament-weighted form dataset...")
    X_wt, y_wt = _build_feature_dataset(use_weighted=True)

    base_cv = _run_cv(X_base, y_base, "Baseline")
    wt_cv = _run_cv(X_wt, y_wt, "WeightedForm")

    base_gain, base_shap = _get_importance(X_base, y_base)
    wt_gain, wt_shap = _get_importance(X_wt, y_wt)

    target_features = ["form_diff", "opp_adj_form_diff", "home_form", "away_form"]
    
    base_comb_gain = sum(base_gain.get(f, 0) for f in target_features)
    wt_comb_gain = sum(wt_gain.get(f, 0) for f in target_features)
    
    base_comb_shap = sum(base_shap.get(f, 0) for f in target_features)
    wt_comb_shap = sum(wt_shap.get(f, 0) for f in target_features)

    gate_passed = (wt_cv["log_loss"] - base_cv["log_loss"] <= -0.005) or (wt_cv["brier_score"] - base_cv["brier_score"] <= -0.002)

    print("\n=== TOURNAMENT-WEIGHTED FORM BENCHMARK ===")
    print(f"Gate Passed: {'YES' if gate_passed else 'NO'}")
    
    print("\n1. CV Metrics (5-Fold TimeSeriesSplit, Platt Calibrated)")
    print(f"{'Metric':<15} | {'Baseline':<10} | {'Weighted':<10} | {'Delta':<10}")
    print("-" * 55)
    print(f"{'Accuracy':<15} | {base_cv['accuracy']:.5f}    | {wt_cv['accuracy']:.5f}    | {_delta(wt_cv['accuracy'], base_cv['accuracy'], False)}")
    print(f"{'Log Loss':<15} | {base_cv['log_loss']:.5f}    | {wt_cv['log_loss']:.5f}    | {_delta(wt_cv['log_loss'], base_cv['log_loss'], True)}")
    print(f"{'Brier Score':<15} | {base_cv['brier_score']:.5f}    | {wt_cv['brier_score']:.5f}    | {_delta(wt_cv['brier_score'], base_cv['brier_score'], True)}")
    print(f"{'Macro F1':<15} | {base_cv['f1_macro']:.5f}    | {wt_cv['f1_macro']:.5f}    | {_delta(wt_cv['f1_macro'], base_cv['f1_macro'], False)}")

    print("\n2. Importance Changes (Combined form_diff, opp_adj_form_diff, home_form, away_form)")
    print(f"{'Metric':<15} | {'Baseline':<10} | {'Weighted':<10} | {'Delta':<10}")
    print("-" * 55)
    print(f"{'Combined Gain':<15} | {base_comb_gain:.5f}    | {wt_comb_gain:.5f}    | {_delta(wt_comb_gain, base_comb_gain, False)}")
    print(f"{'Combined SHAP':<15} | {base_comb_shap:.5f}    | {wt_comb_shap:.5f}    | {_delta(wt_comb_shap, base_comb_shap, False)}")

    print("\nFinal Decision:")
    if gate_passed:
        print("-> ADOPT. The Tournament-Weighted Form successfully cleared the gate. Proceed with merging into production.")
    else:
        print("-> REJECT. The Tournament-Weighted Form failed to provide meaningful lift. Do not merge into production.")

if __name__ == "__main__":
    main()
