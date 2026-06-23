import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import XGB_PARAMS, TRAINING_START, TEAM_NAME_ALIASES, ALIAS_REVERSE
from data.feature_engineering import FEATURE_COLS, tournament_importance
from data.ingestion import FormTracker, H2HRegistry, build_match_records, load_raw_results
from data.elo_tracker import EloTracker
from models.ml_engine import _multiclass_brier_score

def get_ece(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

def _build_corpora():
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
    opp_history = {}
    
    s1_rows = []
    base_rows = []

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
                if not opponents: return 0.0
                return own_form + sum(form_tracker.get_form_score(o) for o in opponents) / len(opponents) - 1.0

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

            is_s1 = h in wc_csv_names or a in wc_csv_names
            is_base = h in wc_csv_names and a in wc_csv_names
            
            if is_s1: s1_rows.append(record)
            if is_base: base_rows.append(record)

        form_tracker.update(h, a, hs, as_)
        h2h.update(h, a, hs, as_)
        opp_history.setdefault(h, []).append(a)
        opp_history.setdefault(a, []).append(h)
        replay_elo.update(h, a, int(row["target"]), tournament, neutral)

    def prep(rows):
        tdf = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
        tdf["form_diff"] = tdf["home_form"].astype(float) - tdf["away_form"].astype(float)
        tdf["match_type_enc"] = tdf["tournament"].map(tournament_importance).fillna(0.40)
        tdf["neutral_venue"] = tdf["neutral"].astype(float)
        return tdf

    return prep(s1_rows), prep(base_rows)

def run_ablation(name, model, X_te, y_te):
    proba = model.predict_proba(X_te)
    ll = log_loss(y_te, proba, labels=[0,1,2])
    brier = _multiclass_brier_score(y_te.values, proba)
    
    # Draw stats
    avg_draw = proba[:, 1].mean() * 100
    y_true_draw = (y_te == 1).astype(int)
    y_prob_draw = proba[:, 1]
    ece_draw = get_ece(y_true_draw.values, y_prob_draw)
    
    print(f"{name:<25} | {ll:>8.5f} | {brier:>8.5f} | {avg_draw:>8.2f}% | {ece_draw:>8.5f}")

def main():
    print("Building datasets...")
    df_s1, df_base = _build_corpora()

    cutoff = pd.Timestamp("2023-01-01")
    
    # Scenario 1 (expanded)
    train_s1 = df_s1[df_s1["date"] < cutoff]
    test_s1 = df_s1[df_s1["date"] >= cutoff]
    X_tr_s1 = train_s1[FEATURE_COLS]
    y_tr_s1 = train_s1["target"].astype(int)
    
    # Baseline (strict)
    train_base = df_base[df_base["date"] < cutoff]
    X_tr_base = train_base[FEATURE_COLS]
    y_tr_base = train_base["target"].astype(int)
    
    # Test identical Top-20 vs Top-20 set
    test_top20 = test_s1[(test_s1["h_rank"] <= 20) & (test_s1["a_rank"] <= 20)]
    X_te = test_top20[FEATURE_COLS]
    y_te = test_top20["target"].astype(int)
    
    print(f"\\nIdentical Test Set: {len(test_top20)} Top-20 vs Top-20 matches")
    actual_draw = (y_te == 1).mean() * 100
    print(f"Actual Draw Rate in Test Set: {actual_draw:.2f}%\\n")

    print(f"{'Ablation':<25} | {'Log Loss':>8} | {'Brier':>8} | {'Avg Draw':>9} | {'Draw ECE':>8}")
    print("-" * 75)
    
    # 1. Baseline Raw XGB
    m_base_raw = xgb.XGBClassifier(**XGB_PARAMS)
    m_base_raw.fit(X_tr_base, y_tr_base)
    run_ablation("Baseline Raw XGB", m_base_raw, X_te, y_te)

    # 2. Baseline + Platt
    m_base_platt = CalibratedClassifierCV(estimator=xgb.XGBClassifier(**XGB_PARAMS), method='sigmoid', cv=3)
    m_base_platt.fit(X_tr_base, y_tr_base)
    run_ablation("Baseline + Platt", m_base_platt, X_te, y_te)

    # 3. S1 Raw XGB
    m_s1_raw = xgb.XGBClassifier(**XGB_PARAMS)
    m_s1_raw.fit(X_tr_s1, y_tr_s1)
    run_ablation("Scenario 1 Raw XGB", m_s1_raw, X_te, y_te)
    
    # 4. S1 + Platt
    m_s1_platt = CalibratedClassifierCV(estimator=xgb.XGBClassifier(**XGB_PARAMS), method='sigmoid', cv=3)
    m_s1_platt.fit(X_tr_s1, y_tr_s1)
    run_ablation("Scenario 1 + Platt", m_s1_platt, X_te, y_te)
    
    # 5. S1 + Isotonic
    m_s1_iso = CalibratedClassifierCV(estimator=xgb.XGBClassifier(**XGB_PARAMS), method='isotonic', cv=3)
    m_s1_iso.fit(X_tr_s1, y_tr_s1)
    run_ablation("Scenario 1 + Isotonic", m_s1_iso, X_te, y_te)

if __name__ == "__main__":
    main()
