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

from core.config import XGB_PARAMS, TRAINING_START, TEAM_NAME_ALIASES, ALIAS_REVERSE, WC2026_TEAMS
from core.logger import get_logger
from data.elo_tracker import EloTracker
from data.feature_engineering import FEATURE_COLS, tournament_importance
from data.ingestion import FormTracker, H2HRegistry, build_match_records, load_raw_results, MatchDataPipeline
from models.ml_engine import MatchOutcomeModel, ProbabilityMatrixBuilder, _multiclass_brier_score
from simulation.monte_carlo import TournamentSimulator

log = get_logger("eval_draw_adj")

def _apply_draw_adj(proba: np.ndarray, is_top20_match: np.ndarray, target_draw_rate: float = 0.325) -> np.ndarray:
    """
    proba: (N, 3) where cols are [Win, Draw, Loss]
    """
    new_proba = proba.copy()
    
    # Calculate current avg draw prob for top20 matches
    if not is_top20_match.any(): return new_proba
    
    current_avg_draw = new_proba[is_top20_match, 1].mean()
    multiplier = target_draw_rate / current_avg_draw if current_avg_draw > 0 else 1.0
    multiplier = min(multiplier, 1.5) # Cap multiplier just in case
    
    for i in range(len(new_proba)):
        if is_top20_match[i]:
            p_w, p_d, p_l = new_proba[i]
            pd_new = min(p_d * multiplier, 0.99)
            diff = pd_new - p_d
            
            if p_w + p_l > 0:
                pw_new = p_w - diff * (p_w / (p_w + p_l))
                pl_new = p_l - diff * (p_l / (p_w + p_l))
            else:
                pw_new, pl_new = 0.0, 0.0
                
            new_proba[i] = [pw_new, pd_new, pl_new]
            
    return new_proba

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

            if h in wc_csv_names or a in wc_csv_names:
                s1_rows.append(record)

        form_tracker.update(h, a, hs, as_)
        h2h.update(h, a, hs, as_)
        opp_history.setdefault(h, []).append(a)
        opp_history.setdefault(a, []).append(h)
        replay_elo.update(h, a, int(row["target"]), tournament, neutral)

    tdf = pd.DataFrame(s1_rows).sort_values("date").reset_index(drop=True)
    tdf["form_diff"] = tdf["home_form"].astype(float) - tdf["away_form"].astype(float)
    tdf["match_type_enc"] = tdf["tournament"].map(tournament_importance).fillna(0.40)
    tdf["neutral_venue"] = tdf["neutral"].astype(float)
    
    return tdf

def main():
    log.info("Building dataset...")
    df_s1 = _build_holdout_corpora()

    cutoff = pd.Timestamp("2023-01-01")
    train_s1 = df_s1[df_s1["date"] < cutoff]
    test_identical = df_s1[df_s1["date"] >= cutoff]

    X_tr = train_s1[FEATURE_COLS].copy()
    y_tr = train_s1["target"].astype(int)
    
    clf = CalibratedClassifierCV(estimator=xgb.XGBClassifier(**XGB_PARAMS), method='sigmoid', cv=3)
    clf.fit(X_tr, y_tr)
    
    sub_df = test_identical[(test_identical["h_rank"] <= 20) & (test_identical["a_rank"] <= 20)].copy()
    X_te = sub_df[FEATURE_COLS].copy()
    y_te = sub_df["target"].astype(int)
    
    proba_raw = clf.predict_proba(X_te)
    ll_raw = log_loss(y_te, proba_raw, labels=[0,1,2])
    brier_raw = _multiclass_brier_score(y_te.values, proba_raw)
    
    # Adjust
    is_top20 = np.ones(len(proba_raw), dtype=bool)
    proba_adj = _apply_draw_adj(proba_raw, is_top20, target_draw_rate=0.325)
    
    ll_adj = log_loss(y_te, proba_adj, labels=[0,1,2])
    brier_adj = _multiclass_brier_score(y_te.values, proba_adj)

    print("\n=== TOP-20 POST-PROCESSING DRAW ADJUSTMENT ===")
    print(f"Log Loss:      Original {ll_raw:.5f}  ->  Adjusted {ll_adj:.5f}")
    print(f"Brier Score:   Original {brier_raw:.5f}  ->  Adjusted {brier_adj:.5f}")
    print("------------------------------------------------------------------")

    # Now let's simulate the tournament with and without this adjustment
    log.info("Running tournament simulation to see effect on Champion probabilities...")
    
    pipeline = MatchDataPipeline()
    
    elo = EloTracker()
    elo.batch_update(pipeline.matches)
    
    form_scores = pipeline.current_form_scores()
    opp_adj_form_scores = pipeline.current_opp_adjusted_form_scores()
    form_gd = pipeline.current_form_gd()
    form_details = pipeline.current_form_details()
    h2h_fn = pipeline.current_h2h
    
    model = MatchOutcomeModel()
    X_tr = df_s1[FEATURE_COLS]
    y_tr = df_s1["target"].astype(int)
    model.fit(X_tr, y_tr)
    
    pmb_raw = ProbabilityMatrixBuilder(model)
    pmb_raw.build(
        teams=WC2026_TEAMS,
        elo_tracker=elo,
        form_scores=form_scores,
        opp_adj_form_scores=opp_adj_form_scores,
        form_gd=form_gd,
        h2h_fn=h2h_fn,
        form_details=form_details,
        ko_resolution="coinflip"
    )
    
    sim_raw = TournamentSimulator(pmb_raw).run(n_sims=10000)
    df_res_raw = sim_raw.to_dataframe()
    top3_raw = df_res_raw.head(3)["champion_pct"].sum()
    
    # Now build adjusted matrix
    pmb_adj = ProbabilityMatrixBuilder(model)
    pmb_adj.build(
        teams=WC2026_TEAMS,
        elo_tracker=elo,
        form_scores=form_scores,
        opp_adj_form_scores=opp_adj_form_scores,
        form_gd=form_gd,
        h2h_fn=h2h_fn,
        form_details=form_details,
        ko_resolution="coinflip" # We still use coinflip to resolve the knockout, but starting from higher Draw probability
    )
    
    # Identify Top 20 teams
    sorted_ratings = sorted(elo.ratings.values(), reverse=True)
    top20_teams = [t for t, e in elo.ratings.items() if e >= sorted_ratings[19]]
    top20_indices = [pmb_adj.team_idx[t] for t in top20_teams if t in pmb_adj.team_idx]
    
    # Manual adjustment on P_win, P_draw, P_loss
    for i in range(pmb_adj.P_win.shape[0]):
        for j in range(pmb_adj.P_win.shape[1]):
            if i == j: continue
            if i in top20_indices and j in top20_indices:
                p_w, p_d, p_l = pmb_adj.P_win[i,j], pmb_adj.P_draw[i,j], pmb_adj.P_loss[i,j]
                multiplier = 1.21 # Based on 32.5 / 26.8
                pd_new = min(p_d * multiplier, 0.99)
                diff = pd_new - p_d
                pw_new = p_w - diff * (p_w / (p_w + p_l))
                pl_new = p_l - diff * (p_l / (p_w + p_l))
                
                pmb_adj.P_win[i,j] = pw_new
                pmb_adj.P_draw[i,j] = pd_new
                pmb_adj.P_loss[i,j] = pl_new
                
                # Re-calculate ko_resolution with coinflip based on the adjusted probabilities
                pmb_adj.P_ko[i,j] = pw_new + (pd_new * 0.5)

    sim_adj = TournamentSimulator(pmb_adj).run(n_sims=10000)
    df_res_adj = sim_adj.to_dataframe()
    top3_adj = df_res_adj.head(3)["champion_pct"].sum()

    print("\n=== TOURNAMENT SIMULATION IMPACT (10,000 Sims) ===")
    print(f"{'Team':<15} | {'Raw Champ%':<12} | {'Adj Champ%':<12} | {'Delta':<10}")
    print("-" * 55)
    
    merged = df_res_raw[['team', 'champion_pct']].merge(df_res_adj[['team', 'champion_pct']], on='team', suffixes=('_raw', '_adj'))
    for _, row in merged.head(10).iterrows():
        print(f"{row['team']:<15} | {row['champion_pct_raw']:>10.2f}% | {row['champion_pct_adj']:>10.2f}% | {row['champion_pct_adj'] - row['champion_pct_raw']:>9.2f}%")
        
    print(f"\nTop-3 Title Concentration: Original {top3_raw:.1f}% -> Adjusted {top3_adj:.1f}%")

if __name__ == "__main__":
    main()
