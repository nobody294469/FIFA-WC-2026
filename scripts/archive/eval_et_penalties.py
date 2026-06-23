import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import XGB_PARAMS, TRAINING_START, TEAM_NAME_ALIASES, ALIAS_REVERSE, WC2026_TEAMS
from core.logger import get_logger
from data.elo_tracker import EloTracker
from data.feature_engineering import FEATURE_COLS
from data.ingestion import MatchDataPipeline
from models.ml_engine import MatchOutcomeModel, ProbabilityMatrixBuilder
from simulation.monte_carlo import TournamentSimulator
from backtesting.world_cup_backtest import (
    _build_backtest_training_dataset,
    _advancement_brier,
    HISTORICAL_WORLD_CUPS,
    HistoricalTournamentSimulator
)

log = get_logger("eval_et_penalties")

def get_p_ko(pw, pd, diff, strategy="coinflip"):
    if strategy == "coinflip":
        return pw + pd * 0.5
    elif strategy == "elo_weighted":
        p_et = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
        return pw + pd * p_et
    elif strategy == "logistic_elo":
        # Dampened sensitivity for penalties/ET
        p_et = 1.0 / (1.0 + np.exp(-diff * 0.003))
        return pw + pd * p_et
    return pw + pd * 0.5

def run_historical_backtests():
    strategies = ["coinflip", "elo_weighted", "logistic_elo"]
    brier_scores = {s: [] for s in strategies}
    
    for tournament in HISTORICAL_WORLD_CUPS:
        log.info(f"Backtesting {tournament.year}...")
        X_tr, y_tr = _build_backtest_training_dataset(tournament.start_date)
        model = MatchOutcomeModel()
        model.fit(X_tr, y_tr)
        
        # We need an elo tracker synced to the tournament start
        pipeline = MatchDataPipeline() # This builds to current by default, wait!
        # Actually _build_backtest_training_dataset already builds the data up to cutoff.
        # But we need the Elo ratings of the teams AT the tournament start.
        # So we can rebuild Elo up to the cutoff.
        elo_tr = EloTracker()
        pre_cutoff = pipeline.matches[pipeline.matches["date"] < pd.Timestamp(tournament.start_date)]
        elo_tr.batch_update(pre_cutoff)
        
        # Build form and h2h logic manually or assume they are static since we only need pmb.build
        # I'll just build PMB manually for the tournament teams
        tourney_teams = [t for g in tournament.groups.values() for t in g]
        
        # Simplified PMB build since we only need the matrices:
        # P_win, P_draw
        n = len(tourney_teams)
        P_win = np.zeros((n, n))
        P_draw = np.zeros((n, n))
        
        # We need features for these teams
        from data.ingestion import FormTracker, H2HRegistry
        ft = FormTracker(window=10)
        h2h = H2HRegistry()
        opp_history = {}
        for _, row in pre_cutoff.iterrows():
            ft.update(row["home_team"], row["away_team"], row["home_score"], row["away_score"])
            h2h.update(row["home_team"], row["away_team"], row["home_score"], row["away_score"])
            opp_history.setdefault(row["home_team"], []).append(row["away_team"])
            opp_history.setdefault(row["away_team"], []).append(row["home_team"])
            
        def get_opp_adj(team):
            t_can = ALIAS_REVERSE.get(team, team)
            csv_name = TEAM_NAME_ALIASES.get(t_can, t_can)
            own = ft.get_form_score(csv_name)
            opps = opp_history.get(csv_name, [])[-10:]
            if not opps: return 0.0
            opp_mean = sum(ft.get_form_score(o) for o in opps) / len(opps)
            return own + opp_mean - 1.0

        records = []
        for i, t_i in enumerate(tourney_teams):
            for j, t_j in enumerate(tourney_teams):
                if i == j: continue
                csv_i = TEAM_NAME_ALIASES.get(t_i, t_i)
                csv_j = TEAM_NAME_ALIASES.get(t_j, t_j)
                
                diff = elo_tr.get_rating(t_i) - elo_tr.get_rating(t_j)
                hw, hd, hl, hgd = ft.get_form(csv_i)
                aw, ad, al, agd = ft.get_form(csv_j)
                
                sorted_ratings = sorted(elo_tr.ratings.values(), reverse=True)
                h_rank = sorted_ratings.index(elo_tr.get_rating(t_i)) + 1
                a_rank = sorted_ratings.index(elo_tr.get_rating(t_j)) + 1
                
                row = {
                    "elo_diff": diff,
                    "h2h_diff": h2h.win_rate_diff(csv_i, csv_j),
                    "home_form": ft.get_form_score(csv_i),
                    "away_form": ft.get_form_score(csv_j),
                    "form_diff": ft.get_form_score(csv_i) - ft.get_form_score(csv_j),
                    "opp_adj_form_diff": get_opp_adj(t_i) - get_opp_adj(t_j),
                    "gd_diff": hgd - agd,
                    "home_win_rate": hw,
                    "away_win_rate": aw,
                    "home_draw_rate": hd,
                    "away_draw_rate": ad,
                    "home_loss_rate": hl,
                    "h_rank": h_rank,
                    "a_rank": a_rank,
                    "match_type_enc": 1.0,
                    "neutral_venue": 1.0
                }
                records.append(row)
                
        if records:
            X_pred = pd.DataFrame(records)[FEATURE_COLS]
            proba = model.predict_proba(X_pred)
            
            idx = 0
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    P_win[i,j] = proba[idx, 0]
                    P_draw[i,j] = proba[idx, 1]
                    idx += 1
                    
        # Dummy PMB for simulator
        class DummyPMB:
            def __init__(self, teams, pw, pd, p_ko):
                self.teams = teams
                self.team_idx = {t: i for i, t in enumerate(teams)}
                self.P_win = pw
                self.P_draw = pd
                self.P_ko = p_ko
                
        for strat in strategies:
            P_ko = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    diff = elo_tr.get_rating(tourney_teams[i]) - elo_tr.get_rating(tourney_teams[j])
                    P_ko[i,j] = get_p_ko(P_win[i,j], P_draw[i,j], diff, strat)
                    
            pmb = DummyPMB(tourney_teams, P_win, P_draw, P_ko)
            sim = HistoricalTournamentSimulator(pmb, tournament.groups)
            res = sim.run(5000)
            df_res = res.to_dataframe()
            
            brier_champ = _advancement_brier(df_res.set_index("team")["champion_pct"], [tournament.actual["champion"]])
            brier_sf = _advancement_brier(df_res.set_index("team")["semifinal_pct"], tournament.actual["semifinalists"])
            brier_qf = _advancement_brier(df_res.set_index("team")["quarterfinal_pct"], tournament.actual["quarterfinalists"])
            brier_g = _advancement_brier(df_res.set_index("team")["round_of_16_pct"], tournament.actual["group_qualifiers"])
            avg_brier = np.mean([brier_champ, brier_sf, brier_qf, brier_g])
            brier_scores[strat].append(avg_brier)
            
    print("\\n=== HISTORICAL BACKTEST (Mean Brier Score over '14, '18, '22) ===")
    for strat in strategies:
        print(f"  {strat:<15}: {np.mean(brier_scores[strat]):.4f}")

def run_wc2026_sims():
    pipeline = MatchDataPipeline()
    elo = EloTracker()
    elo.batch_update(pipeline.matches)
    
    model = MatchOutcomeModel()
    model.load()
    
    pmb_base = ProbabilityMatrixBuilder(model)
    pmb_base.build(
        teams=WC2026_TEAMS,
        elo_tracker=elo,
        form_scores=pipeline.current_form_scores(),
        opp_adj_form_scores=pipeline.current_opp_adjusted_form_scores(),
        form_gd=pipeline.current_form_gd(),
        h2h_fn=pipeline.current_h2h,
        form_details=pipeline.current_form_details(),
        ko_resolution="coinflip"
    )
    
    strategies = ["coinflip", "elo_weighted", "logistic_elo"]
    results = {}
    
    print("\\n=== WC2026 50k SIMULATIONS ===")
    for strat in strategies:
        log.info(f"Simulating {strat}...")
        pmb_strat = ProbabilityMatrixBuilder(model)
        # Deep copy base matrices
        pmb_strat.teams = pmb_base.teams
        pmb_strat.team_idx = pmb_base.team_idx
        pmb_strat.P_win = pmb_base.P_win.copy()
        pmb_strat.P_draw = pmb_base.P_draw.copy()
        pmb_strat.P_ko = np.zeros_like(pmb_base.P_ko)
        
        for i in range(len(WC2026_TEAMS)):
            for j in range(len(WC2026_TEAMS)):
                if i == j: continue
                diff = elo.get_rating(WC2026_TEAMS[i]) - elo.get_rating(WC2026_TEAMS[j])
                pmb_strat.P_ko[i,j] = get_p_ko(pmb_strat.P_win[i,j], pmb_strat.P_draw[i,j], diff, strat)
                
        sim = TournamentSimulator(pmb_strat)
        res = sim.run(50000)
        df_res = res.to_dataframe()
        
        top3_conc = df_res.head(3)["champion_pct"].sum()
        results[strat] = {
            "top3": top3_conc,
            "champs": df_res.head(10)[["team", "champion_pct"]]
        }
        
    print(f"{'Team':<15} | {'Coinflip':<10} | {'Elo-Weight':<10} | {'Logistic':<10}")
    print("-" * 55)
    
    teams = list(results["coinflip"]["champs"]["team"])
    for t in teams:
        c_pct = results["coinflip"]["champs"].loc[results["coinflip"]["champs"]["team"] == t, "champion_pct"].values[0]
        e_pct = results["elo_weighted"]["champs"].loc[results["elo_weighted"]["champs"]["team"] == t, "champion_pct"]
        e_pct = e_pct.values[0] if len(e_pct) > 0 else 0.0
        l_pct = results["logistic_elo"]["champs"].loc[results["logistic_elo"]["champs"]["team"] == t, "champion_pct"]
        l_pct = l_pct.values[0] if len(l_pct) > 0 else 0.0
        print(f"{t:<15} | {c_pct:>9.2f}% | {e_pct:>9.2f}% | {l_pct:>9.2f}%")
        
    print("\\nTop-3 Concentration:")
    for strat in strategies:
        print(f"  {strat:<15}: {results[strat]['top3']:.1f}%")

    # Analytical Draw-Resolution Outcomes
    print("\\n=== DRAW-RESOLUTION OUTCOMES (Top-20 vs Top-20 matrix) ===")
    sorted_ratings = sorted(elo.ratings.values(), reverse=True)
    top20_teams = [t for t, e in elo.ratings.items() if e >= sorted_ratings[19]]
    
    outcomes = {s: [] for s in strategies}
    for t_i in top20_teams:
        for t_j in top20_teams:
            if t_i == t_j: continue
            if t_i not in WC2026_TEAMS or t_j not in WC2026_TEAMS: continue
            
            i = pmb_base.team_idx[t_i]
            j = pmb_base.team_idx[t_j]
            diff = elo.get_rating(t_i) - elo.get_rating(t_j)
            if diff <= 0: continue # Only look at Favorite vs Underdog
            
            pw = pmb_base.P_win[i,j]
            pd = pmb_base.P_draw[i,j]
            
            # Given a draw occurred, what is the P(Favorite advances)?
            for strat in strategies:
                p_adv_total = get_p_ko(pw, pd, diff, strat)
                # p_adv_total = pw + pd * p_adv_given_draw
                # Therefore p_adv_given_draw = (p_adv_total - pw) / pd
                if pd > 0:
                    p_adv_given_draw = (p_adv_total - pw) / pd
                    outcomes[strat].append(p_adv_given_draw)
                    
    for strat in strategies:
        mean_et_win = np.mean(outcomes[strat]) * 100
        print(f"  {strat:<15}: Favorite wins {mean_et_win:.1f}% of Extra Time / Penalty shootouts")

if __name__ == "__main__":
    run_historical_backtests()
    run_wc2026_sims()
