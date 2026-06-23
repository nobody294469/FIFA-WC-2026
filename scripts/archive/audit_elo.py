import pandas as pd
import numpy as np
from data.ingestion import MatchDataPipeline
from data.elo_tracker import EloTracker
from models.ml_engine import MatchOutcomeModel, ProbabilityMatrixBuilder

print("Loading data pipeline...")
pipeline = MatchDataPipeline()
matches = pipeline.matches.copy()

matches['date'] = pd.to_datetime(matches['date'])
pre_2023 = matches[matches['date'] < '2023-01-01'].copy()
post_2023 = matches[matches['date'] >= '2023-01-01'].copy()

print("Initializing Elo up to 2023...")
elo = EloTracker()
elo.batch_update(pre_2023)

targets = ['Spain', 'Argentina', 'France', 'Brazil', 'England', 'Germany', 'Portugal']
history = {t: [] for t in targets}
max_gains = {t: (0, "", "") for t in targets}

print("Simulating Elo match-by-match from 2023 onwards...")
for idx, row in post_2023.iterrows():
    date = row['date'].strftime('%Y-%m-%d')
    home = row['home_team']
    away = row['away_team']
    outcome = row['target']
    tournament = row['tournament']
    neutral = row['neutral']
    
    # Track pre-match
    pre_h = elo.get_rating(home) if home in targets else 0
    pre_a = elo.get_rating(away) if away in targets else 0
    
    # Update Elo
    k = elo._k_factor(tournament)
    elo.update(home, away, outcome, tournament, neutral)
    
    # Record post-match and delta
    if home in targets:
        post = elo.get_rating(home)
        delta = post - pre_h
        res = "Win" if outcome == 0 else "Draw" if outcome == 1 else "Loss"
        history[home].append((date, f"vs {away}", res, pre_h, post, delta, k, tournament))
        if delta > max_gains[home][0]:
            max_gains[home] = (delta, date, f"vs {away} ({tournament})")
            
    if away in targets:
        post = elo.get_rating(away)
        delta = post - pre_a
        res = "Win" if outcome == 2 else "Draw" if outcome == 1 else "Loss"
        history[away].append((date, f"@ {home}", res, pre_a, post, delta, k, tournament))
        if delta > max_gains[away][0]:
            max_gains[away] = (delta, date, f"@ {home} ({tournament})")

print("\n=== ELO EVOLUTION (Jan 2023 - Jun 2026) ===")
for t in targets:
    print(f"\n{t.upper()}")
    print(f"{'Date':<12} | {'Opponent':<20} | {'Res':<5} | {'Pre':<6} | {'Post':<6} | {'Delta':<6} | {'K':<4} | {'Tournament'}")
    print("-" * 90)
    for date, opp, res, pre, post, delta, k, tourn in history[t]:
        # Only show a subset if it's too long, but user asked for "every Elo gain/loss after each match"
        # Since it's only ~3 years, maybe ~30-40 matches per team. We will print all.
        print(f"{date:<12} | {opp:<20} | {res:<5} | {pre:6.1f} | {post:6.1f} | {delta:+6.1f} | {k:<4} | {tourn}")

print("\n=== LARGEST ELO GAINS ===")
for t in targets:
    gain, date, desc = max_gains[t]
    print(f"{t:<10}: {gain:+6.1f} on {date} {desc}")

print("\n=== FINAL RATINGS CONTEXT ===")
for t in targets:
    print(f"{t:<10}: {elo.get_rating(t):.1f}")
    
print("\n=== ELO-IMPLIED vs XGBOOST PROBABILITY ===")
elo_full = EloTracker()
elo_full.batch_update(matches)

# Compare implied Elo prob vs XGBoost matrix prob
model = MatchOutcomeModel()
model.load()
pmb = ProbabilityMatrixBuilder(model)
pmb.build(
    teams=['Spain', 'Brazil', 'England', 'France'],
    elo_tracker=elo_full,
    form_scores=pipeline.current_form_scores(),
    opp_adj_form_scores=pipeline.current_opp_adjusted_form_scores(),
    form_gd=pipeline.current_form_gd(),
    h2h_fn=pipeline.current_h2h,
    form_details=pipeline.current_form_details(),
    ko_resolution="coinflip",
)

def compare_probs(team_a, team_b):
    elo_win_a, elo_draw, elo_win_b = elo_full.expected_probs(team_a, team_b, neutral=True)
    idx_a = pmb.team_idx[team_a]
    idx_b = pmb.team_idx[team_b]
    xgb_win_a = pmb.P_win[idx_a, idx_b]
    
    print(f"\nMatchup: {team_a} vs {team_b}")
    print(f"Elo Only Implied Probability: {team_a} Win {elo_win_a*100:.1f}%")
    print(f"XGBoost Model Probability:    {team_a} Win {xgb_win_a*100:.1f}%")

compare_probs('Spain', 'Brazil')
compare_probs('Spain', 'England')
