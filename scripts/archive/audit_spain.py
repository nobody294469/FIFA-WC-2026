"""
Spain Advantage Decomposition and Counterfactuals
"""
import numpy as np
import pandas as pd
from collections import defaultdict
from dashboard.utils import get_pmb, load_forecast
from core.config import WC2026_TEAMS, WC2026_GROUPS
from simulation.monte_carlo import TournamentSimulator
import json

pmb = get_pmb()
sim = TournamentSimulator(pmb)
n_sims = 50000
rng = np.random.default_rng(42)

teams = ['Spain', 'France', 'Brazil', 'England']

# --- Baseline ---
forecast = load_forecast('coinflip')
base_stats = {}
for t in teams:
    row = forecast[forecast['team'] == t].iloc[0]
    base_stats[t] = {
        'Win%': row.champion_pct,
        'Final%': row.finalist_pct,
        'SF%': row.semifinal_pct,
        'QF%': row.quarterfinal_pct,
        'R16%': row.r16_pct,
        'R32%': row.r32_pct,
        'Grp1st%': row.group_1st_pct,
        'Grp2nd%': row.group_2nd_pct,
        'GrpExit%': row.group_exit_pct
    }

print("Running deep simulation to extract path difficulty...")
# Extracting average knockout opponent strength requires running a custom simulation loop
# We will do a mini-version of the simulation that tracks opponents

# We can estimate Path Difficulty by looking at the P_ko matrix.
# Let P_ko[i, j] be the probability team i knocks out team j.
# A higher avg P_ko against the field means a stronger team overall.
avg_pko = {t: np.mean(pmb.P_ko[pmb.team_idx[t], :]) * 100 for t in teams}

# Average group opponent P_win (lower is better for the team)
group_diff = {}
for t in teams:
    for g, grp_teams in WC2026_GROUPS.items():
        if t in grp_teams:
            opps = [o for o in grp_teams if o != t]
            idx = pmb.team_idx[t]
            opp_idxs = [pmb.team_idx[o] for o in opps]
            # Probability of winning against group opponents
            p_wins = [pmb.P_win[idx, oi] * 100 for oi in opp_idxs]
            group_diff[t] = {
                'avg_win_pct': np.mean(p_wins),
                'opponents': opps
            }

# R32 Path
# According to modal pairings:
# Spain (1H) plays mostly Algeria/Austria/Argentina
# France (1I) plays mostly Egypt/Iran/New Zealand
# Brazil (1C) plays mostly Japan/Netherlands/Tunisia
# England (1L) plays mostly Uzbekistan/DR Congo/Colombia

# Counterfactual simulations
def run_cf_sim(target_team):
    print(f"Running counterfactual: Swapping {target_team} into Spain's slot...")
    
    # Create modified groups
    cf_groups = {k: list(v) for k, v in WC2026_GROUPS.items()}
    
    # Find Spain and target
    spain_grp = None
    spain_idx = -1
    target_grp = None
    target_idx = -1
    
    for g, tlist in cf_groups.items():
        if 'Spain' in tlist:
            spain_grp = g
            spain_idx = tlist.index('Spain')
        if target_team in tlist:
            target_grp = g
            target_idx = tlist.index(target_team)
            
    # Swap them
    cf_groups[spain_grp][spain_idx] = target_team
    cf_groups[target_grp][target_idx] = 'Spain'
    
    # New simulator
    cf_sim = TournamentSimulator(pmb, groups=cf_groups)
    cf_rng = np.random.default_rng(42)
    df_cf = cf_sim.simulate_tournament(cf_rng, n_sims)
    
    row = df_cf[df_cf['team'] == target_team].iloc[0]
    return row.champion_pct, row.finalist_pct

cf_results = {}
for t in ['France', 'Brazil', 'England']:
    cf_win, cf_fin = run_cf_sim(t)
    cf_results[t] = {
        'cf_win_pct': cf_win,
        'cf_fin_pct': cf_fin,
        'base_win_pct': base_stats[t]['Win%'],
        'delta': cf_win - base_stats[t]['Win%']
    }

# Save results
out = {
    'base_stats': base_stats,
    'avg_pko': avg_pko,
    'group_diff': group_diff,
    'cf_results': cf_results
}

with open('reports/forecast_2026/spain_audit.json', 'w') as f:
    json.dump(out, f, indent=2)

print("Audit data generated.")
