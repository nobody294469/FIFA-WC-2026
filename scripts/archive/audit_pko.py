import numpy as np
from dashboard.utils import get_pmb
from simulation.monte_carlo import TournamentSimulator

pmb = get_pmb()
sim = TournamentSimulator(pmb)

teams = sim.teams

def get_pko_profile(team_name):
    idx = sim.team_idx[team_name]
    pko_list = []
    for t in teams:
        if t != team_name:
            t_idx = sim.team_idx[t]
            pko_list.append((t, sim.P_ko[idx, t_idx] * 100))
    pko_list.sort(key=lambda x: x[1], reverse=True)
    return pko_list

spain_pko = get_pko_profile('Spain')
france_pko = get_pko_profile('France')

print("=== SPAIN P_KO PROFILE ===")
print("Top 10 Easiest Opponents:")
for t, p in spain_pko[:10]:
    print(f"  vs {t}: {p:.1f}%")

print("\nTop 10 Hardest Opponents:")
for t, p in reversed(spain_pko[-10:]):
    print(f"  vs {t}: {p:.1f}%")

print("\n=== FRANCE P_KO PROFILE ===")
print("Top 10 Easiest Opponents:")
for t, p in france_pko[:10]:
    print(f"  vs {t}: {p:.1f}%")

print("\nTop 10 Hardest Opponents:")
for t, p in reversed(france_pko[-10:]):
    print(f"  vs {t}: {p:.1f}%")

h2h_targets = ['France', 'Argentina', 'Brazil', 'England', 'Germany', 'Portugal']

print("\n=== SPAIN HEAD-TO-HEAD VS ELITE ===")
for t in h2h_targets:
    if t == 'Spain': continue
    idx1 = sim.team_idx['Spain']
    idx2 = sim.team_idx[t]
    print(f"  Spain vs {t}: {sim.P_ko[idx1, idx2] * 100:.1f}%")

print("\n=== FRANCE HEAD-TO-HEAD VS ELITE ===")
h2h_targets_fr = ['Spain', 'Argentina', 'Brazil', 'England', 'Germany', 'Portugal']
for t in h2h_targets_fr:
    if t == 'France': continue
    idx1 = sim.team_idx['France']
    idx2 = sim.team_idx[t]
    print(f"  France vs {t}: {sim.P_ko[idx1, idx2] * 100:.1f}%")

# Distribution metrics
spain_probs = [p for t, p in spain_pko]
france_probs = [p for t, p in france_pko]

print("\n=== DISTRIBUTION SUMMARY ===")
print(f"Spain Average: {np.mean(spain_probs):.1f}%")
print(f"France Average: {np.mean(france_probs):.1f}%")
print(f"Spain Median: {np.median(spain_probs):.1f}%")
print(f"France Median: {np.median(france_probs):.1f}%")
print(f"Spain % of opponents where win prob > 60%: {sum(1 for p in spain_probs if p > 60) / 47 * 100:.1f}%")
print(f"France % of opponents where win prob > 60%: {sum(1 for p in france_probs if p > 60) / 47 * 100:.1f}%")
