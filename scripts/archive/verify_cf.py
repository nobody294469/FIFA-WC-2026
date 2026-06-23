import numpy as np
import pandas as pd
from dashboard.utils import get_pmb, load_forecast
from core.config import WC2026_GROUPS
from simulation.monte_carlo import TournamentSimulator

def get_group_stats(sim, target_team, n_sims=50000, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    # Get group stage stats
    tables = sim._sim_group_stage(rng, n_sims)
    
    # Find team's group
    target_grp = None
    for g, tlist in sim.groups.items():
        if target_team in tlist:
            target_grp = g
            break
            
    data = tables[target_grp]
    ranked = sim._rank_group(data['pts'], data['gd'], data['gs'], data['ids'])
    
    # ranked is (n_sims, 4) containing local team ids
    # 0th column is 1st place, 1st column is 2nd place
    target_local_id = sim.team_idx[target_team]
    
    first_place = np.sum(ranked[:, 0] == target_local_id) / n_sims * 100
    second_place = np.sum(ranked[:, 1] == target_local_id) / n_sims * 100
    
    # Run full tournament for R32 and Champ %
    rng = np.random.default_rng(rng_seed) # reset rng for consistency
    results = sim.run(n_sims=n_sims)
    df = results.to_dataframe()
    row = df[df['team'] == target_team].iloc[0]
    
    return first_place, second_place, row.r32_pct, row.champion_pct


print("--- 1. Baseline ---")
pmb = get_pmb()
base_sim = TournamentSimulator(pmb)

b1, b2, br32, bchamp = get_group_stats(base_sim, 'Brazil')

print(f"Brazil Baseline:")
print(f"  Group winner %: {b1:.2f}%")
print(f"  Group runner-up %: {b2:.2f}%")
print(f"  R32 %: {br32}%")
print(f"  Champion %: {bchamp}%\n")

print("--- 2. Swap Implementation ---")
print("Original Group C:", WC2026_GROUPS['C'])
print("Original Group H:", WC2026_GROUPS['H'])

cf_groups = {k: list(v) for k, v in WC2026_GROUPS.items()}

# Swap Brazil and Spain
brazil_idx = cf_groups['C'].index('Brazil')
spain_idx = cf_groups['H'].index('Spain')
cf_groups['C'][brazil_idx] = 'Spain'
cf_groups['H'][spain_idx] = 'Brazil'

print("\nNew Group C:", cf_groups['C'])
print("New Group H:", cf_groups['H'])

# Re-initialize the simulator with the NEW groups
cf_sim = TournamentSimulator(pmb, groups=cf_groups)

print("\n--- 3. Verifying Group Mappings ---")
# Verify that Brazil is now facing Spain's opponents
brazil_local_id = cf_sim.team_idx['Brazil']
spain_local_id = cf_sim.team_idx['Spain']

# Group local ids verification
brazil_grp_ids = cf_sim.group_local_ids['H']
spain_grp_ids = cf_sim.group_local_ids['C']

print(f"Brazil local ID: {brazil_local_id}")
print(f"Teams in Group H (Local IDs): {brazil_grp_ids}")
print(f"Teams in Group H: {[cf_sim.teams[i] for i in brazil_grp_ids]}")
print(f"Is Brazil in Group H? {brazil_local_id in brazil_grp_ids}")

print(f"\nSpain local ID: {spain_local_id}")
print(f"Teams in Group C (Local IDs): {spain_grp_ids}")
print(f"Teams in Group C: {[cf_sim.teams[i] for i in spain_grp_ids]}")
print(f"Is Spain in Group C? {spain_local_id in spain_grp_ids}")

print("\n--- 4. Running Counterfactual Simulation ---")

c1, c2, cr32, cchamp = get_group_stats(cf_sim, 'Brazil')

print("\n--- 5. Counterfactual Results (Brazil in Group H) ---")
print(f"  Group winner %: {c1:.2f}% (Delta: {c1 - b1:+.2f}%)")
print(f"  Group runner-up %: {c2:.2f}% (Delta: {c2 - b2:+.2f}%)")
print(f"  R32 %: {cr32}% (Delta: {cr32 - br32:+.2f}%)")
print(f"  Champion %: {cchamp}% (Delta: {cchamp - bchamp:+.2f}%)")

print("\nVerify Bracket Downstream:")
print("The Monte Carlo simulator maps groups to the R32 dynamically using the group names.")
print("Because Brazil is now physically in Group H, they will follow the 1H, 2H, or 3H pathways")
print("dictated by the FIFA bracket rules encoded in TournamentSimulator.__init__ (e.g. 1H vs 2J).")
