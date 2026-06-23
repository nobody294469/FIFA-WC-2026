"""
analyze_stability.py - Analyzes the stability of the 2026 forecast.
"""
import sys
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import N_SIMULATIONS, RANDOM_SEED, WC2026_GROUPS, WC2026_TEAMS
from data.elo_tracker import EloTracker
from data.ingestion import MatchDataPipeline
from models.ml_engine import MatchOutcomeModel, ProbabilityMatrixBuilder
from simulation.monte_carlo import TournamentSimulator

class StabilitySimulator(TournamentSimulator):
    def run_stability(self, n_sims: int):
        rng = np.random.default_rng(RANDOM_SEED)
        
        N = self.n_teams
        r32_cnt = np.zeros(N, dtype=np.int32)
        group_1st = np.zeros(N, dtype=np.int32)
        group_2nd = np.zeros(N, dtype=np.int32)
        group_3rd = np.zeros(N, dtype=np.int32)
        
        r32_configs = Counter()
        r32_matchups = Counter()
        
        from core.config import N_BEST_THIRD
        
        tables = self._sim_group_stage(rng, n_sims)
        
        winners_g = {}
        runners_g = {}
        t3_pts_cols, t3_gd_cols, t3_gs_cols, t3_id_cols = [], [], [], []
        
        for g in self.group_names:
            data = tables[g]
            ranked = self._rank_group(data["pts"], data["gd"], data["gs"], data["ids"])
            winners_g[g] = ranked[:, 0]
            runners_g[g] = ranked[:, 1]
            
            for col in range(4):
                if col == 0:
                    np.add.at(group_1st, ranked[:, col], 1)
                elif col == 1:
                    np.add.at(group_2nd, ranked[:, col], 1)
            
            order_idx = np.lexsort((-data["gs"], -data["gd"], -data["pts"]), axis=1)
            t3_pts_cols.append(data["pts"][np.arange(n_sims), order_idx[:, 2]])
            t3_gd_cols.append( data["gd"][ np.arange(n_sims), order_idx[:, 2]])
            t3_gs_cols.append( data["gs"][ np.arange(n_sims), order_idx[:, 2]])
            t3_id_cols.append(ranked[:, 2])

        all_t3_pts = np.column_stack(t3_pts_cols)
        all_t3_gd  = np.column_stack(t3_gd_cols)
        all_t3_gs  = np.column_stack(t3_gs_cols)
        all_t3_ids = np.column_stack(t3_id_cols)

        t3_order = np.lexsort((-all_t3_gs, -all_t3_gd, -all_t3_pts), axis=1)
        
        chosen_groups = t3_order[:, :N_BEST_THIRD]
        masks = np.sum(1 << chosen_groups, axis=1)
        mapped_gi = self.t3_lookup[masks]
        sim_idx = np.arange(n_sims)[:, None]
        t3_assigned_ids = all_t3_ids[sim_idx, mapped_gi]
        
        # Count who actually advanced as 3rd
        for sim_i in range(n_sims):
            for m in range(8):
                group_3rd[t3_assigned_ids[sim_i, m]] += 1

        r32 = np.empty((n_sims, 32), dtype=np.int32)
        r32[:, 0]  = winners_g["E"];  r32[:, 1]  = t3_assigned_ids[:, 3]
        r32[:, 2]  = winners_g["I"];  r32[:, 3]  = t3_assigned_ids[:, 5]
        r32[:, 4]  = runners_g["A"];  r32[:, 5]  = runners_g["B"]
        r32[:, 6]  = winners_g["F"];  r32[:, 7]  = runners_g["C"]
        r32[:, 8]  = runners_g["K"];  r32[:, 9]  = runners_g["L"]
        r32[:, 10] = winners_g["H"];  r32[:, 11] = runners_g["J"]
        r32[:, 12] = winners_g["D"];  r32[:, 13] = t3_assigned_ids[:, 2]
        r32[:, 14] = winners_g["G"];  r32[:, 15] = t3_assigned_ids[:, 4]
        r32[:, 16] = winners_g["C"];  r32[:, 17] = runners_g["F"]
        r32[:, 18] = runners_g["E"];  r32[:, 19] = runners_g["I"]
        r32[:, 20] = winners_g["A"];  r32[:, 21] = t3_assigned_ids[:, 0]
        r32[:, 22] = winners_g["L"];  r32[:, 23] = t3_assigned_ids[:, 7]
        r32[:, 24] = winners_g["J"];  r32[:, 25] = runners_g["H"]
        r32[:, 26] = runners_g["D"];  r32[:, 27] = runners_g["G"]
        r32[:, 28] = winners_g["B"];  r32[:, 29] = t3_assigned_ids[:, 1]
        r32[:, 30] = winners_g["K"];  r32[:, 31] = t3_assigned_ids[:, 6]
        
        for col in range(32):
            np.add.at(r32_cnt, r32[:, col], 1)

        for sim_i in range(n_sims):
            bracket_tuple = tuple(
                tuple(sorted((int(r32[sim_i, 2*m]), int(r32[sim_i, 2*m+1]))))
                for m in range(16)
            )
            r32_configs[bracket_tuple] += 1
            for m in range(16):
                pair = frozenset((int(r32[sim_i, 2*m]), int(r32[sim_i, 2*m+1])))
                r32_matchups[pair] += 1
                
        return {
            "group_1st": group_1st,
            "group_2nd": group_2nd,
            "group_3rd_adv": group_3rd,
            "r32_cnt": r32_cnt,
            "r32_configs": r32_configs,
            "r32_matchups": r32_matchups
        }

def main():
    data_pipeline = MatchDataPipeline()
    elo = EloTracker()
    elo.batch_update(data_pipeline.matches)
    
    form_scores = data_pipeline.current_form_scores()
    opp_adj_form_scores = data_pipeline.current_opp_adjusted_form_scores()
    form_gd = data_pipeline.current_form_gd()
    form_details = data_pipeline.current_form_details()
    h2h_fn = data_pipeline.current_h2h
    
    model = MatchOutcomeModel()
    model.load()
    
    pmb = ProbabilityMatrixBuilder(model)
    pmb.build(
        teams=WC2026_TEAMS,
        elo_tracker=elo,
        form_scores=form_scores,
        opp_adj_form_scores=opp_adj_form_scores,
        form_gd=form_gd,
        form_details=form_details,
        h2h_fn=h2h_fn
    )
    
    sim = StabilitySimulator(prob_matrix_builder=pmb, groups=WC2026_GROUPS)
    n_sims = 50000
    res = sim.run_stability(n_sims)
    
    print(f"Unique R32 configurations: {len(res['r32_configs'])}")
    print("Top 20 R32 matchups:")
    for pair, cnt in res['r32_matchups'].most_common(20):
        t1, t2 = sorted(list(pair))
        t1_name, t2_name = sim.teams[t1], sim.teams[t2]
        print(f"{t1_name} vs {t2_name}: {cnt/n_sims*100:.2f}%")
        
    print("\nQualification stability:")
    df_rows = []
    for i, t in enumerate(sim.teams):
        q_pct = res["r32_cnt"][i] / n_sims * 100
        g1_pct = res["group_1st"][i] / n_sims * 100
        g2_pct = res["group_2nd"][i] / n_sims * 100
        g3_pct = res["group_3rd_adv"][i] / n_sims * 100
        df_rows.append({
            "Team": t,
            "Qualify %": q_pct,
            "1st %": g1_pct,
            "2nd %": g2_pct,
            "3rd Adv %": g3_pct,
            "3rd Reliance": g3_pct / q_pct if q_pct > 0 else 0
        })
    import pandas as pd
    df = pd.DataFrame(df_rows)
    print("Most stable (highest 1st+2nd %):")
    print(df.sort_values("1st %", ascending=False).head(10).to_string(index=False))
    print("\nHighest reliance on 3rd place:")
    print(df.sort_values("3rd Reliance", ascending=False).head(10).to_string(index=False))

    # Output probabilities of top teams against each other to explain the gap
    print("\nMatch probabilities among top teams:")
    top_teams = ["Spain", "Argentina", "France", "Brazil", "England", "Portugal"]
    for i in range(len(top_teams)):
        for j in range(i+1, len(top_teams)):
            t1 = top_teams[i]
            t2 = top_teams[j]
            t1_idx = pmb.team_idx[t1]
            t2_idx = pmb.team_idx[t2]
            pw, pd, pl = pmb.P_win[t1_idx, t2_idx], pmb.P_draw[t1_idx, t2_idx], pmb.P_ko[t1_idx, t2_idx]
            print(f"{t1} vs {t2}: {t1} KO adv = {pw*100:.1f}%, {t2} KO adv = {(1-pw)*100:.1f}%")

if __name__ == "__main__":
    main()
