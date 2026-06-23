"""
monte_carlo.py — High-speed Monte Carlo WC2026 tournament simulator.

Design
──────
• All match probabilities are pre-loaded from the ProbabilityMatrixBuilder
  numpy arrays: zero model calls inside the simulation loop.
• Group stage: full round-robin (6 matches per group × 12 groups = 72 matches),
  simulated for all N_SIMULATIONS simultaneously via vectorised numpy random draws.
• Standing tiebreaker: points → goal difference (synthetic ±) → goals scored.
• Best-8 third-place: select by points → GD → GS across all 12 groups.
• Knockout rounds (R32 → R16 → QF → SF → Final): draw probabilities are
  redistributed proportionally so every knockout match produces exactly one winner.
• All team IDs are contiguous integers [0, 47]; numpy fancy indexing is used
  throughout for O(1) per-matchup probability lookups.

Target performance: 10,000 simulations in < 15 seconds on a standard laptop.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.config import N_BEST_THIRD, N_SIMULATIONS, RANDOM_SEED, WC2026_GROUPS
from core.logger import get_logger

log = get_logger("monte_carlo")

TEAMS_PER_GROUP = 4


# ═══════════════════════════════════════════════════════════════════════════════
# Result container
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SimulationResults:
    n_sims:              int
    teams:               List[str]
    champion_counts:     np.ndarray
    finalist_counts:     np.ndarray
    semifinal_counts:    np.ndarray
    quarterfinal_counts: np.ndarray
    r16_counts:          np.ndarray
    r32_counts:          np.ndarray
    group_exit_counts:   np.ndarray
    elapsed_s:           float = 0.0

    def to_dataframe(self) -> pd.DataFrame:
        n = self.n_sims
        df = pd.DataFrame({
            "team":             self.teams,
            "champion_pct":     self.champion_counts / n * 100,
            "finalist_pct":     self.finalist_counts / n * 100,
            "semifinal_pct":    self.semifinal_counts / n * 100,
            "quarterfinal_pct": self.quarterfinal_counts / n * 100,
            "r16_pct":          self.r16_counts / n * 100,
            "r32_pct":          self.r32_counts / n * 100,
            "group_exit_pct":   self.group_exit_counts / n * 100,
        }).sort_values("champion_pct", ascending=False).reset_index(drop=True)
        df.insert(0, "rank", range(1, len(df) + 1))
        return df


# ═══════════════════════════════════════════════════════════════════════════════
# Tournament Simulator
# ═══════════════════════════════════════════════════════════════════════════════

class TournamentSimulator:
    """
    Vectorised WC2026 Monte Carlo simulator.

    Initialisation
    ──────────────
    1. Build local integer index [0..47] for the 48 WC teams.
    2. Extract probability sub-matrices for these 48 teams from the full
       ProbabilityMatrixBuilder arrays.
    3. Pre-generate all group-stage match pair indices.
    4. Compute the FIFA 3rd-place to R32 assignment mapping matrix.

    Simulation loop (run() method)
    ──────────────────────────────
    A. Group stage     — 72 matches × 10,000 sims via batch random draws
    B. Best-8 thirds   — numpy lexsort over (12 × 10,000) points tables
    C. R32 (16 matches) → R16 (8) → QF (4) → SF (2) → Final (1)
       All knockout rounds use the P_ko (draw-redistributed) matrix.
    """

    def __init__(
        self,
        prob_matrix_builder,           # ProbabilityMatrixBuilder
        groups: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self.pmb    = prob_matrix_builder
        self.groups = groups or WC2026_GROUPS

        # ── Local integer index ────────────────────────────────────────────────
        all_teams = [t for grp in self.groups.values() for t in grp]
        assert len(all_teams) == 48, f"Expected 48 teams, got {len(all_teams)}"
        self.teams    = all_teams
        self.n_teams  = len(all_teams)
        self.team_idx = {t: i for i, t in enumerate(all_teams)}

        # ── Map from global (ProbabilityMatrixBuilder) index to local ─────────
        global_to_local = {
            self.pmb.team_idx[t]: i for i, t in enumerate(all_teams)
        }

        # Extract probability sub-matrices for these 48 teams only
        N = self.n_teams
        global_idxs = np.array([self.pmb.team_idx[t] for t in all_teams], dtype=np.int32)

        self.P_win_gs  = np.zeros((N, N), dtype=np.float32)
        self.P_draw_gs = np.zeros((N, N), dtype=np.float32)
        self.P_ko      = np.zeros((N, N), dtype=np.float32)

        for li, gi in enumerate(global_idxs):
            for lj, gj in enumerate(global_idxs):
                if li == lj:
                    continue
                self.P_win_gs[li, lj]  = self.pmb.P_win[gi, gj]
                self.P_draw_gs[li, lj] = self.pmb.P_draw[gi, gj]
                self.P_ko[li, lj]      = self.pmb.P_ko[gi, gj]

        # ── Group metadata ─────────────────────────────────────────────────────
        self.group_names = sorted(list(self.groups.keys()))
        self.group_local_ids: Dict[str, np.ndarray] = {
            g: np.array([self.team_idx[t] for t in self.groups[g]], dtype=np.int32)
            for g in self.group_names
        }

        # ── FIFA 3rd-Place Qualification Mapping ───────────────────────────────
        # FIFA dictates which group winner plays which 3rd-placed team based on
        # exactly which 8 groups produce the advancing 3rd-placed teams.
        # Group indices: 0:A, 1:B, 2:C, 3:D, 4:E, 5:F, 6:G, 7:H, 8:I, 9:J, 10:K, 11:L
        
        self.t3_hosts = [0, 1, 3, 4, 6, 8, 10, 11]  # The group winners who play 3rd-placed teams
        t3_allowed = {
            0:  {2, 4, 5, 7, 8},           # 1A plays 3rd from C/E/F/H/I
            1:  {4, 5, 6, 8, 9},           # 1B plays 3rd from E/F/G/I/J
            3:  {1, 4, 5, 8, 9},           # 1D plays 3rd from B/E/F/I/J
            4:  {0, 1, 2, 3, 5},           # 1E plays 3rd from A/B/C/D/F
            6:  {0, 4, 7, 8, 9},           # 1G plays 3rd from A/E/H/I/J
            8:  {2, 3, 5, 6, 7},           # 1I plays 3rd from C/D/F/G/H
            10: {3, 4, 8, 9, 11},          # 1K plays 3rd from D/E/I/J/L
            11: {4, 7, 8, 9, 10},          # 1L plays 3rd from E/H/I/J/K
        }

        # Build a O(1) lookup matrix mapping a 12-bit integer (representing the 8 advancing groups)
        # to an array of 8 assigned group indices ordered to match self.t3_hosts.
        self.t3_lookup = np.zeros((4096, 8), dtype=np.int32)
        
        for combo in itertools.combinations(range(12), 8):
            mask = sum(1 << g for g in combo)
            
            # Simple DFS to find the valid matching for this 8-team combination
            def find_matching(host_idx, current_matching, available_groups):
                if host_idx == 8:
                    return current_matching
                host = self.t3_hosts[host_idx]
                for g in available_groups:
                    if g in t3_allowed[host]:
                        res = find_matching(
                            host_idx + 1, 
                            current_matching + [g], 
                            available_groups - {g}
                        )
                        if res is not None:
                            return res
                return None
            
            matching = find_matching(0, [], set(combo))
            if matching is None:
                matching = list(combo) # Fallback, theoretically unreachable based on FIFA sets
            self.t3_lookup[mask] = matching

        log.info("TournamentSimulator ready: %d teams, %d groups", N, len(self.groups))

    # ── Group stage ────────────────────────────────────────────────────────────

    def _sim_group_stage(
        self, rng: np.random.Generator, n: int
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Simulate all 72 group matches for n tournaments simultaneously.

        Returns
        -------
        {group_name: {pts, gd, gs, ids}}
          pts : (n, 4) int16  — points per team per sim
          gd  : (n, 4) int16  — goal-difference proxy
          gs  : (n, 4) int16  — goals-scored proxy (for 3rd tiebreaker)
          ids : (4,)  int32   — local team IDs in group order
        """
        tables = {}
        for g, ids in self.group_local_ids.items():
            pts = np.zeros((n, TEAMS_PER_GROUP), dtype=np.int16)
            gd  = np.zeros((n, TEAMS_PER_GROUP), dtype=np.int16)
            gs  = np.zeros((n, TEAMS_PER_GROUP), dtype=np.int16)

            for r in range(TEAMS_PER_GROUP):
                for c in range(r + 1, TEAMS_PER_GROUP):
                    gi, gj = int(ids[r]), int(ids[c])
                    u   = rng.random(n, dtype=np.float32)
                    p_w = float(self.P_win_gs[gi, gj])
                    p_d = float(self.P_draw_gs[gi, gj])

                    i_wins = u < p_w
                    draws  = (u >= p_w) & (u < p_w + p_d)
                    j_wins = ~i_wins & ~draws

                    pts[:, r] += (3 * i_wins + draws).astype(np.int16)
                    pts[:, c] += (3 * j_wins + draws).astype(np.int16)
                    gd[:, r]  += i_wins.astype(np.int16) - j_wins.astype(np.int16)
                    gd[:, c]  += j_wins.astype(np.int16) - i_wins.astype(np.int16)

                    # Synthetic goal counts for tiebreaking only (not for display)
                    # Drawn from Poisson with mean tuned to realistic WC scoring
                    # Using integers 0-4 uniformly as a simple, fast proxy
                    gr = rng.integers(0, 5, n, dtype=np.int16)
                    gc = rng.integers(0, 5, n, dtype=np.int16)
                    gs[:, r] += gr
                    gs[:, c] += gc

            tables[g] = {"pts": pts, "gd": gd, "gs": gs, "ids": ids}
        return tables

    @staticmethod
    def _rank_group(
        pts: np.ndarray,  # (n, 4)
        gd:  np.ndarray,  # (n, 4)
        gs:  np.ndarray,  # (n, 4)
        ids: np.ndarray,  # (4,)
    ) -> np.ndarray:
        """
        Rank 4 teams for n simulations.
        Primary: pts DESC → gd DESC → gs DESC (FIFA tiebreaker order).
        Returns ranked_local_ids of shape (n, 4): col 0 = winner.
        """
        # numpy lexsort: last key is primary — so we reverse the desired order
        order = np.lexsort((-gs, -gd, -pts), axis=1)   # (n, 4) position indices
        return ids[order]                                # (n, 4) local team IDs

    # ── Knockout round ─────────────────────────────────────────────────────────

    def _knockout_round(
        self,
        rng: np.random.Generator,
        matchups: np.ndarray,   # (n, 2*k) — interleaved [a0, b0, a1, b1, …]
    ) -> np.ndarray:
        """
        Simulate one knockout round for all n simulations.
        Uses the P_ko (draw-redistributed) matrix for clean winners.

        Returns winners: (n, k) array of local team IDs.
        """
        n = matchups.shape[0]
        k = matchups.shape[1] // 2
        winners = np.empty((n, k), dtype=np.int32)
        for m in range(k):
            a = matchups[:, 2 * m]
            b = matchups[:, 2 * m + 1]
            p_a = self.P_ko[a, b]
            u   = rng.random(n, dtype=np.float32)
            winners[:, m] = np.where(u < p_a, a, b)
        return winners

    # ── Main run ───────────────────────────────────────────────────────────────

    def run(self, n_sims: int = N_SIMULATIONS) -> SimulationResults:
        """Execute n_sims complete WC2026 tournament simulations."""
        rng   = np.random.default_rng(RANDOM_SEED)
        start = time.perf_counter()
        log.info("Starting %d Monte Carlo simulations…", n_sims)

        N = self.n_teams
        champ_cnt   = np.zeros(N, dtype=np.int32)
        final_cnt   = np.zeros(N, dtype=np.int32)
        sf_cnt      = np.zeros(N, dtype=np.int32)
        qf_cnt      = np.zeros(N, dtype=np.int32)
        r16_cnt     = np.zeros(N, dtype=np.int32)
        r32_cnt     = np.zeros(N, dtype=np.int32)
        group_x_cnt = np.zeros(N, dtype=np.int32)

        # ── A. Group Stage ─────────────────────────────────────────────────────
        tables = self._sim_group_stage(rng, n_sims)

        winners_g: Dict[str, np.ndarray] = {}
        runners_g: Dict[str, np.ndarray] = {}

        # Collect third-place data across all 12 groups
        t3_pts_cols: list = []
        t3_gd_cols:  list = []
        t3_gs_cols:  list = []
        t3_id_cols:  list = []

        # Iterate in strict alphabetical order to maintain stable group indices
        for g in self.group_names:
            data = tables[g]
            ranked = self._rank_group(data["pts"], data["gd"], data["gs"], data["ids"])
            # ranked: (n_sims, 4) — local team IDs, col 0 = winner
            winners_g[g] = ranked[:, 0]   # (n_sims,)
            runners_g[g] = ranked[:, 1]

            # Extract 3rd-place stats: find position index of 3rd-place team
            order_idx = np.lexsort(
                (-data["gs"], -data["gd"], -data["pts"]), axis=1
            )  # (n, 4) — indices into teams within group
            t3_pts_cols.append(data["pts"][np.arange(n_sims), order_idx[:, 2]])
            t3_gd_cols.append( data["gd"][ np.arange(n_sims), order_idx[:, 2]])
            t3_gs_cols.append( data["gs"][ np.arange(n_sims), order_idx[:, 2]])
            t3_id_cols.append(ranked[:, 2])

            # 4th-place team exits in group stage
            np.add.at(group_x_cnt, ranked[:, 3], 1)

        # ── B. Best-8 Third-Place ──────────────────────────────────────────────
        all_t3_pts = np.column_stack(t3_pts_cols)   # (n_sims, 12)
        all_t3_gd  = np.column_stack(t3_gd_cols)
        all_t3_gs  = np.column_stack(t3_gs_cols)
        all_t3_ids = np.column_stack(t3_id_cols)

        # Rank all 12 thirds per sim by pts→gd→gs (descending)
        t3_order = np.lexsort((-all_t3_gs, -all_t3_gd, -all_t3_pts), axis=1)  # (n, 12)
        worst4_ids = all_t3_ids[np.arange(n_sims)[:, None], t3_order[:, N_BEST_THIRD:]]

        # Worst 4 thirds exit at group stage
        for col in range(4):
            np.add.at(group_x_cnt, worst4_ids[:, col], 1)

        # Map 8 best 3rd-place teams to their round of 32 opponents
        chosen_groups = t3_order[:, :N_BEST_THIRD]  # (n_sims, 8) — group indices (0-11)
        masks = np.sum(1 << chosen_groups, axis=1)  # (n_sims,)
        mapped_group_indices = self.t3_lookup[masks] # (n_sims, 8)
        
        sim_idx = np.arange(n_sims)[:, None]
        t3_assigned_ids = all_t3_ids[sim_idx, mapped_group_indices] # (n_sims, 8)

        # ── C. Build R32 Bracket (32 teams, 16 matches) ────────────────────────
        # Bracket matches ordered such that consecutive pairs in R32 output
        # properly feed into R16 -> QF -> SF in a clean binary tree structure.
        # Tree pairings:
        # (74,77)->89, (73,75)->90  => 89,90 -> 97 (SF1)
        # (83,84)->93, (81,82)->94  => 93,94 -> 98 (SF1)
        # (76,78)->91, (79,80)->92  => 91,92 -> 99 (SF2)
        # (86,88)->95, (85,87)->96  => 95,96 -> 100 (SF2)

        r32 = np.empty((n_sims, 32), dtype=np.int32)
        
        # M74: 1E vs 3rd(E)
        r32[:, 0] = winners_g["E"]; r32[:, 1] = t3_assigned_ids[:, 3]
        # M77: 1I vs 3rd(I)
        r32[:, 2] = winners_g["I"]; r32[:, 3] = t3_assigned_ids[:, 5]
        
        # M73: 2A vs 2B
        r32[:, 4] = runners_g["A"]; r32[:, 5] = runners_g["B"]
        # M75: 1F vs 2C
        r32[:, 6] = winners_g["F"]; r32[:, 7] = runners_g["C"]
        
        # M83: 2K vs 2L
        r32[:, 8] = runners_g["K"]; r32[:, 9] = runners_g["L"]
        # M84: 1H vs 2J
        r32[:, 10] = winners_g["H"]; r32[:, 11] = runners_g["J"]
        
        # M81: 1D vs 3rd(D)
        r32[:, 12] = winners_g["D"]; r32[:, 13] = t3_assigned_ids[:, 2]
        # M82: 1G vs 3rd(G)
        r32[:, 14] = winners_g["G"]; r32[:, 15] = t3_assigned_ids[:, 4]
        
        # M76: 1C vs 2F
        r32[:, 16] = winners_g["C"]; r32[:, 17] = runners_g["F"]
        # M78: 2E vs 2I
        r32[:, 18] = runners_g["E"]; r32[:, 19] = runners_g["I"]
        
        # M79: 1A vs 3rd(A)
        r32[:, 20] = winners_g["A"]; r32[:, 21] = t3_assigned_ids[:, 0]
        # M80: 1L vs 3rd(L)
        r32[:, 22] = winners_g["L"]; r32[:, 23] = t3_assigned_ids[:, 7]
        
        # M86: 1J vs 2H
        r32[:, 24] = winners_g["J"]; r32[:, 25] = runners_g["H"]
        # M88: 2D vs 2G
        r32[:, 26] = runners_g["D"]; r32[:, 27] = runners_g["G"]
        
        # M85: 1B vs 3rd(B)
        r32[:, 28] = winners_g["B"]; r32[:, 29] = t3_assigned_ids[:, 1]
        # M87: 1K vs 3rd(K)
        r32[:, 30] = winners_g["K"]; r32[:, 31] = t3_assigned_ids[:, 6]

        for col in range(32):
            np.add.at(r32_cnt, r32[:, col], 1)

        # ── D. R32 → R16 (16 matches → 16 winners) ────────────────────────────
        r32_w = self._knockout_round(rng, r32)              # (n, 16)
        for col in range(16):
            np.add.at(r16_cnt, r32_w[:, col], 1)

        # ── E. R16 → QF (8 matches → 8 winners) ──────────────────────────────
        r16 = np.empty((n_sims, 16), dtype=np.int32)
        for k in range(8):
            r16[:, 2*k]   = r32_w[:, 2*k]
            r16[:, 2*k+1] = r32_w[:, 2*k+1]
        r16_w = self._knockout_round(rng, r16)              # (n, 8)
        for col in range(8):
            np.add.at(qf_cnt, r16_w[:, col], 1)

        # ── F. QF → SF (4 matches → 4 winners) ────────────────────────────────
        qf = np.empty((n_sims, 8), dtype=np.int32)
        for k in range(4):
            qf[:, 2*k]   = r16_w[:, 2*k]
            qf[:, 2*k+1] = r16_w[:, 2*k+1]
        qf_w = self._knockout_round(rng, qf)                # (n, 4)
        for col in range(4):
            np.add.at(sf_cnt, qf_w[:, col], 1)

        # ── G. SF → Final (2 matches → 2 winners) ─────────────────────────────
        sf = np.empty((n_sims, 4), dtype=np.int32)
        sf[:, 0] = qf_w[:, 0]; sf[:, 1] = qf_w[:, 1]
        sf[:, 2] = qf_w[:, 2]; sf[:, 3] = qf_w[:, 3]
        sf_w = self._knockout_round(rng, sf)                 # (n, 2)
        for col in range(2):
            np.add.at(final_cnt, sf_w[:, col], 1)

        # ── H. Final (1 match → 1 champion) ───────────────────────────────────
        final = np.empty((n_sims, 2), dtype=np.int32)
        final[:, 0] = sf_w[:, 0]
        final[:, 1] = sf_w[:, 1]
        champs = self._knockout_round(rng, final)            # (n, 1)
        np.add.at(champ_cnt, champs[:, 0], 1)

        elapsed = time.perf_counter() - start
        log.info("[DONE] %d simulations in %.2fs  (%.0f sims/sec)",
                 n_sims, elapsed, n_sims / elapsed)

        return SimulationResults(
            n_sims              = n_sims,
            teams               = self.teams,
            champion_counts     = champ_cnt,
            finalist_counts     = final_cnt,
            semifinal_counts    = sf_cnt,
            quarterfinal_counts = qf_cnt,
            r16_counts          = r16_cnt,
            r32_counts          = r32_cnt,
            group_exit_counts   = group_x_cnt,
            elapsed_s           = elapsed,
        )