"""
eval_poisson_goals.py — Phase 1 Benchmark: Poisson Goal-Scoring vs Current Simulator

Evaluates whether replacing the current binary ±1 GD + uniform-random goals scorer
with a Poisson goal-scoring model materially improves historical World Cup realism.

What changes between the two simulators
─────────────────────────────────────────
Current (BINARY):
  • Match outcome (W/D/L) drawn from probability matrix                [unchanged]
  • Goal difference proxy: +1 per win, -1 per loss                    [synthetic]
  • Goals scored proxy:    uniform integer [0, 4] per team per match   [synthetic]

Poisson (NEW):
  • Goals for each team sampled from Poisson(λ) where:               [realistic]
      λ_h = 2 × μ_wc × P_win_h / (P_win_h + P_win_a)
      λ_a = 2 × μ_wc × P_win_a / (P_win_h + P_win_a)
      μ_wc = historical WC average goals per team per match (calibrated)
  • Outcome (W/D/L) derived directly from sampled goals               [realistic]
  • GD = actual goal difference from sampled goals                    [realistic]
  • GS = actual goals scored                                          [realistic]
  NOTE: knockout rounds are UNCHANGED in both — only group stage differs.

Evaluation metrics
──────────────────
1. Group advancement accuracy     — top-2 per group vs historical WC fact
2. Knockout qualification accuracy — champion, SF, QF, R16 stage Brier scores
3. Champion probability calibration — champion's simulated win probability
4. Goal distribution statistics    — mean goals/match, draw rate, GD distribution
5. Standings disagreement rate     — % of sims where top-2 changes vs current
6. Computational overhead          — wall-clock time ratio (Poisson / Binary)

Gate criterion (ADOPT if any one is met)
─────────────────────────────────────────
  Mean advancement Brier delta ≤ −0.005  (lower is better)
  OR mean group advancement accuracy delta ≥ +0.025
  OR draw rate closer to historical WC rate (22 ± 3%) by >5 percentage points

PRODUCTION CODE IS NOT MODIFIED BY THIS SCRIPT.

Usage:
    python scripts/eval_poisson_goals.py [--sims N] [--bt-sims N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import RANDOM_SEED, REPORT_DIR
from core.logger import get_logger
from backtesting.world_cup_backtest import (
    HISTORICAL_WORLD_CUPS,
    HistoricalSimulationResults,
    HistoricalTournamentSimulator,
    _advancement_brier,
    _build_pre_tournament_state,
    _stage_top_accuracy,
)
from data.ingestion import load_raw_results
from models.ml_engine import MatchOutcomeModel, ProbabilityMatrixBuilder
from scripts.eval_mov_elo import _build_feature_dataset   # reuse dataset builder

log = get_logger("eval_poisson_goals")

REPORT_PATH      = REPORT_DIR / "poisson_goals_benchmark.md"
REPORT_JSON_PATH = REPORT_DIR / "poisson_goals_benchmark.json"

# Historical WC average goals per team per match (calibrated from 2010-2022 WC data)
# Source: FIFA match records — computed below at runtime from results.csv
_WC_GOALS_FALLBACK = 1.36   # fallback if calibration fails


# ══════════════════════════════════════════════════════════════════════════════
# λ Calibration from historical WC data
# ══════════════════════════════════════════════════════════════════════════════

def calibrate_wc_goals_per_team(cutoff_date: Optional[str] = None) -> float:
    """
    Compute the historical WC average goals per team per match from results.csv.
    Optionally restrict to matches before cutoff_date for OOS use.
    """
    raw = load_raw_results()
    wc = raw[raw["tournament"] == "FIFA World Cup"].copy()
    if cutoff_date:
        wc = wc[wc["date"] < pd.Timestamp(cutoff_date)]
    if wc.empty:
        return _WC_GOALS_FALLBACK
    avg = float((wc["home_score"].sum() + wc["away_score"].sum()) / (2 * len(wc)))
    log.info("Calibrated WC goals/team/match = %.4f  (n=%d matches)", avg, len(wc))
    return avg


# ══════════════════════════════════════════════════════════════════════════════
# Poisson Group-Stage Simulator (subclass — no production code modified)
# ══════════════════════════════════════════════════════════════════════════════

class PoissonGroupStageSimulator(HistoricalTournamentSimulator):
    """
    Extends HistoricalTournamentSimulator by replacing the binary ±1 GD + uniform
    goals scorer with Poisson-distributed goals derived from the probability matrix.

    Only _sim_group_stage() is overridden.  All knockout logic is inherited unchanged.

    λ calibration
    ─────────────
    For a match between team i and team j:
        total_goals = 2 × μ_wc           (preserves historical scoring rate)
        strength_h  = P_win[i,j] / (P_win[i,j] + P_win[j,i])
        λ_h = total_goals × strength_h
        λ_a = total_goals × (1 − strength_h)

    The draw probability from the probability matrix is NOT used to set λ —
    only the relative win probabilities drive the λ ratio.  This is intentional:
    allowing λ_h ≈ λ_a for evenly matched teams naturally produces ~25% draws
    via Poisson coincidence, which aligns with historical WC draw rates.
    """

    def __init__(
        self,
        prob_matrix_builder: ProbabilityMatrixBuilder,
        groups: Dict[str, List[str]],
        wc_goals_per_team: float = _WC_GOALS_FALLBACK,
    ) -> None:
        super().__init__(prob_matrix_builder, groups)
        self.mu = wc_goals_per_team
        self._build_lambda_matrices()

    def _build_lambda_matrices(self) -> None:
        """Pre-compute the λ matrix for all ordered team pairs."""
        n = len(self.teams)
        eps = 1e-9
        self.lambda_h = np.zeros((n, n), dtype=np.float32)
        self.lambda_a = np.zeros((n, n), dtype=np.float32)
        total = 2.0 * self.mu

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                p_win_ij = float(self.p_win[i, j])
                p_win_ji = float(self.p_win[j, i])
                denom = p_win_ij + p_win_ji
                if denom < eps:
                    # If both probabilities are near-zero (shouldn't happen), split evenly
                    self.lambda_h[i, j] = total / 2.0
                    self.lambda_a[i, j] = total / 2.0
                else:
                    strength_h = p_win_ij / denom
                    self.lambda_h[i, j] = total * strength_h
                    self.lambda_a[i, j] = total * (1.0 - strength_h)

    def _sim_group_stage(
        self,
        rng: np.random.Generator,
        n_sims: int,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Poisson goal-scoring group stage.

        For each match:
          goals_h ~ Poisson(λ_h[i,j])
          goals_a ~ Poisson(λ_a[i,j])
          outcome = sign(goals_h - goals_a)
          GD and GS are the actual sampled values (not synthetic proxies).
        """
        tables: Dict[str, Dict[str, np.ndarray]] = {}
        for group, ids in self.group_local_ids.items():
            pts = np.zeros((n_sims, 4), dtype=np.int16)
            gd  = np.zeros((n_sims, 4), dtype=np.int16)
            gs  = np.zeros((n_sims, 4), dtype=np.int16)

            for home_pos in range(4):
                for away_pos in range(home_pos + 1, 4):
                    home_id = int(ids[home_pos])
                    away_id = int(ids[away_pos])

                    lh = float(self.lambda_h[home_id, away_id])
                    la = float(self.lambda_a[home_id, away_id])

                    # Sample goals — vectorised across all n_sims at once
                    goals_h = rng.poisson(lh, n_sims).astype(np.int16)
                    goals_a = rng.poisson(la, n_sims).astype(np.int16)

                    home_win = goals_h > goals_a
                    draw     = goals_h == goals_a
                    away_win = goals_h < goals_a

                    pts[:, home_pos] += (3 * home_win + draw).astype(np.int16)
                    pts[:, away_pos] += (3 * away_win + draw).astype(np.int16)

                    # Real GD: actual simulated goal difference
                    match_gd = (goals_h - goals_a).astype(np.int16)
                    gd[:, home_pos] += match_gd
                    gd[:, away_pos] -= match_gd

                    # Real GS: actual simulated goals scored
                    gs[:, home_pos] += goals_h
                    gs[:, away_pos] += goals_a

            tables[group] = {"pts": pts, "gd": gd, "gs": gs, "ids": ids}
        return tables


# ══════════════════════════════════════════════════════════════════════════════
# Disagreement Rate Measurement
# ══════════════════════════════════════════════════════════════════════════════

def _measure_disagreement(
    binary_sim: HistoricalTournamentSimulator,
    poisson_sim: PoissonGroupStageSimulator,
    n_sims: int,
    seed: int = RANDOM_SEED,
) -> Dict[str, float]:
    """
    Run both simulators with the same seed and compare group stage outcomes.

    Returns:
      top2_disagreement_rate: fraction of (sim, group) pairs where top-2 differs
      top1_disagreement_rate: fraction of (sim, group) pairs where winner differs
      draw_rate_binary:  observed draw rate in binary sim's group stage
      draw_rate_poisson: observed draw rate in Poisson sim's group stage
      mean_goals_per_match: observed in Poisson sim
      mean_gd_abs_binary:  mean |GD| in binary sim (should be ~1.0 by design)
      mean_gd_abs_poisson: mean |GD| in Poisson sim (should be ~1.2-1.6)
    """
    log.info("Measuring disagreement rate (%d sims)...", n_sims)
    rng_b = np.random.default_rng(seed)
    rng_p = np.random.default_rng(seed)   # same seed → same match ordering

    tables_b = binary_sim._sim_group_stage(rng_b, n_sims)
    tables_p = poisson_sim._sim_group_stage(rng_p, n_sims)

    n_groups     = len(binary_sim.group_names)
    top2_disagree = 0
    top1_disagree = 0
    total_matches = 0  # total match-pair observations

    draw_count_b = 0
    draw_count_p = 0
    goals_per_match_p: List[float] = []
    abs_gd_b: List[float] = []
    abs_gd_p: List[float] = []

    for group in binary_sim.group_names:
        ids = binary_sim.group_local_ids[group]
        n   = 4  # teams per group

        # Rank both tables
        ranked_b = HistoricalTournamentSimulator._rank_group(
            tables_b[group]["pts"],
            tables_b[group]["gd"],
            tables_b[group]["gs"],
            ids,
        )  # (n_sims, 4)
        ranked_p = HistoricalTournamentSimulator._rank_group(
            tables_p[group]["pts"],
            tables_p[group]["gd"],
            tables_p[group]["gs"],
            ids,
        )

        top2_b = set_rows(ranked_b[:, :2])   # (n_sims, 2) — sorted set per row
        top2_p = set_rows(ranked_p[:, :2])
        top2_disagree += int(np.sum(top2_b != top2_p))

        top1_disagree += int(np.sum(ranked_b[:, 0] != ranked_p[:, 0]))

        # Draw rate: count matches where draw occurred in this group
        pts_b = tables_b[group]["pts"]   # (n_sims, 4)
        pts_p = tables_p[group]["pts"]
        gd_b  = tables_b[group]["gd"]
        gd_p  = tables_p[group]["gd"]
        gs_p  = tables_p[group]["gs"]

        for home_pos in range(4):
            for away_pos in range(home_pos + 1, 4):
                total_matches += n_sims
                # Draw detection: pts delta = 1 each → draw
                # In binary: draw adds 1 to both → check gd unchanged (0 net for both)
                # In Poisson: goals_h == goals_a → net gd contribution = 0 for this match
                # Simpler: reconstruct from pts difference per pair is complex.
                # Instead, use the simulated gd columns — a draw contributes 0 to gd.
                # Not trivial to decompose. Track directly in a separate pass.

        # Track GD statistics from table columns
        abs_gd_b.extend(np.abs(tables_b[group]["gd"]).ravel().tolist())
        abs_gd_p.extend(np.abs(tables_p[group]["gd"]).ravel().tolist())

    # Run a SMALL separate pass to count draws and goals (more tractable)
    draw_count_b, draw_count_p, total_match_obs, goals_sum_p, goals_count_p = \
        _count_draws_goals(binary_sim, poisson_sim, n_sims, seed)

    mean_goals_pm = goals_sum_p / max(goals_count_p, 1)
    draw_rate_b   = draw_count_b / max(total_match_obs, 1)
    draw_rate_p   = draw_count_p / max(total_match_obs, 1)

    total_group_sims = n_sims * n_groups
    result = {
        "top2_disagreement_rate": top2_disagree / total_group_sims,
        "top1_disagreement_rate": top1_disagree / total_group_sims,
        "draw_rate_binary":       draw_rate_b,
        "draw_rate_poisson":      draw_rate_p,
        "mean_goals_per_match":   mean_goals_pm,
        "mean_abs_gd_binary":     float(np.mean(abs_gd_b)) if abs_gd_b else 0.0,
        "mean_abs_gd_poisson":    float(np.mean(abs_gd_p)) if abs_gd_p else 0.0,
    }
    log.info(
        "Disagreement — top-2: %.2f%%  top-1: %.2f%%  "
        "draw_rate: binary=%.2f%% poisson=%.2f%%  "
        "goals/match: %.3f  |GD| binary: %.3f  poisson: %.3f",
        result["top2_disagreement_rate"] * 100,
        result["top1_disagreement_rate"] * 100,
        result["draw_rate_binary"] * 100,
        result["draw_rate_poisson"] * 100,
        result["mean_goals_per_match"],
        result["mean_abs_gd_binary"],
        result["mean_abs_gd_poisson"],
    )
    return result


def set_rows(arr: np.ndarray) -> np.ndarray:
    """Sort each row so (A,B) == (B,A) for comparison."""
    return np.sort(arr, axis=1)


def _count_draws_goals(
    binary_sim: HistoricalTournamentSimulator,
    poisson_sim: PoissonGroupStageSimulator,
    n_sims: int,
    seed: int,
) -> Tuple[int, int, int, float, int]:
    """Count draw occurrences and goal totals per match by re-running match pairs."""
    rng_b = np.random.default_rng(seed + 999_001)
    rng_p = np.random.default_rng(seed + 999_001)

    draw_b = 0
    draw_p = 0
    total  = 0
    goals_p = 0.0
    goals_count = 0

    for group, ids in binary_sim.group_local_ids.items():
        for hp in range(4):
            for ap in range(hp + 1, 4):
                hi = int(ids[hp])
                ai = int(ids[ap])

                # Binary
                u   = rng_b.random(n_sims, dtype=np.float32)
                p_w = float(binary_sim.p_win[hi, ai])
                p_d = float(binary_sim.p_draw[hi, ai])
                b_draw = ((u >= p_w) & (u < p_w + p_d)).sum()
                draw_b += int(b_draw)

                # Poisson
                lh = float(poisson_sim.lambda_h[hi, ai])
                la = float(poisson_sim.lambda_a[hi, ai])
                gh = rng_p.poisson(lh, n_sims).astype(np.int16)
                ga = rng_p.poisson(la, n_sims).astype(np.int16)
                draw_p  += int((gh == ga).sum())
                goals_p += float(gh.sum() + ga.sum())
                goals_count += n_sims  # per match, both teams

                total += n_sims

    return draw_b, draw_p, total, goals_p, goals_count


# ══════════════════════════════════════════════════════════════════════════════
# Historical Backtest Runner
# ══════════════════════════════════════════════════════════════════════════════

def _run_historical_backtest(
    use_poisson: bool,
    n_sims: int,
    label: str,
) -> Dict:
    """
    OOS backtest on WC 2014, 2018, 2022.
    Builds model from pre-tournament data, runs n_sims, measures accuracy metrics.
    """
    results_per_year = []

    for tournament in HISTORICAL_WORLD_CUPS:
        log.info("[%s] Backtest WC%d (cutoff %s)...", label, tournament.year, tournament.start_date)
        t0 = time.perf_counter()

        # Build pre-tournament state (OOS — only data before tournament.start_date)
        elo, form_scores, opp_adj_form, form_gd, h2h, form_details = \
            _build_pre_tournament_state(tournament.start_date)

        teams = [team for group in tournament.groups.values() for team in group]

        # Train ML model on pre-tournament data
        X_bt, y_bt, _, _, _, _ = _build_feature_dataset(
            cutoff_date=tournament.start_date, use_mov=False
        )
        model_bt = MatchOutcomeModel()
        model_bt.fit(X_bt, y_bt)

        pmb = ProbabilityMatrixBuilder(model_bt)
        pmb.build(
            teams               = teams,
            elo_tracker         = elo,
            form_scores         = form_scores,
            opp_adj_form_scores = opp_adj_form,
            form_gd             = form_gd,
            h2h_fn              = h2h.win_rate_diff,
            form_details        = form_details,
        )

        # Calibrate μ from WC history BEFORE this tournament (strict OOS)
        mu = calibrate_wc_goals_per_team(cutoff_date=tournament.start_date)

        if use_poisson:
            sim = PoissonGroupStageSimulator(pmb, tournament.groups, wc_goals_per_team=mu)
        else:
            sim = HistoricalTournamentSimulator(pmb, tournament.groups)

        res = sim.run(n_sims=n_sims)
        res_df = res.to_dataframe()

        actual = tournament.actual
        champ  = str(actual["champion"])
        gq     = list(actual["group_qualifiers"])
        qf     = list(actual["quarterfinalists"])
        sf     = list(actual["semifinalists"])

        champ_prob = float(res_df.loc[res_df["team"] == champ, "champion_pct"].values[0])
        grp_acc  = _stage_top_accuracy(res_df, "round_of_16_pct", gq, len(gq))
        qf_acc   = _stage_top_accuracy(res_df, "quarterfinal_pct", qf, len(qf))
        sf_acc   = _stage_top_accuracy(res_df, "semifinal_pct", sf, len(sf))

        g_brier  = _advancement_brier(res_df.set_index("team")["round_of_16_pct"], gq)
        qf_brier = _advancement_brier(res_df.set_index("team")["quarterfinal_pct"], qf)
        sf_brier = _advancement_brier(res_df.set_index("team")["semifinal_pct"], sf)
        ch_brier = _advancement_brier(res_df.set_index("team")["champion_pct"], [champ])
        adv_brier = float(np.mean([g_brier, qf_brier, sf_brier, ch_brier]))

        runtime = time.perf_counter() - t0
        log.info(
            "[%s] WC%d — champ=%.2f%%  grp=%.3f  qf=%.3f  sf=%.3f  brier=%.4f  (%.1fs)",
            label, tournament.year, champ_prob, grp_acc, qf_acc, sf_acc, adv_brier, runtime,
        )
        results_per_year.append({
            "year":        tournament.year,
            "champion":    champ,
            "champ_pct":   champ_prob,
            "grp_acc":     grp_acc,
            "qf_acc":      qf_acc,
            "sf_acc":      sf_acc,
            "g_brier":     g_brier,
            "qf_brier":    qf_brier,
            "sf_brier":    sf_brier,
            "ch_brier":    ch_brier,
            "adv_brier":   adv_brier,
            "runtime_s":   runtime,
        })

    return {
        "per_year":          results_per_year,
        "mean_grp_acc":      float(np.mean([r["grp_acc"]    for r in results_per_year])),
        "mean_qf_acc":       float(np.mean([r["qf_acc"]     for r in results_per_year])),
        "mean_sf_acc":       float(np.mean([r["sf_acc"]     for r in results_per_year])),
        "mean_adv_brier":    float(np.mean([r["adv_brier"]  for r in results_per_year])),
        "mean_champ_pct":    float(np.mean([r["champ_pct"]  for r in results_per_year])),
        "total_runtime_s":   float(sum(r["runtime_s"]        for r in results_per_year)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Computational overhead measurement
# ══════════════════════════════════════════════════════════════════════════════

def _measure_overhead(
    binary_sim: HistoricalTournamentSimulator,
    poisson_sim: PoissonGroupStageSimulator,
    n_sims: int,
    n_repeats: int = 3,
) -> Dict[str, float]:
    """Time both simulators over n_repeats runs and report mean wall-clock time."""
    log.info("Measuring computational overhead (%d sims × %d repeats)...", n_sims, n_repeats)

    times_b = []
    times_p = []

    for _ in range(n_repeats):
        t0 = time.perf_counter()
        binary_sim.run(n_sims=n_sims)
        times_b.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        poisson_sim.run(n_sims=n_sims)
        times_p.append(time.perf_counter() - t0)

    mean_b = float(np.mean(times_b))
    mean_p = float(np.mean(times_p))
    ratio  = mean_p / max(mean_b, 1e-6)

    log.info(
        "Overhead — Binary: %.3fs  Poisson: %.3fs  Ratio: %.2f×",
        mean_b, mean_p, ratio,
    )
    return {
        "binary_s":  mean_b,
        "poisson_s": mean_p,
        "overhead_ratio": ratio,
        "n_sims":    n_sims,
        "n_repeats": n_repeats,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Goal distribution statistics from actual WC data
# ══════════════════════════════════════════════════════════════════════════════

def _get_historical_wc_stats() -> Dict[str, float]:
    """Compute reference statistics from historical FIFA World Cup matches."""
    raw = load_raw_results()
    wc  = raw[raw["tournament"] == "FIFA World Cup"].copy()
    if wc.empty:
        return {"draw_rate": 0.224, "goals_per_match": 2.72, "mean_gd": 1.31}

    total_matches = len(wc)
    draws         = (wc["home_score"] == wc["away_score"]).sum()
    total_goals   = (wc["home_score"] + wc["away_score"]).sum()
    abs_gd        = (wc["home_score"] - wc["away_score"]).abs()

    return {
        "draw_rate":        float(draws / total_matches),
        "goals_per_match":  float(total_goals / total_matches),
        "mean_abs_gd":      float(abs_gd[abs_gd > 0].mean()),   # average non-draw GD
        "total_matches":    int(total_matches),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Report builder
# ══════════════════════════════════════════════════════════════════════════════

def _dlt(new: float, old: float, lower_is_better: bool = True) -> str:
    d = new - old
    sign = "+" if d > 0 else ""
    if lower_is_better and d < -0.0005:
        return f"{sign}{d:.5f} ✓"
    if not lower_is_better and d > 0.0005:
        return f"{sign}{d:.5f} ✓"
    return f"{sign}{d:.5f}"


def _build_report(
    b: Dict,
    p: Dict,
    disagreement: Dict,
    overhead: Dict,
    hist_stats: Dict,
    gate_passed: bool,
    recommendation: str,
    wc_mu: float,
) -> str:
    hist_draw_pct = hist_stats["draw_rate"] * 100
    hist_goals    = hist_stats["goals_per_match"]

    lines = [
        "# Poisson Goal-Scoring — Phase 1 Benchmark Report",
        "",
        f"> **Historical WC draw rate**: {hist_draw_pct:.1f}%  "
        f"|  **Historical WC goals/match**: {hist_goals:.2f}  "
        f"|  **λ calibration μ**: {wc_mu:.4f} goals/team/match",
        f"> **Backtest**: WC 2014, 2018, 2022 (OOS)  "
        f"|  **Gate**: {'PASSED ✓' if gate_passed else 'FAILED ✗'}",
        "",
        "---",
        "",
        "## 1. Goal Distribution Statistics",
        "",
        "How well does each simulator reproduce historical WC goal-scoring patterns?",
        "",
        "| Metric | Historical WC | Binary Sim | Poisson Sim | Δ (vs historical) |",
        "|:---|---:|---:|---:|---:|",
        f"| Draw rate | {hist_draw_pct:.2f}% | {disagreement['draw_rate_binary']*100:.2f}% | "
        f"{disagreement['draw_rate_poisson']*100:.2f}% | "
        f"Binary: {(disagreement['draw_rate_binary']-hist_stats['draw_rate'])*100:+.2f}pp  "
        f"Poisson: {(disagreement['draw_rate_poisson']-hist_stats['draw_rate'])*100:+.2f}pp |",
        f"| Goals/match | {hist_goals:.2f} | N/A (uniform) | "
        f"{disagreement['mean_goals_per_match']:.2f} | "
        f"Poisson: {disagreement['mean_goals_per_match']-hist_goals:+.2f} |",
        f"| Mean \\|GD\\| | {hist_stats['mean_abs_gd']:.2f} | "
        f"{disagreement['mean_abs_gd_binary']:.2f} | "
        f"{disagreement['mean_abs_gd_poisson']:.2f} | — |",
        "",
        "---",
        "",
        "## 2. Standings Disagreement Rate",
        "",
        "How often do group standings differ between the two simulators?",
        "",
        "| Metric | Value |",
        "|:---|---:|",
        f"| Top-2 advancement disagreement | {disagreement['top2_disagreement_rate']*100:.2f}% of group-sims |",
        f"| Group winner disagreement | {disagreement['top1_disagreement_rate']*100:.2f}% of group-sims |",
        "",
        "> A higher disagreement rate means the tiebreaker change has material effect on which teams advance.",
        "",
        "---",
        "",
        "## 3. Historical WC Backtest — Accuracy Metrics",
        "",
        "| Year | Champion | Binary Champ% | Poisson Champ% | Binary GrpAcc | Poisson GrpAcc | Binary Brier | Poisson Brier |",
        "|:---|:---|---:|---:|---:|---:|---:|---:|",
    ]

    for i, yr in enumerate(b["per_year"]):
        pr = p["per_year"][i]
        lines.append(
            f"| {yr['year']} | {yr['champion']} | {yr['champ_pct']:.2f}% | {pr['champ_pct']:.2f}% | "
            f"{yr['grp_acc']:.3f} | {pr['grp_acc']:.3f} | "
            f"{yr['adv_brier']:.4f} | {pr['adv_brier']:.4f} |"
        )

    lines += [
        f"| **Mean** | — | {b['mean_champ_pct']:.2f}% | {p['mean_champ_pct']:.2f}% | "
        f"{b['mean_grp_acc']:.3f} | {p['mean_grp_acc']:.3f} | "
        f"**{b['mean_adv_brier']:.4f}** | **{p['mean_adv_brier']:.4f}** |",
        "",
        "### Aggregate Deltas (Poisson − Binary)",
        "",
        "| Metric | Binary | Poisson | Delta | Gate? |",
        "|:---|---:|---:|---:|:---|",
        f"| **Mean Advancement Brier** | {b['mean_adv_brier']:.5f} | {p['mean_adv_brier']:.5f} | "
        f"{_dlt(p['mean_adv_brier'], b['mean_adv_brier'])} | Δ ≤ −0.005 |",
        f"| Mean Group Accuracy | {b['mean_grp_acc']:.5f} | {p['mean_grp_acc']:.5f} | "
        f"{_dlt(p['mean_grp_acc'], b['mean_grp_acc'], lower_is_better=False)} | Δ ≥ +0.025 |",
        f"| Mean QF Accuracy | {b['mean_qf_acc']:.5f} | {p['mean_qf_acc']:.5f} | "
        f"{_dlt(p['mean_qf_acc'], b['mean_qf_acc'], lower_is_better=False)} | — |",
        f"| Mean SF Accuracy | {b['mean_sf_acc']:.5f} | {p['mean_sf_acc']:.5f} | "
        f"{_dlt(p['mean_sf_acc'], b['mean_sf_acc'], lower_is_better=False)} | — |",
        f"| Mean Champion % on actual winner | {b['mean_champ_pct']:.3f} | {p['mean_champ_pct']:.3f} | "
        f"{_dlt(p['mean_champ_pct'], b['mean_champ_pct'], lower_is_better=False)} | — |",
        "",
        "---",
        "",
        "## 4. Computational Overhead",
        "",
        "| Metric | Value |",
        "|:---|---:|",
        f"| Binary simulator mean time | {overhead['binary_s']:.3f}s per {overhead['n_sims']:,} sims |",
        f"| Poisson simulator mean time | {overhead['poisson_s']:.3f}s per {overhead['n_sims']:,} sims |",
        f"| Overhead ratio | **{overhead['overhead_ratio']:.2f}×** slower |",
        "",
        "> Poisson sampling replaces `rng.integers()` with `rng.poisson()`, which is marginally slower.",
        "> An overhead ratio < 2.0 is acceptable; the simulation remains fast.",
        "",
        "---",
        "",
        "## 5. Verdict",
        "",
        f"### {'✅ RECOMMENDATION: ADOPT' if gate_passed else '❌ RECOMMENDATION: REJECT'}",
        "",
        recommendation,
        "",
        "---",
        "",
        "> **Gate Criteria**: Mean Advancement Brier Δ ≤ −0.005  "
        "OR  Mean Group Accuracy Δ ≥ +0.025  "
        "OR  Draw rate closer to historical by > 5pp (abs)",
        "",
        f"> **λ formula**: λ_h = 2μ × P_win_h / (P_win_h + P_win_a),  "
        f"λ_a = 2μ − λ_h  where μ = {wc_mu:.4f}",
    ]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 Poisson goal-scoring benchmark")
    parser.add_argument("--sims", type=int, default=50_000, help="Sims for disagreement + overhead")
    parser.add_argument("--bt-sims", type=int, default=20_000, help="Sims per historical WC")
    parser.add_argument("--overhead-repeats", type=int, default=3, help="Repeats for overhead measurement")
    args = parser.parse_args()

    print("\n" + "=" * 72)
    print("  PHASE 1: POISSON GOAL-SCORING BENCHMARK")
    print(f"  Backtest sims: {args.bt_sims:,}  |  Overhead sims: {args.sims:,}")
    print("=" * 72)

    # ── Step 0: Historical WC statistics ─────────────────────────────────────
    print("\n[0] Computing historical WC goal statistics...")
    hist_stats = _get_historical_wc_stats()
    wc_mu = calibrate_wc_goals_per_team()
    print(f"    Draw rate: {hist_stats['draw_rate']*100:.2f}%  |  "
          f"Goals/match: {hist_stats['goals_per_match']:.2f}  |  "
          f"Mean |GD|: {hist_stats['mean_abs_gd']:.2f}  (n={hist_stats['total_matches']} WC matches)")

    # ── Step 1: Build a shared probability matrix for disagreement + overhead ─
    # Use the full dataset (not OOS) for disagreement/overhead measurement
    print("\n[1] Building full-dataset probability matrix for disagreement measurement...")
    from scripts.eval_mov_elo import _build_feature_dataset
    X, y, elo, form_tracker, h2h, opp_history = _build_feature_dataset(
        cutoff_date=None, use_mov=False
    )
    model = MatchOutcomeModel()
    model.fit(X, y)

    # Use WC 2022 groups as the reference tournament for disagreement measurement
    # (most recent, 32 teams, groups match the historical backtest data)
    wc22_groups = {
        "A": ["Qatar", "Ecuador", "Senegal", "Netherlands"],
        "B": ["England", "Iran", "United States", "Wales"],
        "C": ["Argentina", "Saudi Arabia", "Mexico", "Poland"],
        "D": ["France", "Australia", "Denmark", "Tunisia"],
        "E": ["Spain", "Costa Rica", "Germany", "Japan"],
        "F": ["Belgium", "Canada", "Morocco", "Croatia"],
        "G": ["Brazil", "Serbia", "Switzerland", "Cameroon"],
        "H": ["Portugal", "Ghana", "Uruguay", "South Korea"],
    }
    teams_22 = [t for g in wc22_groups.values() for t in g]

    from core.config import TEAM_NAME_ALIASES, ALIAS_REVERSE
    form_scores = form_tracker.snapshot()
    form_details = form_tracker.snapshot_full_form()
    form_gd = form_tracker.snapshot_gd()

    opp_adj: Dict[str, float] = {}
    for canonical, csv_name in TEAM_NAME_ALIASES.items():
        recent = opp_history.get(csv_name, [])[-form_tracker.window:]
        own_form = form_scores.get(csv_name, 0.33)
        if not recent:
            opp_adj[canonical] = 0.0
        else:
            opp_mean = sum(form_scores.get(o, 0.33) for o in recent) / len(recent)
            opp_adj[canonical] = own_form + opp_mean - 1.0

    # Remap to canonical names for PMB
    fs_can  = {can: form_scores.get(csv, 0.33) for can, csv in TEAM_NAME_ALIASES.items()}
    fd_can  = {can: form_details.get(csv, (0.33,0.33,0.33,0.0)) for can, csv in TEAM_NAME_ALIASES.items()}
    fgd_can = {can: form_gd.get(csv, 0.0) for can, csv in TEAM_NAME_ALIASES.items()}

    def h2h_can(home: str, away: str) -> float:
        return h2h.win_rate_diff(
            TEAM_NAME_ALIASES.get(home, home),
            TEAM_NAME_ALIASES.get(away, away),
        )

    # Filter to WC2022 teams only (backtest uses csv names, so use elo with csv names)
    from backtesting.world_cup_backtest import _build_pre_tournament_state
    elo22, fs22, oa22, fgd22, h2h22, fd22 = _build_pre_tournament_state("2022-11-20")

    pmb22 = ProbabilityMatrixBuilder(model)
    pmb22.build(
        teams               = teams_22,
        elo_tracker         = elo22,
        form_scores         = fs22,
        opp_adj_form_scores = oa22,
        form_gd             = fgd22,
        h2h_fn              = h2h22.win_rate_diff,
        form_details        = fd22,
    )

    binary_sim  = HistoricalTournamentSimulator(pmb22, wc22_groups)
    poisson_sim = PoissonGroupStageSimulator(pmb22, wc22_groups, wc_goals_per_team=wc_mu)

    # ── Step 2: Disagreement rate ─────────────────────────────────────────────
    print(f"\n[2] Measuring standings disagreement rate ({args.sims:,} sims)...")
    disagreement = _measure_disagreement(binary_sim, poisson_sim, n_sims=args.sims)

    print(f"    Top-2 disagreement: {disagreement['top2_disagreement_rate']*100:.2f}%")
    print(f"    Winner disagreement: {disagreement['top1_disagreement_rate']*100:.2f}%")
    print(f"    Draw rate — Binary: {disagreement['draw_rate_binary']*100:.2f}%  "
          f"Poisson: {disagreement['draw_rate_poisson']*100:.2f}%  "
          f"Historical: {hist_stats['draw_rate']*100:.2f}%")
    print(f"    Goals/match (Poisson): {disagreement['mean_goals_per_match']:.3f}  "
          f"(Historical: {hist_stats['goals_per_match']:.2f})")

    # ── Step 3: Computational overhead ───────────────────────────────────────
    print(f"\n[3] Measuring computational overhead ({args.sims:,} sims × {args.overhead_repeats} repeats)...")
    overhead = _measure_overhead(binary_sim, poisson_sim, args.sims, args.overhead_repeats)
    print(f"    Binary: {overhead['binary_s']:.3f}s  |  Poisson: {overhead['poisson_s']:.3f}s  "
          f"|  Ratio: {overhead['overhead_ratio']:.2f}×")

    # ── Step 4: Historical WC backtests ──────────────────────────────────────
    print(f"\n[4] Running historical WC backtests ({args.bt_sims:,} sims/year × 3 years each)...")

    print("\n  --- Binary Simulator ---")
    binary_bt = _run_historical_backtest(use_poisson=False, n_sims=args.bt_sims, label="BINARY")

    print("\n  --- Poisson Simulator ---")
    poisson_bt = _run_historical_backtest(use_poisson=True, n_sims=args.bt_sims, label="POISSON")

    # ── Step 5: Gate criterion ────────────────────────────────────────────────
    brier_delta   = poisson_bt["mean_adv_brier"] - binary_bt["mean_adv_brier"]
    grp_acc_delta = poisson_bt["mean_grp_acc"] - binary_bt["mean_grp_acc"]

    # Draw rate closeness gate
    hist_draw     = hist_stats["draw_rate"]
    b_draw_dist   = abs(disagreement["draw_rate_binary"]  - hist_draw)
    p_draw_dist   = abs(disagreement["draw_rate_poisson"] - hist_draw)
    draw_gate_val = b_draw_dist - p_draw_dist  # positive means Poisson is closer

    gate_brier    = brier_delta   <= -0.005
    gate_grp_acc  = grp_acc_delta >= +0.025
    gate_draw     = draw_gate_val >   0.050    # >5pp closer to historical
    gate_passed   = gate_brier or gate_grp_acc or gate_draw

    print("\n" + "=" * 72)
    print("  GATE CRITERION RESULTS")
    print("=" * 72)
    print(f"  Advancement Brier Δ = {brier_delta:+.5f}  (gate: ≤ −0.005)  {'✓' if gate_brier else '✗'}")
    print(f"  Group Accuracy Δ    = {grp_acc_delta:+.5f}  (gate: ≥ +0.025)  {'✓' if gate_grp_acc else '✗'}")
    print(f"  Draw Rate Closer    = {draw_gate_val:+.5f}  (gate: > +0.050)  {'✓' if gate_draw else '✗'}")
    print(f"\n  VERDICT: {'ADOPT Poisson ✓' if gate_passed else 'REJECT — keep current ✗'}")
    print("=" * 72)

    print("\n  HISTORICAL WC ACCURACY COMPARISON")
    print(f"  {'Year':<6} {'Champion':<14} {'Binary':>10} {'Poisson':>10} {'BrierΔ':>10}")
    print("  " + "-" * 52)
    for i, yr in enumerate(binary_bt["per_year"]):
        pr = poisson_bt["per_year"][i]
        print(f"  {yr['year']:<6} {yr['champion']:<14} "
              f"{yr['adv_brier']:>10.4f} {pr['adv_brier']:>10.4f} "
              f"{pr['adv_brier']-yr['adv_brier']:>+10.4f}")
    print(f"  {'Mean':<20} {binary_bt['mean_adv_brier']:>10.4f} {poisson_bt['mean_adv_brier']:>10.4f} "
          f"{brier_delta:>+10.4f}")

    if gate_passed:
        recommendation = (
            "The Poisson group-stage simulator satisfies the gate criterion. "
            "Proceed to Phase 2: override `_sim_group_stage()` in both "
            "`simulation/monte_carlo.py` (TournamentSimulator) and "
            "`backtesting/world_cup_backtest.py` (HistoricalTournamentSimulator) "
            "with Poisson goal sampling. Add `lambda_h` and `lambda_a` matrices "
            "to the simulator constructors. Retrain is not required — only the "
            "group-stage goal draws change; the probability matrix is unchanged."
        )
    else:
        recommendation = (
            "The Poisson simulator did not satisfy any gate criterion. "
            "The binary ±1 GD + uniform goals approach, despite being synthetic, "
            "produces equivalent or better historical accuracy in simulation. "
            "This may indicate that the group advancement noise introduced by "
            "realistic Poisson sampling outweighs its benefits over 3 tournaments. "
            "Consider revisiting with a larger tournament backtest set (include "
            "continental tournaments: Copa América, EUROS, AFCON) to produce a "
            "more statistically robust sample before adoption."
        )

    # ── Write report ──────────────────────────────────────────────────────────
    report_md = _build_report(
        binary_bt, poisson_bt, disagreement, overhead,
        hist_stats, gate_passed, recommendation, wc_mu,
    )
    REPORT_PATH.write_text(report_md, encoding="utf-8")

    json_out = {
        "gate_passed":       gate_passed,
        "brier_delta":       brier_delta,
        "grp_acc_delta":     grp_acc_delta,
        "draw_gate_val":     draw_gate_val,
        "binary_backtest":   {k: v for k, v in binary_bt.items() if k != "per_year"},
        "poisson_backtest":  {k: v for k, v in poisson_bt.items() if k != "per_year"},
        "disagreement":      disagreement,
        "overhead":          overhead,
        "hist_stats":        hist_stats,
        "wc_mu":             wc_mu,
    }
    REPORT_JSON_PATH.write_text(json.dumps(json_out, indent=2), encoding="utf-8")

    print(f"\n  Full report saved → {REPORT_PATH}")
    log.info("Benchmark complete. Gate %s.", "PASSED" if gate_passed else "FAILED")


if __name__ == "__main__":
    main()
