"""
main.py — WC2026 Prediction Pipeline Orchestrator
==================================================

Phase 1  │ Real data ingestion
         │   • Load results.csv (martj42/international_results)
         │   • Drop future/incomplete rows (NA scores)
         │   • Normalise team names to canonical WC2026 names
         │   • Build chronological form scores and H2H registry

Phase 2  │ Elo rating construction
         │   • Batch-update EloTracker from the full 49k-row results history
         │   • K-factors scaled by tournament importance
         │   • Final ratings represent each team's true strength into WC2026

Phase 3  │ Feature engineering
         │   • Replay Elo over training window to avoid data leakage
         │   • Attach real form scores, H2H differentials
         │   • Transfermarkt squad values (May 2026) → log market-value diff
         │   • FIFA rankings (April 2026) → rank differential
         │   • Haversine travel fatigue, climate mismatch

Phase 4  │ XGBoost training
         │   • Expanding-window time-aware CV (5 folds) for forward-looking performance estimate
         │   • Full-dataset refit produces the production model
         │   • SHAP TreeExplainer for local feature attribution

Phase 5  │ 48×48 probability matrix pre-computation
         │   • Single XGBoost batch pass over 2,256 ordered team pairs
         │   • Neutral-venue calibration via symmetric averaging
         │   • Knockout matrix: draw probability redistributed proportionally

Phase 6  │ Monte Carlo simulation (10,000 tournaments)
         │   • All 72 group matches vectorised over n_sims simultaneously
         │   • Best-8 thirds selected via numpy lexsort
         │   • Knockout rounds use O(1) matrix lookups — zero model calls

Phase 7  │ Reporting
         │   • Elo ratings table
         │   • Model CV performance + SHAP feature importances
         │   • Sample match predictions with SHAP drivers
         │   • Full 48-team tournament probability forecast
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from core.config import N_SIMULATIONS, WC2026_GROUPS, WC2026_TEAMS
from core.logger import get_logger
from data.ingestion import MatchDataPipeline
from data.elo_tracker import EloTracker
from data.feature_engineering import (
    FEATURE_COLS,
    build_training_features,
    build_inference_features,
)
from models.ml_engine import MatchOutcomeModel, ProbabilityMatrixBuilder
from simulation.monte_carlo import TournamentSimulator
from reports.generator import ReportGenerator

log    = get_logger("pipeline.main")
_DIV  = "═" * 82
_SDIV = "─" * 82


def _phase(n: int, title: str) -> None:
    log.info("")
    log.info(_DIV)
    log.info("  PHASE %d  │  %s", n, title)
    log.info(_DIV)


# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline() -> None:
    t0 = time.perf_counter()

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 1 — Data ingestion
    # ──────────────────────────────────────────────────────────────────────────
    _phase(1, "Real Data Ingestion  (results.csv)")

    data_pipeline = MatchDataPipeline()

    log.info(
        "Training window (WC teams only, ≥ %s): %d matches",
        "2010-01-01", len(data_pipeline.training_df),
    )
    log.info(
        "Target distribution — 0=home win: %d  1=draw: %d  2=away win: %d",
        (data_pipeline.training_df["target"] == 0).sum(),
        (data_pipeline.training_df["target"] == 1).sum(),
        (data_pipeline.training_df["target"] == 2).sum(),
    )

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 2 — Elo rating construction
    # ──────────────────────────────────────────────────────────────────────────
    _phase(2, "Rolling Elo Rating Construction")

    elo = EloTracker()
    elo.batch_update(data_pipeline.matches)
    elo.save()

    elo_df = elo.ratings_df()
    reporter = ReportGenerator()
    print(reporter.elo_ratings(elo_df, top_n=20))

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 3 — Feature engineering
    # ──────────────────────────────────────────────────────────────────────────
    _phase(3, "Feature Engineering")

    X_train, y_train = build_training_features(
        training_df = data_pipeline.training_df,
        elo_tracker = elo,
    )

    log.info("Training matrix: %d rows × %d features", len(X_train), len(FEATURE_COLS))
    log.info("Features: %s", FEATURE_COLS)

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 4 — XGBoost training
    # ──────────────────────────────────────────────────────────────────────────
    _phase(4, "XGBoost Training  (5-fold Expanding-Window CV)")

    model = MatchOutcomeModel()
    model.fit(X_train, y_train)
    model.save()

    print(reporter.model_performance(model.cv_metrics, model.importance_report()))

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 5 — Pre-compute 48×48 probability matrix
    # ──────────────────────────────────────────────────────────────────────────
    _phase(5, "Pre-computing 48×48 Matchup Probability Matrix")

    form_scores = data_pipeline.current_form_scores()
    opp_adj_form_scores = data_pipeline.current_opp_adjusted_form_scores()
    form_gd = data_pipeline.current_form_gd()
    h2h_fn      = data_pipeline.current_h2h   # callable(home, away) → float
    form_details = data_pipeline.current_form_details()

    pmb = ProbabilityMatrixBuilder(model)
    pmb.build(
    teams        = WC2026_TEAMS,
    elo_tracker  = elo,
    form_scores  = form_scores,
    opp_adj_form_scores = opp_adj_form_scores,
    form_gd      = form_gd,
    form_details = form_details,
    h2h_fn       = h2h_fn,
)

    # Print a selection of interesting matchup probabilities
    print("\n" + _SDIV)
    print("  PRE-COMPUTED MATCHUP PROBABILITIES  (neutral venue)")
    print(_SDIV)
    showcases = [
        ("Argentina",  "France"),
        ("England",    "Brazil"),
        ("Spain",      "Germany"),
        ("Morocco",    "Netherlands"),
        ("USA",        "Mexico"),
        ("Portugal",   "Belgium"),
        ("Japan",      "Senegal"),
        ("Colombia",   "Uruguay"),
    ]
    for h, a in showcases:
        if h in pmb.team_idx and a in pmb.team_idx:
            pw, pd_, pl = pmb.lookup(h, a)
            print(f"  {h:<22} vs {a:<22}  win={pw:.3f}  draw={pd_:.3f}  loss={pl:.3f}")
    print(_SDIV)

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 6 — Monte Carlo simulation
    # ──────────────────────────────────────────────────────────────────────────
    _phase(6, f"Monte Carlo Simulation  ({N_SIMULATIONS:,} tournaments)")

    simulator   = TournamentSimulator(prob_matrix_builder=pmb, groups=WC2026_GROUPS)
    sim_results = simulator.run(n_sims=N_SIMULATIONS)
    results_df  = sim_results.to_dataframe()

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 7 — Reporting
    # ──────────────────────────────────────────────────────────────────────────
    _phase(7, "Reports")

    # SHAP-annotated match predictions for selected matchups
    print()
    for home, away in [
        ("Argentina", "France"),
        ("England", "Brazil"),
        ("Spain", "Germany"),
        ("Morocco", "Senegal"),
    ]:
        if home not in pmb.team_idx or away not in pmb.team_idx:
            continue
        feats = build_inference_features(
            pairs       = [(home, away)],
            elo_tracker = elo,
            form_scores = form_scores,
            opp_adj_form_scores = opp_adj_form_scores,
            form_details = form_details,
            form_gd     = form_gd,
            h2h_fn      = h2h_fn,
            neutral     = True,
            tournament  = "FIFA World Cup",
        )
        feat_dict = dict(zip(FEATURE_COLS, feats[FEATURE_COLS].iloc[0]))
        ph, pd_, pa = model.predict_proba_single(feat_dict)
        shap_df     = model.explain_single(feat_dict, class_idx=0, top_n=5)
        print(reporter.match_prediction(home, away, (ph, pd_, pa), shap_df))

    # Full tournament forecast
    forecast_report = reporter.tournament_forecast(
        results_df, N_SIMULATIONS, sim_results.elapsed_s
    )
    print(forecast_report)

    reporter.save_json(results_df)

    # ── Summary ────────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0
    print()
    print(_DIV)
    print("  TOP-15 CHAMPIONSHIP PROBABILITIES  —  2026 FIFA WORLD CUP")
    print(_DIV)
    for _, row in results_df.head(15).iterrows():
        bar = "█" * round(float(row["champion_pct"]) / 1.5)
        print(f"  {int(row['rank']):>2}. {row['team']:<26} {float(row['champion_pct']):5.2f}%  {bar}")
    print(_DIV)
    print(f"\n  Total pipeline runtime: {elapsed:.1f}s")
    print(_DIV)


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_pipeline()
