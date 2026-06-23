"""
eval_mov_elo.py — Phase 1 Benchmark: Margin-of-Victory Elo vs Binary Elo

Runs a full self-contained evaluation comparing the current binary Elo system
against World Football Elo-style margin-of-victory (MoV) Elo.

Evaluation layers:
  1. Elo-only metrics  : correlation with outcomes, Kendall tau ranking accuracy
                         on historical WC results, Elo rating distributions
  2. ML metrics        : 5-fold TimeSeriesSplit CV on each Elo dataset
                         (Accuracy, Log Loss, Brier Score, Macro F1, elo_diff importance)
  3. Historical backtest: WC 2014, 2018, 2022 OOS tournament prediction
  4. WC2026 forecast   : 50,000 simulations for both systems

Gate criterion (ADOPT if any one is met, REJECT if none):
  - CV Log Loss delta <= -0.005   (lower is better)
  - CV Brier Score delta <= -0.002
  - Historical WC Brier Score delta <= -0.003 (mean over 3 tournaments)

PRODUCTION CODE IS NOT MODIFIED BY THIS SCRIPT.

Usage:
    python scripts/eval_mov_elo.py [--sims N]
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
from scipy.stats import kendalltau
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
)
from sklearn.model_selection import TimeSeriesSplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import (
    ALIAS_REVERSE,
    N_SIMULATIONS,
    RANDOM_SEED,
    REPORT_DIR,
    TEAM_NAME_ALIASES,
    TRAINING_START,
    WC2026_GROUPS,
    WC2026_TEAMS,
)
from core.logger import get_logger
from data.elo_tracker import EloTracker, _SCORE_MAP
from data.feature_engineering import FEATURE_COLS, build_inference_features, tournament_importance
from data.ingestion import (
    FormTracker,
    H2HRegistry,
    build_match_records,
    load_raw_results,
)
from models.ml_engine import MatchOutcomeModel, ProbabilityMatrixBuilder, _multiclass_brier_score

log = get_logger("eval_mov_elo")

REPORT_PATH = REPORT_DIR / "mov_elo_benchmark.md"

# ──────────────────────────────────────────────────────────────────────────────
# Historical WC data (mirrors backtesting/world_cup_backtest.py — no import)
# ──────────────────────────────────────────────────────────────────────────────
HISTORICAL_WCS = [
    {
        "year": 2014,
        "start": "2014-06-12",
        "champion": "Germany",
        "group_qualifiers": [
            "Brazil", "Mexico", "Netherlands", "Chile",
            "Colombia", "Greece", "Costa Rica", "Uruguay",
            "France", "Switzerland", "Argentina", "Nigeria",
            "Germany", "United States", "Belgium", "Algeria",
        ],
        "quarterfinalists": [
            "Brazil", "Colombia", "France", "Germany",
            "Netherlands", "Costa Rica", "Argentina", "Belgium",
        ],
        "semifinalists": ["Germany", "Argentina", "Brazil", "Netherlands"],
        "groups": {
            "A": ["Brazil", "Croatia", "Mexico", "Cameroon"],
            "B": ["Spain", "Netherlands", "Chile", "Australia"],
            "C": ["Colombia", "Greece", "Ivory Coast", "Japan"],
            "D": ["Uruguay", "Costa Rica", "England", "Italy"],
            "E": ["Switzerland", "Ecuador", "France", "Honduras"],
            "F": ["Argentina", "Bosnia and Herzegovina", "Iran", "Nigeria"],
            "G": ["Germany", "Portugal", "Ghana", "United States"],
            "H": ["Belgium", "Algeria", "Russia", "South Korea"],
        },
    },
    {
        "year": 2018,
        "start": "2018-06-14",
        "champion": "France",
        "group_qualifiers": [
            "Uruguay", "Russia", "Spain", "Portugal",
            "France", "Denmark", "Croatia", "Argentina",
            "Brazil", "Switzerland", "Sweden", "Mexico",
            "Belgium", "England", "Colombia", "Japan",
        ],
        "quarterfinalists": [
            "Uruguay", "France", "Brazil", "Belgium",
            "Sweden", "England", "Russia", "Croatia",
        ],
        "semifinalists": ["France", "Croatia", "Belgium", "England"],
        "groups": {
            "A": ["Russia", "Saudi Arabia", "Egypt", "Uruguay"],
            "B": ["Portugal", "Spain", "Morocco", "Iran"],
            "C": ["France", "Australia", "Peru", "Denmark"],
            "D": ["Argentina", "Iceland", "Croatia", "Nigeria"],
            "E": ["Brazil", "Switzerland", "Costa Rica", "Serbia"],
            "F": ["Germany", "Mexico", "Sweden", "South Korea"],
            "G": ["Belgium", "Panama", "Tunisia", "England"],
            "H": ["Poland", "Senegal", "Colombia", "Japan"],
        },
    },
    {
        "year": 2022,
        "start": "2022-11-20",
        "champion": "Argentina",
        "group_qualifiers": [
            "Netherlands", "Senegal", "England", "United States",
            "Argentina", "Poland", "France", "Australia",
            "Japan", "Spain", "Morocco", "Croatia",
            "Brazil", "Switzerland", "Portugal", "South Korea",
        ],
        "quarterfinalists": [
            "Netherlands", "Argentina", "Croatia", "Brazil",
            "England", "France", "Morocco", "Portugal",
        ],
        "semifinalists": ["Argentina", "France", "Croatia", "Morocco"],
        "groups": {
            "A": ["Qatar", "Ecuador", "Senegal", "Netherlands"],
            "B": ["England", "Iran", "United States", "Wales"],
            "C": ["Argentina", "Saudi Arabia", "Mexico", "Poland"],
            "D": ["France", "Australia", "Denmark", "Tunisia"],
            "E": ["Spain", "Costa Rica", "Germany", "Japan"],
            "F": ["Belgium", "Canada", "Morocco", "Croatia"],
            "G": ["Brazil", "Serbia", "Switzerland", "Cameroon"],
            "H": ["Portugal", "Ghana", "Uruguay", "South Korea"],
        },
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# Margin-of-Victory EloTracker (subclass — no production code modified)
# ══════════════════════════════════════════════════════════════════════════════

class MarginOfVictoryEloTracker(EloTracker):
    """
    Extends EloTracker with the World Football Elo margin-of-victory multiplier.

    Formula: G = min((11 + GD) / 8, 3.0)  for wins/losses
             G = 1.0                        for draws

    The parent class update() and batch_update() are NOT used here.
    Instead, batch_update_with_scores() ingests (home_score, away_score)
    directly to compute GD multipliers without data leakage.
    """

    @staticmethod
    def _mov_multiplier(goal_diff: int) -> float:
        """Return the MoV multiplier G for a given goal difference."""
        if goal_diff == 0:
            return 1.0   # draw — no multiplier
        elif goal_diff == 1:
            return 1.0
        elif goal_diff == 2:
            return 1.5
        else:
            return min((11.0 + goal_diff) / 8.0, 3.0)

    def update_with_score(
        self,
        home_team: str,
        away_team: str,
        home_score: int,
        away_score: int,
        tournament: str,
        neutral: bool = False,
    ) -> None:
        """Score-aware update applying the MoV multiplier."""
        goal_diff = abs(home_score - away_score)
        G = self._mov_multiplier(goal_diff)

        if home_score > away_score:
            outcome = 0
        elif home_score == away_score:
            outcome = 1
        else:
            outcome = 2

        r_h = self._get(home_team)
        r_a = self._get(away_team)
        k   = self._k_factor(tournament)
        e_h = self._expected(r_h, r_a, home_team_is_a=not neutral)
        s_h, s_a = _SCORE_MAP[outcome]

        delta_h = k * G * (s_h - e_h)
        delta_a = k * G * (s_a - (1.0 - e_h))

        self.ratings[home_team] = r_h + delta_h
        self.ratings[away_team] = r_a + delta_a

    def batch_update_with_scores(self, matches: pd.DataFrame) -> None:
        """
        Score-aware batch update. Requires columns:
          home_team, away_team, home_score, away_score, tournament, neutral
        """
        log.info(
            "Building MoV Elo ratings from %d historical matches...", len(matches)
        )
        for _, row in matches.iterrows():
            self.update_with_score(
                home_team  = str(row["home_team"]),
                away_team  = str(row["away_team"]),
                home_score = int(row["home_score"]),
                away_score = int(row["away_score"]),
                tournament = str(row["tournament"]),
                neutral    = bool(row["neutral"]),
            )
        log.info(
            "MoV Elo batch update complete. %d unique teams tracked.", len(self.ratings)
        )


# ══════════════════════════════════════════════════════════════════════════════
# Feature dataset builders — parallel pipelines, same form/H2H, different Elo
# ══════════════════════════════════════════════════════════════════════════════

def _build_feature_dataset(
    cutoff_date: Optional[str] = None,
    use_mov: bool = False,
) -> Tuple[pd.DataFrame, pd.Series, EloTracker, FormTracker, H2HRegistry, Dict]:
    """
    Build (X, y) feature dataset up to cutoff_date using either binary or MoV Elo.

    Returns:
      X           : feature DataFrame (FEATURE_COLS columns)
      y           : outcome Series
      elo         : fully-updated EloTracker (for inference state)
      form_tracker: fully-updated FormTracker
      h2h         : fully-updated H2HRegistry
      opp_history : {team: [recent_opponents]}
    """
    training_start = pd.Timestamp(TRAINING_START)
    cutoff = pd.Timestamp(cutoff_date) if cutoff_date else None
    wc_csv_names = set(TEAM_NAME_ALIASES.values())

    raw         = load_raw_results()
    all_matches = build_match_records(raw)
    # Attach raw scores to all_matches (build_match_records drops them)
    raw_clean = raw.dropna(subset=["home_score", "away_score"]).copy()
    raw_clean["home_score"] = raw_clean["home_score"].astype(int)
    raw_clean["away_score"] = raw_clean["away_score"].astype(int)
    raw_clean = raw_clean.sort_values("date").reset_index(drop=True)

    # Merge scores back onto all_matches
    all_matches = all_matches.copy()
    all_matches["home_score"] = raw_clean["home_score"].values
    all_matches["away_score"] = raw_clean["away_score"].values

    if cutoff is not None:
        history = all_matches[all_matches["date"] < cutoff].copy()
    else:
        history = all_matches.copy()

    # ── State trackers ────────────────────────────────────────────────────────
    if use_mov:
        elo = MarginOfVictoryEloTracker()
    else:
        elo = EloTracker()

    form_tracker  = FormTracker(window=10)
    h2h           = H2HRegistry()
    opp_history: Dict[str, List[str]] = {}
    training_rows: List[Dict] = []

    # Separate Elo replay tracker for pre-match snapshot (avoids leakage)
    pre_training_hist = history[history["date"] < training_start]
    if use_mov:
        replay_elo = MarginOfVictoryEloTracker()
        replay_elo.batch_update_with_scores(pre_training_hist)
    else:
        replay_elo = EloTracker()
        replay_elo.batch_update(pre_training_hist)

    for _, row in history.iterrows():
        h   = str(row["home_team"])
        a   = str(row["away_team"])
        hs  = int(row["home_score"])
        as_ = int(row["away_score"])
        date = row["date"]
        tournament = str(row["tournament"])
        neutral    = bool(row["neutral"])

        # ── Training row capture (pre-match, no leakage) ─────────────────────
        if (date >= training_start and h in wc_csv_names and a in wc_csv_names):
            h_can = ALIAS_REVERSE.get(h, h)
            a_can = ALIAS_REVERSE.get(a, a)

            # Elo snapshot before this match
            rh = replay_elo._get(h)
            ra = replay_elo._get(a)
            elo_diff = rh - ra

            h_w, h_d, h_l, h_gd = form_tracker.get_form(h)
            a_w, a_d, a_l, a_gd = form_tracker.get_form(a)
            h_form = form_tracker.get_form_score(h)
            a_form = form_tracker.get_form_score(a)

            recent_opp_h = opp_history.get(h, [])[-form_tracker.window:]
            recent_opp_a = opp_history.get(a, [])[-form_tracker.window:]

            def _opp_adj(own_form, opponents):
                if not opponents:
                    return 0.0
                opp_mean = sum(form_tracker.get_form_score(o) for o in opponents) / len(opponents)
                return own_form + opp_mean - 1.0

            opp_adj_h = _opp_adj(h_form, recent_opp_h)
            opp_adj_a = _opp_adj(a_form, recent_opp_a)

            training_rows.append({
                "date":              date,
                "home_team":         h_can,
                "away_team":         a_can,
                "tournament":        tournament,
                "neutral":           neutral,
                "target":            int(row["target"]),
                "elo_diff":          elo_diff,
                "home_form":         h_form,
                "away_form":         a_form,
                "opp_adj_form_diff": opp_adj_h - opp_adj_a,
                "gd_diff":           h_gd - a_gd,
                "home_win_rate":     h_w,
                "away_win_rate":     a_w,
                "home_draw_rate":    h_d,
                "away_draw_rate":    a_d,
                "home_loss_rate":    h_l,
            })

        # ── Update state AFTER capture (no leakage) ──────────────────────────
        form_tracker.update(h, a, hs, as_)
        h2h.update(h, a, hs, as_)
        opp_history.setdefault(h, []).append(a)
        opp_history.setdefault(a, []).append(h)

        if use_mov:
            replay_elo.update_with_score(h, a, hs, as_, tournament, neutral)
        else:
            replay_elo.update(h, a, int(row["target"]), tournament, neutral)

    if not training_rows:
        raise ValueError("No training rows found — check date range and team filters.")

    tdf = pd.DataFrame(training_rows).sort_values("date").reset_index(drop=True)
    tdf["form_diff"]      = tdf["home_form"].astype(float) - tdf["away_form"].astype(float)
    tdf["match_type_enc"] = tdf["tournament"].map(tournament_importance).fillna(0.40)
    tdf["neutral_venue"]  = tdf["neutral"].astype(float)

    missing = [c for c in FEATURE_COLS if c not in tdf.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    # Build the final Elo state for inference (full history)
    if use_mov:
        final_elo = MarginOfVictoryEloTracker()
        final_elo.batch_update_with_scores(history)
    else:
        final_elo = EloTracker()
        final_elo.batch_update(history)

    return (
        tdf[FEATURE_COLS].copy(),
        tdf["target"].astype(int),
        final_elo,
        form_tracker,
        h2h,
        opp_history,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Elo-only evaluation metrics
# ══════════════════════════════════════════════════════════════════════════════

def _eval_elo_only(
    X: pd.DataFrame,
    y: pd.Series,
    label: str,
) -> Dict[str, float]:
    """
    Evaluate predictive power of elo_diff alone (no ML model).

    Metrics:
      - Pearson correlation of elo_diff with match outcome (encoded as 0/0.5/1)
      - Accuracy of Elo-implied winner (sign of elo_diff → home/away win)
      - Elo-only log loss (using a logistic function to convert elo_diff to P(home win))
    """
    elo_diffs = X["elo_diff"].values
    outcomes  = y.values  # 0=home win, 1=draw, 2=away win

    # Encode outcomes as home-team score [0, 0.5, 1]
    home_score = np.where(outcomes == 0, 1.0, np.where(outcomes == 1, 0.5, 0.0))

    # Pearson correlation
    corr = float(np.corrcoef(elo_diffs, home_score)[0, 1])

    # Sign accuracy: elo_diff > 0 → predict home win, < 0 → predict away win
    elo_pred = np.where(elo_diffs > 0, 0, 2)  # ignore draws for sign accuracy
    non_draw_mask = outcomes != 1
    sign_acc = float(
        np.mean(elo_pred[non_draw_mask] == outcomes[non_draw_mask])
    ) if non_draw_mask.sum() > 0 else float("nan")

    # Elo-implied 3-class log loss using logistic conversion
    # P(home win) = logistic(elo_diff / 400)
    # Approximate draw probability using a Gaussian draw-band model
    p_home_raw = 1.0 / (1.0 + 10.0 ** (-elo_diffs / 400.0))
    draw_prob  = np.clip(0.27 * np.exp(-0.5 * ((p_home_raw - 0.50) / 0.21) ** 2), 0.05, 0.45)
    p_home = np.clip(p_home_raw - draw_prob / 2, 0.05, 0.90)
    p_away = np.clip(1.0 - p_home_raw - draw_prob / 2, 0.05, 0.90)
    total = p_home + draw_prob + p_away
    proba_3class = np.column_stack([p_home / total, draw_prob / total, p_away / total])
    elo_logloss = float(log_loss(outcomes, proba_3class))

    log.info(
        "[%s] Elo-only — corr=%.4f  sign_acc=%.4f  logloss=%.4f",
        label, corr, sign_acc, elo_logloss,
    )
    return {
        "pearson_correlation": corr,
        "sign_accuracy":       sign_acc,
        "elo_logloss":         elo_logloss,
    }


def _eval_elo_ranking_vs_actual(
    elo: EloTracker,
    wc_data: Dict,
    label: str,
) -> Dict[str, float]:
    """
    Measure Kendall tau between Elo ranks and actual tournament performance.
    Uses the pre-tournament Elo state for the historical WC.
    """
    all_teams = [t for g in wc_data["groups"].values() for t in g]
    # Get Elo ratings using csv names directly (backtest uses csv names, not canonical)
    elo_ratings = {t: elo._get(t) for t in all_teams}
    elo_rank = {t: rank for rank, t in enumerate(sorted(elo_ratings, key=elo_ratings.get, reverse=True), 1)}

    # Encode actual performance: champion=1, finalist=2, semifinalist=3/4, QF=5-8, R16=9-16, group=17+
    def _perf_score(team: str) -> int:
        if team == wc_data["champion"]:
            return 1
        if team in wc_data["semifinalists"]:
            return 3
        if team in wc_data["quarterfinalists"]:
            return 5
        if team in wc_data["group_qualifiers"]:
            return 9
        return 17

    actual_rank = {t: _perf_score(t) for t in all_teams}

    elo_r  = [elo_rank[t]   for t in all_teams]
    perf_r = [actual_rank[t] for t in all_teams]

    tau, p_val = kendalltau(elo_r, perf_r)
    log.info("[%s] WC%d Elo ranking Kendall tau=%.4f (p=%.4f)", label, wc_data["year"], tau, p_val)
    return {"kendall_tau": float(tau), "p_value": float(p_val)}


# ══════════════════════════════════════════════════════════════════════════════
# ML cross-validation evaluation
# ══════════════════════════════════════════════════════════════════════════════

def _run_cv(X: pd.DataFrame, y: pd.Series, label: str) -> Dict[str, float]:
    """Run 5-fold TimeSeriesSplit CV and return mean metrics."""
    log.info("[%s] Starting 5-fold TimeSeriesSplit CV on %d samples...", label, len(X))
    tscv = TimeSeriesSplit(n_splits=5)
    accs, losses, briers, f1s = [], [], [], []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        m = MatchOutcomeModel()
        m.fit(X_tr, y_tr)
        proba = m.predict_proba(X_val)
        preds = proba.argmax(axis=1)

        accs.append(accuracy_score(y_val, preds))
        losses.append(log_loss(y_val, proba))
        briers.append(_multiclass_brier_score(y_val, proba))
        f1s.append(f1_score(y_val, preds, average="macro"))
        log.info(
            "[%s] Fold %d — acc=%.4f logloss=%.4f brier=%.4f f1=%.4f",
            label, fold, accs[-1], losses[-1], briers[-1], f1s[-1],
        )

    result = {
        "accuracy":    float(np.mean(accs)),
        "log_loss":    float(np.mean(losses)),
        "brier_score": float(np.mean(briers)),
        "f1_macro":    float(np.mean(f1s)),
    }
    log.info(
        "[%s] CV Summary — acc=%.4f logloss=%.4f brier=%.4f f1=%.4f",
        label, result["accuracy"], result["log_loss"], result["brier_score"], result["f1_macro"],
    )
    return result


def _get_elo_diff_importance(X: pd.DataFrame, y: pd.Series, label: str) -> float:
    """Train one model on all data and return elo_diff feature importance (gain)."""
    m = MatchOutcomeModel()
    m.fit(X, y)
    return round(m.feat_importance.get("elo_diff", 0.0), 6)


# ══════════════════════════════════════════════════════════════════════════════
# Historical WC backtest
# ══════════════════════════════════════════════════════════════════════════════

def _advancement_brier(team_probs: pd.Series, actual_teams: List[str]) -> float:
    actual = team_probs.index.to_series().isin(actual_teams).astype(float).to_numpy()
    pred   = team_probs.to_numpy() / 100.0
    return float(np.mean((pred - actual) ** 2))


def _stage_acc(df: pd.DataFrame, col: str, actual: List[str], k: int) -> float:
    return len(set(df.nlargest(k, col)["team"]) & set(actual)) / k


def _run_historical_backtest(use_mov: bool, n_sims: int, label: str) -> Dict:
    """Run WC 2014, 2018, 2022 historical backtest with the chosen Elo variant."""
    results_per_year = []

    for wc in HISTORICAL_WCS:
        log.info("[%s] Historical backtest WC%d (cutoff %s)...", label, wc["year"], wc["start"])
        t0 = time.perf_counter()

        X_bt, y_bt, elo_state, form_tracker, h2h, opp_history = _build_feature_dataset(
            cutoff_date=wc["start"], use_mov=use_mov
        )

        model_bt = MatchOutcomeModel()
        model_bt.fit(X_bt, y_bt)

        # Build form snapshots from the form_tracker state at cutoff
        form_scores  = form_tracker.snapshot()
        form_details = form_tracker.snapshot_full_form()
        form_gd      = form_tracker.snapshot_gd()

        opp_adj: Dict[str, float] = {}
        for team, own_form in form_scores.items():
            recent = opp_history.get(team, [])[-form_tracker.window:]
            if not recent:
                opp_adj[team] = 0.0
            else:
                opp_mean = sum(form_scores.get(o, 0.33) for o in recent) / len(recent)
                opp_adj[team] = own_form + opp_mean - 1.0

        teams = [t for g in wc["groups"].values() for t in g]
        pmb = ProbabilityMatrixBuilder(model_bt)
        pmb.build(
            teams               = teams,
            elo_tracker         = elo_state,
            form_scores         = form_scores,
            opp_adj_form_scores = opp_adj,
            form_gd             = form_gd,
            form_details        = form_details,
            h2h_fn              = h2h.win_rate_diff,
        )

        # Simple 32-team simulation (mirror HistoricalTournamentSimulator)
        from backtesting.world_cup_backtest import (
            HistoricalTournamentSimulator, HistoricalTournament,
        )
        ht = HistoricalTournament(
            year=wc["year"], start_date=wc["start"],
            groups=wc["groups"], actual={}
        )
        simulator = HistoricalTournamentSimulator(pmb, wc["groups"])
        res = simulator.run(n_sims=n_sims)
        res_df = res.to_dataframe()

        # Elo-only ranking metrics
        tau_metrics = _eval_elo_ranking_vs_actual(elo_state, wc, label)

        champ_prob  = float(res_df.loc[res_df["team"] == wc["champion"], "champion_pct"].values[0])
        grp_acc     = _stage_acc(res_df, "round_of_16_pct", wc["group_qualifiers"], 16)
        qf_acc      = _stage_acc(res_df, "quarterfinal_pct", wc["quarterfinalists"], 8)
        sf_acc      = _stage_acc(res_df, "semifinal_pct", wc["semifinalists"], 4)

        g_brier  = _advancement_brier(res_df.set_index("team")["round_of_16_pct"], wc["group_qualifiers"])
        qf_brier = _advancement_brier(res_df.set_index("team")["quarterfinal_pct"], wc["quarterfinalists"])
        sf_brier = _advancement_brier(res_df.set_index("team")["semifinal_pct"], wc["semifinalists"])
        ch_brier = _advancement_brier(res_df.set_index("team")["champion_pct"], [wc["champion"]])
        adv_brier = float(np.mean([g_brier, qf_brier, sf_brier, ch_brier]))

        results_per_year.append({
            "year":         wc["year"],
            "champion":     wc["champion"],
            "champion_pct": champ_prob,
            "group_acc":    grp_acc,
            "qf_acc":       qf_acc,
            "sf_acc":       sf_acc,
            "adv_brier":    adv_brier,
            "kendall_tau":  tau_metrics["kendall_tau"],
            "runtime_s":    time.perf_counter() - t0,
        })
        log.info(
            "[%s] WC%d — champ=%.2f%%  grp=%.3f  qf=%.3f  sf=%.3f  brier=%.4f  tau=%.4f",
            label, wc["year"], champ_prob, grp_acc, qf_acc, sf_acc, adv_brier, tau_metrics["kendall_tau"],
        )

    aggregate_brier = float(np.mean([r["adv_brier"] for r in results_per_year]))
    return {"per_year": results_per_year, "mean_adv_brier": aggregate_brier}


# ══════════════════════════════════════════════════════════════════════════════
# WC2026 forecast
# ══════════════════════════════════════════════════════════════════════════════

def _run_wc2026_forecast(use_mov: bool, n_sims: int, label: str) -> pd.DataFrame:
    """Run full WC2026 50k-sim forecast using the chosen Elo variant."""
    from scripts.wc2026_forecast import ExtendedTournamentSimulator

    log.info("[%s] Building WC2026 feature dataset...", label)
    X, y, elo_state, form_tracker, h2h, opp_history = _build_feature_dataset(
        cutoff_date=None, use_mov=use_mov
    )

    model = MatchOutcomeModel()
    model.fit(X, y)

    form_scores  = form_tracker.snapshot()
    form_details = form_tracker.snapshot_full_form()
    form_gd      = form_tracker.snapshot_gd()

    opp_adj: Dict[str, float] = {}
    for canonical, csv_name in TEAM_NAME_ALIASES.items():
        recent = opp_history.get(csv_name, [])[-form_tracker.window:]
        own_form = form_scores.get(csv_name, 0.33)
        if not recent:
            opp_adj[canonical] = 0.0
        else:
            opp_mean = sum(form_scores.get(o, 0.33) for o in recent) / len(recent)
            opp_adj[canonical] = own_form + opp_mean - 1.0

    # Remap form scores from csv names to canonical names
    form_scores_can = {
        can: form_scores.get(csv, 0.33)
        for can, csv in TEAM_NAME_ALIASES.items()
    }
    form_details_can = {
        can: form_details.get(csv, (0.33, 0.33, 0.33, 0.0))
        for can, csv in TEAM_NAME_ALIASES.items()
    }
    form_gd_can = {
        can: form_gd.get(csv, 0.0)
        for can, csv in TEAM_NAME_ALIASES.items()
    }

    def h2h_fn_can(home: str, away: str) -> float:
        h_csv = TEAM_NAME_ALIASES.get(home, home)
        a_csv = TEAM_NAME_ALIASES.get(away, away)
        return h2h.win_rate_diff(h_csv, a_csv)

    pmb = ProbabilityMatrixBuilder(model)
    pmb.build(
        teams               = WC2026_TEAMS,
        elo_tracker         = elo_state,
        form_scores         = form_scores_can,
        opp_adj_form_scores = opp_adj,
        form_gd             = form_gd_can,
        form_details        = form_details_can,
        h2h_fn              = h2h_fn_can,
    )

    log.info("[%s] Running %d WC2026 simulations...", label, n_sims)
    sim = ExtendedTournamentSimulator(prob_matrix_builder=pmb, groups=WC2026_GROUPS)
    results, _ = sim.run_extended(n_sims=n_sims)
    return results.to_dataframe()


# ══════════════════════════════════════════════════════════════════════════════
# Report builder
# ══════════════════════════════════════════════════════════════════════════════

def _delta(new: float, old: float, lower_is_better: bool = True) -> str:
    d = new - old
    sign = "+" if d > 0 else ""
    marker = ""
    if lower_is_better and d < -0.001:
        marker = " ✓"
    elif not lower_is_better and d > 0.001:
        marker = " ✓"
    return f"{sign}{d:.5f}{marker}"


def _build_report(
    baseline: Dict,
    mov: Dict,
    gate_passed: bool,
    recommendation: str,
) -> str:
    b_cv = baseline["cv"]
    m_cv = mov["cv"]
    b_elo = baseline["elo_only"]
    m_elo = mov["elo_only"]

    lines = [
        "# Margin-of-Victory Elo — Phase 1 Benchmark Report",
        "",
        f"> **Simulations**: 50,000  |  **CV folds**: 5 (TimeSeriesSplit)  |  **Gate**: {'PASSED ✓' if gate_passed else 'FAILED ✗'}",
        "",
        "---",
        "",
        "## 1. Elo-Only Evaluation (No ML Model)",
        "",
        "These metrics evaluate the raw Elo signal quality before any XGBoost model is involved.",
        "",
        "| Metric | Binary Elo | MoV Elo | Delta |",
        "|:---|---:|---:|---:|",
        f"| Pearson Correlation (elo_diff vs outcome) | {b_elo['pearson_correlation']:.5f} | {m_elo['pearson_correlation']:.5f} | {_delta(m_elo['pearson_correlation'], b_elo['pearson_correlation'], lower_is_better=False)} |",
        f"| Sign Accuracy (non-draw matches) | {b_elo['sign_accuracy']:.5f} | {m_elo['sign_accuracy']:.5f} | {_delta(m_elo['sign_accuracy'], b_elo['sign_accuracy'], lower_is_better=False)} |",
        f"| Elo-Implied Log Loss | {b_elo['elo_logloss']:.5f} | {m_elo['elo_logloss']:.5f} | {_delta(m_elo['elo_logloss'], b_elo['elo_logloss'], lower_is_better=True)} |",
        "",
        "---",
        "",
        "## 2. ML Cross-Validation Metrics (5-Fold TimeSeriesSplit)",
        "",
        "| Metric | Binary Elo | MoV Elo | Delta | Gate? |",
        "|:---|---:|---:|---:|:---|",
        f"| Accuracy | {b_cv['accuracy']:.5f} | {m_cv['accuracy']:.5f} | {_delta(m_cv['accuracy'], b_cv['accuracy'], lower_is_better=False)} | — |",
        f"| **Log Loss** | {b_cv['log_loss']:.5f} | {m_cv['log_loss']:.5f} | {_delta(m_cv['log_loss'], b_cv['log_loss'])} | Δ ≤ −0.005 |",
        f"| **Brier Score** | {b_cv['brier_score']:.5f} | {m_cv['brier_score']:.5f} | {_delta(m_cv['brier_score'], b_cv['brier_score'])} | Δ ≤ −0.002 |",
        f"| Macro F1 | {b_cv['f1_macro']:.5f} | {m_cv['f1_macro']:.5f} | {_delta(m_cv['f1_macro'], b_cv['f1_macro'], lower_is_better=False)} | — |",
        f"| elo_diff Feature Importance | {baseline['elo_importance']:.6f} | {mov['elo_importance']:.6f} | {_delta(mov['elo_importance'], baseline['elo_importance'], lower_is_better=False)} | — |",
        "",
        "---",
        "",
        "## 3. Historical WC Backtest (2014 / 2018 / 2022)",
        "",
        "| Year | Champion | Binary Champ% | MoV Champ% | Binary Brier | MoV Brier | Binary Tau | MoV Tau |",
        "|:---|:---|---:|---:|---:|---:|---:|---:|",
    ]

    for i, year_data in enumerate(baseline["backtest"]["per_year"]):
        m_year = mov["backtest"]["per_year"][i]
        lines.append(
            f"| {year_data['year']} | {year_data['champion']} | "
            f"{year_data['champion_pct']:.2f}% | {m_year['champion_pct']:.2f}% | "
            f"{year_data['adv_brier']:.4f} | {m_year['adv_brier']:.4f} | "
            f"{year_data['kendall_tau']:.4f} | {m_year['kendall_tau']:.4f} |"
        )

    b_agg_brier = baseline["backtest"]["mean_adv_brier"]
    m_agg_brier = mov["backtest"]["mean_adv_brier"]
    lines += [
        f"| **Mean** | — | — | — | **{b_agg_brier:.4f}** | **{m_agg_brier:.4f}** | — | — |",
        "",
        f"> **Historical Brier gate** (Δ ≤ −0.003): Δ = {m_agg_brier - b_agg_brier:+.5f}  {'✓ PASSED' if m_agg_brier - b_agg_brier <= -0.003 else '✗ FAILED'}",
        "",
        "---",
        "",
        "## 4. WC2026 Championship Probabilities (Top 20)",
        "",
        "| Rank | Team | Binary Champ% | MoV Champ% | Delta |",
        "|:---|:---|---:|---:|---:|",
    ]

    b_fc = baseline["wc2026"].set_index("team")
    m_fc = mov["wc2026"].set_index("team")
    all_teams_sorted = baseline["wc2026"].sort_values("champion_pct", ascending=False)["team"].tolist()

    for rank, team in enumerate(all_teams_sorted[:20], 1):
        b_pct = float(b_fc.loc[team, "champion_pct"]) if team in b_fc.index else 0.0
        m_pct = float(m_fc.loc[team, "champion_pct"]) if team in m_fc.index else 0.0
        delta = m_pct - b_pct
        sign  = "+" if delta >= 0 else ""
        lines.append(f"| {rank} | {team} | {b_pct:.2f}% | {m_pct:.2f}% | {sign}{delta:.2f}pp |")

    b_top3 = sum(float(b_fc.loc[t, "champion_pct"]) for t in all_teams_sorted[:3] if t in b_fc.index)
    m_top3 = sum(float(m_fc.loc[t, "champion_pct"]) for t in all_teams_sorted[:3] if t in m_fc.index)

    lines += [
        "",
        f"**Top-3 concentration**: Binary = {b_top3:.1f}%  |  MoV = {m_top3:.1f}%  |  Δ = {m_top3 - b_top3:+.1f}pp",
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
        "> **Gate Criteria**: Log Loss Δ ≤ −0.005  OR  Brier Score Δ ≤ −0.002  OR  Historical WC Brier Δ ≤ −0.003",
        f"> **Formula used**: G = min((11 + GD) / 8, 3.0) for wins; G = 1.0 for draws",
    ]

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 MoV Elo benchmark")
    parser.add_argument("--sims", type=int, default=50_000, help="Simulations for WC2026 forecast")
    parser.add_argument("--bt-sims", type=int, default=10_000, help="Simulations per historical WC")
    args = parser.parse_args()

    print("\n" + "=" * 72)
    print("  PHASE 1: MARGIN-OF-VICTORY ELO BENCHMARK")
    print(f"  WC2026 sims: {args.sims:,}  |  Historical WC sims: {args.bt_sims:,}")
    print("=" * 72)

    results: Dict[str, Dict] = {"baseline": {}, "mov": {}}

    for label, use_mov in [("BINARY", False), ("MOV", True)]:
        print(f"\n[{label}] Building feature dataset...")
        X, y, elo, form_tracker, h2h, opp_hist = _build_feature_dataset(
            cutoff_date=None, use_mov=use_mov
        )

        print(f"[{label}] Evaluating Elo-only metrics...")
        elo_metrics = _eval_elo_only(X, y, label)

        print(f"[{label}] Running ML cross-validation...")
        cv_metrics = _run_cv(X, y, label)

        print(f"[{label}] Computing elo_diff feature importance...")
        elo_importance = _get_elo_diff_importance(X, y, label)

        print(f"[{label}] Running historical WC backtest...")
        backtest = _run_historical_backtest(use_mov=use_mov, n_sims=args.bt_sims, label=label)

        print(f"[{label}] Running WC2026 forecast ({args.sims:,} sims)...")
        wc2026_df = _run_wc2026_forecast(use_mov=use_mov, n_sims=args.sims, label=label)

        key = "baseline" if not use_mov else "mov"
        results[key] = {
            "elo_only":      elo_metrics,
            "cv":            cv_metrics,
            "elo_importance": elo_importance,
            "backtest":      backtest,
            "wc2026":        wc2026_df,
        }

    # ── Gate criterion ────────────────────────────────────────────────────────
    b = results["baseline"]
    m = results["mov"]

    logloss_delta  = m["cv"]["log_loss"]    - b["cv"]["log_loss"]
    brier_delta    = m["cv"]["brier_score"] - b["cv"]["brier_score"]
    ht_brier_delta = m["backtest"]["mean_adv_brier"] - b["backtest"]["mean_adv_brier"]

    gate_logloss  = logloss_delta  <= -0.005
    gate_brier    = brier_delta    <= -0.002
    gate_ht_brier = ht_brier_delta <= -0.003
    gate_passed   = gate_logloss or gate_brier or gate_ht_brier

    gate_lines = [
        f"  CV Log Loss Δ = {logloss_delta:+.5f}  (gate: ≤ −0.005)  {'✓' if gate_logloss else '✗'}",
        f"  CV Brier Δ    = {brier_delta:+.5f}  (gate: ≤ −0.002)  {'✓' if gate_brier else '✗'}",
        f"  Hist Brier Δ  = {ht_brier_delta:+.5f}  (gate: ≤ −0.003)  {'✓' if gate_ht_brier else '✗'}",
    ]

    if gate_passed:
        recommendation = (
            "MoV Elo satisfies the gate criterion. Proceed to Phase 2: modify "
            "`data/elo_tracker.py`, `data/feature_engineering.py`, and "
            "`backtesting/world_cup_backtest.py` to adopt the MoV multiplier "
            "in production. Retrain, rerun simulations, and update the dashboard."
        )
    else:
        recommendation = (
            "MoV Elo did not satisfy any gate criterion. The marginal improvement "
            "is not sufficient to justify modifying production code. The current "
            "binary Elo remains the production standard. Consider revisiting with "
            "a larger dataset or exploring alternative Elo extensions."
        )

    print("\n" + "=" * 72)
    print("  GATE CRITERION RESULTS")
    print("=" * 72)
    for line in gate_lines:
        print(line)
    print(f"\n  VERDICT: {'ADOPT MoV ELO ✓' if gate_passed else 'REJECT — keep binary Elo ✗'}")
    print("=" * 72)

    # ── Print comparison table ────────────────────────────────────────────────
    print("\n  TOP-10 WC2026 CHAMPIONSHIP PROBABILITIES")
    print(f"  {'Team':<24} {'Binary%':>10} {'MoV%':>10} {'Delta':>10}")
    print("  " + "-" * 56)
    b_fc = b["wc2026"].set_index("team")
    m_fc = m["wc2026"].set_index("team")
    for rank, team in enumerate(b["wc2026"].sort_values("champion_pct", ascending=False)["team"].tolist()[:10], 1):
        bp = float(b_fc.loc[team, "champion_pct"])
        mp = float(m_fc.loc[team, "champion_pct"]) if team in m_fc.index else 0.0
        print(f"  {rank:>2}. {team:<22} {bp:>9.2f}% {mp:>9.2f}% {mp - bp:>+9.2f}pp")

    # ── Save report ───────────────────────────────────────────────────────────
    report_md = _build_report(b, m, gate_passed, recommendation)
    REPORT_PATH.write_text(report_md, encoding="utf-8")
    log.info("Benchmark report saved -> %s", REPORT_PATH)
    print(f"\n  Full report saved to: {REPORT_PATH}")

    # ── Save structured JSON for downstream consumption ───────────────────────
    json_out = REPORT_DIR / "mov_elo_benchmark.json"
    json_out.write_text(json.dumps({
        "gate_passed":     gate_passed,
        "logloss_delta":   logloss_delta,
        "brier_delta":     brier_delta,
        "ht_brier_delta":  ht_brier_delta,
        "baseline_cv":     b["cv"],
        "mov_cv":          m["cv"],
        "baseline_elo":    b["elo_only"],
        "mov_elo":         m["elo_only"],
        "baseline_bt_brier": b["backtest"]["mean_adv_brier"],
        "mov_bt_brier":      m["backtest"]["mean_adv_brier"],
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
