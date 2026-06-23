"""
Historical World Cup backtesting using the current production model.

This module simulates the 2014, 2018, and 2022 FIFA World Cups using only
match information available before each tournament started.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from core.config import (
    ALIAS_REVERSE,
    DATA_DIR,
    N_SIMULATIONS,
    REPORT_DIR,
    TEAM_NAME_ALIASES,
    TRAINING_START,
)
from core.logger import get_logger
from data.elo_tracker import EloTracker
from data.feature_engineering import (
    FEATURE_COLS,
    build_inference_features,
    tournament_importance,
)
from data.ingestion import FormTracker, H2HRegistry, build_match_records, load_raw_results
from models.ml_engine import MatchOutcomeModel, ProbabilityMatrixBuilder

log = get_logger("backtesting")

_BACKTEST_TXT  = REPORT_DIR / "historical_backtest.txt"
_BACKTEST_JSON = REPORT_DIR / "historical_backtest.json"
_OOS_TXT       = REPORT_DIR / "oos_backtest.txt"
_OOS_JSON      = REPORT_DIR / "oos_backtest.json"
_LINE          = "-" * 82
_DLINE         = "═" * 82


@dataclass(frozen=True)
class HistoricalTournament:
    year: int
    start_date: str
    groups: Dict[str, List[str]]
    actual: Dict[str, object]


HISTORICAL_WORLD_CUPS: List[HistoricalTournament] = [
    HistoricalTournament(
        year=2014,
        start_date="2014-06-12",
        groups={
            "A": ["Brazil", "Croatia", "Mexico", "Cameroon"],
            "B": ["Spain", "Netherlands", "Chile", "Australia"],
            "C": ["Colombia", "Greece", "Ivory Coast", "Japan"],
            "D": ["Uruguay", "Costa Rica", "England", "Italy"],
            "E": ["Switzerland", "Ecuador", "France", "Honduras"],
            "F": ["Argentina", "Bosnia and Herzegovina", "Iran", "Nigeria"],
            "G": ["Germany", "Portugal", "Ghana", "United States"],
            "H": ["Belgium", "Algeria", "Russia", "South Korea"],
        },
        actual={
            "champion": "Germany",
            "finalist": ["Germany", "Argentina"],
            "semifinalists": ["Germany", "Argentina", "Brazil", "Netherlands"],
            "quarterfinalists": [
                "Brazil", "Colombia", "France", "Germany",
                "Netherlands", "Costa Rica", "Argentina", "Belgium",
            ],
            "group_qualifiers": [
                "Brazil", "Mexico", "Netherlands", "Chile",
                "Colombia", "Greece", "Costa Rica", "Uruguay",
                "France", "Switzerland", "Argentina", "Nigeria",
                "Germany", "United States", "Belgium", "Algeria",
            ],
        },
    ),
    HistoricalTournament(
        year=2018,
        start_date="2018-06-14",
        groups={
            "A": ["Russia", "Saudi Arabia", "Egypt", "Uruguay"],
            "B": ["Portugal", "Spain", "Morocco", "Iran"],
            "C": ["France", "Australia", "Peru", "Denmark"],
            "D": ["Argentina", "Iceland", "Croatia", "Nigeria"],
            "E": ["Brazil", "Switzerland", "Costa Rica", "Serbia"],
            "F": ["Germany", "Mexico", "Sweden", "South Korea"],
            "G": ["Belgium", "Panama", "Tunisia", "England"],
            "H": ["Poland", "Senegal", "Colombia", "Japan"],
        },
        actual={
            "champion": "France",
            "finalist": ["France", "Croatia"],
            "semifinalists": ["France", "Croatia", "Belgium", "England"],
            "quarterfinalists": [
                "Uruguay", "France", "Brazil", "Belgium",
                "Sweden", "England", "Russia", "Croatia",
            ],
            "group_qualifiers": [
                "Uruguay", "Russia", "Spain", "Portugal",
                "France", "Denmark", "Croatia", "Argentina",
                "Brazil", "Switzerland", "Sweden", "Mexico",
                "Belgium", "England", "Colombia", "Japan",
            ],
        },
    ),
    HistoricalTournament(
        year=2022,
        start_date="2022-11-20",
        groups={
            "A": ["Qatar", "Ecuador", "Senegal", "Netherlands"],
            "B": ["England", "Iran", "United States", "Wales"],
            "C": ["Argentina", "Saudi Arabia", "Mexico", "Poland"],
            "D": ["France", "Australia", "Denmark", "Tunisia"],
            "E": ["Spain", "Costa Rica", "Germany", "Japan"],
            "F": ["Belgium", "Canada", "Morocco", "Croatia"],
            "G": ["Brazil", "Serbia", "Switzerland", "Cameroon"],
            "H": ["Portugal", "Ghana", "Uruguay", "South Korea"],
        },
        actual={
            "champion": "Argentina",
            "finalist": ["Argentina", "France"],
            "semifinalists": ["Argentina", "France", "Croatia", "Morocco"],
            "quarterfinalists": [
                "Netherlands", "Argentina", "Croatia", "Brazil",
                "England", "France", "Morocco", "Portugal",
            ],
            "group_qualifiers": [
                "Netherlands", "Senegal", "England", "United States",
                "Argentina", "Poland", "France", "Australia",
                "Japan", "Spain", "Morocco", "Croatia",
                "Brazil", "Switzerland", "Portugal", "South Korea",
            ],
        },
    ),
]


@dataclass
class HistoricalSimulationResults:
    n_sims: int
    teams: List[str]
    champion_counts: np.ndarray
    finalist_counts: np.ndarray
    semifinal_counts: np.ndarray
    quarterfinal_counts: np.ndarray
    round_of_16_counts: np.ndarray
    group_exit_counts: np.ndarray
    elapsed_s: float

    def to_dataframe(self) -> pd.DataFrame:
        n = self.n_sims
        df = pd.DataFrame({
            "team": self.teams,
            "champion_pct": self.champion_counts / n * 100,
            "finalist_pct": self.finalist_counts / n * 100,
            "semifinal_pct": self.semifinal_counts / n * 100,
            "quarterfinal_pct": self.quarterfinal_counts / n * 100,
            "round_of_16_pct": self.round_of_16_counts / n * 100,
            "group_exit_pct": self.group_exit_counts / n * 100,
        })
        return df.sort_values("champion_pct", ascending=False).reset_index(drop=True)


class HistoricalTournamentSimulator:
    """32-team World Cup simulator using the production probability matrix."""

    def __init__(self, prob_matrix_builder: ProbabilityMatrixBuilder, groups: Dict[str, List[str]]) -> None:
        self.pmb = prob_matrix_builder
        self.groups = groups
        self.teams = [team for group in groups.values() for team in group]
        self.team_idx = {team: idx for idx, team in enumerate(self.teams)}
        self.group_names = sorted(groups)
        self.group_local_ids = {
            group: np.array([self.team_idx[team] for team in teams], dtype=np.int32)
            for group, teams in groups.items()
        }

        global_idxs = np.array([self.pmb.team_idx[t] for t in self.teams], dtype=np.int32)
        n = len(self.teams)
        self.p_win = np.zeros((n, n), dtype=np.float32)
        self.p_draw = np.zeros((n, n), dtype=np.float32)
        self.p_ko = np.zeros((n, n), dtype=np.float32)
        for li, gi in enumerate(global_idxs):
            for lj, gj in enumerate(global_idxs):
                if li == lj:
                    continue
                self.p_win[li, lj] = self.pmb.P_win[gi, gj]
                self.p_draw[li, lj] = self.pmb.P_draw[gi, gj]
                self.p_ko[li, lj] = self.pmb.P_ko[gi, gj]

    def _sim_group_stage(
        self, rng: np.random.Generator, n_sims: int
    ) -> Dict[str, Dict[str, np.ndarray]]:
        tables: Dict[str, Dict[str, np.ndarray]] = {}
        for group, ids in self.group_local_ids.items():
            pts = np.zeros((n_sims, 4), dtype=np.int16)
            gd = np.zeros((n_sims, 4), dtype=np.int16)
            gs = np.zeros((n_sims, 4), dtype=np.int16)
            for home_pos in range(4):
                for away_pos in range(home_pos + 1, 4):
                    home_id = int(ids[home_pos])
                    away_id = int(ids[away_pos])
                    u = rng.random(n_sims, dtype=np.float32)
                    p_w = float(self.p_win[home_id, away_id])
                    p_d = float(self.p_draw[home_id, away_id])
                    home_win = u < p_w
                    draw = (u >= p_w) & (u < p_w + p_d)
                    away_win = ~home_win & ~draw

                    pts[:, home_pos] += (3 * home_win + draw).astype(np.int16)
                    pts[:, away_pos] += (3 * away_win + draw).astype(np.int16)
                    gd[:, home_pos] += home_win.astype(np.int16) - away_win.astype(np.int16)
                    gd[:, away_pos] += away_win.astype(np.int16) - home_win.astype(np.int16)

                    gsh = rng.integers(0, 5, n_sims, dtype=np.int16)
                    gsa = rng.integers(0, 5, n_sims, dtype=np.int16)
                    gs[:, home_pos] += gsh
                    gs[:, away_pos] += gsa

            tables[group] = {"pts": pts, "gd": gd, "gs": gs, "ids": ids}
        return tables

    @staticmethod
    def _rank_group(pts: np.ndarray, gd: np.ndarray, gs: np.ndarray, ids: np.ndarray) -> np.ndarray:
        order = np.lexsort((-gs, -gd, -pts), axis=1)
        return ids[order]

    def _knockout_round(self, rng: np.random.Generator, matchups: np.ndarray) -> np.ndarray:
        n = matchups.shape[0]
        k = matchups.shape[1] // 2
        winners = np.empty((n, k), dtype=np.int32)
        for match in range(k):
            a = matchups[:, 2 * match]
            b = matchups[:, 2 * match + 1]
            probs = self.p_ko[a, b]
            winners[:, match] = np.where(rng.random(n, dtype=np.float32) < probs, a, b)
        return winners

    def run(self, n_sims: int) -> HistoricalSimulationResults:
        rng = np.random.default_rng(2026)
        start = time.perf_counter()
        n_teams = len(self.teams)

        champion_counts = np.zeros(n_teams, dtype=np.int32)
        finalist_counts = np.zeros(n_teams, dtype=np.int32)
        semifinal_counts = np.zeros(n_teams, dtype=np.int32)
        quarterfinal_counts = np.zeros(n_teams, dtype=np.int32)
        round_of_16_counts = np.zeros(n_teams, dtype=np.int32)
        group_exit_counts = np.zeros(n_teams, dtype=np.int32)

        tables = self._sim_group_stage(rng, n_sims)
        winners: Dict[str, np.ndarray] = {}
        runners: Dict[str, np.ndarray] = {}
        for group in self.group_names:
            ranked = self._rank_group(
                tables[group]["pts"],
                tables[group]["gd"],
                tables[group]["gs"],
                tables[group]["ids"],
            )
            winners[group] = ranked[:, 0]
            runners[group] = ranked[:, 1]
            np.add.at(group_exit_counts, ranked[:, 2], 1)
            np.add.at(group_exit_counts, ranked[:, 3], 1)
            np.add.at(round_of_16_counts, ranked[:, 0], 1)
            np.add.at(round_of_16_counts, ranked[:, 1], 1)

        r16 = np.column_stack([
            winners["A"], runners["B"],
            winners["C"], runners["D"],
            winners["E"], runners["F"],
            winners["G"], runners["H"],
            winners["B"], runners["A"],
            winners["D"], runners["C"],
            winners["F"], runners["E"],
            winners["H"], runners["G"],
        ])
        qf = self._knockout_round(rng, r16)
        for col in range(qf.shape[1]):
            np.add.at(quarterfinal_counts, qf[:, col], 1)

        sf_matchups = np.column_stack([
            qf[:, 0], qf[:, 1],
            qf[:, 2], qf[:, 3],
            qf[:, 4], qf[:, 5],
            qf[:, 6], qf[:, 7],
        ])
        sf = self._knockout_round(rng, sf_matchups)
        for col in range(sf.shape[1]):
            np.add.at(semifinal_counts, sf[:, col], 1)

        final_matchups = np.column_stack([
            sf[:, 0], sf[:, 1],
            sf[:, 2], sf[:, 3],
        ])
        finalists = self._knockout_round(rng, final_matchups)
        for col in range(finalists.shape[1]):
            np.add.at(finalist_counts, finalists[:, col], 1)

        final = np.column_stack([finalists[:, 0], finalists[:, 1]])
        champions = self._knockout_round(rng, final)
        np.add.at(champion_counts, champions[:, 0], 1)

        return HistoricalSimulationResults(
            n_sims=n_sims,
            teams=self.teams,
            champion_counts=champion_counts,
            finalist_counts=finalist_counts,
            semifinal_counts=semifinal_counts,
            quarterfinal_counts=quarterfinal_counts,
            round_of_16_counts=round_of_16_counts,
            group_exit_counts=group_exit_counts,
            elapsed_s=time.perf_counter() - start,
        )


def _build_pre_tournament_state(cutoff_date: str) -> Tuple[EloTracker, Dict[str, float], Dict[str, float], Dict[str, float], H2HRegistry, Dict[str, List[str]]]:
    raw = load_raw_results()
    matches = build_match_records(raw)
    cutoff = pd.Timestamp(cutoff_date)
    history = matches[matches["date"] < cutoff].copy()

    elo = EloTracker()
    elo.batch_update(history)

    form_tracker = FormTracker(window=10)
    h2h = H2HRegistry()
    opponent_history: Dict[str, List[str]] = {}
    for _, row in history.iterrows():
        home = str(row["home_team"])
        away = str(row["away_team"])
        hs = int(row["home_score"])
        as_ = int(row["away_score"])
        form_tracker.update(home, away, hs, as_)
        h2h.update(home, away, hs, as_)
        opponent_history.setdefault(home, []).append(away)
        opponent_history.setdefault(away, []).append(home)

    form_scores = form_tracker.snapshot()
    form_details = form_tracker.snapshot_full_form()
    form_gd = form_tracker.snapshot_gd()

    opp_adj: Dict[str, float] = {}
    for team, own_form in form_scores.items():
        recent_opponents = opponent_history.get(team, [])[-form_tracker.window :]
        if not recent_opponents:
            opp_adj[team] = 0.0
            continue
        opp_mean = sum(form_scores.get(opponent, 0.33) for opponent in recent_opponents) / len(recent_opponents)
        opp_adj[team] = own_form - opp_mean

    return elo, form_scores, opp_adj, form_gd, h2h, form_details


def _advancement_brier(team_probs: pd.Series, actual_teams: List[str]) -> float:
    actual = team_probs.index.to_series().isin(actual_teams).astype(float).to_numpy()
    pred = team_probs.to_numpy() / 100.0
    return float(np.mean((pred - actual) ** 2))


def _stage_top_accuracy(results_df: pd.DataFrame, column: str, actual_teams: List[str], top_k: int) -> float:
    predicted = set(results_df.nlargest(top_k, column)["team"])
    actual = set(actual_teams)
    return len(predicted & actual) / top_k


def _build_report(summary: Dict[str, object]) -> str:
    lines = [
        _LINE,
        "  HISTORICAL WORLD CUP BACKTEST  —  Production Model",
        _LINE,
    ]
    for tournament in summary["tournaments"]:
        lines.extend([
            f"  {tournament['year']} FIFA World Cup",
            f"    Champion probability on actual winner ({tournament['actual_champion']}): "
            f"{tournament['winner_probability_for_actual_champion']:.2f}%",
            f"    Group qualification accuracy : {tournament['group_stage_qualification_accuracy']:.3f}",
            f"    Quarterfinalist accuracy     : {tournament['quarterfinalist_accuracy']:.3f}",
            f"    Semifinalist accuracy        : {tournament['semifinalist_accuracy']:.3f}",
            f"    Advancement Brier score      : {tournament['advancement_brier_score']:.4f}",
            f"    Model strengths              : {tournament['strengths']}",
            f"    Model failures               : {tournament['failures']}",
            "",
        ])

    aggregate = summary["aggregate"]
    lines.extend([
        "  Aggregate Summary",
        f"    Mean champion probability on actual winners : "
        f"{aggregate['mean_winner_probability_for_actual_champion']:.2f}%",
        f"    Mean group qualification accuracy           : "
        f"{aggregate['mean_group_stage_qualification_accuracy']:.3f}",
        f"    Mean quarterfinalist accuracy               : "
        f"{aggregate['mean_quarterfinalist_accuracy']:.3f}",
        f"    Mean semifinalist accuracy                  : "
        f"{aggregate['mean_semifinalist_accuracy']:.3f}",
        f"    Mean advancement Brier score                : "
        f"{aggregate['mean_advancement_brier_score']:.4f}",
        _LINE,
    ])
    return "\n".join(lines)


def run_world_cup_backtest(
    n_sims: int = N_SIMULATIONS,
    model_path: Path | None = None,
) -> Dict[str, object]:
    model = MatchOutcomeModel()
    model.load() if model_path is None else model.load()

    tournaments_summary = []

    for tournament in HISTORICAL_WORLD_CUPS:
        log.info("Backtesting %d World Cup using cutoff %s", tournament.year, tournament.start_date)
        elo, form_scores, opp_adj_form, form_gd, h2h, form_details = _build_pre_tournament_state(
            tournament.start_date
        )

        teams = [team for group in tournament.groups.values() for team in group]
        pmb = ProbabilityMatrixBuilder(model)
        pmb.build(
            teams=teams,
            elo_tracker=elo,
            form_scores=form_scores,
            opp_adj_form_scores=opp_adj_form,
            form_gd=form_gd,
            h2h_fn=h2h.win_rate_diff,
            form_details=form_details,
        )

        simulator = HistoricalTournamentSimulator(pmb, tournament.groups)
        results = simulator.run(n_sims=n_sims)
        results_df = results.to_dataframe()

        actual_champion = str(tournament.actual["champion"])
        actual_group_qualifiers = list(tournament.actual["group_qualifiers"])
        actual_quarterfinalists = list(tournament.actual["quarterfinalists"])
        actual_semifinalists = list(tournament.actual["semifinalists"])

        champion_prob = float(
            results_df.loc[results_df["team"] == actual_champion, "champion_pct"].iloc[0]
        )
        group_acc = _stage_top_accuracy(results_df, "round_of_16_pct", actual_group_qualifiers, 16)
        qf_acc = _stage_top_accuracy(results_df, "quarterfinal_pct", actual_quarterfinalists, 8)
        sf_acc = _stage_top_accuracy(results_df, "semifinal_pct", actual_semifinalists, 4)

        group_brier = _advancement_brier(results_df.set_index("team")["round_of_16_pct"], actual_group_qualifiers)
        qf_brier = _advancement_brier(results_df.set_index("team")["quarterfinal_pct"], actual_quarterfinalists)
        sf_brier = _advancement_brier(results_df.set_index("team")["semifinal_pct"], actual_semifinalists)
        champ_brier = _advancement_brier(results_df.set_index("team")["champion_pct"], [actual_champion])
        advancement_brier = float(np.mean([group_brier, qf_brier, sf_brier, champ_brier]))

        predicted_group = set(results_df.nlargest(16, "round_of_16_pct")["team"])
        predicted_qf = set(results_df.nlargest(8, "quarterfinal_pct")["team"])
        predicted_sf = set(results_df.nlargest(4, "semifinal_pct")["team"])
        strengths = []
        failures = []
        if actual_champion in set(results_df.nlargest(5, "champion_pct")["team"]):
            strengths.append("eventual champion placed in top-5 winner probabilities")
        else:
            failures.append("eventual champion was outside the top-5 winner probabilities")
        if len(predicted_group & set(actual_group_qualifiers)) >= 12:
            strengths.append("group-stage qualification signal was strong")
        else:
            failures.append("group-stage qualification misses were material")
        if len(predicted_sf & set(actual_semifinalists)) >= 2:
            strengths.append("semi-finalist shortlist captured at least half of actual semi-finalists")
        else:
            failures.append("semi-finalist shortlist missed most of the eventual final four")
        if len(predicted_qf & set(actual_quarterfinalists)) < 4:
            failures.append("quarterfinal field was less stable than simulated probabilities suggested")

        tournaments_summary.append({
            "year": tournament.year,
            "actual_champion": actual_champion,
            "winner_probability_for_actual_champion": champion_prob,
            "group_stage_qualification_accuracy": group_acc,
            "quarterfinalist_accuracy": qf_acc,
            "semifinalist_accuracy": sf_acc,
            "advancement_brier_score": advancement_brier,
            "simulation_runtime_s": results.elapsed_s,
            "top5_winner_probabilities": (
                results_df[["team", "champion_pct"]].head(5).to_dict(orient="records")
            ),
            "strengths": strengths or ["none noted"],
            "failures": failures or ["none noted"],
        })

    aggregate = {
        "mean_winner_probability_for_actual_champion": float(np.mean([
            t["winner_probability_for_actual_champion"] for t in tournaments_summary
        ])),
        "mean_group_stage_qualification_accuracy": float(np.mean([
            t["group_stage_qualification_accuracy"] for t in tournaments_summary
        ])),
        "mean_quarterfinalist_accuracy": float(np.mean([
            t["quarterfinalist_accuracy"] for t in tournaments_summary
        ])),
        "mean_semifinalist_accuracy": float(np.mean([
            t["semifinalist_accuracy"] for t in tournaments_summary
        ])),
        "mean_advancement_brier_score": float(np.mean([
            t["advancement_brier_score"] for t in tournaments_summary
        ])),
    }

    summary = {"tournaments": tournaments_summary, "aggregate": aggregate}
    report = _build_report(summary)
    _BACKTEST_TXT.write_text(report, encoding="utf-8")
    _BACKTEST_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Historical backtest report saved → %s", _BACKTEST_TXT)
    return summary


# ==============================================================================
# True Out-of-Sample Backtest
# ==============================================================================

def _opp_adjusted_form(
    own_form: float,
    recent_opponents: List[str],
    ft: FormTracker,
) -> float:
    """Opponent-adjusted form: own_form minus mean form of recent opponents."""
    if not recent_opponents:
        return 0.0
    opp_mean = sum(ft.get_form_score(o) for o in recent_opponents) / len(recent_opponents)
    return own_form + opp_mean - 1.0


def _build_backtest_training_dataset(
    cutoff_date: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build (X, y) using only matches that predate cutoff_date.

    Replicates the production feature pipeline exactly:
      - FormTracker, H2HRegistry, EloTracker all rebuilt from scratch
      - Features captured pre-match (no leakage)
      - Elo replay mirrors build_training_features logic
      - Only matches between WC2026 teams in [TRAINING_START, cutoff_date)
        contribute training rows (identical filter to MatchDataPipeline)

    Parameters
    ----------
    cutoff_date : ISO date string — tournament start date (exclusive upper bound)

    Returns
    -------
    (X, y) where X has exactly the columns in FEATURE_COLS and y is the
    3-class outcome label (0=home win, 1=draw, 2=away win).
    """
    cutoff         = pd.Timestamp(cutoff_date)
    training_start = pd.Timestamp(TRAINING_START)
    wc_csv_names   = set(TEAM_NAME_ALIASES.values())

    raw         = load_raw_results()
    all_matches = build_match_records(raw)
    pre_cutoff  = all_matches[all_matches["date"] < cutoff].copy()

    form_tracker  = FormTracker(window=10)
    h2h_reg       = H2HRegistry()
    opp_history: Dict[str, List[str]] = {}
    training_rows: list = []

    for _, row in pre_cutoff.iterrows():
        h   = str(row["home_team"])
        a   = str(row["away_team"])
        hs  = int(row["home_score"])
        as_ = int(row["away_score"])
        date = row["date"]

        # Capture pre-match state only for Scenario 1 filter (at least one WC team)
        if (date >= training_start
                and (h in wc_csv_names or a in wc_csv_names)):

            h_can = ALIAS_REVERSE.get(h, h)
            a_can = ALIAS_REVERSE.get(a, a)

            h_w, h_d, h_l, h_gd = form_tracker.get_form(h)
            a_w, a_d, a_l, a_gd = form_tracker.get_form(a)
            h_form = form_tracker.get_form_score(h)
            a_form = form_tracker.get_form_score(a)

            recent_opp_h = opp_history.get(h, [])[-form_tracker.window:]
            recent_opp_a = opp_history.get(a, [])[-form_tracker.window:]

            opp_adj_h = _opp_adjusted_form(h_form, recent_opp_h, form_tracker)
            opp_adj_a = _opp_adjusted_form(a_form, recent_opp_a, form_tracker)

            training_rows.append({
                "date":             date,
                "home_team":        h_can,
                "away_team":        a_can,
                "tournament":       str(row["tournament"]),
                "neutral":          bool(row["neutral"]),
                "target":           int(row["target"]),
                "h2h_diff":         h2h_reg.win_rate_diff(h, a),
                "home_form":        h_form,
                "away_form":        a_form,
                "opp_adj_form_diff": opp_adj_h - opp_adj_a,
                "gd_diff":          h_gd - a_gd,
                "home_win_rate":    h_w,
                "away_win_rate":    a_w,
                "home_draw_rate":   h_d,
                "away_draw_rate":   a_d,
                "home_loss_rate":   h_l,
            })

        # -- Update state AFTER capture -- no leakage ----------------------------
        form_tracker.update(h, a, hs, as_)
        h2h_reg.update(h, a, hs, as_)
        opp_history.setdefault(h, []).append(a)
        opp_history.setdefault(a, []).append(h)

    if not training_rows:
        raise ValueError(
            f"No training samples found in [{TRAINING_START}, {cutoff_date}). "
            f"Ensure results.csv covers this period."
        )

    tdf = pd.DataFrame(training_rows).sort_values("date").reset_index(drop=True)

    # -- Elo replay -- mirrors build_training_features exactly ------------------
    replay_elo     = EloTracker()
    pre_train_hist = pre_cutoff[pre_cutoff["date"] < training_start]
    replay_elo.batch_update(pre_train_hist)  # seed with pre-2010 history

    elo_diffs: List[float] = []
    for _, row in tdf.iterrows():
        h_csv = TEAM_NAME_ALIASES.get(str(row["home_team"]), str(row["home_team"]))
        a_csv = TEAM_NAME_ALIASES.get(str(row["away_team"]), str(row["away_team"]))
        rh = replay_elo._get(h_csv)
        ra = replay_elo._get(a_csv)
        elo_diffs.append(rh - ra)
        replay_elo.update(
            home_team  = h_csv,
            away_team  = a_csv,
            outcome    = int(row["target"]),
            tournament = str(row["tournament"]),
            neutral    = bool(row["neutral"]),
        )

    tdf["elo_diff"]       = elo_diffs
    tdf["form_diff"]      = tdf["home_form"].astype(float) - tdf["away_form"].astype(float)
    tdf["match_type_enc"] = tdf["tournament"].map(tournament_importance).fillna(0.40)
    tdf["neutral_venue"]  = tdf["neutral"].astype(float)

    missing = [c for c in FEATURE_COLS if c not in tdf.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    log.info(
        "[OOS] Training dataset for cutoff=%s: %d samples × %d features  "
        "(class dist: 0=%d 1=%d 2=%d)",
        cutoff_date, len(tdf), len(FEATURE_COLS),
        (tdf["target"] == 0).sum(),
        (tdf["target"] == 1).sum(),
        (tdf["target"] == 2).sum(),
    )
    return tdf[FEATURE_COLS].copy(), tdf["target"].astype(int)


def _build_oos_report(summary: Dict[str, object]) -> str:
    lines = [
        "=" * 82,
        "  TRUE OUT-OF-SAMPLE WORLD CUP BACKTEST",
        "  (Each model trained exclusively on pre-tournament data)",
        "=" * 82,
    ]
    for t in summary["tournaments"]:
        lines.extend([
            f"  {t['year']} FIFA World Cup  "
            f"[{t['training_samples']} training samples]",
            f"    Champion probability on actual winner ({t['actual_champion']}): "
            f"{t['winner_probability_for_actual_champion']:.2f}%",
            f"    Group qualification accuracy : {t['group_stage_qualification_accuracy']:.3f}",
            f"    Quarterfinalist accuracy     : {t['quarterfinalist_accuracy']:.3f}",
            f"    Semifinalist accuracy        : {t['semifinalist_accuracy']:.3f}",
            f"    Advancement Brier score      : {t['advancement_brier_score']:.4f}",
            f"    Strengths : {t['strengths']}",
            f"    Failures  : {t['failures']}",
            "",
        ])
    agg = summary["aggregate"]
    lines.extend([
        "  Aggregate",
        f"    Mean champion probability  : {agg['mean_winner_probability_for_actual_champion']:.2f}%",
        f"    Mean group accuracy        : {agg['mean_group_stage_qualification_accuracy']:.3f}",
        f"    Mean QF accuracy           : {agg['mean_quarterfinalist_accuracy']:.3f}",
        f"    Mean SF accuracy           : {agg['mean_semifinalist_accuracy']:.3f}",
        f"    Mean Brier score           : {agg['mean_advancement_brier_score']:.4f}",
        "=" * 82,
    ])
    return "\n".join(lines)


def _build_comparison_report(
    oos: Dict[str, object],
    contaminated: Dict[str, object],
) -> str:
    """
    Side-by-side comparison table: contaminated vs out-of-sample backtest.
    """
    W = 16   # column width

    def row(label: str, cont_val: str, oos_val: str, delta: str) -> str:
        return (
            f"  {label:<34}  {cont_val:>{W}}  {oos_val:>{W}}  {delta:>{W}}"
        )

    def pct_row(label, cont_f, oos_f):
        d = oos_f - cont_f
        sign = "+" if d >= 0 else ""
        return row(label, f"{cont_f:.2f}%", f"{oos_f:.2f}%", f"{sign}{d:.2f}pp")

    def acc_row(label, cont_f, oos_f):
        d = oos_f - cont_f
        sign = "+" if d >= 0 else ""
        return row(label, f"{cont_f:.3f}", f"{oos_f:.3f}", f"{sign}{d:.3f}")

    lines = [
        "=" * 82,
        "  CONTAMINATED vs TRUE OUT-OF-SAMPLE BACKTEST -- COMPARISON",
        "=" * 82,
        row("Metric", "Contaminated", "Out-of-Sample", "D (OOS - Cont)"),
        "  " + "-" * 78,
    ]

    cont_ts = {t["year"]: t for t in contaminated["tournaments"]}
    oos_ts  = {t["year"]: t for t in oos["tournaments"]}

    for year in sorted(cont_ts):
        c = cont_ts[year]
        o = oos_ts[year]
        lines.append(f"  -- WC {year}  (actual champion: {c['actual_champion']}, "
                     f"OOS training set: {o['training_samples']} rows) --")
        lines.append(pct_row("  Champion probability",
                             c["winner_probability_for_actual_champion"],
                             o["winner_probability_for_actual_champion"]))
        lines.append(acc_row("  Group qualification accuracy",
                             c["group_stage_qualification_accuracy"],
                             o["group_stage_qualification_accuracy"]))
        lines.append(acc_row("  Quarterfinal accuracy",
                             c["quarterfinalist_accuracy"],
                             o["quarterfinalist_accuracy"]))
        lines.append(acc_row("  Semifinal accuracy",
                             c["semifinalist_accuracy"],
                             o["semifinalist_accuracy"]))
        lines.append(acc_row("  Brier score (lower = better)",
                             c["advancement_brier_score"],
                             o["advancement_brier_score"]))
        lines.append("")

    ca = contaminated["aggregate"]
    oa = oos["aggregate"]
    lines.append("  -- AGGREGATE --")
    lines.append(pct_row("  Mean champion probability",
                         ca["mean_winner_probability_for_actual_champion"],
                         oa["mean_winner_probability_for_actual_champion"]))
    lines.append(acc_row("  Mean group accuracy",
                         ca["mean_group_stage_qualification_accuracy"],
                         oa["mean_group_stage_qualification_accuracy"]))
    lines.append(acc_row("  Mean QF accuracy",
                         ca["mean_quarterfinalist_accuracy"],
                         oa["mean_quarterfinalist_accuracy"]))
    lines.append(acc_row("  Mean SF accuracy",
                         ca["mean_semifinalist_accuracy"],
                         oa["mean_semifinalist_accuracy"]))
    lines.append(acc_row("  Mean Brier score",
                         ca["mean_advancement_brier_score"],
                         oa["mean_advancement_brier_score"]))
    lines.append("")
    lines.append("  Negative D on accuracy = OOS is harder (good -- this is expected).")
    lines.append("  Positive D on Brier    = OOS Brier is higher (worse = expected).")
    lines.append("=" * 82)
    return "\n".join(lines)


def run_oos_world_cup_backtest(
    n_sims: int = N_SIMULATIONS,
    compare: bool = False,
) -> Dict[str, object]:
    """
    True out-of-sample historical World Cup backtest.

    For each of the three historical World Cups (2014, 2018, 2022) a fresh
    XGBoost model is trained using **only** matches that predated the
    tournament start date.  The production model is never loaded, used,
    or modified.

    Parameters
    ----------
    n_sims  : number of Monte Carlo tournament simulations per year
    compare : if True, load the contaminated backtest results (if available)
              and print a side-by-side comparison table

    Returns
    -------
    OOS summary dict (same schema as run_world_cup_backtest)
    """
    log.info("Starting TRUE out-of-sample World Cup backtest (%d sims/year)", n_sims)
    tournaments_summary = []

    for tournament in HISTORICAL_WORLD_CUPS:
        log.info(
            "[OOS] %d WC — building training dataset (cutoff: %s) …",
            tournament.year, tournament.start_date,
        )

        # -- Step 1: train a model on pre-tournament data only ----------------
        X_bt, y_bt = _build_backtest_training_dataset(tournament.start_date)
        model_bt   = MatchOutcomeModel()
        model_bt.fit(X_bt, y_bt)
        # Intentionally NOT saved — this model is backtest-only.

        # -- Step 2: build pre-tournament feature state (time-locked) ---------
        elo, form_scores, opp_adj_form, form_gd, h2h, form_details = (
            _build_pre_tournament_state(tournament.start_date)
        )

        # -- Step 3: probability matrix via the backtest model ----------------
        teams = [t for grp in tournament.groups.values() for t in grp]
        pmb   = ProbabilityMatrixBuilder(model_bt)
        pmb.build(
            teams               = teams,
            elo_tracker         = elo,
            form_scores         = form_scores,
            opp_adj_form_scores = opp_adj_form,
            form_gd             = form_gd,
            h2h_fn              = h2h.win_rate_diff,
            form_details        = form_details,
        )

        # -- Step 4: simulate -------------------------------------------------
        simulator  = HistoricalTournamentSimulator(pmb, tournament.groups)
        results    = simulator.run(n_sims=n_sims)
        results_df = results.to_dataframe()

        # -- Step 5: evaluate -------------------------------------------------
        actual_champion         = str(tournament.actual["champion"])
        actual_group_qualifiers = list(tournament.actual["group_qualifiers"])
        actual_quarterfinalists = list(tournament.actual["quarterfinalists"])
        actual_semifinalists    = list(tournament.actual["semifinalists"])

        champion_prob = float(
            results_df.loc[
                results_df["team"] == actual_champion, "champion_pct"
            ].iloc[0]
        )
        group_acc = _stage_top_accuracy(
            results_df, "round_of_16_pct", actual_group_qualifiers, 16
        )
        qf_acc = _stage_top_accuracy(
            results_df, "quarterfinal_pct", actual_quarterfinalists, 8
        )
        sf_acc = _stage_top_accuracy(
            results_df, "semifinal_pct", actual_semifinalists, 4
        )

        group_brier = _advancement_brier(
            results_df.set_index("team")["round_of_16_pct"], actual_group_qualifiers
        )
        qf_brier = _advancement_brier(
            results_df.set_index("team")["quarterfinal_pct"], actual_quarterfinalists
        )
        sf_brier = _advancement_brier(
            results_df.set_index("team")["semifinal_pct"], actual_semifinalists
        )
        champ_brier = _advancement_brier(
            results_df.set_index("team")["champion_pct"], [actual_champion]
        )
        advancement_brier = float(
            np.mean([group_brier, qf_brier, sf_brier, champ_brier])
        )

        top5      = set(results_df.nlargest(5, "champion_pct")["team"])
        pred_grp  = set(results_df.nlargest(16, "round_of_16_pct")["team"])
        pred_sf   = set(results_df.nlargest(4, "semifinal_pct")["team"])
        pred_qf   = set(results_df.nlargest(8, "quarterfinal_pct")["team"])
        strengths, failures = [], []
        if actual_champion in top5:
            strengths.append("eventual champion placed in top-5 winner probabilities")
        else:
            failures.append("eventual champion outside top-5 winner probabilities")
        if len(pred_grp & set(actual_group_qualifiers)) >= 12:
            strengths.append("group-stage qualification signal was strong")
        else:
            failures.append("group-stage qualification misses were material")
        if len(pred_sf & set(actual_semifinalists)) >= 2:
            strengths.append("semi-finalist shortlist captured ≥ half of actual semi-finalists")
        else:
            failures.append("semi-finalist shortlist missed most of the eventual final four")
        if len(pred_qf & set(actual_quarterfinalists)) < 4:
            failures.append("quarterfinal field was less stable than simulated probabilities suggested")

        log.info(
            "[OOS] %d — champion_prob=%.2f%%  group_acc=%.3f  "
            "qf_acc=%.3f  sf_acc=%.3f  brier=%.4f",
            tournament.year, champion_prob, group_acc, qf_acc, sf_acc, advancement_brier,
        )

        tournaments_summary.append({
            "year":                   tournament.year,
            "actual_champion":        actual_champion,
            "training_samples":       int(len(X_bt)),
            "winner_probability_for_actual_champion": champion_prob,
            "group_stage_qualification_accuracy":     group_acc,
            "quarterfinalist_accuracy":               qf_acc,
            "semifinalist_accuracy":                  sf_acc,
            "advancement_brier_score":                advancement_brier,
            "simulation_runtime_s":                   results.elapsed_s,
            "top5_winner_probabilities": (
                results_df[["team", "champion_pct"]].head(5).to_dict(orient="records")
            ),
            "strengths": strengths or ["none noted"],
            "failures":  failures  or ["none noted"],
        })

    aggregate = {
        "mean_winner_probability_for_actual_champion": float(np.mean([
            t["winner_probability_for_actual_champion"] for t in tournaments_summary
        ])),
        "mean_group_stage_qualification_accuracy": float(np.mean([
            t["group_stage_qualification_accuracy"] for t in tournaments_summary
        ])),
        "mean_quarterfinalist_accuracy": float(np.mean([
            t["quarterfinalist_accuracy"] for t in tournaments_summary
        ])),
        "mean_semifinalist_accuracy": float(np.mean([
            t["semifinalist_accuracy"] for t in tournaments_summary
        ])),
        "mean_advancement_brier_score": float(np.mean([
            t["advancement_brier_score"] for t in tournaments_summary
        ])),
    }

    oos_summary = {"tournaments": tournaments_summary, "aggregate": aggregate}

    # -- Persist OOS results --------------------------------------------------
    oos_report = _build_oos_report(oos_summary)
    _OOS_TXT.write_text(oos_report, encoding="utf-8")
    _OOS_JSON.write_text(json.dumps(oos_summary, indent=2), encoding="utf-8")
    log.info("OOS backtest report saved -> %s", _OOS_TXT)

    # -- Comparison table -----------------------------------------------------
    if compare and _BACKTEST_JSON.exists():
        contaminated = json.loads(_BACKTEST_JSON.read_text())
        comp_report  = _build_comparison_report(oos_summary, contaminated)
        comp_path    = REPORT_DIR / "backtest_comparison.txt"
        comp_path.write_text(comp_report, encoding="utf-8")
        log.info("Comparison report saved -> %s", comp_path)
        print(comp_report)
    elif compare:
        log.warning(
            "Contaminated backtest results not found at %s - "
            "run run_world_cup_backtest() first to enable comparison.",
            _BACKTEST_JSON,
        )

    return oos_summary
