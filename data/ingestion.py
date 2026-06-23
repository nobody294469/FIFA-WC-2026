"""
ingestion.py — Real-data ingestion layer.

Reads the martj42/international_results dataset (results.csv) and:
  1. Normalises team names to canonical WC2026 names
  2. Filters to completed matches only (no NA scores)
  3. Derives home/away/draw outcome labels
  4. Computes rolling 5-match form rates per team (win/draw/loss rates)
  5. Exposes head-to-head win-rate differentials
  6. Provides a clean training DataFrame for the ML engine

NO synthetic data is generated anywhere in this module.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from core.config import (
    ALIAS_REVERSE,
    RESULTS_CSV,
    TEAM_NAME_ALIASES,
    TRAINING_START,
    WC2026_TEAMS,
)
from core.logger import get_logger

log = get_logger("ingestion")

# Columns we need from results.csv
_REQUIRED_COLS = {"date", "home_team", "away_team", "home_score", "away_score", "tournament", "neutral"}


# ═══════════════════════════════════════════════════════════════════════════════
# Low-level loader
# ═══════════════════════════════════════════════════════════════════════════════

def load_raw_results(path: Path = RESULTS_CSV) -> pd.DataFrame:
    """
    Load results.csv, validate schema, drop future/incomplete rows,
    parse dates, and return a clean DataFrame.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"results.csv not found at {path}.\n"
            "Download it from: https://github.com/martj42/international_results\n"
            "and place it at data/results.csv"
        )

    df = pd.read_csv(path, parse_dates=["date"])
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"results.csv is missing expected columns: {missing}")

    # Drop rows where scores are missing (future fixtures, forfeits, etc.)
    before = len(df)
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    after = len(df)
    log.info("Loaded %d completed matches (dropped %d future/incomplete rows)", after, before - after)

    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"]    = df["neutral"].map({"TRUE": True, "FALSE": False, True: True, False: False})
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _normalise_name(name: str) -> Optional[str]:
    """
    Map a results.csv team name to the canonical WC2026 team name.
    Returns None if the team is not one of the 48 WC2026 teams.
    """
    # Direct reverse alias lookup
    if name in ALIAS_REVERSE:
        return ALIAS_REVERSE[name]
    # Already a canonical name
    if name in TEAM_NAME_ALIASES:
        return name
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Match record processor
# ═══════════════════════════════════════════════════════════════════════════════

def build_match_records(raw: pd.DataFrame) -> pd.DataFrame:
    """
    From the raw results DataFrame, build a clean match-level record table
    with:
      - canonical team names
      - outcome label (0=home win, 1=draw, 2=away win)
      - goal difference
      - tournament string (kept for Elo K-factor lookup)
      - neutral venue flag

    Only rows where BOTH teams are known (not necessarily WC2026 teams, since
    we want full history to build Elo) are retained. The target column is added.
    """
    df = raw.copy()

    # Outcome label
    df["target"] = np.where(
        df["home_score"] > df["away_score"], 0,
        np.where(df["home_score"] == df["away_score"], 1, 2)
    ).astype(np.int8)

    df["goal_diff"] = df["home_score"] - df["away_score"]

    # Keep a copy of the raw names for Elo (we use ALL teams, not just WC ones)
    df = df[["date", "home_team", "away_team", "home_score", "away_score",
             "goal_diff", "tournament", "neutral", "target"]].copy()
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Form computer — rolling N-match form per team
# ═══════════════════════════════════════════════════════════════════════════════

class FormTracker:
    """
    Computes a rolling-window form score for each team from real match results.

    For each team we maintain an ordered list of outcomes (W/D/L from that
    team's perspective) and compute:
      - recent_win_rate   : fraction of last N games that were wins
      - recent_draw_rate  : fraction of last N games that were draws
      - recent_loss_rate  : fraction of last N games that were losses
      - recent_gd_avg     : average goal difference over last N games

    The form state at any date reflects only matches BEFORE that date
    (no data leakage).
    """

    def __init__(self, window: int = 10) -> None:
        self.window = window
        # {team_name: deque of (win_flag, draw_flag, loss_flag, gd)}
        self._records: Dict[str, list] = {}

    def _ensure(self, team: str) -> None:
        if team not in self._records:
            self._records[team] = []

    def update(self, home: str, away: str, home_score: int, away_score: int) -> None:
        """Record a completed match. Call in chronological order."""
        self._ensure(home)
        self._ensure(away)
        gd = home_score - away_score

        if home_score > away_score:
            self._records[home].append((1, 0, 0, gd))
            self._records[away].append((0, 0, 1, -gd))
        elif home_score == away_score:
            self._records[home].append((0, 1, 0, 0))
            self._records[away].append((0, 1, 0, 0))
        else:
            self._records[home].append((0, 0, 1, gd))
            self._records[away].append((1, 0, 0, -gd))

        # Keep only last window * 3 (generous buffer; we trim on read)
        if len(self._records[home]) > self.window * 3:
            self._records[home] = self._records[home][-self.window * 3:]
        if len(self._records[away]) > self.window * 3:
            self._records[away] = self._records[away][-self.window * 3:]

    def get_form(self, team: str) -> Tuple[float, float, float, float]:
        """
        Returns (win_rate, draw_rate, loss_rate, avg_gd) over the last N matches.
        If fewer than N matches exist, uses all available.
        """
        recs = self._records.get(team, [])
        recent = recs[-self.window:] if len(recs) >= self.window else recs
        if not recent:
            return 0.33, 0.33, 0.33, 0.0

        n = len(recent)
        wins  = sum(r[0] for r in recent) / n
        draws = sum(r[1] for r in recent) / n
        losses= sum(r[2] for r in recent) / n
        avg_gd= sum(r[3] for r in recent) / n
        return wins, draws, losses, avg_gd

    def get_form_score(self, team: str) -> float:
        """Scalar form score: win=3pts, draw=1pt, normalised to [0,1]."""
        w, d, l, _ = self.get_form(team)
        return (3 * w + d) / 3.0

    def snapshot(self) -> Dict[str, float]:
        """Return current form scores for all tracked teams."""
        return {t: self.get_form_score(t) for t in self._records}

    def snapshot_gd(self) -> Dict[str, float]:
        """Return current average goal difference for all tracked teams."""
        result = {}
        for t in self._records:
            _, _, _, avg_gd = self.get_form(t)
            result[t] = avg_gd
        return result
    
    def snapshot_full_form(self) -> Dict[str, tuple]:
        """
    Returns:
    {
        team: (win_rate, draw_rate, loss_rate, avg_gd)
    }
    """
        return {t: self.get_form(t) for t in self._records}

# ═══════════════════════════════════════════════════════════════════════════════
# Head-to-head registry
# ═══════════════════════════════════════════════════════════════════════════════

class H2HRegistry:
    """
    Maintains head-to-head win rates between teams from real results.
    All lookups are direction-aware: h2h_win_rate(A, B) = fraction of
    A vs B historical matches won by A.
    """

    def __init__(self) -> None:
        # (team_a, team_b) → [wins_by_a, total]
        self._data: Dict[Tuple[str, str], list] = {}

    def _key(self, a: str, b: str) -> Tuple[str, str]:
        return (a, b)

    def update(self, home: str, away: str, home_score: int, away_score: int) -> None:
        h_key = self._key(home, away)
        a_key = self._key(away, home)
        for k in (h_key, a_key):
            if k not in self._data:
                self._data[k] = [0, 0]

        self._data[h_key][1] += 1
        self._data[a_key][1] += 1

        if home_score > away_score:
            self._data[h_key][0] += 1
        elif away_score > home_score:
            self._data[a_key][0] += 1
        # draw: neither team's win count increments

    def win_rate(self, team_a: str, team_b: str) -> float:
        """Fraction of historical matches won by team_a against team_b."""
        data = self._data.get(self._key(team_a, team_b))
        if not data or data[1] == 0:
            return 0.5   # neutral prior
        return data[0] / data[1]

    def win_rate_diff(self, home: str, away: str) -> float:
        """win_rate(home vs away) − win_rate(away vs home)"""
        return self.win_rate(home, away) - self.win_rate(away, home)


# ═══════════════════════════════════════════════════════════════════════════════
# Main data pipeline class
# ═══════════════════════════════════════════════════════════════════════════════

class MatchDataPipeline:
    """
    Loads and processes all historical data. Exposes:
      - self.matches        : full cleaned DataFrame (all teams, all eras)
      - self.training_df    : modern-era (≥ TRAINING_START) for ML training
      - self.form_tracker   : FormTracker with final state after all history
      - self.h2h            : H2HRegistry with final state after all history
      - self.wc_team_set    : set of results.csv names for the 48 WC2026 teams
    """

    def __init__(self) -> None:
        self.raw         = load_raw_results()
        self.matches     = build_match_records(self.raw)
        self.form_tracker= FormTracker(window=10)
        self.h2h         = H2HRegistry()
        self._opponent_history: Dict[str, list] = {}

        # Build the set of results.csv team names that correspond to WC2026 teams
        self.wc_csv_names: set = set(TEAM_NAME_ALIASES.values())

        self._process_chronological()
        log.info("MatchDataPipeline ready. Training rows: %d", len(self.training_df))

    def _process_chronological(self) -> None:
        """
        Walk all matches chronologically:
          1. Update FormTracker and H2HRegistry for every match
          2. For training rows (≥ TRAINING_START, both teams = WC2026),
             record the form/elo snapshot at that moment (before updating)
        """
        cutoff = pd.Timestamp(TRAINING_START)
        training_rows = []

        for _, row in self.matches.iterrows():
            h = row["home_team"]
            a = row["away_team"]
            hs = int(row["home_score"])
            as_ = int(row["away_score"])
            date = row["date"]

            def opponent_adjusted_form(team: str) -> float:
                recent_opponents = self._opponent_history.get(team, [])[-self.form_tracker.window :]
                own_form = self.form_tracker.get_form_score(team)
                if not recent_opponents:
                    return 0.0
                opp_forms = [self.form_tracker.get_form_score(opponent) for opponent in recent_opponents]
                return own_form + (sum(opp_forms) / len(opp_forms)) - 1.0

            # Capture pre-match form state for training records
            if (date >= cutoff and
                    (h in self.wc_csv_names or
                     a in self.wc_csv_names)):

                h_can = ALIAS_REVERSE.get(h, h)
                a_can = ALIAS_REVERSE.get(a, a)

                h_form = self.form_tracker.get_form_score(h)
                a_form = self.form_tracker.get_form_score(a)
                # Calculate opp mean
                h_recent = self._opponent_history.get(h, [])[-self.form_tracker.window :]
                h_opp_mean = sum(self.form_tracker.get_form_score(o) for o in h_recent) / len(h_recent) if h_recent else 0.0
                
                a_recent = self._opponent_history.get(a, [])[-self.form_tracker.window :]
                a_opp_mean = sum(self.form_tracker.get_form_score(o) for o in a_recent) / len(a_recent) if a_recent else 0.0

                h_opp_adj_form = opponent_adjusted_form(h)
                a_opp_adj_form = opponent_adjusted_form(a)

                h_w, h_d, h_l, h_gd = self.form_tracker.get_form(h)
                a_w, a_d, a_l, a_gd = self.form_tracker.get_form(a)
                h2h_diff = self.h2h.win_rate_diff(h, a)

                training_rows.append({
                    "date":          date,
                    "home_team":     h_can,
                    "away_team":     a_can,
                    "home_score":    hs,
                    "away_score":    as_,
                    "tournament":    row["tournament"],
                    "neutral":       bool(row["neutral"]),
                    "target":        int(row["target"]),
                    "home_form":     h_form,
                    "away_form":     a_form,
                    "home_opp_mean": h_opp_mean,
                    "away_opp_mean": a_opp_mean,
                    "h2h_diff":      h2h_diff,
                    "opp_adj_form_diff": h_opp_adj_form - a_opp_adj_form,
                    "home_gd":       h_gd,
                    "away_gd":       a_gd,
                    "gd_diff":       h_gd - a_gd,

                    "home_win_rate":  h_w,
                    "away_win_rate":  a_w,

                    "home_draw_rate": h_d,
                    "away_draw_rate": a_d,

                    "home_loss_rate": h_l,
                    "away_loss_rate": a_l,
                })

            # Update state after capture (no leakage)
            self.form_tracker.update(h, a, hs, as_)
            self.h2h.update(h, a, hs, as_)
            self._opponent_history.setdefault(h, []).append(a)
            self._opponent_history.setdefault(a, []).append(h)

        self.training_df = pd.DataFrame(training_rows)
        log.info("Chronological processing complete. "
                 "Form and H2H state built from %d total matches.", len(self.matches))

    def current_form_scores(self) -> Dict[str, float]:
        """
        Return the current (post-all-history) form score for every
        WC2026 team using their canonical name.
        """
        raw_snap = self.form_tracker.snapshot()
        result: Dict[str, float] = {}
        for canonical, csv_name in TEAM_NAME_ALIASES.items():
            result[canonical] = raw_snap.get(csv_name, 0.33)
        return result

    def current_h2h(self, home: str, away: str) -> float:
        """
        H2H win-rate differential for canonical team names.
        home and away are canonical WC2026 names.
        """
        h_csv = TEAM_NAME_ALIASES.get(home, home)
        a_csv = TEAM_NAME_ALIASES.get(away, away)
        return self.h2h.win_rate_diff(h_csv, a_csv)
    
    def current_form_gd(self) -> Dict[str, float]:
        """
        Goal differential for canonical team names.
        """
        raw_snap = self.form_tracker.snapshot_gd()

        result = {}
        for canonical, csv_name in TEAM_NAME_ALIASES.items():
            result[canonical] = raw_snap.get(csv_name, 0.0)

        return result
    
    def current_form_details(self) -> Dict[str, tuple]:
        raw = self.form_tracker.snapshot_full_form()

        result = {}
        for canonical, csv_name in TEAM_NAME_ALIASES.items():
            result[canonical] = raw.get(csv_name, (0.33, 0.33, 0.33, 0.0))

        return result

    def current_opp_adjusted_form_scores(self) -> Dict[str, float]:
        """
        Return opponent-adjusted form scores for canonical WC2026 teams.

        Score = team's current form score minus the average current form score of
        its recent opponents.
        """
        raw_forms = self.form_tracker.snapshot()
        result: Dict[str, float] = {}

        for canonical, csv_name in TEAM_NAME_ALIASES.items():
            recent_opponents = self._opponent_history.get(csv_name, [])[-self.form_tracker.window :]
            own_form = raw_forms.get(csv_name, 0.33)
            if not recent_opponents:
                result[canonical] = 0.0
                continue
            opp_mean = sum(raw_forms.get(opponent, 0.33) for opponent in recent_opponents) / len(recent_opponents)
            result[canonical] = own_form + opp_mean - 1.0

        return result
