"""
elo_tracker.py — Rolling Elo rating system built from real historical results.

Algorithm
---------
Standard FIDE-style Elo with:
  • Variable K-factor based on match importance (tournament type)
  • Home advantage offset (+100 pts to home expected score)
  • Chronological batch update from the full results.csv
  • Persistence to JSON cache for fast restarts

All ratings are derived exclusively from real match results. No synthetic
initialisation or random seeding is used beyond the standard 1500-point default
for teams with no prior history in the dataset.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from core.config import (
    CACHE_DIR,
    ELO_HOME_ADVANTAGE,
    ELO_INITIAL,
    ELO_K_DEFAULT,
    ELO_K_MAP,
    TEAM_NAME_ALIASES,
    WC2026_TEAMS,
)
from core.logger import get_logger

log = get_logger("elo_tracker")

_ELO_CACHE_PATH = CACHE_DIR / "elo_ratings.json"

# Outcome → (score_home, score_away)
_SCORE_MAP: Dict[int, Tuple[float, float]] = {
    0: (1.0, 0.0),  # home win
    1: (0.5, 0.5),  # draw
    2: (0.0, 1.0),  # away win
}


class EloTracker:
    """
    Chronological Elo tracker.

    Initialisation seeds every team at ELO_INITIAL. As matches are processed,
    ratings diverge based solely on results. The final state after running the
    full history represents the best available estimate of each team's true
    strength heading into WC2026.
    """

    def __init__(self) -> None:
        # {csv_team_name: float}  — keyed by results.csv names for universal coverage
        self.ratings: Dict[str, float] = {}

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _get(self, team: str) -> float:
        if team not in self.ratings:
            self.ratings[team] = ELO_INITIAL
        return self.ratings[team]

    @staticmethod
    def _k_factor(tournament: str) -> float:
        return ELO_K_MAP.get(tournament, ELO_K_DEFAULT)

    def _expected(self, rating_a: float, rating_b: float, home_team_is_a: bool) -> float:
        """Expected score for team A (logistic function with optional home boost)."""
        adj_a = rating_a + (ELO_HOME_ADVANTAGE if home_team_is_a else 0.0)
        return 1.0 / (1.0 + 10.0 ** ((rating_b - adj_a) / 400.0))

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(
        self,
        home_team: str,
        away_team: str,
        outcome: int,           # 0=home win, 1=draw, 2=away win
        tournament: str,
        neutral: bool = False,
    ) -> None:
        """Update both teams' ratings after a single match."""
        r_h = self._get(home_team)
        r_a = self._get(away_team)
        k   = self._k_factor(tournament)

        e_h = self._expected(r_h, r_a, home_team_is_a=not neutral)
        s_h, s_a = _SCORE_MAP[outcome]

        delta_h = k * (s_h - e_h)
        delta_a = k * (s_a - (1.0 - e_h))

        self.ratings[home_team] = r_h + delta_h
        self.ratings[away_team] = r_a + delta_a

    def batch_update(self, matches: pd.DataFrame) -> None:
        """
        Process a chronologically-sorted DataFrame of all historical matches.

        Required columns: home_team, away_team, tournament, neutral, target
        The matches DataFrame uses results.csv team names (not canonical names).
        """
        log.info("Building Elo ratings from %d historical matches...", len(matches))
        for _, row in matches.iterrows():
            self.update(
                home_team  = str(row["home_team"]),
                away_team  = str(row["away_team"]),
                outcome    = int(row["target"]),
                tournament = str(row["tournament"]),
                neutral    = bool(row["neutral"]),
            )
        log.info("Elo batch update complete. %d unique teams tracked.", len(self.ratings))

    def get_rating(self, team: str) -> float:
        """
        Get Elo rating for a CANONICAL WC2026 team name.
        Handles the alias mapping automatically.
        """
        csv_name = TEAM_NAME_ALIASES.get(team, team)
        return self._get(csv_name)

    def elo_diff(self, home: str, away: str) -> float:
        """Signed Elo differential: home − away (canonical names)."""
        return self.get_rating(home) - self.get_rating(away)

    def expected_probs(
        self, team_a: str, team_b: str, neutral: bool = True
    ) -> Tuple[float, float, float]:
        """
        Convert Elo ratings to a (p_win_A, p_draw, p_win_B) triple.

        Uses a Gaussian draw-band model: draw probability peaks when teams are
        evenly matched (expected score ≈ 0.5) and falls off on either side.
        """
        r_a = self.get_rating(team_a)
        r_b = self.get_rating(team_b)
        e_a = self._expected(r_a, r_b, home_team_is_a=not neutral)

        draw = 0.27 * np.exp(-0.5 * ((e_a - 0.50) / 0.21) ** 2)
        p_a  = float(np.clip(e_a   - draw / 2, 0.05, 0.90))
        p_b  = float(np.clip(1 - e_a - draw / 2, 0.05, 0.90))
        draw_  = float(np.clip(draw, 0.05, 0.50))
        total  = p_a + draw_ + p_b
        return p_a / total, draw_ / total, p_b / total

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: Path = _ELO_CACHE_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.ratings, indent=2))
        log.info("Elo ratings cached → %s (%d teams)", path.name, len(self.ratings))

    @classmethod
    def load(cls, path: Path = _ELO_CACHE_PATH) -> Optional["EloTracker"]:
        """Load from cache. Returns None if cache does not exist."""
        if not path.exists():
            return None
        obj = cls()
        obj.ratings = json.loads(path.read_text())
        log.info("Loaded Elo ratings from cache (%d teams)", len(obj.ratings))
        return obj

    def ratings_df(self) -> pd.DataFrame:
        """
        Returns a DataFrame of WC2026 team Elo ratings sorted descending.
        Uses canonical team names.
        """
        rows = []
        for canonical in WC2026_TEAMS:
            rows.append({
                "team": canonical,
                "elo":  round(self.get_rating(canonical), 1),
            })
        df = (
            pd.DataFrame(rows)
            .sort_values("elo", ascending=False)
            .reset_index(drop=True)
        )
        df.insert(0, "elo_rank", range(1, len(df) + 1))
        return df