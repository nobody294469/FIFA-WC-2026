"""
feature_engineering.py — Feature construction from real data sources.

Features used in the ML model and the probability matrix:

  Real-data features (all derived from results.csv or config metadata):
  ─────────────────────────────────────────────────────────────────────
  elo_diff           : Elo rating differential (home − away)
  form_diff          : Rolling 10-match form score differential
  h2h_diff           : Head-to-head win-rate differential (all-time real results)
  home_form          : Home team's rolling form score [0,1]
  away_form          : Away team's rolling form score [0,1]
  travel_fatigue_diff: Away − Home travel distance to WC centroid (normalised [0,1])
  clim_mismatch      : Climate zone distance (home team vs USA host climate)
  match_type_enc     : Encoded match importance (0=friendly … 1=WC)
  neutral_venue      : 1 if neutral, 0 if home/away

  Derived at training time from real match rows:
  ──────────────────────────────────────────────
  home_elo, away_elo : raw Elo ratings at match time

No synthetic features. No player micro-metrics that require a paid API.
"""
from __future__ import annotations

from math import asin, cos, radians, sin, sqrt
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from core.config import (
    CLIMATE_NUM,
    ELO_K_MAP,
    TEAM_CLIMATE,
    TEAM_COORDS,
    WC_VENUE_CENTROID,
    get_fifa_rank,
    get_squad_value,
)
from core.logger import get_logger

log = get_logger("feature_engineering")

# ── Feature column order (must be consistent across train and inference) ────────
FEATURE_COLS = [
    "elo_diff",
    "h2h_diff",

    "form_diff",
    "opp_adj_form_diff",
    "gd_diff",

    "home_form",
    "away_form",

    "home_win_rate",
    "away_win_rate",

    "home_draw_rate",
    "away_draw_rate",

    "home_loss_rate",
    "match_type_enc",
    "neutral_venue",
]
_TOURNEY_IMPORTANCE: Dict[str, float] = {
    "FIFA World Cup":                        1.00,
    "UEFA Euro":                             0.90,
    "Copa América":                          0.90,
    "Africa Cup of Nations":                 0.88,
    "African Cup of Nations":                0.88,
    "AFC Asian Cup":                         0.88,
    "CONCACAF Gold Cup":                     0.85,
    "OFC Nations Cup":                       0.80,
    "Confederations Cup":                    0.85,
    "UEFA Nations League":                   0.75,
    "CONCACAF Nations League":               0.72,
    "FIFA World Cup qualification":          0.70,
    "UEFA Euro qualification":               0.68,
    "African Cup of Nations qualification":  0.65,
    "AFC Asian Cup qualification":           0.65,
    "CONCACAF Championship":                 0.65,
    "Copa América qualification":            0.60,
    "Friendly":                              0.20,
}
_DEFAULT_IMPORTANCE = 0.40


def tournament_importance(tournament: str) -> float:
    return _TOURNEY_IMPORTANCE.get(tournament, _DEFAULT_IMPORTANCE)


# ── Travel distance ────────────────────────────────────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    φ1, φ2 = radians(lat1), radians(lat2)
    Δφ, Δλ = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(Δφ / 2) ** 2 + cos(φ1) * cos(φ2) * sin(Δλ / 2) ** 2
    return 2 * R * asin(sqrt(max(a, 0.0)))


_TRAVEL_KM_MAX = 18_000.0
_TRAVEL_KM_MIN = 0.0


# @lru_cache(maxsize=128)


# ═══════════════════════════════════════════════════════════════════════════════
# Training feature builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_training_features(
    training_df: pd.DataFrame,
    elo_tracker,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Construct (X, y) from the chronologically-processed training DataFrame
    that comes out of MatchDataPipeline._process_chronological().

    The training_df already contains:
      home_team (canonical), away_team (canonical), date, tournament, neutral,
      home_form, away_form, h2h_diff, target

    We add Elo-based features by replaying the match-level Elo state.

    Parameters
    ----------
    training_df : output of MatchDataPipeline (post-chronological-processing)
    elo_tracker : a fully-updated EloTracker (after batch_update on full history)

    NOTE: We cannot use the final Elo state for training features (data leakage).
    Instead we rebuild a second lightweight Elo replay using only the training-set
    matches, which gives us the Elo state *before* each match.
    """
    from data.elo_tracker import EloTracker  # local import to avoid circular

    # ── Replay Elo over training matches to get pre-match ratings ─────────────
    # We use the full raw results for the Elo baseline up to TRAINING_START,
    # then continue updating as we process each row.
    # The elo_tracker passed in already holds the full-history final state.
    # For training features, we need to know the Elo *at the time* of the match.
    # We achieve this by attaching the ELO snapshot at capture time — which
    # ingestion.py already handles in MatchDataPipeline._process_chronological.
    # However, that module stores form not elo. We add elo here by:
    #   (a) building a replay EloTracker from scratch for the training window
    #   (b) walking the training_df chronologically

    # Build a mini EloTracker from scratch for the training window
    # (re-read from full results up to training_df's earliest date)
    from core.config import TRAINING_START, RESULTS_CSV, TEAM_NAME_ALIASES, ALIAS_REVERSE
    import pandas as _pd

    raw = _pd.read_csv(RESULTS_CSV, parse_dates=["date"])
    raw = raw.dropna(subset=["home_score", "away_score"]).copy()
    raw["home_score"] = raw["home_score"].astype(int)
    raw["away_score"] = raw["away_score"].astype(int)
    raw["target"] = (_pd.np.where(
        raw["home_score"] > raw["away_score"], 0,
        _pd.np.where(raw["home_score"] == raw["away_score"], 1, 2)
    ) if hasattr(_pd, "np") else
    __import__("numpy").where(
        raw["home_score"] > raw["away_score"], 0,
        __import__("numpy").where(raw["home_score"] == raw["away_score"], 1, 2)
    ))
    raw = raw.sort_values("date").reset_index(drop=True)

    cutoff = _pd.Timestamp(TRAINING_START)
    pre_training = raw[raw["date"] < cutoff]
_TOURNEY_IMPORTANCE: Dict[str, float] = {
    "FIFA World Cup":                        1.00,
    "UEFA Euro":                             0.90,
    "Copa América":                          0.90,
    "Africa Cup of Nations":                 0.88,
    "African Cup of Nations":                0.88,
    "AFC Asian Cup":                         0.88,
    "CONCACAF Gold Cup":                     0.85,
    "OFC Nations Cup":                       0.80,
    "Confederations Cup":                    0.85,
    "UEFA Nations League":                   0.75,
    "CONCACAF Nations League":               0.72,
    "FIFA World Cup qualification":          0.70,
    "UEFA Euro qualification":               0.68,
    "African Cup of Nations qualification":  0.65,
    "AFC Asian Cup qualification":           0.65,
    "CONCACAF Championship":                 0.65,
    "Copa América qualification":            0.60,
    "Friendly":                              0.20,
}
_DEFAULT_IMPORTANCE = 0.40


def tournament_importance(tournament: str) -> float:
    return _TOURNEY_IMPORTANCE.get(tournament, _DEFAULT_IMPORTANCE)


# ── Travel distance ────────────────────────────────────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    φ1, φ2 = radians(lat1), radians(lat2)
    Δφ, Δλ = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(Δφ / 2) ** 2 + cos(φ1) * cos(φ2) * sin(Δλ / 2) ** 2
    return 2 * R * asin(sqrt(max(a, 0.0)))


_TRAVEL_KM_MAX = 18_000.0
_TRAVEL_KM_MIN = 0.0


# @lru_cache(maxsize=128)


# ═══════════════════════════════════════════════════════════════════════════════
# Training feature builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_training_features(
    training_df: pd.DataFrame,
    elo_tracker,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Construct (X, y) from the chronologically-processed training DataFrame
    that comes out of MatchDataPipeline._process_chronological().

    The training_df already contains:
      home_team (canonical), away_team (canonical), date, tournament, neutral,
      home_form, away_form, h2h_diff, target

    We add Elo-based features by replaying the match-level Elo state.

    Parameters
    ----------
    training_df : output of MatchDataPipeline (post-chronological-processing)
    elo_tracker : a fully-updated EloTracker (after batch_update on full history)

    NOTE: We cannot use the final Elo state for training features (data leakage).
    Instead we rebuild a second lightweight Elo replay using only the training-set
    matches, which gives us the Elo state *before* each match.
    """
    from data.elo_tracker import EloTracker  # local import to avoid circular

    # ── Replay Elo over training matches to get pre-match ratings ─────────────
    # We use the full raw results for the Elo baseline up to TRAINING_START,
    # then continue updating as we process each row.
    # The elo_tracker passed in already holds the full-history final state.
    # For training features, we need to know the Elo *at the time* of the match.
    # We achieve this by attaching the ELO snapshot at capture time — which
    # ingestion.py already handles in MatchDataPipeline._process_chronological.
    # However, that module stores form not elo. We add elo here by:
    #   (a) building a replay EloTracker from scratch for the training window
    #   (b) walking the training_df chronologically

    # Build a mini EloTracker from scratch for the training window
    # (re-read from full results up to training_df's earliest date)
    from core.config import TRAINING_START, RESULTS_CSV, TEAM_NAME_ALIASES, ALIAS_REVERSE
    import pandas as _pd

    raw = _pd.read_csv(RESULTS_CSV, parse_dates=["date"])
    raw = raw.dropna(subset=["home_score", "away_score"]).copy()
    raw["home_score"] = raw["home_score"].astype(int)
    raw["away_score"] = raw["away_score"].astype(int)
    raw["target"] = (_pd.np.where(
        raw["home_score"] > raw["away_score"], 0,
        _pd.np.where(raw["home_score"] == raw["away_score"], 1, 2)
    ) if hasattr(_pd, "np") else
    __import__("numpy").where(
        raw["home_score"] > raw["away_score"], 0,
        __import__("numpy").where(raw["home_score"] == raw["away_score"], 1, 2)
    ))
    raw = raw.sort_values("date").reset_index(drop=True)

    cutoff = _pd.Timestamp(TRAINING_START)
    pre_training = raw[raw["date"] < cutoff]

    replay_elo = EloTracker()
    # Seed with pre-training history
    replay_elo.batch_update(pre_training)

    # Now walk training rows and capture pre-match elo
    elo_diffs   = []
    home_elos   = []
    away_elos   = []
    home_ranks  = []
    away_ranks  = []

    tdf = training_df.sort_values("date").reset_index(drop=True)
    from core.config import TEAM_NAME_ALIASES
    for _, row in tdf.iterrows():
        h   = row["home_team"]   # canonical
        a   = row["away_team"]   # canonical
        h_csv = TEAM_NAME_ALIASES.get(h, h)
        a_csv = TEAM_NAME_ALIASES.get(a, a)

        rh = replay_elo._get(h_csv)
        ra = replay_elo._get(a_csv)
        
        # Calculate rank dynamically
        sorted_ratings = sorted(replay_elo.ratings.values(), reverse=True)
        hrk = sorted_ratings.index(rh) + 1 if rh in sorted_ratings else 200
        ark = sorted_ratings.index(ra) + 1 if ra in sorted_ratings else 200
        
        home_elos.append(rh)
        away_elos.append(ra)
        home_ranks.append(hrk)
        away_ranks.append(ark)
        elo_diffs.append(rh - ra)

        # Update replay elo for this match
        replay_elo.update(
            home_team  = h_csv,
            away_team  = a_csv,
            outcome    = int(row["target"]),
            tournament = str(row["tournament"]),
            neutral    = bool(row["neutral"]),
        )

    tdf = tdf.copy()
    tdf["elo_diff"]  = elo_diffs
    tdf["home_elo"]  = home_elos
    tdf["away_elo"]  = away_elos
    tdf["h_rank"]    = home_ranks
    tdf["a_rank"]    = away_ranks
    tdf["form_diff"] = (
    tdf["home_form"].astype(float) -
    tdf["away_form"].astype(float)
)

    # ── Compute static features ────────────────────────────────────────────────
    tdf["match_type_enc"] = tdf["tournament"].map(tournament_importance)
    tdf["neutral_venue"]  = tdf["neutral"].astype(float)

    # Rename to match FEATURE_COLS
    tdf = tdf.rename(columns={"h2h_diff": "h2h_diff"})   # already named correctly

    missing = [c for c in FEATURE_COLS if c not in tdf.columns]
    if missing:
        raise ValueError(f"Missing training feature columns: {missing}")

    X = tdf[FEATURE_COLS + ["h_rank", "a_rank"]].copy()
    y = tdf["target"].astype(int)

    log.info(
        "Training features built: %d rows × %d features. "
        "Target dist: {0: %d, 1: %d, 2: %d}",
        len(X), len(FEATURE_COLS),
        (y == 0).sum(), (y == 1).sum(), (y == 2).sum(),
    )
    return X, y


# ═══════════════════════════════════════════════════════════════════════════════
# Inference feature builder (for probability matrix pre-computation)
# ═══════════════════════════════════════════════════════════════════════════════

def build_inference_features(
    pairs: List[Tuple[str, str]],
    elo_tracker,
    form_scores: Dict[str, float],
    opp_adj_form_scores: Dict[str, float],
    form_gd: Dict[str, float],
    form_details: Dict[str, tuple],
    h2h_fn,
    neutral: bool = True,
    tournament: str = "FIFA World Cup",
) -> pd.DataFrame:
    """
    Build the inference feature matrix for a list of (home, away) canonical pairs.

    Parameters
    ----------
    pairs        : list of (home_team, away_team) canonical names
    elo_tracker  : fully-trained EloTracker
    form_scores  : {canonical_team: form_score} dict (from FormTracker.snapshot)
    h2h_fn       : callable taking (home, away) canonical names → float
    neutral      : all WC matches are at neutral venue
    tournament   : used for match_type_enc

    Returns a DataFrame with columns = FEATURE_COLS + ["home_team", "away_team"]
    """
    rows = []
    mt_enc = tournament_importance(tournament)

    for home, away in pairs:
        elo_h  = elo_tracker.get_rating(home)
        elo_a  = elo_tracker.get_rating(away)
        f_h    = form_scores.get(home, 0.33)
        f_a    = form_scores.get(away, 0.33)
        of_h   = opp_adj_form_scores.get(home, 0.0)
        of_a   = opp_adj_form_scores.get(away, 0.0)
        gd_h   = form_gd.get(home, 0.0)
        gd_a   = form_gd.get(away, 0.0)
        mv_h   = np.log1p(get_squad_value(home))
        mv_a   = np.log1p(get_squad_value(away))
        rk_h   = get_fifa_rank(home)
        rk_a   = get_fifa_rank(away)
        
        # Unpack detail metrics
        h_w, h_d, h_l, _ = form_details.get(home, (0.33, 0.33, 0.33, 0.0))
        a_w, a_d, a_l, _ = form_details.get(away, (0.33, 0.33, 0.33, 0.0))

        rows.append({
            "home_team":            home,
            "away_team":            away,
            "elo_diff":             elo_h - elo_a,
            "h2h_diff":             h2h_fn(home, away),
            "form_diff":            f_h - f_a,
            "opp_adj_form_diff":    of_h - of_a,
            "gd_diff":              gd_h - gd_a,
            "home_form":            f_h,
            "away_form":            f_a,
            "match_type_enc":       mt_enc,
            "neutral_venue":        1.0 if neutral else 0.0,
            "home_win_rate":        h_w,
            "away_win_rate":        a_w,
            "home_draw_rate":       h_d,
            "away_draw_rate":       a_d,
            "home_loss_rate":       h_l,
            "away_loss_rate":       a_l,
            "h_rank":               rk_h,
            "a_rank":               rk_a,
        })

    return pd.DataFrame(rows)
