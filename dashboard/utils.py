"""
utils.py — Shared helpers for the WC2026 Streamlit dashboard.
"""
from __future__ import annotations
import base64
import sys
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FLAGS_DIR = Path(__file__).parent / "flags"

# ISO2 code for every WC2026 team
TEAM_ISO2: dict[str, str] = {
    "Mexico": "mx",
    "South Africa": "za",
    "South Korea": "kr",
    "Czechia": "cz",
    "Canada": "ca",
    "Switzerland": "ch",
    "Qatar": "qa",
    "Bosnia and Herzegovina": "ba",
    "Brazil": "br",
    "Morocco": "ma",
    "Haiti": "ht",
    "Scotland": "gb-sct",
    "USA": "us",
    "Paraguay": "py",
    "Australia": "au",
    "Türkiye": "tr",
    "Germany": "de",
    "Curaçao": "cw",
    "Ivory Coast": "ci",
    "Ecuador": "ec",
    "Netherlands": "nl",
    "Japan": "jp",
    "Tunisia": "tn",
    "Sweden": "se",
    "Belgium": "be",
    "Egypt": "eg",
    "Iran": "ir",
    "New Zealand": "nz",
    "Spain": "es",
    "Cape Verde": "cv",
    "Saudi Arabia": "sa",
    "Uruguay": "uy",
    "France": "fr",
    "Senegal": "sn",
    "Norway": "no",
    "Iraq": "iq",
    "Argentina": "ar",
    "Algeria": "dz",
    "Austria": "at",
    "Jordan": "jo",
    "Portugal": "pt",
    "Uzbekistan": "uz",
    "Colombia": "co",
    "DR Congo": "cd",
    "England": "gb-eng",
    "Croatia": "hr",
    "Ghana": "gh",
    "Panama": "pa",
}


@st.cache_data
def _flag_b64(iso2: str) -> str | None:
    path = FLAGS_DIR / f"{iso2}.png"
    if path.exists():
        return base64.b64encode(path.read_bytes()).decode()
    return None


def flag_img_html(team: str, width: int = 28) -> str:
    """Return an <img> HTML tag for the team's flag, or team name if flag missing."""
    iso2 = TEAM_ISO2.get(team)
    if iso2:
        b64 = _flag_b64(iso2)
        if b64:
            return (
                f'<img src="data:image/png;base64,{b64}" '
                f'width="{width}" height="{int(width*0.67)}" '
                f'style="vertical-align:middle; border-radius:2px; margin-right:6px;">'
            )
    return ""


def flag_label(team: str, width: int = 20) -> str:
    """Inline HTML: flag image + team name."""
    return f'{flag_img_html(team, width)}<span style="vertical-align:middle">{team}</span>'


@st.cache_resource(show_spinner="Loading ML model and probability matrices…")
def get_pmb():
    from data.ingestion import MatchDataPipeline
    from data.elo_tracker import EloTracker
    from models.ml_engine import MatchOutcomeModel, ProbabilityMatrixBuilder
    from core.config import WC2026_TEAMS

    pipeline = MatchDataPipeline()
    elo = EloTracker()
    elo.batch_update(pipeline.matches)
    model = MatchOutcomeModel()
    model.load()
    pmb = ProbabilityMatrixBuilder(model)
    pmb.build(
        teams=WC2026_TEAMS,
        elo_tracker=elo,
        form_scores=pipeline.current_form_scores(),
        opp_adj_form_scores=pipeline.current_opp_adjusted_form_scores(),
        form_gd=pipeline.current_form_gd(),
        h2h_fn=pipeline.current_h2h,
        form_details=pipeline.current_form_details(),
        ko_resolution="coinflip",
    )
    return pmb


@st.cache_data
def load_forecast(method: str = "coinflip"):
    import pandas as pd
    path = ROOT / f"reports/ko_sensitivity_{method}.csv"
    if path.exists():
        return pd.read_csv(path).sort_values("champion_pct", ascending=False).reset_index(drop=True)
    return pd.DataFrame()


def init_session_state():
    """Ensure all simulator keys exist in session_state."""
    defaults = {
        "sim_mode": None,          # "quick" | "custom"
        "sim_step": 1,             # 1=groups, 2=thirds, 3=bracket
        "groups_ordered": None,    # {group: [team, team, team, team]} after drag
        "groups_locked": False,
        "best_thirds_selected": [],
        "r32_bracket": None,       # list[str] 32 teams interleaved
        "bracket_round": "r32",    # current round: r32|r16|qf|sf|final
        "bracket_state": {         # winners per round, None = not yet decided
            "r32": [None] * 16,
            "r16": [None] * 8,
            "qf":  [None] * 4,
            "sf":  [None] * 2,
            "final": None,
        },
        "forecast_method": "coinflip",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_simulator():
    keys = [
        "sim_mode", "sim_step", "groups_ordered", "groups_locked",
        "best_thirds_selected", "r32_bracket", "bracket_round", "bracket_state",
    ]
    for k in keys:
        if k in st.session_state:
            del st.session_state[k]
    init_session_state()
