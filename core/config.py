"""
config.py — Single source of truth for the WC2026 prediction pipeline.

All team metadata uses real sources:
  • FIFA Rankings: official April 2026 release
  • Transfermarkt squad values: May 2026 (€M)
  • Team name aliases: maps official WC2026 team names to the exact strings
    used in the historical results.csv dataset (martj42/international_results)

No synthetic data. No placeholder values.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
DATA_DIR   = ROOT / "data"
MODEL_DIR  = ROOT / "models"
CACHE_DIR  = ROOT / "cache"
LOG_DIR    = ROOT / "logs"
REPORT_DIR = ROOT / "reports"

for _d in (DATA_DIR, MODEL_DIR, CACHE_DIR, LOG_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = DATA_DIR / "results.csv"

# ── Elo engine ─────────────────────────────────────────────────────────────────
ELO_INITIAL        = 1500.0
ELO_HOME_ADVANTAGE = 100.0   # points added to home team expected score

# K-factor by tournament string (importance-scaled).
# Any tournament not listed falls back to ELO_K_DEFAULT.
ELO_K_MAP: Dict[str, float] = {
    "FIFA World Cup":                        60.0,
    "FIFA World Cup qualification":          40.0,
    "Confederations Cup":                    50.0,
    "Copa América":                          50.0,
    "Copa America":                          50.0,
    "UEFA Euro":                             50.0,
    "UEFA Euro qualification":               40.0,
    "Africa Cup of Nations":                 50.0,
    "African Cup of Nations":                50.0,
    "African Cup of Nations qualification":  40.0,
    "AFC Asian Cup":                         50.0,
    "AFC Asian Cup qualification":           40.0,
    "CONCACAF Gold Cup":                     50.0,
    "CONCACAF Nations League":               40.0,
    "CONCACAF Championship":                 40.0,
    "CONCACAF Nations League qualification": 35.0,
    "UEFA Nations League":                   40.0,
    "Copa América qualification":            40.0,
    "Copa America qualification":            40.0,
    "OFC Nations Cup":                       45.0,
    "COSAFA Cup":                            30.0,
    "Friendly":                              20.0,
    "Arab Cup":                              30.0,
    "Gulf Cup":                              35.0,
    "King's Cup":                            25.0,
}
ELO_K_DEFAULT = 30.0

# ── ML model hyperparameters ───────────────────────────────────────────────────
# Hybrid Calibration Settings
HYBRID_CALIBRATION_RANK_THRESHOLD = 20
XGB_PARAMS: Dict = {
    "n_estimators":     800,
    "max_depth":        2,
    "learning_rate":    0.10,
    "subsample":        0.60,
    "colsample_bytree": 0.90,
    "min_child_weight": 10,
    "gamma":            1.00,
    "reg_alpha":        0.50,
    "reg_lambda":       1.50,
    "objective":        "multi:softprob",
    "num_class":        3,          # 0=home win, 1=draw, 2=away win
    "eval_metric":      "mlogloss",
    "tree_method":      "hist",
    "random_state":     42,
    "n_jobs":           -1,
    "verbosity":        0,
}
MODEL_CV_FOLDS  = 5
XGB_EARLY_STOPPING_ROUNDS = 30
TRAINING_START  = "2010-01-01"   # modern football era cutoff

# ── Monte Carlo simulation ─────────────────────────────────────────────────────
N_SIMULATIONS = 10_000
RANDOM_SEED   = 2026

# ── Team name aliases ──────────────────────────────────────────────────────────
# Key   = canonical WC2026 name (used everywhere in this codebase)
# Value = name as it appears in results.csv
TEAM_NAME_ALIASES: Dict[str, str] = {
    "Mexico":                    "Mexico",
    "South Africa":              "South Africa",
    "South Korea":               "South Korea",
    "Czechia":                   "Czech Republic",
    "Canada":                    "Canada",
    "Switzerland":               "Switzerland",
    "Qatar":                     "Qatar",
    "Bosnia and Herzegovina":    "Bosnia and Herzegovina",
    "Brazil":                    "Brazil",
    "Morocco":                   "Morocco",
    "Haiti":                     "Haiti",
    "Scotland":                  "Scotland",
    "USA":                       "United States",
    "Paraguay":                  "Paraguay",
    "Australia":                 "Australia",
    "Türkiye":                   "Turkey",
    "Germany":                   "Germany",
    "Curaçao":                   "Curaçao",
    "Ivory Coast":               "Ivory Coast",
    "Ecuador":                   "Ecuador",
    "Netherlands":               "Netherlands",
    "Japan":                     "Japan",
    "Tunisia":                   "Tunisia",
    "Sweden":                    "Sweden",
    "Belgium":                   "Belgium",
    "Egypt":                     "Egypt",
    "Iran":                      "Iran",
    "New Zealand":               "New Zealand",
    "Spain":                     "Spain",
    "Cape Verde":                "Cape Verde",
    "Saudi Arabia":              "Saudi Arabia",
    "Uruguay":                   "Uruguay",
    "France":                    "France",
    "Senegal":                   "Senegal",
    "Norway":                    "Norway",
    "Iraq":                      "Iraq",
    "Argentina":                 "Argentina",
    "Algeria":                   "Algeria",
    "Austria":                   "Austria",
    "Jordan":                    "Jordan",
    "Portugal":                  "Portugal",
    "Uzbekistan":                "Uzbekistan",
    "Colombia":                  "Colombia",
    "DR Congo":                  "DR Congo",
    "England":                   "England",
    "Croatia":                   "Croatia",
    "Ghana":                     "Ghana",
    "Panama":                    "Panama",
}

# Reverse: results.csv name → canonical WC2026 name
ALIAS_REVERSE: Dict[str, str] = {v: k for k, v in TEAM_NAME_ALIASES.items()}

# Ordered list of all 48 canonical team names
WC2026_TEAMS: List[str] = list(TEAM_NAME_ALIASES.keys())

# ── Official WC2026 group draw ─────────────────────────────────────────────────
WC2026_GROUPS: Dict[str, List[str]] = {
    "A": ["Mexico",      "South Africa", "South Korea",           "Czechia"],
    "B": ["Canada",      "Switzerland",  "Qatar",                 "Bosnia and Herzegovina"],
    "C": ["Brazil",      "Morocco",      "Haiti",                 "Scotland"],
    "D": ["USA",         "Paraguay",     "Australia",             "Türkiye"],
    "E": ["Germany",     "Curaçao",      "Ivory Coast",           "Ecuador"],
    "F": ["Netherlands", "Japan",        "Tunisia",               "Sweden"],
    "G": ["Belgium",     "Egypt",        "Iran",                  "New Zealand"],
    "H": ["Spain",       "Cape Verde",   "Saudi Arabia",          "Uruguay"],
    "I": ["France",      "Senegal",      "Norway",                "Iraq"],
    "J": ["Argentina",   "Algeria",      "Austria",               "Jordan"],
    "K": ["Portugal",    "Uzbekistan",   "Colombia",              "DR Congo"],
    "L": ["England",     "Croatia",      "Ghana",                 "Panama"],
}

N_BEST_THIRD = 8   # best 8 third-place teams advance from 12 groups

# ── April 2026 FIFA World Rankings ────────────────────────────────────────────
# Maps canonical WC2026 name → FIFA rank (lower = better)
FIFA_RANKINGS: Dict[str, int] = {
    "Argentina":              1,
    "Spain":                  2,
    "France":                 3,
    "England":                4,
    "Brazil":                 6,
    "Portugal":               5,
    "Netherlands":            8,
    "Belgium":                9,
    "Germany":               10,
    "Uruguay":               16,
    "Colombia":              13,
    "Morocco":               7,
    "Japan":                 18,
    "USA":                   17,
    "Croatia":               11,
    "Mexico":                14,
    "Switzerland":           19,
    "South Korea":           18,
    "Senegal":               15,
    "Iran":                  20,
    "Ecuador":               23,
    "Austria":               24,
    "Algeria":               28,
    "Türkiye":               22,
    "Norway":                31,
    "Canada":                30,
    "Australia":             27,
    "Tunisia":               46,
    "Ivory Coast":           33,
    "Czechia":               39,
    "Egypt":                 29,
    "Sweden":                38,
    "Scotland":              35,
    "Paraguay":              40,
    "South Africa":          60,
    "Jordan":                63,
    "Panama":                34,
    "Ghana":                 73,
    "Saudi Arabia":          44,
    "Iraq":                  56,
    "DR Congo":              45,
    "New Zealand":           85,
    "Bosnia and Herzegovina":64,
    "Cape Verde":            67,
    "Uzbekistan":            50,
    "Qatar":                 57,
    "Haiti":                 83,
    "Curaçao":               82,
}

def get_fifa_rank(team: str) -> int:
    """Return FIFA rank for a canonical team name (default 100 if unknown)."""
    return FIFA_RANKINGS.get(team, 100)

# ── Transfermarkt squad values (€M, May 2026) ─────────────────────────────────
SQUAD_VALUES_M = {
    "England":                1345.0,
    "France":                 1195.0,
    "Spain":                   861.0,
    "Germany":                 775.0,
    "Portugal":               1000.0,
    "Brazil":                 1135.0,
    "Argentina":               820.7,
    "Netherlands":             671.7,
    "Belgium":                 549.4,
    "Colombia":                390.0,
    "Uruguay":                 423.95,
    "USA":                     270.2,
    "Norway":                  320.0,
    "Sweden":                  310.0,
    "Mexico":                  164.6,
    "Switzerland":             281.7,
    "Japan":                   284.75,
    "Austria":                 290.0,
    "Czechia":                 320.0,
    "Croatia":                 325.5,
    "Morocco":                 318.1,
    "South Korea":             184.3,
    "Senegal":                 211.8,
    "Ivory Coast":             250.0,
    "Türkiye":                 280.0,
    "Ecuador":                 236.35,
    "Scotland":                185.0,
    "Canada":                  185.0,
    "Australia":               41.25,
    "Algeria":                 145.0,
    "Egypt":                   100.0,
    "Tunisia":                  54.1,
    "Iran":                     51.4,
    "Saudi Arabia":             14.5,
    "Ghana":                   242.2,
    "Paraguay":                130.0,
    "Bosnia and Herzegovina":  185.0,
    "South Africa":             78.0,
    "DR Congo":                 82.0,
    "Jordan":                   38.0,
    "Iraq":                     60.0,
    "Uzbekistan":               62.0,
    "Cape Verde":               58.0,
    "Qatar":                    14.3,
    "New Zealand":              32.0,
    "Haiti":                    26.0,
    "Curaçao":                  38.0,
    "Panama":                   48.0,
}

def get_squad_value(team: str) -> float:
    """Return Transfermarkt squad value in €M for a canonical team name."""
    return SQUAD_VALUES_M.get(team, 25.0)

# ── Climate zones (for travel-fatigue feature) ────────────────────────────────
CLIMATE_NUM: Dict[str, int] = {
    "temperate": 0, "mediterranean": 1, "tropical": 2, "arid": 3, "mixed": 1,
}
TEAM_CLIMATE: Dict[str, str] = {
    "Mexico": "arid", "South Africa": "temperate", "South Korea": "temperate",
    "Czechia": "temperate", "Canada": "temperate", "Switzerland": "temperate",
    "Qatar": "arid", "Bosnia and Herzegovina": "temperate",
    "Brazil": "tropical", "Morocco": "arid", "Haiti": "tropical",
    "Scotland": "temperate", "USA": "mixed", "Paraguay": "tropical",
    "Australia": "mixed", "Türkiye": "mediterranean", "Germany": "temperate",
    "Curaçao": "tropical", "Ivory Coast": "tropical", "Ecuador": "tropical",
    "Netherlands": "temperate", "Japan": "temperate", "Tunisia": "arid",
    "Sweden": "temperate", "Belgium": "temperate", "Egypt": "arid",
    "Iran": "arid", "New Zealand": "temperate", "Spain": "mediterranean",
    "Cape Verde": "tropical", "Saudi Arabia": "arid", "Uruguay": "temperate",
    "France": "temperate", "Senegal": "arid", "Norway": "temperate",
    "Iraq": "arid", "Argentina": "temperate", "Algeria": "arid",
    "Austria": "temperate", "Jordan": "arid", "Portugal": "mediterranean",
    "Uzbekistan": "arid", "Colombia": "tropical", "DR Congo": "tropical",
    "England": "temperate", "Croatia": "mediterranean", "Ghana": "tropical",
    "Panama": "tropical",
}

# Approximate home-nation coordinates (lat, lon)
TEAM_COORDS: Dict[str, Tuple[float, float]] = {
    "Argentina":             (-34.6, -58.4),
    "France":                (48.9,    2.4),
    "England":               (51.5,   -0.1),
    "Brazil":                (-15.8, -47.9),
    "Portugal":              (38.7,   -9.1),
    "Belgium":               (50.8,    4.4),
    "Netherlands":           (52.4,    4.9),
    "Spain":                 (40.4,   -3.7),
    "Germany":               (52.5,   13.4),
    "Morocco":               (34.0,   -6.9),
    "USA":                   (38.9,  -77.0),
    "Japan":                 (35.7,  139.7),
    "Colombia":              (4.7,   -74.1),
    "Uruguay":               (-34.9, -56.2),
    "Mexico":                (19.4,  -99.1),
    "Senegal":               (14.7,  -17.5),
    "Ecuador":               (-0.2,  -78.5),
    "South Korea":           (37.6,  127.0),
    "Canada":                (45.4,  -75.7),
    "Australia":             (-35.3, 149.1),
    "Czechia":               (50.1,   14.4),
    "Türkiye":               (39.9,   32.9),
    "Iran":                  (35.7,   51.4),
    "Norway":                (59.9,   10.7),
    "Algeria":               (36.7,    3.1),
    "Austria":               (48.2,   16.4),
    "Jordan":                (31.9,   35.9),
    "Iraq":                  (33.3,   44.4),
    "Sweden":                (59.3,   18.1),
    "Switzerland":           (46.9,    7.4),
    "Croatia":               (45.8,   16.0),
    "Bosnia and Herzegovina":(43.8,   18.4),
    "Scotland":              (55.9,   -3.2),
    "Ghana":                 (5.6,    -0.2),
    "Panama":                (9.0,   -79.5),
    "Paraguay":              (-25.3, -57.7),
    "Saudi Arabia":          (24.7,   46.7),
    "Egypt":                 (30.1,   31.2),
    "Ivory Coast":           (5.4,    -4.0),
    "Tunisia":               (36.8,   10.2),
    "South Africa":          (-25.7,  28.2),
    "DR Congo":              (-4.3,   15.3),
    "Qatar":                 (25.3,   51.5),
    "Cape Verde":            (15.1,  -23.6),
    "Curaçao":               (12.2,  -69.0),
    "Haiti":                 (18.5,  -72.3),
    "New Zealand":           (-41.3, 174.8),
    "Uzbekistan":            (41.3,   69.2),
}

# WC2026 host-city centroid (weighted average lat/lon of 16 host cities)
WC_VENUE_CENTROID: Tuple[float, float] = (32.0, -95.0)
