import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from core.config import RESULTS_CSV, TRAINING_START, TEAM_NAME_ALIASES
from data.ingestion import load_raw_results, build_match_records, FormTracker, H2HRegistry
from data.elo_tracker import EloTracker

# Output directory for plots
OUT_DIR = Path(r"C:\Users\SAMMYAG\.gemini\antigravity-ide\brain\b0b4286a-aeef-45c1-af1a-fb05ef3c13c1")

def _build_distributions():
    training_start = pd.Timestamp(TRAINING_START)
    wc_csv_names = set(TEAM_NAME_ALIASES.values())

    raw = load_raw_results()
    all_matches = build_match_records(raw)

    raw_clean = raw.dropna(subset=["home_score", "away_score"]).copy()
    raw_clean["home_score"] = raw_clean["home_score"].astype(int)
    raw_clean["away_score"] = raw_clean["away_score"].astype(int)
    raw_clean = raw_clean.sort_values("date").reset_index(drop=True)

    all_matches = all_matches.copy()
    all_matches["home_score"] = raw_clean["home_score"].values
    all_matches["away_score"] = raw_clean["away_score"].values

    form_tracker = FormTracker(window=10)
    
    # We only need elo_diff, form_diff, gd_diff for the modern era.
    # We must chronologically process all matches.
    pre_training_hist = all_matches[all_matches["date"] < training_start]
    replay_elo = EloTracker()
    replay_elo.batch_update(pre_training_hist)
    
    baseline_rows = []
    scen1_rows = []
    scen2_rows = []

    for _, row in all_matches.iterrows():
        h = str(row["home_team"])
        a = str(row["away_team"])
        hs = int(row["home_score"])
        as_ = int(row["away_score"])
        date = row["date"]
        tournament = str(row["tournament"])
        neutral = bool(row["neutral"])

        if date >= training_start:
            rh = replay_elo._get(h)
            ra = replay_elo._get(a)
            elo_diff = rh - ra

            _, _, _, h_gd = form_tracker.get_form(h)
            _, _, _, a_gd = form_tracker.get_form(a)
            h_form = form_tracker.get_form_score(h)
            a_form = form_tracker.get_form_score(a)

            record = {
                "elo_diff": elo_diff,
                "form_diff": h_form - a_form,
                "gd_diff": h_gd - a_gd
            }

            scen2_rows.append(record)

            if h in wc_csv_names or a in wc_csv_names:
                scen1_rows.append(record)
                
            if h in wc_csv_names and a in wc_csv_names:
                baseline_rows.append(record)

        form_tracker.update(h, a, hs, as_)
        replay_elo.update(h, a, int(row["target"]), tournament, neutral)

    return pd.DataFrame(baseline_rows), pd.DataFrame(scen1_rows), pd.DataFrame(scen2_rows)

def main():
    print("Building datasets...")
    df_base, df_s1, df_s2 = _build_distributions()
    
    print(f"Baseline rows: {len(df_base)}")
    print(f"Scenario 1 rows: {len(df_s1)}")
    print(f"Scenario 2 rows: {len(df_s2)}")
    
    features = ["elo_diff", "form_diff", "gd_diff"]
    
    print("\n=== SUMMARY STATISTICS ===")
    for feat in features:
        print(f"\n--- {feat} ---")
        stats = pd.DataFrame({
            "Baseline": df_base[feat].describe(),
            "Scenario 1": df_s1[feat].describe(),
            "Scenario 2": df_s2[feat].describe()
        })
        print(stats.to_string())

    print("\nGenerating histograms...")
    
    for feat in features:
        plt.figure(figsize=(10, 6))
        plt.hist(df_base[feat], bins=50, density=True, alpha=0.5, label=f'Baseline (n={len(df_base)})')
        plt.hist(df_s1[feat], bins=50, density=True, alpha=0.5, label=f'Scenario 1 (n={len(df_s1)})')
        plt.hist(df_s2[feat], bins=50, density=True, alpha=0.5, label=f'Scenario 2 (n={len(df_s2)})')
        plt.title(f'Distribution of {feat}')
        plt.legend()
        out_path = OUT_DIR / f"dist_{feat}.png"
        plt.savefig(out_path)
        plt.close()
        print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()
