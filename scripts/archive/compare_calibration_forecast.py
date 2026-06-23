import pandas as pd
from pathlib import Path

FORECAST_DIR = Path("reports/forecast_2026")
raw_df = pd.read_csv(FORECAST_DIR / "champion_probabilities_raw.csv")
cal_df = pd.read_csv(FORECAST_DIR / "champion_probabilities_calibrated.csv")

# Ensure order doesn't matter by merging on team
df = pd.merge(
    raw_df[["team", "champion_pct", "finalist_pct", "semifinal_pct", "model_rank"]],
    cal_df[["team", "champion_pct", "finalist_pct", "semifinal_pct", "model_rank"]],
    on="team",
    suffixes=("_raw", "_cal")
)

# Calculate Deltas
df["champ_delta"] = df["champion_pct_cal"] - df["champion_pct_raw"]

# Sort by calibrated rank
df = df.sort_values("model_rank_cal").reset_index(drop=True)

# 1. Comparison Tables (Top 20 based on Calibrated)
print("=== Top 20 Teams Comparison ===")
print(f"{'Team':<18} | {'Raw Champ%':>10} | {'Cal Champ%':>10} | {'Delta':>8} | {'Raw Final%':>10} | {'Cal Final%':>10} | {'Raw Semi%':>9} | {'Cal Semi%':>9}")
print("-" * 105)
for _, row in df.head(20).iterrows():
    print(f"{row['team']:<18} | {row['champion_pct_raw']:>9.2f}% | {row['champion_pct_cal']:>9.2f}% | {row['champ_delta']:>7.2f}% | {row['finalist_pct_raw']:>9.2f}% | {row['finalist_pct_cal']:>9.2f}% | {row['semifinal_pct_raw']:>8.2f}% | {row['semifinal_pct_cal']:>8.2f}%")

# 2. Probability Mass Concentrations
top3_raw = df.sort_values("champion_pct_raw", ascending=False).head(3)["champion_pct_raw"].sum()
top3_cal = df.sort_values("champion_pct_cal", ascending=False).head(3)["champion_pct_cal"].sum()

top5_raw = df.sort_values("champion_pct_raw", ascending=False).head(5)["champion_pct_raw"].sum()
top5_cal = df.sort_values("champion_pct_cal", ascending=False).head(5)["champion_pct_cal"].sum()

ranks_10_20_raw = df[(df["model_rank_raw"] >= 10) & (df["model_rank_raw"] <= 20)]["champion_pct_raw"].sum()
ranks_10_20_cal = df[(df["model_rank_cal"] >= 10) & (df["model_rank_cal"] <= 20)]["champion_pct_cal"].sum()

ranks_21_48_raw = df[df["model_rank_raw"] > 20]["champion_pct_raw"].sum()
ranks_21_48_cal = df[df["model_rank_cal"] > 20]["champion_pct_cal"].sum()

print("\n=== Probability Mass Concentration ===")
print(f"Top 3 Teams:       {top3_raw:5.2f}% (Raw) -> {top3_cal:5.2f}% (Calibrated)   [Delta: {top3_cal - top3_raw:+.2f}%]")
print(f"Top 5 Teams:       {top5_raw:5.2f}% (Raw) -> {top5_cal:5.2f}% (Calibrated)   [Delta: {top5_cal - top5_raw:+.2f}%]")
print(f"Ranks 10-20:       {ranks_10_20_raw:5.2f}% (Raw) -> {ranks_10_20_cal:5.2f}% (Calibrated)   [Delta: {ranks_10_20_cal - ranks_10_20_raw:+.2f}%]")
print(f"Ranks 21-48:       {ranks_21_48_raw:5.2f}% (Raw) -> {ranks_21_48_cal:5.2f}% (Calibrated)   [Delta: {ranks_21_48_cal - ranks_21_48_raw:+.2f}%]")

# 3. Risers and Fallers
risers = df.sort_values("champ_delta", ascending=False).head(5)
fallers = df.sort_values("champ_delta", ascending=True).head(5)

print("\n=== Biggest Risers (Calibrated - Raw) ===")
for _, row in risers.iterrows():
    print(f"{row['team']:<18}: {row['champ_delta']:+.2f}%")

print("\n=== Biggest Fallers (Calibrated - Raw) ===")
for _, row in fallers.iterrows():
    print(f"{row['team']:<18}: {row['champ_delta']:+.2f}%")
