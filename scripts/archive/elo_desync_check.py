# Diagnostic: compare pre-match replay Elo vs final/full-history Elo
from pathlib import Path
import json
import sys
import pandas as pd
import numpy as np

# Ensure repo root is on sys.path when running this script directly
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.ingestion import MatchDataPipeline
from data.elo_tracker import EloTracker
from core.config import TEAM_NAME_ALIASES, TRAINING_START

OUT_JSON = Path("models/elo_desync_report.json")
OUT_CSV = Path("models/elo_desync_examples.csv")

p = MatchDataPipeline()
training = p.training_df.copy()
# final/full-history Elo
final_elo = EloTracker()
final_elo.batch_update(p.matches)

# replay Elo seeded with pre-training history
raw = p.raw  # results.csv cleaned
cutoff = pd.Timestamp(TRAINING_START)
pre_training = raw[raw["date"] < cutoff]
replay = EloTracker()
# Ensure pre_training has a 'target' column like build_training_features expects
pre_training = pre_training.copy()
pre_training["target"] = (
    np.where(pre_training["home_score"] > pre_training["away_score"], 0,
             np.where(pre_training["home_score"] == pre_training["away_score"], 1, 2))
)
replay.batch_update(pre_training)

rows = []
# training rows are canonical names
for _, row in training.sort_values("date").iterrows():
    h_can = row["home_team"]
    a_can = row["away_team"]
    # csv names used by EloTracker internals
    h_csv = TEAM_NAME_ALIASES.get(h_can, h_can)
    a_csv = TEAM_NAME_ALIASES.get(a_can, a_can)

    pre_h = replay._get(h_csv)
    pre_a = replay._get(a_csv)
    pre_diff = pre_h - pre_a

    final_h = final_elo.get_rating(h_can)
    final_a = final_elo.get_rating(a_can)
    final_diff = final_h - final_a

    delta = pre_diff - final_diff
    sign_flip = np.sign(pre_diff) != np.sign(final_diff)

    rows.append({
        "date": str(row["date"].date()),
        "home": h_can,
        "away": a_can,
        "pre_home_elo": pre_h,
        "pre_away_elo": pre_a,
        "pre_diff": pre_diff,
        "final_home_elo": final_h,
        "final_away_elo": final_a,
        "final_diff": final_diff,
        "delta_diff": delta,
        "sign_flip": bool(sign_flip),
        "target": int(row["target"]),
    })

    # update replay with this match
    replay.update(home_team=h_csv, away_team=a_csv, outcome=int(row["target"]), tournament=str(row["tournament"]), neutral=bool(row["neutral"]))

# Create DataFrame
df = pd.DataFrame(rows)

deltas = df["delta_diff"].astype(float)
abs_deltas = deltas.abs()

summary = {
    "n_rows": int(len(df)),
    "mean_delta": float(deltas.mean()),
    "median_delta": float(deltas.median()),
    "std_delta": float(deltas.std()),
    "max_abs_delta": float(abs_deltas.max()),
    "pct_sign_flip": float(df["sign_flip"].mean()),
}

# Select some concrete examples
examples = pd.concat([
    df.sort_values("delta_diff", key=lambda s: s.abs(), ascending=False).head(5),
    df[df["sign_flip"]].head(5)
]).drop_duplicates().reset_index(drop=True)

# Save outputs
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps(summary, indent=2))
examples.to_csv(OUT_CSV, index=False)

print("Saved summary to", OUT_JSON)
print("Saved examples to", OUT_CSV)
print(json.dumps(summary, indent=2))
