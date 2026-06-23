import xgboost as xgb
import numpy as np
import pandas as pd
from data.ingestion import MatchDataPipeline
from data.elo_tracker import EloTracker
from models.ml_engine import MatchOutcomeModel
from data.feature_engineering import build_inference_features

print("Initializing Pipeline...")
pipeline = MatchDataPipeline()
elo = EloTracker()
elo.batch_update(pipeline.matches)
model = MatchOutcomeModel()
model.load()

teams = ['Spain', 'France', 'Brazil', 'England', 'Germany', 'Portugal', 'Argentina']

print("\n=== RAW TEAM STATE VALUES ===")
form_scores = pipeline.current_form_scores()
opp_adj_form_scores = pipeline.current_opp_adjusted_form_scores()
form_gd = pipeline.current_form_gd()
form_details = pipeline.current_form_details()

print(f"{'Team':<12} | {'Elo':<6} | {'Form':<5} | {'Opp-Adj':<7} | {'Form GD':<7} | {'Win% (L5)':<9} | {'Away Win%':<9}")
print("-" * 70)
for t in teams:
    e = elo.get_rating(t)
    f = form_scores.get(t, 0)
    o = opp_adj_form_scores.get(t, 0)
    g = form_gd.get(t, 0)
    w_pct, u_pct, a_win, a_und = form_details.get(t, (0,0,0,0))
    print(f"{t:<12} | {e:6.1f} | {f:5.2f} | {o:7.2f} | {g:7.2f} | {w_pct:8.1f}% | {a_win:8.1f}%")

pairs = [
    ('Spain', 'England'),
    ('Spain', 'Brazil'),
    ('Spain', 'Germany')
]

feat_df = build_inference_features(
    pairs=pairs,
    elo_tracker=elo,
    form_scores=form_scores,
    opp_adj_form_scores=opp_adj_form_scores,
    form_gd=form_gd,
    form_details=form_details,
    h2h_fn=pipeline.current_h2h,
    neutral=True,
    tournament="FIFA World Cup"
)

from data.feature_engineering import build_inference_features, FEATURE_COLS

features = FEATURE_COLS
feat_df = feat_df[FEATURE_COLS]

# Get SHAP contributions
dmatrix = xgb.DMatrix(feat_df)
# shape: (n_samples, n_classes, n_features + 1)
# Class 0 is Home Win (Spain Win)
contribs = model.model.get_booster().predict(dmatrix, pred_contribs=True)

print("\n=== MATCHUP FEATURE VECTORS & SHAP CONTRIBUTIONS ===")
print("Note: SHAP values represent the log-odds additive contribution to the 'Spain Win' class.")

for i, (home, away) in enumerate(pairs):
    print(f"\n{home} vs {away}")
    print("-" * 50)
    
    # Extract feature values for this pair
    row_vals = feat_df.iloc[i].to_dict()
    
    # Extract SHAP values for class 0 (Home Win)
    shap_vals = contribs[i, 0, :-1] # Exclude bias
    bias = contribs[i, 0, -1]
    
    # Combine feature, value, shap
    data = []
    for j, f in enumerate(features):
        data.append((f, row_vals[f], shap_vals[j]))
        
    # Sort by absolute SHAP value to see most impactful features
    data.sort(key=lambda x: abs(x[2]), reverse=True)
    
    print(f"{'Feature':<25} | {'Value':<10} | {'SHAP (Log-Odds Contribution)':<20}")
    print("-" * 65)
    for f, v, s in data:
        print(f"{f:<25} | {v:<10.3f} | {s:+.4f}")
    
    print(f"{'Base Margin (Bias)':<25} | {'-':<10} | {bias:+.4f}")
    total_log_odds = bias + np.sum(shap_vals)
    prob = 1 / (1 + np.exp(-total_log_odds)) # roughly, though it's softmax in reality
    # Softmax conversion requires other classes, but we know the actual P(Win) from earlier
    print(f"Sum of SHAP + Bias = {total_log_odds:+.4f}")
