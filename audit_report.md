# WC2026 Prediction Pipeline — Deep Technical Audit

> **Scope**: Full code review of all Python modules.
> **Date**: June 2026 | **Reviewer**: Antigravity

---

## Priority Matrix

| # | Finding | Impact | Confidence | Effort |
|---|---------|--------|------------|--------|
| 1 | Backtest uses production model trained on future data | **Critical** | High | Medium |
| 2 | Dead features computed at inference but not in FEATURE_COLS | **High** | High | Low |
| 3 | `h2h_diff` still computed at training time and stored in training rows | **High** | High | Low |
| 4 | Early stopping selects on last-fold only, biasing n_estimators | **High** | High | Low |
| 5 | Fold 5 performance cliff — expanding window folds are unequal in size | **Medium** | High | Low |
| 6 | Neutral-venue calibration is not symmetric-safe for group stage | **Medium** | High | Medium |
| 7 | Synthetic goals tiebreaker is independent of match probabilities | **Medium** | Medium | Medium |
| 8 | `away_loss_rate` is computed at inference but absent from FEATURE_COLS | **Medium** | High | Low |
| 9 | H2H registry uses raw CSV names at training time, canonical at inference | **Medium** | High | Low |
| 10 | Best-8 third-place assignment lookup may silently fall back | **Medium** | Medium | Medium |
| 11 | `_rank_group` uses `ids[order]` — numpy advanced-indexing footgun | **Medium** | High | Low |
| 12 | `match_type_enc` leaks tournament importance into group-stage tiebreaker | Low | Medium | Low |
| 13 | Log-loss mean reported without std across folds | Low | High | Trivial |
| 14 | Backtest `group_acc` uses `round_of_16_pct` column, not `group_acc` | Low | High | Low |
| 15 | `_pd.np` deprecated usage in feature_engineering.py | Low | High | Trivial |

---

## Finding 1 — Backtest Uses the Production Model Trained on Future Data

**Severity: Critical | Confidence: High | Effort to fix: Medium**

### What the code does
`run_world_cup_backtest()` ([world_cup_backtest.py:402](file:///c:/Users/SAMMYAG/OneDrive/Desktop/wc2026_v3/backtesting/world_cup_backtest.py#L402)) calls `model.load()`, which loads the **production XGBoost model** (`xgb_wc2026.ubj`). That model was trained on matches from 2010 through the present day (June 2026). It is then used to predict the *2014, 2018, and 2022* World Cups.

```python
# world_cup_backtest.py:402-403
model = MatchOutcomeModel()
model.load() if model_path is None else model.load()   # always loads production model
```

`_build_pre_tournament_state` correctly builds a time-locked Elo/form/H2H state at the tournament cutoff date. But the **model weights themselves** were learned from match outcomes that postdate those cutoffs. The model has effectively memorised which teams performed well in 2014–2022, then "backtests" against exactly that knowledge.

### Why this inflates the reported metrics
- The tree splits on `elo_diff`, `form_diff`, and `opp_adj_form_diff` were chosen to minimise loss over a training set that *includes* the 2014–2022 match outcomes.
- Even if individual feature values are computed from pre-cutoff data, the threshold values inside the trees encode information from post-cutoff results.
- The 14.5% mean champion probability figure and 66.7% group-stage accuracy are therefore optimistic. They cannot be taken as evidence that the model generalises to unseen tournaments.

### Fix
Retrain the model at each cutoff date, or at minimum use a genuine out-of-sample fold (all matches before 2014 → predict 2014, all before 2018 → predict 2018, etc.). This is the same problem as the already-fixed `rank_diff` leakage but one level higher: it applies to the model itself.

---

## Finding 2 — Dead Features Computed at Inference but Not Used by the Model

**Severity: High | Confidence: High | Effort to fix: Low**

### What the code does
`build_inference_features()` ([feature_engineering.py:280-300](file:///c:/Users/SAMMYAG/OneDrive/Desktop/wc2026_v3/data/feature_engineering.py#L280-L300)) computes these values for every inference call:

```python
mv_h   = np.log1p(get_squad_value(home))   # market value
mv_a   = np.log1p(get_squad_value(away))
rk_h   = get_fifa_rank(home)               # FIFA ranking
rk_a   = get_fifa_rank(away)
```

The resulting row dict stores `mv_diff` and `rank_diff`, and also `h2h_diff` and `away_loss_rate`:

```python
rows.append({
    ...
    "mv_diff":   mv_h - mv_a,
    "rank_diff": float(rk_h - rk_a),
    "h2h_diff":  h2h,
    "away_loss_rate": a_l,   # at line 316
    ...
})
```

None of these four columns are in `FEATURE_COLS` ([feature_engineering.py:47-66](file:///c:/Users/SAMMYAG/OneDrive/Desktop/wc2026_v3/data/feature_engineering.py#L47-L66)). `predict_proba()` calls `X[FEATURE_COLS]`, so they are silently dropped before the model sees them.

### Impact
- `mv_diff` and `rank_diff` are computed (wasting CPU), stored in the output DataFrame, and then thrown away. If they were ever intended to be model inputs, the model has never seen them.
- `h2h_diff` is especially concerning: it is also stored in training rows (`ingestion.py:338`) but equally absent from `FEATURE_COLS`. That means the model has *never* used H2H differentials despite the significant compute spent building the H2H registry.
- `away_loss_rate` is populated in the row dict but not in `FEATURE_COLS`; it is therefore ignored during inference.

### Fix
Decision: either (a) add these columns to `FEATURE_COLS` and retrain, or (b) remove the dead code from `build_inference_features`. Given that H2H was already explicitly removed as "harmful," option (b) is safer. The compute waste is trivial, but the false confidence that these features are contributing is a real risk.

---

## Finding 3 — `h2h_diff` Is Still Stored in Training Rows

**Severity: High | Confidence: High | Effort to fix: Low**

### What the code does
`ingestion.py:325-338` computes and stores `h2h_diff` in every training row:

```python
h2h_diff = self.h2h.win_rate_diff(h, a)
training_rows.append({
    ...
    "h2h_diff": h2h_diff,
    ...
})
```

The `build_training_features()` function in `feature_engineering.py` receives this column in `tdf` but never adds it to `X = tdf[FEATURE_COLS]`. The column sits in memory, doing nothing, but is passed to `build_inference_features` conceptually. The key risk is developer confusion: the next person to "fix" this will likely add `h2h_diff` to `FEATURE_COLS` without auditing whether the current column has a different character at training vs inference time.

At training time, `h2h_diff` is the H2H differential using **raw CSV names** (e.g. `"United States"`, `"Czech Republic"`). At inference time, `h2h_fn` is called with **canonical names** (`"USA"`, `"Czechia"`). This name mismatch would cause silent zero-lookups at inference for any team with a name alias, producing a false neutral prior of 0.5.

### Fix
Remove `h2h_diff` computation from `_process_chronological` entirely, or ensure both paths use the same name space before re-adding it to `FEATURE_COLS`.

---

## Finding 4 — Early Stopping Is Determined by the Last CV Fold Only

**Severity: High | Confidence: High | Effort to fix: Low**

### What the code does
`ml_engine.py:135-150` runs a *separate* early stopping experiment using only the last chronological fold:

```python
tr_idx, val_idx = cv_splits[-1]   # last fold only
...
stop_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], ...)
self.production_best_iteration = int(getattr(stop_model, "best_iteration", ...)) + 1
```

This single fold determines `n_estimators` for the production model trained on **all data**.

### Why this is wrong
1. The last fold is the most recent slice of data — the smallest temporal window and likely the hardest distribution to model well. The early stopping count it produces is biased toward this specific slice.
2. The CV loop already collects `best_iteration` per fold (stored in `fold_metrics`). The mean of those (31.6 rounds, from `cv_metrics.json`) is a much better estimate than the single-fold value (65 rounds for fold 5).
3. The discrepancy between folds 1–4 (~16–31 rounds) and fold 5 (65 rounds) is significant: the last fold likely has a different optimal depth due to a structural shift in the validation data.

### Fix
Use `round(np.mean(best_iterations))` from the CV loop as `production_best_iteration`. This already uses the correct data and is already computed; the current code simply ignores it.

---

## Finding 5 — Expanding Window CV Produces Radically Unequal Fold Sizes

**Severity: Medium | Confidence: High | Effort to fix: Low (awareness)**

### What the code does
`TimeSeriesSplit(n_splits=5)` with the default `test_size=None` uses test blocks of equal size but training blocks that grow from ~16% to ~83% of the data. The CV metrics therefore measure performance on identical test-size windows, which is correct for forward-looking estimation.

However, fold 5 shows a dramatic performance cliff:

```
fold 1: acc=0.5125, logloss=1.0071
fold 2: acc=0.5219, logloss=1.0048
fold 3: acc=0.5156, logloss=0.9948
fold 4: acc=0.5250, logloss=0.9927
fold 5: acc=0.4562, logloss=1.0519   ← 7pp accuracy drop
```

### Why this matters
Fold 5 is the most recent data — the slice that best represents the WC2026 prediction problem. Its accuracy (45.6%) is **below random chance for a three-class problem at uniform priors** (33%), though not below a realistic home-win-biased prior. But it is 7 percentage points worse than fold 4, suggesting either:
- A distributional shift in recent matches (post-2022 football is different)
- The training window for fold 5 is simply too large and overfitting is occurring at some features
- Fold 5 validation data happened to contain a higher proportion of upsets

This cliff is currently hidden by the averaging in `cv_metrics`. The production model is trained on all data including this hard recent window, but `n_estimators` is derived from fold 5's early stopping (65 rounds), which may be compensating for this distributional shift by being more conservative.

### Fix
Log and inspect fold 5 validation dates explicitly. Consider adding a diagnostic that reports the temporal span of each fold's validation set alongside its metrics.

---

## Finding 6 — Neutral-Venue Calibration Assumption Is Fragile

**Severity: Medium | Confidence: High | Effort to fix: Medium**

### What the code does
`ProbabilityMatrixBuilder.build()` ([ml_engine.py:366-382](file:///c:/Users/SAMMYAG/OneDrive/Desktop/wc2026_v3/models/ml_engine.py#L366-L382)) applies symmetric averaging to remove home-field bias for neutral venues:

```python
p_win_ij = (raw_win[i, j] + raw_loss[j, i]) / 2.0
p_win_ji = (raw_win[j, i] + raw_loss[i, j]) / 2.0
p_draw   = (raw_draw[i, j] + raw_draw[j, i]) / 2.0
total    = p_win_ij + p_win_ji + p_draw
```

### Issues

**Issue 6a — All matchups are inferred with `neutral=True` from the start**
`build_inference_features()` is called with `neutral=True` in both the matrix builder and the showcases. The `neutral_venue` feature is already set to `1.0` in the feature vector passed to the model. The symmetric averaging then doubles down on the neutralisation. The model already adjusts for neutral venue through the `neutral_venue` feature; the post-hoc averaging re-applies an assumption that may not be validated.

**Issue 6b — The averages do not guarantee P_win + P_loss + P_draw = 1 before normalisation**
The code normalises by `total`, which corrects this. But if `total` ever drifts significantly from 1.0 (which can happen when the model's raw forward and backward predictions are inconsistent), the normalisation amplifies any model miscalibration rather than correcting it.

**Issue 6c — The group-stage matrix (`P_win_gs`, `P_draw_gs`) and knockout matrix (`P_ko`) are treated as independent**
`P_ko[i,j]` is derived from `P_win[i,j]` and `P_loss[i,j]` after symmetrisation. This is correct. However, for the group stage, `P_win_gs` and `P_draw_gs` are used separately, but `P_loss_gs` is never stored; it is implicitly `1 - P_win_gs - P_draw_gs`. If the symmetrisation's normalisation is not perfect, the group-stage draw probabilities may be inconsistent.

---

## Finding 7 — Goal Tiebreaker Is Decoupled From Match Outcome

**Severity: Medium | Confidence: Medium | Effort to fix: Medium**

### What the code does
The group-stage simulation ([monte_carlo.py:225-228](file:///c:/Users/SAMMYAG/OneDrive/Desktop/wc2026_v3/simulation/monte_carlo.py#L225-L228)) generates goals independently of match outcome:

```python
gr = rng.integers(0, 5, n, dtype=np.int16)
gc = rng.integers(0, 5, n, dtype=np.int16)
gs[:, r] += gr
gs[:, c] += gc
```

This creates structurally incoherent simulations:
- In a simulated home-win, the away team's goals are drawn uniformly from {0,1,2,3,4}, so the away team will "outscore" the home team ~50% of the time in goal-tally, even though the match was recorded as a home win.
- The goal difference (`gd`) is computed correctly from match outcome (±1 per match), but the goals-scored (`gs`) used as the third tiebreaker is noise uncorrelated with the match result.
- FIFA's actual tiebreaker is head-to-head results *within the group*, then GD, then GS — the code skips the head-to-head tiebreaker entirely.

### Impact
For tiebreakers, the `gs` column is pure noise. When the top 8 third-place teams are selected, their ranking by `gs` in the tiebreaker is effectively random. This likely has small average effect (ties are infrequent) but creates scenarios where clearly inferior teams advance ahead of superior ones in edge cases.

### Fix
If implementing a full goal model is out of scope, at minimum condition goals on outcome:
- Win: sample `gs_winner ~ Poisson(λ=1.8)`, `gs_loser ~ Poisson(λ=0.7)`
- Draw: `gs_both ~ Poisson(λ=1.1)`

This makes GD derivable from GS (or vice versa), which is the correct relationship.

---

## Finding 8 — `away_loss_rate` Is Populated at Inference but Not Used

**Severity: Medium | Confidence: High | Effort to fix: Low**

The inference row dict ([feature_engineering.py:316](file:///c:/Users/SAMMYAG/OneDrive/Desktop/wc2026_v3/data/feature_engineering.py#L316)) includes:

```python
"away_loss_rate": a_l,
```

`FEATURE_COLS` contains `home_loss_rate` but **not** `away_loss_rate`. The model therefore has asymmetric information: home team loss rate is used, away team loss rate is silently dropped. This introduces a structural asymmetry between the home and away team representations.

The training rows ([ingestion.py:350-351](file:///c:/Users/SAMMYAG/OneDrive/Desktop/wc2026_v3/data/ingestion.py#L350-L351)) store `home_loss_rate` and `away_loss_rate`, but since `away_loss_rate` is absent from `FEATURE_COLS`, the model never trains on it either. So the asymmetry is at least consistent, but it is still a modeling choice that appears unintentional.

### Fix
Either add `away_loss_rate` to `FEATURE_COLS` (and retrain) or remove it from `build_inference_features` row dict to make the dead code explicit.

---

## Finding 9 — H2H Registry Uses Inconsistent Name Spaces

**Severity: Medium | Confidence: High | Effort to fix: Low**

### Training path
`ingestion.py:325` calls `self.h2h.win_rate_diff(h, a)` where `h` and `a` are **raw CSV names** (e.g. `"United States"`, `"Czech Republic"`, `"Turkey"`).

### Inference path
`main.py:150` assigns `h2h_fn = data_pipeline.current_h2h`, which calls `MatchDataPipeline.current_h2h()` ([ingestion.py:375-382](file:///c:/Users/SAMMYAG/OneDrive/Desktop/wc2026_v3/data/ingestion.py#L375-L382)):

```python
def current_h2h(self, home: str, away: str) -> float:
    h_csv = TEAM_NAME_ALIASES.get(home, home)
    a_csv = TEAM_NAME_ALIASES.get(away, away)
    return self.h2h.win_rate_diff(h_csv, a_csv)
```

This correctly maps canonical → CSV names. However, since `h2h_diff` is not in `FEATURE_COLS`, neither path's H2H value reaches the model. The name-space inconsistency is therefore currently dormant — but it is a latent bug that will silently produce zeros for aliased teams the moment `h2h_diff` is re-added to `FEATURE_COLS`.

---

## Finding 10 — Third-Place Bracket Assignment May Silently Fall Back

**Severity: Medium | Confidence: Medium | Effort to fix: Medium**

### What the code does
`monte_carlo.py:158-180` precomputes a lookup table mapping every combination of 8 advancing third-place groups to their R32 opponents. If the DFS finds no valid matching for a given 8-group combination, it falls back to `list(combo)` with the comment "theoretically unreachable":

```python
matching = find_matching(0, [], set(combo))
if matching is None:
    matching = list(combo)  # Fallback, theoretically unreachable based on FIFA sets
self.t3_lookup[mask] = matching
```

### Issues
1. The `t3_allowed` dictionary ([monte_carlo.py:143-152](file:///c:/Users/SAMMYAG/OneDrive/Desktop/wc2026_v3/simulation/monte_carlo.py#L143-L152)) represents 5-element allowed sets for each of 8 hosts. With 12 groups and 8 advancing thirds, there are C(12,8) = 495 combinations. These are not all guaranteed to have valid matchings under the specified `t3_allowed` constraints, especially if the 48-team WC2026 bracket rules differ from the original FIFA specification the sets were sourced from.
2. The fallback assignment `list(combo)` assigns the `i`-th advancing group to the `i`-th host in `t3_hosts` order — an arbitrary assignment that almost certainly violates bracket rules for some combinations.
3. There is no runtime warning when the fallback fires, so invalid bracket assignments would be silent.

### Fix
Add a validation step that iterates all 495 combinations and asserts no fallback fires. Log a warning if any does. Verify `t3_allowed` against the official FIFA 2026 technical documentation.

---

## Finding 11 — `_rank_group` Has a NumPy Advanced-Indexing Bug

**Severity: Medium | Confidence: High | Effort to fix: Low**

### What the code does
`monte_carlo.py:246-247`:

```python
order = np.lexsort((-gs, -gd, -pts), axis=1)   # (n, 4) position indices
return ids[order]                                # (n, 4) local team IDs
```

`ids` is a 1-D array of shape `(4,)`. `order` is a 2-D array of shape `(n, 4)`. NumPy's advanced indexing broadcasts `ids[order]` as `ids[order[i,j]]` which correctly returns the team ID at each ranked position. **This works correctly** as long as `order` contains values in `[0, 3]`, which it does because `lexsort` returns positional indices into the group's 4 teams.

However, the same function is also called at line 311–317 in the third-place extraction path:

```python
order_idx = np.lexsort((-data["gs"], -data["gd"], -data["pts"]), axis=1)  # (n, 4)
t3_pts_cols.append(data["pts"][np.arange(n_sims), order_idx[:, 2]])
```

This advanced indexing is correct. But `_rank_group` is called on `data["pts"], data["gd"], data["gs"], data["ids"]` — the `ids` here are **local** team IDs (integers 0–47), not positional indices 0–3. The `order` from `lexsort` produces positional indices 0–3, and indexing `ids[order]` maps these to local team IDs. This is correct only if `ids` is a contiguous 4-element array in group order — which it is, per `group_local_ids`.

The bug risk is subtle: if `_rank_group` were ever called with an `ids` array that isn't simply `[group_teams[0..3]]`, the indexing would silently produce wrong team IDs. The current code is safe but fragile.

---

## Finding 12 — `match_type_enc` Encoding Leaks Tournament-Level Prior at Group Stage

**Severity: Low | Confidence: Medium | Effort to fix: Low**

Every inference call passes `tournament="FIFA World Cup"` which maps to `match_type_enc = 1.0`. During training, `match_type_enc` carries the full distribution across tournament types. The model learns that high `match_type_enc` correlates with certain outcome distributions (e.g., fewer home wins, more competitive draws).

At inference, every single match gets `match_type_enc = 1.0`. The model is therefore in a slightly different regime at inference than during any individual training fold (where the average `match_type_enc` is much lower than 1.0). This is the correct and intended behavior — WC matches should be treated as WC matches — but the feature now functions essentially as a constant at inference, contributing zero discriminative power across matchups.

**Consequence**: `match_type_enc` has 6.8% feature importance (gain), but all of that gain is from training-time signal. At inference it cannot differentiate between matchups. The model is using this slot for information it cannot leverage in production.

---

## Finding 13 — Log-Loss Std Not Reported Across Folds

**Severity: Low | Confidence: High | Effort to fix: Trivial**

`cv_metrics.json` reports `std_accuracy` (2.8pp) but only `mean_logloss` (no std). Given the fold-5 logloss cliff (1.0519 vs 0.9927 for fold 4), the standard deviation would be informative:

```
logloss values: [1.0071, 1.0048, 0.9948, 0.9927, 1.0519]
std ≈ 0.022
```

The mean 1.010 with this variation is not well-characterised by the mean alone.

---

## Finding 14 — Backtest `group_acc` Is Computed From Wrong Column

**Severity: Low | Confidence: High | Effort to fix: Low**

`world_cup_backtest.py:437`:

```python
group_acc = _stage_top_accuracy(results_df, "round_of_16_pct", actual_group_qualifiers, 16)
```

The `HistoricalSimulationResults.to_dataframe()` uses `round_of_16_pct` which is populated from `round_of_16_counts`. In the historical 32-team simulator, `round_of_16_counts` is incremented for group winners *and* runners-up ([world_cup_backtest.py:258-259](file:///c:/Users/SAMMYAG/OneDrive/Desktop/wc2026_v3/backtesting/world_cup_backtest.py#L258-L259)):

```python
np.add.at(round_of_16_counts, ranked[:, 0], 1)
np.add.at(round_of_16_counts, ranked[:, 1], 1)
```

This is only the top-2 from each of 8 groups = 16 teams, which is correct for a 32-team tournament. But the column is named `round_of_16_pct` in the output, and it is being used to evaluate "group qualification accuracy." The naming is misleading: qualifying from the group *is* reaching the Round of 16 in a 32-team format. This is logically correct but confusingly named in a codebase where "R16" has a different meaning in the 48-team structure.

---

## Finding 15 — Deprecated `_pd.np` Access

**Severity: Low | Confidence: High | Effort to fix: Trivial**

`feature_engineering.py:162-169` uses:

```python
raw["target"] = (_pd.np.where(...) if hasattr(_pd, "np") else ...)
```

`pandas.np` was deprecated in pandas 1.0 and removed in pandas 2.0. The `hasattr` guard will silently fall through to the `numpy` fallback on modern pandas installations, making the first branch dead code. There is a runtime cost to the `hasattr` check on every training run.

---

## Summary of Structural Concerns

### What the metrics actually tell you

| Metric | Value | What it means |
|--------|-------|----------------|
| CV Accuracy 50.6% | Better than 33% random | Model extracts signal |
| CV Accuracy vs. baseline | Must compare to majority-class baseline | If home-win rate in training data is ~48%, a trivial classifier scores ~48% |
| Fold 5 accuracy 45.6% | **Below training-data distribution** | Recent football may be harder to predict |
| Backtest champion probs | Inflated (Finding 1) | Not credible as out-of-sample evidence |
| Feature importance (gain) | elo_diff dominates at 17.7% | Model is an Elo wrapper with noisy adjuncts |

### Dead code inventory

| Computed | Stored | Used by model |
|----------|--------|---------------|
| `mv_diff` | Yes (inference row) | **No** |
| `rank_diff` | Yes (inference row) | **No** |
| `h2h_diff` | Yes (training + inference rows) | **No** |
| `away_loss_rate` | Yes (training + inference rows) | **No** |

**Four features are computed on every run but contribute zero to predictions.** Either they should be in the model or their computation should be removed.

### Key risk summary

The most important single issue is **Finding 1**: the backtest numbers are not out-of-sample. Everything that was "validated" by the historical World Cup backtest must be treated as suspect. The model may well generalise, but you do not currently have evidence of it. All other findings are fixable with low-to-medium effort and most are code consistency issues rather than fundamental modeling errors.
