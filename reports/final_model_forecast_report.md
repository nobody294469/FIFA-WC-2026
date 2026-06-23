# 2026 FIFA World Cup — Final Model Forecast Report

This document finalizes the state of the Match Outcome Model following the Corpus Expansion audit and benchmarks. It includes the finalized feature set, CV/OOS metrics, and the post-expansion World Cup 2026 forecast odds, derived from 100,000 Monte Carlo simulations.

---

## 1. Final Model Architecture

The production model uses an `XGBClassifier` optimized with an expanding-window time-aware cross-validation scheme. The model probabilities are calibrated using **Platt Scaling (Sigmoid)**.

### Final Feature List (14 Features)
1. `elo_diff`
2. `h2h_diff`
3. `form_diff`
4. `opp_adj_form_diff`
5. `gd_diff`
6. `home_form`
7. `away_form`
8. `home_win_rate`
9. `away_win_rate`
10. `home_draw_rate`
11. `away_draw_rate`
12. `home_loss_rate`
13. `match_type_enc`
14. `neutral_venue`

---

## 2. Final Model Metrics

### Cross-Validation Metrics (5-Fold TimeSeriesSplit)
*Trained on the expanded 7,326-match Scenario 1 corpus.*

* **Mean Accuracy:** 0.6010 ± 0.0284
* **Mean Log Loss:** 0.8760
* **Mean Brier Score:** 0.5146
* **Mean Macro F1:** 0.4467

### True Out-of-Sample (OOS) Backtest Metrics
*Historical World Cup prediction testing (2014, 2018, 2022).*

* **Mean Brier Score:** 0.1194
* **Mean Group Qualification Accuracy:** 0.625
* **Mean Quarterfinal Accuracy:** 0.542
* **Mean Semifinal Accuracy:** 0.417

---

## 3. Post-Expansion WC2026 Forecast
*Simulated 100,000 times using the expanded production model.*

### Top-10 Championship Probabilities
| Rank | Team | Champion % | Elo Rating | Elo Rank | Δ vs Pre-Expansion |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **Spain** | 18.73% | 2094 | 1 | +5.27% |
| 2 | **Argentina** | 15.43% | 2072 | 2 | +4.46% |
| 3 | **France** | 11.17% | 2025 | 3 | +1.46% |
| 4 | **Brazil** | 6.18% | 1951 | 5 | +1.54% |
| 5 | **England** | 5.54% | 1952 | 4 | +1.21% |
| 6 | **Portugal** | 4.83% | 1948 | 6 | +0.01% |
| 7 | **Colombia** | 4.00% | 1940 | 7 | 0.00% |
| 8 | **Ecuador** | 3.33% | 1930 | 8 | +0.02% |
| 9 | **Netherlands** | 3.26% | 1903 | 10 | -0.47% |
| 10 | **Germany** | 3.11% | 1906 | 9 | -0.32% |

### Forecast Stability & Odds Concentration
The most notable result of the expanded corpus was a **sharp increase in the concentration of title odds among the heavy favorites**. By training the model on thousands of additional David-vs-Goliath matches (World Cup teams playing non-World Cup teams), the model learned a much higher degree of certainty regarding favorites dispatching underdogs. 

* **Pre-Expansion Top-3 Concentration:** 34.14%
* **Post-Expansion Top-3 Concentration:** 45.33% (+11.19%)

Spain (+5.27%) and Argentina (+4.46%) saw massive bumps to their title probabilities, while mid-tier teams saw proportional decreases:
* Morocco: -1.29%
* Switzerland: -0.96%
* Japan: -0.89%

---

## 4. Audit Decisions Summary

### Accepted Improvements
* **Corpus Expansion (Scenario 1):** Allowed the model to train on matches where at least one team was a WC2026 competitor. Added ~5,400 training rows and massively improved CV Log Loss (1.017 → 0.876).
* **Head-to-Head (H2H) feature:** Fixed a bug causing the raw `h2h_diff` feature to be dropped from production.

### Rejected Improvements
* **Time-Decayed H2H:** Rejected because the raw baseline feature had <0.3% global importance. Squeezing marginal value out of it wasn't worth the architectural complexity.
* **Tournament-Weighted Form:** The benchmark proved that time-weighting the form actually degraded Log Loss, Brier, and Accuracy relative to the unweighted 10-match rolling window.
* **Isotonic Calibration:** Failed the calibration benchmark, performing strictly worse than Platt Scaling on CV splits.

---

## 5. Current Limitations & Recommended Future Work
1. **Knockout ET/Penalty Model:** The model currently treats all draws in the knockout stage rigidly. We urgently need a dedicated mathematical resolution model for Extra Time and Penalty Shootouts.
2. **Host Advantage Dynamics:** The model evaluates all matches as taking place at a purely neutral venue. This under-values the true probability of USA, Mexico, and Canada succeeding in the actual tournament.
