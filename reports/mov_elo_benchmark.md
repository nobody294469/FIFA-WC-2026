# Margin-of-Victory Elo — Phase 1 Benchmark Report

> **Simulations**: 50,000  |  **CV folds**: 5 (TimeSeriesSplit)  |  **Gate**: FAILED ✗

---

## 1. Elo-Only Evaluation (No ML Model)

These metrics evaluate the raw Elo signal quality before any XGBoost model is involved.

| Metric | Binary Elo | MoV Elo | Delta |
|:---|---:|---:|---:|
| Pearson Correlation (elo_diff vs outcome) | 0.36002 | 0.36197 | +0.00195 ✓ |
| Sign Accuracy (non-draw matches) | 0.69556 | 0.69417 | -0.00139 |
| Elo-Implied Log Loss | 1.03476 | 1.04911 | +0.01435 |

---

## 2. ML Cross-Validation Metrics (5-Fold TimeSeriesSplit)

| Metric | Binary Elo | MoV Elo | Delta | Gate? |
|:---|---:|---:|---:|:---|
| Accuracy | 0.51938 | 0.52375 | +0.00437 ✓ | — |
| **Log Loss** | 1.00448 | 1.00351 | -0.00097 | Δ ≤ −0.005 |
| **Brier Score** | 0.60102 | 0.60041 | -0.00062 | Δ ≤ −0.002 |
| Macro F1 | 0.38449 | 0.38779 | +0.00330 ✓ | — |
| elo_diff Feature Importance | 0.242551 | 0.228766 | -0.01378 | — |

---

## 3. Historical WC Backtest (2014 / 2018 / 2022)

| Year | Champion | Binary Champ% | MoV Champ% | Binary Brier | MoV Brier | Binary Tau | MoV Tau |
|:---|:---|---:|---:|---:|---:|---:|---:|
| 2014 | Germany | 11.89% | 11.73% | 0.1198 | 0.1163 | 0.3097 | 0.3780 |
| 2018 | France | 5.26% | 6.21% | 0.1227 | 0.1209 | 0.3487 | 0.3780 |
| 2022 | Argentina | 16.81% | 15.94% | 0.1164 | 0.1200 | 0.3487 | 0.3292 |
| **Mean** | — | — | — | **0.1196** | **0.1191** | — | — |

> **Historical Brier gate** (Δ ≤ −0.003): Δ = -0.00059  ✗ FAILED

---

## 4. WC2026 Championship Probabilities (Top 20)

| Rank | Team | Binary Champ% | MoV Champ% | Delta |
|:---|:---|---:|---:|---:|
| 1 | Spain | 13.85% | 14.54% | +0.69pp |
| 2 | Argentina | 11.44% | 11.54% | +0.11pp |
| 3 | France | 9.20% | 9.68% | +0.49pp |
| 4 | Brazil | 5.31% | 5.62% | +0.30pp |
| 5 | Portugal | 5.14% | 4.57% | -0.57pp |
| 6 | England | 5.03% | 6.16% | +1.13pp |
| 7 | Ecuador | 4.40% | 4.21% | -0.19pp |
| 8 | Colombia | 4.34% | 5.28% | +0.93pp |
| 9 | Netherlands | 3.52% | 3.66% | +0.14pp |
| 10 | Germany | 3.17% | 2.81% | -0.36pp |
| 11 | Morocco | 3.07% | 2.79% | -0.28pp |
| 12 | Japan | 2.79% | 2.76% | -0.02pp |
| 13 | Mexico | 2.64% | 2.12% | -0.52pp |
| 14 | Uruguay | 2.53% | 2.32% | -0.20pp |
| 15 | Türkiye | 2.20% | 1.85% | -0.35pp |
| 16 | Croatia | 2.14% | 1.80% | -0.34pp |
| 17 | Switzerland | 2.13% | 2.43% | +0.31pp |
| 18 | Belgium | 1.94% | 1.95% | +0.01pp |
| 19 | Senegal | 1.47% | 1.01% | -0.46pp |
| 20 | Canada | 1.46% | 1.40% | -0.06pp |

**Top-3 concentration**: Binary = 34.5%  |  MoV = 35.8%  |  Δ = +1.3pp

---

## 5. Verdict

### ❌ RECOMMENDATION: REJECT

MoV Elo did not satisfy any gate criterion. The marginal improvement is not sufficient to justify modifying production code. The current binary Elo remains the production standard. Consider revisiting with a larger dataset or exploring alternative Elo extensions.

---

> **Gate Criteria**: Log Loss Δ ≤ −0.005  OR  Brier Score Δ ≤ −0.002  OR  Historical WC Brier Δ ≤ −0.003
> **Formula used**: G = min((11 + GD) / 8, 3.0) for wins; G = 1.0 for draws