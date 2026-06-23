# Poisson Goal-Scoring — Phase 1 Benchmark Report

> **Historical WC draw rate**: 22.2%  |  **Historical WC goals/match**: 2.82  |  **λ calibration μ**: 1.4108 goals/team/match
> **Backtest**: WC 2014, 2018, 2022 (OOS)  |  **Gate**: FAILED ✗

---

## 1. Goal Distribution Statistics

How well does each simulator reproduce historical WC goal-scoring patterns?

| Metric | Historical WC | Binary Sim | Poisson Sim | Δ (vs historical) |
|:---|---:|---:|---:|---:|
| Draw rate | 22.20% | 24.84% | 20.80% | Binary: +2.64pp  Poisson: -1.40pp |
| Goals/match | 2.82 | N/A (uniform) | 2.82 | Poisson: -0.00 |
| Mean \|GD\| | 1.92 | 1.28 | 3.22 | — |

---

## 2. Standings Disagreement Rate

How often do group standings differ between the two simulators?

| Metric | Value |
|:---|---:|
| Top-2 advancement disagreement | 89.47% of group-sims |
| Group winner disagreement | 59.99% of group-sims |

> A higher disagreement rate means the tiebreaker change has material effect on which teams advance.

---

## 3. Historical WC Backtest — Accuracy Metrics

| Year | Champion | Binary Champ% | Poisson Champ% | Binary GrpAcc | Poisson GrpAcc | Binary Brier | Poisson Brier |
|:---|:---|---:|---:|---:|---:|---:|---:|
| 2014 | Germany | 11.93% | 13.30% | 0.625 | 0.625 | 0.1223 | 0.1226 |
| 2018 | France | 5.12% | 5.45% | 0.750 | 0.750 | 0.1227 | 0.1213 |
| 2022 | Argentina | 16.90% | 18.37% | 0.562 | 0.562 | 0.1163 | 0.1196 |
| **Mean** | — | 11.32% | 12.37% | 0.646 | 0.646 | **0.1204** | **0.1212** |

### Aggregate Deltas (Poisson − Binary)

| Metric | Binary | Poisson | Delta | Gate? |
|:---|---:|---:|---:|:---|
| **Mean Advancement Brier** | 0.12042 | 0.12116 | +0.00074 | Δ ≤ −0.005 |
| Mean Group Accuracy | 0.64583 | 0.64583 | 0.00000 | Δ ≥ +0.025 |
| Mean QF Accuracy | 0.54167 | 0.58333 | +0.04167 ✓ | — |
| Mean SF Accuracy | 0.41667 | 0.41667 | 0.00000 | — |
| Mean Champion % on actual winner | 11.315 | 12.372 | +1.05667 ✓ | — |

---

## 4. Computational Overhead

| Metric | Value |
|:---|---:|
| Binary simulator mean time | 0.557s per 50,000 sims |
| Poisson simulator mean time | 0.609s per 50,000 sims |
| Overhead ratio | **1.09×** slower |

> Poisson sampling replaces `rng.integers()` with `rng.poisson()`, which is marginally slower.
> An overhead ratio < 2.0 is acceptable; the simulation remains fast.

---

## 5. Verdict

### ❌ RECOMMENDATION: REJECT

The Poisson simulator did not satisfy any gate criterion. The binary ±1 GD + uniform goals approach, despite being synthetic, produces equivalent or better historical accuracy in simulation. This may indicate that the group advancement noise introduced by realistic Poisson sampling outweighs its benefits over 3 tournaments. Consider revisiting with a larger tournament backtest set (include continental tournaments: Copa América, EUROS, AFCON) to produce a more statistically robust sample before adoption.

---

> **Gate Criteria**: Mean Advancement Brier Δ ≤ −0.005  OR  Mean Group Accuracy Δ ≥ +0.025  OR  Draw rate closer to historical by > 5pp (abs)

> **λ formula**: λ_h = 2μ × P_win_h / (P_win_h + P_win_a),  λ_a = 2μ − λ_h  where μ = 1.4108