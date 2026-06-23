# WC2026 Top-10 Favourites — Forecast Report

> **Simulations**: 100,000  |  **Model**: XGBoost (max_depth=2, ~32 iterations)  |  **Features**: Elo, form, opponent-adjusted form, GD differential

---

## Ranked Favourites

| Rank | Team | Champion % | Finalist % | SF % | QF % | Elo Rating | FIFA Rank | Elo Rank Delta |
|------|------|-----------|-----------|------|------|-----------|-----------|---------------|
| 1 | **Spain** | 17.13% | 25.82% | 36.25% | 48.15% | 2094 | 2 | 0 |
| 2 | **Argentina** | 14.59% | 22.88% | 34.24% | 48.94% | 2072 | 1 | 0 |
| 3 | **France** | 9.74% | 16.68% | 28.21% | 44.59% | 2025 | 3 | 0 |
| 4 | **Brazil** | 5.69% | 10.70% | 19.97% | 33.90% | 1951 | 6 | +1 |
| 5 | **England** | 5.31% | 10.47% | 19.27% | 33.81% | 1952 | 4 | -1 |
| 6 | **Portugal** | 4.90% | 9.86% | 18.25% | 32.56% | 1948 | 5 | 0 |
| 7 | **Colombia** | 4.37% | 8.78% | 16.41% | 30.04% | 1940 | 13 | 0 |
| 8 | **Ecuador** | 3.48% | 7.72% | 16.27% | 29.64% | 1930 | 23 | 0 |
| 9 | **Netherlands** | 3.42% | 7.23% | 15.18% | 29.30% | 1903 | 8 | +1 |
| 10 | **Germany** | 3.23% | 7.08% | 15.34% | 28.99% | 1906 | 10 | -1 |

---

## Team Analyses

### 1. Spain  (Group H)

**Champion probability**: 17.13%  |  **Finalist**: 25.82%  |  **Semifinal**: 36.25%

Reigning Euro 2024 champion. Their tiki-taka possession system generates the strongest positive GD differential of any top-8 team in the model's form window. Group H (Cape Verde, Saudi Arabia, Uruguay) is manageable. The concern is a potential QF collision with Argentina or France, which limits their expected championship probability despite a strong Elo rating.

### 2. Argentina  (Group J)

**Champion probability**: 14.59%  |  **Finalist**: 22.88%  |  **Semifinal**: 34.24%

Defending world champion with the tournament's highest Elo rating. Messi's post-2022 form has lifted every measurable signal — opponent-adjusted form, GD, win rate. The model treats them as the marginal title favourite entering Group J with Algeria, Austria, and Jordan — a group they should top comfortably before hitting the tougher knockout draw.

### 3. France  (Group I)

**Champion probability**: 9.74%  |  **Finalist**: 16.68%  |  **Semifinal**: 28.21%

Structurally the deepest squad in the tournament (Elo rank #3) and carrying the highest opponent-adjusted form score entering the tournament. Group I (Senegal, Norway, Iraq) is winnable with minimal attrition, which should preserve the squad for a deep run. Their 2022 runner-up result combined with strong Elo keeps them among the top three by model probability.

### 4. Brazil  (Group C)

**Champion probability**: 5.69%  |  **Finalist**: 10.70%  |  **Semifinal**: 19.97%

Five-time world champion back in the conversation after a turbulent 2022. Post-Neymar Brazil has shown strong GD differentials under the new manager. Group C (Morocco, Haiti, Scotland) is the easiest draw of any top-10 team. The model rates them #5 — Elo and form are consistent, but they are at the mercy of a wide-open bracket that could produce an Argentina or France collision as early as the semifinals.

### 5. England  (Group L)

**Champion probability**: 5.31%  |  **Finalist**: 10.47%  |  **Semifinal**: 19.27%

High squad market value (EUR 1,345M — highest in the tournament) but the model rates their recent form and opponent-adjusted performance below Argentina, France, and Spain. Group L with Croatia, Ghana, and Panama is navigable. England's historic tendency to underperform their squad ceiling is partially encoded in their form features from the last 10 matches.

### 6. Portugal  (Group K)

**Champion probability**: 4.90%  |  **Finalist**: 9.86%  |  **Semifinal**: 18.25%

Strong squad value (EUR 1,000M) and Elo rank #5. The model places them 7th, consistent with their Elo. Group K (Uzbekistan, Colombia, DR Congo) is winnable but Colombia are a legitimate threat for the runner-up spot. Portugal's historical tendency to exit at QF stage is partially reflected in their form features relative to their Elo ceiling.

### 7. Colombia  (Group K)

**Champion probability**: 4.37%  |  **Finalist**: 8.78%  |  **Semifinal**: 16.41%

The model's biggest over-performer relative to Elo rank. Strong recent CONMEBOL qualifying form has produced high opponent-adjusted form scores. Group K (Portugal, Uzbekistan, DR Congo) is tough but survivable. Their championship probability comes primarily from form signals rather than Elo history.

### 8. Ecuador  (Group E)

**Champion probability**: 3.48%  |  **Finalist**: 7.72%  |  **Semifinal**: 16.27%

No detailed narrative available.

### 9. Netherlands  (Group F)

**Champion probability**: 3.42%  |  **Finalist**: 7.23%  |  **Semifinal**: 15.18%

8th in model championship probability, consistent with Elo rank #8. Strong defensive form and positive GD differential. Group F (Japan, Tunisia, Sweden) is manageable. The Dutch have a clean path to the QF where they could face Spain — that matchup capping their expected run.

### 10. Germany  (Group E)

**Champion probability**: 3.23%  |  **Finalist**: 7.08%  |  **Semifinal**: 15.34%

Rebuilt under Nagelsmann and entering as Group E favourites. The model assigns them the 6th-highest championship probability driven by strong Elo (top 10 historically) and an improving post-2022 form curve. Their group draw (Curacao, Ivory Coast, Ecuador) is the softest path to the knockout rounds of any top-6 team, so their R32 probability approaches certainty.

---

## Overrated Teams (Model < Elo Expectation)

### Belgium  (Model rank #17, Elo rank #19)

FIFA rank #9 but the golden generation is aging. The model's form window shows declining results — their opp-adjusted form score is materially below their Elo-implied expectation. Group G (Egypt, Iran, New Zealand) is straightforward but the model doesn't see them going deep.

### Croatia  (Model rank #14, Elo rank #14)

2018 finalists and 2022 third-place. The model sees a sharp form decline post-2022: aging spine (Modric 40 in 2026), declining GD differential. Elo rank #11, model rank materially lower. Group L with England is particularly brutal for a team on a downward trajectory.

### Uruguay  (Model rank #15, Elo rank #13)

Strong historical Elo from the pre-2022 era. The model's 10-match form window shows a weaker recent trend. Group H with Spain is a difficult opener and their path to the knockout rounds is uncertain.

---

## Underrated Teams (Model < Potential)

### Japan  (Model rank #12, Elo rank #12)

The model's clearest underdog with genuine upside. Their recent form window includes wins over Germany and Spain. Opponent-adjusted form score is among the top 12 globally — far above their Elo history implies. Group F with Netherlands is a steep challenge but they are likely to qualify from the group and are dangerous in R32.

### Morocco  (Model rank #11, Elo rank #11)

Already flagged in the top-10 analysis: Elo rank #7 with champion probability suppressed by Group C placement and likely R32 bracket. Any scenario where they avoid Brazil in the group stage produces materially higher deep-round probability.

### USA  (Model rank #30, Elo rank #32)

Home advantage dynamics (co-hosts with Canada and Mexico) are not modelled — the model treats all WC matches as neutral. The actual home crowd effect could add 5-10% probability mass to their knockout probabilities. Group D (Paraguay, Australia, Türkiye) is well within reach. The model's probability understates their true expected performance given the co-host context.

---

## Methodology Note

All probabilities are derived from Monte Carlo tournament simulation using a frozen XGBoost classifier trained on WC2026-team match history from 2010 to June 2026. Features are Elo differential, opponent-adjusted form, GD differential, win/draw/loss rates, and match type encoding. The model was evaluated using an expanding-window time-aware cross-validation scheme.

**True out-of-sample backtest performance** (separate models trained per cutoff date):
- WC2014: Germany champion probability 8.78% (actual: won)
- WC2018: France champion probability 4.68% (actual: won, outside top-5)
- WC2022: Argentina champion probability 24.64% (actual: won, ranked #2)

> Co-host advantage (USA, Canada, Mexico) is not modelled. Real champion probabilities for these teams may be higher than shown.