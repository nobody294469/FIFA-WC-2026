# FIFA World Cup 2026 Forecast Model

## Overview

This project is a machine learning-based forecasting system for the FIFA World Cup 2026.

The pipeline combines Elo ratings, recent team form, historical head-to-head performance, and calibrated probabilistic modeling to estimate match outcomes and simulate the entire tournament using large-scale Monte Carlo simulation.

The final model was evaluated using strict temporal validation and historical World Cup backtesting before generating tournament forecasts.

---

## Key Features

* Dynamic Elo rating system
* Form-based team strength metrics
* Head-to-head performance features
* XGBoost multiclass outcome prediction
* Probability calibration using Platt Scaling
* Hybrid calibration for elite-vs-elite matchups
* Monte Carlo World Cup simulation engine
* Historical World Cup backtesting framework
* Interactive dashboard for forecast visualization

---

## Dataset

Source:

* International football results dataset (1872–present)
* FIFA World Cup 2026 qualified teams
* Historical international match results

Final production training corpus:

* 7,326 modern international matches

---

## Model Features

The final production model uses 14 predictive features:

1. elo_diff
2. h2h_diff
3. form_diff
4. opp_adj_form_diff
5. gd_diff
6. home_form
7. away_form
8. home_win_rate
9. away_win_rate
10. home_draw_rate
11. away_draw_rate
12. home_loss_rate
13. match_type_enc
14. neutral_venue

---

## Final Validation Metrics

### Time-Series Cross Validation

| Metric      | Score  |
| ----------- | ------ |
| Accuracy    | 0.6018 |
| Log Loss    | 0.8760 |
| Brier Score | 0.5146 |
| Macro F1    | 0.4498 |

### Historical World Cup Backtest

| Metric                       | Score  |
| ---------------------------- | ------ |
| Group Qualification Accuracy | 0.625  |
| Quarterfinal Accuracy        | 0.542  |
| Semifinal Accuracy           | 0.417  |
| Mean Brier Score             | 0.1194 |

---

## FIFA World Cup 2026 Forecast

Based on 100,000 Monte Carlo simulations.

| Rank | Team        | Championship Probability |
| ---- | ----------- | ------------------------ |
| 1    | Spain       | 17.13%                   |
| 2    | Argentina   | 14.59%                   |
| 3    | France      | 9.74%                    |
| 4    | Brazil      | 5.69%                    |
| 5    | England     | 5.31%                    |
| 6    | Portugal    | 4.90%                    |
| 7    | Colombia    | 4.37%                    |
| 8    | Ecuador     | 3.48%                    |
| 9    | Netherlands | 3.42%                    |
| 10   | Germany     | 3.23%                    |

---

## Project Structure

```text
backtesting/
cache/
core/
dashboard/
data/
models/
reports/
scripts/
simulation/
```

---

## Running The Project

Train the model:

```bash
python scripts/train.py
```

Generate forecasts:

```bash
python scripts/wc2026_forecast.py --sims 100000
```

Launch dashboard:

```bash
streamlit run dashboard/app.py
```

---

## Future Improvements

* Dynamic host nation advantage
* Player-level injury and squad availability modelling
* Transfer-market based roster strength adjustments
* Automated forecast dashboard deployment
* Continuous forecast updates

---

## Disclaimer

This project is an educational and research-oriented forecasting system. Football remains highly uncertain, and forecast probabilities should not be interpreted as guarantees of future outcomes.
