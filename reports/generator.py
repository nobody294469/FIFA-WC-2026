"""reports/generator.py — Formatted console and file reports."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
from core.config import REPORT_DIR
from core.logger import get_logger

log = get_logger("reports")
_LINE = "─" * 82


def _pct(v: float) -> str:
    return f"{v:6.2f}%"


def _bar(v: float, width: int = 22) -> str:
    filled = round(float(v) / 100 * width)
    return "█" * filled + "░" * (width - filled)


class ReportGenerator:
    def __init__(self, out: Path = REPORT_DIR) -> None:
        self.out = out
        out.mkdir(parents=True, exist_ok=True)

    def model_performance(self, cv_metrics: Dict, importance_df: pd.DataFrame) -> str:
        lines = [
            _LINE,
            "  MODEL PERFORMANCE  —  XGBClassifier  (Home Win / Draw / Away Win)",
            _LINE,
            f"  CV Folds        : {len(cv_metrics.get('folds', []))}",
            f"  Mean Accuracy   : {cv_metrics.get('mean_accuracy',0):.4f}  "
            f"± {cv_metrics.get('std_accuracy',0):.4f}",
            f"  Mean Log-Loss   : {cv_metrics.get('mean_logloss',0):.4f}",
            f"  Mean Brier Score: {cv_metrics.get('mean_brier',0):.4f}",
            f"  Mean F1 (macro) : {cv_metrics.get('mean_f1_macro',0):.4f}",
            "",
            "  Per-fold breakdown:",
        ]
        for fold in cv_metrics.get("folds", []):
            lines.append(
                f"    Fold {fold['fold']}: acc={fold['accuracy']:.4f}  "
                f"logloss={fold['logloss']:.4f}  brier={fold.get('brier', 0):.4f}  f1={fold['f1_macro']:.4f}"
            )
        lines += ["", "  Feature Importance (normalised gain):"]
        for _, row in importance_df.iterrows():
            bar = _bar(float(row["importance_gain"]) * 100, width=18)
            lines.append(f"    {row['feature']:<30} {bar}  {float(row['importance_gain']):.4f}")
        lines.append(_LINE)
        report = "\n".join(lines)
        (self.out / "model_performance.txt").write_text(
    report,
    encoding="utf-8"
)
        return report

    def match_prediction(
        self,
        home: str,
        away: str,
        probs: tuple,
        shap_df: Optional[pd.DataFrame] = None,
    ) -> str:
        p_h, p_d, p_a = probs
        lines = [
            _LINE,
            f"  MATCH PREDICTION  |  {home}  vs  {away}",
            _LINE,
            f"  {home:<22} Win  : {_pct(p_h*100)}  {_bar(p_h*100)}",
            f"  {'Draw':<26}  : {_pct(p_d*100)}  {_bar(p_d*100)}",
            f"  {away:<22} Win  : {_pct(p_a*100)}  {_bar(p_a*100)}",
        ]
        if shap_df is not None and not shap_df.empty:
            lines += ["", "  ■ Top SHAP Drivers (Home Win class):"]
            for _, row in shap_df.iterrows():
                lines.append(
                    f"    {row['feature']:<30}  val={float(row['feature_value']):+.3f}"
                    f"  shap={float(row['shap_value']):+.4f}  {row['direction']}"
                )
        lines.append(_LINE)
        return "\n".join(lines)

    def elo_ratings(self, elo_df: pd.DataFrame, top_n: int = 20) -> str:
        lines = [
            _LINE,
            f"  ELO RATINGS  —  Top {top_n} WC2026 Teams  (built from real results)",
            _LINE,
            f"  {'Rank':<6} {'Team':<28} {'Elo':>8}",
            "  " + "─" * 45,
        ]
        for _, row in elo_df.head(top_n).iterrows():
            lines.append(f"  {int(row['elo_rank']):<6} {row['team']:<28} {row['elo']:>8.1f}")
        lines.append(_LINE)
        return "\n".join(lines)

    def tournament_forecast(
        self, results_df: pd.DataFrame, n_sims: int, elapsed: float
    ) -> str:
        lines = [
            _LINE,
            "  2026 FIFA WORLD CUP  —  MONTE CARLO TOURNAMENT FORECAST",
            f"  Simulations : {n_sims:,}   |   Runtime: {elapsed:.2f}s",
            _LINE,
            f"  {'#':<4} {'Team':<26} {'Champion':>10} {'Final':>8} "
            f"{'Semi':>8} {'QF':>8} {'R16':>8} {'R32':>8} {'Group Exit':>11}",
            "  " + "─" * 81,
        ]
        for _, row in results_df.iterrows():
            lines.append(
                f"  {int(row['rank']):<4} {row['team']:<26}"
                f" {_pct(row['champion_pct']):>10}"
                f" {_pct(row['finalist_pct']):>8}"
                f" {_pct(row['semifinal_pct']):>8}"
                f" {_pct(row['quarterfinal_pct']):>8}"
                f" {_pct(row['r16_pct']):>8}"
                f" {_pct(row['r32_pct']):>8}"
                f" {_pct(row['group_exit_pct']):>11}"
            )
        lines.append(_LINE)
        report = "\n".join(lines)
        (self.out / "tournament_forecast.txt").write_text(
    report,
    encoding="utf-8"
)
        log.info("Forecast saved → %s", self.out / "tournament_forecast.txt")
        return report

    def save_json(self, results_df: pd.DataFrame, name: str = "wc2026_forecast.json") -> None:
        p = self.out / name
        results_df.to_json(p, orient="records", indent=2)
        log.info("JSON results → %s", p)