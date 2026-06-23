"""
ml_engine.py — XGBoost multi-class classifier trained on real match history.

Architecture
────────────
• Expanding-window time-aware CV (5 folds) for forward-looking performance estimation
• Full-dataset refit produces the production model
• SHAP TreeExplainer for local feature attribution
• ProbabilityMatrixBuilder: pre-computes the full 48×48 match probability
• Full-dataset refit produces the production model
• SHAP TreeExplainer for local feature attribution
• ProbabilityMatrixBuilder: pre-computes the full 48×48 match probability
  matrix (2,256 ordered pairs) in one XGBoost forward pass, then stores
  three (48×48) numpy arrays: P_win, P_draw, P_loss. The Monte Carlo loop
  performs O(1) array lookups — no model calls inside the simulation.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

from core.config import (
    HYBRID_CALIBRATION_RANK_THRESHOLD,
    MODEL_CV_FOLDS,
    MODEL_DIR,
    XGB_EARLY_STOPPING_ROUNDS,
    XGB_PARAMS,
)
from core.logger import get_logger
from data.feature_engineering import FEATURE_COLS, build_inference_features

warnings.filterwarnings("ignore", category=UserWarning)
log = get_logger("ml_engine")

_MODEL_PATH = MODEL_DIR / "xgb_model.json"
_CALIBRATOR_PATH = MODEL_DIR / "platt_calibrator.pkl"
_HYBRID_CALIBRATOR_PATH = MODEL_DIR / "hybrid_calibrator.pkl"
_IMPORTANCE_PATH = MODEL_DIR / "feature_importance.json"
_CV_METRICS_PATH = MODEL_DIR / "cv_metrics.json"


def _multiclass_brier_score(y_true: pd.Series, proba: np.ndarray) -> float:
    y_idx = np.asarray(y_true, dtype=np.int64)
    y_onehot = np.eye(proba.shape[1], dtype=np.float64)[y_idx]
    return float(np.mean(np.sum((proba - y_onehot) ** 2, axis=1)))


# ═══════════════════════════════════════════════════════════════════════════════
# Match Outcome Model
# ═══════════════════════════════════════════════════════════════════════════════

class MatchOutcomeModel:
    """
    XGBoost 3-class classifier: 0=home win, 1=draw, 2=away win.

    Outputs calibrated probability distributions via the softprob objective.
    SHAP TreeExplainer provides exact local feature attributions.
    """

    def __init__(self) -> None:
        self.model:            Optional[xgb.XGBClassifier] = None
        self.calibrated_model: Optional[CalibratedClassifierCV] = None
        self.local_isotonic_models: Optional[List[IsotonicRegression]] = None
        self._explainer:       Optional[shap.TreeExplainer] = None
        self.feat_importance:  Dict[str, float] = {}
        self.cv_metrics:       Dict = {}
        self.production_best_iteration: Optional[int] = None
        self.final_training_config: Dict = {}

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MatchOutcomeModel":
        """Train with expanding-window time-aware CV then refit on full dataset."""
        log.info("Training XGBClassifier on %d samples × %d features", len(X), X.shape[1])

        tscv = TimeSeriesSplit(n_splits=MODEL_CV_FOLDS)
        fold_metrics = []
        best_iterations = []
        cv_splits = list(tscv.split(X))

        for fold, (tr_idx, val_idx) in enumerate(cv_splits, 1):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

            cv_params = dict(XGB_PARAMS)
            cv_params["early_stopping_rounds"] = XGB_EARLY_STOPPING_ROUNDS
            clf = xgb.XGBClassifier(**cv_params)
            clf.fit(
                X_tr,
                y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            proba = clf.predict_proba(X_val)
            preds = proba.argmax(axis=1)
            best_iteration = int(getattr(clf, "best_iteration", XGB_PARAMS["n_estimators"] - 1)) + 1
            best_iterations.append(best_iteration)
            fold_metrics.append({
                "fold":     fold,
                "accuracy": round(accuracy_score(y_val, preds), 4),
                "logloss":  round(log_loss(y_val, proba), 4),
                "brier":    round(_multiclass_brier_score(y_val, proba), 4),
                "f1_macro": round(f1_score(y_val, preds, average="macro"), 4),
                "best_iteration": best_iteration,
            })
            m = fold_metrics[-1]
            log.info(
                "  Fold %d — acc=%.4f | logloss=%.4f | brier=%.4f | f1_macro=%.4f | best_iter=%d",
                m["fold"],
                m["accuracy"],
                m["logloss"],
                m["brier"],
                m["f1_macro"],
                m["best_iteration"],
            )

        df_cv = pd.DataFrame(fold_metrics)
        self.cv_metrics = {
            "folds":          fold_metrics,
            "mean_accuracy":  float(df_cv["accuracy"].mean()),
            "std_accuracy":   float(df_cv["accuracy"].std()),
            "mean_logloss":   float(df_cv["logloss"].mean()),
            "mean_brier":     float(df_cv["brier"].mean()),
            "mean_f1_macro":  float(df_cv["f1_macro"].mean()),
            "mean_best_iteration": float(np.mean(best_iterations)),
        }
        log.info("CV summary — acc=%.4f±%.4f | logloss=%.4f | brier=%.4f | f1=%.4f",
                 self.cv_metrics["mean_accuracy"], self.cv_metrics["std_accuracy"],
                 self.cv_metrics["mean_logloss"], self.cv_metrics["mean_brier"], self.cv_metrics["mean_f1_macro"])

        # Use the latest chronological block for early stopping, then refit on
        # the full dataset with the selected boosting length.
        tr_idx, val_idx = cv_splits[-1]
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        stop_params = dict(XGB_PARAMS)
        stop_params["early_stopping_rounds"] = XGB_EARLY_STOPPING_ROUNDS
        stop_model = xgb.XGBClassifier(**stop_params)
        stop_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        self.production_best_iteration = (
            int(getattr(stop_model, "best_iteration", XGB_PARAMS["n_estimators"] - 1)) + 1
        )

        final_params = dict(XGB_PARAMS)
        final_params["n_estimators"] = self.production_best_iteration
        self.final_training_config = {
            "cv_scheme": "expanding_window",
            "early_stopping_rounds": XGB_EARLY_STOPPING_ROUNDS,
            "selection_train_size": int(len(tr_idx)),
            "selection_val_size": int(len(val_idx)),
            "selection_val_start_index": int(val_idx[0]),
            "selection_val_end_index": int(val_idx[-1]),
            "selected_n_estimators": self.production_best_iteration,
            "xgb_params": final_params,
        }

        log.info(
            "Production early stopping selected best_iter=%d using latest temporal holdout (%d train / %d val)",
            self.production_best_iteration,
            len(tr_idx),
            len(val_idx),
        )
        log.info("Refitting on full dataset (%d samples) with n_estimators=%d…", len(X), self.production_best_iteration)
        self.model = xgb.XGBClassifier(**final_params)
        self.model.fit(X[FEATURE_COLS], y, verbose=False)
        
        log.info("Training Platt Scaling (Sigmoid) Calibrated Classifier...")
        self.calibrated_model = CalibratedClassifierCV(
            estimator=xgb.XGBClassifier(**final_params),
            method='sigmoid',
            cv=5
        )
        self.calibrated_model.fit(X[FEATURE_COLS], y)
        
        log.info(f"Training Local Isotonic Regression for Top-{HYBRID_CALIBRATION_RANK_THRESHOLD} teams...")
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        oof_proba = np.zeros((len(X), 3))
        X_feat = X[FEATURE_COLS]
        for tr_idx, va_idx in skf.split(X_feat, y):
            m = xgb.XGBClassifier(**final_params)
            m.fit(X_feat.iloc[tr_idx], y.iloc[tr_idx])
            oof_proba[va_idx] = m.predict_proba(X_feat.iloc[va_idx])
            
        is_elite = (X["h_rank"] <= HYBRID_CALIBRATION_RANK_THRESHOLD) & (X["a_rank"] <= HYBRID_CALIBRATION_RANK_THRESHOLD)
        oof_elite = oof_proba[is_elite]
        y_elite = y[is_elite].values
        
        self.local_isotonic_models = []
        for c in range(3):
            ir = IsotonicRegression(out_of_bounds='clip')
            ir.fit(oof_elite[:, c], (y_elite == c).astype(float))
            self.local_isotonic_models.append(ir)
            
        log.info("Hybrid Calibration models ready.")

        # Feature importance (gain-normalised)
        scores = self.model.get_booster().get_score(importance_type="gain")
        total  = sum(scores.values()) or 1.0
        self.feat_importance = {
            k: round(v / total, 6)
            for k, v in sorted(scores.items(), key=lambda x: -x[1])
        }

        # SHAP explainer
        self._explainer = shap.TreeExplainer(self.model)
        log.info("SHAP TreeExplainer ready")
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns (N, 3) probability matrix: [P(home_win), P(draw), P(away_win)]."""
        if self.calibrated_model is None:
            raise RuntimeError("Model not trained — call fit() first")
            
        global_proba = self.calibrated_model.predict_proba(X[FEATURE_COLS])
        
        if self.local_isotonic_models is not None and "h_rank" in X.columns and "a_rank" in X.columns:
            is_elite = (X["h_rank"] <= HYBRID_CALIBRATION_RANK_THRESHOLD) & (X["a_rank"] <= HYBRID_CALIBRATION_RANK_THRESHOLD)
            if is_elite.any():
                raw_proba = self.model.predict_proba(X[FEATURE_COLS])
                iso_proba = np.zeros((is_elite.sum(), 3))
                for c in range(3):
                    iso_proba[:, c] = self.local_isotonic_models[c].predict(raw_proba[is_elite, c])
                
                iso_sums = iso_proba.sum(axis=1, keepdims=True)
                iso_sums[iso_sums == 0] = 1.0
                iso_proba /= iso_sums
                
                global_proba[is_elite] = iso_proba
                
        return global_proba

    def predict_proba_single(
        self, feat_dict: Dict[str, float]
    ) -> Tuple[float, float, float]:
        row = pd.DataFrame([feat_dict])[FEATURE_COLS]
        p   = self.predict_proba(row)[0]
        return float(p[0]), float(p[1]), float(p[2])

    # ── Explainability ────────────────────────────────────────────────────────

    def explain(self, X: pd.DataFrame, class_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute SHAP values for class `class_idx`.
        Returns (shap_values, base_values) both shape (N, n_features).
        """
        if self._explainer is None:
            raise RuntimeError("SHAP explainer not initialised — model must be trained")
        sv = self._explainer(X[FEATURE_COLS])
        if sv.values.ndim == 3:
            return sv.values[:, :, class_idx], sv.base_values[:, class_idx]
        return sv.values, sv.base_values

    def explain_single(
        self,
        feat_dict: Dict[str, float],
        class_idx: int = 0,
        top_n: int = 6,
    ) -> pd.DataFrame:
        """
        Return a ranked DataFrame of SHAP feature contributions for one prediction.
        class_idx: 0=home win, 1=draw, 2=away win
        """
        X = pd.DataFrame([feat_dict])[FEATURE_COLS]
        sv, bv = self.explain(X, class_idx)
        return (
            pd.DataFrame({
                "feature":       FEATURE_COLS,
                "feature_value": X.iloc[0].values,
                "shap_value":    sv[0],
            })
            .assign(
                abs_shap=lambda d: d["shap_value"].abs(),
                direction=lambda d: d["shap_value"].map(
                    lambda x: "↑ favours home" if x > 0 else "↓ favours away"
                ),
            )
            .sort_values("abs_shap", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
            [["feature", "feature_value", "shap_value", "direction"]]
        )

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self) -> None:
        if self.model is None or self.calibrated_model is None:
            raise RuntimeError("Nothing to save — model not trained")
        self.model.save_model(_MODEL_PATH)
        joblib.dump(self.calibrated_model, _CALIBRATOR_PATH)
        if self.local_isotonic_models is not None:
            joblib.dump(self.local_isotonic_models, _HYBRID_CALIBRATOR_PATH)
        _IMPORTANCE_PATH.write_text(json.dumps(self.feat_importance, indent=2))
        _CV_METRICS_PATH.write_text(json.dumps(self.cv_metrics, indent=2))
        log.info("Model saved → %s", _MODEL_PATH)

    def load(self) -> "MatchOutcomeModel":
        if not _MODEL_PATH.exists() or not _CALIBRATOR_PATH.exists():
            raise FileNotFoundError("Missing model files — run training first")
        self.model = xgb.XGBClassifier()
        self.model.load_model(_MODEL_PATH)
        self.calibrated_model = joblib.load(_CALIBRATOR_PATH)
        if _HYBRID_CALIBRATOR_PATH.exists():
            self.local_isotonic_models = joblib.load(_HYBRID_CALIBRATOR_PATH)
        else:
            self.local_isotonic_models = None
        self._explainer = shap.TreeExplainer(self.model)
        if _IMPORTANCE_PATH.exists():
            self.feat_importance = json.loads(_IMPORTANCE_PATH.read_text())
        if _CV_METRICS_PATH.exists():
            self.cv_metrics = json.loads(_CV_METRICS_PATH.read_text())
        log.info("Model loaded from %s", _MODEL_PATH)
        return self

    def importance_report(self) -> pd.DataFrame:
        return (
            pd.DataFrame(
                list(self.feat_importance.items()),
                columns=["feature", "importance_gain"],
            )
            .sort_values("importance_gain", ascending=False)
            .reset_index(drop=True)
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Probability Matrix Builder
# ═══════════════════════════════════════════════════════════════════════════════

class ProbabilityMatrixBuilder:
    """
    Pre-computes the full N×N matchup probability matrix for all 48 WC teams.

    Strategy
    ────────
    1. Enumerate all N*(N-1) ordered (home, away) pairs.
    2. Build feature matrix in one vectorised pass (no per-pair loop in Python).
    3. Run XGBoost predict_proba once → (N*(N-1), 3) probability array.
    4. Reshape into three (N, N) matrices: P_win, P_draw, P_loss.
    5. Apply neutral-venue calibration: average the A-vs-B and B-vs-A perspectives.

    The Monte Carlo layer reads directly from these arrays via integer indexing —
    zero model calls inside the simulation loop.
    """

    def __init__(self, model: MatchOutcomeModel) -> None:
        self.model     = model
        self.teams:    List[str] = []
        self.team_idx: Dict[str, int] = {}
        self.P_win:    np.ndarray = np.empty(0)
        self.P_draw:   np.ndarray = np.empty(0)
        self.P_loss:   np.ndarray = np.empty(0)
        self.P_ko:     np.ndarray = np.empty(0)   # knockout (no-draw) P(team_i beats team_j)

    def build(
        self,
        teams: List[str],
        elo_tracker,
        form_scores: Dict[str, float],
        opp_adj_form_scores: Dict[str, float],
        form_gd: Dict[str, float],
        h2h_fn,
        form_details: Dict[str, Tuple[float, float, float, float]],
        ko_resolution: str = "coinflip",
    ) -> None:
        """
        Build the full probability matrix.

        Parameters
        ----------
        teams        : ordered list of canonical team names
        elo_tracker  : trained EloTracker
        form_scores  : {team: form_score} from FormTracker current state
        h2h_fn       : callable(home, away) → h2h win-rate differential
        ko_resolution: "proportional", "coinflip", or "hybrid"
        """
        self.teams    = teams
        self.team_idx = {t: i for i, t in enumerate(teams)}
        N = len(teams)

        log.info("Building %d×%d probability matrix (%d pair predictions)…",
                 N, N, N * (N - 1))

        # All ordered pairs
        pairs = [(teams[i], teams[j]) for i in range(N) for j in range(N) if i != j]

        # Build feature matrix (vectorised)
        feat_df = build_inference_features(
            pairs       = pairs,
            elo_tracker = elo_tracker,
            form_scores = form_scores,
            opp_adj_form_scores = opp_adj_form_scores,
            form_gd     = form_gd,
            form_details = form_details,
            h2h_fn      = h2h_fn,
            neutral     = True,
            tournament  = "FIFA World Cup",
        )
        proba = self.model.predict_proba(feat_df)   # (N*(N-1), 3)

        # Fill raw matrices
        raw_win  = np.zeros((N, N), dtype=np.float64)
        raw_draw = np.zeros((N, N), dtype=np.float64)
        raw_loss = np.zeros((N, N), dtype=np.float64)

        for idx, (h, a) in enumerate(pairs):
            i = self.team_idx[h]
            j = self.team_idx[a]
            raw_win[i, j]  = proba[idx, 0]
            raw_draw[i, j] = proba[idx, 1]
            raw_loss[i, j] = proba[idx, 2]

        # Neutral-venue calibration: symmetrically average forward/backward
        self.P_win  = np.zeros((N, N), dtype=np.float64)
        self.P_draw = np.zeros((N, N), dtype=np.float64)
        self.P_loss = np.zeros((N, N), dtype=np.float64)

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                # Average (i vs j) win prob with (j vs i) loss prob
                p_win_ij = (raw_win[i, j]  + raw_loss[j, i]) / 2.0
                p_win_ji = (raw_win[j, i]  + raw_loss[i, j]) / 2.0
                p_draw   = (raw_draw[i, j] + raw_draw[j, i]) / 2.0
                total    = p_win_ij + p_win_ji + p_draw
                self.P_win[i, j]  = p_win_ij / total
                self.P_draw[i, j] = p_draw   / total
                self.P_loss[i, j] = p_win_ji / total

        # Knockout matrix: redistribute draw prob
        self.P_ko = np.zeros((N, N), dtype=np.float64)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                pw = self.P_win[i, j]
                pl = self.P_loss[i, j]
                pd = self.P_draw[i, j]
                
                if ko_resolution == "proportional":
                    denom = pw + pl
                    self.P_ko[i, j] = pw / denom if denom > 0 else 0.5
                elif ko_resolution == "coinflip":
                    self.P_ko[i, j] = pw + (pd * 0.5)
                elif ko_resolution == "hybrid":
                    if pw > pl:
                        self.P_ko[i, j] = pw + (pd * 0.575)
                    elif pw < pl:
                        self.P_ko[i, j] = pw + (pd * 0.425)
                    else:
                        self.P_ko[i, j] = pw + (pd * 0.5)
                elif ko_resolution == "historical_hybrid":
                    # Empirical: in 34 WC knockout draws (1986-2022),
                    # the pre-match favorite only advanced 14/34 = 41.2% of the time.
                    # Underdog advanced 20/34 = 58.8%.
                    FAV_SHARE = 14 / 34   # 0.4118
                    DOG_SHARE = 20 / 34   # 0.5882
                    if pw > pl:
                        self.P_ko[i, j] = pw + (pd * FAV_SHARE)
                    elif pw < pl:
                        self.P_ko[i, j] = pw + (pd * DOG_SHARE)
                    else:
                        self.P_ko[i, j] = pw + (pd * 0.5)
                else:
                    raise ValueError(f"Unknown ko_resolution: {ko_resolution}")

        # Spot-check
        sample_pairs = [
            (teams[0], teams[1]),
            (teams[2], teams[3]),
        ]
        for h, a in sample_pairs:
            i, j = self.team_idx[h], self.team_idx[a]
            log.info(
                "  Matrix sample — %s vs %s: win=%.3f draw=%.3f loss=%.3f",
                h, a, self.P_win[i,j], self.P_draw[i,j], self.P_loss[i,j]
            )

        log.info("Probability matrix built.")

    def lookup(self, team_a: str, team_b: str) -> Tuple[float, float, float]:
        """Return (P_win_A, P_draw, P_win_B) from the matrix."""
        i = self.team_idx[team_a]
        j = self.team_idx[team_b]
        return float(self.P_win[i,j]), float(self.P_draw[i,j]), float(self.P_loss[i,j])
