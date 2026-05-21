"""
ml_model.py
-----------
CancellationPredictor
    Trains a cancellation-risk model on historical event data and exposes a
    single-event prediction method.

Public API
----------
    CancellationPredictor.train(file_bytes, file_name)   → dict | None
        Cached via @st.cache_data.  Returns a result dict or None when there
        is insufficient data.

    CancellationPredictor.predict(model_result, ...)     → float (0–100 %)
"""

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline

from src.config import ML_FEATURES, WEEKDAY_MAP, PARTICIPANT_MAX
from src.data_pipeline import DataPipeline


class CancellationPredictor:
    """Stateless ML helpers; all state is returned in a plain dict."""

    # ── Training ───────────────────────────────────────────────────────────────

    @staticmethod
    @st.cache_data(show_spinner=False)
    def train(file_bytes: bytes, file_name: str) -> dict | None:
        """
        Load *file_bytes*, engineer features, run 5-fold CV on three candidate
        models, select the best by PR-AUC, refit on the full set, and return a
        result dict.  Returns None when the dataset is too small.
        """
        df, *_ = DataPipeline.load_and_clean(file_bytes, file_name)
        X, y, room_enc, act_enc, d = CancellationPredictor._prepare(df)

        if len(np.unique(y)) < 2 or len(y) < 50 or y.sum() < 10:
            return None

        cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        smote = SMOTE(random_state=42, k_neighbors=min(5, int(y.sum()) - 1))

        candidates = {
            "Random Forest": ImbPipeline([
                ("smote", smote),
                ("clf", RandomForestClassifier(
                    n_estimators=200, max_depth=8,
                    class_weight="balanced", random_state=42,
                )),
            ]),
            "Gradient Boosting": ImbPipeline([
                ("smote", smote),
                ("clf", GradientBoostingClassifier(
                    n_estimators=150, max_depth=4,
                    learning_rate=0.08, random_state=42,
                )),
            ]),
            "Logistic Regression": ImbPipeline([
                ("smote", smote),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    class_weight="balanced", max_iter=500, random_state=42,
                )),
            ]),
        }

        scores: dict[str, np.ndarray] = {}
        for name, model in candidates.items():
            try:
                cv_scores = cross_val_score(
                    model, X, y, cv=cv, scoring="average_precision", n_jobs=-1,
                )
                scores[name] = cv_scores
            except Exception:
                scores[name] = np.zeros(5)

        best_name  = max(scores, key=lambda k: scores[k].mean())
        best_model = candidates[best_name]
        best_model.fit(X, y)

        # Feature importances
        clf_step = best_model.named_steps.get("clf")
        if clf_step is not None and hasattr(clf_step, "feature_importances_"):
            importances = clf_step.feature_importances_
        elif clf_step is not None and hasattr(clf_step, "coef_"):
            importances = clf_step.coef_[0]
        else:
            importances = np.zeros(len(ML_FEATURES))

        # Optimal decision threshold (maximise F1, clamped to [0.2, 0.6])
        probas = best_model.predict_proba(X)[:, 1]
        precision_arr, recall_arr, thresholds = precision_recall_curve(y, probas)
        f1_scores = np.where(
            (precision_arr + recall_arr) == 0,
            0,
            2 * precision_arr * recall_arr / (precision_arr + recall_arr),
        )
        best_thresh_idx = np.argmax(f1_scores[:-1])
        threshold = float(
            np.clip(thresholds[best_thresh_idx], 0.2, 0.6)
        )

        y_pred = (probas >= threshold).astype(int)
        cm     = confusion_matrix(y, y_pred)

        room_risk = (
            d.groupby("room")["target"]
            .agg(total="count", cancelled="sum")
            .assign(cancel_rate=lambda x: (x["cancelled"] / x["total"] * 100).round(1))
            .sort_values("cancel_rate", ascending=False)
            .reset_index()
        )

        d["pred_proba"] = probas
        high_risk = (
            d[d["pred_proba"] >= threshold]
            .groupby(["room", "activity_type", "weekday"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(10)
        )

        return {
            "model":       best_model,
            "best_name":   best_name,
            "scores":      scores,
            "importances": importances,
            "cm":          cm,
            "room_risk":   room_risk,
            "high_risk":   high_risk,
            "room_enc":    room_enc,
            "act_enc":     act_enc,
            "X":           X,
            "y":           y,
            "n_samples":   len(y),
            "cancel_rate": round(float(y.mean()) * 100, 1),
            "threshold":   threshold,
            "pr_auc":      scores[best_name].mean(),
        }

    # ── Single-event prediction ────────────────────────────────────────────────

    @staticmethod
    def predict(
        model_result: dict,
        hour: int, month: int, weekday: str,
        duration: float, room: str, activity: str, participants: int,
    ) -> float:
        """Return an estimated cancellation probability (0–100 %)."""
        room_enc = model_result["room_enc"]
        act_enc  = model_result["act_enc"]

        def safe_encode(enc: LabelEncoder, val: str) -> int:
            classes = list(enc.classes_)
            return classes.index(val) if val in classes else 0

        row = np.array([[
            hour,
            month,
            WEEKDAY_MAP.get(weekday, 0),
            min(duration, 72),
            safe_encode(room_enc, room),
            safe_encode(act_enc, activity),
            min(participants, PARTICIPANT_MAX),
        ]])
        proba = model_result["model"].predict_proba(row)[0][1]
        return round(proba * 100, 1)

    # ── Feature engineering ────────────────────────────────────────────────────

    @staticmethod
    def _prepare(df: pd.DataFrame):
        d = df.copy()
        d = d[d["status"].isin(["Tenu", "Annulé"])].copy()
        d["target"]             = (d["status"] == "Annulé").astype(int)
        d["weekday_num"]        = d["weekday"].map(WEEKDAY_MAP).fillna(0).astype(int)
        d["participants_clean"] = d["participants"].fillna(0).clip(0, PARTICIPANT_MAX)
        d["duration_hours"]     = d["duration_hours"].fillna(0).clip(0, 72)
        d["hour"]               = d["hour"].fillna(0).astype(int)
        d["month"]              = d["month"].fillna(1).astype(int)

        room_enc = LabelEncoder()
        act_enc  = LabelEncoder()
        d["room_enc"]     = room_enc.fit_transform(d["room"].fillna("Unknown"))
        d["activity_enc"] = act_enc.fit_transform(d["activity_type"].fillna("Unknown"))

        X = d[ML_FEATURES].values
        y = d["target"].values
        return X, y, room_enc, act_enc, d
