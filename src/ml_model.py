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

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from src.config import ML_FEATURES, WEEKDAY_MAP, PARTICIPANT_MAX
from src.data_pipeline import DataPipeline


# Extended feature list — adds engineered features on top of base ML_FEATURES
ML_FEATURES_EXT = ML_FEATURES + [
    "is_weekend",
    "is_morning",
    "is_large_event",
    "booking_lead_days",
]


class CancellationPredictor:
    """Stateless ML helpers; all state is returned in a plain dict."""

    # ── Training ───────────────────────────────────────────────────────────────

    @staticmethod
    @st.cache_data(show_spinner=False)
    def train(file_bytes: bytes, file_name: str) -> dict | None:
        """
        Load *file_bytes*, engineer features, run 5-fold CV on candidate
        models including XGBoost, select the best by PR-AUC, refit on the
        full set, and return a result dict.
        Returns None when the dataset is too small.
        """
        df, *_ = DataPipeline.load_and_clean(file_bytes, file_name)
        X, y, room_enc, act_enc, d = CancellationPredictor._prepare(df)

        if len(np.unique(y)) < 2 or len(y) < 50 or y.sum() < 10:
            return None

        # Class weight ratio mirrors the imbalance in the data
        n_neg = int((y == 0).sum())
        n_pos = int((y == 1).sum())
        pos_w = round(n_neg / max(n_pos, 1), 1)
        cw    = {0: 1, 1: pos_w}

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        candidates = {
            "Random Forest": Pipeline([
                ("clf", RandomForestClassifier(
                    n_estimators=100,
                    max_depth=4,
                    min_samples_leaf=10,
                    class_weight=cw,
                    random_state=42,
                )),
            ]),
            "Gradient Boosting": Pipeline([
                ("clf", GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.05,
                    min_samples_leaf=10,
                    random_state=42,
                )),
            ]),
            "Logistic Regression": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    class_weight=cw,
                    max_iter=500,
                    random_state=42,
                )),
            ]),
        }

        # Add XGBoost if available — handles imbalance natively via scale_pos_weight
        if XGBOOST_AVAILABLE:
            candidates["XGBoost"] = Pipeline([
                ("clf", XGBClassifier(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.05,
                    scale_pos_weight=pos_w,  # handles imbalance natively
                    random_state=42,
                    eval_metric="aucpr",
                    verbosity=0,
                    use_label_encoder=False,
                )),
            ])

        # ── Cross-validation ───────────────────────────────────────────────
        scores: dict[str, np.ndarray] = {}
        for name, model in candidates.items():
            try:
                cv_scores = cross_val_score(
                    model, X, y, cv=cv, scoring="average_precision", n_jobs=-1,
                )
                scores[name] = cv_scores
            except Exception:
                scores[name] = np.zeros(5)

        # ── Dummy baseline — tells us if the model actually adds value ─────
        dummy = DummyClassifier(strategy="most_frequent")
        try:
            dummy_scores = cross_val_score(
                dummy, X, y, cv=cv, scoring="average_precision", n_jobs=-1,
            )
            dummy_score = float(dummy_scores.mean())
        except Exception:
            dummy_score = float(y.mean())  # fallback: just the positive rate

        # ── Select & refit best model ──────────────────────────────────────
        best_name  = max(scores, key=lambda k: scores[k].mean())
        best_model = candidates[best_name]
        best_model.fit(X, y)

        # ── Feature importances ────────────────────────────────────────────
        clf_step = best_model.named_steps.get("clf")
        if clf_step is not None and hasattr(clf_step, "feature_importances_"):
            importances = clf_step.feature_importances_
        elif clf_step is not None and hasattr(clf_step, "coef_"):
            importances = np.abs(clf_step.coef_[0])
        else:
            importances = np.zeros(len(ML_FEATURES_EXT))

        # ── Optimal decision threshold — clamped to [0.15, 0.5] ──────────
        probas = best_model.predict_proba(X)[:, 1]
        precision_arr, recall_arr, thresholds = precision_recall_curve(y, probas)
        f1_scores = np.where(
            (precision_arr + recall_arr) == 0,
            0,
            2 * precision_arr * recall_arr / (precision_arr + recall_arr),
        )
        best_thresh_idx = np.argmax(f1_scores[:-1])
        threshold = float(np.clip(thresholds[best_thresh_idx], 0.15, 0.5))

        y_pred = (probas >= threshold).astype(int)
        cm     = confusion_matrix(y, y_pred)

        # ── Room risk table ────────────────────────────────────────────────
        room_risk = (
            d.groupby("room")["target"]
            .agg(total="count", cancelled="sum")
            .assign(cancel_rate=lambda x: (x["cancelled"] / x["total"] * 100).round(1))
            .sort_values("cancel_rate", ascending=False)
            .reset_index()
        )

        # ── High-risk combinations (by cancel rate, not raw count) ─────────
        d["pred_proba"] = probas
        high_risk = (
            d.groupby(["room", "activity_type", "weekday"])
            .agg(
                total=("target", "count"),
                cancelled=("target", "sum"),
                avg_risk=("pred_proba", "mean"),
            )
            .reset_index()
            .assign(cancel_rate=lambda x: (x["cancelled"] / x["total"] * 100).round(1))
            .query("total >= 5")
            .sort_values("cancel_rate", ascending=False)
            .head(10)
        )

        pr_auc      = float(scores[best_name].mean())
        model_gain  = round((pr_auc - dummy_score) / max(dummy_score, 0.001) * 100, 1)

        return {
            "model":        best_model,
            "best_name":    best_name,
            "scores":       scores,
            "dummy_score":  dummy_score,
            "model_gain":   model_gain,       # % improvement over random baseline
            "importances":  importances,
            "feature_names": ML_FEATURES_EXT,
            "cm":           cm,
            "room_risk":    room_risk,
            "high_risk":    high_risk,
            "room_enc":     room_enc,
            "act_enc":      act_enc,
            "X":            X,
            "y":            y,
            "n_samples":    len(y),
            "cancel_rate":  round(float(y.mean()) * 100, 1),
            "threshold":    threshold,
            "pr_auc":       pr_auc,
            "class_weight": cw,
            "xgboost_used": XGBOOST_AVAILABLE,
        }

    # ── Single-event prediction ────────────────────────────────────────────────

    @staticmethod
    def predict(
        model_result: dict,
        hour: int, month: int, weekday: str,
        duration: float, room: str, activity: str, participants: int,
        booking_lead_days: float = 0.0,
    ) -> float:
        """Return an estimated cancellation probability (0–100 %)."""
        room_enc = model_result["room_enc"]
        act_enc  = model_result["act_enc"]

        def safe_encode(enc: LabelEncoder, val: str) -> int:
            classes = list(enc.classes_)
            return classes.index(val) if val in classes else 0

        weekday_num = WEEKDAY_MAP.get(weekday, 0)
        is_weekend  = int(weekday_num >= 5)
        is_morning  = int(hour < 12)
        is_large    = int(participants > 100)
        lead_days   = float(np.clip(booking_lead_days, 0, 365))

        row = np.array([[
            hour,
            month,
            weekday_num,
            min(duration, 72),
            safe_encode(room_enc, room),
            safe_encode(act_enc, activity),
            min(participants, PARTICIPANT_MAX),
            is_weekend,
            is_morning,
            is_large,
            lead_days,
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

        # Engineered features
        d["is_weekend"]     = (d["weekday_num"] >= 5).astype(int)
        d["is_morning"]     = (d["hour"] < 12).astype(int)
        d["is_large_event"] = (d["participants_clean"] > 100).astype(int)

        # booking_lead_days — days between booking and event date
        if "booking_date" in d.columns and "start_time" in d.columns:
            start   = pd.to_datetime(d["start_time"],   errors="coerce")
            booking = pd.to_datetime(d["booking_date"], errors="coerce")
            d["booking_lead_days"] = (
                (start - booking).dt.total_seconds() / 86400
            ).clip(0, 365).fillna(0)
        else:
            d["booking_lead_days"] = 0.0

        room_enc = LabelEncoder()
        act_enc  = LabelEncoder()
        d["room_enc"]     = room_enc.fit_transform(d["room"].fillna("Unknown"))
        d["activity_enc"] = act_enc.fit_transform(d["activity_type"].fillna("Unknown"))

        X = d[ML_FEATURES_EXT].values
        y = d["target"].values
        return X, y, room_enc, act_enc, d