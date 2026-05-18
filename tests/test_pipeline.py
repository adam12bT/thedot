"""
tests/test_pipeline.py
Covers the core data-pipeline logic — no Streamlit runtime needed.
Run: pytest tests/ -v
"""

import datetime
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ── Import pure functions from app (no st.* calls at import time) ────────────
from app import (
    parse_date,
    map_status,
    map_space,
    remove_outliers,
    PARTICIPANT_MAX,
)


# ─────────────────────────────────────────────────────────────────────────────
# parse_date
# ─────────────────────────────────────────────────────────────────────────────

class TestParseDate:
    def test_iso_format(self):
        result = parse_date("2024-03-15T09:30:00")
        assert result == pd.Timestamp("2024-03-15 09:30:00")

    def test_js_date_string(self):
        result = parse_date("Fri Mar 15 2024 09:30:00")
        assert isinstance(result, pd.Timestamp)

    def test_nan_returns_none(self):
        assert parse_date(np.nan) is None

    def test_empty_string_returns_none(self):
        assert parse_date("") is None

    def test_plain_date(self):
        result = parse_date("2024-06-01")
        assert result is not None
        assert result.year == 2024


# ─────────────────────────────────────────────────────────────────────────────
# map_status
# ─────────────────────────────────────────────────────────────────────────────

class TestMapStatus:
    def test_finished_maps_to_tenu(self):
        assert map_status("finished") == "Tenu"

    def test_canceled_maps_to_annule(self):
        assert map_status("canceled") == "Annulé"

    def test_rejected_maps_to_annule(self):
        assert map_status("rejected") == "Annulé"

    def test_accepted_maps_to_tenu(self):
        assert map_status("accepted") == "Tenu"

    def test_case_insensitive(self):
        assert map_status("FINISHED") == "Tenu"
        assert map_status("Canceled") == "Annulé"

    def test_already_tenu(self):
        assert map_status("tenu") == "Tenu"

    def test_nan_returns_nan(self):
        result = map_status(np.nan)
        assert pd.isna(result)

    def test_unknown_value_passthrough(self):
        assert map_status("pending") == "pending"


# ─────────────────────────────────────────────────────────────────────────────
# map_space
# ─────────────────────────────────────────────────────────────────────────────

class TestMapSpace:
    def test_salle_polyvalente(self):
        assert map_space("Salle Polyvalente") == "Salle de formation"

    def test_training_room(self):
        assert map_space("Training Room") == "Salle de formation"

    def test_empty_string_returns_empty(self):
        assert map_space("") == ""

    def test_nan_returns_empty(self):
        assert map_space(np.nan) == ""

    def test_unknown_passthrough(self):
        assert map_space("Some New Room") == "Some New Room"


# ─────────────────────────────────────────────────────────────────────────────
# remove_outliers  (threshold > 1000)
# ─────────────────────────────────────────────────────────────────────────────

class TestRemoveOutliers:
    def _series(self, values):
        return pd.Series(values, dtype=float)

    def test_removes_above_1000(self):
        s = self._series([10, 50, 200, 1001, 5000])
        result = remove_outliers(s)
        assert 1001 not in result.values
        assert 5000 not in result.values

    def test_keeps_exactly_1000(self):
        s = self._series([100, 500, 1000])
        result = remove_outliers(s)
        assert 1000 in result.values

    def test_keeps_zero(self):
        s = self._series([0, 10, 50])
        result = remove_outliers(s)
        assert 0 in result.values

    def test_empty_series(self):
        result = remove_outliers(self._series([]))
        assert len(result) == 0

    def test_all_outliers(self):
        s = self._series([1500, 2000, 9999])
        result = remove_outliers(s)
        assert len(result) == 0

    def test_participant_max_constant(self):
        assert PARTICIPANT_MAX == 1000


# ─────────────────────────────────────────────────────────────────────────────
# Integration — build a minimal DataFrame and run the full cleaning logic
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration:
    def _make_df(self):
        return pd.DataFrame({
            "event_name":    ["Event A", "Event B", "Event C", "Event D"],
            "start_time":    pd.to_datetime(["2024-01-10 09:00", "2024-02-15 14:00",
                                             "2024-03-20 10:00", "2024-04-05 11:00"]),
            "end_time":      pd.to_datetime(["2024-01-10 11:00", "2024-02-15 16:00",
                                             "2024-03-20 12:00", "2024-04-05 13:00"]),
            "status":        ["Tenu", "Annulé", "Tenu", "Tenu"],
            "participants":  [50.0, 120.0, 2000.0, 80.0],
            "room":          ["Salle de formation", "Terrasse", "BALE", "Podcast"],
            "organization":  ["Org A", "Org B", "Org C", "Org A"],
            "activity_type": ["Evénement interne"] * 4,
            "duration_hours":[2.0, 2.0, 2.0, 2.0],
        })

    def test_outlier_flag_applied(self):
        df = self._make_df()
        df["participant_outlier"] = ~df["participants"].isna() & (df["participants"] > PARTICIPANT_MAX)
        assert df.loc[df["event_name"] == "Event C", "participant_outlier"].iloc[0] == True
        assert df.loc[df["event_name"] == "Event A", "participant_outlier"].iloc[0] == False

    def test_avg_excludes_outliers(self):
        df = self._make_df()
        df["participant_outlier"] = ~df["participants"].isna() & (df["participants"] > PARTICIPANT_MAX)
        clean = df.loc[~df["participant_outlier"], "participants"]
        avg = clean.mean()
        # 50 + 120 + 80 / 3 = 83.33 — 2000 must NOT influence this
        assert avg < 200
        assert round(avg, 2) == round((50 + 120 + 80) / 3, 2)

    def test_held_cancelled_counts(self):
        df = self._make_df()
        assert (df["status"] == "Tenu").sum() == 3
        assert (df["status"] == "Annulé").sum() == 1

    def test_duration_calculation(self):
        df = self._make_df()
        df["duration_hours"] = (
            (df["end_time"] - df["start_time"]).dt.total_seconds() / 3600
        ).round(2)
        assert all(df["duration_hours"] == 2.0)
