import io
import re
import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from rapidfuzz import fuzz

from src.config import PARTICIPANT_MAX
from src.utils import parse_date, find_col, map_status, map_space


class DataPipeline:
    """Stateless data-cleaning pipeline; all logic lives in class/static methods."""

    # ── Public entry-point (Streamlit-cached) ──────────────────────────────────

    @staticmethod
    @st.cache_data(show_spinner=False)
    def load_and_clean(file_bytes: bytes, file_name: str):
        """
        Read *file_bytes* (CSV or Excel), clean it, and return a 5-tuple:
            (df, empty_rows, duplicate_rows, negative_rows, outlier_rows)
        """
        df_raw = DataPipeline._read_raw(file_bytes, file_name)
        col_map = DataPipeline._detect_columns(df_raw)
        records = DataPipeline._build_records(df_raw, col_map)

        df = pd.DataFrame(records)

        df, empty_rows     = DataPipeline._drop_empty(df)
        df                 = DataPipeline._clean_event_names(df)       # fix whitespace, pipes, caps
        df                 = DataPipeline._normalize_similar_names(df) # fix near-typos (fuzzy)
        df, duplicate_rows = DataPipeline._drop_duplicates(df)         # 3-tier dedup
        df                 = DataPipeline._normalise_types(df)
        df, negative_rows  = DataPipeline._fix_negative_durations(df)
        df                 = DataPipeline._add_derived_columns(df)
        df, outlier_rows   = DataPipeline._remove_participant_outliers(df)

        return df, empty_rows, duplicate_rows, negative_rows, outlier_rows

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _read_raw(file_bytes: bytes, file_name: str) -> pd.DataFrame:
        ext = Path(file_name).suffix.lower()
        return (
            pd.read_csv(io.BytesIO(file_bytes), dtype=str)
            if ext == ".csv"
            else pd.read_excel(io.BytesIO(file_bytes), dtype=str)
        )

    @staticmethod
    def _detect_columns(df_raw: pd.DataFrame) -> dict:
        """Map semantic column names → actual DataFrame column names."""
        fc = lambda *c: find_col(df_raw, *c)  # noqa: E731
        return {
            "title":        fc("title", "event_name", "name", "nom"),
            "start":        fc("startTime", "start_time", "start", "début"),
            "end":          fc("endTime",   "end_time",   "end",   "fin"),
            "status":       fc("status",    "statut",     "état"),
            "visibility":   fc("visibility", "type",      "visibilité"),
            "space":        fc("event_proposals.space.name", "room", "salle", "space"),
            "email":        fc("organizer.email",            "email", "courriel"),
            "first_name":   fc("organizer.firstName",        "firstName", "prenom", "first_name"),
            "last_name":    fc("organizer.lastname",         "lastName",  "last_name"),
            "org":          fc("organizer.organization.name","organization","organisation","org"),
            "participants": fc("participantNb", "participants", "participant_count"),
            "policy":       fc("policy", "declaration", "signed"),
            "theme":        fc("theme", "thème", "category", "catégorie"),
            "booking":      fc("booking_date", "bookingDate", "reservation_date"),
        }

    @staticmethod
    def _build_records(df_raw: pd.DataFrame, col_map: dict) -> list:
        """Convert raw rows into normalised record dicts."""

        def gv(row, key):
            col = col_map.get(key)
            return row[col] if col and col in row.index else None

        records = []
        for _, row in df_raw.iterrows():
            fn  = gv(row, "first_name") or ""
            ln  = gv(row, "last_name")  or ""
            vis = gv(row, "visibility")

            activity_type = (
                "Evénement externe" if vis and str(vis).lower() == "public"
                else ("Evénement interne" if vis else None)
            )

            policy_val = gv(row, "policy")
            signed = (
                "Oui"
                if pd.notna(policy_val) and str(policy_val).strip() not in ["", "nan", "NaN", "-"]
                else "Non"
            )

            records.append({
                "event_name":         gv(row, "title"),
                "activity_type":      activity_type,
                "start_time":         parse_date(gv(row, "start")),
                "end_time":           parse_date(gv(row, "end")),
                "room":               map_space(gv(row, "space")),
                "booking_date":       parse_date(gv(row, "booking")),
                "organizer_email":    gv(row, "email"),
                "organizer_name":     f"{fn} {ln}".strip(),
                "organization":       gv(row, "org"),
                "participants":       pd.to_numeric(gv(row, "participants"), errors="coerce"),
                "status":             map_status(gv(row, "status")),
                "signed_declaration": signed,
                "comment":            gv(row, "theme"),
            })
        return records

    # ── Cleaning steps ─────────────────────────────────────────────────────────

    @staticmethod
    def _drop_empty(df: pd.DataFrame):
        key_cols = [c for c in ["event_name", "start_time", "end_time", "status"] if c in df.columns]
        empty_mask = df[key_cols].isna().all(axis=1)
        return df[~empty_mask].reset_index(drop=True), df[empty_mask].copy()

    @staticmethod
    def _clean_event_names(df: pd.DataFrame) -> pd.DataFrame:
        """
        Fix 4 known dirty-name patterns:
          1. Extra / collapsed whitespace  →  single space, stripped
          2. Pipe-duplicated segments      →  "X | X | Y"  →  "X | Y"
          3. Inconsistent pipe spacing     →  "X| Y"       →  "X | Y"
          4. Accidental ALL-CAPS names     →  title-cased
        """
        def _fix(name):
            if pd.isna(name):
                return name

            s = str(name)

            # 1. Collapse whitespace
            s = re.sub(r'\s+', ' ', s).strip()

            # 2 & 3. Deduplicate + normalise pipe-separated segments
            if "|" in s:
                parts = [p.strip() for p in s.split("|")]
                seen = []
                for p in parts:
                    if p.lower() not in [x.lower() for x in seen]:
                        seen.append(p)
                s = " | ".join(seen)

            # 4. All-caps → title case (preserve intentional acronyms inside mixed names)
            if s == s.upper() and len(s) > 5:
                s = s.title()

            return s

        df["event_name"] = df["event_name"].apply(_fix)
        return df

    @staticmethod
    def _normalize_similar_names(df: pd.DataFrame) -> pd.DataFrame:
        """
        If two event names are ≥95% similar (fuzzy match), normalize the rarer
        one to match the more common one. This catches near-typos like:
          "Pitching and evaluation intermédiairedes candidats"
          "Pitching and evaluation intermédiaire des candidats"
        Only applies to names that share the same organizer email AND same room,
        to avoid false positives on short or generic names.
        """
        name_counts = df["event_name"].value_counts()
        names = name_counts.index.tolist()  # sorted most → least frequent

        corrections = {}
        for i, a in enumerate(names):
            if a in corrections:
                continue
            for b in names[i + 1:]:
                if b in corrections:
                    continue
                # Only compare names of similar length to avoid false positives
                if abs(len(a) - len(b)) > 10:
                    continue
                if fuzz.ratio(a.lower(), b.lower()) >= 95:
                    # Keep the more frequent name (a is already more frequent)
                    corrections[b] = a

        if corrections:
            df["event_name"] = df["event_name"].replace(corrections)
        return df

    @staticmethod
    def _drop_duplicates(df: pd.DataFrame):
        df["_name_key"] = df["event_name"].str.strip().str.lower()
        df["_date_key"] = df["start_time"].dt.date  # ← new

        all_dupe_indices = set()

        # ── Tier 1: exact duplicates ──────────────────────────────────────────
        exact_cols = ["_name_key", "start_time", "end_time", "room", "participants"]
        exact_mask = df.duplicated(subset=exact_cols, keep="first")
        all_dupe_indices.update(df[exact_mask].index.tolist())

        # ── Tier 2: same event same DAY, one Tenu / one Annulé → keep Tenu ───
        status_key = ["_name_key", "_date_key"]  # ← was ["_name_key","room","participants"]
        status_order = {"Tenu": 0, "Annulé": 1}
        df["_status_rank"] = df["status"].map(status_order).fillna(2)
        df_sorted = df.sort_values("_status_rank")
        status_mask = df_sorted.duplicated(subset=status_key, keep="first")
        all_dupe_indices.update(df_sorted[status_mask].index.tolist())

        # ── Tier 3: multi-room same event ────────────────────────────────────
        multi_key = ["_name_key", "start_time", "end_time"]
        for _, group in df.groupby(multi_key):
            if group["room"].nunique() > 1 and group["participants"].nunique() == 1:
                rooms_combined = " | ".join(group["room"].dropna().unique())
                kept_idx = group.index[0]
                df.at[kept_idx, "room"] = rooms_combined
                all_dupe_indices.update(group.index[1:].tolist())

        dupe_mask = df.index.isin(all_dupe_indices)
        duplicate_rows = df[dupe_mask].copy()
        df = (
            df[~dupe_mask]
            .drop(columns=["_name_key", "_status_rank", "_date_key"])  # ← drop _date_key too
            .reset_index(drop=True)
        )
        return df, duplicate_rows

    @staticmethod
    def _normalise_types(df: pd.DataFrame) -> pd.DataFrame:
        activity_map = {
            "événement interne":  "Evénement interne",
            "evénement interne":  "Evénement interne",
            "evenement interne":  "Evénement interne",
            "evénement externe":  "Evénement externe",
            "evenement externe":  "Evénement externe",
        }
        df["activity_type"] = df["activity_type"].apply(
            lambda x: activity_map.get(x.lower() if x else x, x) if pd.notna(x) else x
        )
        df["status"] = df["status"].str.strip().replace({"Reporté": "Annulé"})
        return df

    @staticmethod
    def _fix_negative_durations(df: pd.DataFrame):
        def fix_time(dt):
            if pd.isna(dt):
                return dt
            if isinstance(dt, datetime.datetime) and dt.hour == 0 and dt.minute < 24:
                return dt.replace(hour=dt.minute, minute=0)
            return dt

        df["start_time"] = df["start_time"].apply(fix_time)
        df["end_time"]   = df["end_time"].apply(fix_time)
        df["duration_hours"] = (
            (df["end_time"] - df["start_time"]).dt.total_seconds() / 3600
        ).round(2)

        neg_mask = df["duration_hours"] < 0
        negative_rows = df[neg_mask].copy()
        df.loc[neg_mask, ["start_time", "end_time"]] = (
            df.loc[neg_mask, ["end_time", "start_time"]].values
        )
        df.loc[neg_mask, "duration_hours"] = df.loc[neg_mask, "duration_hours"].abs()
        return df, negative_rows

    @staticmethod
    def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
        df["duration_flag"] = df["duration_hours"].apply(
            lambda x: "multi-day" if pd.notna(x) and x > 24 else ""
        )
        df["date"]       = df["start_time"].dt.date
        df["year"]       = df["start_time"].dt.year
        df["month"]      = df["start_time"].dt.month
        df["month_name"] = df["start_time"].dt.strftime("%b")
        df["weekday"]    = df["start_time"].dt.day_name()
        df["hour"]       = df["start_time"].dt.hour
        df["room"]       = df["room"].str.replace(r"^0,\s*", "", regex=True).str.strip()
        return df

    @staticmethod
    def _remove_participant_outliers(df: pd.DataFrame):
        outlier_mask = ~df["participants"].isna() & (
            (df["participants"] < 0) | (df["participants"] > PARTICIPANT_MAX)
        )
        outlier_rows = df[outlier_mask].copy()

        # Compute mean from valid (non-outlier) rows only, then round to nearest int
        valid_mask = ~df["participants"].isna() & ~outlier_mask
        mean_participants = round(df.loc[valid_mask, "participants"].mean())

        # Replace outliers with the mean instead of dropping them
        df.loc[outlier_mask, "participants"] = mean_participants

        return df.reset_index(drop=True), outlier_rows