"""
app.py
------
EventAnalyticsApp
    Orchestrates the Streamlit UI.  All data-processing, charting, export, ML,
    and AI logic is delegated to the dedicated classes in the other modules.

Entry-point:
    if __name__ == "__main__":
        EventAnalyticsApp().run()
"""

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
load_dotenv()  # loads your .env file

from src.config import (
    CSS, PARTICIPANT_MAX,
    INDIGO, TEAL, ROSE, AMBER, EMERALD, VIOLET, SKY, BLACK, WHITE, MGRAY,
)
from src.utils import chart_style
from src.data_pipeline import DataPipeline
from src.charts import ChartBuilder
from src.exporters import ExcelExporter
from src.ml_model import CancellationPredictor
from src.ai_query import AIQueryEngine

EXAMPLE_QUESTIONS = [
    "How many events were cancelled last year?",
    "Which room had the most events?",
    "What's the average number of participants per event?",
    "Which organisation booked the most events?",
    "Show me events with more than 200 participants",
    "What day of the week has the highest cancellation rate?",
    "How many unique organisations used the space?",
    "What's the longest event we've ever held?",
]

ML_FEATURE_LABELS = [
    "Time of day", "Month of year", "Day of week", "Event duration",
    "Room used", "Type of activity", "Number of guests",
]


class EventAnalyticsApp:
    """Main application class.  Call `.run()` to launch the Streamlit UI."""

    def run(self):
        st.set_page_config(
            page_title="Event Analytics",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded",
        )
        st.markdown(CSS, unsafe_allow_html=True)
        chart_style()

        uploaded = self._render_sidebar()
        self._render_header()

        if uploaded is None:
            self._render_landing()
            st.stop()

        df, empty_rows, duplicate_rows, negative_rows, outlier_rows = self._load(uploaded)
        kpis   = self._compute_kpis(df)
        dff    = self._render_filters(df, kpis)

        if len(dff) == 0:
            st.warning("No events match the selected filters.")
            st.stop()

        self._render_kpi_strip(kpis)
        self._render_filter_count(len(dff), kpis["total_events"])

        tabs = st.tabs([
            "📊  Charts",
            "🗂  Data",
            "🧹  Cleaning",
            "📋  Statistics",
            "💾  Export",
            "🔮  Cancellation Predictor",
            "💬  Ask your data",
        ])

        with tabs[0]: self._tab_charts(dff)
        with tabs[1]: self._tab_data(dff)
        with tabs[2]: self._tab_cleaning(df, empty_rows, duplicate_rows, negative_rows, outlier_rows, dff)
        with tabs[3]: self._tab_statistics(dff)
        with tabs[4]: self._tab_export(dff, empty_rows, duplicate_rows, negative_rows, outlier_rows)
        with tabs[5]: self._tab_ml(uploaded)
        with tabs[6]: self._tab_chat(dff)

    # ══════════════════════════════════════════════════════════════════════════
    # Sidebar & Header
    # ══════════════════════════════════════════════════════════════════════════

    def _render_sidebar(self):
        with st.sidebar:
            st.markdown("""
            <div style="background:linear-gradient(135deg,#4F46E5,#3730A3);
                        padding:28px 24px 24px;margin:-1rem -1rem 20px;border-bottom:1px solid #312E81;">
            <div style="font-family:'Inter',sans-serif;font-size:1.35rem;font-weight:700;
                        color:#FFFFFF;letter-spacing:-0.02em;line-height:1.2;">Event Analytics</div>
            </div>
            """, unsafe_allow_html=True)

            uploaded = st.file_uploader(
                "Upload data file", type=["csv", "xlsx", "xls"],
                help="data.csv or data.xlsx — columns are auto-detected",
            )

            st.markdown("<hr style='border-color:#EBEBEB;margin:20px 0'>", unsafe_allow_html=True)
            st.markdown("""
            <div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;
                        letter-spacing:0.12em;color:#818CF8;margin-bottom:10px;">Supported columns</div>
            <div style="font-size:0.78rem;color:#A5B4FC;line-height:2.1;">
            title · startTime · endTime<br>status · visibility<br>
            room / space · organization<br>participantNb · theme
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<hr style='border-color:#EBEBEB;margin:20px 0'>", unsafe_allow_html=True)
            st.caption("Powered by Streamlit · matplotlib · openpyxl · Gemini")

        return uploaded

    def _render_header(self):
        st.markdown("""
        <div style="background:#FFFFFF;padding:24px 32px 22px;margin:-4rem -4rem 0;
                    margin-bottom:28px;border-bottom:1px solid #EBEBEB;
                    display:flex;align-items:center;gap:16px;">
        <div style="width:44px;height:44px;background:#4F46E5;border-radius:10px;
                    display:flex;align-items:center;justify-content:center;flex-shrink:0;">
            <span style="font-size:1.2rem;color:white;">📊</span>
        </div>
        <div>
            <div style="font-family:'Inter',sans-serif;font-size:1.3rem;
                        font-weight:700;color:#1A1A2E;letter-spacing:-0.02em;line-height:1.2;">
            Event Analytics Pipeline
            </div>
        </div>
        <div style="margin-left:auto;">
            <span style="background:#EEF2FF;color:#4F46E5;font-size:0.68rem;font-weight:600;
                        letter-spacing:0.06em;padding:4px 10px;border-radius:6px;
                        border:1px solid #C7D2FE;">v4.3</span>
        </div>
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # Landing (no file)
    # ══════════════════════════════════════════════════════════════════════════

    def _render_landing(self):
        st.markdown("""
        <div style="background:#FFFFFF;border-radius:14px;padding:52px;text-align:center;
                    border:1.5px dashed #E2E8F0;margin-top:20px;">
        <div style="font-size:3rem;margin-bottom:12px;">📂</div>
        <div style="font-family:'Inter',sans-serif;font-size:1.3rem;font-weight:700;
                    color:#1A1A2E;letter-spacing:-0.01em;">No data loaded</div>
        <div style="color:#94A3B8;margin-top:8px;font-size:0.9rem;">
            Upload a <strong>data.csv</strong> or <strong>data.xlsx</strong> in the sidebar to begin
        </div>
        </div>
        """, unsafe_allow_html=True)

        feature_cards = [
            ("🧹", "Clean",   "Removes empty rows, dupes, bad timestamps & participant outliers", EMERALD),
            ("📊", "Analyse", "12+ charts auto-generated from your data",                         INDIGO),
            ("🔍", "Filter",  "Slice by status, type & year live",                                AMBER),
            ("💾", "Export",  "Styled Excel + removed rows report + CSV",                         VIOLET),
        ]
        for col, (icon, label, desc, color) in zip(st.columns(4), feature_cards):
            col.markdown(f"""
            <div style="background:#FFFFFF;border-radius:12px;padding:24px;text-align:center;
                        border:1px solid #F0F0F0;box-shadow:0 1px 6px rgba(0,0,0,0.04);margin-top:16px;">
            <div style="font-size:1.6rem;margin-bottom:10px;">{icon}</div>
            <div style="font-weight:700;font-size:0.85rem;color:{color};
                        letter-spacing:0.04em;margin-bottom:6px;">{label}</div>
            <div style="font-size:0.77rem;color:#94A3B8;line-height:1.5;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # Data loading & KPI helpers
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _load(uploaded):
        with st.spinner("Processing data…"):
            return DataPipeline.load_and_clean(uploaded.read(), uploaded.name)

    @staticmethod
    def _compute_kpis(df: pd.DataFrame) -> dict:
        total    = len(df)
        held     = int((df["status"] == "Tenu").sum())
        canc     = int((df["status"] == "Annulé").sum())
        return {
            "total_events":  total,
            "held":          held,
            "cancelled":     canc,
            "cancel_rate":   round(canc / total * 100, 1) if total else 0,
            "total_hours":   round(df["duration_hours"].sum(), 1),
            "total_part":    int(df["participants"].sum(skipna=True)),
            "avg_part":      round(df["participants"].mean(skipna=True), 1),
            "unique_orgs":   int(df["organization"].nunique()),
            "unique_rooms":  int(df["room"].replace("", pd.NA).dropna().nunique()),
        }

    @staticmethod
    def _render_filters(df: pd.DataFrame, kpis: dict) -> pd.DataFrame:
        st.markdown('<div class="section-label">Filters</div>', unsafe_allow_html=True)
        f1, f2, f3 = st.columns(3)
        with f1:
            sel_status = st.selectbox("Status", ["All"] + sorted(df["status"].dropna().unique().tolist()))
        with f2:
            sel_type = st.selectbox("Activity Type", ["All"] + sorted(df["activity_type"].dropna().unique().tolist()))
        with f3:
            years     = sorted(df["year"].dropna().unique().tolist())
            sel_years = st.multiselect("Year(s)", years, default=years)

        dff = df.copy()
        if sel_status != "All":  dff = dff[dff["status"] == sel_status]
        if sel_type   != "All":  dff = dff[dff["activity_type"] == sel_type]
        if sel_years:            dff = dff[dff["year"].isin(sel_years)]
        return dff

    def _render_kpi_strip(self, kpis: dict):
        st.markdown('<div class="section-label">Key Metrics</div>', unsafe_allow_html=True)
        kpi_list = [
            ("Total Events",  f"{kpis['total_events']:,}",         "#4F46E5"),
            ("Held",          f"{kpis['held']:,}",                  "#059669"),
            ("Cancelled",     f"{kpis['cancelled']:,}",             "#E11D48"),
            ("Cancel Rate",   f"{kpis['cancel_rate']}%",            "#7C3AED"),
            ("Total Hours",   f"{kpis['total_hours']:,.1f}h",       "#0284C7"),
            ("Participants",  f"{kpis['total_part']:,}",            "#0D9488"),
            ("Avg / Event",   f"{kpis['avg_part']:,.1f}",           "#D97706"),
            ("Organisations", f"{kpis['unique_orgs']:,}",           "#DB2777"),
            ("Rooms",         f"{kpis['unique_rooms']:,}",          "#1A1A2E"),
        ]
        for col, (label, value, color) in zip(st.columns(len(kpi_list)), kpi_list):
            col.markdown(f"""
            <div class="kpi-card" style="border-top:3px solid {color};">
            <div class="kpi-value">{value}</div>
            <div class="kpi-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    @staticmethod
    def _render_filter_count(filtered: int, total: int):
        st.markdown(
            f'<div style="font-size:0.76rem;color:#94A3B8;margin-bottom:4px;">'
            f'Showing <strong style="color:#1A1A2E">{filtered:,}</strong> of '
            f'<strong style="color:#1A1A2E">{total:,}</strong> events</div>',
            unsafe_allow_html=True,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Tab: Charts
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _chart_card(fig):
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        if fig:
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No data available for this chart with current filters.")
        st.markdown("</div>", unsafe_allow_html=True)

    def _tab_charts(self, dff: pd.DataFrame):
        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a: self._chart_card(ChartBuilder.status(dff))
        with col_b: self._chart_card(ChartBuilder.weekday(dff))

        self._chart_card(ChartBuilder.monthly(dff))
        self._chart_card(ChartBuilder.yearly_trend(dff))

        col_c, col_d = st.columns(2)
        with col_c: self._chart_card(ChartBuilder.activity(dff))
        with col_d: self._chart_card(ChartBuilder.hour(dff))

        col_e, col_f = st.columns(2)
        with col_e: self._chart_card(ChartBuilder.top_rooms(dff))
        with col_f: self._chart_card(ChartBuilder.top_orgs(dff))

        col_g, col_h = st.columns(2)
        with col_g: self._chart_card(ChartBuilder.avg_participants(dff))
        with col_h:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            cr_fig = ChartBuilder.cancel_rate_by_room(dff)
            if cr_fig:
                st.pyplot(cr_fig, use_container_width=True)
            else:
                st.info("No cancellation data with current filters.")
            st.markdown("</div>", unsafe_allow_html=True)

        self._chart_card(ChartBuilder.room_hours(dff))
        self._chart_card(ChartBuilder.participants_dist(dff))
        self._chart_card(ChartBuilder.heatmap(dff))

    # ══════════════════════════════════════════════════════════════════════════
    # Tab: Data
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _tab_data(dff: pd.DataFrame):
        show_cols = [c for c in [
            "event_name", "activity_type", "start_time", "end_time",
            "room", "status", "organization", "participants", "duration_hours",
        ] if c in dff.columns]
        st.dataframe(dff[show_cols], use_container_width=True, height=520)
        st.caption(f"{len(dff):,} rows · {len(show_cols)} columns shown")

    # ══════════════════════════════════════════════════════════════════════════
    # Tab: Cleaning
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _tab_cleaning(df, empty_rows, duplicate_rows, negative_rows, outlier_rows, dff):
        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

        n_orig  = len(df) + len(empty_rows) + len(duplicate_rows)
        stats   = [
            ("Original Rows",                f"{n_orig:,}",           "#4F46E5"),
            ("Empty Removed",                f"{len(empty_rows):,}",  "#E11D48"),
            ("Duplicates Removed",           f"{len(duplicate_rows):,}", "#D97706"),
            ("Neg Duration Fixed",           f"{len(negative_rows):,}", "#0D9488"),
            ("Participant Outliers Removed", f"{len(outlier_rows):,}", "#7C3AED"),
        ]
        for col, (label, value, color) in zip(st.columns(5), stats):
            col.markdown(f"""
            <div style="background:#FFFFFF;border-radius:10px;padding:18px 12px 14px;
                        border:1px solid #E2E8F0;border-top:3px solid {color};
                        box-shadow:0 1px 6px rgba(0,0,0,0.04);text-align:center;margin-bottom:16px;">
            <div style="font-family:'Inter',sans-serif;font-size:1.6rem;font-weight:700;
                        color:#1A1A2E;letter-spacing:-0.02em;line-height:1;">{value}</div>
            <div style="font-size:0.63rem;font-weight:600;text-transform:uppercase;
                        letter-spacing:0.09em;color:#94A3B8;margin-top:6px;">{label}</div>
            </div>
            """, unsafe_allow_html=True)

        if not empty_rows.empty:
            with st.expander(f"Empty rows  ·  {len(empty_rows):,}"):
                st.dataframe(empty_rows, use_container_width=True)
        if not duplicate_rows.empty:
            with st.expander(f"Duplicate rows  ·  {len(duplicate_rows):,}"):
                st.dataframe(duplicate_rows, use_container_width=True)
        if not negative_rows.empty:
            with st.expander(f"Negative-duration fixed  ·  {len(negative_rows):,}"):
                st.dataframe(negative_rows, use_container_width=True)
        if not outlier_rows.empty:
            with st.expander(f"Participant outliers removed (> {PARTICIPANT_MAX:,})  ·  {len(outlier_rows):,}"):
                st.dataframe(
                    outlier_rows[["event_name", "start_time", "room", "participants", "organization"]],
                    use_container_width=True,
                )
        multi = dff[dff["duration_flag"] == "multi-day"]
        if not multi.empty:
            with st.expander(f"Multi-day events kept  ·  {len(multi):,}"):
                st.dataframe(
                    multi[["event_name", "start_time", "end_time", "duration_hours"]],
                    use_container_width=True,
                )

    # ══════════════════════════════════════════════════════════════════════════
    # Tab: Statistics
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _tab_statistics(dff: pd.DataFrame):
        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

        total_part_f = dff["participants"].sum(skipna=True)
        total_dur_f  = dff["duration_hours"].sum()
        avg_dur_f    = dff["duration_hours"].mean()
        avg_part_f   = dff["participants"].mean(skipna=True)

        st.markdown('<div class="section-label">Aggregate Totals — Filtered View</div>', unsafe_allow_html=True)
        agg_items = [
            (f"{int(total_part_f):,}", "Total Participants",    TEAL,   ""),
            (f"{total_dur_f:,.1f}",    "Total Hours Booked",    SKY,    "h"),
            (f"{avg_part_f:.1f}",      "Avg Participants / Event", AMBER, ""),
            (f"{avg_dur_f:.2f}",       "Avg Duration / Event",  VIOLET, "h"),
        ]
        for col, (val, label, color, suffix) in zip(st.columns(4), agg_items):
            col.markdown(f"""
            <div style="background:#FFFFFF;border-radius:12px;padding:22px 16px 18px;
                        border-left:4px solid {color};border:1px solid #F0F0F0;
                        box-shadow:0 1px 6px rgba(0,0,0,0.04);text-align:center;margin-bottom:16px;">
            <div style="font-family:'Inter',sans-serif;font-size:2rem;font-weight:700;
                        color:#1A1A2E;letter-spacing:-0.02em;line-height:1;">
                {val}<span style="font-size:1rem;color:{color};margin-left:2px;">{suffix}</span>
            </div>
            <div style="font-size:0.63rem;font-weight:600;text-transform:uppercase;
                        letter-spacing:0.1em;color:#94A3B8;margin-top:8px;">{label}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-label">Descriptive Statistics</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Duration (hours)**")
            st.dataframe(dff["duration_hours"].describe().rename("duration_hours").to_frame(), use_container_width=True)
        with c2:
            st.markdown("**Participants**")
            st.dataframe(dff["participants"].describe().rename("participants").to_frame(), use_container_width=True)

        st.markdown('<div class="section-label">Events by Month × Activity Type</div>', unsafe_allow_html=True)
        pivot = dff.groupby(["month_name", "activity_type"]).size().unstack(fill_value=0)
        st.dataframe(pivot, use_container_width=True)

        st.markdown('<div class="section-label">Participants by Activity Type</div>', unsafe_allow_html=True)
        part_by_type = dff.groupby("activity_type")["participants"].agg(
            Total="sum", Average="mean", Max="max", Min="min", Count="count"
        ).round(1)
        part_by_type["Total"] = part_by_type["Total"].apply(lambda x: f"{int(x):,}")
        st.dataframe(part_by_type, use_container_width=True)

        st.markdown('<div class="section-label">Duration by Room</div>', unsafe_allow_html=True)
        dur_by_room = (
            dff[dff["room"] != ""]
            .groupby("room")["duration_hours"]
            .agg(Total_Hours="sum", Avg_Hours="mean", Events="count")
            .round(1)
            .sort_values("Total_Hours", ascending=False)
        )
        st.dataframe(dur_by_room, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # Tab: Export
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _tab_export(dff, empty_rows, duplicate_rows, negative_rows, outlier_rows):
        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
        XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

        cards = [
            (
                "📗", "Cleaned Events",
                "Styled Excel with dropdowns, table format & SUMMARY sheet",
                EMERALD, "Download cleaned_events.xlsx", "dl_clean",
                lambda: ExcelExporter.build_cleaned_excel(dff),
                "cleaned_events.xlsx", XLSX_MIME,
            ),
            (
                "🗑️", "Removed Rows",
                "Empty rows · duplicates · negative durations · participant outliers",
                AMBER, "Download removed_rows.xlsx", "dl_removed",
                lambda: ExcelExporter.build_removed_excel(empty_rows, duplicate_rows, negative_rows, outlier_rows),
                "removed_rows.xlsx", XLSX_MIME,
            ),
            (
                "📄", "Filtered CSV",
                "Current filtered view as plain CSV — ready for any tool",
                SKY, "Download filtered_events.csv", "dl_csv",
                lambda: ExcelExporter.build_filtered_csv(dff),
                "filtered_events.csv", "text/csv",
            ),
        ]
        for col, (icon, title, desc, color, btn, key, build_fn, fname, mime) in zip(st.columns(3), cards):
            with col:
                st.markdown(f"""
                <div style="background:#FFFFFF;border-radius:12px;padding:24px 20px 20px;
                            text-align:center;box-shadow:0 1px 6px rgba(0,0,0,0.04);
                            border:1px solid #F0F0F0;border-top:3px solid {color};margin-bottom:8px;">
                <div style="font-size:1.8rem;margin-bottom:10px;">{icon}</div>
                <div style="font-weight:700;font-size:0.95rem;color:#1A1A2E;margin-bottom:8px;">{title}</div>
                <div style="font-size:0.78rem;color:#94A3B8;margin-bottom:18px;line-height:1.5;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
                with st.spinner("Building…"):
                    file_data = build_fn()
                st.download_button(btn, data=file_data, file_name=fname, mime=mime,
                                   key=key, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # Tab: ML Predictor
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _tab_ml(uploaded):
        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Will my event get cancelled?</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#F0F0FF;border-radius:10px;padding:14px 18px;
                    border-left:3px solid #4F46E5;margin-bottom:20px;font-size:0.84rem;color:#374151;line-height:1.6;">
        This tool <strong>learned from your past event history</strong> to estimate whether a new event
        is likely to be cancelled. It looks for patterns — like which rooms or days tend to have more
        cancellations — and uses those to give you an early warning.
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("Analysing your past events…"):
            uploaded.seek(0)
            ml = CancellationPredictor.train(uploaded.read(), uploaded.name)

        if ml is None:
            st.warning("Not enough data to make predictions yet. You need at least 50 events with a Held or Cancelled status.")
            return

        pr_auc = ml.get("pr_auc", ml["scores"][ml["best_name"]].mean())

        for col, (val, label, color) in zip(st.columns(4), [
            ("✅ Ready",                        "Prediction status",          INDIGO),
            (f"{round(pr_auc * 100)}% PR-AUC", "Precision-recall score",     EMERALD),
            (f"{ml['n_samples']:,} events",    "Past events it learned from",TEAL),
            (f"{ml['cancel_rate']}% cancelled","Your overall cancel rate",    ROSE),
        ]):
            col.markdown(f"""
            <div style="background:#FFFFFF;border-radius:10px;padding:18px 12px 14px;
                        border:1px solid #E2E8F0;border-top:3px solid {color};
                        box-shadow:0 1px 6px rgba(0,0,0,0.04);text-align:center;margin-bottom:16px;">
            <div style="font-family:'Inter',sans-serif;font-size:1.1rem;font-weight:700;
                        color:#1A1A2E;line-height:1.2;">{val}</div>
            <div style="font-size:0.63rem;font-weight:600;text-transform:uppercase;
                        letter-spacing:0.09em;color:#94A3B8;margin-top:6px;">{label}</div>
            </div>
            """, unsafe_allow_html=True)

        ca1, ca2 = st.columns(2)
        with ca1:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.pyplot(ChartBuilder.ml_cv_scores(ml["scores"]), use_container_width=True)
            st.markdown('<div style="font-size:0.75rem;color:#94A3B8;margin-top:8px;">Three different methods were tested — the highest bar was chosen.</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with ca2:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<div style="font-size:0.78rem;color:#64748B;margin-bottom:8px;">The diagonal boxes show correct predictions.</div>', unsafe_allow_html=True)
            st.pyplot(ChartBuilder.ml_confusion(ml["cm"]), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.pyplot(ChartBuilder.ml_feature_importance(ml["importances"], ML_FEATURE_LABELS), use_container_width=True)
        st.markdown('<div style="font-size:0.75rem;color:#94A3B8;margin-top:8px;">Longer bars = stronger influence on whether an event gets cancelled.</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        rr_fig = ChartBuilder.ml_room_risk(ml["room_risk"])
        if rr_fig:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.pyplot(rr_fig, use_container_width=True)
            st.markdown('<div style="font-size:0.75rem;color:#94A3B8;margin-top:8px;">Only rooms with 5+ past events are shown.</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        if not ml["high_risk"].empty:
            st.markdown('<div class="section-label">⚠️ Combinations most likely to result in cancellation</div>', unsafe_allow_html=True)
            st.dataframe(ml["high_risk"], use_container_width=True)

        # ── Single-event form ──────────────────────────────────────────────
        st.markdown('<div class="section-label">Check a specific event</div>', unsafe_allow_html=True)

        known_rooms      = sorted([r for r in ml["room_enc"].classes_ if r and r != "Unknown"])
        known_activities = sorted([a for a in ml["act_enc"].classes_ if a and a != "Unknown"])

        if "prediction_result" not in st.session_state:
            st.session_state.prediction_result = None

        with st.form("prediction_form"):
            p1, p2, p3 = st.columns(3)
            p4, p5, p6, p7 = st.columns(4)
            with p1: pred_room     = st.selectbox("Room", known_rooms)
            with p2: pred_activity = st.selectbox("Activity type", known_activities)
            with p3: pred_weekday  = st.selectbox("Day of week",
                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
            with p4: pred_hour = st.slider("Start hour", 0, 23, 9, format="%d:00")
            with p5: pred_month = st.slider("Month", 1, 12, 6)
            with p6: pred_duration = st.slider("Duration (hours)", 0.5, 24.0, 2.0, step=0.5, format="%.1f h")
            with p7: pred_participants = st.slider("Expected guests", 0, PARTICIPANT_MAX, 50)
            submitted = st.form_submit_button("🔮 Predict cancellation risk", use_container_width=True)

        if submitted:
            risk = CancellationPredictor.predict(
                ml, pred_hour, pred_month, pred_weekday,
                pred_duration, pred_room, pred_activity, pred_participants,
            )
            st.session_state.prediction_result = risk

        if st.session_state.prediction_result is not None:
            risk = st.session_state.prediction_result
            if risk >= 60:
                rc, rl, rb, rt = ROSE, "⚠️ High risk of cancellation", "#FFF1F2", "Consider a different room or day."
            elif risk >= 35:
                rc, rl, rb, rt = AMBER, "🟡 Moderate chance of cancellation", "#FFFBEB", "A reminder closer to the date may help."
            else:
                rc, rl, rb, rt = EMERALD, "✅ Unlikely to be cancelled", "#F0FDF4", "Looking good based on past patterns."

            st.markdown(
                f"""<div style="background:{rb};border-radius:14px;padding:28px;text-align:center;
                            border:2px solid {rc};margin-top:12px;">
                    <div style="font-size:0.85rem;font-weight:700;color:{rc};margin-bottom:8px;">{rl}</div>
                    <div style="font-size:3.5rem;font-weight:800;color:{rc};letter-spacing:-0.03em;line-height:1;">{risk}%</div>
                    <div style="font-size:0.82rem;color:#64748B;margin-top:10px;">
                        estimated cancellation probability · based on {ml['n_samples']:,} past events
                    </div>
                    <div style="font-size:0.8rem;color:{rc};margin-top:8px;font-style:italic;">{rt}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    # ══════════════════════════════════════════════════════════════════════════
    # Tab: Ask your data (chat)
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _tab_chat(dff: pd.DataFrame):
        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

        if "chat_history_display"  not in st.session_state: st.session_state.chat_history_display  = []
        if "chat_history_ollama"   not in st.session_state: st.session_state.chat_history_ollama   = []
        if "chat_prefill"          not in st.session_state: st.session_state.chat_prefill           = ""

        # ── Example pills ──────────────────────────────────────────────────
        st.markdown(
            '<div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;'
            'letter-spacing:0.1em;color:#4F46E5;margin-bottom:8px;">Try asking…</div>',
            unsafe_allow_html=True,
        )
        pill_cols = st.columns(4)
        for i, q in enumerate(EXAMPLE_QUESTIONS):
            if pill_cols[i % 4].button(q, key=f"pill_{i}", use_container_width=True):
                st.session_state.chat_prefill = q
                st.rerun()

        st.markdown("<hr style='border-color:#EBEBEB;margin:16px 0'>", unsafe_allow_html=True)

        # ── Message history ────────────────────────────────────────────────
        for msg in st.session_state.chat_history_display:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-bubble-user">{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                meta = msg.get("meta", {})
                if meta.get("error"):
                    st.markdown(f'<div class="chat-error">⚠️ {meta["error"]}</div>', unsafe_allow_html=True)
                else:
                    thought_html = f'<div class="thought">💭 {meta["thought"]}</div>' if meta.get("thought") else ""
                    st.markdown(
                        f'<div class="chat-bubble-assistant">{thought_html}<div>{msg["content"]}</div></div>',
                        unsafe_allow_html=True,
                    )
                    if isinstance(meta.get("result_df"), pd.DataFrame):
                        st.dataframe(meta["result_df"], use_container_width=True)
                    if meta.get("code"):
                        with st.expander("🔍 See generated code", expanded=False):
                            st.code(meta["code"], language="python")

        # ── Input area ─────────────────────────────────────────────────────
        st.markdown('<div class="chat-input-area">', unsafe_allow_html=True)
        left_col, right_col = st.columns([5, 1])
        with left_col:
            user_input = st.text_input(
                "Ask a question about your data",
                value=st.session_state.chat_prefill,
                placeholder="e.g. Which room has the highest cancellation rate?",
                key="chat_input",
                label_visibility="collapsed",
            )
        with right_col:
            send_clicked = st.button("Ask →", key="chat_send", use_container_width=True)

        if st.session_state.chat_prefill and user_input == st.session_state.chat_prefill:
            st.session_state.chat_prefill = ""

        if st.session_state.chat_history_display:
            if st.button("🗑️ Clear conversation", key="chat_clear"):
                st.session_state.chat_history_display = []
                st.session_state.chat_history_ollama  = []
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

        # ── Process question ───────────────────────────────────────────────
        if (send_clicked or user_input) and user_input.strip():
            question = user_input.strip()
            last_user = next(
                (m for m in reversed(st.session_state.chat_history_display) if m["role"] == "user"),
                None,
            )
            if last_user and last_user["content"] == question:
                st.stop()

            st.session_state.chat_history_display.append({"role": "user", "content": question})

            with st.spinner("Thinking…"):
                result = AIQueryEngine.query(
                    question=question,
                    df=dff,
                    history=st.session_state.chat_history_ollama,
                )

            if result.get("error"):
                st.session_state.chat_history_display.append({
                    "role": "assistant", "content": "",
                    "meta": {"error": result["error"], "thought": "", "code": result.get("code", ""), "result_df": None},
                })
            else:
                st.session_state.chat_history_display.append({
                    "role": "assistant", "content": result["answer"],
                    "meta": {
                        "thought": result.get("thought", ""),
                        "code":    result.get("code", ""),
                        "result_df": result.get("result_df"),
                        "error":   None,
                    },
                })
                st.session_state.chat_history_ollama.append({"role": "user",      "content": question})
                st.session_state.chat_history_ollama.append({"role": "assistant", "content": result["answer"]})
                if len(st.session_state.chat_history_ollama) > 20:
                    st.session_state.chat_history_ollama = st.session_state.chat_history_ollama[-20:]

            st.rerun()


# ── Entry-point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    EventAnalyticsApp().run()
