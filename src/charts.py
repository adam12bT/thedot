"""
charts.py
---------
ChartBuilder
    Each public method receives a filtered DataFrame and returns a matplotlib
    Figure (or None when there is not enough data).

    All rendering details (colours, axes styles) are isolated here so the
    application layer only needs to call e.g.::

        fig = ChartBuilder.status(dff)
        st.pyplot(fig)
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

from src.config import (
    CHART_COLORS, INDIGO, TEAL, ROSE, AMBER, EMERALD,
    VIOLET, SKY, BLACK, WHITE, MGRAY,
)
from src.utils import make_fig


class ChartBuilder:
    """Static factory methods, one per chart type."""

    # ── Overview ───────────────────────────────────────────────────────────────

    @staticmethod
    def status(df: pd.DataFrame) -> plt.Figure:
        counts = df["status"].value_counts()
        fig, ax = make_fig(7, 4)
        color_map = {"Tenu": EMERALD, "Annulé": ROSE}
        colors = [color_map.get(s, MGRAY) for s in counts.index]
        bars = ax.bar(
            counts.index, counts.values,
            color=colors, edgecolor=WHITE, linewidth=2, width=0.42, alpha=0.92,
        )
        for b in bars:
            ax.text(
                b.get_x() + b.get_width() / 2, b.get_height() + 0.4,
                f"{int(b.get_height()):,}", ha="center",
                fontweight="600", fontsize=12, color=BLACK,
            )
        ax.set_title("Events by Status")
        ax.set_ylabel("Count", color=MGRAY, fontsize=10)
        ax.tick_params(axis="x", rotation=0, labelsize=11)
        plt.tight_layout()
        return fig

    @staticmethod
    def activity(df: pd.DataFrame) -> plt.Figure:
        counts = df["activity_type"].value_counts().dropna()
        fig, ax = make_fig(9, max(3.5, len(counts) * 0.65 + 1.5))
        colors = CHART_COLORS[: len(counts)]
        ax.barh(counts.index, counts.values, color=colors, edgecolor=WHITE, height=0.52, alpha=0.92)
        for p in ax.patches:
            ax.text(
                p.get_width() + 0.3, p.get_y() + p.get_height() / 2,
                f"{int(p.get_width()):,}", va="center", fontsize=10, color=MGRAY, fontweight="500",
            )
        ax.set_title("Events by Activity Type")
        ax.set_xlabel("Count", color=MGRAY, fontsize=10)
        ax.invert_yaxis()
        plt.tight_layout()
        return fig

    @staticmethod
    def monthly(df: pd.DataFrame) -> plt.Figure:
        monthly = df.groupby(["year", "month"]).size().reset_index(name="count")
        monthly["period"] = pd.to_datetime(monthly[["year", "month"]].assign(day=1))
        monthly = monthly.sort_values("period")
        fig, ax = make_fig(max(9, len(monthly) * 0.8 + 2), 4.5)
        ax.fill_between(monthly["period"], monthly["count"], alpha=0.08, color=INDIGO)
        ax.plot(
            monthly["period"], monthly["count"],
            marker="o", color=INDIGO, linewidth=2.5, markersize=6,
            zorder=3, markerfacecolor=WHITE, markeredgewidth=2.5,
        )
        for x, y in zip(monthly["period"], monthly["count"]):
            ax.annotate(
                str(y), (x, y), textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=8, color=MGRAY, fontweight="500",
            )
        ax.set_title("Events per Month")
        ax.set_ylabel("Events", color=MGRAY, fontsize=10)
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %Y"))
        plt.xticks(rotation=40, ha="right", fontsize=9)
        plt.tight_layout()
        return fig

    @staticmethod
    def weekday(df: pd.DataFrame) -> plt.Figure:
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        counts = df["weekday"].value_counts().reindex(order, fill_value=0)
        fig, ax = make_fig(9, 4.5)
        max_v = counts.max()
        colors = [INDIGO if v == max_v and max_v > 0 else "#CBD5E1" for v in counts.values]
        bars = ax.bar(counts.index, counts.values, color=colors, edgecolor=WHITE, width=0.6, alpha=0.92)
        for b in bars:
            ax.text(
                b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
                f"{int(b.get_height()):,}", ha="center", fontsize=9, color=MGRAY, fontweight="500",
            )
        ax.set_title("Events by Day of Week")
        ax.set_ylabel("Count", color=MGRAY, fontsize=10)
        ax.tick_params(axis="x", rotation=15)
        plt.tight_layout()
        return fig

    @staticmethod
    def hour(df: pd.DataFrame) -> plt.Figure:
        hc = df["hour"].value_counts().sort_index()
        all_h = pd.Series(0, index=range(24))
        all_h.update(hc)
        fig, ax = make_fig(11, 4.2)
        slot_colors = {
            range(0, 6):  "#7C3AED",
            range(6, 9):  "#D97706",
            range(9, 12): "#0284C7",
            range(12, 14):"#D97706",
            range(14, 18):"#059669",
            range(18, 21):"#E11D48",
            range(21, 24):"#0D9488",
        }
        hour_palette = []
        for h in range(24):
            for rng, c in slot_colors.items():
                if h in rng:
                    hour_palette.append(c)
                    break
        max_v = all_h.max()
        colors = [
            INDIGO if v == max_v and max_v > 0 else hour_palette[i]
            for i, v in enumerate(all_h.values)
        ]
        ax.bar(all_h.index, all_h.values, color=colors, edgecolor=WHITE, width=0.75, alpha=0.88)
        ax.set_title("Events by Start Hour")
        ax.set_xlabel("Hour of Day", color=MGRAY, fontsize=10)
        ax.set_ylabel("Count", color=MGRAY, fontsize=10)
        ax.set_xticks(range(24))
        legend_items = [
            Patch(color="#7C3AED", label="Night (0–5)"),
            Patch(color="#D97706", label="Early AM (6–8)"),
            Patch(color="#0284C7", label="Morning (9–11)"),
            Patch(color="#D97706", label="Noon (12–13)"),
            Patch(color="#059669", label="Afternoon (14–17)"),
            Patch(color="#E11D48", label="Evening (18–20)"),
            Patch(color="#0D9488", label="Late (21–23)"),
        ]
        ax.legend(handles=legend_items, loc="upper right", fontsize=7,
                  framealpha=0.95, edgecolor="#EBEBEB", ncol=2)
        plt.tight_layout()
        return fig

    # ── Rooms ──────────────────────────────────────────────────────────────────

    @staticmethod
    def top_rooms(df: pd.DataFrame) -> plt.Figure:
        top = df["room"].replace("", pd.NA).dropna().value_counts().head(10)
        fig, ax = make_fig(9, max(3.5, len(top) * 0.6 + 1))
        colors = CHART_COLORS[: len(top)]
        ax.barh(top.index, top.values, color=colors, edgecolor=WHITE, height=0.58, alpha=0.92)
        for p in ax.patches:
            ax.text(
                p.get_width() + 0.2, p.get_y() + p.get_height() / 2,
                f"{int(p.get_width()):,}", va="center", fontsize=9, color=MGRAY, fontweight="500",
            )
        ax.set_title("Top 10 Most Used Rooms")
        ax.set_xlabel("Events", color=MGRAY, fontsize=10)
        ax.invert_yaxis()
        plt.tight_layout()
        return fig

    @staticmethod
    def room_hours(df: pd.DataFrame) -> plt.Figure:
        rd = (
            df.groupby("room")["duration_hours"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )
        rd = rd[rd.index != ""]
        fig, ax = make_fig(9, max(3.5, len(rd) * 0.65 + 1))
        colors = CHART_COLORS[: len(rd)]
        ax.barh(rd.index, rd.values, color=colors, edgecolor=WHITE, height=0.58, alpha=0.92)
        for p in ax.patches:
            ax.text(
                p.get_width() + 0.5, p.get_y() + p.get_height() / 2,
                f"{p.get_width():.0f}h", va="center", fontsize=9, color=MGRAY, fontweight="500",
            )
        ax.set_title("Rooms by Total Reservation Hours")
        ax.set_xlabel("Total Hours", color=MGRAY, fontsize=10)
        ax.invert_yaxis()
        plt.tight_layout()
        return fig

    @staticmethod
    def cancel_rate_by_room(df: pd.DataFrame):
        """Returns None when there is no cancellation data."""
        rs = df.groupby(["room", "status"]).size().unstack(fill_value=0)
        if "Annulé" not in rs.columns:
            return None
        rs["rate"] = (rs["Annulé"] / rs.sum(axis=1) * 100).round(1)
        top = rs["rate"].sort_values(ascending=False).head(10)
        top = top[top > 0]
        if top.empty:
            return None
        fig, ax = make_fig(9, max(3.5, len(top) * 0.65 + 1))
        max_r = top.max()
        colors = [ROSE if v == max_r else "#FDA4AF" for v in top.values]
        ax.barh(top.index, top.values, color=colors, edgecolor=WHITE, height=0.58, alpha=0.92)
        for p in ax.patches:
            ax.text(
                p.get_width() + 0.3, p.get_y() + p.get_height() / 2,
                f"{p.get_width():.1f}%", va="center", fontsize=9, color=MGRAY, fontweight="500",
            )
        ax.xaxis.set_major_formatter(mticker.PercentFormatter())
        ax.set_title("Cancellation Rate by Room")
        ax.set_xlabel("Cancellation Rate", color=MGRAY, fontsize=10)
        ax.invert_yaxis()
        plt.tight_layout()
        return fig

    # ── Organisations / Participants ───────────────────────────────────────────

    @staticmethod
    def top_orgs(df: pd.DataFrame) -> plt.Figure:
        top = df["organization"].replace("", pd.NA).dropna().value_counts().head(15)
        fig, ax = make_fig(9, max(4, len(top) * 0.5 + 1))
        colors = [INDIGO] + CHART_COLORS[1 : len(top)]
        ax.barh(top.index, top.values, color=colors[: len(top)], edgecolor=WHITE, height=0.58, alpha=0.92)
        for p in ax.patches:
            ax.text(
                p.get_width() + 0.1, p.get_y() + p.get_height() / 2,
                f"{int(p.get_width()):,}", va="center", fontsize=8, color=MGRAY, fontweight="500",
            )
        ax.set_title("Top 15 Organizations")
        ax.set_xlabel("Events", color=MGRAY, fontsize=10)
        ax.invert_yaxis()
        plt.tight_layout()
        return fig

    @staticmethod
    def avg_participants(df: pd.DataFrame) -> plt.Figure:
        avg = (
            df.groupby("activity_type")["participants"]
            .mean()
            .dropna()
            .sort_values(ascending=False)
        )
        fig, ax = make_fig(9, max(3.5, len(avg) * 0.65 + 1))
        colors = CHART_COLORS[: len(avg)]
        ax.barh(avg.index, avg.values, color=colors, edgecolor=WHITE, height=0.52, alpha=0.92)
        for p in ax.patches:
            ax.text(
                p.get_width() + 0.5, p.get_y() + p.get_height() / 2,
                f"{p.get_width():.1f}", va="center", fontsize=9, color=MGRAY, fontweight="500",
            )
        ax.set_title("Avg Participants by Activity Type")
        ax.set_xlabel("Avg Participants", color=MGRAY, fontsize=10)
        ax.invert_yaxis()
        plt.tight_layout()
        return fig

    @staticmethod
    def participants_dist(df: pd.DataFrame) -> plt.Figure:
        clean = df["participants"].dropna()
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), facecolor=WHITE)
        for ax in axes:
            ax.set_facecolor(WHITE)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.spines["bottom"].set_color("#EBEBEB")
            ax.tick_params(colors=MGRAY, length=0)
            ax.yaxis.grid(True, color="#F4F4F8", linewidth=0.7)
            ax.set_axisbelow(True)

        axes[0].boxplot(
            clean, vert=True, patch_artist=True,
            boxprops=dict(facecolor="#EEF2FF", alpha=0.9, linewidth=1.5),
            medianprops=dict(color=INDIGO, linewidth=2.5),
            whiskerprops=dict(color=TEAL, linewidth=1.5),
            capprops=dict(color=TEAL, linewidth=2),
            flierprops=dict(marker="o", color=ROSE, alpha=0.5, markersize=4),
        )
        axes[0].set_title("Participant Distribution", fontsize=11)
        axes[0].set_ylabel("Participants", color=MGRAY, fontsize=10)
        axes[0].set_xticks([])

        axes[1].hist(clean, bins=28, color=INDIGO, edgecolor=WHITE, alpha=0.82)
        axes[1].set_title("Histogram of Participants", fontsize=11)
        axes[1].set_xlabel("Participants", color=MGRAY, fontsize=10)
        axes[1].set_ylabel("Events", color=MGRAY, fontsize=10)
        plt.tight_layout()
        return fig

    # ── Time ──────────────────────────────────────────────────────────────────

    @staticmethod
    def heatmap(df: pd.DataFrame) -> plt.Figure:
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        heat = (
            df.groupby(["weekday", "hour"])
            .size()
            .unstack(fill_value=0)
            .reindex(order, fill_value=0)
        )
        fig, ax = plt.subplots(figsize=(14, 4.5), facecolor=WHITE)
        ax.set_facecolor(WHITE)
        cmap = LinearSegmentedColormap.from_list(
            "indigo", [WHITE, "#EEF2FF", "#A5B4FC", INDIGO, "#312E81"]
        )
        sns.heatmap(
            heat, ax=ax, cmap=cmap, linewidths=0.4, linecolor="#F8FAFC",
            cbar_kws={"label": "Events", "shrink": 0.8},
        )
        ax.set_title("Event Density — Weekday × Start Hour", pad=14)
        ax.set_xlabel("Hour of Day", color=MGRAY, fontsize=10)
        ax.set_ylabel("", color=MGRAY)
        plt.tight_layout()
        return fig

    @staticmethod
    def yearly_trend(df: pd.DataFrame):
        """Returns None when there is only a single year in the data."""
        yearly = df.groupby("year").agg(
            events=("event_name", "count"),
            participants=("participants", "sum"),
            hours=("duration_hours", "sum"),
        ).reset_index()
        if len(yearly) < 2:
            return None
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor=WHITE)
        yr_colors = CHART_COLORS[: len(yearly)]
        for ax in axes:
            ax.set_facecolor(WHITE)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.spines["bottom"].set_color("#EBEBEB")
            ax.tick_params(colors=MGRAY, length=0)
            ax.yaxis.grid(True, color="#F4F4F8", linewidth=0.7)
            ax.set_axisbelow(True)

        axes[0].bar(
            yearly["year"].astype(str), yearly["events"],
            color=yr_colors, edgecolor=WHITE, width=0.5, alpha=0.92,
        )
        axes[0].set_title("Events per Year")
        for b in axes[0].patches:
            axes[0].text(
                b.get_x() + b.get_width() / 2, b.get_height() + 1,
                f"{int(b.get_height()):,}", ha="center", fontsize=9, color=MGRAY, fontweight="500",
            )

        axes[1].bar(
            yearly["year"].astype(str), yearly["participants"].fillna(0),
            color=yr_colors, edgecolor=WHITE, width=0.5, alpha=0.92,
        )
        axes[1].set_title("Total Participants per Year")
        for b in axes[1].patches:
            axes[1].text(
                b.get_x() + b.get_width() / 2, b.get_height() + 1,
                f"{int(b.get_height()):,}", ha="center", fontsize=9, color=MGRAY, fontweight="500",
            )

        axes[2].bar(
            yearly["year"].astype(str), yearly["hours"].fillna(0),
            color=yr_colors, edgecolor=WHITE, width=0.5, alpha=0.92,
        )
        axes[2].set_title("Total Hours per Year")
        for b in axes[2].patches:
            axes[2].text(
                b.get_x() + b.get_width() / 2, b.get_height() + 1,
                f"{b.get_height():.0f}h", ha="center", fontsize=9, color=MGRAY, fontweight="500",
            )
        plt.tight_layout()
        return fig

    # ── ML charts ─────────────────────────────────────────────────────────────

    @staticmethod
    def ml_feature_importance(importances, feature_names: list) -> plt.Figure:
        fig, ax = make_fig(8, 4)
        idx = np.argsort(np.abs(importances))
        colors_bar = [ROSE if v > 0 else INDIGO for v in importances[idx]]
        ax.barh(
            [feature_names[i] for i in idx], importances[idx],
            color=colors_bar, edgecolor=WHITE, height=0.55, alpha=0.9,
        )
        ax.set_title("What influences cancellations the most?")
        ax.set_xlabel(
            "Influence level (higher = more impact on the prediction)",
            color=MGRAY, fontsize=10,
        )
        ax.axvline(0, color=MGRAY, linewidth=0.8, linestyle="--")
        plt.tight_layout()
        return fig

    @staticmethod
    def ml_cv_scores(scores: dict) -> plt.Figure:
        fig, ax = make_fig(8, 4)
        names  = list(scores.keys())
        means  = [scores[n].mean() for n in names]
        stds   = [scores[n].std()  for n in names]
        colors = [INDIGO, TEAL, AMBER]
        bars = ax.bar(
            names, means, yerr=stds, color=colors[: len(names)],
            edgecolor=WHITE, width=0.45, alpha=0.9,
            capsize=6, error_kw=dict(color=MGRAY, linewidth=1.5),
        )
        for b, m in zip(bars, means):
            ax.text(
                b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                f"{round(m * 100)}%", ha="center", fontsize=10, fontweight="600", color=BLACK,
            )
        ax.set_ylim(0, 1)
        ax.set_title("Which prediction method works best on your data?")
        ax.set_ylabel("Accuracy", color=MGRAY, fontsize=10)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        plt.tight_layout()
        return fig

    @staticmethod
    def ml_confusion(cm) -> plt.Figure:
        from matplotlib.colors import LinearSegmentedColormap as LSC
        fig, ax = plt.subplots(figsize=(5, 4), facecolor=WHITE)
        ax.set_facecolor(WHITE)
        labels = [["TN", "FP"], ["FN", "TP"]]
        cmap   = LSC.from_list("cm_cmap", [WHITE, "#C7D2FE", INDIGO])
        ax.imshow(cm, cmap=cmap, aspect="auto")
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, f"{labels[i][j]}\n{cm[i, j]:,}",
                    ha="center", va="center", fontsize=12,
                    fontweight="700",
                    color=BLACK if cm[i, j] < cm.max() * 0.6 else WHITE,
                )
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Predicted: Held", "Predicted: Cancelled"], fontsize=9, color=MGRAY)
        ax.set_yticklabels(["Actually: Held", "Actually: Cancelled"], fontsize=9, color=MGRAY)
        ax.set_title("How often was it right?", pad=12)
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
        plt.tight_layout()
        return fig

    @staticmethod
    def ml_room_risk(room_risk: pd.DataFrame):
        """Returns None when no room has ≥5 events."""
        top = room_risk[room_risk["total"] >= 5].head(12)
        if top.empty:
            return None
        fig, ax = make_fig(9, max(3.5, len(top) * 0.6 + 1))
        max_r  = top["cancel_rate"].max()
        colors = [ROSE if v == max_r else "#FDA4AF" for v in top["cancel_rate"].values]
        ax.barh(top["room"], top["cancel_rate"], color=colors, edgecolor=WHITE, height=0.58, alpha=0.92)
        for p in ax.patches:
            ax.text(
                p.get_width() + 0.5, p.get_y() + p.get_height() / 2,
                f"{p.get_width():.1f}%", va="center", fontsize=9, color=MGRAY, fontweight="500",
            )
        ax.xaxis.set_major_formatter(mticker.PercentFormatter())
        ax.set_title("Which rooms have the most cancellations? (historical)")
        ax.set_xlabel("% of events that were cancelled", color=MGRAY, fontsize=10)
        ax.invert_yaxis()
        plt.tight_layout()
        return fig
