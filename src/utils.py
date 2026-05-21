"""
utils.py
--------
Pure helper functions shared across modules:
  - parse_date      : flexible datetime parser
  - find_col        : case-insensitive column lookup
  - map_status      : normalise status strings → 'Tenu' / 'Annulé'
  - map_space       : normalise room/space names
  - chart_style     : global matplotlib rcParams
  - make_fig        : factory for a styled (fig, ax) pair
"""

import re
import datetime

import pandas as pd
import matplotlib.pyplot as plt

from src.config import MGRAY, BLACK, WHITE


# ── Date parsing ───────────────────────────────────────────────────────────────

def parse_date(s):
    """Parse a wide variety of date/datetime strings into a pandas Timestamp."""
    if pd.isna(s):
        return None
    s = str(s).strip()
    if s == "":
        return None

    m = re.search(r"(\w{3} \w{3} \d{2} \d{4} \d{2}:\d{2}:\d{2})", s)
    if m:
        try:
            return pd.to_datetime(m.group(1), format="%a %b %d %Y %H:%M:%S")
        except Exception:
            pass

    m = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", s)
    if m:
        try:
            return pd.to_datetime(m.group(1))
        except Exception:
            pass

    try:
        return pd.to_datetime(s)
    except Exception:
        return None


# ── Column discovery ───────────────────────────────────────────────────────────

def find_col(df, *candidates):
    """Return the first column name (case-insensitive) that matches any candidate."""
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


# ── Status / space normalisers ─────────────────────────────────────────────────

def map_status(s):
    """Normalise raw status strings to 'Tenu' or 'Annulé'."""
    if pd.isna(s):
        return s
    return {
        "finished": "Tenu", "canceled": "Annulé", "rejected": "Annulé",
        "accepted": "Tenu", "tenu": "Tenu", "annulé": "Annulé", "annule": "Annulé",
    }.get(str(s).strip().lower(), str(s).strip())


def map_space(s):
    """Normalise raw room/space strings to canonical names."""
    if pd.isna(s) or str(s).strip() == "":
        return ""
    return {
        "Salle Polyvalente":      "Salle de formation",
        "ESPACE FONDATION":       "Salle Fondation",
        "Training Room":          "Salle de formation",
        "Terrasse":               "Terrasse",
        "BALE":                   "BALE",
        "Podcast Tunisia":        "Podcast",
        "Salle Design Thinking":  "Salle Design Thinking",
        "Salle de réunion 113":   "Salle de réunion 113",
    }.get(str(s).strip(), str(s).strip())


# ── Matplotlib helpers ─────────────────────────────────────────────────────────

def chart_style():
    """Apply global rcParams for a clean, branded chart style."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.spines.bottom": True,
        "axes.edgecolor": "#EBEBEB", "axes.facecolor": WHITE,
        "figure.facecolor": WHITE, "grid.color": "#F4F4F8",
        "grid.linewidth": 0.6, "xtick.color": MGRAY, "ytick.color": MGRAY,
        "axes.titlesize": 12, "axes.titleweight": "600",
        "axes.titlecolor": BLACK, "axes.titlepad": 14,
    })


def make_fig(w=9, h=4.5):
    """Return a (fig, ax) pair pre-styled for the dashboard."""
    fig, ax = plt.subplots(figsize=(w, h), facecolor=WHITE)
    ax.set_facecolor(WHITE)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#EBEBEB")
    ax.tick_params(colors=MGRAY, length=0)
    ax.yaxis.grid(True, color="#F4F4F8", linewidth=0.7)
    ax.set_axisbelow(True)
    return fig, ax
