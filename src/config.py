"""
config.py
---------
Global constants, colour palette, chart style defaults, and CSS.
"""

import os
import streamlit as st
import google.generativeai as genai

# ── API / Model ────────────────────────────────────────────────────────────────

def _get_gemini_key() -> str:
    try:
        return st.secrets["GEMINI_API_KEY"]
    except (KeyError, FileNotFoundError):
        pass
    return os.environ.get("GEMINI_API_KEY", "")


GEMINI_API_KEY = _get_gemini_key()
GEMINI_MODEL   = "gemini-2.5-flash"

genai.configure(api_key=GEMINI_API_KEY)

# ── Business rules ─────────────────────────────────────────────────────────────

PARTICIPANT_MAX = 1000

# ── Colour palette ─────────────────────────────────────────────────────────────

INDIGO  = "#4F46E5"
TEAL    = "#0D9488"
ROSE    = "#E11D48"
AMBER   = "#D97706"
EMERALD = "#059669"
VIOLET  = "#7C3AED"
SKY     = "#0284C7"
BLACK   = "#1A1A2E"
WHITE   = "#FFFFFF"
MGRAY   = "#94A3B8"
LGRAY   = "#F8FAFC"

CHART_COLORS = [
    "#4F46E5", "#0D9488", "#E11D48", "#D97706",
    "#059669", "#7C3AED", "#0284C7", "#DB2777",
    "#065F46", "#92400E",
]

# ── ML features ────────────────────────────────────────────────────────────────

ML_FEATURES = [
    "hour", "month", "weekday_num", "duration_hours",
    "room_enc", "activity_enc", "participants_clean",
]

WEEKDAY_MAP = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
    "Friday": 4, "Saturday": 5, "Sunday": 6,
}

# ── CSS / Design system ────────────────────────────────────────────────────────

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #FAFAFA !important;
    color: #1A1A2E;
}

[data-testid="stSidebar"] {
    background: #1E1B4B !important;
    border-right: 1px solid #312E81 !important;
}
[data-testid="stSidebar"] > div { padding-top: 0 !important; }
[data-testid="stSidebar"] * { color: #E0E7FF !important; }
[data-testid="stSidebar"] hr { border-color: #312E81 !important; }
[data-testid="stSidebar"] .stFileUploader {
    background: #2D2A5E !important;
    border: 1.5px dashed #6366F1 !important;
    border-radius: 10px;
    padding: 14px;
}
[data-testid="stSidebar"] .stFileUploader label { color: #A5B4FC !important; }
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #2D2A5E !important;
    border-color: #4338CA !important;
    color: #E0E7FF !important;
}
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] .stCaption { color: #6366F1 !important; }

.stApp { background: #FAFAFA !important; }
.block-container { padding-top: 0 !important; padding-bottom: 2rem; max-width: 100% !important; }

.kpi-card {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 18px 16px 14px;
    border: 1px solid #F0F0F0;
    box-shadow: 0 1px 6px rgba(0,0,0,0.04);
    text-align: center;
}
.kpi-value {
    font-size: clamp(1.2rem, 2vw, 1.9rem);
    font-weight: 700;
    color: #1A1A2E;
    letter-spacing: -0.02em;
    line-height: 1;
}
.kpi-label {
    font-size: 0.63rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #94A3B8;
    margin-top: 6px;
}

[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1.5px solid #EBEBEB !important;
    gap: 0;
}
[data-testid="stTabs"] [role="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    color: #94A3B8 !important;
    border-radius: 0 !important;
    padding: 10px 22px !important;
    border: none !important;
    background: transparent !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #4F46E5 !important;
    border-bottom: 2.5px solid #4F46E5 !important;
}

.stDownloadButton > button, .stButton > button {
    background: #4F46E5 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    padding: 10px 22px !important;
    box-shadow: 0 2px 8px rgba(79,70,229,0.25) !important;
    transition: all 0.15s ease !important;
}
.stDownloadButton > button:hover, .stButton > button:hover {
    background: #4338CA !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(79,70,229,0.3) !important;
}

[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    border-color: #C7D2FE !important;
    border-radius: 8px !important;
    background: #FFFFFF !important;
    color: #1A1A2E !important;
}

[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
    background: #EEF2FF !important;
    border: 1px solid #C7D2FE !important;
    border-radius: 6px !important;
    color: #3730A3 !important;
    font-weight: 600 !important;
    font-size: 0.78rem !important;
}
[data-testid="stMultiSelect"] span[data-baseweb="tag"] span { color: #3730A3 !important; }
[data-testid="stMultiSelect"] span[data-baseweb="tag"] button svg { fill: #6366F1 !important; }
[data-testid="stMultiSelect"] span[data-baseweb="tag"] button:hover svg { fill: #3730A3 !important; }
[data-testid="stMultiSelect"] [data-baseweb="select"] > div { background: #FFFFFF !important; }

[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border-radius: 10px !important;
    padding: 16px 14px 12px !important;
    border: 1px solid #E2E8F0 !important;
    border-top: 3px solid #4F46E5 !important;
    box-shadow: 0 1px 6px rgba(0,0,0,0.04) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    color: #1A1A2E !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.67rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: #64748B !important;
}

[data-testid="stExpander"] {
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
    background: #FFFFFF !important;
    margin-bottom: 8px !important;
}
[data-testid="stExpander"] summary {
    font-weight: 600 !important;
    font-size: 0.86rem !important;
    color: #374151 !important;
    padding: 12px 16px !important;
}
[data-testid="stExpander"] summary:hover {
    background: #F8F9FF !important;
    border-radius: 10px !important;
}

.section-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #4F46E5;
    margin: 2rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #EBEBEB;
}

.chart-card {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 24px 24px 18px;
    margin-bottom: 16px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.04);
    border: 1px solid #F0F0F0;
}

.stAlert {
    border-radius: 10px !important;
    border-left: 3px solid #4F46E5 !important;
    background: #F0F0FF !important;
}

[data-testid="stDataFrame"] { border-radius: 10px !important; overflow: hidden !important; }
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #F8F8F8; }
::-webkit-scrollbar-thumb { background: #E2E8F0; border-radius: 4px; }

.chat-bubble-user {
    background: #4F46E5;
    color: #FFFFFF;
    border-radius: 14px 14px 4px 14px;
    padding: 12px 16px;
    margin: 6px 0 6px auto;
    max-width: 72%;
    font-size: 0.88rem;
    line-height: 1.5;
    width: fit-content;
}
.chat-bubble-assistant {
    background: #FFFFFF;
    color: #1A1A2E;
    border-radius: 14px 14px 14px 4px;
    padding: 12px 16px;
    margin: 6px auto 6px 0;
    max-width: 80%;
    font-size: 0.88rem;
    line-height: 1.5;
    border: 1px solid #E2E8F0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    width: fit-content;
}
.chat-bubble-assistant .thought {
    font-size: 0.74rem;
    color: #94A3B8;
    font-style: italic;
    margin-bottom: 6px;
    border-bottom: 1px solid #F0F0F0;
    padding-bottom: 6px;
}
.chat-error {
    background: #FFF1F2;
    border: 1px solid #FECDD3;
    border-radius: 10px;
    padding: 12px 16px;
    color: #E11D48;
    font-size: 0.82rem;
    margin: 6px 0;
}
.chat-code {
    background: #F8F9FF;
    border: 1px solid #E0E7FF;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.76rem;
    font-family: 'Courier New', monospace;
    color: #3730A3;
    margin-top: 6px;
    overflow-x: auto;
    white-space: pre-wrap;
}
.chat-input-area {
    position: sticky;
    bottom: 0;
    background: #FAFAFA;
    padding: 12px 0 4px;
    border-top: 1px solid #EBEBEB;
    margin-top: 16px;
}
</style>
"""
