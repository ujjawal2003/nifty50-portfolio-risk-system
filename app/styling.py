"""
Visual theme for the app: custom CSS injection + a shared Plotly template
so every chart in the app looks consistent.
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio

PRIMARY = "#4C6FFF"
ACCENT = "#00C2A8"
BG_CARD = "rgba(255,255,255,0.03)"
DANGER = "#e74c3c"
WARNING = "#e67e22"
CAUTION = "#f1c40f"
SAFE = "#2ecc71"
TEXT_MUTED = "#9aa4b2"

FONT_FAMILY = "'Inter', 'Segoe UI', sans-serif"


def inject_css():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {{
            font-family: {FONT_FAMILY};
        }}

        /* Force a dark background regardless of the host's light/dark theme
           setting -- the card/text colors below assume a dark canvas. */
        html, body,
        [data-testid="stAppViewContainer"],
        [data-testid="stAppViewContainer"] > .main,
        [data-testid="stHeader"],
        .stApp {{
            background-color: #0e1220 !important;
            color: #f5f7ff !important;
        }}
        [data-testid="stSidebar"] {{
            background-color: #12172b !important;
            color: #f5f7ff !important;
        }}
        [data-testid="stSidebar"] * {{
            color: #f5f7ff !important;
        }}
        [data-testid="stHeader"] {{
            background-color: transparent !important;
        }}

        /* Default text elements (markdown, captions, labels) */
        p, span, li, label, div[data-testid="stMarkdownContainer"] {{
            color: #e8ecf7;
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: #f5f7ff !important;
        }}
        .stCaption, [data-testid="stCaptionContainer"] {{
            color: {TEXT_MUTED} !important;
        }}

        /* Hide default streamlit chrome we don't want */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}

        .block-container {{
            padding-top: 1.6rem;
            padding-bottom: 3rem;
            max-width: 1300px;
        }}

        /* ---- Hero header ---- */
        .hero {{
            background: linear-gradient(135deg, #12172b 0%, #1a2140 45%, #202a52 100%);
            border-radius: 18px;
            padding: 28px 32px;
            margin-bottom: 22px;
            border: 1px solid rgba(255,255,255,0.06);
            box-shadow: 0 10px 30px rgba(15, 20, 40, 0.35);
        }}
        .hero h1 {{
            font-size: 1.9rem;
            font-weight: 800;
            color: #f5f7ff;
            margin-bottom: 2px;
            letter-spacing: -0.02em;
        }}
        .hero p {{
            color: {TEXT_MUTED};
            font-size: 0.95rem;
            margin: 0;
        }}
        .hero .badge-row {{
            margin-top: 14px;
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }}

        /* ---- Pill badges ---- */
        .pill {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.02em;
            border: 1px solid rgba(255,255,255,0.12);
            color: #e8ecf7;
            background: rgba(255,255,255,0.06);
        }}
        .pill-safe {{ background: rgba(46, 204, 113, 0.16); color: #2ecc71; border-color: rgba(46,204,113,0.35);}}
        .pill-caution {{ background: rgba(241, 196, 15, 0.16); color: #f1c40f; border-color: rgba(241,196,15,0.35);}}
        .pill-warning {{ background: rgba(230, 126, 34, 0.16); color: #e67e22; border-color: rgba(230,126,34,0.35);}}
        .pill-danger {{ background: rgba(231, 76, 60, 0.18); color: #e74c3c; border-color: rgba(231,76,60,0.4);}}

        /* ---- Metric / stat cards ---- */
        .stat-card {{
            background: {BG_CARD};
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 14px;
            padding: 16px 18px;
            height: 100%;
        }}
        .stat-card .label {{
            font-size: 0.78rem;
            color: {TEXT_MUTED};
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 6px;
        }}
        .stat-card .value {{
            font-size: 1.6rem;
            font-weight: 800;
            color: #f5f7ff;
        }}
        .stat-card .sub {{
            font-size: 0.78rem;
            color: {TEXT_MUTED};
            margin-top: 4px;
        }}

        /* ---- Section headers ---- */
        .section-title {{
            font-size: 1.15rem;
            font-weight: 700;
            color: #f5f7ff;
            margin-top: 8px;
            margin-bottom: 4px;
        }}
        .section-sub {{
            color: {TEXT_MUTED};
            font-size: 0.85rem;
            margin-bottom: 14px;
        }}

        .info-box {{
            background: rgba(76, 111, 255, 0.08);
            border: 1px solid rgba(76, 111, 255, 0.25);
            border-radius: 12px;
            padding: 14px 16px;
            font-size: 0.87rem;
            color: #d7ddf5;
        }}
        .warn-box {{
            background: rgba(230, 126, 34, 0.1);
            border: 1px solid rgba(230, 126, 34, 0.3);
            border-radius: 12px;
            padding: 14px 16px;
            font-size: 0.87rem;
            color: #f3dcc4;
        }}

        div[data-testid="stMetric"] {{
            background: {BG_CARD};
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 14px;
            padding: 12px 16px;
        }}
        div[data-testid="stMetric"] label {{
            color: {TEXT_MUTED} !important;
        }}
        div[data-testid="stMetricValue"] {{
            color: #f5f7ff !important;
        }}

        /* Tabs */
        button[data-baseweb="tab"] {{
            font-weight: 600;
            font-size: 0.92rem;
            color: {TEXT_MUTED} !important;
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{
            color: #f5f7ff !important;
        }}
        [data-baseweb="tab-border"] {{
            background: rgba(255,255,255,0.08) !important;
        }}
        [data-baseweb="tab-highlight"] {{
            background-color: {PRIMARY} !important;
        }}

        /* Dataframes / tables */
        [data-testid="stDataFrame"] {{
            background: {BG_CARD};
            border-radius: 12px;
        }}

        /* Buttons */
        .stButton > button {{
            background: rgba(76,111,255,0.12);
            color: #e8ecf7;
            border: 1px solid rgba(76,111,255,0.35);
            border-radius: 10px;
        }}
        .stButton > button:hover {{
            background: rgba(76,111,255,0.22);
            border-color: {PRIMARY};
            color: #ffffff;
        }}

        /* Sliders */
        [data-testid="stSlider"] label {{
            color: #e8ecf7 !important;
        }}

        /* Expander */
        [data-testid="stExpander"] {{
            background: {BG_CARD};
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 12px;
        }}

        /* Images (SHAP figures etc.) sit on white PNGs -- give them a light
           card behind them so they don't look like floating cutouts. */
        [data-testid="stImage"] img {{
            border-radius: 10px;
            background: #ffffff;
            padding: 6px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_plotly_theme():
    template = go.layout.Template()
    template.layout = go.Layout(
        font=dict(family=FONT_FAMILY, color="#e8ecf7", size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=[PRIMARY, ACCENT, WARNING, "#9B6BFF", SAFE, DANGER, CAUTION, "#36C5F0"],
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.12)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.12)"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    pio.templates["risk_app"] = template
    pio.templates.default = "risk_app"
