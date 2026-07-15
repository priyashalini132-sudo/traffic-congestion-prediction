# -*- coding: utf-8 -*-
"""
app.py — Traffic Congestion Prediction & Smart Traffic Analytics System
Professional Streamlit Dashboard
"""

import os
import sys
import json
import warnings
import logging
import datetime

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TrafficIQ — Congestion Prediction System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR   = os.path.join(BASE_DIR, "data", "raw")
DATA_PROC_DIR  = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR     = os.path.join(BASE_DIR, "models")
REPORTS_DIR    = os.path.join(BASE_DIR, "reports")

RAW_CSV        = os.path.join(DATA_RAW_DIR, "traffic.csv")
CLEAN_CSV      = os.path.join(DATA_PROC_DIR, "cleaned_traffic.csv")
MODEL_PKL      = os.path.join(MODELS_DIR, "best_model.pkl")
PREP_PKL       = os.path.join(MODELS_DIR, "preprocessor.pkl")
META_JSON      = os.path.join(MODELS_DIR, "model_metadata.json")
COMPARISON_CSV = os.path.join(REPORTS_DIR, "model_comparison.csv")

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stDateInput label { color: #94a3b8 !important; font-size:0.78rem; }

/* ── Main background ── */
.main { background: #0f172a; }
.block-container { padding: 1.5rem 2rem 3rem; }

/* ── Metric cards ── */
.metric-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover { transform: translateY(-3px); box-shadow: 0 8px 30px rgba(99,102,241,0.2); }
.metric-label { font-size: 0.72rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.4rem; }
.metric-value { font-size: 2rem; font-weight: 700; color: #e2e8f0; line-height: 1; }
.metric-sub   { font-size: 0.78rem; color: #94a3b8; margin-top: 0.3rem; }

/* ── Risk badge ── */
.badge-low    { background:#064e3b; color:#34d399; border:1px solid #065f46; border-radius:8px; padding:4px 14px; font-weight:600; }
.badge-medium { background:#78350f; color:#fbbf24; border:1px solid #92400e; border-radius:8px; padding:4px 14px; font-weight:600; }
.badge-high   { background:#7f1d1d; color:#f87171; border:1px solid #991b1b; border-radius:8px; padding:4px 14px; font-weight:600; }

/* ── Section header ── */
.section-header {
    font-size: 1.05rem; font-weight: 700; color: #818cf8;
    text-transform: uppercase; letter-spacing: 0.1em;
    border-bottom: 2px solid rgba(99,102,241,0.3);
    padding-bottom: 0.5rem; margin-bottom: 1.2rem;
}

/* ── Alert boxes ── */
.info-box {
    background: rgba(99,102,241,0.1); border: 1px solid rgba(99,102,241,0.3);
    border-radius: 12px; padding: 1rem 1.2rem; color: #c7d2fe; font-size: 0.9rem;
}
.warn-box {
    background: rgba(245,158,11,0.1); border: 1px solid rgba(245,158,11,0.3);
    border-radius: 12px; padding: 1rem 1.2rem; color: #fde68a; font-size: 0.9rem;
}
.success-box {
    background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.3);
    border-radius: 12px; padding: 1rem 1.2rem; color: #6ee7b7; font-size: 0.9rem;
}

/* ── Progress bar ── */
.risk-bar-bg { background:#1e293b; border-radius:100px; height:14px; width:100%; overflow:hidden; }
.risk-bar-fill { height:14px; border-radius:100px; transition:width 0.5s; }

/* ── Plotly dark theme fix ── */
.js-plotly-plot .plotly { background: transparent !important; }

/* ── Hide Streamlit branding ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { gap: 8px; background: transparent; border-bottom: 1px solid rgba(99,102,241,0.2); }
.stTabs [data-baseweb="tab"] {
    background: transparent; border: 1px solid rgba(99,102,241,0.2);
    border-radius: 8px 8px 0 0; color: #94a3b8; font-weight: 500; padding: 8px 20px;
}
.stTabs [aria-selected="true"] { background: rgba(99,102,241,0.15) !important; color: #818cf8 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; } ::-webkit-scrollbar-track { background: #0f172a; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Dark Plotly template ───────────────────────────────────────────────────────
PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(15,23,42,0)",
    plot_bgcolor="rgba(15,23,42,0)",
    font=dict(family="Inter", color="#e2e8f0"),
    title_font=dict(size=15, color="#818cf8"),
    legend=dict(bgcolor="rgba(30,41,59,0.7)", bordercolor="rgba(99,102,241,0.3)", borderwidth=1),
    margin=dict(l=30, r=20, t=50, b=30),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False),
)

# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_clean_data() -> pd.DataFrame | None:
    path = CLEAN_CSV if os.path.exists(CLEAN_CSV) else RAW_CSV
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["DateTime"])
    if "Hour" not in df.columns:
        df["Hour"] = df["DateTime"].dt.hour
    if "DayOfWeek" not in df.columns:
        df["DayOfWeek"] = df["DateTime"].dt.dayofweek
    if "IsWeekend" not in df.columns:
        df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
    if "Month" not in df.columns:
        df["Month"] = df["DateTime"].dt.month
    if "IsRushHour" not in df.columns:
        df["IsRushHour"] = ((df["Hour"].isin([7, 8, 9, 17, 18])) & (df["IsWeekend"] == 0)).astype(int)
    return df

@st.cache_resource(show_spinner=False)
def load_model_artifacts():
    import joblib
    if not os.path.exists(MODEL_PKL) or not os.path.exists(PREP_PKL):
        return None, None
    model = joblib.load(MODEL_PKL)
    prep  = joblib.load(PREP_PKL)
    return model, prep

@st.cache_data(show_spinner=False)
def load_metadata() -> dict | None:
    if not os.path.exists(META_JSON):
        return None
    with open(META_JSON) as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def load_comparison() -> pd.DataFrame | None:
    if not os.path.exists(COMPARISON_CSV):
        return None
    return pd.read_csv(COMPARISON_CSV)

def data_exists() -> bool:
    return os.path.exists(RAW_CSV) or os.path.exists(CLEAN_CSV)

def model_exists() -> bool:
    return os.path.exists(MODEL_PKL) and os.path.exists(PREP_PKL)


def run_predict(dt: pd.Timestamp, junction: int, model, prep) -> dict:
    """Run single-point prediction using the loaded model + preprocessor."""
    from src.feature_engineering import build_features
    from src.risk_scoring import calculate_congestion_metrics

    raw_row = pd.DataFrame([{"DateTime": dt, "Junction": int(junction)}])
    profiles = prep["historical_profiles"]
    feat_row = build_features(raw_row, profiles)

    for j in [1, 2, 3, 4]:
        feat_row[f"Junction_{j}"] = 1 if junction == j else 0

    feature_cols = prep["feature_names"]
    X_raw = feat_row[feature_cols].copy().fillna(0.0)

    cols_to_scale = ["Hour", "Day", "DayOfWeek", "Month", "Year", "Hist_Traffic_J_D_H", "Hist_Traffic_J_H"]
    scaler = prep["scaler"]
    X_scaled = X_raw.copy()
    X_scaled[cols_to_scale] = scaler.transform(X_raw[cols_to_scale])

    predicted = float(max(0.0, model.predict(X_scaled.values)[0]))
    metrics = calculate_congestion_metrics(predicted, junction, prep["risk_thresholds"])
    return metrics


def risk_color(level: str) -> str:
    return {"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"}.get(level, "#6b7280")

def risk_badge(level: str) -> str:
    cls = {"Low": "badge-low", "Medium": "badge-medium", "High": "badge-high"}.get(level, "")
    return f'<span class="{cls}">{level}</span>'

def apply_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(**PLOTLY_THEME)
    return fig

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚦 TrafficIQ")
    st.markdown("<p style='color:#64748b;font-size:0.78rem;margin-top:-0.5rem;'>Congestion Prediction System v1.0</p>", unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Overview", "🔮 Live Predict", "📊 EDA & Analysis", "🤖 Model Performance", "ℹ️ About"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("<p class='metric-label'>System Status</p>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            f"<div style='text-align:center;padding:6px;border-radius:8px;background:{'rgba(16,185,129,0.15)' if data_exists() else 'rgba(239,68,68,0.15)'};border:1px solid {'#065f46' if data_exists() else '#7f1d1d'}'>"
            f"<div style='font-size:0.65rem;color:#64748b'>DATA</div>"
            f"<div style='font-size:0.85rem;font-weight:700;color:{'#34d399' if data_exists() else '#f87171'}'>{'✔ Ready' if data_exists() else '✘ Missing'}</div></div>",
            unsafe_allow_html=True
        )
    with col_b:
        st.markdown(
            f"<div style='text-align:center;padding:6px;border-radius:8px;background:{'rgba(16,185,129,0.15)' if model_exists() else 'rgba(239,68,68,0.15)'};border:1px solid {'#065f46' if model_exists() else '#7f1d1d'}'>"
            f"<div style='font-size:0.65rem;color:#64748b'>MODEL</div>"
            f"<div style='font-size:0.85rem;font-weight:700;color:{'#34d399' if model_exists() else '#f87171'}'>{'✔ Ready' if model_exists() else '✘ Untrained'}</div></div>",
            unsafe_allow_html=True
        )
    st.markdown("")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("# 🚦 Traffic Congestion Prediction")
    st.markdown("<p style='color:#64748b;font-size:1.05rem;'>Smart Traffic Analytics · Real-time Risk Scoring · Multi-Model ML</p>", unsafe_allow_html=True)
    st.markdown("")

    # ── Setup instructions if data/model missing ──────────────────────────────
    if not data_exists():
        st.markdown('<div class="warn-box">⚠️ <strong>Dataset not found.</strong><br>'
                    'Download <code>traffic.csv</code> from '
                    '<a href="https://www.kaggle.com/datasets/fedesoriano/traffic-prediction-dataset" target="_blank" style="color:#fbbf24">Kaggle — Traffic Prediction Dataset</a>'
                    ' and place it in <code>data/raw/traffic.csv</code>.</div>', unsafe_allow_html=True)
        st.markdown("")

    if not model_exists():
        st.markdown('<div class="info-box">💡 <strong>Models not yet trained.</strong><br>'
                    'After placing the dataset, run in your terminal:<br>'
                    '<code style="background:rgba(99,102,241,0.15);padding:2px 8px;border-radius:4px;">'
                    'python -m src.train</code></div>', unsafe_allow_html=True)
        st.markdown("")

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    df = load_clean_data()
    meta = load_metadata()

    if df is not None:
        total_records = len(df)
        junctions = df["Junction"].nunique()
        peak_hour = int(df.groupby("Hour")["Vehicles"].mean().idxmax())
        avg_vehicles = df["Vehicles"].mean()
        max_vehicles = df["Vehicles"].max()
        date_range = f"{df['DateTime'].min().strftime('%b %Y')} – {df['DateTime'].max().strftime('%b %Y')}"

        c1, c2, c3, c4, c5 = st.columns(5)
        cards = [
            (c1, "📋 Total Records",  f"{total_records:,}",    date_range),
            (c2, "🔀 Junctions",      str(junctions),           "Monitored Nodes"),
            (c3, "⏰ Peak Hour",       f"{peak_hour}:00",        "Daily Congestion Peak"),
            (c4, "🚗 Avg Vehicles/hr", f"{avg_vehicles:.0f}",   "Across All Junctions"),
            (c5, "📈 Max Recorded",   f"{max_vehicles:.0f}",    "Vehicles / Hour"),
        ]
        for col, label, val, sub in cards:
            with col:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">{label}</div>'
                    f'<div class="metric-value">{val}</div>'
                    f'<div class="metric-sub">{sub}</div></div>',
                    unsafe_allow_html=True
                )
        st.markdown("")

        # ── Quick heatmap ──────────────────────────────────────────────────────
        st.markdown('<div class="section-header">Congestion Heat Map · Junction × Hour</div>', unsafe_allow_html=True)
        pivot = df.groupby(["Junction", "Hour"])["Vehicles"].mean().reset_index()
        pivot = pivot.pivot(index="Junction", columns="Hour", values="Vehicles")
        fig = px.imshow(
              pivot,
        labels=dict(x="Hour", y="Junction", color="Avg Vehicles"),
        aspect="auto"
        )
       
        apply_theme(fig)
        fig.update_layout(height=260, title="")
        st.plotly_chart(fig, use_container_width=True)

        # ── Overview charts row ────────────────────────────────────────────────
        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.markdown('<div class="section-header">Hourly Traffic Trends by Junction</div>', unsafe_allow_html=True)
            hourly = df.groupby(["Junction", "Hour"])["Vehicles"].mean().reset_index()
            hourly["Junction"] = "Junction " + hourly["Junction"].astype(str)
            fig2 = px.line(hourly, x="Hour", y="Vehicles", color="Junction",
                           markers=True, color_discrete_sequence=px.colors.qualitative.Vivid)
            apply_theme(fig2)
            fig2.update_layout(height=280, xaxis_title="Hour of Day", yaxis_title="Avg Vehicles",
                                hovermode="x unified")
            st.plotly_chart(fig2, use_container_width=True)

        with col_right:
            st.markdown('<div class="section-header">Weekday vs Weekend Split</div>', unsafe_allow_html=True)
            wk = df.groupby(["Junction", "IsWeekend"])["Vehicles"].mean().reset_index()
            wk["Period"] = wk["IsWeekend"].map({0: "Weekday", 1: "Weekend"})
            wk["Junction"] = "J" + wk["Junction"].astype(str)
            fig3 = px.bar(wk, x="Junction", y="Vehicles", color="Period", barmode="group",
                          color_discrete_map={"Weekday": "#6366f1", "Weekend": "#f59e0b"})
            apply_theme(fig3)
            fig3.update_layout(height=280, xaxis_title="Junction", yaxis_title="Avg Vehicles", legend_title="")
            st.plotly_chart(fig3, use_container_width=True)

        # ── Model summary if trained ───────────────────────────────────────────
        if meta:
            st.markdown('<div class="section-header">Best Model Summary</div>', unsafe_allow_html=True)
            best_name = meta.get("best_model_name", "–")
            best_metrics = next((m for m in meta.get("metrics", []) if m["Model"] == best_name), {})
            m1, m2, m3, m4 = st.columns(4)
            for col, label, key, fmt in [
                (m1, "🏆 Best Model", "Model", "{}"),
                (m2, "📉 MAE", "MAE", "{:.2f}"),
                (m3, "📐 RMSE", "RMSE", "{:.2f}"),
                (m4, "🎯 R² Score", "R2", "{:.4f}"),
            ]:
                val = best_metrics.get(key, best_name if key == "Model" else "–")
                display = fmt.format(val) if val != "–" else val
                with col:
                    st.markdown(
                        f'<div class="metric-card"><div class="metric-label">{label}</div>'
                        f'<div class="metric-value" style="font-size:1.35rem">{display}</div></div>',
                        unsafe_allow_html=True
                    )
    else:
        st.markdown('<div class="info-box">📂 No data loaded yet. Follow the setup instructions above to get started.</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: LIVE PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Live Predict":
    st.markdown("# 🔮 Live Traffic Prediction")
    st.markdown("<p style='color:#64748b'>Predict traffic volume and congestion risk for any date, time, and junction.</p>", unsafe_allow_html=True)
    st.markdown("")

    if not model_exists():
        st.markdown('<div class="warn-box">⚠️ <strong>No trained model found.</strong><br>'
                    'Please run <code>python -m src.train</code> first.</div>', unsafe_allow_html=True)
        st.stop()

    model, prep = load_model_artifacts()
    if model is None:
        st.error("Failed to load model artifacts.")
        st.stop()

    meta = load_metadata()

    # ── Input form ────────────────────────────────────────────────────────────
    with st.container():
        st.markdown('<div class="section-header">🗓️ Prediction Parameters</div>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
        with col1:
            pred_date = st.date_input("Date", value=datetime.date.today(), min_value=datetime.date(2015, 1, 1))
        with col2:
            pred_hour = st.slider("Hour of Day", 0, 23, datetime.datetime.now().hour)
        with col3:
            pred_junction = st.selectbox("Junction", [1, 2, 3, 4], format_func=lambda x: f"Junction {x}")
        with col4:
            st.markdown("<br>", unsafe_allow_html=True)
            predict_btn = st.button("🚀 Predict", use_container_width=True, type="primary")

    if predict_btn:
        dt = pd.Timestamp(datetime.datetime.combine(pred_date, datetime.time(pred_hour, 0)))
        with st.spinner("Running prediction…"):
            try:
                result = run_predict(dt, pred_junction, model, prep)
            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.stop()

        st.markdown("")
        st.markdown('<div class="section-header">🎯 Prediction Results</div>', unsafe_allow_html=True)

        lvl   = result["congestion_level"]
        score = result["risk_score"]
        veh   = result["predicted_vehicles"]
        rec   = result["recommendation"]
        color = risk_color(lvl)

        # ── Result KPIs ────────────────────────────────────────────────────────
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">🚗 Predicted Vehicles</div>'
                f'<div class="metric-value">{veh:.0f}</div><div class="metric-sub">vehicles / hour</div></div>',
                unsafe_allow_html=True
            )
        with r2:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">🔥 Congestion Risk Score</div>'
                f'<div class="metric-value" style="color:{color}">{score:.1f}</div>'
                f'<div class="metric-sub">out of 100</div></div>',
                unsafe_allow_html=True
            )
        with r3:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">🚦 Congestion Level</div>'
                f'<div style="margin-top:0.5rem">{risk_badge(lvl)}</div>'
                f'<div class="metric-sub" style="margin-top:0.6rem">{lvl} Risk</div></div>',
                unsafe_allow_html=True
            )
        with r4:
            day_name = dt.strftime("%A")
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">📅 Prediction For</div>'
                f'<div class="metric-value" style="font-size:1.1rem">{dt.strftime("%d %b %Y")}</div>'
                f'<div class="metric-sub">{day_name}, {pred_hour:02d}:00 · Junction {pred_junction}</div></div>',
                unsafe_allow_html=True
            )

        st.markdown("")
        # ── Risk bar ──────────────────────────────────────────────────────────
        st.markdown(f'<div class="metric-label">Congestion Risk Level</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="risk-bar-bg"><div class="risk-bar-fill" style="width:{score}%;background:linear-gradient(90deg,{color}88,{color})"></div></div>',
            unsafe_allow_html=True
        )
        st.markdown(f"<br><div class='info-box'>💬 <strong>Advisory:</strong> {rec}</div>", unsafe_allow_html=True)
        st.markdown("")

        # ── 24-hour forecast chart ─────────────────────────────────────────────
        st.markdown('<div class="section-header">📈 24-Hour Forecast for Junction {}</div>'.format(pred_junction), unsafe_allow_html=True)
        hours = list(range(24))
        forecasts = []
        for h in hours:
            dt_h = pd.Timestamp(datetime.datetime.combine(pred_date, datetime.time(h, 0)))
            try:
                r = run_predict(dt_h, pred_junction, model, prep)
                forecasts.append({
                    "Hour": h,
                    "Vehicles": r["predicted_vehicles"],
                    "Risk": r["risk_score"],
                    "Level": r["congestion_level"]
                })
            except Exception:
                forecasts.append({"Hour": h, "Vehicles": 0, "Risk": 0, "Level": "Low"})

        fore_df = pd.DataFrame(forecasts)
        level_colors = fore_df["Level"].map({"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"})

        fig_fore = go.Figure()
        fig_fore.add_trace(go.Scatter(
            x=fore_df["Hour"], y=fore_df["Vehicles"],
            mode="lines+markers",
            line=dict(color="#6366f1", width=3),
            marker=dict(color=level_colors.tolist(), size=10, line=dict(color="#0f172a", width=2)),
            name="Predicted Vehicles",
            hovertemplate="Hour: %{x}:00<br>Vehicles: %{y:.0f}<extra></extra>"
        ))
        fig_fore.add_vrect(x0=pred_hour - 0.3, x1=pred_hour + 0.3, fillcolor="rgba(99,102,241,0.3)",
                           line_width=0, annotation_text="Selected", annotation_position="top")
        apply_theme(fig_fore)
        fig_fore.update_layout(height=300, xaxis=dict(tickvals=list(range(0, 24, 2)),
                                                       ticktext=[f"{h:02d}:00" for h in range(0, 24, 2)]),
                               yaxis_title="Vehicles / hr")
        st.plotly_chart(fig_fore, use_container_width=True)

        # ── All junctions comparison ───────────────────────────────────────────
        st.markdown(f'<div class="section-header">🔀 Junction Comparison at {pred_hour:02d}:00</div>', unsafe_allow_html=True)
        jcomp = []
        for j in [1, 2, 3, 4]:
            try:
                r = run_predict(dt, j, model, prep)
                jcomp.append({"Junction": f"Junction {j}", "Vehicles": r["predicted_vehicles"], "Risk": r["risk_score"], "Level": r["congestion_level"]})
            except Exception:
                jcomp.append({"Junction": f"Junction {j}", "Vehicles": 0, "Risk": 0, "Level": "Low"})

        jcomp_df = pd.DataFrame(jcomp)
        bar_colors = jcomp_df["Level"].map({"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"}).tolist()

        fig_j = go.Figure(go.Bar(
            x=jcomp_df["Junction"], y=jcomp_df["Vehicles"],
            marker_color=bar_colors,
            text=jcomp_df["Level"], textposition="auto",
            hovertemplate="%{x}<br>Vehicles: %{y:.0f}<extra></extra>"
        ))
        apply_theme(fig_j)
        fig_j.update_layout(height=280,title=NONE, yaxis_title="Predicted Vehicles")
        st.plotly_chart(fig_j, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA & ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA & Analysis":
    st.markdown("# 📊 Exploratory Data Analysis")
    st.markdown("<p style='color:#64748b'>Deep-dive into historical traffic patterns across junctions and time periods.</p>", unsafe_allow_html=True)

    df = load_clean_data()
    if df is None:
        st.markdown('<div class="warn-box">⚠️ Dataset not found. Please add <code>data/raw/traffic.csv</code>.</div>', unsafe_allow_html=True)
        st.stop()

    # ── Filters ───────────────────────────────────────────────────────────────
    with st.expander("🔍 Filters", expanded=False):
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            sel_junctions = st.multiselect("Junction(s)", sorted(df["Junction"].unique()),
                                            default=sorted(df["Junction"].unique()), key="eda_j")
        with col_f2:
            min_d = df["DateTime"].min().date()
            max_d = df["DateTime"].max().date()
            date_from = st.date_input("From", min_d, min_value=min_d, max_value=max_d, key="eda_from")
        with col_f3:
            date_to = st.date_input("To", max_d, min_value=min_d, max_value=max_d, key="eda_to")

    dff = df[
        (df["Junction"].isin(sel_junctions)) &
        (df["DateTime"].dt.date >= date_from) &
        (df["DateTime"].dt.date <= date_to)
    ].copy()

    st.markdown(f"<p style='color:#64748b;font-size:0.82rem'>Showing {len(dff):,} records from {date_from} to {date_to} for junctions {sel_junctions}</p>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["⏰ Hourly", "📅 Daily/Weekly", "🌡️ Heat Maps", "📦 Distributions", "📈 Time Series"])

    with tab1:
        hourly = dff.groupby(["Junction", "Hour"])["Vehicles"].mean().reset_index()
        hourly["Junction"] = "Junction " + hourly["Junction"].astype(str)
        fig = px.line(hourly, x="Hour", y="Vehicles", color="Junction", markers=True,
                      color_discrete_sequence=px.colors.qualitative.Vivid)
        apply_theme(fig)
        fig.update_layout(height=380, title="Average Vehicles by Hour", xaxis_title="Hour of Day")
        st.plotly_chart(fig, use_container_width=True)

        # Rush hour analysis
        rush_df = dff.copy()
        rush_df["Period"] = rush_df.apply(
            lambda r: "AM Rush (7-9)" if r["Hour"] in [7,8,9] and r["IsWeekend"]==0
            else ("PM Rush (17-19)" if r["Hour"] in [17,18,19] and r["IsWeekend"]==0
                  else ("Weekend" if r["IsWeekend"]==1 else "Off-Peak")), axis=1
        )
        rush_avg = rush_df.groupby(["Junction", "Period"])["Vehicles"].mean().reset_index()
        rush_avg["Junction"] = "Junction " + rush_avg["Junction"].astype(str)
        col1, col2 = st.columns(2)
        with col1:
            fig_r = px.bar(rush_avg, x="Junction", y="Vehicles", color="Period",
                           barmode="group", title="Traffic by Time Period",
                           color_discrete_sequence=px.colors.qualitative.Bold)
            apply_theme(fig_r)
            fig_r.update_layout(height=300)
            st.plotly_chart(fig_r, use_container_width=True)
        with col2:
            hour_mean = dff.groupby("Hour")["Vehicles"].mean().reset_index()
            fig_b = go.Figure(go.Bar(
                x=hour_mean["Hour"], y=hour_mean["Vehicles"],
                marker_color=[("#ef4444" if h in [7,8,9,17,18,19] else "#6366f1") for h in hour_mean["Hour"]],
                hovertemplate="Hour %{x}:00<br>Avg: %{y:.0f}<extra></extra>"
            ))
            apply_theme(fig_b)
            fig_b.update_layout(height=300, title="Avg Vehicles per Hour (All Junctions)",
                                 xaxis_title="Hour", yaxis_title="Avg Vehicles")
            st.plotly_chart(fig_b, use_container_width=True)

    with tab2:
        day_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
        dff["DayName"] = dff["DayOfWeek"].map(day_map)
        daily = dff.groupby(["Junction", "DayOfWeek", "DayName"])["Vehicles"].mean().reset_index().sort_values("DayOfWeek")
        daily["Junction"] = "Junction " + daily["Junction"].astype(str)

        fig_d = px.bar(daily, x="DayName", y="Vehicles", color="Junction", barmode="group",
                       category_orders={"DayName": ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]},
                       color_discrete_sequence=px.colors.qualitative.Vivid)
        apply_theme(fig_d)
        fig_d.update_layout(height=350, title="Average Traffic by Day of Week")
        st.plotly_chart(fig_d, use_container_width=True)

        col_wk1, col_wk2 = st.columns(2)
        with col_wk1:
            month_map = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                         7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
            monthly = dff.groupby("Month")["Vehicles"].mean().reset_index()
            monthly["MonthName"] = monthly["Month"].map(month_map)
            fig_m = px.line(monthly, x="MonthName", y="Vehicles", markers=True,
                            title="Monthly Average Traffic Volume",
                            color_discrete_sequence=["#818cf8"])
            apply_theme(fig_m)
            fig_m.update_layout(height=280)
            st.plotly_chart(fig_m, use_container_width=True)
        with col_wk2:
            wknd = dff.groupby(["Junction", "IsWeekend"])["Vehicles"].mean().reset_index()
            wknd["Period"] = wknd["IsWeekend"].map({0: "Weekday", 1: "Weekend"})
            wknd["Junction"] = "J" + wknd["Junction"].astype(str)
            fig_wk = px.bar(wknd, x="Junction", y="Vehicles", color="Period", barmode="group",
                            title="Weekday vs Weekend Comparison",
                            color_discrete_map={"Weekday":"#6366f1","Weekend":"#f59e0b"})
            apply_theme(fig_wk)
            fig_wk.update_layout(height=280)
            st.plotly_chart(fig_wk, use_container_width=True)

    with tab3:
        pivot_h = dff.groupby(["Junction", "Hour"])["Vehicles"].mean().reset_index()
        pivot_h = pivot_h.pivot(index="Junction", columns="Hour", values="Vehicles")

        fig_hm = go.Figure(go.Heatmap(
            z=pivot_h.values, x=list(pivot_h.columns),
            y=[f"Junction {i}" for i in pivot_h.index],
            colorscale=[[0,"#0f172a"],[0.33,"#1d4ed8"],[0.66,"#f59e0b"],[1,"#ef4444"]],
            hovertemplate="Junction %{y}<br>Hour %{x}:00<br>Avg Vehicles: %{z:.0f}<extra></extra>",
            colorbar=dict(title="Avg Vehicles", tickfont=dict(color="#94a3b8"))
        ))
        apply_theme(fig_hm)
        fig_hm.update_layout(height=320, title="Junction × Hour Congestion Heatmap")
        st.plotly_chart(fig_hm, use_container_width=True)

        # DayOfWeek × Hour
        pivot_dh = dff.groupby(["DayOfWeek", "Hour"])["Vehicles"].mean().reset_index()
        pivot_dh = pivot_dh.pivot(index="DayOfWeek", columns="Hour", values="Vehicles")
        dow_labels = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

        fig_dh = go.Figure(go.Heatmap(
            z=pivot_dh.values, x=list(pivot_dh.columns),
            y=[dow_labels[i] for i in pivot_dh.index],
            colorscale=[[0,"#0f172a"],[0.33,"#1d4ed8"],[0.66,"#f59e0b"],[1,"#ef4444"]],
            hovertemplate="Day: %{y}<br>Hour: %{x}:00<br>Avg Vehicles: %{z:.0f}<extra></extra>",
            colorbar=dict(title="Avg Vehicles", tickfont=dict(color="#94a3b8"))
        ))
        apply_theme(fig_dh)
        fig_dh.update_layout(height=300, title="Day of Week × Hour Congestion Heatmap")
        st.plotly_chart(fig_dh, use_container_width=True)

    with tab4:
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            fig_hist = px.histogram(dff, x="Vehicles", nbins=60, color_discrete_sequence=["#6366f1"],
                                    title="Distribution of Vehicle Counts",
                                    labels={"Vehicles": "Vehicles per Hour"})
            apply_theme(fig_hist)
            fig_hist.update_layout(height=300)
            st.plotly_chart(fig_hist, use_container_width=True)
        with col_d2:
            fig_box = px.box(dff, x="Junction", y="Vehicles",
                             title="Vehicle Count Distribution by Junction",
                             color_discrete_sequence=px.colors.qualitative.Vivid)
            apply_theme(fig_box)
            fig_box.update_layout(height=300)
            st.plotly_chart(fig_box, use_container_width=True)

        # Correlation matrix
        st.markdown('<div class="section-header">Feature Correlation Matrix</div>', unsafe_allow_html=True)
        num_cols = dff.select_dtypes(include=[np.number]).drop(columns=["ID","id"], errors="ignore").columns.tolist()
        corr = dff[num_cols].corr()
        fig_corr = px.imshow(corr, color_continuous_scale="RdBu", zmin=-1, zmax=1,
                              text_auto=".2f", title="Correlation Matrix",
                              color_continuous_midpoint=0)
        apply_theme(fig_corr)
        fig_corr.update_layout(height=400)
        st.plotly_chart(fig_corr, use_container_width=True)

    with tab5:
        st.markdown('<div class="section-header">Traffic Volume Over Time</div>', unsafe_allow_html=True)
        sel_j_ts = st.selectbox("Select Junction", sorted(dff["Junction"].unique()),
                                  format_func=lambda x: f"Junction {x}", key="ts_j")
        ts = dff[dff["Junction"] == sel_j_ts].set_index("DateTime")["Vehicles"]

        # Resample for readability
        resample_opt = st.radio("Granularity", ["Hourly", "Daily", "Weekly"], horizontal=True, key="ts_gran")
        rule_map = {"Hourly": "h", "Daily": "D", "Weekly": "W-SUN"}
        ts_res = ts.resample(rule_map[resample_opt]).mean().dropna().reset_index()

        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=ts_res["DateTime"], y=ts_res["Vehicles"],
                                     mode="lines", line=dict(color="#6366f1", width=1.5),
                                     name="Traffic Volume", fill="tozeroy",
                                     fillcolor="rgba(99,102,241,0.1)"))
        apply_theme(fig_ts)
        fig_ts.update_layout(height=380, title=f"Junction {sel_j_ts} — {resample_opt} Traffic Volume",
                              xaxis_title="Date", yaxis_title="Avg Vehicles")
        st.plotly_chart(fig_ts, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Performance":
    st.markdown("# 🤖 Model Performance & Comparison")
    st.markdown("<p style='color:#64748b'>Comprehensive evaluation of all trained ML models.</p>", unsafe_allow_html=True)

    if not model_exists():
        st.markdown('<div class="warn-box">⚠️ No model artifacts found. Run <code>python -m src.train</code> first.</div>', unsafe_allow_html=True)
        st.stop()

    meta = load_metadata()
    comp = load_comparison()

    if meta is None or comp is None:
        st.error("Model metadata or comparison file not found.")
        st.stop()

    best_name = meta.get("best_model_name", "–")

    # ── Model Comparison Table ────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Model Comparison</div>', unsafe_allow_html=True)

    comp_styled = comp.copy()
    comp_styled["Best"] = comp_styled["Model"].apply(lambda x: "🏆" if x == best_name else "")
    comp_styled = comp_styled[["Best", "Model", "MAE", "RMSE", "R2"]].rename(columns={"R2": "R² Score"})
    comp_styled["MAE"] = comp_styled["MAE"].map("{:.3f}".format)
    comp_styled["RMSE"] = comp_styled["RMSE"].map("{:.3f}".format)
    comp_styled["R² Score"] = comp_styled["R² Score"].map("{:.4f}".format)
    st.dataframe(comp_styled, use_container_width=True, hide_index=True)

    # ── Bar chart comparison ───────────────────────────────────────────────────
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        fig_r2 = go.Figure(go.Bar(
            x=comp["Model"], y=comp["R2"],
            marker_color=["#fbbf24" if m == best_name else "#6366f1" for m in comp["Model"]],
            text=[f"{v:.4f}" for v in comp["R2"]], textposition="auto",
            hovertemplate="%{x}<br>R² Score: %{y:.4f}<extra></extra>"
        ))
        apply_theme(fig_r2)
        fig_r2.update_layout(height=320, title="R² Score by Model", yaxis_title="R² Score",
                              yaxis_range=[max(0, comp["R2"].min() - 0.05), 1.0])
        st.plotly_chart(fig_r2, use_container_width=True)

    with col_m2:
        metrics_long = comp.melt(id_vars="Model", value_vars=["MAE", "RMSE"], var_name="Metric", value_name="Value")
        fig_err = px.bar(metrics_long, x="Model", y="Value", color="Metric", barmode="group",
                         title="MAE & RMSE by Model",
                         color_discrete_map={"MAE": "#6366f1", "RMSE": "#f59e0b"})
        apply_theme(fig_err)
        fig_err.update_layout(height=320)
        st.plotly_chart(fig_err, use_container_width=True)

    # ── Feature Importance ────────────────────────────────────────────────────
    if meta.get("feature_importances"):
        st.markdown('<div class="section-header">🔍 Feature Importance — {}</div>'.format(best_name), unsafe_allow_html=True)
        fi = pd.DataFrame(list(meta["feature_importances"].items()), columns=["Feature", "Importance"])
        fi = fi.sort_values("Importance", ascending=True).tail(15)

        fig_fi = go.Figure(go.Bar(
            x=fi["Importance"], y=fi["Feature"], orientation="h",
            marker=dict(
                color=fi["Importance"],
                colorscale=[[0,"#1d4ed8"],[0.5,"#6366f1"],[1,"#818cf8"]],
                showscale=False
            ),
            hovertemplate="%{y}<br>Importance: %{x:.4f}<extra></extra>"
        ))
        apply_theme(fig_fi)
        fig_fi.update_layout(height=420, title=f"Top Feature Importances — {best_name}",
                              xaxis_title="Relative Importance", yaxis_title="")
        st.plotly_chart(fig_fi, use_container_width=True)

    # ── Training Summary ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📋 Training Summary</div>', unsafe_allow_html=True)
    ps = meta.get("preprocessing_summary", {})
    ts_info = [
        ("Total Records", f"{ps.get('raw_shape', [0])[0]:,}"),
        ("Cleaned Records", f"{ps.get('clean_shape', [0])[0]:,}"),
        ("Duplicates Removed", str(ps.get("duplicates_removed", 0))),
        ("Missing Values (Raw)", str(ps.get("missing_values_raw", 0))),
        ("Train Period", meta.get("train_date_range", {}).get("min", "–")[:10] + " → " + meta.get("train_date_range", {}).get("max", "–")[:10]),
        ("Test Period", meta.get("test_date_range", {}).get("min", "–")[:10] + " → " + meta.get("test_date_range", {}).get("max", "–")[:10]),
        ("Junctions", str(ps.get("junctions", []))),
        ("Best Model", best_name),
    ]
    col_s1, col_s2 = st.columns(2)
    for i, (k, v) in enumerate(ts_info):
        with (col_s1 if i % 2 == 0 else col_s2):
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;padding:6px 0;'
                f'border-bottom:1px solid rgba(255,255,255,0.05)">'
                f'<span style="color:#94a3b8;font-size:0.85rem">{k}</span>'
                f'<span style="color:#e2e8f0;font-weight:600;font-size:0.85rem">{v}</span></div>',
                unsafe_allow_html=True
            )

    # ── Run Training button ───────────────────────────────────────────────────
    st.markdown("")
    st.markdown('<div class="section-header">🔄 Retrain Models</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">To retrain all models from scratch, run in your terminal:<br>'
                '<code style="background:rgba(99,102,241,0.15);padding:2px 8px;border-radius:4px;display:inline-block;margin-top:6px">'
                'python -m src.train</code></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.markdown("# ℹ️ About This Project")

    st.markdown("""
<div class="info-box">
<strong>🚦 Traffic Congestion Prediction & Smart Traffic Analytics System</strong><br>
An end-to-end machine learning pipeline for predicting hourly traffic volume and generating real-time congestion risk scores.
</div>
""", unsafe_allow_html=True)

    st.markdown("")
    st.markdown("## 🏗️ Project Architecture")

    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.markdown("""
**Data Pipeline**
- `src/data_loader.py` — Auto-download & validation
- `src/preprocessing.py` — Cleaning, deduplication, interpolation
- `src/feature_engineering.py` — Temporal, cyclical & historical features

**ML Pipeline**
- `src/train.py` — Multi-model training & selection
- `src/evaluate.py` — MAE, RMSE, R² metrics
- `src/predict.py` — Real-time inference engine
- `src/risk_scoring.py` — 0-100 risk score + congestion level
""")

    with col_a2:
        st.markdown("""
**Dashboard**
- `app.py` — Streamlit interactive dashboard
- `src/visualization.py` — Plotly & Matplotlib charts

**Models Compared**
- Linear Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost (typically best)

**Dataset**: Kaggle — Traffic Prediction Dataset (fedesoriano)
""")

    st.markdown("## 🚀 Quick Start")
    st.code("""# 1. Install dependencies
pip install -r requirements.txt

# 2. Add dataset
# Download traffic.csv from Kaggle and place at:
#   data/raw/traffic.csv

# 3. Train models
python -m src.train

# 4. Launch dashboard
streamlit run app.py
""", language="bash")

    st.markdown("## 📊 Features")
    features = [
        "✅ Automatic dataset download (falls back to Kaggle instructions)",
        "✅ Chronological train/test split to prevent data leakage",
        "✅ Historical traffic profiling per junction × hour × day-of-week",
        "✅ Cyclical feature encoding (sin/cos) for temporal signals",
        "✅ 5-model comparison with automated best-model selection",
        "✅ Congestion risk score 0–100 per junction",
        "✅ Low / Medium / High congestion categorization",
        "✅ 24-hour forecast chart for any date & junction",
        "✅ All-junction comparison at selected time",
        "✅ Interactive EDA with heatmaps, distributions, time series",
        "✅ Feature importance visualization",
        "✅ Fully dark-themed professional Streamlit dashboard",
    ]
    for f in features:
        st.markdown(f)
