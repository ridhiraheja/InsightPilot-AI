import json
import re
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from google import genai
from sklearn.ensemble import IsolationForest


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="InsightPilot AI | Business Intelligence",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DESIGN SYSTEM
# ============================================================

st.markdown(
    """
    <style>
    /* ---------- Global ---------- */
    .stApp {
        background: #f7f9fc;
    }

    .main .block-container {
        max-width: 1500px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    [data-testid="stHeader"] {
        background: rgba(247, 249, 252, 0.92);
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: #0b1220;
        border-right: 1px solid #1f2937;
    }

    section[data-testid="stSidebar"] * {
        color: #e5e7eb;
    }

    section[data-testid="stSidebar"] .stMarkdown p {
        color: #aeb8c8;
    }

    .sidebar-brand {
        padding: 0.3rem 0 1.2rem 0;
    }

    .sidebar-brand-title {
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #ffffff;
    }

    .sidebar-brand-sub {
        font-size: 0.78rem;
        color: #94a3b8;
        margin-top: 0.2rem;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 0.65rem;
        border-radius: 999px;
        font-size: 0.74rem;
        font-weight: 700;
        margin-top: 0.5rem;
        border: 1px solid rgba(255,255,255,0.12);
    }

    .status-good { background: rgba(34,197,94,0.12); color: #86efac; }
    .status-warn { background: rgba(245,158,11,0.12); color: #fcd34d; }

    /* ---------- Hero ---------- */
    .hero {
        position: relative;
        overflow: hidden;
        border-radius: 24px;
        padding: 2.1rem 2.2rem;
        margin-bottom: 1.35rem;
        background: linear-gradient(135deg, #111827 0%, #172554 52%, #1e3a8a 100%);
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.16);
        color: white;
    }

    .hero:after {
        content: "";
        position: absolute;
        width: 280px;
        height: 280px;
        right: -90px;
        top: -110px;
        border-radius: 50%;
        background: rgba(255,255,255,0.08);
    }

    .hero-kicker {
        position: relative;
        z-index: 1;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #bfdbfe;
        margin-bottom: 0.55rem;
    }

    .hero-title {
        position: relative;
        z-index: 1;
        font-size: clamp(2rem, 4vw, 3.25rem);
        line-height: 1.03;
        font-weight: 850;
        letter-spacing: -0.045em;
        margin: 0;
    }

    .hero-subtitle {
        position: relative;
        z-index: 1;
        max-width: 800px;
        color: #dbeafe;
        font-size: 1.02rem;
        line-height: 1.6;
        margin-top: 0.75rem;
        margin-bottom: 0;
    }

    .hero-chip-row {
        position: relative;
        z-index: 1;
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 1.15rem;
    }

    .hero-chip {
        padding: 0.38rem 0.7rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.10);
        border: 1px solid rgba(255,255,255,0.13);
        color: #e0f2fe;
        font-size: 0.78rem;
        font-weight: 650;
    }

    /* ---------- Section headers ---------- */
    .section-heading {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        margin: 1.35rem 0 0.75rem 0;
    }

    .section-icon {
        width: 34px;
        height: 34px;
        display: grid;
        place-items: center;
        border-radius: 10px;
        background: #e0e7ff;
        color: #3730a3;
        font-size: 1rem;
    }

    .section-title {
        font-size: 1.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #111827;
    }

    .section-subtitle {
        color: #64748b;
        font-size: 0.84rem;
        margin: -0.35rem 0 0.8rem 2.75rem;
    }

    /* ---------- Cards ---------- */
    .soft-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 1rem 1.05rem;
        box-shadow: 0 5px 18px rgba(15,23,42,0.045);
    }

    .mini-label {
        color: #64748b;
        font-size: 0.72rem;
        font-weight: 750;
        letter-spacing: 0.07em;
        text-transform: uppercase;
    }

    .mini-value {
        color: #0f172a;
        font-size: 1.55rem;
        font-weight: 850;
        margin-top: 0.25rem;
    }

    .mini-note {
        color: #64748b;
        font-size: 0.76rem;
        margin-top: 0.15rem;
    }

    .health-card {
        border-radius: 16px;
        padding: 1rem 1.1rem;
        background: #ffffff;
        border: 1px solid #e5e7eb;
        box-shadow: 0 5px 18px rgba(15,23,42,0.045);
    }

    .health-score {
        font-size: 2.15rem;
        font-weight: 900;
        letter-spacing: -0.04em;
    }

    .health-bar {
        height: 8px;
        border-radius: 999px;
        background: #e2e8f0;
        overflow: hidden;
        margin-top: 0.6rem;
    }

    .health-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #4f46e5, #2563eb);
    }

    .insight-card {
        border: 1px solid #e2e8f0;
        border-left: 4px solid #4f46e5;
        background: #ffffff;
        border-radius: 12px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.65rem;
        box-shadow: 0 4px 14px rgba(15,23,42,0.035);
    }

    .insight-title {
        font-weight: 800;
        color: #111827;
        font-size: 0.88rem;
    }

    .insight-text {
        color: #475569;
        font-size: 0.84rem;
        line-height: 1.5;
        margin-top: 0.22rem;
    }

    .ai-card {
        border: 1px solid #c7d2fe;
        border-radius: 20px;
        padding: 1.2rem;
        background: linear-gradient(145deg, #ffffff 0%, #f8faff 100%);
        box-shadow: 0 10px 28px rgba(79,70,229,0.08);
    }

    .ai-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.3rem 0.62rem;
        border-radius: 999px;
        background: #eef2ff;
        color: #4338ca;
        font-size: 0.73rem;
        font-weight: 800;
    }

    .question-chip {
        background: #ffffff;
        border: 1px solid #dbe2ea;
        border-radius: 10px;
        padding: 0.55rem 0.7rem;
        color: #334155;
        font-size: 0.8rem;
    }

    .empty-state {
        text-align: center;
        padding: 4rem 1rem;
        border: 1px dashed #cbd5e1;
        border-radius: 20px;
        background: #ffffff;
    }

    .empty-icon {
        font-size: 2.7rem;
        margin-bottom: 0.5rem;
    }

    .empty-title {
        font-size: 1.2rem;
        font-weight: 800;
        color: #0f172a;
    }

    .empty-text {
        max-width: 560px;
        margin: 0.35rem auto 0;
        color: #64748b;
        line-height: 1.55;
    }

    /* ---------- Streamlit controls ---------- */
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 15px;
        padding: 0.85rem 1rem;
        box-shadow: 0 5px 18px rgba(15,23,42,0.04);
    }

    div[data-testid="stMetricLabel"] {
        color: #64748b;
    }

    div[data-testid="stMetricValue"] {
        color: #0f172a;
        font-weight: 850;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 700;
        min-height: 2.55rem;
    }

    .stTextArea textarea {
        border-radius: 12px;
    }

    .stSelectbox [data-baseweb="select"] > div,
    .stMultiSelect [data-baseweb="select"] > div {
        border-radius: 10px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.35rem;
        border-bottom: 1px solid #e2e8f0;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 9px 9px 0 0;
        padding: 0.6rem 1rem;
        font-weight: 700;
    }

    .stTabs [aria-selected="true"] {
        color: #3730a3;
    }

    /* Keep tables visually clean */
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #e5e7eb;
    }

    /* Mobile */
    @media (max-width: 768px) {
        .main .block-container { padding: 1rem 0.75rem 2rem; }
        .hero { padding: 1.4rem; border-radius: 18px; }
        .hero-title { font-size: 2rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================


def section_header(icon, title, subtitle=None):
    st.markdown(
        f"""
        <div class="section-heading">
            <div class="section-icon">{icon}</div>
            <div class="section-title">{title}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if subtitle:
        st.markdown(
            f'<div class="section-subtitle">{subtitle}</div>',
            unsafe_allow_html=True,
        )


def fmt_num(value):
    try:
        value = float(value)
        if abs(value) >= 1_000_000:
            return f"{value / 1_000_000:.2f}M"
        if abs(value) >= 1_000:
            return f"{value / 1_000:.2f}K"
        return f"{value:,.2f}"
    except Exception:
        return str(value)


def make_chart_layout(fig, height=390):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=48, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Arial, sans-serif", color="#334155"),
        title=dict(font=dict(size=16, color="#0f172a")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    fig.update_xaxes(showgrid=False, linecolor="#e2e8f0")
    fig.update_yaxes(gridcolor="#eef2f7", zeroline=False)
    return fig


def safe_percent_change(first_value, last_value):
    if first_value == 0:
        return None
    return ((last_value - first_value) / abs(first_value)) * 100


def data_health_score(dataframe):
    total_cells = max(dataframe.shape[0] * dataframe.shape[1], 1)
    missing_ratio = dataframe.isnull().sum().sum() / total_cells
    duplicate_ratio = dataframe.duplicated().sum() / max(len(dataframe), 1)
    score = 100 - (missing_ratio * 70) - (duplicate_ratio * 30)
    return int(max(0, min(100, round(score))))


def health_label(score):
    if score >= 95:
        return "Excellent"
    if score >= 85:
        return "Healthy"
    if score >= 70:
        return "Needs review"
    return "Needs attention"


# ============================================================
# GEMINI CLIENT
# ============================================================

client = None
try:
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
except Exception:
    client = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">✦ InsightPilot AI</div>
            <div class="sidebar-brand-sub">Explainable business intelligence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if client is not None:
        st.markdown(
            '<div class="status-pill status-good">● AI engine configured</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-pill status-warn">● AI key not configured</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("**Workflow**")
    st.markdown("1. Upload business data")
    st.markdown("2. Review data health")
    st.markdown("3. Explore KPIs & trends")
    st.markdown("4. Ask the AI decision engine")

    st.divider()
    st.markdown("**Built for**")
    st.caption("Business analysts • Managers • Data teams")
    st.caption("CSV / XLSX / XLS")


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">AI DECISION ENGINE · PS-04</div>
        <div class="hero-title">InsightPilot AI</div>
        <p class="hero-subtitle">
            Turn raw business data into verified evidence, interactive analytics,
            and explainable decisions — without losing sight of the numbers.
        </p>
        <div class="hero-chip-row">
            <span class="hero-chip">✓ Verified evidence</span>
            <span class="hero-chip">✓ KPI intelligence</span>
            <span class="hero-chip">✓ Trend detection</span>
            <span class="hero-chip">✓ Anomaly detection</span>
            <span class="hero-chip">✓ AI recommendations</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FILE UPLOAD
# ============================================================

section_header("↑", "Start with your business data", "Upload a dataset and InsightPilot will build the analysis workspace automatically.")

upload_col, info_col = st.columns([1.55, 1], gap="large")

with upload_col:
    uploaded_file = st.file_uploader(
        "Drop a CSV or Excel file here",
        type=["csv", "xlsx", "xls"],
        help="Supported formats: CSV, XLSX and XLS",
    )

with info_col:
    st.markdown(
        """
        <div class="soft-card">
            <div class="mini-label">Analysis pipeline</div>
            <div style="font-size:0.9rem;font-weight:750;color:#0f172a;margin-top:0.4rem;">
                Upload → Profile → Explore → Ask → Decide
            </div>
            <div class="mini-note">
                Calculations are performed with Python first. The AI layer explains the verified evidence.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# EMPTY STATE
# ============================================================

if uploaded_file is None:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-icon">📊</div>
            <div class="empty-title">Your decision workspace is ready</div>
            <div class="empty-text">
                Upload your business dataset above to unlock the executive overview,
                data quality checks, interactive analytics, anomaly detection,
                and explainable AI recommendations.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_header("✦", "What InsightPilot analyzes", "Everything below is generated from the uploaded dataset.")
    cards = st.columns(5, gap="small")
    features = [
        ("01", "Data profile", "Rows, columns, types and data health"),
        ("02", "KPI analysis", "Summary statistics and business metrics"),
        ("03", "Patterns", "Categories, trends and correlations"),
        ("04", "Risk signals", "Potential anomalies and missing data"),
        ("05", "AI decisions", "Evidence-backed interpretation and actions"),
    ]
    for card, (num, title, desc) in zip(cards, features):
        with card:
            st.markdown(
                f"""
                <div class="soft-card" style="height:150px;">
                    <div class="mini-label">{num}</div>
                    <div style="font-weight:800;color:#0f172a;margin-top:0.45rem;">{title}</div>
                    <div class="mini-note">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.stop()


# ============================================================
# READ FILE
# ============================================================

try:
    if uploaded_file.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
except Exception as exc:
    st.error(f"Could not read the uploaded file: {exc}")
    st.stop()

if df.empty:
    st.warning("The uploaded file does not contain any usable rows.")
    st.stop()

# Basic cleaning only; do not silently alter business values.
df.columns = df.columns.astype(str).str.strip()
df = df.dropna(how="all").dropna(axis=1, how="all")

if df.empty or df.shape[1] == 0:
    st.warning("No usable data remains after removing completely empty rows and columns.")
    st.stop()


# ============================================================
# DATA TYPE DETECTION
# ============================================================

numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
categorical_columns = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

date_columns = []
for column in df.columns:
    if column in numeric_columns:
        continue
    try:
        converted = pd.to_datetime(df[column], errors="coerce")
        if converted.notna().mean() >= 0.80:
            date_columns.append(column)
    except Exception:
        pass

categorical_columns = [c for c in categorical_columns if c not in date_columns]


# ============================================================
# CORE DATA HEALTH
# ============================================================

total_missing = int(df.isnull().sum().sum())
duplicate_rows = int(df.duplicated().sum())
health_score = data_health_score(df)


# ============================================================
# VERIFIED BUSINESS EVIDENCE
# ============================================================

@st.cache_data(show_spinner=False)
def generate_business_evidence(dataframe):
    evidence = {}

    evidence["dataset_overview"] = {
        "rows": int(dataframe.shape[0]),
        "columns": int(dataframe.shape[1]),
        "missing_values": int(dataframe.isnull().sum().sum()),
        "duplicate_rows": int(dataframe.duplicated().sum()),
    }

    numeric_cols = dataframe.select_dtypes(include=np.number).columns.tolist()
    numeric_statistics = {}
    for column in numeric_cols:
        series = dataframe[column].dropna()
        if len(series) == 0:
            continue
        numeric_statistics[column] = {
            "count": int(series.count()),
            "mean": round(float(series.mean()), 4),
            "median": round(float(series.median()), 4),
            "minimum": round(float(series.min()), 4),
            "maximum": round(float(series.max()), 4),
            "standard_deviation": round(float(series.std()), 4),
        }
    evidence["numeric_statistics"] = numeric_statistics

    categorical_cols = dataframe.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    category_statistics = {}
    for category_column in categorical_cols[:8]:
        unique_count = dataframe[category_column].nunique(dropna=True)
        if unique_count > 50:
            continue
        category_statistics[category_column] = {}
        for numeric_column in numeric_cols[:8]:
            try:
                grouped = (
                    dataframe.groupby(category_column)[numeric_column]
                    .agg(mean="mean", sum="sum", count="count")
                    .sort_values("mean", ascending=False)
                    .head(10)
                    .reset_index()
                )
                rows = []
                for _, row in grouped.iterrows():
                    rows.append(
                        {
                            "category": str(row[category_column]),
                            "average": round(float(row["mean"]), 4),
                            "total": round(float(row["sum"]), 4),
                            "count": int(row["count"]),
                        }
                    )
                category_statistics[category_column][numeric_column] = rows
            except Exception:
                continue
    evidence["category_statistics"] = category_statistics

    missing_statistics = {}
    for column in dataframe.columns:
        missing_count = int(dataframe[column].isnull().sum())
        if missing_count > 0:
            missing_statistics[column] = {
                "missing_count": missing_count,
                "missing_percentage": round(float(dataframe[column].isnull().mean() * 100), 2),
            }
    evidence["missing_values"] = missing_statistics

    correlation_pairs = []
    if len(numeric_cols) >= 2:
        correlation_matrix = dataframe[numeric_cols].corr()
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                column_a = numeric_cols[i]
                column_b = numeric_cols[j]
                correlation = correlation_matrix.loc[column_a, column_b]
                if pd.notna(correlation):
                    correlation_pairs.append(
                        {
                            "column_1": column_a,
                            "column_2": column_b,
                            "correlation": round(float(correlation), 4),
                        }
                    )
        correlation_pairs = sorted(
            correlation_pairs,
            key=lambda x: abs(x["correlation"]),
            reverse=True,
        )[:15]
    evidence["strongest_correlations"] = correlation_pairs

    trend_statistics = {}
    detected_dates = []
    for column in dataframe.columns:
        if column in numeric_cols:
            continue
        try:
            converted = pd.to_datetime(dataframe[column], errors="coerce")
            if converted.notna().mean() >= 0.80:
                detected_dates.append(column)
        except Exception:
            pass

    for date_column in detected_dates[:3]:
        converted = pd.to_datetime(dataframe[date_column], errors="coerce")
        temp = dataframe.copy()
        temp["_detected_date"] = converted
        temp = temp.dropna(subset=["_detected_date"])
        if len(temp) < 2:
            continue
        trend_statistics[date_column] = {}
        for numeric_column in numeric_cols[:8]:
            try:
                grouped = (
                    temp.groupby("_detected_date")[numeric_column]
                    .sum()
                    .sort_index()
                )
                if len(grouped) >= 2:
                    first_value = float(grouped.iloc[0])
                    last_value = float(grouped.iloc[-1])
                    change = safe_percent_change(first_value, last_value)
                    trend_statistics[date_column][numeric_column] = {
                        "start_date": str(grouped.index[0].date()),
                        "end_date": str(grouped.index[-1].date()),
                        "starting_value": round(first_value, 4),
                        "ending_value": round(last_value, 4),
                        "percentage_change": round(change, 2) if change is not None else None,
                    }
            except Exception:
                continue
    evidence["trend_statistics"] = trend_statistics

    anomaly_statistics = {}
    for numeric_column in numeric_cols[:8]:
        series = dataframe[numeric_column].dropna()
        if len(series) < 20:
            continue
        try:
            model = IsolationForest(contamination=0.05, random_state=42)
            predictions = model.fit_predict(series.to_frame())
            anomaly_count = int((predictions == -1).sum())
            anomaly_statistics[numeric_column] = {
                "anomaly_count": anomaly_count,
                "anomaly_percentage": round(anomaly_count / len(series) * 100, 2),
            }
        except Exception:
            continue
    evidence["anomalies"] = anomaly_statistics
    return evidence


with st.spinner("Preparing verified business evidence..."):
    business_evidence = generate_business_evidence(df)


# ============================================================
# TOP STATUS BAR
# ============================================================

st.markdown(
    f"""
    <div style="display:flex;flex-wrap:wrap;gap:0.5rem;margin-bottom:0.8rem;">
        <span class="status-pill" style="background:#ffffff;color:#334155;border:1px solid #e2e8f0;">📄 {uploaded_file.name}</span>
        <span class="status-pill" style="background:#ffffff;color:#334155;border:1px solid #e2e8f0;">{df.shape[0]:,} rows</span>
        <span class="status-pill" style="background:#ffffff;color:#334155;border:1px solid #e2e8f0;">{df.shape[1]:,} columns</span>
        <span class="status-pill" style="background:#ffffff;color:#334155;border:1px solid #e2e8f0;">Health {health_score}/100 · {health_label(health_score)}</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MAIN TABS
# ============================================================

tab_overview, tab_explore, tab_ai, tab_data = st.tabs(
    ["◈ Executive Overview", "◌ Explore & Detect", "✦ AI Decision Engine", "▦ Data & Quality"]
)


# ============================================================
# TAB 1 — EXECUTIVE OVERVIEW
# ============================================================

with tab_overview:
    section_header("◈", "Executive overview", "A fast, decision-focused snapshot of the uploaded business data.")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4, gap="medium")
    with kpi1:
        st.metric("Rows analyzed", f"{df.shape[0]:,}")
    with kpi2:
        st.metric("Columns detected", f"{df.shape[1]:,}")
    with kpi3:
        st.metric("Missing values", f"{total_missing:,}")
    with kpi4:
        st.metric("Duplicate rows", f"{duplicate_rows:,}")

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    left, right = st.columns([1.1, 1], gap="large")

    with left:
        score_color = "#16a34a" if health_score >= 85 else "#d97706" if health_score >= 70 else "#dc2626"
        st.markdown(
            f"""
            <div class="health-card">
                <div class="mini-label">Data quality signal</div>
                <div style="display:flex;align-items:end;gap:0.6rem;margin-top:0.25rem;">
                    <div class="health-score" style="color:{score_color};">{health_score}</div>
                    <div style="color:#64748b;font-size:0.85rem;padding-bottom:0.35rem;">/ 100 · {health_label(health_score)}</div>
                </div>
                <div class="health-bar"><div class="health-fill" style="width:{health_score}%;"></div></div>
                <div class="mini-note" style="margin-top:0.65rem;">
                    Based on missing-cell and duplicate-row signals. Use the Data & Quality tab for column-level detail.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            f"""
            <div class="health-card">
                <div class="mini-label">Detected structure</div>
                <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:0.8rem;margin-top:0.65rem;">
                    <div><div class="mini-value">{len(numeric_columns)}</div><div class="mini-note">Numeric</div></div>
                    <div><div class="mini-value">{len(categorical_columns)}</div><div class="mini-note">Categorical</div></div>
                    <div><div class="mini-value">{len(date_columns)}</div><div class="mini-note">Date</div></div>
                </div>
                <div class="mini-note" style="margin-top:0.8rem;">
                    InsightPilot uses this detected structure to choose relevant analyses automatically.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    section_header("💡", "Automatic signals", "Deterministic observations generated directly from your dataset.")

    automatic_insights = []
    if total_missing > 0:
        automatic_insights.append(("Data quality", f"The dataset contains {total_missing:,} missing values."))
    else:
        automatic_insights.append(("Data quality", "No missing values were detected."))

    if duplicate_rows > 0:
        automatic_insights.append(("Duplicates", f"The dataset contains {duplicate_rows:,} duplicate rows."))
    else:
        automatic_insights.append(("Duplicates", "No duplicate rows were detected."))

    if numeric_columns:
        averages = {c: df[c].mean() for c in numeric_columns if pd.notna(df[c].mean())}
        if averages:
            highest = max(averages, key=averages.get)
            automatic_insights.append(("Numeric signal", f"{highest} has the highest average value among numeric columns: {fmt_num(averages[highest])}."))

    if business_evidence["anomalies"]:
        highest_anomaly_column = max(
            business_evidence["anomalies"],
            key=lambda c: business_evidence["anomalies"][c]["anomaly_percentage"],
        )
        anomaly_percentage = business_evidence["anomalies"][highest_anomaly_column]["anomaly_percentage"]
        automatic_insights.append(("Risk signal", f"{highest_anomaly_column} has {anomaly_percentage:.2f}% potential anomalies according to Isolation Forest."))

    insight_cols = st.columns(2, gap="medium")
    for i, (title, text) in enumerate(automatic_insights):
        with insight_cols[i % 2]:
            st.markdown(
                f"""
                <div class="insight-card">
                    <div class="insight-title">{title}</div>
                    <div class="insight-text">{text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if numeric_columns:
        section_header("📊", "KPI snapshot", "Average values across the first available numeric measures.")
        kpi_cols = numeric_columns[:4]
        cards = st.columns(len(kpi_cols), gap="medium")
        for card, column in zip(cards, kpi_cols):
            value = df[column].mean()
            with card:
                st.markdown(
                    f"""
                    <div class="soft-card">
                        <div class="mini-label">Average {column}</div>
                        <div class="mini-value">{fmt_num(value)}</div>
                        <div class="mini-note">Median: {fmt_num(df[column].median())}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ============================================================
# TAB 2 — EXPLORE & DETECT
# ============================================================

with tab_explore:
    section_header("◌", "Explore & detect", "Interactive analysis for distributions, categories, trends, anomalies and relationships.")

    if numeric_columns:
        st.markdown("#### Numeric distribution")
        selected_numeric = st.selectbox("Choose a numeric measure", numeric_columns, key="explore_numeric")
        chart_col, stats_col = st.columns([1.65, 1], gap="large")

        with chart_col:
            fig_hist = px.histogram(
                df,
                x=selected_numeric,
                marginal="box",
                nbins=30,
                title=f"Distribution of {selected_numeric}",
            )
            fig_hist = make_chart_layout(fig_hist, 410)
            st.plotly_chart(fig_hist, use_container_width=True)

        with stats_col:
            summary = df[selected_numeric].describe().round(2)
            summary_df = pd.DataFrame({"Statistic": summary.index, "Value": summary.values})
            st.dataframe(summary_df, use_container_width=True, hide_index=True, height=410)

    if categorical_columns and numeric_columns:
        section_header("▥", "Category comparison", "Compare average or total performance across business categories.")
        control1, control2, control3 = st.columns([1, 1, 0.7], gap="medium")
        with control1:
            category_column = st.selectbox("Category", categorical_columns, key="category_column")
        with control2:
            value_column = st.selectbox("Measure", numeric_columns, key="category_value")
        with control3:
            category_metric = st.selectbox("Metric", ["Average", "Total"], key="category_metric")

        category_limit = min(15, max(5, int(df[category_column].nunique(dropna=True))))
        category_summary = (
            df.groupby(category_column)[value_column]
            .agg(mean="mean", sum="sum", count="count")
            .sort_values("mean", ascending=False)
            .head(category_limit)
            .reset_index()
        )
        chart_metric = "mean" if category_metric == "Average" else "sum"
        y_title = f"{category_metric} {value_column}"
        fig_category = px.bar(
            category_summary,
            x=category_column,
            y=chart_metric,
            title=f"{y_title} by {category_column}",
            text_auto=".2f",
        )
        fig_category = make_chart_layout(fig_category, 420)
        st.plotly_chart(fig_category, use_container_width=True)

        with st.expander("View category statistics"):
            st.dataframe(category_summary, use_container_width=True, hide_index=True)

    if date_columns and numeric_columns:
        section_header("◷", "Trend analysis", "Track a numeric measure over a detected date field.")
        trend_controls = st.columns([1, 1], gap="medium")
        with trend_controls[0]:
            trend_date = st.selectbox("Date column", date_columns, key="trend_date")
        with trend_controls[1]:
            trend_value = st.selectbox("Value column", numeric_columns, key="trend_value")

        trend_df = df[[trend_date, trend_value]].copy()
        trend_df[trend_date] = pd.to_datetime(trend_df[trend_date], errors="coerce")
        trend_df = trend_df.dropna(subset=[trend_date, trend_value])
        trend_df = trend_df.groupby(trend_date)[trend_value].sum().reset_index().sort_values(trend_date)

        if len(trend_df) > 1:
            fig_trend = px.line(
                trend_df,
                x=trend_date,
                y=trend_value,
                markers=True,
                title=f"{trend_value} over time",
            )
            fig_trend = make_chart_layout(fig_trend, 420)
            st.plotly_chart(fig_trend, use_container_width=True)
            change = safe_percent_change(float(trend_df[trend_value].iloc[0]), float(trend_df[trend_value].iloc[-1]))
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Start", fmt_num(trend_df[trend_value].iloc[0]))
            with c2:
                st.metric("End", fmt_num(trend_df[trend_value].iloc[-1]))
            with c3:
                st.metric("Overall change", f"{change:.2f}%" if change is not None else "N/A")
        else:
            st.info("Not enough valid date/value pairs to build a trend.")

    if numeric_columns and len(df) >= 20:
        section_header("⚠", "Anomaly detection", "Potentially unusual records identified with Isolation Forest.")
        anomaly_column = st.selectbox("Measure to inspect", numeric_columns, key="anomaly_column")
        anomaly_data = df[[anomaly_column]].dropna().copy()
        if len(anomaly_data) >= 20:
            model = IsolationForest(contamination=0.05, random_state=42)
            predictions = model.fit_predict(anomaly_data[[anomaly_column]])
            anomaly_data["Status"] = np.where(predictions == -1, "Potential anomaly", "Normal")
            anomaly_count = int((predictions == -1).sum())
            normal_count = int((predictions == 1).sum())

            a1, a2, a3 = st.columns(3)
            with a1:
                st.metric("Normal records", f"{normal_count:,}")
            with a2:
                st.metric("Potential anomalies", f"{anomaly_count:,}")
            with a3:
                st.metric("Anomaly rate", f"{anomaly_count / len(anomaly_data) * 100:.2f}%")

            fig_anomaly = px.scatter(
                anomaly_data.reset_index(),
                x="index",
                y=anomaly_column,
                color="Status",
                title=f"Potential anomalies — {anomaly_column}",
                color_discrete_map={"Normal": "#94a3b8", "Potential anomaly": "#dc2626"},
            )
            fig_anomaly = make_chart_layout(fig_anomaly, 420)
            st.plotly_chart(fig_anomaly, use_container_width=True)
        else:
            st.info("Not enough valid records for anomaly detection.")

    if len(numeric_columns) >= 2:
        section_header("⌁", "Correlation map", "Correlation shows relationships between numeric measures; it does not prove causation.")
        correlation_matrix = df[numeric_columns].corr()
        fig_corr = px.imshow(
            correlation_matrix,
            text_auto=".2f",
            aspect="auto",
            title="Numeric feature correlation",
            color_continuous_scale="Blues",
            zmin=-1,
            zmax=1,
        )
        fig_corr = make_chart_layout(fig_corr, 480)
        st.plotly_chart(fig_corr, use_container_width=True)

        pairs = business_evidence["strongest_correlations"][:5]
        if pairs:
            st.markdown("**Strongest relationships**")
            pair_cols = st.columns(min(5, len(pairs)), gap="small")
            for card, pair in zip(pair_cols, pairs):
                with card:
                    st.markdown(
                        f"""
                        <div class="soft-card">
                            <div class="mini-label">Correlation</div>
                            <div style="font-weight:800;color:#0f172a;margin-top:0.3rem;">{pair['column_1']} ↔ {pair['column_2']}</div>
                            <div class="mini-value" style="font-size:1.25rem;">{pair['correlation']:.2f}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


# ============================================================
# AI DECISION ENGINE
# ============================================================


def run_ai_analysis(user_question, evidence):
    if client is None:
        return None, "Gemini API is not configured. Add GEMINI_API_KEY to Streamlit secrets."

    evidence_text = json.dumps(evidence, indent=2, default=str)
    prompt = f"""
You are InsightPilot AI, an explainable business decision engine.

Analyze ONLY the verified evidence calculated directly from the user's dataset.

Rules:
1. Never invent numbers, categories, columns, or findings.
2. If the evidence is insufficient, explicitly say so.
3. Keep numerical values consistent with the evidence.
4. Separate factual findings from business interpretation.
5. Missing values and duplicate rows are data-quality issues.
6. Anomalies are analytical findings, not automatically data-quality errors.
7. Correlation describes a relationship; it does not prove causation.
8. Recommendations must be tied to the evidence.
9. Do not claim that a recommendation is guaranteed to work.
10. Mention the exact columns and values supporting the answer when relevant.
11. Be concise and business-friendly.

VERIFIED DATASET EVIDENCE:
{evidence_text}

USER'S BUSINESS QUESTION:
{user_question}

Return exactly these sections:
### Answer
Directly answer the question.

### Evidence
List the specific dataset facts supporting the answer.

### Business Interpretation
Explain what those facts mean from a business perspective.

### Recommendation
Give practical next steps based on the evidence.

### Confidence
Choose exactly one: High, Medium, Low.
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response, None
    except Exception as exc:
        error_text = str(exc)
        if "429" in error_text or "resource_exhausted" in error_text.lower() or "quota" in error_text.lower():
            return None, "Gemini usage quota is temporarily exhausted. The dashboard analytics are still available; AI questions can be retried after the quota resets."
        return None, error_text


with tab_ai:
    section_header("✦", "AI decision engine", "Ask a business question. InsightPilot calculates the evidence first, then asks Gemini to explain it.")

    st.markdown(
        """
        <div class="ai-card">
            <span class="ai-badge">● Evidence-grounded AI</span>
            <div style="font-size:1.2rem;font-weight:850;color:#111827;margin-top:0.7rem;">
                From question → evidence → interpretation → action
            </div>
            <div style="color:#64748b;font-size:0.86rem;line-height:1.55;margin-top:0.35rem;">
                The AI does not receive a raw spreadsheet and guess. Python computes the verified business evidence first,
                and the model is instructed to explain only those facts.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Try a question")
    suggestions = [
        "Which category performs best by average value?",
        "Which segment has the highest total value?",
        "What are the strongest relationships in the data?",
        "What data-quality issues should I review first?",
    ]
    suggestion_cols = st.columns(2, gap="small")
    for i, suggestion in enumerate(suggestions):
        with suggestion_cols[i % 2]:
            if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                st.session_state["ai_question"] = suggestion

    if "ai_question" not in st.session_state:
        st.session_state["ai_question"] = ""

    with st.form("ai_decision_form"):
        question = st.text_area(
            "Business question",
            key="ai_question",
            height=110,
            placeholder="Example: Which region generates the most total sales, and what should the business investigate next?",
        )
        submitted = st.form_submit_button("✦ Analyze with InsightPilot AI", type="primary", use_container_width=True)

    if submitted:
        if not question.strip():
            st.warning("Enter a business question first.")
        else:
            start_time = time.time()
            with st.spinner("Analyzing verified evidence..."):
                response, error = run_ai_analysis(question.strip(), business_evidence)
            response_time = time.time() - start_time
            st.session_state["ai_response"] = response.text if response else None
            st.session_state["ai_error"] = error
            st.session_state["ai_response_time"] = response_time
            st.session_state["ai_question_used"] = question.strip()

    if st.session_state.get("ai_error"):
        st.error(st.session_state["ai_error"])

    if st.session_state.get("ai_response"):
        st.markdown("---")
        result_left, result_right = st.columns([1.55, 0.45], gap="large")
        with result_left:
            st.markdown("#### Decision brief")
            st.markdown(st.session_state["ai_response"])

        with result_right:
            st.markdown(
                """
                <div class="soft-card">
                    <div class="mini-label">Run details</div>
                    <div style="margin-top:0.7rem;color:#334155;font-size:0.84rem;line-height:1.7;">
                        <b>Evidence:</b> Verified<br>
                        <b>Model:</b> Gemini 2.5 Flash<br>
                        <b>Grounding:</b> Python statistics<br>
                        <b>Human review:</b> Required
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.metric("Response time", f"{st.session_state.get('ai_response_time', 0):.2f}s")
            st.download_button(
                "↓ Download decision",
                data=st.session_state["ai_response"],
                file_name="insightpilot_ai_analysis.txt",
                mime="text/plain",
                use_container_width=True,
            )


# ============================================================
# TAB 4 — DATA & QUALITY
# ============================================================

with tab_data:
    section_header("▦", "Data & quality", "Inspect the uploaded records, detected types and missing-value profile.")

    preview_col, type_col = st.columns([1.35, 1], gap="large")
    with preview_col:
        st.markdown("#### Dataset preview")
        preview_rows = st.slider("Preview rows", 5, min(50, max(5, len(df))), min(10, max(5, len(df))), key="preview_rows")
        st.dataframe(df.head(preview_rows), use_container_width=True, hide_index=True, height=360)

    with type_col:
        st.markdown("#### Detected data types")
        type_df = pd.DataFrame(
            {
                "Column": df.columns,
                "Data type": [str(df[c].dtype) for c in df.columns],
                "Role": [
                    "Numeric" if c in numeric_columns else "Date" if c in date_columns else "Categorical"
                    for c in df.columns
                ],
            }
        )
        st.dataframe(type_df, use_container_width=True, hide_index=True, height=360)

    section_header("⌁", "Missing-value profile", "Columns with missing data are surfaced so they can be reviewed before decisions are made.")
    missing_df = pd.DataFrame(
        {
            "Column": df.columns,
            "Missing values": [int(df[c].isnull().sum()) for c in df.columns],
            "Missing %": [round(float(df[c].isnull().mean() * 100), 2) for c in df.columns],
        }
    ).sort_values("Missing values", ascending=False)

    nonzero_missing = missing_df[missing_df["Missing values"] > 0]
    if nonzero_missing.empty:
        st.success("No missing values were detected in the uploaded dataset.")
    else:
        st.dataframe(nonzero_missing, use_container_width=True, hide_index=True)

    section_header("∑", "Numeric summary", "Descriptive statistics for all detected numeric columns.")
    if numeric_columns:
        numeric_summary = df[numeric_columns].describe().T.round(2)
        st.dataframe(numeric_summary, use_container_width=True)
    else:
        st.info("No numeric columns were detected.")

    section_header("⇩", "Download", "Export the cleaned preview or verified evidence for your project report.")
    download1, download2 = st.columns(2, gap="medium")
    with download1:
        st.download_button(
            "Download dataset CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="insightpilot_dataset.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with download2:
        st.download_button(
            "Download verified evidence JSON",
            data=json.dumps(business_evidence, indent=2, default=str),
            file_name="insightpilot_verified_evidence.json",
            mime="application/json",
            use_container_width=True,
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.markdown(
    """
    <div style="display:flex;justify-content:space-between;gap:1rem;flex-wrap:wrap;color:#64748b;font-size:0.78rem;">
        <div><b style="color:#334155;">InsightPilot AI</b> · Turning Business Data into Explainable Decisions</div>
        <div>AI recommendations are decision support and should be reviewed by a human.</div>
    </div>
    """,
    unsafe_allow_html=True,
)
