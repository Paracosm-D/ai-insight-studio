
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from openai import OpenAI
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import mutual_info_regression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler

from epic_theme_snippets import inject_epic_serene_theme
# ----------------------------- 页面配置 -----------------------------
st.set_page_config(page_title="AI Insight Studio", page_icon="✦", layout="wide", initial_sidebar_state="expanded")
inject_epic_serene_theme()


# ----------------------------- 视觉风格 -----------------------------
def inject_custom_css():
    st.markdown("""
    <style>
    :root {
        --bg0: #07111f;
        --bg1: #0b1728;
        --bg2: rgba(15, 25, 42, 0.82);
        --panel: rgba(10, 19, 34, 0.72);
        --panel-2: rgba(16, 28, 48, 0.78);
        --line: rgba(125, 211, 252, 0.16);
        --line-strong: rgba(125, 211, 252, 0.34);
        --text: #e8f3ff;
        --muted: #91a9c4;
        --accent: #5eead4;
        --accent-2: #60a5fa;
        --accent-3: #a78bfa;
        --good: #34d399;
        --warn: #fbbf24;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(56, 189, 248, 0.12), transparent 26%),
            radial-gradient(circle at 85% 10%, rgba(167, 139, 250, 0.12), transparent 24%),
            linear-gradient(180deg, #07111f 0%, #08111d 45%, #060d16 100%);
        color: var(--text);
    }

    [data-testid="stAppViewContainer"] > .main {
        background: transparent;
    }

    .main .block-container {
        max-width: 1320px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(8,15,29,0.98) 0%, rgba(7,17,31,0.95) 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.12);
    }

    section[data-testid="stSidebar"] * {
        color: #d7e7f8 !important;
    }

    h1, h2, h3, h4 {
        letter-spacing: -0.02em;
        color: var(--text);
    }

    .hero-wrap {
        position: relative;
        overflow: hidden;
        padding: 1.6rem 1.7rem 1.5rem 1.7rem;
        border: 1px solid var(--line);
        border-radius: 28px;
        background:
            linear-gradient(135deg, rgba(15, 23, 42, 0.94), rgba(9, 17, 31, 0.8)),
            radial-gradient(circle at top right, rgba(94, 234, 212, 0.12), transparent 28%);
        box-shadow: 0 24px 60px rgba(2, 8, 23, 0.35);
        margin-bottom: 1.2rem;
    }

    .hero-wrap::before {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(90deg, transparent, rgba(94, 234, 212, 0.06), transparent);
        transform: skewX(-22deg) translateX(-40%);
        pointer-events: none;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.38rem 0.78rem;
        border-radius: 999px;
        border: 1px solid rgba(94, 234, 212, 0.24);
        background: rgba(13, 27, 43, 0.68);
        color: #c8f6f0;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 0.9rem;
    }

    .hero-title {
        font-size: 2.8rem;
        line-height: 1.05;
        font-weight: 800;
        margin: 0 0 0.65rem 0;
    }

    .hero-title .grad {
        background: linear-gradient(90deg, #dff7ff 0%, #8bdcfb 35%, #7dd3fc 60%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .hero-subtitle {
        font-size: 1rem;
        line-height: 1.75;
        color: var(--muted);
        max-width: 820px;
        margin-bottom: 1rem;
    }

    .hero-pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 0.6rem;
    }

    .hero-pill {
        padding: 0.45rem 0.82rem;
        border-radius: 999px;
        background: rgba(12, 23, 41, 0.85);
        border: 1px solid rgba(148, 163, 184, 0.16);
        color: #dbeafe;
        font-size: 0.84rem;
    }

    .section-shell {
        margin: 0.7rem 0 1rem 0;
    }

    .section-label {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.35rem 0.78rem;
        border: 1px solid rgba(96, 165, 250, 0.16);
        border-radius: 999px;
        color: #c9dcf7;
        background: rgba(13, 25, 44, 0.62);
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.45rem;
    }

    .section-title {
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0;
    }

    .section-desc {
        color: var(--muted);
        margin-top: 0.45rem;
        font-size: 0.96rem;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(180deg, rgba(10, 20, 34, 0.84), rgba(10, 20, 34, 0.64));
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 22px;
        padding: 0.9rem 1rem;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
    }

    [data-testid="stMetric"] label,
    [data-testid="stMetricValue"],
    [data-testid="stMetricDelta"] {
        color: var(--text) !important;
    }

    [data-testid="stVerticalBlock"] div:has(> [data-testid="stPlotlyChart"]),
    [data-testid="stVerticalBlock"] div:has(> [data-testid="stDataFrame"]),
    [data-testid="stExpander"] {
        border-radius: 22px;
    }

    [data-testid="stPlotlyChart"],
[data-testid="stDataFrame"],
div[data-testid="stFileUploader"],
[data-testid="stExpander"] {
    background: rgba(10, 18, 32, 0.68);
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 22px;
    box-shadow: 0 10px 28px rgba(2, 6, 23, 0.18);
}

/* ===== Selectbox / Multiselect 外层 ===== */
div[data-baseweb="select"] {
    background: transparent !important;
    border-radius: 18px !important;
}

/* ===== 真正可见的输入框区域 ===== */
div[data-baseweb="select"] > div,
div[data-baseweb="base-input"] > div {
    background: rgba(11, 22, 40, 0.92) !important;
    border: 1px solid rgba(125, 211, 252, 0.16) !important;
    border-radius: 18px !important;
    min-height: 54px !important;
    box-shadow: 0 8px 20px rgba(2, 8, 23, 0.18) !important;
    color: #e8f3ff !important;
}

/* ===== hover / focus 状态 ===== */
div[data-baseweb="select"] > div:hover,
div[data-baseweb="base-input"] > div:hover {
    border-color: rgba(125, 211, 252, 0.28) !important;
    background: rgba(13, 25, 45, 0.96) !important;
}

div[data-baseweb="select"] > div:focus-within,
div[data-baseweb="base-input"] > div:focus-within {
    border-color: rgba(94, 234, 212, 0.5) !important;
    box-shadow: 0 0 0 1px rgba(94, 234, 212, 0.18), 0 10px 28px rgba(2, 8, 23, 0.22) !important;
    background: rgba(13, 25, 45, 0.98) !important;
}

/* ===== 输入文字 / 已选文字 / placeholder ===== */
div[data-baseweb="select"] input,
div[data-baseweb="select"] span,
div[data-baseweb="select"] div,
div[data-baseweb="base-input"] input,
div[data-baseweb="base-input"] span,
div[data-baseweb="base-input"] div {
    color: #e8f3ff !important;
    -webkit-text-fill-color: #e8f3ff !important;
}

div[data-baseweb="select"] input::placeholder,
div[data-baseweb="base-input"] input::placeholder {
    color: #8ea6c1 !important;
    opacity: 1 !important;
}

/* ===== 下拉箭头 ===== */
div[data-baseweb="select"] svg {
    fill: #b9d7f2 !important;
}

/* ===== 下拉菜单弹层 ===== */
div[data-baseweb="popover"] {
    background: transparent !important;
}

div[data-baseweb="popover"] ul,
div[data-baseweb="popover"] [role="listbox"] {
    background: rgba(10, 18, 32, 0.98) !important;
    border: 1px solid rgba(125, 211, 252, 0.14) !important;
    border-radius: 16px !important;
    box-shadow: 0 18px 40px rgba(2, 8, 23, 0.35) !important;
    padding: 6px !important;
}

div[data-baseweb="popover"] li,
div[data-baseweb="popover"] [role="option"] {
    color: #e8f3ff !important;
    background: transparent !important;
    border-radius: 12px !important;
}

div[data-baseweb="popover"] li:hover,
div[data-baseweb="popover"] [role="option"]:hover {
    background: rgba(96, 165, 250, 0.10) !important;
}

div[data-baseweb="popover"] [aria-selected="true"] {
    background: rgba(94, 234, 212, 0.12) !important;
    color: #f4fbff !important;
}

/* ===== 上传区域 ===== */
div[data-testid="stFileUploader"] {
    background: rgba(10, 18, 32, 0.72) !important;
    border: 1px solid rgba(148, 163, 184, 0.12) !important;
    border-radius: 22px !important;
    padding: 0.6rem 0.75rem 0.45rem 0.75rem !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: rgba(11, 22, 40, 0.92) !important;
    border: 1px dashed rgba(125, 211, 252, 0.22) !important;
    border-radius: 18px !important;
}

[data-testid="stFileUploaderDropzone"]:hover {
    background: rgba(13, 25, 45, 0.96) !important;
    border-color: rgba(94, 234, 212, 0.36) !important;
}

[data-testid="stFileUploaderDropzone"] * {
    color: #dbeafe !important;
}

/* ===== slider 颜色也顺手统一一下 ===== */
[data-baseweb="slider"] > div > div {
    background: rgba(148, 163, 184, 0.18) !important;
}

[data-baseweb="slider"] [role="slider"] {
    background: #5eead4 !important;
    border-color: #5eead4 !important;
}

[data-baseweb="slider"] div[aria-valuenow] + div {
    background: linear-gradient(90deg, #60a5fa, #5eead4) !important;
}
    .stButton > button, .stDownloadButton > button {
        border-radius: 999px;
        border: 1px solid rgba(94, 234, 212, 0.24);
        background: linear-gradient(135deg, rgba(10, 20, 34, 0.96), rgba(14, 28, 46, 0.92));
        color: white;
        font-weight: 600;
        padding: 0.58rem 1rem;
        box-shadow: 0 8px 24px rgba(2, 8, 23, 0.22);
    }

    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: rgba(94, 234, 212, 0.5);
        transform: translateY(-1px);
    }

    [data-baseweb="tab-list"] {
        gap: 0.4rem;
    }

    [data-baseweb="tab"] {
        background: rgba(12, 22, 40, 0.75);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 999px;
        padding: 0.25rem 0.8rem;
    }

    [data-baseweb="radio"] > div {
        background: rgba(10, 20, 34, 0.48);
        border-radius: 999px;
        padding: 0.1rem;
    }

    .stAlert {
        border-radius: 18px;
        border: 1px solid rgba(148, 163, 184, 0.12);
    }

    .muted-caption {
        color: var(--muted);
        font-size: 0.88rem;
    }

    .tiny-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.85rem;
        margin-top: 0.75rem;
    }

    .tiny-card {
        padding: 0.9rem 1rem;
        border-radius: 18px;
        background: rgba(10, 18, 32, 0.62);
        border: 1px solid rgba(148, 163, 184, 0.12);
    }

    .tiny-label {
        color: var(--muted);
        font-size: 0.78rem;
        margin-bottom: 0.25rem;
    }

    .tiny-value {
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--text);
    }

    footer {visibility: hidden;}

    @media (max-width: 900px) {
        .hero-title { font-size: 2.1rem; }
        .tiny-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    </style>
    """, unsafe_allow_html=True)


def render_hero():
    st.markdown(
        """
        <div class="hero-wrap">
            <div class="hero-badge">✦ Portfolio-grade AI Analytics Product</div>
            <div class="hero-title">AI <span class="grad">Insight Studio</span></div>
            <div class="hero-subtitle">
                 Dora的可演示型 AI 数据产品原型。
            </div>
            <div class="hero-pill-row">
                <span class="hero-pill">Comment Intelligence</span>
                <span class="hero-pill">Business Diagnostics</span>
                <span class="hero-pill">Retention Analytics</span>
                <span class="hero-pill">LLM-ready Briefing Panel</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def polish_figure(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#EAF2FF"),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=False
        ),
    )
    return fig
def render_section_header(label: str, title: str, desc: str = ""):
    st.markdown(
        f"""
        <div class="section-shell">
            <div class="section-label">{label}</div>
            <div class="section-title">{title}</div>
            {f'<div class="section-desc">{desc}</div>' if desc else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_topline_stats(df_preview=None, schema=None):
    llm_status = "Connected" if LLM_CLIENT is not None else "Offline"
    rows = f"{len(df_preview):,}" if df_preview is not None else "—"
    cols = f"{df_preview.shape[1]:,}" if df_preview is not None else "—"
    time_cols = len(schema["recommendations"]["time"]) if schema else "—"
    text_cols = len(schema["recommendations"]["text"]) if schema else "—"
    st.markdown(
        f"""
        <div class="tiny-grid">
            <div class="tiny-card"><div class="tiny-label">Dataset Rows</div><div class="tiny-value">{rows}</div></div>
            <div class="tiny-card"><div class="tiny-label">Dataset Columns</div><div class="tiny-value">{cols}</div></div>
            <div class="tiny-card"><div class="tiny-label">Text / Time Signals</div><div class="tiny-value">{text_cols} / {time_cols}</div></div>
            <div class="tiny-card"><div class="tiny-label">LLM Layer</div><div class="tiny-value">{llm_status}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def polish_plotly(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=24, r=24, t=36, b=24),
        font=dict(color="#dbeafe", size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.12)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.12)", zeroline=False)
    return fig


def show_fig(fig, use_container_width=True):
    polish_plotly(fig)
    st.plotly_chart(fig, use_container_width=use_container_width)


inject_custom_css()
render_hero()
st.markdown('<div class="muted-caption">Designed for interview demos: clean, technical, and defensible.</div>', unsafe_allow_html=True)

try:
    st.set_option("client.showErrorDetails", False)
except Exception:
    pass


# ----------------------------- LLM 客户端 -----------------------------
@st.cache_resource
def init_llm_client():
    api_key = None
    base_url = None

    try:
        api_key = st.secrets.get("LLM_API_KEY")
        base_url = st.secrets.get("LLM_BASE_URL")
    except Exception:
        pass

    api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = base_url or os.getenv("LLM_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL")

    if not api_key:
        return None

    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


LLM_CLIENT = init_llm_client()


# ----------------------------- 通用工具 -----------------------------
def safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_text(text: str) -> str:
    text = safe_str(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@[A-Za-z0-9_\-\u4e00-\u9fa5]+", " ", text)
    text = re.sub(r"#([^#]+)#", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def pct(v: float) -> str:
    if pd.isna(v):
        return "-"
    return f"{v:.1%}"


def unique_keep_order(items: List[str]) -> List[str]:
    seen = set()
    output = []
    for item in items:
        if item is None:
            continue
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output


def safe_numeric(series: pd.Series) -> pd.Series:
    if isinstance(series, pd.DataFrame):
        if series.shape[1] == 0:
            return pd.Series(dtype=float)
        series = series.iloc[:, 0]
    return pd.to_numeric(series, errors="coerce")


def try_parse_datetime(series: pd.Series) -> pd.Series:
    if isinstance(series, pd.DataFrame):
        if series.shape[1] == 0:
            return pd.Series(dtype="datetime64[ns]")
        series = series.iloc[:, 0]
    return pd.to_datetime(series, errors="coerce")


def bucket_time(dt_series: pd.Series) -> Tuple[pd.Series, str]:
    dt_series = pd.to_datetime(dt_series, errors="coerce")
    valid = dt_series.dropna()
    if valid.empty:
        return pd.Series(["未知"] * len(dt_series), index=dt_series.index), "none"
    span_days = max((valid.max() - valid.min()).days, 1)
    if span_days <= 21:
        return dt_series.dt.strftime("%Y-%m-%d"), "day"
    if span_days <= 180:
        return dt_series.dt.to_period("W").astype(str), "week"
    return dt_series.dt.to_period("M").astype(str), "month"


def infer_column_by_keywords(columns: List[str], keywords: List[str]) -> Optional[str]:
    lowered = {c: str(c).lower() for c in columns}
    for kw in keywords:
        for c, lc in lowered.items():
            if kw in lc:
                return c
    return None


def robust_read_csv(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    errors = []
    tried = []

    candidates = [
        {"encoding": "utf-8-sig"},
        {"encoding": "utf-8"},
        {"encoding": "gb18030"},
        {"encoding": "gbk"},
        {"encoding": "latin1"},
        {"encoding": "utf-8-sig", "sep": ";"},
        {"encoding": "utf-8-sig", "sep": "\t"},
    ]

    for params in candidates:
        try:
            tried.append(str(params))
            return pd.read_csv(pd.io.common.BytesIO(raw), **params)
        except Exception as e:
            errors.append(f"{params}: {e}")

    raise ValueError("CSV 读取失败。请检查文件编码、分隔符和内容格式。")


def _sanitize_column_name(name: str, idx: int) -> str:
    name = safe_str(name)
    name = re.sub(r"\s+", " ", name).strip()
    if not name or name.lower().startswith("unnamed"):
        return f"unnamed_col_{idx+1}"
    return name


def make_columns_unique(columns: List[str]) -> Tuple[List[str], List[str]]:
    counts = {}
    new_cols = []
    rename_logs = []

    for i, col in enumerate(columns):
        base = _sanitize_column_name(col, i)
        counts[base] = counts.get(base, 0) + 1
        new_col = base if counts[base] == 1 else f"{base}__{counts[base]}"
        new_cols.append(new_col)
        if safe_str(col) != new_col:
            rename_logs.append(f"“{col}” → “{new_col}”")
    return new_cols, rename_logs


def preprocess_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    work = df.copy()
    new_cols, rename_logs = make_columns_unique(work.columns.tolist())
    work.columns = new_cols

    issues = []
    original_rows = len(work)
    work = work.dropna(how="all")
    dropped_rows = original_rows - len(work)
    if dropped_rows > 0:
        issues.append(f"已自动移除 {dropped_rows} 行全空记录。")

    empty_cols = [c for c in work.columns if work[c].isna().all()]
    if empty_cols:
        issues.append(f"检测到 {len(empty_cols)} 列全为空：{', '.join(empty_cols[:6])}" + (" ..." if len(empty_cols) > 6 else ""))

    return work, rename_logs, issues


def infer_series_profile(col: str, series: pd.Series) -> Dict:
    s = series.copy()
    non_null = s.dropna()
    non_null_count = int(non_null.shape[0])
    total_count = int(s.shape[0])
    non_null_ratio = non_null_count / total_count if total_count else 0.0
    nunique = int(non_null.nunique()) if non_null_count else 0
    unique_ratio = nunique / non_null_count if non_null_count else 0.0
    sample_values = [safe_str(v) for v in non_null.head(3).tolist()]

    name_lower = str(col).lower()
    as_str = non_null.astype(str).str.strip()

    numeric_score = 0.0
    dt_score = 0.0
    avg_len = float(as_str.str.len().mean()) if len(as_str) else 0.0

    if non_null_count:
        numeric_try = pd.to_numeric(as_str.str.replace(",", "", regex=False), errors="coerce")
        numeric_score = float(numeric_try.notna().mean())

        if any(k in name_lower for k in ["date", "time", "created", "时间", "日期", "timestamp", "day"]):
            dt_try = pd.to_datetime(as_str, errors="coerce")
            dt_score = float(dt_try.notna().mean())
        elif not pd.api.types.is_numeric_dtype(s):
            dt_try = pd.to_datetime(as_str, errors="coerce")
            dt_score = float(dt_try.notna().mean())

    is_text_name = any(k in name_lower for k in ["comment", "content", "text", "review", "评论", "内容", "帖子", "弹幕", "文案"])
    is_id_name = any(k in name_lower for k in ["id", "uid", "user", "player", "用户", "玩家", "账号", "role"])
    is_time_name = any(k in name_lower for k in ["date", "time", "created", "时间", "日期", "timestamp", "day"])
    is_metric_name = any(k in name_lower for k in ["score", "amount", "cnt", "count", "num", "price", "rate", "gmv", "收入", "数", "量", "金额", "时长"])

    roles = []
    base_type = "category"

    if non_null_ratio < 0.1:
        roles.append("稀疏列")

    if pd.api.types.is_numeric_dtype(s) or numeric_score >= 0.85:
        base_type = "numeric"
        roles.append("数值指标")
    if pd.api.types.is_datetime64_any_dtype(s) or dt_score >= 0.75 or is_time_name:
        if dt_score >= 0.5 or is_time_name:
            base_type = "datetime" if base_type == "category" else base_type
            roles.append("时间列")
    if is_text_name or (base_type == "category" and avg_len >= 8 and unique_ratio >= 0.25):
        roles.append("文本列")
    if is_id_name or (unique_ratio >= 0.9 and avg_len <= 30 and base_type != "datetime"):
        roles.append("ID/主键候选")
    if base_type == "category" and unique_ratio <= 0.4:
        roles.append("分组维度")

    if is_metric_name and "数值指标" not in roles:
        roles.append("数值指标候选")
    if "文本列" in roles and "分组维度" in roles and avg_len >= 15:
        roles = [r for r in roles if r != "分组维度"]

    display_type = {
        "numeric": "数值",
        "datetime": "时间",
        "category": "类别/文本",
    }.get(base_type, "未知")

    return {
        "column": col,
        "display_type": display_type,
        "roles": " / ".join(unique_keep_order(roles)) if roles else "未识别",
        "non_null_ratio": non_null_ratio,
        "unique_ratio": unique_ratio,
        "numeric_score": numeric_score,
        "datetime_score": dt_score,
        "avg_text_len": avg_len,
        "sample_values": " | ".join(sample_values[:3]),
    }


def build_schema(df: pd.DataFrame, rename_logs: List[str], import_issues: List[str]) -> Dict:
    profile_rows = [infer_series_profile(c, df[c]) for c in df.columns]
    profile_df = pd.DataFrame(profile_rows)

    def cols_with(role_keyword: str) -> List[str]:
        mask = profile_df["roles"].astype(str).str.contains(role_keyword, na=False)
        return profile_df.loc[mask, "column"].tolist()

    numeric_cols = profile_df.loc[
        (profile_df["display_type"] == "数值") | (profile_df["roles"].str.contains("数值指标候选", na=False)),
        "column",
    ].tolist()
    time_cols = cols_with("时间列")
    text_cols = cols_with("文本列")
    id_cols = cols_with("ID/主键候选")
    segment_cols = cols_with("分组维度")
    sparse_cols = cols_with("稀疏列")

    issues = list(import_issues)
    if rename_logs:
        issues.append(f"系统已自动规范 {len(rename_logs)} 个列名，避免重名或空列名引发报错。")
    if sparse_cols:
        issues.append(f"有 {len(sparse_cols)} 列缺失较多，分析前建议谨慎使用：{', '.join(sparse_cols[:6])}" + (" ..." if len(sparse_cols) > 6 else ""))
    if len(df) < 10:
        issues.append("当前样本量较小，部分聚类、异常检测和留存分析结果可能不稳定。")

    recommendations = {
        "text": text_cols,
        "time": time_cols,
        "numeric": numeric_cols,
        "id": id_cols,
        "segment": segment_cols or [c for c in df.columns if c not in text_cols[:1]],
    }

    return {
        "profile_df": profile_df,
        "issues": issues,
        "rename_logs": rename_logs,
        "recommendations": recommendations,
    }


def ordered_options(all_cols: List[str], preferred: List[str], include_none: bool = False) -> List[str]:
    preferred = [c for c in preferred if c in all_cols]
    remainder = [c for c in all_cols if c not in preferred]
    options = preferred + remainder
    if include_none:
        return ["不使用"] + options
    return options


def pick_default(options: List[str], preferred: Optional[str]) -> int:
    if preferred in options:
        return options.index(preferred)
    return 0


def render_user_message(level: str, title: str, bullets: List[str], icon: str = "⚠️"):
    if level == "error":
        st.error(f"{icon} {title}")
    elif level == "warning":
        st.warning(f"{icon} {title}")
    else:
        st.info(f"{icon} {title}")
    for item in bullets:
        st.markdown(f"- {item}")


def translate_exception(e: Exception, module_name: str) -> Tuple[str, List[str]]:
    msg = str(e)
    msg_lower = msg.lower()

    if "duplicate keys" in msg_lower or "keyerror" in msg_lower or "duplicate" in msg_lower:
        return (
            f"{module_name} 暂时无法继续，因为当前选择的字段存在重复或冲突。",
            [
                "不要把同一列同时选成两个角色，例如同时作为“用户 ID 列”和“时间列”。",
                "优先使用系统推荐的字段。",
                "如果 CSV 本身有重名表头，请先检查源文件，或重新导出一份列名清晰的数据。",
            ],
        )
    if "to_datetime" in msg_lower or "datetime" in msg_lower:
        return (
            f"{module_name} 需要可识别的时间列。",
            [
                "请改选包含日期/时间的信息列，例如 date、time、created_at、时间。",
                "时间列中应尽量避免混入纯文本说明。",
                "可先在 Excel 中统一成 YYYY-MM-DD 或 YYYY-MM-DD HH:MM:SS。",
            ],
        )
    if "numeric" in msg_lower or "could not convert" in msg_lower or "unsupported operand" in msg_lower:
        return (
            f"{module_name} 需要可计算的数值列。",
            [
                "请选择点赞、金额、次数、时长、评分等数值字段。",
                "若数字列里混有单位或文本，请先清洗，例如把“123次”改成“123”。",
                "也可以改用系统识别出的“数值指标”列。",
            ],
        )
    if "empty vocabulary" in msg_lower or "after pruning" in msg_lower:
        return (
            f"{module_name} 没有拿到足够可分析的文本。",
            [
                "请确认文本列里不是空值、单字或重复噪音。",
                "可以调低“最短文本长度过滤”。",
                "至少准备十几条以上较完整的评论文本，聚类会更稳定。",
            ],
        )
    return (
        f"{module_name} 在处理当前数据时遇到了问题。",
        [
            "请优先使用系统推荐字段。",
            "检查是否把文本列选成了数值列，或把 ID 列选成了时间列。",
            "必要时先清洗源数据后再上传。",
        ],
    )


def render_data_health(df: pd.DataFrame, schema: Dict):
    with st.expander("🩺 数据集体检与字段识别", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("样本量", f"{len(df):,}")
        c2.metric("字段数", f"{df.shape[1]:,}")
        c3.metric("识别为数值列", len(schema["recommendations"]["numeric"]))
        c4.metric("识别为时间列", len(schema["recommendations"]["time"]))

        if schema["issues"]:
            render_user_message("info", "系统检测到以下数据特点", schema["issues"], icon="🔎")

        if schema["rename_logs"]:
            st.markdown("**自动重命名记录**")
            for item in schema["rename_logs"][:12]:
                st.markdown(f"- {item}")
            if len(schema["rename_logs"]) > 12:
                st.caption(f"其余 {len(schema['rename_logs']) - 12} 项已省略。")

        profile_df = schema["profile_df"].copy()
        profile_df["非空占比"] = profile_df["non_null_ratio"].map(pct)
        profile_df["唯一值占比"] = profile_df["unique_ratio"].map(pct)
        st.dataframe(
            profile_df[["column", "display_type", "roles", "非空占比", "唯一值占比", "sample_values"]]
            if "非空占比" in profile_df.columns else profile_df,
            use_container_width=True,
        )

        st.markdown("**系统推荐字段**")
        rec = schema["recommendations"]
        st.markdown(f"- 文本列：{', '.join(rec['text'][:6]) if rec['text'] else '未识别到明显文本列'}")
        st.markdown(f"- 时间列：{', '.join(rec['time'][:6]) if rec['time'] else '未识别到明显时间列'}")
        st.markdown(f"- 数值列：{', '.join(rec['numeric'][:8]) if rec['numeric'] else '未识别到明显数值列'}")
        st.markdown(f"- ID 列：{', '.join(rec['id'][:6]) if rec['id'] else '未识别到明显 ID 列'}")


def make_openai_summary(prompt: str) -> Optional[str]:
    if LLM_CLIENT is None:
        return None
    try:
        response = LLM_CLIENT.chat.completions.create(
            model=os.getenv("LLM_MODEL", "deepseek-chat"),
            messages=[
                {"role": "system", "content": "你是资深商业分析师和游戏运营分析师，请用简洁专业的中文输出。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return None


# ----------------------------- 文本分析工具 -----------------------------
POSITIVE_WORDS = [
    "喜欢", "好玩", "优秀", "不错", "惊喜", "上头", "满意", "精致", "神", "稳",
    "love", "great", "good", "amazing", "nice", "fun", "perfect", "awesome", "solid"
]
NEGATIVE_WORDS = [
    "卡", "闪退", "崩溃", "掉帧", "bug", "问题", "失望", "无聊", "垃圾", "难受", "劝退",
    "贵", "氪", "肝", "烦", "不行", "差", "坑", "崩", "延迟", "发热", "耗电",
    "bad", "boring", "lag", "buggy", "crash", "terrible", "awful", "expensive", "grind"
]
INTENSIFIERS = ["很", "非常", "太", "特别", "极其", "超级", "真", "really", "so", "too", "very"]
NEGATIONS = ["不", "没", "无", "别", "不是", "never", "not", "no"]

TAG_RULES = {
    "性能/稳定性": ["卡", "闪退", "崩", "掉帧", "延迟", "发热", "耗电", "bug", "crash", "lag"],
    "付费/商业化": ["氪", "充值", "抽卡", "付费", "氪金", "price", "expensive", "monetization"],
    "新手/引导困惑": ["看不懂", "不会", "怎么", "哪儿", "教程", "新手", "指引", "guide", "where", "why"],
    "内容诉求": ["活动", "地图", "副本", "剧情", "皮肤", "联机", "角色", "boss", "content", "event"],
    "对比竞品/前作": ["不如", "像", "对比", "相比", "前作", "页游", "compared", "than", "like"],
    "社交/组队": ["好友", "组队", "公会", "联机", "社交", "一起", "guild", "team", "party"],
}


def sentiment_score(text: str) -> float:
    text = normalize_text(text).lower()
    if not text:
        return 0.0

    pos = 0.0
    neg = 0.0

    for word in POSITIVE_WORDS:
        if word in text:
            pos += 1.0
    for word in NEGATIVE_WORDS:
        if word in text:
            neg += 1.0

    for intensifier in INTENSIFIERS:
        if intensifier in text:
            pos *= 1.08
            neg *= 1.08

    for neg_word in NEGATIONS:
        if neg_word in text:
            pos, neg = pos * 0.92, neg * 1.05

    if "!" in text or "！" in text:
        pos *= 1.05
        neg *= 1.05
    if "?" in text or "？" in text:
        neg *= 1.03

    raw = pos - neg
    if raw == 0:
        return 0.0
    return float(np.tanh(raw / 3.0))


def sentiment_label(score: float) -> str:
    if score >= 0.2:
        return "积极"
    if score <= -0.2:
        return "消极"
    return "中性"


def tag_comment(text: str) -> str:
    text = normalize_text(text).lower()
    tags = []
    for tag, kws in TAG_RULES.items():
        if any(kw.lower() in text for kw in kws):
            tags.append(tag)
    return " / ".join(tags) if tags else "未命中规则"


def detect_special_flags(text: str, sentiment: float, engagement: float, median_eng: float) -> str:
    text_norm = normalize_text(text).lower()
    flags = []
    if engagement >= median_eng * 1.8 and sentiment <= -0.2:
        flags.append("高热负面")
    if engagement >= median_eng * 1.8 and abs(sentiment) < 0.15:
        flags.append("沉默高关注")
    if ("?" in text_norm or "？" in text_norm or "怎么" in text_norm) and sentiment < 0:
        flags.append("困惑信号")
    if any(k in text_norm for k in ["不如", "相比", "compared", "than", "页游"]):
        flags.append("竞品/前作比较")
    if any(k in text_norm for k in ["闪退", "崩", "卡", "掉帧", "bug", "lag", "crash"]):
        flags.append("故障风险")
    return " / ".join(flags) if flags else "普通"


# ----------------------------- 商业分析工具 -----------------------------
def compute_weighted_gap_table(df: pd.DataFrame, segment_col: str, target_col: str, weight_col: Optional[str]) -> pd.DataFrame:
    selected_cols = unique_keep_order([segment_col, target_col] + ([weight_col] if weight_col else []))
    temp = df.loc[:, selected_cols].copy()
    temp[target_col] = safe_numeric(temp[target_col])
    temp = temp.dropna(subset=[segment_col, target_col])

    if temp.empty:
        return pd.DataFrame()

    overall = temp[target_col].mean()

    if weight_col and weight_col in temp.columns:
        temp[weight_col] = safe_numeric(temp[weight_col]).fillna(1)
    else:
        weight_col = "__w__"
        temp[weight_col] = 1

    grouped = temp.groupby(segment_col).agg(
        sample_size=(target_col, "size"),
        avg_target=(target_col, "mean"),
        total_weight=(weight_col, "sum"),
    ).reset_index()

    grouped["weight_share"] = grouped["total_weight"] / grouped["total_weight"].sum()
    grouped["gap_vs_overall"] = grouped["avg_target"] - overall
    grouped["priority_score"] = grouped["gap_vs_overall"].abs() * grouped["weight_share"]
    return grouped.sort_values("priority_score", ascending=False)


def compute_driver_table(df: pd.DataFrame, target_col: str, numeric_candidates: List[str], exclude_cols: List[str]) -> pd.DataFrame:
    X_cols = [c for c in numeric_candidates if c != target_col and c not in exclude_cols]
    if not X_cols:
        return pd.DataFrame()

    temp = df[unique_keep_order(X_cols + [target_col])].copy()
    for c in temp.columns:
        temp[c] = safe_numeric(temp[c])

    temp = temp.dropna()
    if len(temp) < 30:
        return pd.DataFrame()

    X = temp[X_cols]
    y = temp[target_col]

    mi = mutual_info_regression(X, y, random_state=42)
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X, y)

    out = pd.DataFrame({
        "feature": X_cols,
        "mutual_info": mi,
        "rf_importance": rf.feature_importances_,
    })
    out["composite_score"] = out["mutual_info"].rank(pct=True) * 0.5 + out["rf_importance"].rank(pct=True) * 0.5
    return out.sort_values("composite_score", ascending=False)


# ----------------------------- 留存分析工具 -----------------------------
def retention_matrix(events: pd.DataFrame, user_col: str, time_col: str, freq: str = "W") -> pd.DataFrame:
    selected_cols = unique_keep_order([user_col, time_col])
    data = events.loc[:, selected_cols].copy()
    data[time_col] = try_parse_datetime(data[time_col])
    data = data.dropna(subset=[user_col, time_col])
    if data.empty:
        return pd.DataFrame()

    if freq == "D":
        data["period"] = data[time_col].dt.to_period("D")
    elif freq == "M":
        data["period"] = data[time_col].dt.to_period("M")
    else:
        data["period"] = data[time_col].dt.to_period("W")

    first_period = data.groupby(user_col)["period"].min().rename("cohort_period")
    data = data.merge(first_period, left_on=user_col, right_index=True, how="left")
    data["period_index"] = (data["period"] - data["cohort_period"]).apply(lambda x: x.n)
    cohort_size = data.groupby("cohort_period")[user_col].nunique().rename("cohort_size")
    retained = data.groupby(["cohort_period", "period_index"])[user_col].nunique().rename("users").reset_index()
    retained = retained.merge(cohort_size, left_on="cohort_period", right_index=True)
    retained["retention"] = retained["users"] / retained["cohort_size"]
    matrix = retained.pivot(index="cohort_period", columns="period_index", values="retention").sort_index()
    matrix.index = matrix.index.astype(str)
    return matrix


def user_state_table(events: pd.DataFrame, user_col: str, time_col: str) -> pd.DataFrame:
    selected_cols = unique_keep_order([user_col, time_col])
    data = events.loc[:, selected_cols].copy()
    data[time_col] = try_parse_datetime(data[time_col])
    data = data.dropna(subset=[user_col, time_col])
    if data.empty:
        return pd.DataFrame()

    max_day = data[time_col].max().normalize()
    data["day"] = data[time_col].dt.normalize()

    last_7_start = max_day - pd.Timedelta(days=6)
    prev_7_start = max_day - pd.Timedelta(days=13)
    prev_7_end = max_day - pd.Timedelta(days=7)

    active_curr = set(data.loc[data["day"].between(last_7_start, max_day), user_col])
    active_prev = set(data.loc[data["day"].between(prev_7_start, prev_7_end), user_col])

    user_first = data.groupby(user_col)["day"].min()
    user_last = data.groupby(user_col)["day"].max()
    life_days = data.groupby(user_col)["day"].nunique()

    rows = []
    for uid in data[user_col].dropna().unique():
        in_curr = uid in active_curr
        in_prev = uid in active_prev
        first_seen = user_first.get(uid)
        last_seen = user_last.get(uid)
        active_days = int(life_days.get(uid, 1))
        days_since_last = int((max_day - last_seen).days)

        if in_curr and not in_prev and first_seen >= last_7_start:
            state = "新用户"
        elif in_curr and in_prev:
            state = "连续留存"
        elif in_curr and not in_prev:
            state = "回流"
        elif (not in_curr) and in_prev:
            state = "流失风险"
        else:
            state = "沉默"

        rows.append({
            user_col: uid,
            "state": state,
            "first_seen": first_seen,
            "last_seen": last_seen,
            "active_days": active_days,
            "days_since_last_seen": days_since_last,
        })
    return pd.DataFrame(rows)


# ----------------------------- 模块校验 -----------------------------
def validate_text_column(df: pd.DataFrame, col: str, schema: Dict) -> Tuple[bool, List[str]]:
    if col not in df.columns:
        return False, ["所选文本列不存在。"]
    clean = df[col].astype(str).map(normalize_text)
    usable_ratio = (clean.str.len() >= 3).mean()
    bullets = []
    if usable_ratio < 0.3:
        bullets.append("该列可用文本比例过低，可能不是评论文本列。")
    if clean.nunique(dropna=True) < 8:
        bullets.append("该列的有效文本过少，难以稳定聚类。")
    if bullets:
        rec = schema["recommendations"]["text"]
        if rec:
            bullets.append(f"更适合的文本列候选：{', '.join(rec[:5])}")
        return False, bullets
    return True, []


def validate_numeric_column(df: pd.DataFrame, col: str, schema: Dict) -> Tuple[bool, List[str]]:
    if col == "不使用":
        return True, []
    s = safe_numeric(df[col])
    valid_ratio = s.notna().mean()
    if valid_ratio < 0.5:
        rec = schema["recommendations"]["numeric"]
        bullets = [f"“{col}” 可识别为数值的比例较低，不适合直接计算。"]
        if rec:
            bullets.append(f"可改选这些数值列：{', '.join(rec[:6])}")
        bullets.append("如果原列包含单位或文字，请先清洗成纯数字。")
        return False, bullets
    return True, []


def validate_time_column(df: pd.DataFrame, col: str, schema: Dict, required: bool = False) -> Tuple[bool, List[str]]:
    if col == "不使用":
        if required:
            return False, ["当前模块必须选择时间列。"]
        return True, []

    parsed = try_parse_datetime(df[col])
    valid_ratio = parsed.notna().mean()
    if valid_ratio < 0.5:
        rec = schema["recommendations"]["time"]
        bullets = [f"“{col}” 无法被稳定识别为时间列。"]
        if rec:
            bullets.append(f"建议改选：{', '.join(rec[:5])}")
        bullets.append("时间格式建议统一成 YYYY-MM-DD 或 YYYY-MM-DD HH:MM:SS。")
        return False, bullets
    return True, []


def validate_retention_inputs(df: pd.DataFrame, user_col: str, time_col: str, schema: Dict) -> Tuple[bool, List[str]]:
    if user_col == time_col:
        return False, ["用户 ID 列与时间列不能相同。", "请把时间列改成 date / time / created_at 之类字段。"]

    ok_time, bullets = validate_time_column(df, time_col, schema, required=True)
    if not ok_time:
        return False, bullets

    user_non_null = df[user_col].dropna()
    if user_non_null.empty:
        return False, ["用户 ID 列全为空。"]
    repeat_ratio = 1 - (user_non_null.nunique() / max(len(user_non_null), 1))
    if repeat_ratio <= 0.02:
        return False, [
            "当前“用户 ID 列”几乎没有重复值，更像交易号或事件号，而不是用户标识。",
            "留存分析需要同一用户在多个时间点重复出现。",
            f"可优先尝试这些 ID 候选：{', '.join(schema['recommendations']['id'][:5]) if schema['recommendations']['id'] else '未识别到明显 ID 列'}",
        ]
    return True, []


# ----------------------------- 页面导航 -----------------------------
st.sidebar.title("场景导航")
st.sidebar.caption("选择一个分析工作流，系统会根据数据结构推荐可用字段。")
st.sidebar.markdown("---")
module = st.sidebar.radio(
    "选择分析分区",
    ["总览", "运营洞察", "商业分析", "留存分析"],
)

render_section_header("Data Intake", "Upload your dataset", "建议优先使用结构清晰的 CSV；系统会先自动体检字段，再开放适配的分析路径。")
uploaded_file = st.file_uploader("上传 CSV 数据文件", type=["csv"])

df = None
schema = None
rename_logs = []
import_issues = []

if uploaded_file is None:
    st.info("请上传 CSV 开始体验。推荐优先演示一份社媒评论数据或行为事件数据。")
    st.stop()

try:
    raw_df = robust_read_csv(uploaded_file)
    df, rename_logs, import_issues = preprocess_dataframe(raw_df)
    schema = build_schema(df, rename_logs, import_issues)
    st.success(f"已读取数据：{df.shape[0]} 行 × {df.shape[1]} 列")
    render_topline_stats(df, schema)
except Exception as e:
    title, bullets = translate_exception(e, "数据导入")
    render_user_message("error", title, bullets)
    st.stop()

if df is None or schema is None:
    st.info("请通过 Streamlit 页面上传 CSV 后再继续。")
    st.stop()

render_data_health(df, schema)

with st.expander("查看原始数据预览", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)


# ----------------------------- 模块渲染 -----------------------------
def render_overview():
    render_section_header("Overview", "Portfolio Product Snapshot", "用一眼能看懂的结构告诉面试官：这不是 notebook，而是有产品意识的 AI analytics prototype。")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("当前版本状态", "Portfolio Demo")
    c2.metric("已开启", "数据体检", "列语义识别 + 自动防呆")
    c3.metric("已开启", "用户友好报错", "不再直接暴露代码异常")
    c4.metric("当前重点", "场景工作流", "游戏 / 商业 / 留存")


def render_game_module():
    render_section_header("Game Ops", "Comment Intelligence Workbench", "围绕社媒评论的情绪、主题、异常信号与机会点，输出可汇报的运营洞察。")

    columns = df.columns.tolist()
    rec = schema["recommendations"]
    text_options = ordered_options(columns, rec["text"])
    time_options = ordered_options(columns, rec["time"], include_none=True)
    numeric_options = ordered_options(columns, rec["numeric"], include_none=True)

    default_text = infer_column_by_keywords(text_options, ["comment", "content", "text", "评论", "内容", "帖子"])
    default_time = infer_column_by_keywords(time_options, ["date", "time", "created", "发布时间", "时间"])
    default_like = infer_column_by_keywords(numeric_options, ["like", "点赞"])
    default_reply = infer_column_by_keywords(numeric_options, ["reply", "comment_count", "回复"])
    default_share = infer_column_by_keywords(numeric_options, ["share", "转发", "收藏", "fav"])

    left, right = st.columns([1.3, 1])
    with left:
        text_col = st.selectbox("评论文本列", text_options, index=pick_default(text_options, default_text))
        time_col = st.selectbox("时间列（可选）", time_options, index=pick_default(time_options, default_time))
    with right:
        like_col = st.selectbox("点赞列（可选）", numeric_options, index=pick_default(numeric_options, default_like))
        reply_col = st.selectbox("回复列（可选）", numeric_options, index=pick_default(numeric_options, default_reply))
        share_col = st.selectbox("转发/收藏列（可选）", numeric_options, index=pick_default(numeric_options, default_share))

    max_clusters = st.slider("主题聚类数", min_value=4, max_value=10, value=6)
    min_text_len = st.slider("最短文本长度过滤", min_value=2, max_value=20, value=4)

    ok_text, text_bullets = validate_text_column(df, text_col, schema)
    if not ok_text:
        render_user_message("warning", "当前选择的文本列不适合做评论分析", text_bullets)
        return

    for col in [like_col, reply_col, share_col]:
        ok_num, num_bullets = validate_numeric_column(df, col, schema)
        if not ok_num:
            render_user_message("warning", f"“{col}” 目前不适合作为互动指标", num_bullets)
            return

    ok_time, time_bullets = validate_time_column(df, time_col, schema, required=False)
    if not ok_time:
        render_user_message("warning", "当前时间列无法稳定用于趋势分析", time_bullets)
        time_col = "不使用"

    work = df.copy()
    work["clean_text"] = work[text_col].astype(str).map(normalize_text)
    work = work[work["clean_text"].str.len() >= min_text_len].copy()

    if work.empty or work["clean_text"].nunique() < 8:
        render_user_message(
            "warning",
            "当前文本不足以稳定完成评论聚类",
            [
                "请降低“最短文本长度过滤”，或换一列更完整的评论文本。",
                "聚类分析建议至少准备 15 条以上较完整的评论。",
            ],
        )
        return

    engagement_parts = []
    for col_name, weight in [(like_col, 1.0), (reply_col, 1.6), (share_col, 1.8)]:
        if col_name != "不使用":
            numeric_series = safe_numeric(work[col_name]).fillna(0)
            engagement_parts.append(np.log1p(np.maximum(numeric_series, 0)) * weight)

    if engagement_parts:
        work["engagement_score"] = np.sum(engagement_parts, axis=0)
    else:
        work["engagement_score"] = 1.0

    work["sentiment_score"] = work["clean_text"].map(sentiment_score)
    work["sentiment_label"] = work["sentiment_score"].map(sentiment_label)
    work["rule_tags"] = work["clean_text"].map(tag_comment)
    median_eng = float(np.median(work["engagement_score"])) if len(work) else 1.0
    work["special_flags"] = work.apply(
        lambda r: detect_special_flags(r["clean_text"], r["sentiment_score"], r["engagement_score"], median_eng),
        axis=1,
    )

    try:
        vectorizer = TfidfVectorizer(max_features=2500, analyzer="char_wb", ngram_range=(2, 4), min_df=2)
        X = vectorizer.fit_transform(work["clean_text"])
    except Exception:
        render_user_message(
            "warning",
            "文本特征提取失败",
            [
                "当前文本可能过短、重复过多，或几乎都是空值。",
                "建议换一个更像评论内容的列，或使用更完整的数据样本。",
            ],
        )
        return

    if X.shape[1] <= 2 or len(work) < max_clusters + 4:
        render_user_message(
            "warning",
            "当前样本不足以稳定聚类",
            [
                "建议减少主题聚类数，或上传更多评论。",
                f"当前有效文本数：{len(work)}；当前特征数：{X.shape[1]}。",
            ],
        )
        return

    n_components = max(2, min(20, X.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_red = svd.fit_transform(X)

    kmeans = KMeans(n_clusters=max_clusters, random_state=42, n_init=10)
    work["cluster"] = kmeans.fit_predict(X_red).astype(str)

    feature_names = np.array(vectorizer.get_feature_names_out())
    top_terms = {}
    centers = kmeans.cluster_centers_
    for i in range(max_clusters):
        top_idx = np.argsort(centers[i])[-4:][::-1]
        terms = [feature_names[j] for j in top_idx if j < len(feature_names)]
        top_terms[str(i)] = " / ".join(terms[:3]) if terms else f"主题 {i+1}"

    work["topic_name"] = work["cluster"].map(top_terms)

    if len(work) >= 10:
        lof_input = np.column_stack([
            X_red[:, 0],
            X_red[:, 1],
            MinMaxScaler().fit_transform(work[["engagement_score"]]).flatten(),
            work["sentiment_score"].values,
        ])
        neighbors = min(20, max(5, len(work) - 1))
        lof = LocalOutlierFactor(n_neighbors=neighbors, contamination=min(0.08, max(0.02, 10 / len(work))))
        outlier_flag = lof.fit_predict(lof_input)
        work["outlier_flag"] = np.where(outlier_flag == -1, "异常/稀有", "常规")
        work["novelty_score"] = -lof.negative_outlier_factor_
    else:
        work["outlier_flag"] = "常规"
        work["novelty_score"] = 0.0

    if time_col != "不使用":
        work["parsed_time"] = try_parse_datetime(work[time_col])
        valid_time_ratio = work["parsed_time"].notna().mean()
        if valid_time_ratio >= 0.5:
            work["time_bucket"], bucket_mode = bucket_time(work["parsed_time"])
        else:
            work["time_bucket"] = "未知"
            bucket_mode = "none"
    else:
        work["parsed_time"] = pd.NaT
        work["time_bucket"] = "未知"
        bucket_mode = "none"

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("评论数", f"{len(work):,}")
    k2.metric("消极评论占比", pct((work["sentiment_label"] == "消极").mean()))
    k3.metric("高热负面条数", int(work["special_flags"].str.contains("高热负面", na=False).sum()))
    k4.metric("异常/稀有信号", int((work["outlier_flag"] == "异常/稀有").sum()))

    st.markdown("### ① 语义星图")
    fig_semantic = px.scatter(
        work,
        x=X_red[:, 0],
        y=X_red[:, 1],
        color="topic_name",
        size="engagement_score",
        symbol="sentiment_label",
        hover_data={
            text_col: True,
            "topic_name": True,
            "sentiment_score": ":.2f",
            "engagement_score": ":.2f",
            "special_flags": True,
        },
        labels={"x": "语义维度 1", "y": "语义维度 2"},
        height=560,
    )
    fig_semantic.update_traces(marker=dict(opacity=0.75, line=dict(width=0.5, color="white")))
    show_fig(fig_semantic, use_container_width=True)

    st.markdown("### ② 情绪 × 关注机会矩阵")
    fig_opp = px.scatter(
        work,
        x="sentiment_score",
        y="engagement_score",
        color="topic_name",
        size="novelty_score",
        hover_data={text_col: True, "special_flags": True, "rule_tags": True},
        height=520,
    )
    fig_opp.add_vline(x=0, line_dash="dash", line_width=1)
    fig_opp.add_hline(y=float(work["engagement_score"].median()), line_dash="dash", line_width=1)
    show_fig(fig_opp, use_container_width=True)

    if bucket_mode != "none":
        st.markdown("### ③ 主题漂移热带图")
        drift = work.groupby(["time_bucket", "topic_name"]).size().rename("cnt").reset_index()
        drift["share"] = drift.groupby("time_bucket")["cnt"].transform(lambda s: s / s.sum())
        heat = drift.pivot(index="topic_name", columns="time_bucket", values="share").fillna(0)
        fig_heat = go.Figure(
            data=go.Heatmap(
                z=heat.values,
                x=heat.columns,
                y=heat.index,
                text=np.round(heat.values, 3),
                hovertemplate="主题: %{y}<br>时间: %{x}<br>占比: %{z:.1%}<extra></extra>",
            )
        )
        fig_heat.update_layout(height=430, margin=dict(l=20, r=20, t=30, b=20))
        show_fig(fig_heat, use_container_width=True)

    st.markdown("### ④ 主题摘要与特殊信号")
    cluster_summary = (
        work.groupby("topic_name")
        .agg(
            comments=("clean_text", "size"),
            avg_sentiment=("sentiment_score", "mean"),
            avg_engagement=("engagement_score", "mean"),
            rare_ratio=("outlier_flag", lambda s: (s == "异常/稀有").mean()),
        )
        .reset_index()
        .sort_values(["comments", "avg_engagement"], ascending=[False, False])
    )
    st.dataframe(cluster_summary, use_container_width=True)

    special_table = (
        work.sort_values(["novelty_score", "engagement_score"], ascending=[False, False])[
            ["clean_text", "topic_name", "sentiment_score", "engagement_score", "novelty_score", "rule_tags", "special_flags", "outlier_flag"]
        ]
        .head(20)
        .copy()
    )
    st.dataframe(special_table, use_container_width=True)

    st.markdown("### ⑤ AI 洞察面板")
    selected_cols = ["clean_text", "topic_name", "sentiment_label", "special_flags", "rule_tags"]
    examples = work.sort_values(["engagement_score", "novelty_score"], ascending=[False, False])[selected_cols].head(12)
    prompt = f"""
你正在读取一个游戏运营社媒评论分析看板，请根据以下数据总结输出：
1）3个最值得汇报的发现
2）2个容易被忽略但值得跟进的信号
3）面向产品/运营/社区管理各给1条建议

主题汇总：
{cluster_summary.to_string(index=False)}

高关注/高新颖样本：
{examples.to_string(index=False)}
"""
    summary = make_openai_summary(prompt)
    if summary:
        st.success("已生成 AI 洞察")
        st.markdown(summary)
    else:
        st.info("当前未配置 LLM Key，已跳过 AI 总结，仅展示规则分析与可视化结果。")

    csv_bytes = work.to_csv(index=False).encode("utf-8-sig")
    st.download_button("下载处理后的评论洞察表", data=csv_bytes, file_name="game_comment_insights.csv", mime="text/csv")


def render_business_module():
    render_section_header("Business", "Business Diagnostics Workbench", "面向业务指标拆解、分群优先级和驱动因子识别的管理分析界面。")

    rec = schema["recommendations"]
    numeric_cols = rec["numeric"]
    all_cols = df.columns.tolist()
    segment_candidates = ordered_options(all_cols, rec["segment"])
    numeric_options = ordered_options(numeric_cols, numeric_cols, include_none=False)
    weight_options = ordered_options(numeric_cols, numeric_cols, include_none=True)

    if not numeric_cols:
        render_user_message(
            "warning",
            "当前数据未识别到足够稳定的数值列",
            [
                "商业诊断模块需要至少一个数值指标列。",
                "请检查金额、评分、次数、时长等字段是否被存成了纯文本。",
            ],
        )
        return

    target_col = st.selectbox("核心目标指标", numeric_options, index=0)
    segment_col = st.selectbox("分群列", segment_candidates, index=0)
    weight_col = st.selectbox("规模/权重列（可选）", weight_options, index=0)

    remaining_numeric = [c for c in numeric_cols if c != target_col]
    x_options = remaining_numeric or [target_col]
    x_col = st.selectbox("效率前沿 X 轴", x_options, index=0)
    y_pool = [c for c in numeric_cols if c not in {target_col, x_col}] or [target_col]
    y_col = st.selectbox("效率前沿 Y 轴", y_pool, index=0)

    if segment_col == target_col:
        render_user_message(
            "warning",
            "当前字段组合不合理",
            [
                "“分群列”不能与“核心目标指标”相同。",
                "分群列更适合选择渠道、城市、品类、用户层级等类别字段。",
            ],
        )
        return

    for col in [target_col, x_col, y_col, weight_col]:
        ok_num, num_bullets = validate_numeric_column(df, col, schema)
        if not ok_num:
            render_user_message("warning", f"“{col}” 暂不适合做业务指标计算", num_bullets)
            return

    gap_table = compute_weighted_gap_table(
        df=df,
        segment_col=segment_col,
        target_col=target_col,
        weight_col=None if weight_col == "不使用" else weight_col,
    )

    if gap_table.empty:
        render_user_message(
            "warning",
            "当前数据不足以生成分群优先级",
            [
                "请检查分群列是否为空过多，或目标指标列是否大多无法转成数字。",
                "建议优先选择非空率高、类别清晰的分群列。",
            ],
        )
        return

    st.markdown("### ① 分群优先级矩阵")
    fig_gap = px.scatter(
        gap_table,
        x="gap_vs_overall",
        y="weight_share",
        size="sample_size",
        color="priority_score",
        hover_data={segment_col: True, "avg_target": ":.2f", "sample_size": True},
        text=segment_col,
        labels={"gap_vs_overall": "相对整体差值", "weight_share": "体量占比"},
        height=520,
    )
    fig_gap.add_vline(x=0, line_dash="dash", line_width=1)
    show_fig(fig_gap, use_container_width=True)

    st.markdown("### ② 效率前沿图")
    frontier_cols = unique_keep_order([segment_col, x_col, y_col] + ([weight_col] if weight_col != "不使用" else []))
    frontier_df = df.loc[:, frontier_cols].copy()
    frontier_df[x_col] = safe_numeric(frontier_df[x_col])
    frontier_df[y_col] = safe_numeric(frontier_df[y_col])
    frontier_df = frontier_df.dropna(subset=[x_col, y_col])

    if weight_col != "不使用":
        frontier_df[weight_col] = safe_numeric(frontier_df[weight_col]).fillna(1)
        volume_col = weight_col
    else:
        frontier_df["__w__"] = 1
        volume_col = "__w__"

    segment_frontier = frontier_df.groupby(segment_col).agg(
        x_metric=(x_col, "mean"),
        y_metric=(y_col, "mean"),
        volume=(volume_col, "sum"),
    ).reset_index()

    fig_frontier = px.scatter(
        segment_frontier,
        x="x_metric",
        y="y_metric",
        size="volume",
        text=segment_col,
        hover_data={segment_col: True},
        labels={"x_metric": x_col, "y_metric": y_col},
        height=520,
    )
    show_fig(fig_frontier, use_container_width=True)

    st.markdown("### ③ 驱动因子扫描")
    drivers = compute_driver_table(df, target_col=target_col, numeric_candidates=numeric_cols, exclude_cols=[segment_col])
    if drivers.empty:
        st.info("样本量或可用数值特征不足，暂时无法稳定计算驱动因子。")
    else:
        fig_driver = px.scatter(
            drivers.head(12),
            x="mutual_info",
            y="rf_importance",
            size="composite_score",
            text="feature",
            hover_data={"feature": True, "composite_score": ":.2f"},
            height=480,
        )
        show_fig(fig_driver, use_container_width=True)
        st.dataframe(drivers.head(12), use_container_width=True)

    prompt = f"""
你正在看一个商业分析诊断看板。
目标指标：{target_col}
分群列：{segment_col}

分群优先级：
{gap_table.head(10).to_string(index=False)}

驱动因子：
{drivers.head(10).to_string(index=False) if not drivers.empty else "暂无"}

请输出：
1）一句高层摘要
2）三个优先动作
"""
    summary = make_openai_summary(prompt)
    if summary:
        st.markdown("### ④ AI 管理摘要")
        st.markdown(summary)


def render_retention_module():
    render_section_header("Lifecycle", "Retention & Lifecycle Workbench", "把行为事件数据转换成 cohort、衰减曲线和用户状态分层，支持留存与召回讨论。")

    cols = df.columns.tolist()
    rec = schema["recommendations"]
    user_options = ordered_options(cols, rec["id"] or cols)
    time_options = ordered_options(cols, rec["time"] or cols)
    event_options = ordered_options(cols, cols, include_none=True)

    default_user = infer_column_by_keywords(user_options, ["user", "uid", "player", "用户", "角色"])
    default_time = infer_column_by_keywords(time_options, ["date", "time", "created", "event_time", "时间"])
    default_event = infer_column_by_keywords(event_options, ["event", "action", "行为"])

    user_col = st.selectbox("用户 ID 列", user_options, index=pick_default(user_options, default_user))
    time_col = st.selectbox("时间列", time_options, index=pick_default(time_options, default_time))
    event_col = st.selectbox("事件列（可选）", event_options, index=pick_default(event_options, default_event))
    freq = st.radio("留存粒度", ["D", "W", "M"], horizontal=True, index=1)

    ok_inputs, bullets = validate_retention_inputs(df, user_col, time_col, schema)
    if not ok_inputs:
        render_user_message("warning", "当前字段组合不适合做留存分析", bullets)
        return

    matrix = retention_matrix(df, user_col=user_col, time_col=time_col, freq=freq)
    if matrix.empty or matrix.shape[1] == 0:
        render_user_message(
            "warning",
            "暂时无法生成留存矩阵",
            [
                "请检查用户 ID 是否重复出现、时间列是否可识别。",
                "留存分析通常要求同一用户在多个时间点有多次行为记录。",
            ],
        )
        return

    st.markdown("### ① Cohort 留存矩阵")
    fig_ret = go.Figure(
        data=go.Heatmap(
            z=matrix.values,
            x=[f"T+{c}" for c in matrix.columns],
            y=matrix.index,
            text=np.round(matrix.values, 3),
            hovertemplate="Cohort: %{y}<br>阶段: %{x}<br>留存: %{z:.1%}<extra></extra>",
        )
    )
    fig_ret.update_layout(height=500, margin=dict(l=20, r=20, t=30, b=20))
    show_fig(fig_ret, use_container_width=True)

    st.markdown("### ② Decay Signature")
    avg_curve = matrix.mean(axis=0).reset_index()
    avg_curve.columns = ["period_index", "retention"]
    fig_curve = px.line(avg_curve, x="period_index", y="retention", markers=True, height=420)
    show_fig(fig_curve, use_container_width=True)

    st.markdown("### ③ 生命周期状态分布")
    states = user_state_table(df, user_col=user_col, time_col=time_col)
    if states.empty:
        st.info("最近 14 天状态暂时无法计算。")
        state_cnt = pd.DataFrame(columns=["state", "users"])
    else:
        state_cnt = states["state"].value_counts().rename_axis("state").reset_index(name="users")
        fig_state = px.funnel_area(state_cnt, names="state", values="users", height=420)
        show_fig(fig_state, use_container_width=True)

        st.markdown("### ④ 回流机会散点图")
        fig_reactivation = px.scatter(
            states,
            x="days_since_last_seen",
            y="active_days",
            color="state",
            hover_data={user_col: True},
            height=460,
        )
        show_fig(fig_reactivation, use_container_width=True)
        st.dataframe(
            states.sort_values(["days_since_last_seen", "active_days"], ascending=[False, False]).head(20),
            use_container_width=True,
        )

    prompt = f"""
你正在查看留存分析面板。
平均留存曲线：
{avg_curve.to_string(index=False)}

用户状态分布：
{state_cnt.to_string(index=False) if not state_cnt.empty else "暂无"}

请输出：
1）当前生命周期最关键的问题
2）一个留存动作、一个召回动作
3）一句适合作品集展示的产品描述
"""
    summary = make_openai_summary(prompt)
    if summary:
        st.markdown("### ⑤ AI 生命周期建议")
        st.markdown(summary)


if module == "总览":
    render_overview()
elif module == "运营洞察":
    try:
        render_game_module()
    except Exception as e:
        title, bullets = translate_exception(e, "游戏运营模块")
        render_user_message("error", title, bullets)
elif module == "商业分析":
    try:
        render_business_module()
    except Exception as e:
        title, bullets = translate_exception(e, "商业分析模块")
        render_user_message("error", title, bullets)
elif module == "留存分析":
    try:
        render_retention_module()
    except Exception as e:
        title, bullets = translate_exception(e, "留存分析模块")
        render_user_message("error", title, bullets)
