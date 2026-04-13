import streamlit as st


def inject_epic_serene_theme():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
            --bg: #07111F;
            --bg-2: #0B1830;
            --surface: rgba(11, 24, 48, 0.58);
            --surface-strong: rgba(11, 24, 48, 0.78);
            --border: rgba(180, 214, 255, 0.16);
            --text: #EAF2FF;
            --muted: #9FB0C8;
            --accent: #89D5FF;
            --accent-2: #A89BFF;
            --glow: rgba(116, 190, 255, 0.18);
            --shadow: 0 12px 40px rgba(0,0,0,0.28);
        }

        html, body, [class*="css"]  {
            font-family: 'Inter', 'PingFang SC', 'Noto Sans SC', sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 18% 18%, rgba(126, 184, 255, 0.14), transparent 24%),
                radial-gradient(circle at 80% 22%, rgba(157, 135, 255, 0.10), transparent 22%),
                radial-gradient(circle at 50% -10%, rgba(255,255,255,0.05), transparent 28%),
                linear-gradient(180deg, #050B14 0%, #091321 42%, #07111F 100%);
            color: var(--text);
        }

        .stApp::before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            background:
                linear-gradient(rgba(255,255,255,0.018) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,0.018) 1px, transparent 1px);
            background-size: 44px 44px;
            mask-image: radial-gradient(circle at center, black 42%, transparent 88%);
            opacity: .22;
        }

        [data-testid="stHeader"] {
            background: rgba(0,0,0,0);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(10,18,34,.94), rgba(7,17,31,.88));
            border-right: 1px solid var(--border);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 5rem;
            max-width: 1240px;
        }

        h1, h2, h3, h4, p, label, span, div {
            color: var(--text);
        }

        .epic-hero {
            position: relative;
            overflow: hidden;
            border: 1px solid var(--border);
            background: linear-gradient(180deg, rgba(14,27,50,.62), rgba(8,16,30,.72));
            border-radius: 28px;
            padding: 38px 42px 34px;
            box-shadow: var(--shadow);
            backdrop-filter: blur(18px);
            margin-bottom: 22px;
        }

        .epic-hero::after {
            content: "";
            position: absolute;
            width: 520px;
            height: 520px;
            right: -120px;
            top: -160px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(137,213,255,.15), rgba(137,213,255,0));
            filter: blur(8px);
        }

        .eyebrow {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 999px;
            border: 1px solid rgba(180,214,255,.18);
            color: var(--accent);
            font-size: 12px;
            letter-spacing: .08em;
            text-transform: uppercase;
            background: rgba(255,255,255,.03);
            margin-bottom: 14px;
        }

        .hero-title {
            font-size: 52px;
            line-height: 1.05;
            font-weight: 700;
            letter-spacing: -.02em;
            margin: 0 0 12px 0;
            max-width: 760px;
        }

        .hero-subtitle {
            font-size: 16px;
            line-height: 1.85;
            color: var(--muted);
            max-width: 760px;
            margin-bottom: 22px;
        }

        .hero-pills {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }

        .hero-pill {
            padding: 8px 12px;
            border-radius: 999px;
            background: rgba(255,255,255,.04);
            border: 1px solid rgba(255,255,255,.06);
            color: #DCE8FF;
            font-size: 13px;
        }

        .soft-card {
            border: 1px solid var(--border);
            background: var(--surface);
            border-radius: 24px;
            padding: 18px 18px 16px;
            box-shadow: var(--shadow);
            backdrop-filter: blur(16px);
        }

        div[data-testid="stMetric"] {
            border: 1px solid var(--border);
            background: rgba(255,255,255,.03);
            border-radius: 22px;
            padding: 14px 16px;
            backdrop-filter: blur(10px);
        }

        .stButton > button, .stDownloadButton > button {
            border-radius: 999px;
            border: 1px solid rgba(137,213,255,.22);
            background: linear-gradient(180deg, rgba(137,213,255,.16), rgba(137,213,255,.08));
            color: var(--text);
            min-height: 42px;
            padding: 0 18px;
        }

        .stFileUploader, .stAlert, .stExpander {
            border-radius: 20px;
        }

        .stDataFrame, div[data-testid="stTable"] {
            border: 1px solid var(--border);
            border-radius: 18px;
            overflow: hidden;
        }

        [data-testid="stExpander"] {
            border: 1px solid var(--border);
            background: rgba(255,255,255,.025);
        }

        .section-title {
            font-size: 24px;
            font-weight: 600;
            margin: 18px 0 12px 0;
            letter-spacing: -.01em;
        }

        .section-note {
            color: var(--muted);
            font-size: 14px;
            margin-top: -4px;
            margin-bottom: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

