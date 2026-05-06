"""
╔══════════════════════════════════════════════════════════════════╗
║         NEXUS — Customer Churn Intelligence Platform             ║ 
╚══════════════════════════════════════════════════════════════════╝
"""

import logging
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import settings  # noqa: E402
from src.serving import predict  # noqa: E402
from src.utils import validate_input  # noqa: E402

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NEXUS · Churn Intelligence",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────
# GLOBAL CSS — Sci-Fi Dark Theme with Animations
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── IMPORT FONTS ── */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600&family=Share+Tech+Mono&display=swap');

/* ── CSS VARIABLES ── */
:root {
    --neon-cyan:    #00f5ff;
    --neon-green:   #39ff14;
    --neon-red:     #ff2d55;
    --neon-amber:   #ffb300;
    --bg-deep:      #020408;
    --bg-panel:     #060d18;
    --bg-card:      #0a1628;
    --bg-input:     #071020;
    --grid-line:    rgba(0, 245, 255, 0.07);
    --border-glow:  rgba(0, 245, 255, 0.3);
    --text-primary: #e0f4ff;
    --text-muted:   #4a7a9b;
    --text-dim:     #1e4060;
}

/* ── GLOBAL RESET ── */
* { box-sizing: border-box; }

/* ── MAIN BACKGROUND ── */
.stApp {
    background-color: var(--bg-deep) !important;
    background-image:
        linear-gradient(var(--grid-line) 1px, transparent 1px),
        linear-gradient(90deg, var(--grid-line) 1px, transparent 1px),
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(0,180,255,0.12) 0%, transparent 70%),
        radial-gradient(ellipse 50% 80% at 90% 50%, rgba(0,255,180,0.04) 0%, transparent 60%);
    background-size: 40px 40px, 40px 40px, 100% 100%, 100% 100%;
    font-family: 'Rajdhani', sans-serif;
    color: var(--text-primary) !important;
}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020c1a 0%, #040f20 100%) !important;
    border-right: 1px solid var(--border-glow) !important;
    box-shadow: 4px 0 30px rgba(0, 245, 255, 0.06) !important;
}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p {
    color: var(--text-primary) !important;
}

/* ── HIDE STREAMLIT CHROME ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; max-width: 1400px !important; }

/* ── TYPOGRAPHY ── */
h1, h2, h3 { font-family: 'Orbitron', monospace !important; letter-spacing: 0.08em; }
.stMarkdown p, .stMarkdown li { font-family: 'Rajdhani', sans-serif; font-size: 1.05rem; line-height: 1.7; }

/* ── ANIMATED HERO HEADER ── */
.nexus-hero {
    position: relative;
    text-align: center;
    padding: 3rem 2rem 2rem;
    overflow: hidden;
}
.nexus-hero::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse 60% 80% at 50% 50%, rgba(0,245,255,0.08) 0%, transparent 70%);
    animation: pulse-bg 4s ease-in-out infinite;
}
@keyframes pulse-bg {
    0%, 100% { opacity: 0.5; transform: scale(1); }
    50%       { opacity: 1;   transform: scale(1.05); }
}
.nexus-logo {
    font-family: 'Orbitron', monospace;
    font-size: clamp(2.8rem, 6vw, 5rem);
    font-weight: 900;
    letter-spacing: 0.2em;
    background: linear-gradient(135deg, #00f5ff 0%, #ffffff 40%, #00f5ff 80%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-shadow: none;
    animation: logo-shimmer 3s linear infinite;
    background-size: 200% auto;
}
@keyframes logo-shimmer {
    0%   { background-position: 0% center; }
    100% { background-position: 200% center; }
}
.nexus-tagline {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.85rem;
    color: var(--neon-cyan);
    letter-spacing: 0.35em;
    margin-top: 0.5rem;
    opacity: 0.7;
    text-transform: uppercase;
    animation: fade-in-up 1s ease both;
}
@keyframes fade-in-up {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 0.7; transform: translateY(0); }
}
.nexus-divider {
    width: 180px;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--neon-cyan), transparent);
    margin: 1.2rem auto;
    animation: divider-glow 2s ease-in-out infinite alternate;
}
@keyframes divider-glow {
    from { opacity: 0.4; width: 120px; }
    to   { opacity: 1;   width: 200px; }
}

/* ── SECTION HEADERS ── */
.section-header {
    font-family: 'Orbitron', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.3em;
    color: var(--neon-cyan);
    text-transform: uppercase;
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 1.8rem 0 1rem;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border-glow), transparent);
}
.section-icon {
    font-size: 1.1rem;
    filter: drop-shadow(0 0 6px var(--neon-cyan));
}

/* ── STAT BADGE ── */
.stat-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(0,245,255,0.06);
    border: 1px solid rgba(0,245,255,0.15);
    border-radius: 4px;
    padding: 4px 12px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.78rem;
    color: var(--neon-cyan);
    letter-spacing: 0.05em;
}

/* ── INPUT WIDGETS ── */
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stSlider"] {
    background: var(--bg-input) !important;
    border: 1px solid rgba(0, 245, 255, 0.18) !important;
    border-radius: 4px !important;
    color: var(--text-primary) !important;
    font-family: 'Share Tech Mono', monospace !important;
    transition: border-color 0.3s, box-shadow 0.3s !important;
}
div[data-testid="stNumberInput"] input:focus,
div[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: var(--neon-cyan) !important;
    box-shadow: 0 0 12px rgba(0, 245, 255, 0.2) !important;
    outline: none !important;
}
label, .stSlider label {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    color: var(--text-muted) !important;
    text-transform: uppercase !important;
}

/* ── SLIDER TRACK ── */
div[data-testid="stSlider"] [role="slider"] {
    background: var(--neon-cyan) !important;
    box-shadow: 0 0 8px var(--neon-cyan) !important;
}

/* ── PREDICT BUTTON ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #003d4d 0%, #001a26 100%) !important;
    border: 1px solid var(--neon-cyan) !important;
    color: var(--neon-cyan) !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    padding: 1rem 2rem !important;
    border-radius: 4px !important;
    transition: all 0.3s ease !important;
    position: relative !important;
    overflow: hidden !important;
    box-shadow: 0 0 20px rgba(0,245,255,0.15), inset 0 0 20px rgba(0,245,255,0.05) !important;
}
.stButton > button::before {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(0,245,255,0.15), transparent);
    transition: left 0.5s ease;
}
.stButton > button:hover::before { left: 100%; }
.stButton > button:hover {
    box-shadow: 0 0 40px rgba(0,245,255,0.4), inset 0 0 30px rgba(0,245,255,0.1) !important;
    transform: translateY(-1px) !important;
    border-color: #00f5ff !important;
    color: #ffffff !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── METRICS ── */
div[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid rgba(0,245,255,0.12) !important;
    border-radius: 6px !important;
    padding: 1rem !important;
    position: relative !important;
    overflow: hidden !important;
}
div[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: var(--neon-cyan);
    box-shadow: 0 0 8px var(--neon-cyan);
}
div[data-testid="stMetric"] label {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.7rem !important;
    color: var(--text-muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-family: 'Orbitron', monospace !important;
    font-size: 1.4rem !important;
    color: var(--neon-cyan) !important;
}

/* ── RESULT CARDS ── */
.result-card {
    border-radius: 8px;
    padding: 2.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    animation: card-appear 0.6s cubic-bezier(0.16, 1, 0.3, 1) both;
    margin: 1rem 0;
}
@keyframes card-appear {
    from { opacity: 0; transform: translateY(20px) scale(0.97); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}
.result-card.high-risk {
    background: linear-gradient(135deg, rgba(255,45,85,0.08) 0%, rgba(10,22,40,0.95) 100%);
    border: 1px solid rgba(255,45,85,0.4);
    box-shadow: 0 0 40px rgba(255,45,85,0.15), inset 0 0 60px rgba(255,45,85,0.04);
}
.result-card.low-risk {
    background: linear-gradient(135deg, rgba(57,255,20,0.06) 0%, rgba(10,22,40,0.95) 100%);
    border: 1px solid rgba(57,255,20,0.35);
    box-shadow: 0 0 40px rgba(57,255,20,0.12), inset 0 0 60px rgba(57,255,20,0.03);
}
.result-card::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: conic-gradient(from 0deg, transparent 0deg, rgba(255,255,255,0.01) 60deg, transparent 120deg);
    animation: card-rotate 8s linear infinite;
}
@keyframes card-rotate { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }

.result-icon {
    font-size: 3.5rem;
    display: block;
    margin-bottom: 0.5rem;
    animation: icon-pulse 2s ease-in-out infinite;
    filter: drop-shadow(0 0 20px currentColor);
}
@keyframes icon-pulse {
    0%, 100% { transform: scale(1); }
    50%       { transform: scale(1.08); }
}
.result-label {
    font-family: 'Orbitron', monospace;
    font-size: 1.6rem;
    font-weight: 900;
    letter-spacing: 0.12em;
    margin: 0.3rem 0;
}
.result-label.danger  { color: var(--neon-red); text-shadow: 0 0 20px rgba(255,45,85,0.6); }
.result-label.success { color: var(--neon-green); text-shadow: 0 0 20px rgba(57,255,20,0.6); }
.result-sub {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.1rem;
    color: var(--text-muted);
    margin-top: 0.3rem;
}

/* ── PROBABILITY RING ── */
.prob-ring-wrap {
    display: flex;
    justify-content: center;
    align-items: center;
    margin: 1.5rem 0;
}
.prob-ring-wrap svg { overflow: visible; }

/* ── RECOMMENDATION CARD ── */
.rec-card {
    background: rgba(0,245,255,0.04);
    border: 1px solid rgba(0,245,255,0.12);
    border-left: 3px solid var(--neon-cyan);
    border-radius: 6px;
    padding: 1.2rem 1.5rem;
    margin: 0.7rem 0;
    font-family: 'Rajdhani', sans-serif;
    font-size: 1rem;
    color: var(--text-primary);
    transition: border-color 0.3s, background 0.3s;
    animation: slide-in 0.5s ease both;
}
.rec-card:hover {
    background: rgba(0,245,255,0.08);
    border-left-color: #00f5ff;
}
.rec-card.danger-rec {
    border-left-color: var(--neon-red);
    background: rgba(255,45,85,0.04);
}
@keyframes slide-in {
    from { opacity: 0; transform: translateX(-10px); }
    to   { opacity: 1; transform: translateX(0); }
}

/* ── PRIORITY BADGE ── */
.priority-badge {
    display: inline-block;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    padding: 2px 10px;
    border-radius: 3px;
    letter-spacing: 0.1em;
    font-weight: bold;
    vertical-align: middle;
    margin-left: 8px;
}
.badge-critical { background: rgba(255,45,85,0.2); color: #ff6b8a; border: 1px solid rgba(255,45,85,0.4); }
.badge-high     { background: rgba(255,179,0,0.15); color: var(--neon-amber); border: 1px solid rgba(255,179,0,0.35); }
.badge-medium   { background: rgba(0,245,255,0.1); color: var(--neon-cyan); border: 1px solid rgba(0,245,255,0.25); }
.badge-low      { background: rgba(57,255,20,0.1); color: var(--neon-green); border: 1px solid rgba(57,255,20,0.25); }

/* ── SIDEBAR ELEMENTS ── */
.sidebar-stat {
    background: rgba(0,245,255,0.05);
    border: 1px solid rgba(0,245,255,0.12);
    border-radius: 6px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.82rem;
    color: var(--text-primary);
}
.sidebar-stat .label { color: var(--text-muted); font-size: 0.72rem; letter-spacing: 0.1em; text-transform: uppercase; }
.sidebar-stat .value { color: var(--neon-cyan); font-size: 1.1rem; font-weight: bold; }

.sidebar-logo {
    font-family: 'Orbitron', monospace;
    font-size: 1.5rem;
    font-weight: 900;
    color: var(--neon-cyan);
    letter-spacing: 0.2em;
    text-align: center;
    padding: 1rem 0 0.5rem;
    text-shadow: 0 0 20px rgba(0,245,255,0.5);
}
.sidebar-version {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    color: var(--text-muted);
    text-align: center;
    letter-spacing: 0.2em;
}

/* ── STATUS INDICATOR ── */
.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--neon-green);
    box-shadow: 0 0 6px var(--neon-green);
    animation: blink 2s ease-in-out infinite;
    margin-right: 6px;
}
@keyframes blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.3; }
}

/* ── SPINNER ── */
.stSpinner > div > div {
    border-top-color: var(--neon-cyan) !important;
}

/* ── PROGRESS BAR ── */
div[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, var(--neon-cyan), #00a8b5) !important;
    box-shadow: 0 0 8px var(--neon-cyan) !important;
}
div[data-testid="stProgress"] > div {
    background: rgba(0,245,255,0.08) !important;
    border-radius: 2px !important;
}

/* ── EXPANDER ── */
details {
    background: rgba(0,245,255,0.03) !important;
    border: 1px solid rgba(0,245,255,0.12) !important;
    border-radius: 6px !important;
}
details summary {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.82rem !important;
    color: var(--neon-cyan) !important;
    letter-spacing: 0.1em !important;
}

/* ── DIVIDER ── */
hr { border-color: rgba(0,245,255,0.1) !important; }

/* ── SCANLINES OVERLAY ── */
.stApp::after {
    content: '';
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,0,0,0.03) 2px,
        rgba(0,0,0,0.03) 4px
    );
    pointer-events: none;
    z-index: 9999;
}

/* ── CORNER BRACKETS ── */
.bracket-box {
    position: relative;
    padding: 1.5rem;
    margin: 1rem 0;
}
.bracket-box::before, .bracket-box::after,
.bracket-box > span::before, .bracket-box > span::after {
    content: '';
    position: absolute;
    width: 16px; height: 16px;
    border-color: var(--neon-cyan);
    border-style: solid;
    opacity: 0.5;
}
.bracket-box::before  { top: 0; left: 0;  border-width: 1px 0 0 1px; }
.bracket-box::after   { top: 0; right: 0; border-width: 1px 1px 0 0; }
.bracket-box > span::before { bottom: 0; left: 0;  border-width: 0 0 1px 1px; }
.bracket-box > span::after  { bottom: 0; right: 0; border-width: 0 1px 1px 0; }

/* ── COLUMNS SPACING ── */
div[data-testid="stHorizontalBlock"] { gap: 1rem !important; }

/* ── INPUT LABELS GLOW ON FOCUS ── */
div[data-testid="stNumberInput"]:focus-within label,
div[data-testid="stSelectbox"]:focus-within label {
    color: var(--neon-cyan) !important;
}

/* ── CURSOR TRAIL EFFECT (via JS) ── */
</style>

<script>
// Cursor glow effect
document.addEventListener('mousemove', function(e) {
    const glow = document.getElementById('cursor-glow');
    if (glow) {
        glow.style.left = e.clientX + 'px';
        glow.style.top = e.clientY + 'px';
    }
});
</script>

<div id="cursor-glow" style="
    position: fixed; pointer-events: none; z-index: 99998;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(0,245,255,0.04) 0%, transparent 70%);
    transform: translate(-50%, -50%);
    transition: left 0.15s ease, top 0.15s ease;
"></div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-logo">⬡ NEXUS</div>
        <div class="sidebar-version">CHURN INTELLIGENCE · v2.1.0</div>
        """, unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown("""
        <div class="sidebar-stat">
            <div class="label">Engine</div>
            <div class="value">XGBoost Classifier</div>
        </div>
        <div class="sidebar-stat">
            <div class="label">Total Features</div>
            <div class="value">10 Input Vectors</div>
        </div>
        <div class="sidebar-stat">
            <div class="label">Numeric / Categorical</div>
            <div class="value">7 &nbsp;/&nbsp; 3</div>
        </div>
        <div class="sidebar-stat">
            <div class="label">System Status</div>
            <div class="value"><span class="status-dot"></span>OPERATIONAL</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        with st.expander("◈ OPERATIONAL GUIDE", expanded=False):
            st.markdown("""
**01** · Fill all input vectors  
**02** · Initiate prediction sequence  
**03** · Analyze risk profile output  
**04** · Deploy recommended actions

---
**FIELD CONSTRAINTS**
- Age: 0 – 120 yrs
- Tenure: 0 – 100 months
- Usage Freq: 0 – 100 scale
- Support Calls: 0 – 50
- Payment Delay: 0 – 100 days
- Last Interaction: 0 – 365 days
            """)

        st.markdown("""
        <div style="position:absolute; bottom:1.5rem; left:0; right:0; text-align:center;">
            <div style="font-family:'Share Tech Mono',monospace; font-size:0.68rem; color:#1e4060; letter-spacing:0.15em;">
                MSC ML PROJECT · NEXUS ENGINE<br>
                REAL-TIME INFERENCE ACTIVE
            </div>
        </div>
        """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
# HERO HEADER
# ──────────────────────────────────────────────────────────────────
def render_header():
    st.markdown("""
    <div class="nexus-hero">
        <div class="nexus-logo">NEXUS</div>
        <div class="nexus-tagline">◈ &nbsp; Customer Churn Intelligence Platform &nbsp; ◈</div>
        <div class="nexus-divider"></div>
        <p style="font-family:'Rajdhani',sans-serif; color:#4a7a9b; font-size:1.05rem; max-width:600px; margin:0 auto;">
            Real-time predictive analytics powered by gradient-boosted ensemble learning.
            Enter customer vectors below to compute churn probability.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
# INPUT FORM
# ──────────────────────────────────────────────────────────────────
def render_input_form() -> dict:
    form_data = {}

    # ── DEMOGRAPHICS ──
    st.markdown("""
    <div class="section-header">
        <span class="section-icon">◈</span>
        Customer Demographics
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        form_data["Age"] = st.number_input(
            "Age (years)", min_value=0, max_value=120, value=30, step=1
        )
    with col2:
        form_data["Gender"] = st.selectbox(
            "Gender", options=["Male", "Female"], index=1
        )

    # ── SERVICE & USAGE ──
    st.markdown("""
    <div class="section-header">
        <span class="section-icon">◈</span>
        Service &amp; Usage Metrics
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        form_data["Subscription Type"] = st.selectbox(
            "Subscription Tier", options=["Basic", "Standard", "Premium"], index=1
        )
    with col2:
        form_data["Contract Length"] = st.selectbox(
            "Contract Length", options=["Monthly", "Quarterly", "Annual"], index=2
        )

    col1, col2 = st.columns(2)
    with col1:
        form_data["Usage Frequency"] = st.slider(
            "Usage Frequency (0–100)", min_value=0, max_value=100, value=14
        )
    with col2:
        form_data["Support Calls"] = st.number_input(
            "Support Calls", min_value=0, max_value=50, value=5
        )

    # ── BILLING ──
    st.markdown("""
    <div class="section-header">
        <span class="section-icon">◈</span>
        Account &amp; Billing Intelligence
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        form_data["Tenure"] = st.number_input(
            "Tenure (months)", min_value=0, max_value=100, value=12
        )
    with col2:
        form_data["Total Spend"] = st.number_input(
            "Total Spend ($)", min_value=0.0, value=932.0, step=10.0
        )

    col1, col2 = st.columns(2)
    with col1:
        form_data["Last Interaction"] = st.number_input(
            "Last Interaction (days ago)", min_value=0, max_value=365, value=17
        )
    with col2:
        form_data["Payment Delay"] = st.number_input(
            "Payment Delay (days)", min_value=0, max_value=100, value=18
        )

    return form_data


# ──────────────────────────────────────────────────────────────────
# SVG PROBABILITY RING
# ──────────────────────────────────────────────────────────────────
def probability_ring_svg(prob: float, is_high_risk: bool) -> str:
    pct = prob * 100
    color = "#ff2d55" if is_high_risk else "#39ff14"
    glow = "rgba(255,45,85,0.5)" if is_high_risk else "rgba(57,255,20,0.4)"
    r = 70
    circumference = 2 * 3.14159 * r
    dash = circumference * prob
    gap  = circumference - dash

    return f"""
    <div class="prob-ring-wrap">
    <svg width="200" height="200" viewBox="0 0 200 200">
        <defs>
            <filter id="glow-ring">
                <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
                <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
            </filter>
        </defs>
        <!-- Track -->
        <circle cx="100" cy="100" r="{r}" fill="none" stroke="rgba(255,255,255,0.04)"
                stroke-width="10"/>
        <!-- Tick marks -->
        {"".join([
            f'<line x1="100" y1="24" x2="100" y2="30" stroke="rgba(0,245,255,0.2)" stroke-width="1" '
            f'transform="rotate({i*36} 100 100)"/>'
            for i in range(10)
        ])}
        <!-- Progress arc -->
        <circle cx="100" cy="100" r="{r}" fill="none" stroke="{color}"
                stroke-width="10" stroke-linecap="round"
                stroke-dasharray="{dash:.1f} {gap:.1f}"
                stroke-dashoffset="{circumference * 0.25:.1f}"
                filter="url(#glow-ring)"
                style="transition: stroke-dasharray 1s ease;">
        </circle>
        <!-- Center text -->
        <text x="100" y="92" text-anchor="middle"
              font-family="Orbitron,monospace" font-size="22" font-weight="900"
              fill="{color}">{pct:.1f}%</text>
        <text x="100" y="114" text-anchor="middle"
              font-family="Share Tech Mono,monospace" font-size="9"
              fill="rgba(255,255,255,0.3)" letter-spacing="2">CHURN PROB</text>
    </svg>
    </div>
    """


# ──────────────────────────────────────────────────────────────────
# PREDICTION RESULT
# ──────────────────────────────────────────────────────────────────
def render_prediction(result: dict):
    if "error" in result:
        st.markdown(f"""
        <div style="background:rgba(255,45,85,0.08); border:1px solid rgba(255,45,85,0.35);
                    border-radius:6px; padding:1.5rem; text-align:center; margin:1rem 0;">
            <div style="font-family:'Orbitron',monospace; color:#ff2d55; font-size:1rem;
                        letter-spacing:0.15em;">⚠ INFERENCE ERROR</div>
            <div style="font-family:'Share Tech Mono',monospace; color:#4a7a9b;
                        font-size:0.85rem; margin-top:0.5rem;">{result['error']}</div>
        </div>
        """, unsafe_allow_html=True)
        return

    prediction   = result["prediction"]
    probability  = result["churn_probability"]
    contract     = result.get("contract_type", "Unknown")
    is_high      = prediction == 1

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── RESULT CARD ──
    if is_high:
        st.markdown(f"""
        <div class="result-card high-risk"><span></span>
            <span class="result-icon">⬡</span>
            <div class="result-label danger">HIGH CHURN RISK</div>
            <div class="result-sub">Customer is at elevated probability of disengagement</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-card low-risk"><span></span>
            <span class="result-icon">◈</span>
            <div class="result-label success">LOW CHURN RISK</div>
            <div class="result-sub">Customer exhibits strong retention signals</div>
        </div>
        """, unsafe_allow_html=True)

    # ── PROBABILITY RING + METRICS ──
    col_ring, col_metrics = st.columns([1, 2])

    with col_ring:
        st.markdown(probability_ring_svg(probability, is_high), unsafe_allow_html=True)

    with col_metrics:
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric(
                "Churn Probability",
                f"{probability * 100:.2f}%",
                delta=f"{(probability - 0.5)*100:+.1f}% vs baseline"
            )
        with col_b:
            if is_high:
                risk_label = "CRITICAL" if probability > 0.8 else "HIGH"
            else:
                risk_label = "STABLE"
            st.metric("Risk Classification", risk_label)

        col_c, col_d = st.columns(2)
        with col_c:
            loyalty = (1 - probability) * 100
            st.metric("Loyalty Index", f"{loyalty:.1f}")
        with col_d:
            st.metric("Contract Tier", contract.replace("ly", "").upper() if contract else "—")

    # ── RISK PROGRESS BAR ──
    st.markdown("""
    <div class="section-header"><span class="section-icon">◈</span>Risk Intensity Vector</div>
    """, unsafe_allow_html=True)
    st.progress(probability, text=f"Churn Signal Strength · {probability:.1%}")

    # ── RECOMMENDATIONS ──
    st.markdown("""
    <div class="section-header"><span class="section-icon">◈</span>Strategic Response Playbook</div>
    """, unsafe_allow_html=True)

    if is_high:
        if "Month" in contract:
            recs = [
                ("LOCK-IN OFFER", "critical", "Immediately offer 1-year contract migration with 20–25% discount incentive. Emphasize long-term cost savings and service continuity."),
                ("OUTREACH SEQUENCE", "high",     "Trigger proactive retention call within 24h. Assign dedicated customer success manager to this account."),
                ("LOYALTY REWARD", "high",         "Offer exclusive monthly-to-annual upgrade bonus: free premium tier trial for 60 days upon commitment."),
                ("RISK MONITORING", "medium",       "Escalate to real-time churn monitoring — flag any support ticket or usage drop for immediate escalation."),
            ]
        elif "Quarterly" in contract:
            recs = [
                ("TIER UPGRADE", "high",   "Propose premium tier upgrade with 10% annual discount. Highlight exclusive features unlocked with commitment."),
                ("ENGAGEMENT BOOST", "high", "Schedule a value-review call — demonstrate ROI using their usage data and benchmark improvements."),
                ("LOYALTY PACKAGE", "medium", "Offer loyalty rewards: cashback credits, referral bonuses, or early-access to new platform features."),
            ]
        else:
            recs = [
                ("UPSELL OPPORTUNITY", "medium", "Annual customer showing risk signals — explore cross-sell or feature adoption gaps before renewal window opens."),
                ("EXECUTIVE CHECK-IN", "medium",  "Schedule executive business review — reinforce partnership value before contract renewal date."),
                ("FEATURE ADOPTION", "low",        "Analyze which premium features remain underutilized and offer guided onboarding sessions."),
            ]
        badge_map = {"critical": "badge-critical CRITICAL", "high": "badge-high HIGH", "medium": "badge-medium MEDIUM", "low": "badge-low LOW"}
        for title, level, body in recs:
            badge_cls, badge_txt = badge_map[level].split(" ", 1)
            st.markdown(f"""
            <div class="rec-card danger-rec">
                <strong style="font-family:'Orbitron',monospace; font-size:0.78rem;
                               letter-spacing:0.1em; color:#ff6b8a;">{title}</strong>
                <span class="priority-badge {badge_cls}">{badge_txt}</span>
                <div style="margin-top:0.4rem; color:#c0d8e8;">{body}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        recs = [
            ("SERVICE EXCELLENCE", "Service delivery benchmarks look solid — maintain current engagement cadence and response quality."),
            ("GROWTH OPPORTUNITY",  "Identify upsell or cross-sell pathways; this customer is receptive to value expansion conversations."),
            ("REFERRAL PROGRAM",    "High-satisfaction customers are ideal advocates — activate referral incentives to drive organic acquisition."),
            ("LOYALTY MILESTONE",   "Acknowledge tenure milestones with personalized rewards to deepen emotional brand connection."),
        ]
        for title, body in recs:
            st.markdown(f"""
            <div class="rec-card">
                <strong style="font-family:'Orbitron',monospace; font-size:0.78rem;
                               letter-spacing:0.1em; color:#00f5ff;">{title}</strong>
                <div style="margin-top:0.4rem; color:#c0d8e8;">{body}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── FOOTER STATUS BAR ──
    st.markdown("""
    <div style="margin-top:2rem; padding:0.8rem 1.2rem;
                background:rgba(0,245,255,0.03); border:1px solid rgba(0,245,255,0.08);
                border-radius:4px; display:flex; justify-content:space-between;
                font-family:'Share Tech Mono',monospace; font-size:0.72rem; color:#1e4060;">
        <span><span class="status-dot"></span>INFERENCE COMPLETE</span>
        <span>ENGINE · XGBOOST v2.1</span>
        <span>NEXUS PLATFORM · MSC ML</span>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
# VALIDATION
# ──────────────────────────────────────────────────────────────────
def validate_form(data: dict) -> bool:
    is_valid, error_msg = validate_input(data)
    if not is_valid:
        st.markdown(f"""
        <div style="background:rgba(255,45,85,0.08); border:1px solid rgba(255,45,85,0.35);
                    border-radius:6px; padding:1rem 1.5rem; margin:0.5rem 0;">
            <span style="font-family:'Share Tech Mono',monospace; color:#ff2d55;
                         font-size:0.8rem; letter-spacing:0.1em;">⚠ VALIDATION ERROR</span>
            <div style="font-family:'Rajdhani',sans-serif; color:#c0d8e8;
                        margin-top:0.3rem;">{error_msg}</div>
        </div>
        """, unsafe_allow_html=True)
        return False
    return True


# ──────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────
def main():
    render_sidebar()
    render_header()

    st.markdown("<hr>", unsafe_allow_html=True)

    form_data = render_input_form()

    # ── PREDICT BUTTON ──
    st.markdown("<div style='margin:2rem 0 1rem'>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-header"><span class="section-icon">◈</span>Initiate Prediction Sequence</div>
    """, unsafe_allow_html=True)

    if st.button("⬡  COMPUTE CHURN PROBABILITY  ⬡"):
        if not validate_form(form_data):
            st.stop()

        with st.spinner("Engaging inference engine · Analyzing customer vectors…"):
            try:
                result = predict(form_data)
            except Exception as e:
                result = {"error": str(e)}

        render_prediction(result)

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()