"""
ui_kit.py
---------
Design system for the dashboard: color tokens, typography, injected CSS,
a shared Plotly theme, and the signature "process rail" HTML visual that
represents the T1-T6 workflow as a proportionally-scaled timeline.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
INK = "#14213D"
PAPER = "#F7F5F0"
SURFACE = "#FFFFFF"
BRAND = "#145C64"      # deep teal — primary
BRAND_DARK = "#0D3E44"
SLOW = "#C1442D"        # rust red — bottleneck / breach
FAST = "#4C8C4A"        # moss green — healthy
AMBER = "#D8A13A"       # muted gold — caution
MUTED = "#6B7280"
BORDER = "#E7E3D8"

FONT_DISPLAY = "'Space Grotesk', sans-serif"
FONT_BODY = "'IBM Plex Sans', sans-serif"
FONT_MONO = "'IBM Plex Mono', monospace"

CHART_COLORWAY = [BRAND, SLOW, AMBER, FAST, "#5B7FA6", "#8E5B6E"]


def inject_css():
    # Menggunakan st.html untuk menyuntikkan style secara murni
    st.html(
        f"""
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght=500;600;700&family=IBM+Plex+Sans:wght=400;500;600&family=IBM+Plex+Mono:wght=500;600&display=swap" rel="stylesheet">
        <style>
        html, body, [class*="css"] {{
            font-family: {FONT_BODY};
            color: {INK};
        }}
        .stApp {{
            background-color: {PAPER};
        }}
        h1, h2, h3 {{
            font-family: {FONT_DISPLAY};
            color: {INK};
            letter-spacing: -0.01em;
        }}
        [data-testid="stSidebar"] {{
            background-color: {BRAND_DARK};
        }}
        [data-testid="stSidebar"] * {{
            color: #EDEFEE !important;
        }}
        [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown p {{
            color: #C9D6D4 !important;
        }}
        .kicker {{
            font-family: {FONT_MONO};
            font-size: 0.75rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: {BRAND};
            font-weight: 600;
        }}
        .app-header {{
            padding: 0.25rem 0 1.25rem 0;
            border-bottom: 1px solid {BORDER};
            margin-bottom: 1.5rem;
        }}
        .app-title {{
            font-family: {FONT_DISPLAY};
            font-size: 2.1rem;
            font-weight: 700;
            color: {INK};
            margin: 0.2rem 0 0.1rem 0;
        }}
        .app-subtitle {{
            color: {MUTED};
            font-size: 0.95rem;
            max-width: 780px;
        }}

        /* KPI cards */
        .kpi-card {{
            background: {SURFACE};
            border: 1px solid {BORDER};
            border-radius: 10px;
            padding: 1rem 1.1rem;
            height: 100%;
        }}
        .kpi-label {{
            font-family: {FONT_MONO};
            font-size: 0.72rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: {MUTED};
            margin-bottom: 0.3rem;
        }}
        .kpi-value {{
            font-family: {FONT_DISPLAY};
            font-size: 1.65rem;
            font-weight: 700;
            color: {INK};
            line-height: 1.15;
        }}
        .kpi-delta {{
            font-size: 0.8rem;
            margin-top: 0.25rem;
            font-weight: 500;
        }}
        .kpi-delta.good {{ color: {FAST}; }}
        .kpi-delta.bad {{ color: {SLOW}; }}
        .kpi-delta.neutral {{ color: {MUTED}; }}

        /* Section card */
        .section-card {{
            background: {SURFACE};
            border: 1px solid {BORDER};
            border-radius: 12px;
            padding: 1.25rem 1.4rem;
            margin-bottom: 1rem;
        }}

        /* Recommendation card */
        .rec-card {{
            background: {SURFACE};
            border: 1px solid {BORDER};
            border-left: 5px solid {BRAND};
            border-radius: 8px;
            padding: 1rem 1.2rem;
            margin-bottom: 0.9rem;
        }}
        .rec-card.urgent {{ border-left-color: {SLOW}; }}
        .rec-dept {{
            font-family: {FONT_MONO};
            font-size: 0.72rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: {MUTED};
        }}
        .rec-title {{
            font-family: {FONT_DISPLAY};
            font-weight: 700;
            font-size: 1.15rem;
            color: {INK};
            margin: 0.15rem 0 0.4rem 0;
        }}
        .pill {{
            display: inline-block;
            font-family: {FONT_MONO};
            font-size: 0.72rem;
            font-weight: 600;
            padding: 0.15rem 0.55rem;
            border-radius: 999px;
            margin-right: 0.4rem;
        }}
        .pill.slow {{ background: #FBE7E2; color: {SLOW}; }}
        .pill.fast {{ background: #E7F0E6; color: {FAST}; }}
        .pill.amber {{ background: #FBF0DC; color: {AMBER}; }}

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 4px;
        }}
        .stTabs [data-baseweb="tab"] {{
            font-family: {FONT_DISPLAY};
            font-weight: 600;
            font-size: 0.95rem;
        }}

        /* Process rail */
        .rail-wrap {{
            padding: 0.5rem 0 0.2rem 0;
        }}
        .rail-track {{
            display: flex;
            align-items: stretch;
            width: 100%;
            gap: 3px;
        }}
        .rail-node {{
            flex: 0 0 auto;
            width: 108px;
            text-align: center;
        }}
        .rail-node-badge {{
            font-family: {FONT_MONO};
            font-weight: 600;
            font-size: 0.75rem;
            background: {INK};
            color: {PAPER};
            border-radius: 6px;
            padding: 3px 0;
            margin-bottom: 4px;
        }}
        .rail-node-label {{
            font-size: 0.72rem;
            color: {INK};
            font-weight: 600;
            line-height: 1.2;
        }}
        .rail-node-dept {{
            font-size: 0.65rem;
            color: {MUTED};
        }}
        .rail-hop {{
            flex: 1 1 auto;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            min-width: 40px;
        }}
        .rail-hop-bar {{
            width: 100%;
            border-radius: 999px;
            position: relative;
        }}
        .rail-hop-time {{
            font-family: {FONT_MONO};
            font-size: 0.68rem;
            font-weight: 600;
            margin-top: 3px;
            white-space: nowrap;
        }}
        </style>
        """
    )


def set_plotly_theme():
    pio.templates["multifinance"] = go.layout.Template(
        layout=go.Layout(
            font=dict(family=FONT_BODY, color=INK, size=13),
            title_font=dict(family=FONT_DISPLAY, size=16, color=INK),
            paper_bgcolor=SURFACE,
            plot_bgcolor=SURFACE,
            colorway=CHART_COLORWAY,
            xaxis=dict(gridcolor="#EFEBE0", zerolinecolor=BORDER),
            yaxis=dict(gridcolor="#EFEBE0", zerolinecolor=BORDER),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=10, r=10, t=50, b=10),
        )
    )
    pio.templates.default = "multifinance"


def kpi_card(label: str, value: str, delta: str = None, delta_kind: str = "neutral"):
    delta_html = f'<div class="kpi-delta {delta_kind}">{delta}</div>' if delta else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def process_rail(stage_meta, hop_hours, sla_ok_max=12, sla_warn_max=20):
    """Render the signature horizontal process-rail visual.

    stage_meta: list of dicts with code/label/dept (STAGE_META)
    hop_hours: list of average hours for each of the 5 hops between stages
    """
    max_hop = max(hop_hours) if hop_hours else 1
    nodes_html = ""
    for i, stage in enumerate(stage_meta):
        nodes_html += f"""
        <div class="rail-node">
            <div class="rail-node-badge">{stage['code']}</div>
            <div class="rail-node-label">{stage['label']}</div>
            <div class="rail-node-dept">{stage['dept']}</div>
        </div>
        """
        if i < len(hop_hours):
            hop_h = hop_hours[i]
            width_pct = max(8, min(100, (hop_h / max_hop) * 100))
            if hop_h <= sla_ok_max:
                color = FAST
            elif hop_h <= sla_warn_max:
                color = AMBER
            else:
                color = SLOW
            nodes_html += f"""
            <div class="rail-hop">
                <div class="rail-hop-bar" style="height:8px; background:{color}; width:{width_pct}%; opacity:0.85;"></div>
                <div class="rail-hop-time" style="color:{color};">{hop_h:.1f} j</div>
            </div>
            """
    st.html(
        f"""
        <div class="rail-wrap">
            <div class="rail-track">{nodes_html}</div>
        </div>
        """
    )
