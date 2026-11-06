"""Shared look for the Streamlit dashboards: page chrome, CSS and chart styling."""

from __future__ import annotations

import html

import streamlit as st

# Categorical series colors, in fixed order (validated for colour-vision deficiency on adjacent pairs).
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK = "#1d2327"
MUTED = "#5b6670"
GRID = "#e8ebee"
BORDER = "#dde1e5"
SURFACE = "#ffffff"
CANVAS = "#f6f7f9"
ACCENT = "#1f5fae"
FONT = "Inter, 'Segoe UI', system-ui, sans-serif"

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, .stApp, [class*="st-"], button, input, select, textarea {{ font-family: {FONT}; }}
[data-testid="stIconMaterial"] {{ font-family: 'Material Symbols Rounded' !important; }}
.stApp {{ background: {CANVAS}; color: {INK}; }}
[data-testid="stHeader"] {{ background: transparent; }}
.stAppDeployButton, [data-testid="stDecoration"] {{ display: none; }}
.block-container, .stMainBlockContainer {{ max-width: 1320px; padding-top: 2.25rem; padding-bottom: 4rem; }}
h1, h2, h3, h4 {{ color: {INK}; letter-spacing: -0.01em; }}
h2 {{ font-size: 1.25rem !important; font-weight: 600 !important; }}
h3 {{ font-size: 1.05rem !important; font-weight: 600 !important; }}
[data-testid="stMetricValue"], [data-testid="stDataFrame"] {{ font-variant-numeric: tabular-nums; }}

.nfl-header {{ display: flex; align-items: flex-end; justify-content: space-between; gap: 1rem;
               border-bottom: 1px solid {BORDER}; padding-bottom: 0.9rem; margin-bottom: 1.4rem; }}
.nfl-header .eyebrow {{ font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;
                        color: {ACCENT}; margin-bottom: 0.2rem; }}
.nfl-header .title {{ font-size: 1.65rem; font-weight: 700; line-height: 1.2; color: {INK}; }}
.nfl-header .subtitle {{ color: {MUTED}; font-size: 0.92rem; margin-top: 0.3rem; }}
.nfl-header > div:first-child {{ flex: 1 1 auto; min-width: 0; }}
.nfl-header .meta {{ display: flex; flex-wrap: wrap; gap: 0.4rem; justify-content: flex-end; max-width: 45%; }}
.nfl-chip {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 999px; padding: 0.18rem 0.65rem;
             font-size: 0.78rem; color: {MUTED}; white-space: nowrap; }}
.nfl-chip b {{ color: {INK}; font-weight: 600; }}
.nfl-section {{ font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;
                color: {MUTED}; margin: 1.1rem 0 0.35rem; }}

[data-testid="stSidebar"] {{ background: {SURFACE}; border-right: 1px solid {BORDER}; }}
[data-testid="stSidebar"] .nfl-section:first-child {{ margin-top: 0; }}
[data-testid="stExpander"] details {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 8px; }}
[data-testid="stExpander"] summary {{ font-weight: 500; }}
[data-testid="stMetric"] {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 8px; padding: 0.75rem 1rem; }}
[data-testid="stMetricLabel"] {{ color: {MUTED}; }}
[data-testid="stMetricValue"] {{ font-size: 1.45rem; font-weight: 600; }}
[data-testid="stDataFrame"], [data-testid="stTable"] {{ border: 1px solid {BORDER}; border-radius: 8px; overflow: hidden; }}
[data-testid="stPlotlyChart"], [data-testid="stVegaLiteChart"] {{ background: {SURFACE}; border: 1px solid {BORDER};
                                                               border-radius: 8px; padding: 0.25rem; overflow: hidden; }}
.stButton button, .stDownloadButton button {{ border-radius: 6px; font-weight: 500; }}
[data-testid="stAlert"] {{ border-radius: 8px; }}
[data-testid="stCaptionContainer"] {{ color: {MUTED}; }}
</style>
"""


def page(title: str, icon: str = "🏈") -> None:
    """Configure the page and inject the shared stylesheet. Call first, before any other Streamlit output."""
    st.set_page_config(page_title=title, page_icon=icon, layout="wide", initial_sidebar_state="expanded")
    st.markdown(_CSS, unsafe_allow_html=True)


def header(title: str, subtitle: str | None = None, meta: dict[str, object] | None = None, *,
           eyebrow: str = "NFL modeling") -> None:
    """Draw the page title block, with optional key/value chips (bundle, season, week...) on the right."""
    chips = "".join(f'<span class="nfl-chip">{html.escape(str(key))} <b>{html.escape(str(value))}</b></span>'
                    for key, value in (meta or {}).items() if value not in (None, ""))
    sub = f'<div class="subtitle">{html.escape(subtitle)}</div>' if subtitle else ""
    st.markdown(f'<div class="nfl-header"><div><div class="eyebrow">{html.escape(eyebrow)}</div>'
                f'<div class="title">{html.escape(title)}</div>{sub}</div>'
                f'<div class="meta">{chips}</div></div>', unsafe_allow_html=True)


def section(label: str, container=None) -> None:
    """Small uppercase label that groups the controls under it."""
    (container or st).markdown(f'<div class="nfl-section">{html.escape(label)}</div>', unsafe_allow_html=True)


def style_plotly(figure, *, height: int | None = None):
    """Apply the shared typography, grid and hover styling to a Plotly figure."""
    figure.update_layout(
        font={"family": FONT, "color": INK, "size": 12}, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        colorway=list(SERIES), hoverlabel={"bgcolor": SURFACE, "bordercolor": BORDER, "font": {"family": FONT, "color": INK}},
        legend={"bgcolor": "rgba(0,0,0,0)", "font": {"color": MUTED}}, title={"font": {"size": 15, "color": INK}},
        **({"height": height} if height else {}),
    )
    figure.update_xaxes(gridcolor=GRID, linecolor=BORDER, tickcolor=BORDER, zerolinecolor=BORDER,
                        tickfont={"color": MUTED}, title_font={"color": MUTED})
    figure.update_yaxes(gridcolor=GRID, linecolor=BORDER, tickcolor=BORDER, zerolinecolor=BORDER,
                        tickfont={"color": MUTED}, title_font={"color": MUTED})
    return figure


def style_altair(chart):
    """Apply the shared typography and quiet axes to an Altair chart."""
    return (chart.configure(font=FONT, background=SURFACE)
            .configure_view(strokeWidth=0)
            .configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor=GRID, domainColor=BORDER, tickColor=BORDER,
                            labelFontSize=11, titleFontSize=11)
            .configure_legend(labelColor=MUTED, titleColor=MUTED))
