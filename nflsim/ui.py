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
TINT = "#9ec5f4"
FONT = "Inter, 'Segoe UI', system-ui, sans-serif"
FONT_IMPORT = "@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');"

_CSS = f"""
<style>
{FONT_IMPORT}
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
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{ color: {MUTED}; opacity: 1; }}
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
        legend={"bgcolor": "rgba(0,0,0,0)", "font": {"color": MUTED}}, title={"x": 0.02, "xanchor": "left", "font": {"size": 15, "color": INK}},
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


# ------------------------------------------------------------------ data conventions
# The rules these follow are in dashboard/STYLE_GUIDE.md.
# Distance from neutral at which a cell reaches full color, so columns of one kind compare directly.
SPREAD_SCALE = 9.0
WIN_SCALE = 0.3
POINTS_SCALE = 6.0
TOTAL_SCALE = 8.0
GAP_SCALE = 5.0
GOOD = "#1f5fae"
BAD = "#c8591a"
MIDPOINT = "#f1f2f4"
BLANK = "–"

TABLE_CSS = f"""
table.nfl-table {{ width: 100%; border-collapse: separate; border-spacing: 0; font-size: 13px; background: {SURFACE}; color: {INK};
                   border: 1px solid {BORDER}; border-radius: 8px; font-variant-numeric: tabular-nums; }}
.nfl-table th {{ font-size: 12px; font-weight: 500; color: {MUTED}; padding: 7px 12px; text-align: right; white-space: nowrap;
                 border-bottom: 1px solid {BORDER}; vertical-align: bottom; }}
.nfl-table th.group {{ font-size: 10.5px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; text-align: left;
                       border-bottom: none; padding-bottom: 0; }}
.nfl-table td {{ padding: 0 12px; height: 31px; text-align: right; white-space: nowrap; border-bottom: 1px solid {GRID}; }}
.nfl-table tbody tr:last-child td {{ border-bottom: none; }}
.nfl-table .text {{ text-align: left; }}
.nfl-table .rank {{ color: {MUTED}; }}
.nfl-table .team {{ font-weight: 600; }}
.nfl-table .lead {{ font-weight: 600; }}
.nfl-table .quiet {{ color: {MUTED}; }}
.nfl-table .start {{ border-left: 1px solid {BORDER}; }}
.nfl-table .shaded {{ font-weight: 600; min-width: 58px; }}
.nfl-table img.logo {{ width: 20px; height: 20px; object-fit: contain; vertical-align: middle; margin-right: 7px; }}
.nfl-table .at {{ color: {MUTED}; font-weight: 400; margin: 0 7px; }}
.nfl-note {{ color: {MUTED}; font-size: 12px; line-height: 1.5; margin-top: 10px; }}
.nfl-note a {{ color: inherit; }}
"""


def _channels(color: str) -> tuple:
    return tuple(int(color[position:position + 2], 16) for position in (1, 3, 5))


def _luminance(color: tuple) -> float:
    linear = [(c / 255) / 12.92 if c / 255 <= 0.03928 else ((c / 255 + 0.055) / 1.055) ** 2.4 for c in color]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(first: tuple, second: tuple) -> float:
    high, low = sorted((_luminance(first), _luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def shade(score: float) -> str:
    """Cell style from orange at -1 through gray at 0 to blue at +1, with text that stays readable. Missing stays plain."""
    if score != score:
        return ""
    amount = min(abs(score), 1.0)
    pole = _channels(GOOD if score >= 0 else BAD)
    fill = tuple(round(low + (high - low) * amount) for low, high in zip(_channels(MIDPOINT), pole))
    ink = "#fff" if _contrast(fill, (255, 255, 255)) > _contrast(fill, _channels(INK)) else INK
    return f"background-color:rgb{fill};color:{ink};"


def _text(value, template: str) -> str:
    return BLANK if value is None or value != value else template.format(value)


def signed(value) -> str:
    """One decimal with a sign and a true minus; a value that rounds to zero has no sign."""
    if value is not None and value == value and abs(value) < 0.05:
        return "0.0"
    return _text(value, "{:+.1f}").replace("-", "−")


def number(value) -> str:
    return _text(value, "{:.1f}")


def percent(value) -> str:
    """A 0-1 share as a percent with one decimal."""
    return _text(value, "{:.1%}")


def signed_percent(value) -> str:
    """A change in a 0-1 share, signed like signed()."""
    if value is not None and value == value and abs(value) < 0.0005:
        return "0.0%"
    return _text(value, "{:+.1%}").replace("-", "\u2212")


def table_cell(text: str, score: float, classes: str = "") -> str:
    """A cell of an HTML table, shaded by a score from -1 to 1."""
    style = shade(score)
    return f'<td class="{classes}"' + (f' style="{style}"' if style else "") + f">{text}</td>"


def styled_frame(frame, formats: dict, shades: dict | None = None):
    """A frame for st.dataframe that keeps sorting by value: formats by column, and (neutral value, scale) by shaded column."""
    styler = frame.style.format(formats)
    for column, (neutral, scale) in (shades or {}).items():
        styler = styler.map(lambda value, neutral=neutral, scale=scale: shade((value - neutral) / scale), subset=[column])
    return styler


def style_matplotlib() -> None:
    """Shared type and quiet axes for the charts that are images."""
    import matplotlib

    matplotlib.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"], "text.color": INK, "axes.titlecolor": INK,
        "axes.labelcolor": MUTED, "axes.edgecolor": BORDER, "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURFACE,
    })


# ------------------------------------------------------------------ images for articles
# Substack's post width; images are made at twice this so they stay sharp on high-density screens.
ARTICLE_WIDTH = 728
PLOTLY_JS = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@4.0.0/plotly.min.js"
HTML_TO_IMAGE_JS = "https://cdn.jsdelivr.net/npm/html-to-image@1.11.13/dist/html-to-image.js"

_BUTTONS_CSS = f"""
{FONT_IMPORT}
html, body {{ margin: 0; padding: 0; background: transparent; font-family: {FONT}; }}
.nfl-share {{ display: flex; gap: 8px; align-items: center; padding: 2px 0; }}
.nfl-share button {{ font: 500 12.5px {FONT}; color: {INK}; background: {SURFACE}; border: 1px solid {BORDER};
                     border-radius: 6px; padding: 5px 11px; cursor: pointer; }}
.nfl-share button:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}
.nfl-share .note {{ font-size: 12px; color: {MUTED}; }}
"""

# Copies or saves a PNG made by makeBlob(). The clipboard gets a promise so the click still counts as the user's
# gesture while the image renders.
_BUTTONS_JS = """
const note = document.getElementById('note');
function say(text) { note.textContent = text; clearTimeout(window.sayTimer); window.sayTimer = setTimeout(() => note.textContent = '', 4000); }
document.getElementById('copy').onclick = async () => {
  try {
    await navigator.clipboard.write([new ClipboardItem({'image/png': makeBlob()})]);
    say('Copied. Paste it into your post.');
  } catch (error) { say('This browser blocked the copy. Use Download PNG.'); console.error(error); }
};
document.getElementById('save').onclick = async () => {
  try {
    const url = URL.createObjectURL(await makeBlob());
    const link = Object.assign(document.createElement('a'), {href: url, download: FILE_NAME});
    document.body.appendChild(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
  } catch (error) { say('Could not make the image.'); console.error(error); }
};
"""

_BUTTONS = ('<div class="nfl-share"><button id="copy" title="Copy the chart as an image, ready to paste into Substack">'
            'Copy image</button><button id="save" title="Save the chart as a PNG">Download PNG</button>'
            '<span class="note" id="note"></span></div>')


# A browser turns a chart into a PNG through an image, which can't see web fonts such as Inter, so exports name fonts
# the computer has: Segoe UI on Windows, Helvetica Neue on a Mac.
EXPORT_FONT = "'Segoe UI', 'Helvetica Neue', Arial, sans-serif"


def _system_fonts(value):
    """A Plotly figure spec with every font family swapped for EXPORT_FONT."""
    if isinstance(value, dict):
        return {key: EXPORT_FONT if key == "family" else _system_fonts(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_system_fonts(item) for item in value]
    return value


def _file_name(name: str) -> str:
    import re

    stem = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower() or "chart"
    return f"{stem}.png"


def _script_json(value) -> str:
    """JSON that is safe inside a <script> element."""
    import json

    return json.dumps(value).replace("</", "<\/")


def _share_html(make_blob_js: str, name: str, head: str = "") -> str:
    return (f"<!doctype html><html><head><meta charset='utf-8'><style>{_BUTTONS_CSS}</style>{head}</head><body>{_BUTTONS}"
            f"<script>const FILE_NAME = {_script_json(_file_name(name))};\n{make_blob_js}\n{_BUTTONS_JS}</script></body></html>")


def figure_png(figure, width: int = ARTICLE_WIDTH) -> bytes:
    """A Matplotlib figure as a PNG twice the article width, on the chart surface."""
    import io

    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=2 * width / figure.get_figwidth(), facecolor=SURFACE, bbox_inches="tight")
    return buffer.getvalue()


def image_actions(source, name: str, *, width: int = ARTICLE_WIDTH, container=None) -> None:
    """'Copy image' and 'Download PNG' buttons for a chart, sized for a Substack post.

    source is a Plotly figure (drawn into a PNG in the browser at the article width), a Matplotlib figure, or PNG bytes.
    Copying needs a secure page: localhost or https (Tailscale serve), not a bare network address.
    """
    import base64

    import streamlit.components.v1 as components

    if hasattr(source, "to_plotly_json"):
        import json

        import plotly.io as pio

        spec = _system_fonts(json.loads(pio.to_json(source, validate=False)))  # Plotly's encoder handles numpy arrays
        layout = dict(spec.get("layout", {}))
        layout["font"] = {**layout.get("font", {}), "family": EXPORT_FONT}
        height = int(layout.get("height") or 450)
        layout.update(width=width, height=height)
        head = f"<script src='{PLOTLY_JS}'></script>"
        make_blob = (f"const FIGURE = {_script_json({'data': spec.get('data', []), 'layout': layout})};\n"
                     "async function makeBlob() {\n"
                     "  await document.fonts.load('12px Inter').catch(() => null);\n"
                     "  const holder = document.createElement('div');\n"
                     "  holder.style.cssText = 'position:absolute;left:-10000px;top:0';\n"
                     "  document.body.appendChild(holder);\n"
                     "  try {\n"
                     "    await Plotly.newPlot(holder, FIGURE.data, FIGURE.layout, {staticPlot: true});\n"
                     f"    const url = await Plotly.toImage(holder, {{format: 'png', width: {width}, height: {height}, scale: 2}});\n"
                     "    return await (await fetch(url)).blob();\n"
                     "  } finally { Plotly.purge(holder); holder.remove(); }\n"
                     "}")
    else:
        png = source if isinstance(source, (bytes, bytearray)) else figure_png(source, width)
        head = ""
        make_blob = (f"const PNG = {_script_json(base64.b64encode(png).decode('ascii'))};\n"
                     "async function makeBlob() { return await (await fetch('data:image/png;base64,' + PNG)).blob(); }")
    with (container or st.container()):
        components.html(_share_html(make_blob, name, head), height=38)


def html_with_image_actions(page: str, name: str) -> str:
    """An HTML page (a styled table, say) with 'Copy image' and 'Download PNG' buttons above it that capture the whole
    page except the buttons, on a white background at twice its size, for components.html.

    The page becomes an SVG (fonts and images inlined) that is drawn onto a canvas here rather than by the library,
    whose own drawing waits for an animation frame that never comes in a hidden or background tab."""
    make_blob = ("async function makeBlob() {\n"
                 "  const page = document.getElementById('nfl-capture');\n"
                 "  await document.fonts.ready;\n"
                 "  const width = page.scrollWidth, height = page.scrollHeight;\n"
                 f"  const svg = await htmlToImage.toSvg(page, {{backgroundColor: '{SURFACE}', width, height,\n"
                 "    filter: node => node.tagName !== 'SCRIPT'});\n"
                 "  const image = new Image();\n"
                 "  image.src = svg;\n"
                 "  await image.decode();\n"
                 "  const canvas = Object.assign(document.createElement('canvas'), {width: 2 * width, height: 2 * height});\n"
                 "  const context = canvas.getContext('2d');\n"
                 "  context.scale(2, 2);\n"
                 "  context.drawImage(image, 0, 0, width, height);\n"
                 "  return await new Promise((done, fail) => canvas.toBlob(blob => blob ? done(blob) : fail(new Error('No image')), 'image/png'));\n"
                 "}")
    tools = (f"<style>{_BUTTONS_CSS.replace('html, body', '.nfl-share-host')}</style>{_BUTTONS}"
             f"<script src='{HTML_TO_IMAGE_JS}'></script>"
             f"<script>const FILE_NAME = {_script_json(_file_name(name))};\n{make_blob}\n{_BUTTONS_JS}</script>")
    opening = "<div id='nfl-capture' style='display: flow-root; padding: 12px'>"
    if "<body>" in page and "</body>" in page:
        return page.replace("<body>", "<body>" + tools + opening, 1).replace("</body>", "</div></body>", 1)
    return tools + opening + page + "</div>"
