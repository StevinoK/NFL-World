# Dashboard style guide

Every dashboard page and report follows these rules. Colors, scales and formats live in `nflsim/ui.py`, so use the helpers there instead of writing values into a page.

## Color

| Use | Color | In `ui.py` |
|---|---|---|
| Ours, or the main series | Blue `#1f5fae` | `ACCENT` |
| Better, higher, or favors the home team | Blue `#1f5fae` | `GOOD` |
| Worse, lower, or favors the away team | Orange `#c8591a` | `BAD` |
| Neutral middle of a scale | Gray `#f1f2f4` | `MIDPOINT` |
| Several series on one chart | Fixed order, never cycled by rank | `SERIES` |
| Identifying a team | Logos, and team colors only where the team is the point of a chart | |

The score-distribution "soil to meadow" colormap is the one exception, kept as our signature chart. Other heatmaps of a single quantity use one blue ramp.

The dashboards are light only.

## Numbers

| Value | Format | Helper |
|---|---|---|
| Spreads, margins, gaps | `+3.5`, `−3.5`, `0.0` | `signed` |
| Points, totals | `27.8` | `number` |
| Percents | `67.3%` | `percent` |
| Changes in a percent | `+1.2%` | `signed_percent` |
| Missing | `–` | `BLANK` |

A spread is the named team's margin, or the home team's in a game, so positive favors that team. This is the opposite of sportsbook notation, so each table says so once in its note.

## Tables

- Color most metric columns. Leave rank, names, sample sizes, standard errors and raw data dumps plain.
- Columns of one kind share one scale, so color compares across a row:

| Column | Gray at | Full color at |
|---|---|---|
| Spreads, margins, outside ratings | 0 | ±9 points (`SPREAD_SCALE`) |
| Win % | 50% | ±30 points (`WIN_SCALE`) |
| Points for and against | League average | ±6 points (`POINTS_SCALE`), reversed for points against |
| Totals | Average Vegas total | ±8 points (`TOTAL_SCALE`) |
| Gaps | 0 | ±5 points (`GAP_SCALE`), in bold text in HTML tables |

- Use one rank column. Don't put a rank in each cell.
- Say what the colors mean in a note under the table.
- Use `styled_frame` with `st.dataframe` wherever a table should sort. Use `TABLE_CSS` and `table_cell` for HTML tables in reports.

## Charts

- Plotly for interactive charts, with `style_plotly` and `theme=None`. Altair charts use `style_altair`.
- Matplotlib only for report images and shaded score charts, with `style_matplotlib`.
- Titles sit at the top left. Label only the few points worth naming.
- Comparison charts draw a dashed gray line for agreement.

## Pages

Start with `page` and `header`, then keep this order: controls, main table or chart, details. Put dates, data sources and sample sizes in a caption.

Use the eyebrow, title and subtitle in `header`. Headings are plain sentences or short names, with no decorative dashes.
