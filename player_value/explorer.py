"""Explore saved cumulative weekly estimates without loading the fitting pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd
import player_value.positions as seasonal_positions
from player_value.positions import LOOKUP_NAME, NAMES_NAME, eligible_rows, load_lookup

DATA_DIR = Path(__file__).resolve().parent / 'outputs'
POSITIONS = ('QB', 'RB', 'WR', 'TE', 'OL', 'DL', 'LB', 'DB')
DISPLAY_COLUMNS = ['player_id', 'player_name', 'position', 'team', 'season',
                   'week', 'n_on', 'estimate', 'estimate_kind', 'period']


def prepare_estimates(frame, position):
    if frame.empty:
        return pd.DataFrame(columns=DISPLAY_COLUMNS)
    required = {'player_id', 'team', 'season', 'week', 'n_on'}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    rows = frame.copy()
    for column in ('player_id', 'team'):
        rows[column] = rows[column].astype('string').str.strip().replace('', pd.NA)
    rows['player_name'] = (rows['player_name'] if 'player_name' in rows else rows['player_id'])
    rows['player_name'] = rows['player_name'].astype('string').str.strip().replace('', pd.NA)
    rows['player_name'] = rows['player_name'].fillna(rows['player_id'])
    for column in ('season', 'week', 'n_on'):
        rows[column] = pd.to_numeric(rows[column], errors='coerce')
    bayesian = pd.to_numeric(rows.get('bayesian_plus_minus', pd.Series(np.nan, index=rows.index)), errors='coerce')
    box = pd.to_numeric(rows.get('box_plus_minus', pd.Series(np.nan, index=rows.index)), errors='coerce')
    if 'estimator' in rows:
        method = rows['estimator'].astype('string').str.strip().str.lower()
        method = method.fillna('box' if 'box_plus_minus' in rows else 'bayesian')
    else:
        method = pd.Series('box' if 'box_plus_minus' in rows else 'bayesian', index=rows.index)
    rows['estimate'] = bayesian.where(method.ne('box'), box)
    rows['estimate_kind'] = method.map({'box': 'Box', 'bayesian': 'Bayesian'})
    rows['position'] = position
    valid = (rows[['player_id', 'team', 'estimate_kind']].notna().all(axis=1)
             & np.isfinite(rows[['season', 'week', 'estimate']]).all(axis=1)
             & rows['season'].between(1900, 2200) & rows['season'].mod(1).eq(0)
             & rows['week'].between(1, 22) & rows['week'].mod(1).eq(0))
    rows = rows.loc[valid].copy()
    rows[['season', 'week']] = rows[['season', 'week']].astype(int)
    rows['n_on'] = rows['n_on'].where(np.isfinite(rows['n_on']))
    rows['period'] = rows['season'].astype(str) + ' W' + rows['week'].astype(str).str.zfill(2)
    keys = ['player_id', 'position', 'season', 'week', 'team', 'estimate_kind']
    rows = rows.drop_duplicates(keys, keep='last')
    return rows[DISPLAY_COLUMNS].sort_values(
        ['season', 'week', 'position', 'team', 'player_id'], kind='stable').reset_index(drop=True)


def file_fingerprint(directory=DATA_DIR):
    entries = []
    paths = [(position, Path(directory) / f'plus_minus_weekly_{position}.csv')
             for position in POSITIONS]
    paths.extend([('ROSTER', Path(directory) / LOOKUP_NAME), ('NAMES', Path(directory) / NAMES_NAME),
                  ('POSITION_SOURCE', Path(seasonal_positions.__file__))])
    for position, path in paths:
        try:
            metadata = path.stat()
            entries.append((position, str(path), metadata.st_mtime_ns, metadata.st_size))
        except OSError:
            entries.append((position, str(path), None, None))
    return tuple(entries)


def load_estimates(fingerprint):
    frames, issues = [], []
    roster_entry = next((entry for entry in fingerprint if entry[0] == 'ROSTER'), None)
    if roster_entry is None or roster_entry[2] is None:
        return pd.DataFrame(columns=DISPLAY_COLUMNS), ['Seasonal position lookup is missing or inaccessible.']
    try:
        lookup = load_lookup(roster_entry[1])
    except (OSError, ValueError, KeyError) as error:
        return pd.DataFrame(columns=DISPLAY_COLUMNS), [f'Unable to load seasonal position lookup ({error}).']
    wanted = {'player_id', 'player_name', 'team', 'season', 'week', 'n_on',
              'estimator', 'box_plus_minus', 'bayesian_plus_minus'}
    for position, filename, modified, size in fingerprint:
        if position not in POSITIONS:
            continue
        if modified is None:
            issues.append(f'{position}: weekly CSV is missing or inaccessible.')
            continue
        try:
            raw = pd.read_csv(filename, usecols=lambda column: column in wanted,
                              dtype={'player_id': 'string'}, low_memory=False)
            prepared = eligible_rows(prepare_estimates(raw, position), lookup, position)
            if prepared.empty:
                issues.append(f'{position}: no valid weekly estimates.')
            else:
                frames.append(prepared)
                if len(prepared) < len(raw):
                    issues.append(f'{position}: omitted {len(raw) - len(prepared):,} invalid, duplicate, or roster-ineligible rows.')
        except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
            issues.append(f'{position}: unable to load weekly CSV ({error}).')
    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=DISPLAY_COLUMNS)
    names_entry = next((entry for entry in fingerprint if entry[0] == 'NAMES'), None)
    if names_entry is not None and names_entry[2] is not None and not data.empty:
        # Estimates carry play-by-play names ("P.Mahomes"); show nickname + last name from rosters.
        names = pd.read_parquet(names_entry[1]).set_index('player_id')['player_name']
        data['player_name'] = data['player_id'].map(names).fillna(data['player_name'])
    return data, issues


def player_labels(data):
    ordered = data.sort_values(['season', 'week', 'position', 'team'], kind='stable')
    named = ordered[ordered['player_name'].ne(ordered['player_id'])]
    names = ordered.drop_duplicates('player_id', keep='last').set_index('player_id')['player_name'].to_dict()
    names.update(named.drop_duplicates('player_id', keep='last').set_index('player_id')['player_name'].to_dict())
    positions = data.groupby('player_id')['position'].agg(lambda values: ', '.join(sorted(set(values))))
    name_counts = pd.Series(list(names.values())).value_counts()
    labels = {}
    for player_id, name in names.items():
        label = f'{name} ({positions[player_id]})'
        if name_counts[name] > 1:
            label += f' [{player_id}]'
        labels[player_id] = label
    return dict(sorted(labels.items(), key=lambda item: (item[1].casefold(), item[0])))


def player_series(data, player_id):
    rows = data[data['player_id'].eq(player_id)].sort_values(
        ['season', 'week', 'position', 'team'], kind='stable').copy()
    keys = ['position', 'season', 'team', 'estimate_kind']
    rows['segment'] = rows.groupby(keys, sort=False)['week'].transform(
        lambda weeks: weeks.diff().gt(1).cumsum())
    return rows


def estimate_chart(rows):
    import plotly.graph_objects as go

    figure = go.Figure()
    colors = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948')
    keys = ['position', 'season', 'team', 'estimate_kind', 'segment']
    for trace_index, (identity, series) in enumerate(rows.groupby(keys, sort=False)):
        position, season, team, kind, segment = identity
        figure.add_trace(go.Scatter(
            x=series['period'].tolist(), y=series['estimate'].tolist(), mode='lines+markers',
            name=f'{season} / {team} / {position} / {kind}',
            line={'color': colors[trace_index % len(colors)], 'width': 2},
            marker={'size': 5}, connectgaps=False,
            customdata=series[['season', 'week', 'team', 'player_name', 'n_on', 'estimate_kind', 'position']].to_numpy(),
            hovertemplate=('%{customdata[3]}<br>Season %{customdata[0]} / Week %{customdata[1]}'
                           '<br>%{customdata[2]} / %{customdata[6]}'
                           '<br>%{customdata[5]} estimate: %{y:.4f} EPA/play'
                           '<br>n_on: %{customdata[4]}<extra></extra>'),
        ))
    kinds = sorted(rows['estimate_kind'].unique())
    figure.update_layout(
        height=470, paper_bgcolor='white', plot_bgcolor='white',
        font={'family': "Inter, 'Segoe UI', system-ui, sans-serif", 'color': '#1d2327', 'size': 12},
        margin={'l': 80, 'r': 12, 't': 12, 'b': 12},
        xaxis={'title': 'Season / Week', 'type': 'category', 'categoryorder': 'array',
               'categoryarray': rows['period'].drop_duplicates().tolist(),
               'nticks': 8, 'tickangle': -35, 'showgrid': False, 'linecolor': '#dde1e5',
               'tickfont': {'color': '#5b6670'}, 'title_font': {'color': '#5b6670'}},
         yaxis={'title': {'text': f"{' / '.join(kinds)} Plus-Minus (EPA/play)", 'standoff': 20},
             'automargin': True,
               'gridcolor': '#e8ebee', 'zerolinecolor': '#8a949c', 'zerolinewidth': 1,
               'tickfont': {'color': '#5b6670'}, 'title_font': {'color': '#5b6670'}},
        legend={'orientation': 'h', 'y': -0.3, 'x': 0, 'font': {'size': 11, 'color': '#5b6670'}},
        hoverlabel={'bgcolor': 'white', 'bordercolor': '#dde1e5'},
        hovermode='closest',
    )
    return figure


def main():
    import streamlit as st

    from nflsim import ui

    ui.page('Player Estimates')
    ui.header('Player Estimates', 'Cumulative weekly plus-minus estimates (EPA/play) by player, team and position.',
              eyebrow='Player value')
    cached_load = st.cache_data(show_spinner=False, max_entries=2)(load_estimates)
    data, issues = cached_load(file_fingerprint())
    for issue in issues:
        st.warning(issue)
    if data.empty:
        st.info('No saved weekly player estimates are available in player_value/outputs.')
        return
    labels = player_labels(data)
    picker, *tiles = st.columns([3, 1, 1, 1])
    player_id = picker.selectbox('Player', list(labels), format_func=labels.__getitem__, key='player_id')
    rows = player_series(data, player_id)
    latest = rows.iloc[-1]
    tiles[0].metric(f"Latest ({latest['period']})", f"{latest['estimate']:+.4f}")
    tiles[1].metric('Team / position', f"{latest['team']} · {latest['position']}")
    tiles[2].metric('Weeks estimated', f"{len(rows):,}")
    st.plotly_chart(estimate_chart(rows), use_container_width=True, theme=None,
                    config={'displayModeBar': False, 'responsive': True})


if __name__ == '__main__':
    main()