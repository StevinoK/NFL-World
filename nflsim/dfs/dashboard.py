"""Streamlit dashboard for the DK Showdown DFS lineup optimizer.

Pick the simulations/salaries/entries CSVs, confirm any ambiguous starting kickers,
review/scale each player's simulated fantasy point distribution, then run the full
pipeline and download the filled DK entries CSV.
"""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from nflsim import ui
from nflsim.dfs.optimizer import (
    align_simulation_columns,
    apply_kicker_selection,
    build_field_mean,
    build_field_simulations,
    build_player_map,
    compute_lineup_score_distributions,
    draft_our_entries,
    fetch_draftkings_contest_detail,
    find_ambiguous_kickers,
    generate_valid_lineups,
    load_entries,
    load_prices,
    load_simulations,
    merge_prices_and_sims,
    run_field_simulation,
    scale_simulations,
    summarize_player_projections,
)
from nflsim.dfs.public_projections import (
    blend_projections,
    fetch_dailyfantasyfuel_projections,
    fetch_espn_projections,
    fetch_rotowire_projections,
    fetch_rotowire_slates,
    fetch_sleeper_projections,
)

try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
except ImportError:
    get_script_run_ctx = None

DEFAULT_SEASON = 2026


@st.cache_data(show_spinner="Fetching Sleeper projections...", ttl=300)
def cached_sleeper(season: int, week: int) -> pd.Series:
    return fetch_sleeper_projections(season, week)


@st.cache_data(show_spinner="Fetching ESPN projections...", ttl=300)
def cached_espn(season: int, week: int) -> pd.Series:
    return fetch_espn_projections(season, week)


@st.cache_data(show_spinner="Fetching RotoWire slates...", ttl=300)
def cached_rotowire_slates() -> pd.DataFrame:
    return fetch_rotowire_slates()


@st.cache_data(show_spinner="Fetching RotoWire projections...", ttl=300)
def cached_rotowire(slate_id: int) -> pd.Series:
    return fetch_rotowire_projections(slate_id)


@st.cache_data(show_spinner="Fetching DailyFantasyFuel projections...", ttl=300)
def cached_dailyfantasyfuel(slate_date: str) -> pd.Series:
    return fetch_dailyfantasyfuel_projections(slate_date)


@st.cache_data(show_spinner="Fetching DraftKings contest details...", ttl=300)
def cached_draftkings_contest(contest_id: str) -> tuple[dict, pd.DataFrame]:
    return fetch_draftkings_contest_detail(contest_id)


def external_sources(data: dict) -> dict[str, pd.Series]:
    """The public projection Series that actually returned data this run."""
    candidates = {
        "Sleeper": data.get("sleeper"),
        "RotoWire": data.get("rotowire"),
        "DailyFantasyFuel": data.get("dailyfantasyfuel"),
        "ESPN": data.get("espn"),
    }
    return {label: series for label, series in candidates.items() if series is not None and not series.empty}


def player_column(simulations: pd.DataFrame, short_name: str) -> pd.Series:
    values = simulations[short_name]
    if isinstance(values, pd.DataFrame):  # duplicate short_name columns
        values = values.iloc[:, 0]
    return values.astype(float)


_SOURCE_COLORS = {"Sleeper": ui.SERIES[1], "RotoWire": ui.SERIES[2], "DailyFantasyFuel": ui.SERIES[6], "ESPN": ui.SERIES[3]}


def distribution_chart(values: pd.Series, mean_value: float, source_points: dict[str, float]) -> alt.LayerChart:
    counts, edges = np.histogram(values.to_numpy(), bins=25)
    hist = pd.DataFrame({"points": (edges[:-1] + edges[1:]) / 2.0, "count": counts})
    layers = [
        alt.Chart(hist)
        .mark_bar(color="#9ec5f4", cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
        .encode(x=alt.X("points:Q", title=None), y=alt.Y("count:Q", title=None))
    ]
    rules = [("scaled mean", mean_value, ui.INK)]
    rules += [
        (label, float(value), _SOURCE_COLORS.get(label, ui.MUTED))
        for label, value in source_points.items()
        if value is not None and np.isfinite(value)
    ]
    for label, value, color in rules:
        rule_df = pd.DataFrame({"points": [value], "label": [label]})
        layers.append(
            alt.Chart(rule_df)
            .mark_rule(color=color, size=2)
            .encode(x="points:Q", tooltip=["label", "points"])
        )
    return ui.style_altair(alt.layer(*layers).properties(height=130))


def reset_player_targets() -> None:
    st.session_state["player_targets"] = {}
    st.session_state.pop("last_scale_source", None)
    for key in [k for k in st.session_state if str(k).startswith("target_")]:
        del st.session_state[key]


def apply_external_targets(source: pd.Series, projections: pd.DataFrame) -> None:
    """Set each matched player's target mean to a weighted blend of an external DK-format
    projection and our own simulation mean."""
    weight = float(st.session_state.get("source_blend_weight", 1.0))
    targets = st.session_state.setdefault("player_targets", {})
    matched = 0
    for short_name, row in projections.iterrows():
        points = source.get(short_name)
        if points is None or not np.isfinite(points):
            continue
        base = float(row["sim_mean"]) if pd.notna(row["sim_mean"]) else float(points)
        blended_points = weight * float(points) + (1.0 - weight) * base
        value = float(np.clip(blended_points, 0.0, float(row["upper_bound"])))
        targets[short_name] = value
        st.session_state[f"target_{short_name}"] = value
        matched += 1
    st.session_state["last_scale_source"] = (source.name, matched, len(projections), weight)


def main() -> None:
    ui.page("DFS Showdown Optimizer")
    ui.header("Showdown Optimizer", "Scale simulated player distributions, model the field and draft DraftKings Showdown entries.",
              eyebrow="DFS")
    ui.section("Slate", st.sidebar)

    season = st.sidebar.number_input("Season", min_value=2016, max_value=2100, value=DEFAULT_SEASON, step=1, help='Season of the slate.')
    week = st.sidebar.number_input("Week", min_value=1, max_value=22, value=1, step=1, help='Week of the slate.')

    ui.section("Inputs", st.sidebar)
    # file_uploader's "Browse files" button opens the OS's native file explorer dialog.
    sim_file = st.sidebar.file_uploader("Simulations CSV", type="csv", key="sims_upload", help='Player results from a simulation run: the players.csv in its output folder.')
    prices_file = st.sidebar.file_uploader("DK Salaries CSV", type="csv", key="prices_upload", help='The salary file downloaded from DraftKings for this contest.')
    entries_file = st.sidebar.file_uploader("DK Entries CSV", type="csv", key="entries_upload", help='Your entries file from DraftKings. The optimizer fills these entries so you can upload them back.')

    ui.section("Settings", st.sidebar)
    with st.sidebar.expander("Contest settings", expanded=True):
        salary_min = st.number_input("Salary floor", value=48000, step=500, help='Lineups must spend at least this much.')
        salary_max = st.number_input("Salary cap", value=50000, step=500, help='Lineups may spend at most this much.')
        price_floor = st.number_input("Min player salary to include", value=600, step=50, help='Leave out players cheaper than this.')
        optimization_objective = st.selectbox(
            "Lineup objective",
            ["Omega ratio", "Winner odds"],
            help="Omega uses the DK payout table and field size; Winner odds maximizes the chance one entry beats the field.",
        )

    with st.sidebar.expander("Field-model settings"):
        num_agents = st.number_input("Field agents", value=500, min_value=10, step=50, help='Simulated opponents building lineups, used to estimate how the field will play.')
        team_size = st.number_input("Lineups per agent", value=5, min_value=1, step=1, help='Lineups each simulated opponent builds.')
        iterations = st.number_input("Improvement iterations", value=3, min_value=1, step=1, help='Rounds of improving each lineup. More can find better lineups but takes longer.')
        subsample = st.slider("Agent simulation subsample", 0.05, 1.0, 0.5, help='Share of simulations each opponent sees. Lower values make opponents less alike.')
        pool_size = st.number_input(
            "Candidate lineup pool",
            value=20000,
            min_value=100,
            step=1000,
            help="Agents only choose from this many top-upside lineups. Lower is much faster.",
        )

    with st.sidebar.expander("External projections", expanded=True):
        slates = cached_rotowire_slates()
        rotowire_slate = None
        if slates.empty:
            st.caption("RotoWire slate list unavailable right now.")
        else:
            labels = {
                int(row.slateID): f"{row.contestType} · {row.slateName} · {str(row.startDate)[:16]}"
                for row in slates.itertuples()
            }
            showdown_positions = [i for i, row in enumerate(slates.itertuples()) if row.contestType == "Showdown"]
            rotowire_slate = st.selectbox(
                "RotoWire slate",
                list(labels),
                index=showdown_positions[0] if showdown_positions else 0,
                format_func=lambda slate_id: labels[slate_id],
            help='The RotoWire slate that matches this contest.')
        dff_date = st.date_input("DailyFantasyFuel slate date", value=date.today(), help='Date of the slate to pull from DailyFantasyFuel.')

    if not (sim_file and prices_file and entries_file):
        st.info("Choose simulations, salary, and entries CSVs in the sidebar to begin.")
        return

    try:
        prices = load_prices(prices_file)
    except (OSError, pd.errors.ParserError) as error:
        st.error(f"Could not load salaries CSV: {error}")
        return

    try:
        export_df = load_entries(entries_file)
    except (OSError, pd.errors.ParserError, ValueError) as error:
        st.error(f"Could not load entries CSV: {error}")
        return

    contest_ids = sorted(export_df["Contest ID"].dropna().unique())
    contest_detail = None
    contest_payouts = None
    with st.sidebar.expander("DraftKings contest", expanded=True):
        st.write(f"Our entries: {len(export_df)}")
        if len(contest_ids) != 1:
            st.warning(f"Found {len(contest_ids)} contest IDs in the entries file.")
        for contest_id in contest_ids:
            try:
                contest, payouts = cached_draftkings_contest(contest_id)
            except Exception as error:
                st.warning(f"Could not fetch contest {contest_id}: {error}")
                continue
            if len(contest_ids) == 1:
                contest_detail = contest
                contest_payouts = payouts
            label = contest["name"] or f"Contest {contest_id}"
            st.markdown(f"**{label}**")
            st.write(
                {
                    "contest_id": contest["contest_id"],
                    "entry_fee": contest["entry_fee"],
                    "maximum_entries": contest["maximum_entries"],
                    "entries": contest["entries"],
                }
            )
            if not payouts.empty:
                st.dataframe(payouts, width="stretch", hide_index=True)

    ambiguous = find_ambiguous_kickers(prices)
    kicker_choices = {}
    if ambiguous:
        ui.section("Confirm starting kickers")
        st.caption("More than one kicker is priced for these teams; pick the actual starter.")
        for team, group in ambiguous.items():
            options = group.sort_values("Salary", ascending=False)["Name"].unique().tolist()
            kicker_choices[team] = st.selectbox(f"{team} kicker", options, key=f"kicker_{team}", help='Kicker to use for this team, since projection sources can differ.')
        prices = apply_kicker_selection(prices, kicker_choices)

    if st.button("Load projections", help='Download public projections to compare with the simulations.'):
        try:
            simulations_raw = load_simulations(sim_file)
        except (OSError, pd.errors.ParserError) as error:
            st.error(f"Could not load simulations CSV: {error}")
            return
        loaded_sims, loaded_means, _ = align_simulation_columns(simulations_raw)
        loaded_prices = merge_prices_and_sims(prices, loaded_means, price_floor=int(price_floor))
        if loaded_prices.empty:
            st.error("No priced players matched the simulation columns.")
            return
        sleeper = cached_sleeper(int(season), int(week))
        espn = cached_espn(int(season), int(week))
        rotowire = (
            cached_rotowire(int(rotowire_slate))
            if rotowire_slate is not None
            else pd.Series(dtype=float, name="rotowire")
        )
        dailyfantasyfuel = cached_dailyfantasyfuel(dff_date.isoformat())
        st.session_state["dfs_data"] = {
            "simulations": loaded_sims,
            "merged_prices": loaded_prices,
            "sleeper": sleeper,
            "espn": espn,
            "rotowire": rotowire,
            "dailyfantasyfuel": dailyfantasyfuel,
            "blended": blend_projections(sleeper, espn, rotowire, dailyfantasyfuel),
        }
        reset_player_targets()

    data = st.session_state.get("dfs_data")
    if data is None:
        st.info("Load projections to review player distributions before running the optimizer.")
        return

    simulations = data["simulations"]
    merged_prices = data["merged_prices"]
    sleeper, espn, blended = data["sleeper"], data["espn"], data["blended"]
    sources = external_sources(data)

    if sources:
        st.success(f"Field model means sourced from: {', '.join(sources)} (blended, then filled with our own sim means).")
    else:
        st.warning("Public projection sources unavailable this run; field model falls back to our own simulation means.")

    projections = summarize_player_projections(
        merged_prices,
        simulations,
        sleeper,
        espn,
        blended,
        extra_sources={
            "rotowire_pts": data["rotowire"],
            "dailyfantasyfuel_pts": data["dailyfantasyfuel"],
        },
    )
    external_frame = pd.DataFrame({label: series for label, series in sources.items()}).reindex(projections.index)
    projections["upper_bound"] = (
        np.maximum(projections["sim_mean"].fillna(0.0) * 2.0, external_frame.max(axis=1).fillna(0.0) * 1.1)
        .clip(lower=1.0)
        .round(1)
    )

    ui.section("Review and scale player projections")
    st.caption(
        "Each slider sets a player's target mean; the whole simulated distribution is multiplied "
        "by target / sim mean, so its shape is preserved. Reload after changing the salary filters."
    )

    scale_targets = dict(sources)
    if blended is not None and not blended.empty:
        scale_targets["Source average"] = blended.rename("Source average")
    st.slider(
        "Weight toward source when scaling",
        0.0,
        1.0,
        1.0,
        0.05,
        key="source_blend_weight",
        help="1.0 uses the source's projection outright; 0.5 splits it with our simulation mean.",
    )
    button_cols = st.columns(len(scale_targets) + 1)
    for column, (label, series) in zip(button_cols, scale_targets.items()):
        column.button(
            f"Scale to {label}",
            key=f"scale_to_{label}",
            on_click=apply_external_targets,
            args=(series.rename(label), projections),
            use_container_width=True,
        help="Scale this player's simulated points so the average matches this source.")
    button_cols[-1].button("Reset player scales", on_click=reset_player_targets, use_container_width=True, help='Clear all scaling and return to the simulated values.')
    last_scale = st.session_state.get("last_scale_source")
    if last_scale:
        st.caption(
            f"Scaled {last_scale[1]} of {last_scale[2]} players to {last_scale[0]} projections "
            f"at {last_scale[3]:.0%} source weight."
        )

    global_scale = st.slider("Global scale (all players)", 0.50, 1.50, 1.00, 0.01, help="Multiply every player's simulated points by this amount.")

    targets = st.session_state.setdefault("player_targets", {})
    factors: dict[str, float] = {}
    show_histograms = st.checkbox("Show distribution histograms", value=True, help="Draw each player's simulated points. Slower with many players.")
    max_shown = st.number_input("Players shown", min_value=5, max_value=200, value=min(40, len(projections)), step=5, help='How many players to list, highest salary or projection first.')

    for short_name, row in projections.head(int(max_shown)).iterrows():
        base = float(row["sim_mean"]) if pd.notna(row["sim_mean"]) else 0.0
        player_sources = {
            label: float(series[short_name])
            for label, series in sources.items()
            if short_name in series.index and np.isfinite(series[short_name])
        }

        info_col, slider_col, chart_col = st.columns([2, 2, 3])
        info_col.markdown(f"**{row['Name']}** · {row['Position']} · {row['TeamAbbrev']} · ${int(row['Salary']):,}")
        external_bits = " | ".join(f"{label} {value:.1f}" for label, value in player_sources.items())
        info_col.caption(
            f"sim mean {base:.1f} | p10 {row['sim_p10']:.1f} | p90 {row['sim_p90']:.1f}"
            + (f" | {external_bits}" if external_bits else "")
        )

        if base > 0:
            target = slider_col.slider(
                "Target mean",
                min_value=0.0,
                max_value=float(row["upper_bound"]),
                value=float(min(targets.get(short_name, base), float(row["upper_bound"]))),
                step=0.1,
                key=f"target_{short_name}",
                label_visibility="collapsed",
            help="Scale this player's simulations to average this many points.")
            targets[short_name] = target
            factor = (target / base) * global_scale
        else:
            slider_col.caption("no positive sim mean to scale")
            factor = global_scale
        factors[short_name] = factor
        slider_col.caption(f"scaled mean {base * factor:.1f} ({factor:.2f}x)")

        if show_histograms:
            values = player_column(simulations, short_name) * factor
            chart_col.altair_chart(
                distribution_chart(values, base * factor, player_sources), use_container_width=True
            )

    scaled_summary = projections.drop(columns="upper_bound").assign(
        scale=[factors.get(name, global_scale) for name in projections.index],
    )
    scaled_summary["scaled_mean"] = scaled_summary["sim_mean"] * scaled_summary["scale"]
    with st.expander("All players (scaled)", expanded=False):
        st.dataframe(scaled_summary, width="stretch")

    if not st.button("Run optimizer", type="primary", help='Build lineups with these settings.'):
        return

    scale_series = pd.Series(
        {name: factors.get(name, global_scale) for name in simulations.columns}, dtype="float64"
    )
    simulations = scale_simulations(simulations, scale_series)
    means = simulations.mean(axis=0)
    covs = simulations.cov()
    merged_prices = merged_prices.assign(proj_mean=merged_prices["short_name"].map(means))

    field_mean = build_field_mean(blended, means)

    with st.status("Generating valid lineups...", expanded=True) as status:
        lineups_df = generate_valid_lineups(merged_prices, int(salary_min), int(salary_max))
        if lineups_df.empty:
            status.update(label="No valid lineups found", state="error")
            st.error("No valid lineups found under the current salary settings.")
            return
        status.update(label=f"Found {len(lineups_df)} valid lineups", state="complete")

    player_map = build_player_map(merged_prices)
    lineup_sims_df, lu_scores_array = compute_lineup_score_distributions(lineups_df, simulations, player_map)

    field_simulations = build_field_simulations(covs, field_mean, len(simulations))
    _, market_scores_array = compute_lineup_score_distributions(lineups_df, field_simulations, player_map)

    progress = st.progress(0.0, text="Simulating the field...")

    def on_progress(done: int, total: int, drift: float) -> None:
        progress.progress(done / total, text=f"Field iteration {done}/{total} (ownership drift {drift:.4f})")

    _, counts, field_indices = run_field_simulation(
        lu_scores_array,
        market_scores_array,
        lineup_sims_df,
        num_agents=int(num_agents),
        team_size=int(team_size),
        iterations=int(iterations),
        subsample=float(subsample),
        pool_size=int(pool_size),
        progress_callback=on_progress,
    )

    objective_key = "winner"
    if optimization_objective == "Omega ratio":
        if contest_detail is not None and contest_payouts is not None and not contest_payouts.empty:
            objective_key = "omega"
        else:
            st.warning("Omega ratio objective needs one fetched DK contest payout table; falling back to Winner odds.")

    filled_export, drafted_lineups_df, our_counts = draft_our_entries(
        export_df,
        lu_scores_array,
        field_indices,
        lineup_sims_df,
        player_map,
        payout_tiers=contest_payouts,
        entry_fee=contest_detail["entry_fee"] if contest_detail is not None else None,
        field_size=contest_detail["maximum_entries"] if contest_detail is not None else None,
        objective=objective_key,
        omega_pool_size=int(pool_size),
    )

    ui.section("Results")
    tiles = st.columns(3)
    tiles[0].metric("Valid lineups", f"{len(lineups_df):,}")
    tiles[1].metric("Entries drafted", f"{len(drafted_lineups_df):,}")
    tiles[2].metric("Draft objective", "Omega ratio" if objective_key == "omega" else "Winner odds")

    ownership = pd.concat([counts.rename("field_ownership"), our_counts.rename("our_ownership")], axis=1).fillna(0)
    ownership["diff"] = ownership["our_ownership"] - ownership["field_ownership"]
    st.dataframe(ownership.sort_values("our_ownership", ascending=False), width="stretch")

    ui.section("Drafted lineups")
    st.dataframe(drafted_lineups_df, width="stretch")

    ui.section("Final player projections")
    st.caption("Our scaled simulation distribution plus whichever public sources fed the field model.")
    final_projections = summarize_player_projections(
        merged_prices,
        simulations,
        sleeper,
        espn,
        blended,
        extra_sources={
            "rotowire_pts": data["rotowire"],
            "dailyfantasyfuel_pts": data["dailyfantasyfuel"],
        },
    )
    st.dataframe(final_projections, width="stretch")
    st.bar_chart(final_projections[["sim_p10", "sim_mean", "sim_p90"]].head(30), stack=False,
                 color=[ui.SERIES[0], ui.SERIES[1], ui.SERIES[2]])

    csv_bytes = filled_export.to_csv(index=False).encode("utf-8")
    st.download_button("Download entries CSV", data=csv_bytes, file_name="Entry.csv", mime="text/csv", help='Save the finished lineups in the format DraftKings accepts.')

    save_path = st.text_input("Optional: also save the entries CSV to a path", value="", help='Also write the file to this location on your PC.')
    if save_path:
        filled_export.to_csv(save_path, index=False)
        st.success(f"Saved to {save_path}")


if __name__ == "__main__":
    if get_script_run_ctx is None or get_script_run_ctx() is None:
        os.execv(sys.executable, [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve())])
    main()
