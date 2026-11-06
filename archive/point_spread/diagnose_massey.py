import sys
sys.path.insert(0, r"C:\Users\steve\OneDrive\Desktop\Football\Modeling")

import numpy as np
import pandas as pd
from bayesian_feature_builder import load_pbp_data, _rolling_team_massey_rating, _safe_divide


def _rolling_team_massey_rating_stable(
    pbp: pd.DataFrame,
    team_col: str,
    event_mask: pd.Series,
    value_col: str,
    count_col: str,
    source: str,
    max_age_days: int = 365,
    ridge_lambda: float = 2.0,
) -> pd.DataFrame:
    """Regularized Massey-style rating using a date window and active-team ridge prior."""
    subset = pbp.loc[event_mask].copy()
    if subset.empty:
        return pd.DataFrame(columns=["team", "game_id", "game_date", "rate", "count"])

    required = ["game_id", "game_date", "home_team", "away_team", team_col, value_col, count_col]
    missing = [column for column in required if column not in subset.columns]
    if missing:
        return pd.DataFrame(columns=["team", "game_id", "game_date", "rate", "count"])

    team_prefix = subset[["game_id", "game_date", "home_team", "away_team", team_col, value_col, count_col]].copy()
    team_prefix[count_col] = pd.to_numeric(team_prefix[count_col], errors="coerce").fillna(0.0)
    team_prefix[value_col] = pd.to_numeric(team_prefix[value_col], errors="coerce").fillna(0.0)
    team_game = (
        team_prefix.groupby([team_col, "game_id", "game_date"], dropna=False)
        .agg(team_value=(value_col, "sum"), plays=(count_col, "sum"))
        .reset_index()
        .rename(columns={team_col: "team"})
    )
    team_game["team_eppa"] = _safe_divide(team_game["team_value"], team_game["plays"])

    game_rows = (
        team_prefix[["game_id", "game_date", "home_team", "away_team"]]
        .drop_duplicates()
        .sort_values(["game_date", "game_id"])
        .reset_index(drop=True)
    )
    if game_rows.empty:
        return pd.DataFrame(columns=["team", "game_id", "game_date", "rate", "count"])

    records: list[dict[str, object]] = []

    for _, game in game_rows.iterrows():
        current_game_id = str(game["game_id"])
        current_game_date = pd.to_datetime(game["game_date"])
        home_team = game["home_team"]
        away_team = game["away_team"]
        if pd.isna(home_team) or pd.isna(away_team):
            continue

        cutoff = current_game_date - pd.Timedelta(days=max_age_days)
        window = game_rows[
            (game_rows["game_date"] < current_game_date) & (game_rows["game_date"] >= cutoff)
        ].copy()
        if window.empty:
            continue

        # Active teams only
        active_teams = sorted(
            {
                team
                for team in pd.concat([window["home_team"], window["away_team"]], ignore_index=True)
                .dropna()
                .unique()
                if str(team) != "nan"
            }
        )
        if len(active_teams) < 2:
            continue
        team_to_index = {team: idx for idx, team in enumerate(active_teams)}
        team_count = len(active_teams)

        rows = []
        rhs = []
        weights = []
        for idx, prior_game in window.iterrows():
            prior_id = str(prior_game["game_id"])
            prior_home = prior_game["home_team"]
            prior_away = prior_game["away_team"]
            if pd.isna(prior_home) or pd.isna(prior_away):
                continue
            if prior_home not in team_to_index or prior_away not in team_to_index:
                continue

            home_obs = team_game[(team_game["game_id"] == prior_id) & (team_game["team"] == prior_home)]
            away_obs = team_game[(team_game["game_id"] == prior_id) & (team_game["team"] == prior_away)]
            if home_obs.empty or away_obs.empty:
                continue

            home_rate = float(home_obs["team_eppa"].iat[0])
            away_rate = float(away_obs["team_eppa"].iat[0])
            if np.isnan(home_rate) or np.isnan(away_rate):
                continue

            # Time decay: most recent game in window gets weight 1.0
            age_days = (current_game_date - pd.to_datetime(prior_game["game_date"])).days
            weight = 0.97 ** age_days

            row_home = np.zeros(2 * team_count, dtype=float)
            row_home[team_to_index[prior_home]] = 1.0
            row_home[team_count + team_to_index[prior_away]] = -1.0
            rows.append(row_home)
            rhs.append(home_rate)
            weights.append(weight)

            row_away = np.zeros(2 * team_count, dtype=float)
            row_away[team_to_index[prior_away]] = 1.0
            row_away[team_count + team_to_index[prior_home]] = -1.0
            rows.append(row_away)
            rhs.append(away_rate)
            weights.append(weight)

        if not rows:
            continue

        design = np.vstack(rows)
        observed = np.asarray(rhs, dtype=float)
        game_weights = np.asarray(weights, dtype=float)
        sqrt_w = np.sqrt(np.clip(game_weights, 1e-12, None))
        weighted_design = design * sqrt_w[:, None]
        weighted_rhs = observed * sqrt_w

        # Ridge penalty toward zero for offense and defense coefficients
        penalty = np.eye(2 * team_count) * np.sqrt(ridge_lambda)
        weighted_design = np.vstack([weighted_design, penalty])
        weighted_rhs = np.concatenate([weighted_rhs, np.zeros(2 * team_count)])

        solution, *_ = np.linalg.lstsq(weighted_design, weighted_rhs, rcond=None)

        offense = solution[:team_count]
        defense = solution[team_count:]
        rating = {team: float(offense[team_to_index[team]] - defense[team_to_index[team]]) for team in active_teams}
        offense_map = {team: float(offense[team_to_index[team]]) for team in active_teams}
        defense_map = {team: float(defense[team_to_index[team]]) for team in active_teams}

        for team in [home_team, away_team]:
            if team in rating:
                records.append({
                    "team": team,
                    "game_id": current_game_id,
                    "game_date": current_game_date,
                    "offense_rating": float(offense_map[team]),
                    "defense_rating": float(defense_map[team]),
                    "rate": float(rating[team]),
                })

    if not records:
        return pd.DataFrame(columns=["team", "game_id", "game_date", "offense_rating", "defense_rating", "rate", "count"])

    result = pd.DataFrame.from_records(records)
    result["count"] = 1.0
    return result[["team", "game_id", "game_date", "offense_rating", "defense_rating", "rate", "count"]]


pbp = load_pbp_data(years=list(range(2016, 2026)))
print(f"pbp shape: {pbp.shape}")
print(f"season_types: {pbp['season_type'].unique()}")

# Passing massey
passing_massey = _rolling_team_massey_rating(
    pbp,
    team_col="posteam",
    event_mask=pbp["pass"] == 1,
    value_col="epa",
    count_col="pass",
    source="passing_massey_rating",
)

# Rushing massey
rushing_massey = _rolling_team_massey_rating(
    pbp,
    team_col="posteam",
    event_mask=pbp["rush"] == 1,
    value_col="epa",
    count_col="rush",
    source="rushing_massey_rating",
)

print(f"\nPassing massey records: {len(passing_massey)}")
print(f"Rushing massey records: {len(rushing_massey)}")

passing_stable = _rolling_team_massey_rating_stable(
    pbp,
    team_col="posteam",
    event_mask=pbp["pass"] == 1,
    value_col="epa",
    count_col="pass",
    source="passing_massey_rating_stable",
    max_age_days=365,
    ridge_lambda=2.0,
)

rushing_stable = _rolling_team_massey_rating_stable(
    pbp,
    team_col="posteam",
    event_mask=pbp["rush"] == 1,
    value_col="epa",
    count_col="rush",
    source="rushing_massey_rating_stable",
    max_age_days=365,
    ridge_lambda=2.0,
)

print(f"Stable passing massey records: {len(passing_stable)}")
print(f"Stable rushing massey records: {len(rushing_stable)}")

# Extract NYJ 2025 records
passing_massey["season"] = passing_massey["game_id"].astype(str).str[:4].astype(int)
rushing_massey["season"] = rushing_massey["game_id"].astype(str).str[:4].astype(int)

nyj_pass = passing_massey[(passing_massey["team"] == "NYJ") & (passing_massey["season"] == 2025)].sort_values("game_date")
nyj_rush = rushing_massey[(rushing_massey["team"] == "NYJ") & (rushing_massey["season"] == 2025)].sort_values("game_date")

print("\n=== NYJ 2025 PASSING MASSEY ===")
print(nyj_pass[["game_id", "game_date", "offense_rating", "defense_rating", "rate"]].to_string(index=False))

print("\n=== NYJ 2025 RUSHING MASSEY ===")
print(nyj_rush[["game_id", "game_date", "offense_rating", "defense_rating", "rate"]].to_string(index=False))

# Compare stable version for NYJ
passing_stable["season"] = passing_stable["game_id"].astype(str).str[:4].astype(int)
rushing_stable["season"] = rushing_stable["game_id"].astype(str).str[:4].astype(int)
nyj_pass_stable = passing_stable[(passing_stable["team"] == "NYJ") & (passing_stable["season"] == 2025)].sort_values("game_date")
nyj_rush_stable = rushing_stable[(rushing_stable["team"] == "NYJ") & (rushing_stable["season"] == 2025)].sort_values("game_date")

print("\n=== NYJ 2025 PASSING MASSEY (STABLE) ===")
print(nyj_pass_stable[["game_id", "game_date", "offense_rating", "defense_rating", "rate"]].to_string(index=False))

print("\n=== NYJ 2025 RUSHING MASSEY (STABLE) ===")
print(nyj_rush_stable[["game_id", "game_date", "offense_rating", "defense_rating", "rate"]].to_string(index=False))

# Side-by-side comparison for key weeks
comparison = nyj_pass[["game_id", "game_date", "rate"]].rename(columns={"rate": "original_rate"}).merge(
    nyj_pass_stable[["game_id", "rate"]].rename(columns={"rate": "stable_rate"}),
    on="game_id",
    how="outer",
)
print("\n=== NYJ PASS MASSEY COMPARISON ===")
print(comparison.sort_values("game_date").to_string(index=False))

# Also compute raw EPA/play per game for NYJ
pbp["season"] = pbp["game_id"].astype(str).str[:4].astype(int)
nyj_pbp = pbp[(pbp["posteam"] == "NYJ") & (pbp["season"] == 2025) & (pbp["season_type"] == "REG")].copy()
nyj_pass_games = (
    nyj_pbp[nyj_pbp["pass"] == 1]
    .groupby(["game_id", "game_date"])
    .agg(pass_plays=("pass", "sum"), pass_epa=("epa", "sum"))
    .reset_index()
)
nyj_pass_games["pass_eppa"] = nyj_pass_games["pass_epa"] / nyj_pass_games["pass_plays"]
print("\n=== NYJ 2025 RAW PASS EPA/PLAY BY GAME ===")
print(nyj_pass_games.sort_values("game_date").to_string(index=False))

nyj_rush_games = (
    nyj_pbp[nyj_pbp["rush"] == 1]
    .groupby(["game_id", "game_date"])
    .agg(rush_plays=("rush", "sum"), rush_epa=("epa", "sum"))
    .reset_index()
)
nyj_rush_games["rush_eppa"] = nyj_rush_games["rush_epa"] / nyj_rush_games["rush_plays"]
print("\n=== NYJ 2025 RAW RUSH EPA/PLAY BY GAME ===")
print(nyj_rush_games.sort_values("game_date").to_string(index=False))

# Show the window at week 6 DEN_NYJ
print("\n=== WINDOW COMPOSITION AT 2025_06_DEN_NYJ ===")
target_row = passing_massey[passing_massey["game_id"] == "2025_06_DEN_NYJ"]
print(target_row.head())

# Demonstrate design matrix issue: count unique teams in full dataset vs window
all_games_2025 = pbp[pbp["season"] == 2025]["game_id"].unique()
print(f"\n2025 total games: {len(all_games_2025)}")
print(f"NYJ 2025 games in pbp: {nyj_pbp['game_id'].nunique()}")

# Compute league avg pass epa/play up to each Jets game
print("\n=== LEAGUE AVG PASS EPA/PLAY UP TO EACH NYJ GAME ===")
for _, row in nyj_pass_games.sort_values("game_date").iterrows():
    gdate = row["game_date"]
    prior = pbp[(pbp["pass"] == 1) & (pbp["game_date"] < gdate) & (pbp["season"] == 2025)]
    league_eppa = prior["epa"].mean()
    print(f"{row['game_id']} ({gdate.date()}): NYJ pass eppa={row['pass_eppa']:.3f}, league avg prior={league_eppa:.3f}, plays={row['pass_plays']}")
