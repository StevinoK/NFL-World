"""Game-level context features used by other forecasters, tested against a pregame team-strength rating.

Strength: recency-weighted ridge (Massey) on point margin with a home term, refit before each week on earlier
games only. Each feature is tested by leave-one-season-out RMSE of home margin, 2016-2025 regular season,
and again with the closing spread as the control (does the market already price it?).
"""
import os, numpy as np, pandas as pd
D = os.environ.get("PBP_DIR", "data/pbp")
g = pd.read_csv(f"{D}/games.csv")
g = g[(g.game_type == "REG") & (g.season >= 2010) & g.result.notna()].copy()
LOC = {  # lat, lon, time zone (hours from ET during the season; Arizona counts as Pacific)
 "ARI":(33.53,-112.26,-3),"ATL":(33.76,-84.40,0),"BAL":(39.28,-76.62,0),"BUF":(42.77,-78.79,0),"CAR":(35.23,-80.85,0),
 "CHI":(41.86,-87.62,-1),"CIN":(39.10,-84.52,0),"CLE":(41.51,-81.70,0),"DAL":(32.75,-97.09,-1),"DEN":(39.74,-105.02,-2),
 "DET":(42.34,-83.05,0),"GB":(44.50,-88.06,-1),"HOU":(29.68,-95.41,-1),"IND":(39.76,-86.16,0),"JAX":(30.32,-81.64,0),
 "KC":(39.05,-94.48,-1),"LA":(33.95,-118.34,-3),"STL":(38.63,-90.19,-1),"LAC":(33.95,-118.34,-3),"SD":(32.78,-117.12,-3),
 "LV":(36.09,-115.18,-3),"OAK":(37.75,-122.20,-3),"MIA":(25.96,-80.24,0),"MIN":(44.97,-93.26,-1),"NE":(42.09,-71.26,0),
 "NO":(29.95,-90.08,-1),"NYG":(40.81,-74.07,0),"NYJ":(40.81,-74.07,0),"PHI":(39.90,-75.17,0),"PIT":(40.45,-80.02,0),
 "SEA":(47.60,-122.33,-3),"SF":(37.40,-121.97,-3),"TB":(27.98,-82.50,0),"TEN":(36.17,-86.77,-1),"WAS":(38.91,-76.86,0)}
def miles(a, b):
    la1, lo1, _ = LOC[a]; la2, lo2, _ = LOC[b]
    p1, p2, dl = np.radians(la1), np.radians(la2), np.radians(lo2 - lo1)
    return 3959 * np.arccos(np.clip(np.sin(p1)*np.sin(p2) + np.cos(p1)*np.cos(p2)*np.cos(dl), -1, 1))
g = g[g.location == "Home"].copy()  # neutral/international games left out
g["margin"] = g.result
g["travel_k"] = [miles(a, h) / 1000 for a, h in zip(g.away_team, g.home_team)]
g["tz_shift"] = [LOC[h][2] - LOC[a][2] for a, h in zip(g.away_team, g.home_team)]  # + = away team travels east
et_hour = pd.to_numeric(g.gametime.str[:2], errors="coerce") + pd.to_numeric(g.gametime.str[3:5], errors="coerce") / 60
g["early_east_trip"] = ((et_hour < 14) & (g.tz_shift >= 2)).astype(float)      # west team, 1pm ET kickoff
g["night_west_host"] = ((et_hour >= 20) & (g.tz_shift <= -2)).astype(float)    # east team at a west host at night
g["night_tz"] = (et_hour >= 20) * g.tz_shift
g["div"] = g.div_game.astype(float)
g["rest_diff"] = (g.home_rest - g.away_rest).clip(-7, 7)
g["away_short"] = (g.away_rest <= 4).astype(float); g["home_short"] = (g.home_rest <= 4).astype(float)
g["away_off_bye"] = (g.away_rest >= 13).astype(float); g["home_off_bye"] = (g.home_rest >= 13).astype(float)
turf = lambda s: (~s.fillna("").str.contains("grass")).astype(float)
g["game_turf"] = turf(g.surface)
g = g.sort_values(["gameday", "game_id"]).reset_index(drop=True)
home_surface = {}
mis = []
for r in g.itertuples():
    mis.append(float(home_surface.get(r.away_team, r.game_turf) != r.game_turf))
    home_surface[r.home_team] = r.game_turf
g["surface_mismatch"] = mis
g["temp_f"] = np.where(g.roof.isin(["outdoors", "open"]), g.temp.fillna(60), 70.0)
g["cold_vs_warm_team"] = ((g.temp_f < 40) & np.array([LOC[a][0] < 36 for a in g.away_team])).astype(float)
# --- pregame strength: recency-weighted ridge on margin, refit each week on earlier games
teams = sorted(LOC); idx = {t: i for i, t in enumerate(teams)}
g["wk"] = g.season * 100 + g.week
strength = np.full(len(g), np.nan)
for wk in sorted(g.wk.unique()):
    if wk < 201201: continue
    past = g[g.wk < wk]; now = g.index[g.wk == wk]
    s_now = wk // 100
    age_w = np.where(past.season == s_now, 0.95 ** (wk % 100 - past.week), 0.95 ** (wk % 100) * 0.55 ** (s_now - past.season - 1) * 0.5 ** 0)
    age_w = np.where(past.season == s_now, age_w, 0.95 ** (wk % 100 - 1) * 0.6 * 0.6 ** (s_now - past.season - 1) * 0.97 ** (18 - past.week))
    X = np.zeros((len(past), len(teams) + 1)); X[np.arange(len(past)), [idx[t] for t in past.home_team]] = 1
    X[np.arange(len(past)), [idx[t] for t in past.away_team]] = -1; X[:, -1] = 1
    W = age_w[:, None] * X; P = np.eye(len(teams) + 1) * 8; P[-1, -1] = 0.1
    beta = np.linalg.solve(X.T @ W + P, W.T @ past.margin.to_numpy())
    strength[now] = [beta[idx[h]] - beta[idx[a]] for h, a in zip(g.home_team[now], g.away_team[now])]
g["strength"] = strength
g = g[(g.season >= 2016) & g.strength.notna() & g.spread_line.notna()].copy()
def loso(cols):
    err = []
    for s in sorted(g.season.unique()):
        tr, te = g[g.season != s], g[g.season == s]
        A = np.c_[np.ones(len(tr)), tr[cols]]; b = np.linalg.lstsq(A, tr.margin, rcond=None)[0]
        err.append((te.margin - np.c_[np.ones(len(te)), te[cols]] @ b) ** 2)
    return np.concatenate(err)
def coef(cols):
    A = np.c_[np.ones(len(g)), g[cols]]; b, *_ = np.linalg.lstsq(A, g.margin, rcond=None)
    res = g.margin - A @ b; cov = np.linalg.inv(A.T @ A) * res.var(); return b[-1], np.sqrt(cov[-1, -1])
features = ["rest_diff", "away_short", "home_short", "away_off_bye", "home_off_bye", "travel_k", "tz_shift",
            "early_east_trip", "night_west_host", "night_tz", "div", "surface_mismatch", "cold_vs_warm_team"]
rows = []
for control in ["strength", "spread_line"]:
    base = loso([control]); rows.append((control, "(base)", np.sqrt(base.mean()), 0, 0, np.nan, np.nan))
    for f in features:
        e = loso([control, f]); d = e - base
        b, se = coef([control, f])
        rows.append((control, f, np.sqrt(e.mean()), d.mean(), d.mean() / (d.std() / np.sqrt(len(d))), b, b / se))
r = pd.DataFrame(rows, columns=["control", "feature", "rmse", "mse_change", "mse_change_z", "points_per_unit", "coef_t"])
print("games:", len(g), "| share of games: div", g["div"].mean().round(2), "early_east_trip", g.early_east_trip.mean().round(3),
      "night_west_host", g.night_west_host.mean().round(3), "surface_mismatch", g.surface_mismatch.mean().round(2))
print(r.round(3).to_string(index=False)); r.to_csv("game_context_experiment.csv", index=False)
