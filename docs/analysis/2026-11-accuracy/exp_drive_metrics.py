"""Do drive-level efficiency ratings add pregame information beyond EPA per play?

Both teams' ratings enter (this team's offense vs the opponent defense, and the reverse), as in the simulator.
Team-relative versions subtract each week's league mean, so league-wide rule changes can't pose as team skill.

Per team-game metrics (offense; the same rows give the opponent's defense):
  epa_play      EPA per scrimmage play (no kneels/spikes)            -- the kind of input our team features use
  success       success rate
  ppd           points per drive (TD 7, FG 3; Silver Bulletin-style)
  ppp, plays    points per play and plays per game (ELWAY: points/play x plays)
  edp_avg       expected drive points, an approximation of Fortgang's Earned Drive Points: per drive, the mean
                pre-snap EP over the drive's plays minus the EP of its first play, plus that starting EP, so it
                credits moving the ball whether or not the drive scores. Averaged over drives.
  edp_max       same with the drive's best pre-snap EP instead of the mean
Each metric becomes an opponent-adjusted, recency-weighted ridge rating (offense + defense + home), refit each
week on earlier games. Team points (and home margin, total) are then predicted by leave-one-season-out linear
regression on the ratings' predicted values for that matchup. Test seasons 2017-2025 (2016 warms up).
"""
import os, numpy as np, pandas as pd
D = os.environ.get("PBP_DIR", "data/pbp")  # folder of nflverse pbp_<season>.parquet files
cols = ["game_id", "season", "week", "season_type", "posteam", "defteam", "home_team", "away_team", "play_type", "epa", "ep",
        "success", "qb_kneel", "qb_spike", "fixed_drive", "fixed_drive_result", "drive_start_transition", "drive_inside20", "yardline_100", "game_date", "home_score", "away_score", "location"]
p = pd.concat([pd.read_parquet(f"{D}/pbp_{y}.parquet", columns=[c for c in cols if c != "location"]) for y in range(2016, 2027)])
p = p[(p.season_type == "REG") & p.posteam.notna()]
sc = p[p.play_type.isin(["pass", "run"]) & (p.qb_kneel == 0) & (p.qb_spike == 0) & p.epa.notna()]
tg = sc.groupby(["game_id", "posteam"]).agg(defteam=("defteam", "first"), epa_play=("epa", "mean"), success=("success", "mean"),
                                            plays=("epa", "size")).reset_index()
dr = p[p.fixed_drive.notna()].groupby(["game_id", "posteam", "fixed_drive"])
d = dr.agg(result=("fixed_drive_result", "first"), start=("drive_start_transition", "first"), inside20=("drive_inside20", "max"), min_yl=("yardline_100", "min"), ep0=("ep", "first"), ep_mean=("ep", "mean"), ep_max=("ep", "max")).reset_index()
d["pts"] = d.result.map({"Touchdown": 7.0, "Field goal": 3.0}).fillna(0.0)
d["edp_avg"], d["edp_max"] = d.ep_mean, d.ep_max   # starting EP + (mean or best EP gained) = mean/max EP itself
tr = d.start.fillna("")
d["takeaway_start"] = tr.str.contains("INTERCEPTION|FUMBLE|DOWNS|MUFFED|BLOCKED").astype(float)
d["ep0_kick"] = np.where(tr.eq("KICKOFF"), d.ep0, np.nan); d["ep0_punt"] = np.where(tr.eq("PUNT"), d.ep0, np.nan)
d["rz"] = (d.min_yl <= 20).astype(float); d["rz_td"] = np.where(d.rz == 1, (d.result == "Touchdown").astype(float), np.nan)
d["rz_pts"] = np.where(d.rz == 1, d.pts, np.nan)
d["edp_gain"], d["ppd_over_start"] = d.ep_mean - d.ep0, d.pts - d.ep0   # progress from where the drive started
dg = d.groupby(["game_id", "posteam"]).agg(ppd=("pts", "mean"), drives=("pts", "size"), edp_avg=("edp_avg", "mean"),
                                           edp_max=("edp_max", "mean"), edp_gain=("edp_gain", "mean"), ppd_over_start=("ppd_over_start", "mean"), start_ep=("ep0", "mean"), takeaway_start=("takeaway_start", "mean"), ep0_kick=("ep0_kick", "mean"), ep0_punt=("ep0_punt", "mean"), rz_rate=("rz", "mean"), rz_td=("rz_td", "mean"), rz_pts=("rz_pts", "mean"), points=("pts", "sum")).reset_index()
games = p.groupby("game_id").agg(season=("season", "first"), week=("week", "first"), home=("home_team", "first"),
                                 away=("away_team", "first"), hs=("home_score", "last"), as_=("away_score", "last"),
                                 date=("game_date", "first")).reset_index()
t = tg.merge(dg, on=["game_id", "posteam"]).merge(games, on="game_id")
t["is_home"] = (t.posteam == t.home).astype(float)
t["team_points"] = np.where(t.is_home == 1, t.hs, t.as_)
t["ppp"] = t.points / t.plays
for c in ["ep0_kick", "ep0_punt", "rz_td", "rz_pts"]: t[c] = t[c].fillna(t[c].mean())
t["wk"] = t.season * 100 + t.week
metrics = ["epa_play", "success", "ppd", "ppp", "plays", "edp_avg", "edp_max", "edp_gain", "ppd_over_start", "start_ep", "takeaway_start", "ep0_kick", "ep0_punt", "rz_rate", "rz_td", "rz_pts", "team_points"]
teams = sorted(set(t.posteam)); ix = {k: i for i, k in enumerate(teams)}; n = len(teams)
pred = {m: np.full(len(t), np.nan) for m in metrics}
LAM = {"takeaway_start": 15, "ep0_kick": 15, "ep0_punt": 15, "rz_rate": 15, "rz_td": 15, "rz_pts": 15, "edp_gain": 15, "ppd_over_start": 15, "start_ep": 15, "epa_play": 30, "success": 30, "ppd": 15, "ppp": 15, "plays": 15, "edp_avg": 15, "edp_max": 15, "team_points": 15}
for wk in sorted(t.wk.unique()):
    if wk < 201701: continue
    past, now = t[t.wk < wk], t.index[t.wk == wk]
    s, w = wk // 100, wk % 100
    age = np.where(past.season == s, 0.93 ** (w - past.week), 0.93 ** (w - 1) * 0.5 ** (s - past.season) * 0.97 ** (18 - past.week))
    X = np.zeros((len(past), 2 * n + 2)); r = np.arange(len(past))
    X[r, [ix[k] for k in past.posteam]] = 1; X[r, [n + ix[k] for k in past.defteam]] = 1; X[:, -2] = past.is_home; X[:, -1] = 1
    Xn = np.zeros((len(now), 2 * n + 2)); rn = np.arange(len(now)); cur = t.loc[now]
    Xn[rn, [ix[k] for k in cur.posteam]] = 1; Xn[rn, [n + ix[k] for k in cur.defteam]] = 1; Xn[:, -2] = cur.is_home; Xn[:, -1] = 1
    for m in metrics:
        y = past[m].to_numpy(); wt = age * (past.drives if m in ("ppd", "edp_avg", "edp_max", "edp_gain", "ppd_over_start", "start_ep", "takeaway_start", "ep0_kick", "ep0_punt", "rz_rate", "rz_td", "rz_pts") else past.plays if m in ("epa_play", "success", "ppp") else 1.0)
        wt = np.asarray(wt, dtype=float); wt = wt / wt.mean()
        P = np.full(2 * n + 2, float(LAM[m])); P[-2:] = 1e-3
        b = np.linalg.solve(X.T @ (wt[:, None] * X) + np.diag(P), X.T @ (wt * y))
        pred[m][now] = Xn @ b
for m in metrics: t[f"r_{m}"] = pred[m]
t = t[t.r_epa_play.notna()].copy()
# the other side of the same game: the opponent offense against this team's defense (both teams' strengths, as in the sim)
other = t[["game_id", "posteam"] + [f"r_{m}" for m in metrics]].rename(columns={"posteam": "defteam", **{f"r_{m}": f"o_{m}" for m in metrics}})
t = t.merge(other, on=["game_id", "defteam"], how="inner")
# Team-relative versions: remove the league average of each week's ratings, so a rule change that moves every team
# (e.g. the 2024-25 kickoff rules) can't pose as team skill.
for m in metrics:
    for side in ("r", "o"):
        t[f"{side}d_{m}"] = t[f"{side}_{m}"] - t.groupby("wk")[f"{side}_{m}"].transform("mean")
def loso(features):
    out = pd.Series(np.nan, index=t.index)
    for s in sorted(t.season.unique()):
        tr, te = t[t.season != s], t[t.season == s]
        A = np.c_[np.ones(len(tr)), tr[features]]; b = np.linalg.lstsq(A, tr.team_points, rcond=None)[0]
        out[te.index] = np.c_[np.ones(len(te)), te[features]] @ b
    return out
def scores(pr):
    x = t.assign(pr=pr)
    h = x[x.is_home == 1].set_index("game_id"); a = x[x.is_home == 0].set_index("game_id")
    j = h[["pr", "team_points"]].join(a[["pr", "team_points"]], rsuffix="_a", how="inner")
    m_err = (j.team_points - j.team_points_a) - (j.pr - j.pr_a); t_err = (j.team_points + j.team_points_a) - (j.pr + j.pr_a)
    return np.sqrt(((x.team_points - x.pr) ** 2).mean()), np.sqrt((m_err ** 2).mean()), np.sqrt((t_err ** 2).mean()), (x.team_points - x.pr) ** 2
sets = {
    "EPA/play + plays, both sides (baseline)": ["r_epa_play", "r_plays", "o_epa_play", "o_plays"],
    "+ success rate": ["r_epa_play", "r_plays", "o_epa_play", "o_plays", "r_success", "o_success"],
    "+ points per play (ELWAY-style)": ["r_epa_play", "r_plays", "o_epa_play", "o_plays", "r_ppp", "o_ppp"],
    "+ points per drive (Silver Bulletin)": ["r_epa_play", "r_plays", "o_epa_play", "o_plays", "r_ppd", "o_ppd"],
    "+ expected drive points, mean EP incl. start (EDP approximation)": ["r_epa_play", "r_plays", "o_epa_play", "o_plays", "r_edp_avg", "o_edp_avg"],
    "+ EP gained within drives (mean EP - start EP)": ["r_epa_play", "r_plays", "o_epa_play", "o_plays", "r_edp_gain", "o_edp_gain"],
    "+ points per drive over starting EP": ["r_epa_play", "r_plays", "o_epa_play", "o_plays", "r_ppd_over_start", "o_ppd_over_start"],
    "+ drive start EP (field position)": ["r_epa_play", "r_plays", "o_epa_play", "o_plays", "r_start_ep", "o_start_ep"],
    "+ start EP after kickoffs": ["r_epa_play", "r_plays", "o_epa_play", "o_plays", "r_ep0_kick", "o_ep0_kick"],
    "+ start EP after punts": ["r_epa_play", "r_plays", "o_epa_play", "o_plays", "r_ep0_punt", "o_ep0_punt"],
    "+ share of drives starting from takeaways/downs": ["r_epa_play", "r_plays", "o_epa_play", "o_plays", "r_takeaway_start", "o_takeaway_start"],
    "+ red-zone trips per drive": ["r_epa_play", "r_plays", "o_epa_play", "o_plays", "r_rz_rate", "o_rz_rate"],
    "+ red-zone TD rate": ["r_epa_play", "r_plays", "o_epa_play", "o_plays", "r_rz_td", "o_rz_td"],
    "+ points scored/allowed": ["r_epa_play", "r_plays", "o_epa_play", "o_plays", "r_team_points", "o_team_points"],
    "+ points per play, team-relative": ["r_epa_play", "r_plays", "o_epa_play", "o_plays", "rd_ppp", "od_ppp"],
    "+ points per drive, team-relative": ["r_epa_play", "r_plays", "o_epa_play", "o_plays", "rd_ppd", "od_ppd"],
    "+ drive start EP, team-relative": ["r_epa_play", "r_plays", "o_epa_play", "o_plays", "rd_start_ep", "od_start_ep"],
    "+ red-zone trips per drive, team-relative": ["r_epa_play", "r_plays", "o_epa_play", "o_plays", "rd_rz_rate", "od_rz_rate"],
    "+ points per drive + start EP + red-zone trips, team-relative": ["r_epa_play", "r_plays", "o_epa_play", "o_plays", "rd_ppd", "od_ppd", "rd_start_ep", "od_start_ep", "rd_rz_rate", "od_rz_rate"],
}
base_sq = None; rows = []
for name, f in sets.items():
    tp, mg, tot, sq = scores(loso(f))
    if base_sq is None: base_sq = sq
    dd = sq - base_sq
    rows.append((name, tp, mg, tot, dd.mean(), dd.mean() / (dd.std() / np.sqrt(len(dd))) if dd.std() > 0 else 0.0))
r = pd.DataFrame(rows, columns=["inputs", "team_pts_rmse", "margin_rmse", "total_rmse", "team_mse_change", "z"])
print(f"{len(t)} team-games, test seasons {t.season.min()}-{t.season.max()}")
print(r.round(3).to_string(index=False))
# stability: week-to-week rating of each raw metric -> next-game same metric (signal share)
print("\nHow well each pregame rating predicts its own next-game value (correlation), and next-game points:")
for m in ["epa_play", "ppd", "edp_gain", "start_ep", "ep0_kick", "ep0_punt", "takeaway_start", "rz_rate", "rz_td", "rz_pts", "team_points"]:
    print(f"  {m:12s} own {np.corrcoef(t[f'r_{m}'], t[m])[0,1]:.3f}   points {np.corrcoef(t[f'r_{m}'], t.team_points)[0,1]:.3f}")
r.to_csv("drive_metrics_experiment.csv", index=False)
