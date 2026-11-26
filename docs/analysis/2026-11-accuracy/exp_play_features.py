"""Drive-efficiency team ratings as inputs to the simulator's per-play models.

For each sim model, a LightGBM model on the play's situation plus baseline team ratings is compared with the same
model plus offense/defense ratings of drive metrics. Ratings are opponent-adjusted, recency-weighted ridge
coefficients refit before each week on earlier games only, attached as the offense's offensive rating and the
defense's defensive rating (as the sim feeds team columns). Train 2018-24, test 2025-26 (kickoffs: train through 2025, test 2026, since the 2025 rule has no earlier examples).

Baseline team ratings (stand-ins for the sim's EPA-style team columns): pass EPA/dropback, rush EPA/carry,
success rate, sack rate, plays per game, pass rate.
Results are the % change in held-out log loss (negative = better) with game-clustered z.
Added sets (team-relative: league mean removed each week): points per drive + points per play; drive start EP (all, after punts, after kickoffs); red-zone trips
per drive; rebuilt EDP (needs edp_team_games.parquet from exp_edp.py).
"""
import os, numpy as np, pandas as pd, lightgbm as lgb
D = os.environ.get("PBP_DIR", "data/pbp")
cols = ["game_id", "play_id", "season", "week", "season_type", "game_date", "posteam", "defteam", "home_team", "play_type",
        "down", "ydstogo", "yardline_100", "half_seconds_remaining", "game_seconds_remaining", "score_differential", "qtr",
        "posteam_timeouts_remaining", "defteam_timeouts_remaining", "yards_gained", "epa", "ep", "success", "sack",
        "qb_dropback", "pass_attempt", "rush_attempt", "complete_pass", "interception", "air_yards", "yards_after_catch",
        "qb_kneel", "qb_spike", "qb_scramble", "penalty", "fixed_drive", "fixed_drive_result", "drive_start_transition",
        "punt_attempt", "punt_blocked", "kick_distance", "return_yards", "touchback", "kickoff_attempt", "roof", "wind", "temp"]
p = pd.concat([pd.read_parquet(f"{D}/pbp_{y}.parquet", columns=cols) for y in range(2017, 2027)])
p = p[p.season_type == "REG"].sort_values(["game_id", "play_id"]).reset_index(drop=True)
p["home"] = (p.posteam == p.home_team).astype(float)
p["wk"] = p.season * 100 + p.week
# ---------------- team-game metrics
pos = p[p.posteam.notna()]
sc = pos[pos.play_type.isin(["pass", "run"]) & (pos.qb_kneel == 0) & (pos.qb_spike == 0) & pos.epa.notna()]
tg = sc.groupby(["game_id", "posteam"]).agg(defteam=("defteam", "first"), wk=("wk", "first"), season=("season", "first"),
      week=("week", "first"), plays=("epa", "size"), success=("success", "mean"), pass_rate=("qb_dropback", "mean")).reset_index()
db, ru = sc[sc.qb_dropback == 1], sc[sc.qb_dropback == 0]
tg = tg.merge(db.groupby(["game_id", "posteam"]).agg(pass_epa=("epa", "mean"), sack_rate=("sack", "mean"), n_db=("epa", "size")).reset_index(), on=["game_id", "posteam"], how="left")
tg = tg.merge(ru.groupby(["game_id", "posteam"]).agg(rush_epa=("epa", "mean"), n_ru=("epa", "size")).reset_index(), on=["game_id", "posteam"], how="left")
dr = pos[pos.fixed_drive.notna()].groupby(["game_id", "posteam", "fixed_drive"]).agg(res=("fixed_drive_result", "first"),
      tr=("drive_start_transition", "first"), ep0=("ep", "first"), minyl=("yardline_100", "min")).reset_index()
dr["pts"] = dr.res.map({"Touchdown": 7.0, "Field goal": 3.0}).fillna(0.0)
dr["ep0_punt"] = np.where(dr.tr.eq("PUNT"), dr.ep0, np.nan); dr["ep0_kick"] = np.where(dr.tr.eq("KICKOFF"), dr.ep0, np.nan)
dr["rz"] = (dr.minyl <= 20).astype(float)
dg = dr.groupby(["game_id", "posteam"]).agg(ppd=("pts", "mean"), drives=("pts", "size"), points=("pts", "sum"), start_ep=("ep0", "mean"),
      ep0_punt=("ep0_punt", "mean"), ep0_kick=("ep0_kick", "mean"), rz_rate=("rz", "mean")).reset_index()
tg = tg.merge(dg, on=["game_id", "posteam"])
tg["ppp"] = tg.points / tg.plays
edp_file = "edp_team_games.parquet"
has_edp = os.path.exists(edp_file)
if has_edp:
    tg = tg.merge(pd.read_parquet(edp_file)[["game_id", "posteam", "fk_edp"]], on=["game_id", "posteam"], how="left")
for c in tg.columns:
    if tg[c].dtype.kind == "f": tg[c] = tg[c].fillna(tg[c].mean())
base_m = ["pass_epa", "rush_epa", "success", "sack_rate", "plays", "pass_rate"]
drive_m = ["ppd", "ppp", "start_ep", "ep0_punt", "ep0_kick", "rz_rate"] + (["fk_edp"] if has_edp else [])
weight_of = {"pass_epa": "n_db", "sack_rate": "n_db", "rush_epa": "n_ru", "success": "plays", "pass_rate": "plays", "ppp": "plays",
             "ppd": "drives", "start_ep": "drives", "ep0_punt": "drives", "ep0_kick": "drives", "rz_rate": "drives", "fk_edp": "drives", "plays": None}
# ---------------- weekly ridge coefficients: offense[t], defense[t]
teams = sorted(set(tg.posteam)); ix = {k: i for i, k in enumerate(teams)}; n = len(teams)
tg["is_home"] = tg.game_id.map(p.groupby("game_id").home_team.first()).eq(tg.posteam).astype(float)
rows = []
for wk in sorted(p.wk.unique()):
    if wk < 201801: continue
    past = tg[tg.wk < wk]; s, w = wk // 100, wk % 100
    age = np.where(past.season == s, 0.93 ** (w - past.week), 0.93 ** (w - 1) * 0.5 ** (s - past.season) * 0.97 ** (18 - past.week))
    X = np.zeros((len(past), 2 * n + 2)); r = np.arange(len(past))
    X[r, [ix[k] for k in past.posteam]] = 1; X[r, [n + ix[k] for k in past.defteam]] = 1; X[:, -2] = past.is_home; X[:, -1] = 1
    out = {"wk": wk, "team": teams}
    for m in base_m + drive_m:
        wt = age * (past[weight_of[m]].to_numpy() if weight_of[m] else 1.0); wt = wt / wt.mean()
        P = np.full(2 * n + 2, 15.0); P[-2:] = 1e-3
        b = np.linalg.solve(X.T @ (wt[:, None] * X) + np.diag(P), X.T @ (wt * past[m].to_numpy()))
        out[f"off_{m}"], out[f"def_{m}"] = b[:n], b[n:2 * n]
    rows.append(pd.DataFrame(out))
R = pd.concat(rows)
# Team-relative drive ratings: subtract each week's league mean, so rule changes (2024-25 kickoffs) and scoring
# trends don't show up as out-of-range values the trees never saw.
for m in drive_m:
    for side in ("off", "def"):
        R[f"{side}_{m}"] = R[f"{side}_{m}"] - R.groupby("wk")[f"{side}_{m}"].transform("mean")
offR = R[["wk", "team"] + [c for c in R if c.startswith("off_")]].rename(columns={"team": "posteam"})
defR = R[["wk", "team"] + [c for c in R if c.startswith("def_")]].rename(columns={"team": "defteam"})
p = p[p.season >= 2018].merge(offR, on=["wk", "posteam"], how="left").merge(defR, on=["wk", "defteam"], how="left")
p["outdoor"] = p.roof.isin(["outdoors", "open"]).astype(float)
p["wind_f"] = np.where(p.outdoor == 1, p.wind.fillna(8), 0.0); p["temp_f"] = np.where(p.outdoor == 1, p.temp.fillna(60), 70.0)
situation = ["down", "ydstogo", "yardline_100", "half_seconds_remaining", "game_seconds_remaining", "score_differential", "qtr",
             "posteam_timeouts_remaining", "defteam_timeouts_remaining", "home", "outdoor", "wind_f", "temp_f"]
team = lambda ms: [f"{side}_{m}" for m in ms for side in ("off", "def")]
BASE = situation + team(base_m)
SETS = {"+ points per drive, points per play": team(["ppd", "ppp"]),
        "+ drive start EP (all, punts, kickoffs)": team(["start_ep", "ep0_punt", "ep0_kick"]),
        "+ red-zone trips per drive": team(["rz_rate"])}
if has_edp: SETS["+ EDP (rebuilt)"] = team(["fk_edp"])
SETS["+ all drive features"] = sum(SETS.values(), [])
# ---------------- model targets, as the sim's models are trained
snap = p.play_type.isin(["pass", "run"]) & (p.qb_kneel == 0) & (p.qb_spike == 0) & p.down.notna() & (p.penalty == 0)
binned = lambda x, edges: np.digitize(x, edges)
tasks = {}
pt = p[p.down.notna() & p.play_type.isin(["pass", "run", "punt", "field_goal"]) & (p.qb_kneel == 0) & (p.qb_spike == 0)].copy()
tasks["play (pass/run/punt/FG)"] = (pt, pt.play_type.map({"pass": 0, "run": 1, "punt": 2, "field_goal": 3}), [])
ru = p[snap & (p.rush_attempt == 1) & (p.qb_scramble == 0)].copy()
tasks["rush yards (binned)"] = (ru, binned(ru.yards_gained, [-1, 1, 2, 3, 4, 5, 6, 8, 11, 16, 26]), [])
dbk = p[snap & (p.qb_dropback == 1)].copy()
tasks["sack | dropback"] = (dbk, dbk.sack.astype(int), [])
th = p[snap & (p.pass_attempt == 1) & (p.sack == 0) & p.air_yards.notna()].copy()
tasks["completion | throw"] = (th, th.complete_pass.astype(int), ["air_yards"])
tasks["interception | throw"] = (th, th.interception.astype(int), ["air_yards"])
tasks["air yards (binned)"] = (th, binned(th.air_yards, [-2, 1, 3, 5, 7, 10, 13, 16, 20, 25, 30, 40]), [])
cp = th[(th.complete_pass == 1) & th.yards_after_catch.notna()].copy()
tasks["YAC (binned)"] = (cp, binned(cp.yards_after_catch, [1, 2, 3, 4, 6, 8, 11, 16, 25]), ["air_yards"])
pu = p[(p.punt_attempt == 1) & (p.punt_blocked != 1) & p.kick_distance.notna()].copy()
pu["net"] = pu.kick_distance - pu.return_yards.fillna(0) - 20 * pu.touchback.fillna(0)
tasks["punt net yards (binned)"] = (pu, binned(pu.net, [25, 30, 35, 38, 41, 44, 47, 50, 55]), [])
ko = p[(p.kickoff_attempt == 1)].copy()
nxt = p.groupby("game_id").yardline_100.shift(-1)
ko["start"] = 100 - nxt.loc[ko.index]   # receiving team's own yardline at the next snap
ko = ko[ko.start.between(1, 99)]
ko["rule"] = np.select([ko.season >= 2025, ko.season == 2024], [2, 1], 0)
# on kickoffs nflverse's posteam is the receiving team, defteam the kicking team
# The 2025 kickoff rule has no 2018-24 examples, so kickoffs train through 2025 and test on 2026 (as the live bundle does).
tasks["kickoff start (binned), test 2026"] = (ko, binned(ko.start, [20, 25, 28, 30, 32, 35, 40]), ["rule"])
def run(frame, y, extra, last_train=2024):
    y = np.asarray(y); tr = (frame.season <= last_train).to_numpy(); te = ~tr
    k = len(np.unique(y)); obj = dict(objective="binary") if k == 2 else dict(objective="multiclass", num_class=int(y.max()) + 1)
    def ll(feats):
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=300, subsample=0.8,
                               subsample_freq=1, colsample_bytree=0.8, verbose=-1, **obj)
        m.fit(frame.loc[tr, feats], y[tr])
        pr = m.predict_proba(frame.loc[te, feats]); pr = np.clip(pr[np.arange(te.sum()), y[te]], 1e-12, 1)
        return -np.log(pr)
    base = ll(BASE + extra); out = []
    games = frame.loc[te, "game_id"].to_numpy()
    for name, f in SETS.items():
        d = ll(BASE + extra + f) - base
        g = pd.Series(d).groupby(games).sum(); se = g.std() * np.sqrt(len(g)) / len(d)
        out.append((name, 100 * d.mean() / base.mean(), d.mean() / se))
    return base.mean(), int(te.sum()), out
res = []
for name, (frame, y, extra) in tasks.items():
    b, nt, out = run(frame, y, extra, 2025 if name.startswith("kickoff") else 2024)
    for s, pct, z in out: res.append((name, nt, b, s, pct, z))
    print(name, "done", flush=True)
r = pd.DataFrame(res, columns=["model", "test_rows", "base_logloss", "added", "pct_logloss_change", "z"])
print(r.round(3).to_string(index=False)); r.to_csv("play_features_experiment.csv", index=False)
