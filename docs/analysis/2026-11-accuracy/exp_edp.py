"""Earned Drive Points (Fortgang, SportfolioKings, June 2023), rebuilt from the article's description on public data.

1. Team-aware score expectation: LightGBM on the EP model's situation (down, distance, yardline, half seconds,
   score differential, quarter, timeouts, home) plus rolling, strictly prior offense and defense strength over the
   last 3 and last 16 games (yards per play, success rate, explosive-play rate, sack rate). Target: the points the
   drive ends with (TD 7, FG 3, else 0). Walk forward: season S is scored by a model fit on seasons before S.
2. Per drive, the article's four components:
   gain   expectation before the drive's last snap minus the expectation at its first snap. The last play's own
          result is left out, so a fumble at the 5 keeps the credit for reaching it and a one-play 80-yard TD earns
          little (the article's Team A / Team B examples).
   vol    standard deviation of the snap-to-snap changes in expectation (big swings get "nicked")
   avg    mean expectation over the drive's snaps minus the starting expectation
   ypp    yards per scrimmage play, penalty yardage excluded
3. EDP = a linear combination of the components (plus the starting expectation, so EDP is on a points scale),
   weighted by regressing drive points on them within earlier seasons only. The article doesn't publish weights.
Writes edp_team_games.parquet: one row per team-game with the offense's mean EDP per drive.
"""
import os, numpy as np, pandas as pd, lightgbm as lgb
D = os.environ.get("PBP_DIR", "data/pbp")
cols = ["game_id", "play_id", "season", "week", "season_type", "game_date", "posteam", "defteam", "home_team", "play_type",
        "down", "ydstogo", "yardline_100", "half_seconds_remaining", "score_differential", "qtr", "posteam_timeouts_remaining",
        "defteam_timeouts_remaining", "yards_gained", "success", "sack", "qb_dropback", "pass", "rush", "qb_kneel", "qb_spike",
        "fixed_drive", "fixed_drive_result", "penalty"]
p = pd.concat([pd.read_parquet(f"{D}/pbp_{y}.parquet", columns=cols) for y in range(2016, 2027)])
p = p[(p.season_type == "REG") & p.posteam.notna() & p.fixed_drive.notna()].sort_values(["game_id", "play_id"]).reset_index(drop=True)
p["home"] = (p.posteam == p.home_team).astype(float)
snap = p.play_type.isin(["pass", "run"]) & (p.qb_kneel == 0) & (p.qb_spike == 0) & p.down.notna()
# --- rolling team strength, strictly before each game
sc = p[snap].copy()
sc["explosive"] = ((sc["pass"] == 1) & (sc.yards_gained >= 20)) | ((sc.rush == 1) & (sc.yards_gained >= 10))
tg = sc.groupby(["game_id", "posteam", "defteam"]).agg(date=("game_date", "first"), ypp=("yards_gained", "mean"),
      succ=("success", "mean"), expl=("explosive", "mean"), sack=("sack", "sum"), db=("qb_dropback", "sum")).reset_index()
tg["sackr"] = tg.sack / tg.db.clip(lower=1)
stats = ["ypp", "succ", "expl", "sackr"]
def rolling(side, team_col):
    x = tg.sort_values(["date", "game_id"]).copy()
    out = x[["game_id", team_col]].copy()
    for k in (3, 16):
        r = x.groupby(team_col)[stats].transform(lambda s: s.shift().rolling(k, min_periods=1).mean())
        for c in stats: out[f"{side}_{c}_{k}"] = r[c]
    return out.rename(columns={team_col: "posteam" if side == "off" else "defteam"})
p = p.merge(rolling("off", "posteam"), on=["game_id", "posteam"], how="left").merge(rolling("def", "defteam"), on=["game_id", "defteam"], how="left")
strength = [c for c in p.columns if c.startswith(("off_", "def_")) and c[-2:] in ("_3", "16")]
situation = ["down", "ydstogo", "yardline_100", "half_seconds_remaining", "score_differential", "qtr",
             "posteam_timeouts_remaining", "defteam_timeouts_remaining", "home"]
X = situation + strength
p["drive_pts"] = p.fixed_drive_result.map({"Touchdown": 7.0, "Field goal": 3.0}).fillna(0.0)
p["xs"] = np.nan
s_idx = p.index[snap]
for s in range(2017, 2027):
    tr = p.loc[s_idx][p.loc[s_idx].season < s]; te = s_idx[p.loc[s_idx].season == s]
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=200, verbose=-1)
    m.fit(tr[X], tr.drive_pts); p.loc[te, "xs"] = m.predict(p.loc[te, X])
# --- drive components
q = p.loc[s_idx].dropna(subset=["xs"]).copy()
q["ypp_clean"] = np.where(q.penalty == 1, np.nan, q.yards_gained)
q["dx"] = q.groupby(["game_id", "posteam", "fixed_drive"]).xs.diff()
g = q.groupby(["game_id", "posteam", "fixed_drive"])
d = g.agg(season=("season", "first"), week=("week", "first"), defteam=("defteam", "first"), start=("xs", "first"),
          last=("xs", "last"), avg=("xs", "mean"), vol=("dx", "std"), ypp=("ypp_clean", "mean"), n=("xs", "size"),
          pts=("drive_pts", "first")).reset_index()
d["gain"], d["avg_rel"] = d["last"] - d.start, d.avg - d.start
d["vol"], d["ypp"] = d.vol.fillna(0.0), d.ypp.fillna(0.0)
comp = ["start", "gain", "vol", "avg_rel", "ypp"]
d["edp"] = np.nan
for s in range(2017, 2027):
    tr, te = d[d.season < s], d.index[d.season == s]
    if len(tr) == 0:  # first season: plain sum of the expectation terms
        d.loc[te, "edp"] = d.loc[te, "start"] + d.loc[te, "gain"]; continue
    A = np.c_[np.ones(len(tr)), tr[comp]]; b = np.linalg.lstsq(A, tr.pts, rcond=None)[0]
    d.loc[te, "edp"] = np.c_[np.ones(len(te)), d.loc[te, comp]] @ b
    if s == 2026: print("EDP weights fit on 2017-25 (intercept, " + ", ".join(comp) + "):", np.round(b, 3))
out = d.groupby(["game_id", "posteam"]).agg(fk_edp=("edp", "mean"), fk_start=("start", "mean"), fk_gain=("gain", "mean"), fk_vol=("vol", "mean"),
                                          fk_avg_rel=("avg_rel", "mean"), fk_ypp=("ypp", "mean")).reset_index()
out.to_parquet("edp_team_games.parquet")
print(len(d), "drives;", len(out), "team-games; corr(EDP, drive points) by drive:", round(np.corrcoef(d.edp, d.pts)[0, 1], 3))
