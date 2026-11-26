"""Split, charting and tracking features as inputs to the simulator's per-play models.

Same design as exp_play_features.py: LightGBM per sim model, situation + baseline ratings vs the same plus a
feature family; held-out log loss change (negative = better) with game-clustered z.

Families
  team splits     offense/defense ridge ratings (opponent-adjusted, refit weekly, league mean removed) of pass EPA by
                  location, rush EPA inside/outside and by direction, middle-pass share, 3rd-down conversion,
                  early-down EPA, shotgun and no-huddle rates
  FTN charting    team play-action rate and play-action EPA, screen and motion rates; defense blitz rate, box count,
                  pass rushers (2022+; trained 2023-24, tested 2025-26 on the same rows as its own baseline)
  NGS tracking    player, strictly prior and recency-weighted: QB time to throw, aggressiveness, intended air yards,
                  NGS CPOE; receiver separation, cushion, YAC over expected; rusher RYOE/att, efficiency, time to LOS,
                  8+ box share. Baseline adds the box-score player ratings the sim has (passer EPA, CPOE, sack rate;
                  receiver yards/target, catch rate; rusher yards/carry), built the same way.
  location stage  the play's actual pass location or run location/gap (as if the sim drew it first), alone and with
                  team ratings matched to that location
"""
import os, numpy as np, pandas as pd, lightgbm as lgb
D = os.environ.get("PBP_DIR", "data/pbp")
cols = ["game_id", "play_id", "season", "week", "season_type", "posteam", "defteam", "home_team", "play_type", "down", "ydstogo",
        "yardline_100", "half_seconds_remaining", "game_seconds_remaining", "score_differential", "qtr", "posteam_timeouts_remaining",
        "defteam_timeouts_remaining", "yards_gained", "epa", "success", "sack", "qb_dropback", "pass_attempt", "rush_attempt",
        "complete_pass", "interception", "air_yards", "yards_after_catch", "qb_kneel", "qb_spike", "qb_scramble", "penalty",
        "pass_location", "run_location", "run_gap", "shotgun", "no_huddle", "third_down_converted", "third_down_failed",
        "passer_player_id", "receiver_player_id", "rusher_player_id", "cpoe", "roof", "wind", "temp"]
p = pd.concat([pd.read_parquet(f"{D}/pbp_{y}.parquet", columns=cols) for y in range(2016, 2027)])
p = p[(p.season_type == "REG") & p.posteam.notna()].sort_values(["game_id", "play_id"]).reset_index(drop=True)
p["home"] = (p.posteam == p.home_team).astype(float); p["wk"] = p.season * 100 + p.week
ftn = pd.concat([pd.read_parquet(f"{D}/ftn_{y}.parquet") for y in range(2022, 2027)])
ftn = ftn.rename(columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})
ftn["play_id"] = pd.to_numeric(ftn.play_id, errors="coerce")
fcols = ["is_play_action", "is_screen_pass", "is_motion", "is_rpo", "n_defense_box", "n_blitzers", "n_pass_rushers"]
p = p.merge(ftn[["game_id", "play_id"] + fcols].drop_duplicates(["game_id", "play_id"]), on=["game_id", "play_id"], how="left")
for c in fcols: p[c] = pd.to_numeric(p[c], errors="coerce").astype(float)
scrim = p.play_type.isin(["pass", "run"]) & (p.qb_kneel == 0) & (p.qb_spike == 0) & p.epa.notna()
sc = p[scrim].copy()
sc["is_db"] = sc.qb_dropback == 1
sc["rush_out"] = np.where(~sc.is_db & sc.run_gap.eq("end"), sc.epa, np.nan)
sc["rush_in"] = np.where(~sc.is_db & sc.run_location.notna() & ~sc.run_gap.eq("end"), sc.epa, np.nan)
for loc in ("left", "middle", "right"):
    sc[f"pass_{loc}"] = np.where(sc.is_db & sc.pass_location.eq(loc), sc.epa, np.nan)
    sc[f"rush_{loc}"] = np.where(~sc.is_db & sc.run_location.eq(loc), sc.epa, np.nan)
sc["mid_share"] = np.where(sc.pass_location.notna(), sc.pass_location.eq("middle").astype(float), np.nan)
sc["third_conv"] = np.where(sc.down == 3, sc.third_down_converted.fillna(0), np.nan)
sc["early_epa"] = np.where(sc.down <= 2, sc.epa, np.nan)
sc["pa_rate"] = np.where(sc.is_db & sc.is_play_action.notna(), sc.is_play_action, np.nan)
sc["pa_epa"] = np.where(sc.is_db & (sc.is_play_action == 1), sc.epa, np.nan)
sc["screen_rate"] = np.where(sc.is_db & sc.is_screen_pass.notna(), sc.is_screen_pass, np.nan)
sc["blitz_rate"] = np.where(sc.is_db & sc.n_blitzers.notna(), (sc.n_blitzers > 0).astype(float), np.nan)
sc["rushers"] = np.where(sc.is_db, sc.n_pass_rushers, np.nan)
sc["box"] = sc.n_defense_box; sc["motion_rate"] = sc.is_motion
sc["pass_epa"] = np.where(sc.is_db, sc.epa, np.nan); sc["rush_epa"] = np.where(~sc.is_db, sc.epa, np.nan)
sc["sack_rate"] = np.where(sc.is_db, sc.sack, np.nan); sc["pass_rate"] = sc.is_db.astype(float)
split_m = ["pass_left", "pass_middle", "pass_right", "rush_left", "rush_middle", "rush_right", "rush_in", "rush_out",
           "mid_share", "third_conv", "early_epa", "shotgun", "no_huddle"]
ftn_m = ["pa_rate", "pa_epa", "screen_rate", "motion_rate", "blitz_rate", "box", "rushers"]
base_m = ["pass_epa", "rush_epa", "success", "sack_rate", "pass_rate"]
allm = base_m + split_m + ftn_m
agg = {m: (m, "mean") for m in allm}; cnt = {f"n_{m}": (m, "count") for m in allm}
tg = sc.groupby(["game_id", "posteam"]).agg(defteam=("defteam", "first"), wk=("wk", "first"), season=("season", "first"),
                                            week=("week", "first"), home=("home", "first"), **agg, **cnt).reset_index()
# ---------------- weekly ridge ratings (offense, defense), league mean removed
teams = sorted(set(tg.posteam)); ix = {k: i for i, k in enumerate(teams)}; n = len(teams)
rows = []
for wk in sorted(p.wk.unique()):
    if wk < 201701: continue
    past = tg[tg.wk < wk]; s, w = wk // 100, wk % 100
    age = np.where(past.season == s, 0.93 ** (w - past.week), 0.93 ** (w - 1) * 0.5 ** (s - past.season) * 0.97 ** (18 - past.week))
    out = {"wk": wk, "team": teams}
    for m in allm:
        ok = past[f"n_{m}"].to_numpy() > 0
        if ok.sum() < 64:
            out[f"off_{m}"] = out[f"def_{m}"] = np.full(n, np.nan); continue
        q = past[ok]; X = np.zeros((len(q), 2 * n + 2)); r = np.arange(len(q))
        X[r, [ix[k] for k in q.posteam]] = 1; X[r, [n + ix[k] for k in q.defteam]] = 1; X[:, -2] = q.home; X[:, -1] = 1
        wt = age[ok] * q[f"n_{m}"].to_numpy(); wt = wt / wt.mean()
        P = np.full(2 * n + 2, 15.0); P[-2:] = 1e-3
        b = np.linalg.solve(X.T @ (wt[:, None] * X) + np.diag(P), X.T @ (wt * q[m].to_numpy()))
        out[f"off_{m}"], out[f"def_{m}"] = b[:n] - b[:n].mean(), b[n:2 * n] - b[n:2 * n].mean()
    rows.append(pd.DataFrame(out))
R = pd.concat(rows)
p = p[p.season >= 2018].merge(R[["wk", "team"] + [c for c in R if c.startswith("off_")]].rename(columns={"team": "posteam"}), on=["wk", "posteam"], how="left")
p = p.merge(R[["wk", "team"] + [c for c in R if c.startswith("def_")]].rename(columns={"team": "defteam"}), on=["wk", "defteam"], how="left")
# ---------------- player ratings: strictly prior, recency-weighted (0.92 per game played), from pbp and from NGS weekly
def player_prior(frame, pid, value_cols, weight_col, prefix, decay=0.92):
    """frame has one row per player-week (or player-game) with season, week, values and a weight; returns prior EW means."""
    f = frame.sort_values([pid, "season", "week"]).copy(); res = f[[pid, "season", "week"]].copy()
    for c in value_cols:
        num = (f[c].fillna(0) * f[weight_col]).astype(float); den = f[weight_col].where(f[c].notna(), 0).astype(float)
        def ew(s):
            out = np.empty(len(s)); acc = 0.0
            for i, v in enumerate(s.to_numpy()): out[i] = acc; acc = acc * decay + v
            return pd.Series(out, index=s.index)
        sn, sd = num.groupby(f[pid]).transform(ew), den.groupby(f[pid]).transform(ew)
        res[f"{prefix}_{c}"] = np.where(sd > 3, sn / sd.where(sd > 0), np.nan)
    return res
pg = sc.groupby(["passer_player_id", "season", "week"]).agg(p_epa=("pass_epa", "mean"), p_cpoe=("cpoe", "mean"), p_sack=("sack_rate", "mean"), n=("pass_epa", "count")).reset_index()
pp = player_prior(pg, "passer_player_id", ["p_epa", "p_cpoe", "p_sack"], "n", "qb")
th_all = sc[sc.pass_location.notna() & sc.receiver_player_id.notna()]
rg = th_all.groupby(["receiver_player_id", "season", "week"]).agg(r_ypt=("yards_gained", "mean"), r_catch=("complete_pass", "mean"),
                                                                   r_yac=("yards_after_catch", "mean"), n=("yards_gained", "size")).reset_index()
rp = player_prior(rg, "receiver_player_id", ["r_ypt", "r_catch", "r_yac"], "n", "wr")
ru_all = sc[~sc.is_db & sc.rusher_player_id.notna()]
hg = ru_all.groupby(["rusher_player_id", "season", "week"]).agg(h_ypc=("yards_gained", "mean"), n=("yards_gained", "size")).reset_index()
hp = player_prior(hg, "rusher_player_id", ["h_ypc"], "n", "rb")
ngp = pd.read_parquet(f"{D}/ngs_passing.parquet"); ngr = pd.read_parquet(f"{D}/ngs_receiving.parquet"); ngh = pd.read_parquet(f"{D}/ngs_rushing.parquet")
ngp, ngr, ngh = [x[(x.season_type == "REG") & (x.week > 0)] for x in (ngp, ngr, ngh)]
npass = player_prior(ngp.rename(columns={"player_gsis_id": "passer_player_id"}), "passer_player_id",
                     ["avg_time_to_throw", "aggressiveness", "avg_intended_air_yards", "completion_percentage_above_expectation"], "attempts", "ngsqb")
nrec = player_prior(ngr.rename(columns={"player_gsis_id": "receiver_player_id"}), "receiver_player_id",
                    ["avg_separation", "avg_cushion", "avg_yac_above_expectation", "avg_intended_air_yards"], "targets", "ngswr")
nrush = player_prior(ngh.rename(columns={"player_gsis_id": "rusher_player_id"}), "rusher_player_id",
                     ["rush_yards_over_expected_per_att", "efficiency", "avg_time_to_los", "percent_attempts_gte_eight_defenders"], "rush_attempts", "ngsrb")
def attach(frame, prior, pid):
    # the latest prior row at or before this week (the row for this week holds only earlier weeks)
    pr = prior.sort_values(["season", "week"]); fr = frame.reset_index().sort_values(["season", "week"])
    pr["key"] = pr.season * 100 + pr.week; fr["key"] = fr.season * 100 + fr.week
    out = pd.merge_asof(fr, pr.drop(columns=["season", "week"]).sort_values("key"), on="key", by=pid, direction="backward")
    return out.set_index("index").sort_index().drop(columns="key")
# ---------------- tasks
p["outdoor"] = p.roof.isin(["outdoors", "open"]).astype(float)
p["wind_f"] = np.where(p.outdoor == 1, p.wind.fillna(8), 0.0); p["temp_f"] = np.where(p.outdoor == 1, p.temp.fillna(60), 70.0)
situation = ["down", "ydstogo", "yardline_100", "half_seconds_remaining", "game_seconds_remaining", "score_differential", "qtr",
             "posteam_timeouts_remaining", "defteam_timeouts_remaining", "home", "outdoor", "wind_f", "temp_f"]
team = lambda ms: [f"{s}_{m}" for m in ms for s in ("off", "def")]
snap = p.play_type.isin(["pass", "run"]) & (p.qb_kneel == 0) & (p.qb_spike == 0) & p.down.notna() & (p.penalty == 0)
binned = lambda x, e: np.digitize(x, e)
th = p[snap & (p.pass_attempt == 1) & (p.sack == 0) & p.air_yards.notna()]
th = attach(attach(attach(attach(th, pp, "passer_player_id"), rp, "receiver_player_id"), npass, "passer_player_id"), nrec, "receiver_player_id")
dbk = attach(attach(p[snap & (p.qb_dropback == 1)], pp, "passer_player_id"), npass, "passer_player_id")
ru = p[snap & (p.rush_attempt == 1) & (p.qb_scramble == 0)]
ru = attach(attach(ru, hp, "rusher_player_id"), nrush, "rusher_player_id")
pt = p[p.down.notna() & p.play_type.isin(["pass", "run", "punt", "field_goal"]) & (p.qb_kneel == 0) & (p.qb_spike == 0)]
cp = th[(th.complete_pass == 1) & th.yards_after_catch.notna()]
QB, WR, RB = ["qb_p_epa", "qb_p_cpoe", "qb_p_sack"], ["wr_r_ypt", "wr_r_catch", "wr_r_yac"], ["rb_h_ypc"]
NQB = [c for c in npass if c.startswith("ngsqb_")]; NWR = [c for c in nrec if c.startswith("ngswr_")]; NRB = [c for c in nrush if c.startswith("ngsrb_")]
for f in (th, cp):
    f["loc_middle"] = f.pass_location.eq("middle").astype(float); f["loc_left"] = f.pass_location.eq("left").astype(float)
    for side in ("off", "def"):  # team rating at the location of this throw
        f[f"{side}_pass_at_loc"] = np.select([f.pass_location.eq(l) for l in ("left", "middle", "right")],
                                             [f[f"{side}_pass_{l}"] for l in ("left", "middle", "right")], np.nan)
ru["gap_code"] = ru.run_location.map({"left": 0, "middle": 1, "right": 2}).fillna(1) * 3 + ru.run_gap.map({"end": 0, "tackle": 1, "guard": 2}).fillna(2)
for side in ("off", "def"):
    ru[f"{side}_rush_at_loc"] = np.where(ru.run_gap.eq("end"), ru[f"{side}_rush_out"], ru[f"{side}_rush_in"])
    ru[f"{side}_rush_at_dir"] = np.select([ru.run_location.eq(l) for l in ("left", "middle", "right")], [ru[f"{side}_rush_{l}"] for l in ("left", "middle", "right")], np.nan)
TASKS = {  # name: frame, target, model-specific baseline inputs
    "play call": (pt, pt.play_type.map({"pass": 0, "run": 1, "punt": 2, "field_goal": 3}), []),
    "rush yards": (ru, binned(ru.yards_gained, [-1, 1, 2, 3, 4, 5, 6, 8, 11, 16, 26]), RB),
    "sack | dropback": (dbk, dbk.sack.astype(int), QB),
    "completion | throw": (th, th.complete_pass.astype(int), ["air_yards"] + QB + WR),
    "interception | throw": (th, th.interception.astype(int), ["air_yards"] + QB + WR),
    "air yards": (th, binned(th.air_yards, [-2, 1, 3, 5, 7, 10, 13, 16, 20, 25, 30, 40]), QB + WR),
    "YAC": (cp, binned(cp.yards_after_catch, [1, 2, 3, 4, 6, 8, 11, 16, 25]), ["air_yards"] + QB + WR),
}
FAMILIES = {
    "team: pass EPA by location + middle share": team(["pass_left", "pass_middle", "pass_right", "mid_share"]),
    "team: rush EPA inside/outside + by direction": team(["rush_in", "rush_out", "rush_left", "rush_middle", "rush_right"]),
    "team: 3rd-down conv, early-down EPA": team(["third_conv", "early_epa"]),
    "team: shotgun, no-huddle rates": team(["shotgun", "no_huddle"]),
}
NGS_FOR = {"sack | dropback": NQB, "completion | throw": NQB + NWR, "interception | throw": NQB + NWR, "air yards": NQB + NWR,
           "YAC": NQB + NWR, "rush yards": NRB}
LOC_FOR = {"completion | throw": (["loc_middle", "loc_left"], ["off_pass_at_loc", "def_pass_at_loc"]),
           "interception | throw": (["loc_middle", "loc_left"], ["off_pass_at_loc", "def_pass_at_loc"]),
           "YAC": (["loc_middle", "loc_left"], ["off_pass_at_loc", "def_pass_at_loc"]),
           "rush yards": (["gap_code"], ["off_rush_at_loc", "def_rush_at_loc", "off_rush_at_dir", "def_rush_at_dir"])}
def fit_ll(frame, y, feats, tr, te):
    k = len(np.unique(y)); obj = dict(objective="binary") if k == 2 else dict(objective="multiclass", num_class=int(y.max()) + 1)
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=300, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, verbose=-1, **obj)
    m.fit(frame.loc[tr, feats], y[tr]); pr = m.predict_proba(frame.loc[te, feats])
    return -np.log(np.clip(pr[np.arange(te.sum()), y[te]], 1e-12, 1))
res = []
def compare(task, frame, y, base, added, label, tr, te):
    b = fit_ll(frame, y, base, tr, te); d = fit_ll(frame, y, base + added, tr, te) - b
    g = pd.Series(d).groupby(frame.loc[te, "game_id"].to_numpy()).sum(); se = g.std() * np.sqrt(len(g)) / len(d)
    res.append((task, label, int(te.sum()), 100 * d.mean() / b.mean(), d.mean() / se)); print(task, "|", label, round(100 * d.mean() / b.mean(), 3), round(d.mean() / se, 2), flush=True)
for task, (frame, y, extra) in TASKS.items():
    frame = frame.reset_index(drop=True); y = np.asarray(y)
    tr = (frame.season <= 2024).to_numpy(); te = ~tr
    base = situation + team(base_m) + extra
    for label, f in FAMILIES.items(): compare(task, frame, y, base, f, label, tr, te)
    if task in NGS_FOR: compare(task, frame, y, base, NGS_FOR[task], "player: Next Gen Stats tracking", tr, te)
    if task in LOC_FOR:
        loc, matched = LOC_FOR[task]
        compare(task, frame, y, base, loc, "location stage: actual location only", tr, te)
        compare(task, frame, y, base + loc, matched, "location stage: + team ratings at that location", tr, te)
    # FTN charting: 2022+ only, own baseline on the same rows
    sub = frame[frame.season >= 2023].reset_index(drop=True); ys = y[(frame.season >= 2023).to_numpy()]
    compare(task, sub, ys, base, team(ftn_m), "team: FTN charting (play action, screens, motion, blitz, box, rushers)",
            (sub.season <= 2024).to_numpy(), (sub.season >= 2025).to_numpy())
r = pd.DataFrame(res, columns=["model", "added", "test_rows", "pct_logloss_change", "z"])
print(r.round(3).to_string(index=False)); r.to_csv("split_features_experiment.csv", index=False)
