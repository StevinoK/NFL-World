"""Does a run-location stage make simulated rush yards more accurate, or only the oracle (actual location)?

Held-out 2025-26 rush yards, log loss of the full predictive distribution:
  A  one-step  P(y | x)                                   x = situation, team ratings, rusher rating
  B  one-step  P(y | x, direction ratings, tendencies)    the same information as the two-step, no stage
  C  two-step  sum_l P(l | x, tendencies) P(y | l, x, ratings at l)    what a sim that draws l would produce
  D  two-step without matchup ratings: sum_l P(l | x) P(y | l, x)
  O  oracle    P(y | actual l, x, ratings at l)           not available before the snap
"""
src = open("exp_split_features.py").read()
exec(src[:src.index("TASKS = {")])
# offense run-direction tendencies: strictly prior, recency-weighted team shares
runs = sc[~sc.is_db & sc.run_location.notna()].copy()
runs["out_share"] = runs.run_gap.eq("end").astype(float)
runs["left_share"] = runs.run_location.eq("left").astype(float); runs["mid_share_run"] = runs.run_location.eq("middle").astype(float)
tt = runs.groupby(["posteam", "season", "week"]).agg(out_share=("out_share", "mean"), left_share=("left_share", "mean"),
                                                     mid_share_run=("mid_share_run", "mean"), n=("out_share", "size")).reset_index()
tend = player_prior(tt, "posteam", ["out_share", "left_share", "mid_share_run"], "n", "tend")
ru2 = attach(ru, tend, "posteam").reset_index(drop=True)
ru2 = ru2[ru2.run_location.notna()].reset_index(drop=True)
ru2["l"] = ru2.run_location.map({"left": 0, "middle": 1, "right": 2}).astype(int) * 3 + ru2.run_gap.map({"end": 0, "tackle": 1, "guard": 2}).fillna(2).astype(int)
codes = sorted(ru2.l.unique()); ru2["l"] = ru2.l.map({c: i for i, c in enumerate(codes)})
y = binned(ru2.yards_gained, [-1, 1, 2, 3, 4, 5, 6, 8, 11, 16, 26]); K = int(y.max()) + 1
tr = (ru2.season <= 2024).to_numpy(); te = ~tr
TEND = ["tend_out_share", "tend_left_share", "tend_mid_share_run"]
DIR = team(["rush_in", "rush_out", "rush_left", "rush_middle", "rush_right"])
base = situation + team(base_m) + RB
def model(obj_k):
    obj = dict(objective="multiclass", num_class=obj_k)
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=300, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.8, verbose=-1, **obj)
def matched(frame, l_col):
    """offense/defense ratings at location l (inside/outside and direction)"""
    loc = frame[l_col].map(dict(enumerate(codes)))
    direction, gap = loc // 3, loc % 3
    out = pd.DataFrame(index=frame.index)
    for side in ("off", "def"):
        out[f"{side}_at_gap"] = np.where(gap == 0, frame[f"{side}_rush_out"], frame[f"{side}_rush_in"])
        out[f"{side}_at_dir"] = np.select([direction == 0, direction == 1, direction == 2],
                                          [frame[f"{side}_rush_left"], frame[f"{side}_rush_middle"], frame[f"{side}_rush_right"]], np.nan)
    out["l"] = frame[l_col].to_numpy()
    return out
def ll(prob): return -np.log(np.clip(prob[np.arange(len(prob)), y[te]], 1e-12, 1))
res = {}
mA = model(K).fit(ru2.loc[tr, base], y[tr]); res["A one-step, base inputs"] = ll(mA.predict_proba(ru2.loc[te, base]))
mB = model(K).fit(ru2.loc[tr, base + DIR + TEND], y[tr]); res["B one-step + direction ratings + tendencies"] = ll(mB.predict_proba(ru2.loc[te, base + DIR + TEND]))
# location model
Xl = base + DIR + TEND
ml = model(len(codes)).fit(ru2.loc[tr, Xl], ru2.l[tr]); pl = ml.predict_proba(ru2.loc[te, Xl])
loc_ll = -np.log(np.clip(pl[np.arange(te.sum()), ru2.l[te]], 1e-12, 1)).mean()
prior = np.bincount(ru2.l[tr], minlength=len(codes)) / tr.sum()
print(f"location model log loss {loc_ll:.3f} vs league frequencies {-np.log(prior[ru2.l[te]]).mean():.3f}")
# outcome models given l
M_tr = matched(ru2.loc[tr], "l"); feats_C = base + list(M_tr.columns)
mC = model(K).fit(pd.concat([ru2.loc[tr, base], M_tr], axis=1)[feats_C], y[tr])
mD = model(K).fit(ru2.loc[tr, base].assign(l=ru2.l[tr]), y[tr])
test = ru2.loc[te].copy()
mixC, mixD = np.zeros((te.sum(), K)), np.zeros((te.sum(), K))
for i in range(len(codes)):
    test["l_draw"] = i
    Mi = matched(test, "l_draw").rename(columns={"l": "l"})
    mixC += pl[:, [i]] * mC.predict_proba(pd.concat([test[base], Mi], axis=1)[feats_C])
    mixD += pl[:, [i]] * mD.predict_proba(test[base].assign(l=i))
res["C two-step, drawn location, matchup ratings"] = ll(mixC)
res["D two-step, drawn location, no matchup ratings"] = ll(mixD)
Mo = matched(test, "l")
res["O oracle: actual location, matchup ratings"] = ll(mC.predict_proba(pd.concat([test[base], Mo], axis=1)[feats_C]))
games = ru2.loc[te, "game_id"].to_numpy(); a = res["A one-step, base inputs"]
print(f"{te.sum()} held-out rushes 2025-26; change vs A in % log loss, z clustered by game")
for k, v in res.items():
    d = v - a; g = pd.Series(d).groupby(games).sum(); z = d.mean() / (g.std() * np.sqrt(len(g)) / len(d)) if d.std() > 0 else 0
    print(f"  {k:50s} {v.mean():.4f}  {100 * d.mean() / a.mean():+6.2f}%  z {z:+5.1f}")
# the same check in expectation: simulated mean yards by team-game vs one-step
