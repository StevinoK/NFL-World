"""Does a drawn pass-location stage (left / middle / right) make completion, interception and YAC more accurate?

Held-out 2025-26 throws. In the sim, air yards and the target are drawn before completion, so the stage sits
between them: P(l | x), with x = situation, team ratings, air yards, passer and receiver ratings, and location
tendencies (offense middle share, defense middle share allowed, receiver's prior left/middle shares).
  A  one-step  P(y | x without tendencies or location ratings)
  B  one-step  P(y | x, location ratings, tendencies)         same information, no stage
  C  two-step  sum_l P(l | x) P(y | l, x, team pass EPA at l)
  D  two-step  sum_l P(l | x) P(y | l, x)                     no location ratings
  O  oracle    P(y | actual l, x, ratings at l)
YAC is conditional on a catch, so its two-step weights are P(l | x, caught) proportional to P(l | x) P(catch | l, x).
"""
src = open("exp_split_features.py").read()
exec(src[:src.index("TASKS = {")])
# receiver location tendencies, strictly prior
rl = sc[sc.is_db & sc.pass_location.notna() & sc.receiver_player_id.notna()].copy()
rl["rec_mid"] = rl.pass_location.eq("middle").astype(float); rl["rec_left"] = rl.pass_location.eq("left").astype(float)
rt = rl.groupby(["receiver_player_id", "season", "week"]).agg(rec_mid=("rec_mid", "mean"), rec_left=("rec_left", "mean"), n=("rec_mid", "size")).reset_index()
rtend = player_prior(rt, "receiver_player_id", ["rec_mid", "rec_left"], "n", "rtend")
T = attach(th, rtend, "receiver_player_id").reset_index(drop=True)
T = T[T.pass_location.isin(["left", "middle", "right"])].reset_index(drop=True)
T["l"] = T.pass_location.map({"left": 0, "middle": 1, "right": 2}).astype(int)
tr = (T.season <= 2024).to_numpy(); te = ~tr
TEND = ["off_mid_share", "def_mid_share", "rtend_rec_mid", "rtend_rec_left"]
LOCR = team(["pass_left", "pass_middle", "pass_right"])
base = situation + team(base_m) + ["air_yards"] + QB + WR
def model(k):
    obj = dict(objective="binary") if k == 2 else dict(objective="multiclass", num_class=k)
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=300, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.8, verbose=-1, **obj)
def at_loc(frame, l):
    l = np.broadcast_to(np.asarray(l), (len(frame),))
    out = pd.DataFrame({"l": l}, index=frame.index)
    for side in ("off", "def"):
        out[f"{side}_at_loc"] = np.select([l == 0, l == 1, l == 2], [frame[f"{side}_pass_left"], frame[f"{side}_pass_middle"], frame[f"{side}_pass_right"]], np.nan)
    return out
Xl = base + TEND + LOCR
ml = model(3).fit(T.loc[tr, Xl], T.l[tr]); PL = ml.predict_proba(T.loc[te, Xl])
prior = np.bincount(T.l[tr], minlength=3) / tr.sum()
print(f"{te.sum()} held-out throws; location model log loss {-np.log(PL[np.arange(te.sum()), T.l[te]]).mean():.3f} vs league frequencies {-np.log(prior[T.l[te]]).mean():.3f}")
test = T.loc[te]
def stage_models(y, rows):
    """fit A, B, C-outcome, D-outcome on training rows (mask over T) for target y (array over T)"""
    trr = tr & rows; k = int(np.nanmax(y[rows])) + 1
    tm = T.loc[trr]
    A = model(k).fit(tm[base], y[trr]); B = model(k).fit(tm[base + TEND + LOCR], y[trr])
    C = model(k).fit(pd.concat([tm[base], at_loc(tm, tm.l.to_numpy())], axis=1), y[trr])
    Dm = model(k).fit(tm[base].assign(l=tm.l.to_numpy()), y[trr])
    return A, B, C, Dm
def probs_given_l(Cm, Dm, frame, l):
    return (Cm.predict_proba(pd.concat([frame[base], at_loc(frame, l)], axis=1)), Dm.predict_proba(frame[base].assign(l=l)))
def report(name, y, rows, weights_fn=None):
    A, B, Cm, Dm = stage_models(y, rows)
    ter = te & rows; f = T.loc[ter]; yt = y[ter].astype(int); idx = np.arange(len(f))
    pl = PL[rows[te]]
    if weights_fn is not None: pl = weights_fn(f, pl)
    mixC = sum(pl[:, [i]] * probs_given_l(Cm, Dm, f, i)[0] for i in range(3))
    mixD = sum(pl[:, [i]] * probs_given_l(Cm, Dm, f, i)[1] for i in range(3))
    res = {"A one-step": A.predict_proba(f[base]), "B one-step + location ratings, tendencies": B.predict_proba(f[base + TEND + LOCR]),
           "C two-step, drawn location + ratings at l": mixC, "D two-step, drawn location, no ratings": mixD,
           "O oracle, actual location": Cm.predict_proba(pd.concat([f[base], at_loc(f, f.l.to_numpy())], axis=1))}
    lls = {k: -np.log(np.clip(v[idx, yt], 1e-12, 1)) for k, v in res.items()}
    a = lls["A one-step"]; games = f.game_id.to_numpy()
    print(f"\n{name} ({len(f)} held-out rows)")
    for k, v in lls.items():
        d = v - a; g = pd.Series(d).groupby(games).sum(); z = d.mean() / (g.std() * np.sqrt(len(g)) / len(d)) if d.std() > 0 else 0
        print(f"  {k:45s} {v.mean():.4f}  {100 * d.mean() / a.mean():+6.2f}%  z {z:+5.1f}")
    return Cm, Dm
everything = np.ones(len(T), bool)
comp = T.complete_pass.astype(int).to_numpy()
Ccomp, Dcomp = report("Completion | throw", comp, everything)
report("Interception | throw", T.interception.astype(int).to_numpy(), everything)
caught = (T.complete_pass == 1).to_numpy() & T.yards_after_catch.notna().to_numpy()
yac = np.full(len(T), np.nan); yac[caught] = binned(T.yards_after_catch[caught], [1, 2, 3, 4, 6, 8, 11, 16, 25])
def given_catch(frame, pl):  # P(l | x, caught) proportional to P(l | x) P(catch | l, x), using the matching completion model
    w = np.column_stack([Ccomp.predict_proba(pd.concat([frame[base], at_loc(frame, i)], axis=1))[:, 1] for i in range(3)])
    post = pl * w; return post / post.sum(1, keepdims=True)
report("YAC | catch", yac, caught, given_catch)
