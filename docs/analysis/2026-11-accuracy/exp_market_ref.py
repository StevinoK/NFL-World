"""(1) Do earlier weeks' closing lines carry team-strength information that results alone miss?
(2) Do referee crews predict a game's penalties beyond the league average?"""
import os, numpy as np, pandas as pd
D = os.environ.get("PBP_DIR", "data/pbp")
g = pd.read_csv(f"{D}/games.csv")
g = g[(g.game_type == "REG") & (g.season >= 2010) & g.result.notna() & g.spread_line.notna()].copy()
g = g.sort_values(["gameday", "game_id"]).reset_index(drop=True)
g["wk"] = g.season * 100 + g.week
teams = sorted(set(g.home_team) | set(g.away_team)); idx = {t: i for i, t in enumerate(teams)}
neutral = (g.location == "Neutral").to_numpy()
def ratings(past, target, wk, lam, cross, decay=0.95):
    s_now, w_now = wk // 100, wk % 100
    w = np.where(past.season == s_now, decay ** (w_now - past.week), decay ** (w_now - 1) * cross * cross ** (s_now - past.season - 1) * (0.97 if decay == 0.95 else decay) ** (18 - past.week))
    X = np.zeros((len(past), len(teams) + 1)); r = np.arange(len(past))
    X[r, [idx[t] for t in past.home_team]] = 1; X[r, [idx[t] for t in past.away_team]] = -1; X[:, -1] = (past.location != "Neutral")
    W = w[:, None] * X; P = np.eye(len(teams) + 1) * lam; P[-1, -1] = 0.1 if lam > 1 else 0.01
    return np.linalg.solve(X.T @ W + P, W.T @ past[target].to_numpy())
res_m, res_l = np.full(len(g), np.nan), np.full(len(g), np.nan)
for wk in sorted(g.wk.unique()):
    if wk < 201201: continue
    past, now = g[(g.wk < wk) & (g.wk >= wk - 300)], g.index[g.wk == wk]
    bm = ratings(past, "result", wk, 8, 0.6); bl = ratings(past, "spread_line", wk, 0.05, 0.8, decay=0.7)
    for b, out in ((bm, res_m), (bl, res_l)):
        out[now] = [b[idx[h]] - b[idx[a]] + b[-1] * (not neutral[i]) for i, h, a in zip(now, g.home_team[now], g.away_team[now])]
g["from_results"], g["from_past_lines"] = res_m, res_l
t = g[(g.season >= 2016) & g.from_results.notna()]
def loso(cols):
    e = []
    for s in sorted(t.season.unique()):
        tr, te = t[t.season != s], t[t.season == s]
        A = np.c_[np.ones(len(tr)), tr[cols]]; b = np.linalg.lstsq(A, tr.result, rcond=None)[0]
        e.append(te.result - np.c_[np.ones(len(te)), te[cols]] @ b)
    return np.sqrt(np.mean(np.concatenate(e) ** 2))
early = t.week <= 4
print("Margin RMSE, 2016-2025 leave-one-season-out (all weeks / weeks 1-4 / weeks 5+):")
for name, cols in [("rating from past results", ["from_results"]), ("rating from past closing lines", ["from_past_lines"]),
                   ("both", ["from_results", "from_past_lines"]), ("this game's closing line", ["spread_line"])]:
    full = loso(cols)
    sub = lambda m: np.sqrt(np.mean([((tt.result - np.polyval(np.polyfit(t.loc[~m][cols[0]], t.loc[~m].result, 1), tt[cols[0]])) ** 2).mean() for tt in [t[m]]]))
    print(f"  {name:32s} {full:6.2f}   {sub(early):6.2f}   {sub(~early):6.2f}")
# (2) referees and penalties
pen = []
for y in range(2016, 2027):
    p = pd.read_parquet(f"{D}/pbp_{y}.parquet", columns=["game_id", "penalty", "season_type"])
    p = p[p.season_type == "REG"]; pen.append(p.groupby("game_id").penalty.sum())
pen = pd.concat(pen).rename("penalties")
r = pd.read_csv(f"{D}/games.csv")[["game_id", "gameday", "season", "referee"]].merge(pen, on="game_id").dropna(subset=["referee"])
r = r.sort_values(["gameday", "game_id"]).reset_index(drop=True)
r["league_prior"] = r.groupby("season").penalties.transform(lambda s: s.expanding().mean().shift()).fillna(r.penalties.mean())
r["resid"] = r.penalties - r.league_prior
gr = r.groupby("referee"); n = gr.cumcount(); cs = gr.resid.cumsum() - r.resid
r["ref_effect"] = cs / (n + 16)  # shrunk toward 0 with 16 games of prior weight
tt = r[r.season >= 2018]
base = ((tt.penalties - tt.league_prior) ** 2).mean(); wref = ((tt.penalties - tt.league_prior - tt.ref_effect) ** 2).mean()
print(f"\nReferee crews, {len(tt)} games 2018-26: penalties/game SD {tt.penalties.std():.2f}; "
      f"MSE league-average {base:.2f} -> with crew effect {wref:.2f} ({100*(base-wref)/base:.1f}% better); "
      f"SD of crew effects {tt.groupby('referee').ref_effect.last().std():.2f} penalties/game")
