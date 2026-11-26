"""The article's stability checks on raw (unadjusted) team-game values: game vs next game, 3 games vs next 3,
season vs next season, and each metric's prediction of next-period offensive points. Offense side."""
import os, numpy as np, pandas as pd
D = os.environ.get("PBP_DIR", "data/pbp")
cols = ["game_id", "season", "week", "season_type", "game_date", "posteam", "play_type", "epa", "success", "qb_kneel", "qb_spike",
        "fixed_drive", "fixed_drive_result"]
p = pd.concat([pd.read_parquet(f"{D}/pbp_{y}.parquet", columns=cols) for y in range(2017, 2027)])
p = p[(p.season_type == "REG") & p.posteam.notna()]
sc = p[p.play_type.isin(["pass", "run"]) & (p.qb_kneel == 0) & (p.qb_spike == 0) & p.epa.notna()]
t = sc.groupby(["game_id", "posteam"]).agg(season=("season", "first"), date=("game_date", "first"), epa=("epa", "mean"), success=("success", "mean")).reset_index()
dr = p[p.fixed_drive.notna()].groupby(["game_id", "posteam", "fixed_drive"]).fixed_drive_result.first().reset_index()
dr["pts"] = dr.fixed_drive_result.map({"Touchdown": 7.0, "Field goal": 3.0}).fillna(0.0)
t = t.merge(dr.groupby(["game_id", "posteam"]).pts.mean().rename("ppd").reset_index(), on=["game_id", "posteam"])
t = t.merge(pd.read_parquet("edp_team_games.parquet"), on=["game_id", "posteam"]).sort_values(["posteam", "date"])
metrics = ["epa", "success", "ppd", "fk_edp", "fk_gain", "fk_avg_rel", "fk_ypp", "fk_start"]
rows = []
for m in metrics:
    g = t.groupby(["posteam", "season"])
    nxt, nxt_pts = g[m].shift(-1), g.ppd.shift(-1)
    r3 = g[m].transform(lambda s: s.rolling(3).mean()); f3 = g[m].transform(lambda s: s[::-1].rolling(3).mean()[::-1].shift(-1))
    f3p = g.ppd.transform(lambda s: s[::-1].rolling(3).mean()[::-1].shift(-1))
    ok1, ok3 = nxt.notna(), r3.notna() & f3.notna()
    seas = t.groupby(["posteam", "season"])[list(dict.fromkeys([m, "ppd"]))].mean().reset_index().sort_values(["posteam", "season"])
    seas["next"], seas["next_ppd"] = seas.groupby("posteam")[m].shift(-1), seas.groupby("posteam").ppd.shift(-1)
    sv = seas.dropna()
    rows.append((m, np.corrcoef(t[m][ok1], nxt[ok1])[0, 1], np.corrcoef(t[m][ok1], nxt_pts[ok1])[0, 1],
                 np.corrcoef(r3[ok3], f3[ok3])[0, 1], np.corrcoef(r3[ok3], f3p[ok3])[0, 1],
                 np.corrcoef(sv[m], sv.next)[0, 1], np.corrcoef(sv[m], sv.next_ppd)[0, 1]))
r = pd.DataFrame(rows, columns=["metric", "game->next game", "game->next PPD", "3g->next 3g", "3g->next 3g PPD", "season->next", "season->next PPD"])
print(r.round(3).to_string(index=False)); r.to_csv("edp_stability.csv", index=False)
