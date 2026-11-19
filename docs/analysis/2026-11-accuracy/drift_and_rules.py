import os
PBP_DIR = os.environ.get("PBP_DIR", "data/pbp")  # folder of nflverse play_by_play_<season>.parquet files saved as pbp_<season>.parquet
import numpy as np, pandas as pd
from load import load
p = load()
p = p[p.season_type=="REG"]
g = p.groupby("game_id").agg(season=("season","first"),home=("home_score","last"),away=("away_score","last"),
      spread=("spread_line","first"),tot=("total_line","first")).dropna()
g["pts"]=g.home+g.away; g["margin"]=g.home-g.away
ngames = g.groupby("season").size()
sc = p[(p.down.notna())&(p.play_type.isin(["pass","run"]))&(p.qb_kneel==0)&(p.qb_spike==0)]
early = sc[(sc.down<=2)&(sc.wp.between(.2,.8))&(sc.half_seconds_remaining>120)]
fg = p[p.field_goal_attempt==1]
xp = p[p.extra_point_attempt==1]
ko = p[p.kickoff_attempt==1]
fd = p[(p.down==4)]
rows=[]
for s,gg in g.groupby("season"):
    n=len(gg); ps=p[p.season==s]
    f4=fd[fd.season==s]
    # Sim's hard-coded rule: FG if (yardline_100+17)<=54 & ydstogo>=2 & not trailing late
    fgdist=f4.yardline_100+17
    trailing_late=(f4.qtr>=4)&(f4.game_seconds_remaining<=300)&(f4.score_differential<=-4)
    rule_kick=(fgdist<=54)&(f4.ydstogo>=2)&~trailing_late
    m=f4[rule_kick & f4.play_type.isin(["pass","run","field_goal","punt"])]
    went = m.play_type.isin(["pass","run"]).mean()
    rows.append(dict(season=s, games=n, pts_pg=gg.pts.mean(), team_pts_sd=np.std(np.r_[gg.home,gg.away]),
        home_away_corr=np.corrcoef(gg.home,gg.away)[0,1], margin_sd=gg.margin.std(),
        spread_err_sd=(gg.margin-gg.spread).std(), total_err_sd=(gg.pts-gg.tot).std(),
        early_pass_rate=early[early.season==s]["pass"].mean(),
        epa_play=sc[sc.season==s].epa.mean(),
        fg_pct_50p=(fg[(fg.season==s)&(fg.kick_distance>=50)].field_goal_result=="made").mean(),
        fg_att_50p_pg=((fg.season==s)&(fg.kick_distance>=50)).sum()/n/2,
        xp_pct=(xp[xp.season==s].extra_point_result=="good").mean(),
        twopt_att_pg=(ps.two_point_attempt==1).sum()/n,
        twopt_succ=(ps[ps.two_point_attempt==1].two_point_conv_result=="success").mean(),
        safeties_pg=(ps.safety==1).sum()/n,
        onside_pg=((ps.kickoff_attempt==1)&ps.desc.str.contains("onside",case=False,na=False)).sum()/n,
        ko_touchback=ko[ko.season==s].touchback.mean(),
        go_rate_in_rule_kick_zone=went, rule_zone_n=len(m),
        timeouts_pg=(ps.timeout==1).sum()/n,
        ))
r=pd.DataFrame(rows).set_index("season")
pd.set_option("display.width",250); pd.set_option("display.max_columns",40)
print(r.round(3).T.to_string())
r.to_csv("season_env.csv")
