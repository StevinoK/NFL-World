import os
PBP_DIR = os.environ.get("PBP_DIR", "data/pbp")  # folder of nflverse play_by_play_<season>.parquet files saved as pbp_<season>.parquet
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
D = PBP_DIR + "/"
p=pd.concat([pd.read_parquet(D+f"pbp_{y}.parquet") for y in range(2016,2027)],ignore_index=True)
p=p[p.season_type=="REG"]
k=p[(p.field_goal_attempt==1)|(p.extra_point_attempt==1)].copy().sort_values(["game_date","game_id","play_id"])
k["made"]=np.where(k.field_goal_attempt==1,k.field_goal_result=="made",k.extra_point_result=="good").astype(int)
k["dist"]=k.kick_distance.fillna(k.yardline_100+18)
k["xp"]=k.extra_point_attempt.fillna(0)
out=k.roof.isin(["outdoors","open"])
k["wind_f"]=np.where(out,k.wind.fillna(8),0.0); k["cold"]=np.where(out,np.clip(50-k.temp.fillna(60),0,None),0.0)
k["season_n"]=k.season-2020
def basis(d): x=d.dist/10; return np.c_[x,x**2,x**3,d.xp]
lr=LogisticRegression(C=10,max_iter=2000).fit(basis(k[k.season<=2024]),k[k.season<=2024].made)
k["xlogit"]=lr.decision_function(basis(k)); k["resid"]=k.made-1/(1+np.exp(-k.xlogit))
g=k.groupby("kicker_player_id"); n=g.cumcount(); cs=g.resid.cumsum()-k.resid
for prior in [50,100,200]:
    k[f"kicker_oe_{prior}"]=cs/(n+prior)
tr=k[(k.season>=2017)&(k.season<=2024)]; te=k[k.season>=2025]
def run(feats):
    X=lambda d:np.c_[basis(d),d[feats].to_numpy()] if feats else basis(d)
    m=LogisticRegression(C=1.0,max_iter=3000).fit(X(tr),tr.made)
    pr=m.predict_proba(X(te))[:,1]
    fg=te.field_goal_attempt==1
    return log_loss(te.made,pr), log_loss(te.made[fg],pr[fg]), log_loss(te.made[~fg],pr[~fg])
rows=[]
base=run([])
for nm,f in {"base (distance, PAT flag)":[],"+wind,cold":["wind_f","cold"],"+kicker (prior 100)":["kicker_oe_100"],
             "+season trend":["season_n"],"+wind,cold,kicker":["wind_f","cold","kicker_oe_100"],
             "+wind,cold,kicker,season":["wind_f","cold","kicker_oe_100","season_n"]}.items():
    a,b,c=run(f); rows.append((nm,a,100*(base[0]-a)/base[0],b,100*(base[1]-b)/base[1],c))
r=pd.DataFrame(rows,columns=["variant","all_ll","pct_all","fg_ll","pct_fg","pat_ll"])
print("n test kicks",len(te),"FG",int((te.field_goal_attempt==1).sum()))
print(r.round(4).to_string(index=False))
# sanity: spread of kicker_oe among kickers with 100+ attempts
last=k[k.season>=2025].groupby("kicker_player_id").kicker_oe_100.last()
print("kicker_oe SD among 2025-26 kickers:",last.std().round(4))
r.to_csv("fg_experiment.csv",index=False)
