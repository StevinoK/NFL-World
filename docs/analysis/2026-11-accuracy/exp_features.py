import os
PBP_DIR = os.environ.get("PBP_DIR", "data/pbp")  # folder of nflverse play_by_play_<season>.parquet files saved as pbp_<season>.parquet
import numpy as np, pandas as pd, lightgbm as lgb, warnings
from sklearn.metrics import log_loss
warnings.filterwarnings("ignore")
D = PBP_DIR + "/"
cols=None
frames=[]
for y in range(2016,2027):
    d=pd.read_parquet(D+f"pbp_{y}.parquet"); frames.append(d)
p=pd.concat(frames,ignore_index=True)
p=p[p.season_type=="REG"].copy()
p["roof_c"]=p.roof.astype("category").cat.codes
p["home"]=(p.posteam==p.home_team).astype(float)
p["outdoor"]=p.roof.isin(["outdoors","open"]).astype(float)
p["wind_f"]=np.where(p.outdoor==1,p.wind,0.0); p["temp_f"]=np.where(p.outdoor==1,p.temp,70.0)
# pregame strength proxy: posteam-perspective spread
p["pos_spread"]=np.where(p.posteam==p.home_team,p.spread_line,-p.spread_line)
P=dict(objective="multiclass",learning_rate=0.05,num_leaves=31,min_child_samples=200,verbose=-1,n_estimators=400)
def fit_eval(name,tr,te,X,y,classes=None,weight=None,binary=False):
    params=dict(P); 
    if binary: params.update(objective="binary")
    else: params["num_class"]=len(classes)
    m=lgb.LGBMClassifier(**params)
    m.fit(tr[X],tr[y],sample_weight=weight)
    pr=m.predict_proba(te[X])
    ll=log_loss(te[y],pr,labels=m.classes_)
    return ll
res=[]
def cmp(task,tr,te,y,base,extras,binary=False,weights=None):
    classes=sorted(tr[y].unique())
    b=fit_eval("base",tr,te,base,y,classes,binary=binary)
    res.append((task,"base",b,0.0,len(te)))
    for nm,ex in extras.items():
        l=fit_eval(nm,tr,te,base+ex,y,classes,binary=binary)
        res.append((task,nm,l,100*(b-l)/b,len(te)))
    if weights:
        for nm,w in weights.items():
            l=fit_eval(nm,tr,te,base,y,classes,binary=binary,weight=w(tr))
            res.append((task,nm,l,100*(b-l)/b,len(te)))
TR=lambda d:d[(d.season>=2017)&(d.season<=2024)]; TE=lambda d:d[d.season>=2025]
gs=["down","ydstogo","yardline_100","game_seconds_remaining","score_differential","posteam_score","defteam_score","home","roof_c"]

# ---- 1. FG / kick make model
k=p[p.field_goal_attempt==1].copy().sort_values(["game_date","game_id","play_id"])
k["made"]=(k.field_goal_result=="made").astype(int)
# kicker prior: shrunk trailing (strictly prior) makes over expected, expected from distance logistic fit on all
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression().fit(np.c_[k.kick_distance,k.kick_distance**2],k.made)
k["xmade"]=lr.predict_proba(np.c_[k.kick_distance,k.kick_distance**2])[:,1]
k["resid"]=k.made-k.xmade
g=k.groupby("kicker_player_id")
k["k_prior_n"]=g.cumcount()
k["k_prior_sum"]=g.resid.cumsum()-k.resid
k["kicker_oe"]=k.k_prior_sum/(k.k_prior_n+40)
k["season_n"]=k.season.astype(float)
cmp("FG make (binary)",TR(k),TE(k),"made",gs,
    {"+weather":["wind_f","temp_f"],"+kicker":["kicker_oe"],"+weather+kicker":["wind_f","temp_f","kicker_oe"]},binary=True)

# ---- 2. play type (pass/run/punt/fg) on downs 1-4, excluding kneels/spikes
s=p[p.down.notna()&p.play_type.isin(["pass","run","punt","field_goal"])&(p.qb_kneel==0)&(p.qb_spike==0)].copy()
s["y"]=s.play_type.map({"pass":0,"run":1,"punt":2,"field_goal":3})
s["season_n"]=s.season.astype(float)
cmp("Play type (4-class)",TR(s),TE(s),"y",gs,
    {"+half_secs,qtr":["half_seconds_remaining","qtr"],
     "+timeouts":["posteam_timeouts_remaining","defteam_timeouts_remaining"],
     "+pregame spread":["pos_spread","total_line"],
     "+season trend":["season_n"],
     "+all four":["half_seconds_remaining","qtr","posteam_timeouts_remaining","defteam_timeouts_remaining","pos_spread","total_line","season_n"]},
    weights={"recency weight (0.8/season)":lambda d:0.8**(2024-d.season)})

# ---- 3. 4th down decision only
f=s[s.down==4].copy()
f["y"]=f.y.map({0:0,1:0,2:1,3:2})  # go / punt / fg
cmp("4th-down go/punt/FG",TR(f),TE(f),"y",gs,
    {"+timeouts,half_secs":["half_seconds_remaining","qtr","posteam_timeouts_remaining","defteam_timeouts_remaining"],
     "+pregame spread":["pos_spread","total_line"],
     "+season trend":["season_n"],
     "+all":["half_seconds_remaining","qtr","posteam_timeouts_remaining","defteam_timeouts_remaining","pos_spread","total_line","season_n"]},
    weights={"recency weight (0.8/season)":lambda d:0.8**(2024-d.season)})
# rule audit on test set: sim rule vs actual
t=TE(f); fgd=t.yardline_100+17
tl=(t.qtr>=4)&(t.game_seconds_remaining<=300)&(t.score_differential<=-4)
forced_fg=(fgd<=54)&(t.ydstogo>=2)&~tl
print("\n4th downs 2025-26:",len(t),"| rule forces FG on",forced_fg.mean().round(3),
      "| of those, actual: go",(t[forced_fg].y==0).mean().round(3),"punt",(t[forced_fg].y==1).mean().round(3),"fg",(t[forced_fg].y==2).mean().round(3))
forced_punt=(fgd>=62)&(t.yardline_100>30)&(t.ydstogo>1)&~tl
print("rule forces punt on",forced_punt.mean().round(3),"| actual go",(t[forced_punt].y==0).mean().round(3))

# ---- 4. clock runoff (time elapsed to next snap), same filter as training
p=p.sort_values(["game_id","play_id"])
p["elapsed"]=p.groupby("game_id").game_seconds_remaining.diff(-1)
c=p[p.play_type.isin(["pass","run","punt","field_goal","no_play"])&p.elapsed.between(1,120)&p.down.notna()].copy()
c["cat"]=np.select([c.incomplete_pass.eq(1),c.complete_pass.eq(1),c.rush_attempt.eq(1),c.sack.eq(1),c.punt_attempt.eq(1),c.interception.eq(1),c.field_goal_attempt.eq(1)],[0,1,2,3,4,5,6],7)
c["y"]=np.minimum((c.elapsed//5).astype(int),12)  # 5-second bins
c["oob"]=c.out_of_bounds.fillna(0)
cmp("Clock runoff (5s bins)",TR(c),TE(c),"y",gs+["cat"],
    {"+half_secs,qtr":["half_seconds_remaining","qtr"],
     "+timeouts":["posteam_timeouts_remaining","defteam_timeouts_remaining"],
     "+half_secs,qtr,timeouts":["half_seconds_remaining","qtr","posteam_timeouts_remaining","defteam_timeouts_remaining"],
     "+...+out of bounds":["half_seconds_remaining","qtr","posteam_timeouts_remaining","defteam_timeouts_remaining","oob","no_huddle"]})

r=pd.DataFrame(res,columns=["task","variant","test_logloss","pct_better_than_base","n_test"])
print(r.round(4).to_string(index=False)); r.to_csv("feature_experiments.csv",index=False)
