import os
PBP_DIR = os.environ.get("PBP_DIR", "data/pbp")  # folder of nflverse play_by_play_<season>.parquet files saved as pbp_<season>.parquet
import numpy as np, pandas as pd
D = PBP_DIR + "/"
p=pd.concat([pd.read_parquet(D+f"pbp_{y}.parquet",columns=["game_id","season","season_type","qtr","half_seconds_remaining","posteam_score_post","defteam_score_post","posteam_score","defteam_score","field_goal_attempt","fixed_drive_result","play_type","total_home_score","total_away_score"]) for y in range(2021,2027)])
p=p[p.season_type=="REG"]
p["pts"]=(p.total_home_score+p.total_away_score).groupby(p.game_id).diff().fillna(0).clip(lower=0)
ng=p.game_id.nunique()
eoh=(p.qtr==2)&(p.half_seconds_remaining<=120)
print("points per game, last 2:00 of 1st half:",round(p.loc[eoh,"pts"].sum()/ng,2),"of",round(p.pts.sum()/ng,2))
print("FG attempts per game in last 0:30 of 1st half:",round(((p.qtr==2)&(p.half_seconds_remaining<=30)&(p.field_goal_attempt==1)).sum()/ng,3))
