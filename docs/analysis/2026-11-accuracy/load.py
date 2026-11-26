import os
PBP_DIR = os.environ.get("PBP_DIR", "data/pbp")  # folder of nflverse play_by_play_<season>.parquet files saved as pbp_<season>.parquet
import pandas as pd, glob
COLS = ["game_id","season","week","season_type","posteam","defteam","home_team","away_team","play_type","down","ydstogo",
 "yardline_100","qtr","game_seconds_remaining","half_seconds_remaining","score_differential","posteam_score","defteam_score",
 "posteam_timeouts_remaining","defteam_timeouts_remaining","field_goal_attempt","field_goal_result","kick_distance","punt_attempt",
 "two_point_attempt","two_point_conv_result","extra_point_attempt","extra_point_result","safety","touchback","kickoff_attempt",
 "own_kick_recovery","return_touchdown","home_score","away_score","result","total","spread_line","total_line","pass","rush",
 "qb_kneel","qb_spike","no_huddle","shotgun","timeout","epa","wp","vegas_wp","posteam_type","roof","penalty","fourth_down_converted",
 "fourth_down_failed","play_clock","drive","fixed_drive","fixed_drive_result","touchdown","td_team","sack","qb_scramble","desc"]
def load(years=range(2016,2027)):
    out=[]
    for y in years:
        df=pd.read_parquet(f"{PBP_DIR}/pbp_{y}.parquet")
        out.append(df[[c for c in COLS if c in df.columns]])
    return pd.concat(out, ignore_index=True)
