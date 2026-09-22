import pandas as pd
import numpy as np
import nfl_data_py as nfl
import matplotlib.pyplot as plt
import plotly.express as px
import requests
from scipy.stats import beta
import statsmodels.api as sm
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import random
import random
import joblib
import time
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import traceback
#print(sys.executable)
#converter = trt.TrtGraphConverterV2(input_saved_model_dir='C:/Users/steve/OneDrive/Desktop/play_model.keras')
#funmodel = load_model('targeted_receiver_model.keras')
#funmodel.predict(X_test_processed)
#preprocessor = joblib.load('targeted_receiver_preprocessor.pkl')
class GameModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GameModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.drop1 = nn.Dropout(0.3)  # Match dropout rate
        self.fc2 = nn.Linear(256, 128) # Corrected input and output dimensions
        self.drop2 = nn.Dropout(0.3)  # Match dropout rate
        self.fc3 = nn.Linear(128, 64) # Corrected input and output dimensions
        self.drop3 = nn.Dropout(0.3)  # Match dropout rate
        self.out = nn.Linear(64, output_dim) # Added a final layer

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=0.1)
        x = self.drop1(x)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.1)
        x = self.drop2(x)
        x = F.leaky_relu(self.fc3(x), negative_slope=0.1)
        x = self.drop3(x)
        return self.out(x)
class CornModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CornModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.drop1 = nn.Dropout(0.3)  # Match dropout rate
        self.fc2 = nn.Linear(256, 128) # Corrected input and output dimensions
        self.drop2 = nn.Dropout(0.3)  # Match dropout rate
        self.fc3 = nn.Linear(128, 64) # Corrected input and output dimensions
        self.drop3 = nn.Dropout(0.3)  # Match dropout rate
        self.out = nn.Linear(64, output_dim-1) # Added a final layer

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=0.1)
        x = self.drop1(x)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.1)
        x = self.drop2(x)
        x = F.leaky_relu(self.fc3(x), negative_slope=0.1)
        x = self.drop3(x)
        return self.out(x)

def get_probs(y_pred_logits):
    #if y_pred_logits[0]==0:
    #print(y_pred_logits)
    probas = torch.sigmoid(torch.from_numpy(y_pred_logits))
    #print(probas)
    probas = torch.cumprod(probas, dim=1).float().detach().numpy()
    #print(probas)
    num_classes = probas.shape[1] + 1

    # Create a list of tensors.
    probs = []

    # First, get probability predictions out of the cumulative logits.
    # Column 0 is Probability that y > 0, so Pr(y = 0) = 1 - Pr(y > 0)
    # Pr(Y = 0) = 1 - s(logit for column 0)
    probs.append(1.0 - probas[:, 0])

    # For the other columns, the probability is:
    # Pr(y = k) = Pr(y > k) - Pr(y > k - 1)
    if num_classes > 2:
        for val in range(1, num_classes - 1):
            probs.append(probas[:, val - 1] - probas[:, val])

    # Special handling of the maximum label value.
    probs.append(probas[:, num_classes - 2])
    probs = np.array(probs).T
    #print(probs)
    return probs
# Initialize model
#%%Load Models
schedules = nfl.import_schedules([i for i in range(2016,2025)])
passer_columns = ["cpoe_bayes", "turnover_bayes","sack_bayes", "cpoe_short", "cpoe_medium", "cpoe_deep", "deep_pass_rate","medium_pass_rate", "short_pass_rate"]
def_cols = ['deep_cpoe_allowed','int_rate','medium_cpoe_allowed','short_cpoe_allowed','yac_oe_allowed','sack_rate','rush_yards_per_carry','def_rush_epa_rate',
    'def_pass_epa_rate', 'def_winning_rate','def_losing_rate','def_tied_rate']
team_stats_cols = ['losing_rate','tied_rate','team_4th_rate','winning_rate','rushing_epa_rate', 'epa_rate', 'opp_def_pass_epa_rate', 'opp_def_rush_epa_rate', 'opp_pass_epa_rate','opp_rushing_epa_rate']
team_stats_columns = team_stats_cols.copy()
#def_cols = ['deep_cpoe_allowed','int_rate','medium_cpoe_allowed','short_cpoe_allowed','yac_oe_allowed','sack_rate','rush_yards_per_carry']
game_state_cols = ["defteam_score","down","game_seconds_remaining","posteam_score","posteam_type","yardline_100","ydstogo","roof","score_differential"]
pass_columns = ['air_yards']
rusher_columns = ['close_bayes','explosive_bayes','ypc_bayes','rush_rate', 'scramble_rate']
position_cols = ['position']
receiver_columns = ['deep','medium','short','fd_bayes','rz_bayes','yac_bayes','adot','ypt']
target_embed_columns = [f'target_embed_{str(i)}' for i in range(1,3)]
running_embed_columns = [f'rb_target_embed_{str(i)}' for i in range(1,3)]

#team_stats_cols = ['losing_rate','tied_rate','team_4th_rate','winning_rate']
play_cols = ['play_type_cat']
play_outcome = ['play_outcome']
models = ['kick','ay','target','comp','int','yac','sack','rush','punt','time','rusher','play', 'fumble', 'def_ret']
model_dict = {}
column_dict = {}
bin_dict = {}
processor = {}
for model in models:
    bins = 0
    try:
        bin_dict[model] = pd.read_csv(f'C:/Users/steve/OneDrive/Desktop/{model}_bins.csv')
        bins = 1
    except:
        bins = 0
        pass
    try:
        column_dict[model] = joblib.load(f'C:/Users/steve/OneDrive/Desktop/{model}_columns.pkl')
    except:
        pass
    try:
        processor[model] = joblib.load(f'C:/Users/steve/OneDrive/Desktop/{model}_preprocessor.pkl')
    except:
        pass
    try:
        if model in ['target','rusher']:
            inp = 0
            for col in column_dict[model]:
                if col in ['receiver_columns','rusher_columns']:
                    inp += len(eval(col))*11
                elif col == 'game_state_cols':
                    inp += len(eval(col))+4
                else:
                    inp += len(eval(col))
            output = 11
        elif model == 'time':
            inp = 0
            for col in column_dict[model]:
                if col == 'play_cols':
                    inp += len(eval(col))*8
                elif col == 'game_state_cols':
                    inp += len(eval(col))+4
                else:
                    inp += len(eval(col))
            output = len(bin_dict[model])
        elif model == 'fumble':
            inp = 0
            for col in column_dict[model]:
                if col == 'play_outcome':
                    inp += len(eval(col))*3
                elif col == 'game_state_cols':
                    inp += len(eval(col))+4
                else:
                    inp += len(eval(col))
            output = 2

        else:
            inp = 0
            for col in column_dict[model]:
                if col == 'game_state_cols':
                    inp += len(eval(col))+4
                elif col == 'position_cols':
                    inp += len(eval(col))+4
                else:
                    inp += len(eval(col))
            output = 2
            try:
                output = len(bin_dict[model])
            except:
                pass
        if model == 'play':
            output = 4
        if bins == 1:
            model_dict[model] = CornModel(inp,output)            
        else:
            model_dict[model] = GameModel(inp,output)
        model_dict[model].load_state_dict(torch.load(f'C:/Users/steve/OneDrive/Desktop/{model}_model.pt'))
        model_dict[model].eval()
    except:
        pass
#%%
player_stats_wide = pd.read_csv("C:/Users/steve/OneDrive/Desktop/player_stats_wide.csv")
team_stats_wide = pd.read_csv("C:/Users/steve/OneDrive/Desktop/team_stats_wide.csv")
depth_chart = pd.read_csv('C:/Users/steve/OneDrive/Desktop/depth_charts.csv', index_col = 0)
depth_chart['season'] = depth_chart['game_id'].str.split('_').str[0]
target_means = np.array(pd.read_csv('C:/Users/steve/OneDrive/Desktop/wr_means.csv'))
target_covs = np.array(pd.read_csv('C:/Users/steve/OneDrive/Desktop/wr_covs.csv'))
rb_means = np.array(pd.read_csv('C:/Users/steve/OneDrive/Desktop/rb_means.csv'))
rb_covs = np.array(pd.read_csv('C:/Users/steve/OneDrive/Desktop/rb_covs.csv'))
#%%
#game = depth_chart['game_id'].sample().reset_index().values[0,1] #'2022_02_MIA_BAL'
#depth_chart = depth_chart[depth_chart['game_id']==game]
#indices = []
# for t in depth_chart['team']:
#     for col in ['QB1','WR1','WR2','WR3','WR4','RB1','RB2','RB3','TE1','TE2','TE3']:
#         indices.append(depth_chart.loc[depth_chart['team']==t,col].values[0])
# player_stats_wide = player_stats_wide[player_stats_wide['player_id'].isin(indices)]
#%%
# Create a dictionary to collect filtered rows
# closest_prior_rows = []
# depth_chart['game_date'] = pd.to_datetime(depth_chart['game_date'])
# player_stats_wide['game_date'] = pd.to_datetime(player_stats_wide['game_date'])
# target_date = depth_chart.loc[depth_chart.index[0],'game_date']
# player_stats_wide['diff'] = target_date-player_stats_wide['game_date']
# filtered = player_stats_wide[player_stats_wide['diff'] < pd.Timedelta(0)]

# # Get the latest game_date per player_id (i.e., max game_date where game_date < target_date)
# temp_player = (
#     filtered
#     .sort_values(['player_id', 'game_date'],ascending=True)
#     .groupby('player_id', as_index=False)
#     .first()
# )
# temp_game = team_stats_wide[team_stats_wide['next_game_id']==game]
#%%
class Player:
    def __init__(self, name, position, player_id, attributes):
        self.name = name
        self.position = position
        self.player_id = player_id
        self.stats = {
            'passing_yards': 0,
            'rushing_yards': 0,
            'rushing_attempts': 0,
            'receiving_yards': 0,
            'passing_touchdowns': 0,
            'receiving_touchdowns': 0,
            'rushing_touchdowns': 0,
            'interceptions': 0,
            'receptions': 0,
            'sacks': 0,
            'punts': 0,
            'punt_yards': 0,
            'pat_attempts': 0,
            'pat_made': 0,
            'pass_attempts': 0,
            'completions':0,
            'air_yards':0,
            'fumble_lost':0
        }
        self.attributes = attributes

#%%
class Team:
    def __init__(self, name, players, home, attributes):
        self.name = name
        self.players = players  # list of Player objects
        self.score = 0
        self.home = home
        self.attributes = attributes
        self.rb_embed = np.random.multivariate_normal(rb_means[:,0], rb_covs[:,:])
        self.wr_embed = np.random.multivariate_normal(target_means[:,0], target_covs[:,:])
        self.stats = {
            'sack': 0,
            'pat_made': 0,
            'kick_made': 0,
            'fg 0-39': 0,
            'fg 40-49': 0,
            'fg 50-59': 0,
            'interception':0,
            'pa':0,
            'fumble_recovered':0,
            'defensive_touchdown':0}
    def get_random_player(self, position):
        filtered = [p for p in self.players if position in p.position]
        return random.choice(filtered) if filtered else None
#%%
class Play:
    def __init__(self, down, distance, yardline, game):
        self.down = down
        self.distance = distance
        self.yardline = yardline
        self.description = ""
        self.yards_gained = 0
        self.turnover = False
        self.defensive_touchdown = False
        self.touchdown = False
        self.pat_attempt = False
        self.punt = False
        self.game = game
        self.field_goal_attempt = False
        self.type = "other"
    def simulate(self, offense, defense, is_pat=False):
        defense.stats['pa'] = offense.score-7*defense.stats['defensive_touchdown']
        offense.stats['pa'] = defense.score-7*offense.stats['defensive_touchdown']
        gc = [defense.score, self.game.down, 3600-(self.game.quarter-1)*900 - (900 - self.game.clock), offense.score, offense.home, 100-self.game.yardline, self.game.distance, self.game.dome,  offense.score-defense.score]
        dc = [defense.attributes[col] for col in def_cols]
        tc = [offense.attributes[col] for col in team_stats_cols]
        #print(offense.name, tc)
        play_pred_columns = gc.copy()
        play_pred_columns.extend(tc.copy())
        play_pred_columns.extend(dc.copy())
        cols = []
        for collist in column_dict['play']:
            cols.extend(eval(collist))
        play = pd.DataFrame(np.array(play_pred_columns).reshape(1,-1), columns = cols)
        play_inp = processor['play'].transform(play)
        play_inp[np.isnan(play_inp)]=0
        #play_selection = model_dict['play'].predict(tf.data.Dataset.from_tensor_slices(play_inp).batch(1), verbose=0)
        t1 = time.time()
        play_selection_test = F.softmax(model_dict['play'](torch.as_tensor(play_inp.astype(np.float32))), dim=1).detach().numpy()
        #print(play_selection_test)
        t2 = time.time()
        #play_selection = np.random.choice(['kick','pass','punt','run'],p=play_selection[0,:])
        play_selection_2 = np.random.choice(['kick','pass','punt','run'],p=play_selection_test[0,:])
        play_type = play_selection_2

        #print(play_selection_2)
        if is_pat or play_type == 'kick':
            if is_pat:
                gc = [defense.score, 1, 3600-(self.game.quarter-1)*900 - (900 - self.game.clock), offense.score, offense.home, 15, 10, self.game.dome,  offense.score-defense.score]
            kick_columns = gc.copy()
            cols = []
            for collist in column_dict['kick']:
                cols.extend(eval(collist))
            kick = pd.DataFrame(np.array(kick_columns).reshape(1,-1), columns = cols)
            kick_inp = processor['kick'].transform(kick)
            kick_inp[np.isnan(kick_inp)]=0
            kick = F.softmax(model_dict['kick'](torch.as_tensor(kick_inp.astype(np.float32))), dim=1).detach().numpy()[0][1]
            good = random.random() < kick
            if is_pat:
                if good:
                    offense.stats['pat_made'] += 1
                    offense.score += 1
                    self.description = f"{offense.name} made the PAT"
                else:
                    self.description = f"{offense.name} missed the PAT"
                self.pat_attempt = True
            else:
                self.type = 'field_goal_attempt'
                if good:
                    offense.stats['kick_made'] += 1
                    if self.game.yardline > 79:
                        offense.stats['fg 0-39'] += 1
                    elif self.game.yardline > 69:
                        offense.stats['fg 40-49'] += 1
                    else:
                        offense.stats['fg 50-59'] += 1
                    offense.score += 3
                    self.description = f"{offense.name} made the kick"
                else:
                    self.description = f"{offense.name} missed the kick"
                    self.yards_gained -= 8
                self.field_goal_attempt = True
            return


        if play_type == 'pass':
            qb = offense.get_random_player('QB')
            #print(qb.name, offense.name)
            passer_c = [qb.attributes[col].values[0] for col in passer_columns]
            ay_pred_columns = gc.copy()
            ay_pred_columns.extend(dc.copy())
            ay_pred_columns.extend(passer_c.copy())
            ay_pred_columns.extend(tc.copy())
            cols = []
            for collist in column_dict['ay']:
                cols.extend(eval(collist))
            ay = pd.DataFrame(np.array(ay_pred_columns).reshape(1,-1), columns = cols)
            ay_inp = processor['ay'].transform(ay)
            ay_inp[np.isnan(ay_inp)]=0
            air_yards = get_probs(model_dict['ay'](torch.as_tensor(ay_inp.astype(np.float32))).detach().numpy())
            air_yards = min(np.random.choice(bin_dict['ay']['0'],p=air_yards[0,:]),100-self.yardline)
            
            
            receiver_selection_columns = gc.copy()
            receiver_selection_columns.append(air_yards)
            receiver_selection_columns.extend(list(offense.wr_embed))
            all_positions = ['QB1','WR1','WR2','WR3','WR4','RB1','RB2','RB3','TE1','TE2','TE3']
            for player in offense.players:
                receiver_selection_columns.extend([player.attributes[col].values[0] for col in receiver_columns])
            cols = []
            for collist in column_dict['target']:
                if collist == 'receiver_columns':
                    continue
                cols.extend(eval(collist))
            filled_positions = []
            for player in offense.players:
                l = eval('receiver_columns')
                cols.extend([ind+"_"+player.position  for ind in l])
                filled_positions.append(player.position)
            for player in [pos for pos in all_positions if pos not in filled_positions]:
                l = eval('receiver_columns')
                cols.extend([ind+"_"+player  for ind in l])
                filled_positions.append(player)
                for _ in range(len(l)):
                    receiver_selection_columns.append(np.nan)
            target = pd.DataFrame(np.array(receiver_selection_columns).reshape(1,-1), columns = cols)
            target = processor['target'].transform(target)
            target[np.isnan(target)]=0
            target = F.softmax(model_dict['target'](torch.as_tensor(target.astype(np.float32))), dim=1).detach().numpy()
            wr = None
            selecting = True
            while selecting:
                targ = np.random.choice(all_positions,p=target[0,:])
                for player in offense.players:
                    if player.position == targ:
                        wr = player
                        position = player.position[0:2]
                        selecting = False
                        break
            receiver_c = [wr.attributes[col].values[0] for col in receiver_columns]

            
            sack_pred_columns = gc.copy()
            sack_pred_columns.extend(dc.copy())
            sack_pred_columns.extend(passer_c.copy())
            sack_pred_columns.extend(tc.copy())
            cols = []
            for collist in column_dict['sack']:
                cols.extend(eval(collist))
            sack = pd.DataFrame(np.array(sack_pred_columns).reshape(1,-1), columns = cols)
            sack_inp = processor['sack'].transform(sack)
            sack_inp[np.isnan(sack_inp)]=0
            sack = F.softmax(model_dict['sack'](torch.as_tensor(sack_inp.astype(np.float32))), dim=1).detach().numpy()[0][1]
            comp_columns = gc.copy()
            comp_columns.extend(dc.copy())
            comp_columns.extend(passer_c.copy())
            comp_columns.append(air_yards)
            comp_columns.extend(tc.copy())
            cols = []
            for collist in column_dict['comp']:
                cols.extend(eval(collist))
            comp = pd.DataFrame(np.array(comp_columns).reshape(1,-1), columns = cols)
            comp_inp = processor['comp'].transform(comp)
            comp_inp[np.isnan(comp_inp)]=0
            comp = F.softmax(model_dict['comp'](torch.as_tensor(comp_inp.astype(np.float32))), dim=1).detach().numpy()[0][1]
            int_prob = pd.DataFrame(np.array(comp_columns).reshape(1,-1), columns = cols)
            int_inp = processor['int'].transform(int_prob)
            int_inp[np.isnan(int_inp)]=0
            int_prob = F.softmax(model_dict['int'](torch.as_tensor(int_inp.astype(np.float32))), dim=1).detach().numpy()[0][1]
            inc = max(1-sack-comp-int_prob,np.array(0))
            if inc == 0:
                inc =inc

            probs = np.array([sack, comp, int_prob, inc])           
            probs = probs/probs.sum()
            outcome = np.random.choice([0,1,2,3],p=probs[:])
            sack, complete, interception, incomplete = [0,0,0,0]
            if outcome == 0:
                sack = 1
            elif outcome == 1:
                complete = 1
            elif outcome == 2:
                interception = 1
            else:
                incomplete = 1

            if complete:
                #print("complete")
                self.type = 'complete_pass'
                comp_columns = gc.copy()
                comp_columns.extend(dc.copy())
                comp_columns.extend(receiver_c.copy())
                comp_columns.append(air_yards)
                comp_columns.extend(tc.copy())
                comp_columns.append(position)
                cols = []
                for collist in column_dict['yac']:
                    cols.extend(eval(collist))
                yac_prob = pd.DataFrame(np.array(comp_columns).reshape(1,-1), columns = cols)
                yac_inp = processor['yac'].transform(yac_prob)
                yac_inp[np.isnan(yac_inp)]=0
                #print(processor['yac'].get_feature_names_out()[(abs(yac_inp)>3)[0,:]])
                #print(yac_prob.loc[0,['team_4th_rate','epa_rate']])
                yac_prob = get_probs(model_dict['yac'](torch.as_tensor(yac_inp.astype(np.float32))).detach().numpy())
                #plt.plot(yac_prob[0,:],bin_dict['yac']['0'])
                yac_yards = np.random.choice(bin_dict['yac']['0'],p=yac_prob[0,:])
                qb.stats['pass_attempts']+=1                
                qb.stats['completions']+=1
                yards = air_yards+yac_yards
                if yards + self.yardline >= 100:
                    yards = 100- self.yardline
                self.yards_gained = yards
                wr.stats['receiving_yards'] += yards
                wr.stats['receptions'] += 1
                qb.stats['passing_yards'] += yards
                qb.stats['air_yards'] += air_yards
                self.description = f"{qb.name} completed a pass to {wr.name} for {yards} yards. Now at the {self.yardline+yards} yardline"
                fumb_columns = gc.copy()
                fumb_columns.extend(tc.copy())
                fumb_columns.extend(dc.copy())
                fumb_columns.append('completion')
                cols = []
                for collist in column_dict['fumble']:
                    cols.extend(eval(collist))
                if yards + self.yardline < 100:
                    fumb = pd.DataFrame(np.array(fumb_columns).reshape(1,-1), columns = cols)
                    fumb_inp = processor['fumble'].transform(fumb)
                    fumb_inp[np.isnan(fumb_inp)]=0
                    fumb = F.softmax(model_dict['fumble'](torch.as_tensor(fumb_inp.astype(np.float32))), dim=1).detach().numpy()[0][1]/2
                    if random.random() < fumb:
                        self.turnover = True
                        self.description = f"{qb.name} completed a pass to {wr.name} for {yards} yards. Now at the {self.yardline+yards} yardline. {wr.name} fumbled and it was recovered by the defense"
                        wr.stats['fumble_lost'] += 1
                        defense.stats['fumble_recovered'] += 1
                        ret_columns = gc.copy()
                        cols = []
                        for collist in column_dict['def_ret']:
                            cols.extend(eval(collist))
                        ret_prob = pd.DataFrame(np.array(ret_columns).reshape(1,-1), columns = cols)
                        ret_inp = processor['def_ret'].transform(ret_prob)
                        ret_inp[np.isnan(ret_inp)]=0
                        ret_prob = get_probs(model_dict['def_ret'](torch.as_tensor(ret_inp.astype(np.float32))).detach().numpy())
                        ret_yards = np.random.choice(bin_dict['def_ret']['0'],p=ret_prob[0,:])
                        self.yards_gained -= ret_yards
                        if self.yards_gained + self.yardline <= 0:
                            self.defensive_touchdown = True
                            defense.stats['defensive_touchdown'] += 1
                    
            elif interception:
                self.type = 'interception'
                qb.stats['pass_attempts']+=1
                defense.stats['interception'] += 1
                if air_yards + self.yardline >= 100:
                    air_yards = 80-(air_yards+self.yardline)
                self.yards_gained = air_yards
                self.turnover = True
                qb.stats['interceptions'] += 1
                qb.stats['air_yards'] += air_yards
                self.description = f"{qb.name} threw an interception!"
                ret_columns = gc.copy()
                cols = []
                for collist in column_dict['def_ret']:
                    cols.extend(eval(collist))
                ret_prob = pd.DataFrame(np.array(ret_columns).reshape(1,-1), columns = cols)
                ret_inp = processor['def_ret'].transform(ret_prob)
                ret_inp[np.isnan(ret_inp)]=0
                ret_prob = get_probs(model_dict['def_ret'](torch.as_tensor(ret_inp.astype(np.float32))).detach().numpy())
                ret_yards = np.random.choice(bin_dict['def_ret']['0'],p=ret_prob[0,:])
                self.yards_gained -= ret_yards
                if self.yards_gained + self.yardline <= 0:
                    self.defensive_touchdown = True
                    defense.stats['defensive_touchdown'] += 1

            elif sack:
                self.type = 'sack'
                qb.stats['sacks'] += 1
                defense.stats['sack'] += 1
                self.yards_gained = np.random.randint(-10,-1)
                self.description = f"{qb.name} was sacked for {self.yards_gained} yards"
                fumb_columns = gc.copy()
                fumb_columns.extend(tc.copy())
                fumb_columns.extend(dc.copy())
                fumb_columns.append('sack')
                cols = []
                for collist in column_dict['fumble']:
                    cols.extend(eval(collist))
                fumb = pd.DataFrame(np.array(fumb_columns).reshape(1,-1), columns = cols)
                fumb_inp = processor['fumble'].transform(fumb)
                fumb_inp[np.isnan(fumb_inp)]=0
                fumb = F.softmax(model_dict['fumble'](torch.as_tensor(fumb_inp.astype(np.float32))), dim=1).detach().numpy()[0][1]/2
                if random.random() < fumb:
                    self.turnover = True
                    self.description = f"{qb.name} was sacked for {self.yards_gained} yards and fumbled. It was recovered by the defense"
                    qb.stats['fumble_lost'] += 1
                    defense.stats['fumble_recovered'] += 1
                    ret_columns = gc.copy()
                    cols = []
                    for collist in column_dict['def_ret']:
                        cols.extend(eval(collist))
                    ret_prob = pd.DataFrame(np.array(ret_columns).reshape(1,-1), columns = cols)
                    ret_inp = processor['def_ret'].transform(ret_prob)
                    ret_inp[np.isnan(ret_inp)]=0
                    ret_prob = get_probs(model_dict['def_ret'](torch.as_tensor(ret_inp.astype(np.float32))).detach().numpy())
                    ret_yards = np.random.choice(bin_dict['def_ret']['0'],p=ret_prob[0,:])
                    self.yards_gained -= ret_yards
                    if self.yards_gained + self.yardline <= 0:
                        self.defensive_touchdown = True
                        defense.stats['defensive_touchdown'] += 1
            else:
                self.type = 'incomplete_pass'
                #qb.stats['air_yards'] += air_yards
                qb.stats['pass_attempts']+=1
                self.description = f"{qb.name} threw an incomplete pass"

        elif play_type == 'run':
            self.type = 'rush'
            rusher_selection_columns = gc.copy()
            rusher_selection_columns.extend(list(offense.rb_embed))

            all_positions = ['QB1','WR1','WR2','WR3','WR4','RB1','RB2','RB3','TE1','TE2','TE3']
            for player in offense.players:
                rusher_selection_columns.extend([player.attributes[col].values[0] for col in rusher_columns])
            cols = []
            for collist in column_dict['rusher']:
                if collist == 'rusher_columns':
                    continue
                cols.extend(eval(collist))
            filled_positions = []
            for player in offense.players:
                l = eval('rusher_columns')
                cols.extend([ind+"_"+player.position  for ind in l])
                filled_positions.append(player.position)
            for player in [pos for pos in all_positions if pos not in filled_positions]:
                l = eval('rusher_columns')
                cols.extend([ind+"_"+player  for ind in l])
                filled_positions.append(player)
                for _ in range(len(l)):
                    rusher_selection_columns.append(np.nan)
            # for colname, col in zip(rusher_selection_columns, cols):
            #     print(colname, col)
            rusher = pd.DataFrame(np.array(rusher_selection_columns).reshape(1,-1), columns = cols)
            rusher = processor['rusher'].transform(rusher)
            rusher[np.isnan(rusher)]=0
            rusher = F.softmax(model_dict['rusher'](torch.as_tensor(rusher.astype(np.float32))), dim=1).detach().numpy()
            rb = None
            selecting = True
            while selecting:
                targ = np.random.choice(all_positions,p=rusher[0,:])
                for player in offense.players:
                    if player.position == targ:
                        rb = player
                        selecting = False
                        break
#            rb = offense.get_random_player('RB')
            rusher_c = [rb.attributes[col].values[0] for col in rusher_columns]
            ry_pred_columns = gc.copy()
            ry_pred_columns.extend(dc.copy())
            ry_pred_columns.extend(rusher_c.copy())
            ry_pred_columns.extend(tc.copy())
            ry_pred_columns.append(rb.position[0:2])
            
            cols = []
            for collist in column_dict['rush']:
                if collist == 'rusher_cols':
                    collist = 'rusher_columns'
                cols.extend(eval(collist))
            rb.stats['rushing_attempts'] += 1
            ry = pd.DataFrame(np.array(ry_pred_columns).reshape(1,-1), columns = cols)
            ry_inp = processor['rush'].transform(ry)
            ry_inp[np.isnan(ry_inp)]=0
            rush_yards = get_probs(model_dict['rush'](torch.as_tensor(ry_inp.astype(np.float32))).detach().numpy())
            rush_yards = min(np.random.choice(bin_dict['rush']['0'],p=rush_yards[0,:]),100-self.yardline+1)
            self.yards_gained = rush_yards
            rb.stats['rushing_yards'] += rush_yards
            self.description = f"{rb.name} ran for {rush_yards} yards. Now at the {self.yardline+rush_yards} yardline"
            if rush_yards + self.yardline < 100:
                fumb_columns = gc.copy()
                fumb_columns.extend(tc.copy())
                fumb_columns.extend(dc.copy())
                fumb_columns.append('run')
                cols = []
                for collist in column_dict['fumble']:
                    cols.extend(eval(collist))
                fumb = pd.DataFrame(np.array(fumb_columns).reshape(1,-1), columns = cols)
                fumb_inp = processor['fumble'].transform(fumb)
                fumb_inp[np.isnan(fumb_inp)]=0
                fumb = F.softmax(model_dict['fumble'](torch.as_tensor(fumb_inp.astype(np.float32))), dim=1).detach().numpy()[0][1]/2
                if random.random() < fumb:
                    self.turnover = True
                    self.description = f"{rb.name} ran for {rush_yards} yards. Now at the {self.yardline+rush_yards} yardline. He fumbled and the defense recovered"
                    rb.stats['fumble_lost'] += 1
                    defense.stats['fumble_recovered'] += 1
                    ret_columns = gc.copy()
                    cols = []
                    for collist in column_dict['def_ret']:
                        cols.extend(eval(collist))
                    ret_prob = pd.DataFrame(np.array(ret_columns).reshape(1,-1), columns = cols)
                    ret_inp = processor['def_ret'].transform(ret_prob)
                    ret_inp[np.isnan(ret_inp)]=0
                    ret_prob = get_probs(model_dict['def_ret'](torch.as_tensor(ret_inp.astype(np.float32))).detach().numpy())
                    ret_yards = np.random.choice(bin_dict['def_ret']['0'],p=ret_prob[0,:])
                    self.yards_gained -= ret_yards
                    if self.yards_gained + self.yardline <= 0:
                        self.defensive_touchdown = True
                        defense.stats['defensive_touchdown'] += 1


        # Check for touchdown
        if self.yardline + self.yards_gained >= 100 and not self.turnover:
            self.touchdown = True
            self.yards_gained = 100 - self.yardline
            self.description += " — TOUCHDOWN!"
            offense.score += 6
            if play_type == 'pass':
                qb.stats['passing_touchdowns'] += 1
                wr.stats['receiving_touchdowns'] += 1
            elif play_type == 'run':
                rb.stats['rushing_touchdowns'] += 1
        if play_type == 'punt':
            self.type = 'punt'
            punter = offense.get_random_player('P')
            p_pred_columns = gc.copy()
            cols = []
            for collist in column_dict['punt']:
                cols.extend(eval(collist))
            punt = pd.DataFrame(np.array(p_pred_columns).reshape(1,-1), columns = cols)
            punt_inp = processor['punt'].transform(punt)
            punt_inp[np.isnan(punt_inp)]=0
            punt_yards = get_probs(model_dict['punt'](torch.as_tensor(punt_inp.astype(np.float32))).detach().numpy())
            punt_yards = np.random.choice(bin_dict['punt']['0'],p=punt_yards[0,:])
            self.yards_gained = punt_yards
            if punt_yards + self.yardline >= 100:
                self.yards_gained = 80-self.yardline
            if punt_yards + self.yardline <= 0:
                self.defensive_touchdown = True
            self.punt = True
            if punter:
                punter.stats['punts'] += 1
                punter.stats['punt_yards'] += punt_yards
            self.description = f"{punter.name if punter else 'Punter'} punts for {self.yards_gained} yards"
    def simulate_punt(self, offense, defense):
        gc = [defense.score, self.game.down, 3600-(self.game.quarter-1)*900 - (900 - self.game.clock), offense.score, offense.home, 100- self.game.yardline, self.game.distance, self.game.dome, offense.score-defense.score]
        punter = offense.get_random_player('P')
        p_pred_columns = gc.copy()
        cols = []
        for collist in column_dict['punt']:
            cols.extend(eval(collist))
        punt = pd.DataFrame(np.array(p_pred_columns).reshape(1,-1), columns = cols)
        punt_inp = processor['punt'].transform(punt)
        punt_inp[np.isnan(punt_inp)]=0
        punt_yards = F.softmax(model_dict['punt'](torch.as_tensor(punt_inp.astype(np.float32))), dim=1).detach().numpy()
        punt_yards = np.random.choice(bin_dict['punt']['0'],p=punt_yards[0,:])
        self.yards_gained = punt_yards
        if punt_yards + self.yardline >= 100:
            self.yards_gained = 80-self.yardline
        self.punt = True
        if punter:
            punter.stats['punts'] += 1
            punter.stats['punt_yards'] += punt_yards
        self.description = f"{punter.name if punter else 'Punter'} punts for {self.yards_gained} yards"

#%%
class Game:
    def __init__(self, team1, team2, dome):
        self.team1 = team1
        self.team2 = team2
        for player in team1.players:
            for stat in player.stats:
                player.stats[stat] = 0
        for player in team2.players:
            for stat in player.stats:
                player.stats[stat] = 0
        if random.random() < 0.5:
            self.second = team2
            self.possession = team1
            self.defense = team2
        else:
            self.possession = team2
            self.defense = team1            
        self.yardline = 30
        self.down = 1
        self.distance = 10
        self.plays = []
        self.play_number = 0
        self.quarter = 1
        self.clock = 900  # seconds left in quarter (15:00)
        self.dome = dome

    def switch_possession(self, yardline = 70):
        #print(f"starting drive at the {100-yardline} yardline")
        self.possession, self.defense = self.defense, self.possession
        self.yardline = 100-yardline
        self.down = 1
        self.distance = 10

    def tick_clock(self, play_type):
        #time_elapsed = random.randint(25, 40)
        gc = [self.defense.score, self.down, 3600-(self.quarter-1)*900 - (900 - self.clock), self.possession.score, self.possession.home, 100-self.yardline, self.distance, self.dome, self.possession.score-self.defense.score]
        tick_cols = gc.copy()
        tick_cols.append(play_type)
        cols = []
        for collist in column_dict['time']:
            cols.extend(eval(collist))
        time = pd.DataFrame(np.array(tick_cols).reshape(1,-1), columns = cols)
        time = processor['time'].transform(time)
        time[np.isnan(time)]=0
        #print(time)
        time = get_probs(model_dict['time'](torch.as_tensor(time.astype(np.float32))).detach().numpy())
        time_elapsed = int(np.random.choice(bin_dict['time']['0'],p=time[0,:]))
        self.clock -= time_elapsed
        if self.clock <= 0:
            self.advance_quarter()

    def advance_quarter(self):
        self.quarter += 1
        if self.quarter <= 4:
            #print(f"--- End of Q{self.quarter - 1} ---")
            self.clock = 900
        else:
            self.clock = 0

    def simulate_drive(self):
        while self.clock > 0:
            if self.down <= 4:
                #print("Play")
                play = Play(self.down, self.distance, self.yardline, game = self)
                play.simulate(self.possession, self.defense)
                self.plays.append(play)
                self.play_number += 1
                
                #print(f"Q{self.quarter} | {self.format_time()} | {self.yardline} | {self.team1.name}: {self.team1.score} | {self.team2.name}: {self.team2.score} | {self.down} | {self.distance}: {play.description}")
                self.tick_clock(play.type)
                #print('ticked clock')
                if play.field_goal_attempt:
                    if play.yards_gained != 0:
                        self.switch_possession(self.yardline + play.yards_gained)
                        continue
                    else:
                        self.switch_possession(70)
                if play.touchdown:
                    pat = Play(0, 0, 0, game = self)
                    pat.simulate(self.possession, self.defense, is_pat=True)
                    self.plays.append(pat)
                    self.play_number += 1
                    #print(f"Q{self.quarter} | {self.format_time()} | Play {self.play_number} | {self.down} | {self.distance}: {pat.description}")
                    self.switch_possession(70)
                    break
                if play.defensive_touchdown:
#                    print('defensive touchdown')
                    self.defense.score += 7
                    self.down = 1
                    self.yardline = 30
                    self.distance = 10
                    continue
    
                self.yardline += play.yards_gained
                if play.yards_gained >= self.distance:
                    self.down = 1
                    self.distance = 10
                else:
                    self.down += 1
                    self.distance -= play.yards_gained
                if play.punt:
                    self.switch_possession(self.yardline)
                    continue
                if play.turnover:
                    self.switch_possession(self.yardline)
                    continue
            else:
                self.switch_possession(self.yardline)


    def format_time(self):
        minutes = self.clock // 60
        seconds = self.clock % 60
        return f"{minutes:02}:{seconds:02}"

    def simulate_game(self):
        #print("Starting game...")
        while self.quarter <= 4 and self.clock > 0:
            self.simulate_drive()
        print(f"\nFinal Score:\n{self.team1.name}: {self.team1.score}\n{self.team2.name}: {self.team2.score}")
        print(len(self.plays))
#%%
roof_type = {
    "ARI": "dome",
    "ATL": "dome",
    "BAL": "outdoors",
    "BUF": "outdoors",
    "CAR": "outdoors",
    "CHI": "outdoors",
    "CIN": "outdoors",
    "CLE": "outdoors",
    "DAL": "dome",
    "DEN": "outdoors",
    "DET": "dome",
    "GB": "outdoors",
    "HOU": "dome",
    "IND": "dome",
    "JAX": "outdoors",
    "KC": "outdoors",
    "LV": "dome",
    "LAC": "dome",
    "LAR": "dome",
    "LA": "dome",
    "OAK": "outdoors",
    "MIA": "outdoors",
    "MIN": "dome",
    "NE": "outdoors",
    "NO": "dome",
    "NYG": "outdoors",
    "NYJ": "outdoors",
    "PHI": "outdoors",
    "PIT": "outdoors",
    "SF": "outdoors",
    "SEA": "outdoors",
    "TB": "outdoors",
    "TEN": "outdoors",
    "WAS": "outdoors"
}
#%%
schedules = schedules[schedules['season']==2024]
schedules = schedules[schedules['week']<19]
combined = pd.DataFrame()
for _, row in depth_chart.iterrows():#['game_id']:
    game = row['game_id']
    if row['season']!='2024':
        continue
    if game not in team_stats_wide['next_game_id'].unique():
        continue
    #print(g)
    dc = depth_chart.copy()[depth_chart['game_id']==game].reset_index(drop=False)
    if len(dc) != 2:
        continue
    try:
        df = pd.read_csv(f"C:/Users/steve/OneDrive/Desktop/Game Results/{game}.csv")
    except:
        continue
    
    schedules.loc[schedules['game_id']==game,'sim_total'] = np.quantile(df['home']+df['away'],0.5)
    schedules.loc[schedules['game_id']==game,'sim_spread'] = np.quantile(df['home']-df['away'],0.5)
    #print("\n\n\n",sum(df['home']>df['away'])/max(1,sum(df['home']!=df['away'])))
    schedules.loc[schedules['game_id']==game,'wp'] = sum(df['home']>df['away'])/max(1,sum(df['home']!=df['away']))
    df['game'] = game
    combined = pd.concat([combined,df])
schedules['home_odds'] = schedules['home_moneyline'].apply(lambda x: 1/((-100/x)+1) if x<0 else 1/((x/100)+1))
schedules['unadjusted_home_odds'] = schedules['home_odds']
schedules['away_odds'] = schedules['away_moneyline'].apply(lambda x: 1/((-100/x)+1) if x<0 else 1/((x/100)+1))
t = schedules['home_odds']/(schedules['home_odds']+schedules['away_odds']), schedules['away_odds']/(schedules['home_odds']+schedules['away_odds'])
schedules['home_odds'] = t[0]
schedules['away_odds'] = t[1]
combined.to_csv("C:/Users/steve/OneDrive/Desktop/Game Results/combined.csv")
#%%
#game = depth_chart['game_id'].sample().reset_index().values[0,1] #'2022_02_MIA_BAL'
row = depth_chart.loc[1223,:]
for _, row in depth_chart.iterrows():#['game_id']:
    game = row['game_id']
    if row['season']!='2024':
        continue
    if game not in team_stats_wide['next_game_id'].unique():
        continue
    dc = depth_chart.copy()[depth_chart['game_id']==game].reset_index(drop=False)
    if len(dc) != 2:
        continue
    try:
        df = pd.read_csv(f"C:/Users/steve/OneDrive/Desktop/Game Results/{game}.csv")
    except:
        df = pd.DataFrame()
    if len(df)> 199:
        continue
    indices = []
    for t in dc['team']:
        for col in ['QB1','WR1','WR2','WR3','WR4','RB1','RB2','RB3','TE1','TE2','TE3']:
            indices.append(dc.loc[dc['team']==t,col].values[0])
    psw = player_stats_wide.copy()[player_stats_wide['player_id'].isin(indices)]
    closest_prior_rows = []
    dc['game_date'] = pd.to_datetime(dc['game_date'])
    psw['game_date'] = pd.to_datetime(psw['game_date'])
    target_date = dc.loc[dc.index[0],'game_date']
    psw['diff'] = psw['game_date']-target_date
    filtered = psw[psw['diff'] < pd.Timedelta(0)]
    temp_player = (
        filtered
        .sort_values(['player_id', 'game_date'],ascending=False)
        .groupby('player_id', as_index=False)
        .first()
    )
    ids = dict(zip(psw["player_id"], psw["player_name"]))
    for i, name in ids.items():
        if i not in temp_player['player_id'].unique():
            temp_player.loc[len(temp_player)+1,'player_id']=i
            temp_player.loc[len(temp_player),'player_name']=name
    temp_game = team_stats_wide.copy()[team_stats_wide['next_game_id']==game]
    team_2, team_1 = temp_game['next_game_id'].values[0].split('_')[2:] #second team is home team
    temp_game.set_index('team',inplace=True,drop=True)
    dc.set_index('team',inplace=True,drop=True)
    stadium = roof_type[team_1]
    team1_players = []
    team2_players = []
    if len(temp_player)< 14:
        continue
    for col in dc.columns:
        if col in ['game_id','game_date']:
            continue
        try:
            team1_players.append(Player(temp_player.loc[temp_player['player_id']==dc.loc[team_1,col],'player_name'].values[0],
                      col,
                      temp_player.loc[temp_player['player_id']==dc.loc[team_1,col],'player_id'].values[0],
                      temp_player.loc[temp_player['player_id']==dc.loc[team_1,col],temp_player.columns[5:-1]]))
        except:
            print(col)
            pass
    for col in dc.columns:
        if col in ['game_id','game_date']:
            continue
        try:
            team2_players.append(Player(temp_player.loc[temp_player['player_id']==dc.loc[team_2,col],'player_name'].values[0],
                      col,
                      temp_player.loc[temp_player['player_id']==dc.loc[team_2,col],'player_id'].values[0],
                      temp_player.loc[temp_player['player_id']==dc.loc[team_2,col],temp_player.columns[5:-1]]))
        except:
            print(col)
            pass
    # Teams
    
    # Game
    home_score, away_score = [], []
    for _ in range(200):
        away = Team(team_2, team2_players, "away", temp_game.loc[team_2,temp_game.columns[3:]])
        home = Team(team_1, team1_players, "home", temp_game.loc[team_1,temp_game.columns[3:]])
        g = Game(away, home, stadium)
        try:
            t1 = time.time()
            g.simulate_game()
            t2 = time.time()
            print(t2-t1)
        except Exception as e:
            print("Something has broken: ", str(e))
            break
        home_score.append(home.score)
        away_score.append(away.score)
        differential = np.array(home_score)-np.array(away_score)
        mean = differential.mean()
        total = np.array(home_score)+np.array(away_score)
        df_ind = len(df)
        df.loc[df_ind,'home'] = home.score
        df.loc[df_ind,'away'] = away.score
        schedules.loc[schedules['game_id']==game,'sim_total'] = np.quantile(df['home']+df['away'],0.5)
        schedules.loc[schedules['game_id']==game,'sim_spread'] = np.quantile(df['home']-df['away'],0.5)
        print("\n\n\n",sum(df['home']>df['away'])/max(1,sum(df['home']!=df['away'])))
        schedules.loc[schedules['game_id']==game,'wp'] = sum(df['home']>df['away'])/max(1,sum(df['home']!=df['away']))
        df.to_csv(f"C:/Users/steve/OneDrive/Desktop/Game Results/{game}.csv", index=0)
        if len(df) > 199:
            break
        
    # t = (schedules['sim_spread']-schedules['result'])**2
    # t = t[~np.isnan(t)]
    # t = t/len(t)
    # t = np.sqrt(t.sum())
    # actual = np.sqrt(sum((schedules['result']-schedules['spread_line'])**2/len(schedules)))
    # win_pick = (schedules['sim_spread']>0)*(schedules['result']>0)
    # win_pick = win_pick.sum()/(schedules['sim_spread']>0).sum()
    # win_pick = (schedules['spread_line']>0)*(schedules['result']>0)
    # win_pick = win_pick.sum()/(schedules['spread_line']>0).sum()

    # Example: print QB stats
    for player in away.players + home.players:
        if player.position in ["QB1"]:#,"WR1","WR2","RB1","RB2","TE1","TE2","WR3","WR4","RB3"]:
            print(f"{player.name} Stats: {player.stats}")
    t = (schedules['sim_spread']-schedules['result'])**2
    t = t[~np.isnan(t)]
    t = t/len(t)
    temp = np.sqrt(t.sum())
    actual = np.sqrt(sum((schedules['result']-schedules['spread_line'])**2/len(schedules)))
    win_pick = (schedules['wp']>0.5)*(schedules['result']>0)+(schedules['wp']<0.5)*(schedules['result']<0)
    our_win_pick = win_pick.sum()/(schedules['wp']>0.000).sum()
    home_wins = ((schedules['result']>0)*(schedules['wp']>0)).sum()/(schedules['wp']>0).sum()
    win_pick = (schedules['spread_line']>0)*(schedules['result']>0)*(schedules['wp']>0)+(schedules['spread_line']<0)*(schedules['result']<0)*(schedules['wp']>0)
    win_pick = win_pick.sum()/((schedules['spread_line']>0)*(schedules['wp']>0)+(schedules['spread_line']<0)*(schedules['wp']>0)).sum()
    brier = brier_score_loss(schedules.loc[schedules['wp']>0, 'result']>0,
                         schedules.loc[schedules['wp']>0, 'wp'])
    vegas_brier = brier_score_loss(schedules.loc[schedules['wp']>0, 'result']>0,
                         schedules.loc[schedules['wp']>0, 'home_odds'])
    unadj_vegas_brier = brier_score_loss(schedules.loc[schedules['wp']>0, 'result']>0,
                         schedules.loc[schedules['wp']>0, 'unadjusted_home_odds'])
    
    prob_true, prob_pred = calibration_curve(schedules[schedules['wp']>0]['result']>0, schedules[schedules['wp']>0]['wp'], n_bins=6, strategy='quantile')
    v_prob_true, v_prob_pred = calibration_curve(schedules['result']>0, schedules['home_odds'], n_bins=6, strategy='quantile')
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker = '+', color='green', label='McCarthy System')
    plt.plot(v_prob_pred, v_prob_true, marker='o', label='Vegas Model')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')  # 45-degree line
    plt.xlabel('Predicted Probability')
    plt.ylabel('Observed Frequency')
    plt.title('Probability by Probability (Calibration Curve)')
    plt.legend()
    plt.grid(True)
    s = (schedules['sim_total']-schedules['total'])**2
    s = s[~np.isnan(s)]
    s = s/len(s)
    s = np.sqrt(s.sum())
    total_actual = np.sqrt(sum((schedules['total']-schedules['total_line'])**2/len(schedules)))
    plt.figure()
    plt.hist(schedules['sim_spread']-schedules['result'], color='g', density=True, alpha = 0.5, label = 'McCarthy System')
    plt.hist(schedules['spread_line']-schedules['result'], color='b', density=True, alpha = 0.5, label = 'Vegas')
    plt.legend()
    plt.xlabel('Difference in Points, Spread vs Actual')
    plt.figure()
    plt.hist(schedules['sim_total']-schedules['total'], color='g', density=True, alpha = 0.5, label = 'McCarthy System')
    plt.hist(schedules['total_line']-schedules['total'], color='b', density=True, alpha = 0.5, label = 'Vegas')
    plt.figure()
    plt.plot(schedules['sim_spread'],schedules['spread_line'],'k.')
    plt.figure(figsize=(6, 6))
    s = (schedules['sim_total']-schedules['total'])**2
    s = s[~np.isnan(s)]
    s = s/len(s)
    s = np.sqrt(s.sum())
    total_actual = np.sqrt(sum((schedules['total']-schedules['total_line'])**2/len(schedules)))
schedules['sim_brier_score'] = (schedules['wp']-(schedules['result']>0).astype(float))**2
schedules['vegas_brier_score'] = (schedules['home_odds']-(schedules['result']>0).astype(float))**2
weeks = schedules.groupby(by='week')[['sim_brier_score','vegas_brier_score']].mean()
#%%
df['total'] = df['home']+df['away']
df['Baltimore Points'] = df['home']
df['Cleveland Points'] = df['away']
df['home_margin'] = df['home']-df['away']
df['home_win'] = df['home']>df['away']
plt.plot(df['home'],df['away'],'k.')
#%%
plt.figure(figsize=(8,6))
ax = sns.kdeplot(
    data=df,
    x="Baltimore Points",
    y="Cleveland Points",
    fill=True,      # fills the contours
    multiple='stack',
    cmap="GnBu", # colormap
    thresh=0,       # show full range
    levels=10      # smoother gradations
    #cbar = True,
    #cbar_kws={'label': 'Density'}
)
cbar = ax.collections[-1].colorbar
#cbar.set_ticks([])
plt.xlim(0,40)       # adjust to your desired range for total
plt.ylim(0,40)
plt.plot([0, 40], [0, 40], 'k--')  # 45-degree line
#%%Testing inference
injuries = nfl.import_injuries([2024])
injuries = injuries[injuries['report_status']=='Out']
most_recent_psw = player_stats_wide.groupby(by="player_id").last()
date_mask = (pd.to_datetime(most_recent_psw['game_date']) - max(pd.to_datetime(most_recent_psw['game_date']))).dt.days<-500
most_recent_psw.loc[date_mask,most_recent_psw.columns[most_recent_psw.columns.get_loc("close_bayes")]:] = np.nan
most_recent_tsw = team_stats_wide.groupby(by="team").last()
new_depth_chart = nfl.import_depth_charts([2025])
new_depth_chart['position'] = new_depth_chart['pos_abb']
latest_depth_chart = new_depth_chart[
    new_depth_chart["dt"] == new_depth_chart.groupby("team")["dt"].transform("max")
]
rosters = nfl.import_weekly_rosters([2025])
rosters = rosters.merge(most_recent_psw, left_on=['player_id'], right_index=True, how='left')

ids = nfl.import_ids()
schedules = nfl.import_schedules([2025])
actually_act = []
rosters = rosters[(rosters['status']=='ACT') | (rosters['player_id']).isin(actually_act)]
rosters = rosters[rosters['week']==max(rosters['week'])]
positions = ['WR','RB','TE','QB']
rosters = rosters[rosters['position'].isin(['WR','RB','TE','QB'])]
inactives = ["00-0036754","00-0036963"]
date_mask = (pd.to_datetime(rosters['game_date']) - max(pd.to_datetime(rosters['game_date']))).dt.days<-500
rosters.loc[date_mask,rosters.columns[rosters.columns.get_loc("close_bayes")]:] = np.nan
for position in ['WR','RB','TE','QB']:
    rosters.loc[(date_mask*rosters['position']==position),rosters.columns[rosters.columns.get_loc("close_bayes"):]] = rosters.loc[(date_mask*rosters['position']==position),rosters.columns[rosters.columns.get_loc("close_bayes"):]].fillna(rosters.loc[(rosters['position']==position),rosters.columns[rosters.columns.get_loc("close_bayes"):]].mean(axis = 0))
rosters = rosters[~rosters['player_id'].isin(inactives)]
depth_chart_qb = latest_depth_chart[latest_depth_chart['position'] == 'QB']

qb_roster = rosters[rosters['position'] == 'QB']

# Merge on game_id and player_id to get depth info for each QB in roster
qb_with_depth = qb_roster.merge(
    depth_chart_qb.set_index(['team', 'gsis_id'])['pos_rank'], 
    left_on=['team', 'player_id'],
    right_index=True,
    how='left'
)
qb_with_depth_sorted = qb_with_depth.sort_values(by=['team', 'pos_rank'])
qb_with_depth_sorted.dropna(subset='pos_rank',inplace=True)
top_qb_per_game = (
    qb_with_depth_sorted
    .groupby('team')
    .first()
    .reset_index()
)
#%%
wr_te_sorted = (
    rosters[rosters['position'].isin(['WR', 'TE'])]
    .sort_values(by='medium', ascending=False)
).reset_index(drop=True)

# Sort RB by 'rush_rate'
rb_sorted = (
    rosters[rosters['position'] == 'RB']
    .sort_values(by='rush_rate', ascending=False)
).reset_index(drop=True)
sorted_players = pd.concat([top_qb_per_game,wr_te_sorted, rb_sorted], ignore_index=True)
#%%
position_limits = {'QB': 1, 'WR': 4, 'RB': 3, 'TE': 3}
# Keep only relevant positions
sorted_players['rank'] = sorted_players.groupby(['team', 'position']).cumcount()
#sorted_players = sorted_players.sort_values(by=['team']).reset_index(drop=True)

# Filter to top N by position
position_filtered = pd.concat([
    sorted_players[sorted_players['position'] == pos].groupby(['team']).head(n)
    for pos, n in position_limits.items()
])

# Assign a label like QB1, WR1, WR2, ..., TE3
position_filtered['pos_rank'] = position_filtered.groupby(['team', 'position']).cumcount() + 1
position_filtered['label'] = position_filtered['position'] + position_filtered['pos_rank'].astype(str)

# Pivot to get GSIS IDs per label
top_skill_df = position_filtered.pivot(index=['team'], columns='label', values='player_id').reset_index()

# Optional: sort columns in desired order
column_order = ['game_id'] + [f'QB1'] + [f'WR{i}' for i in range(1, 5)] + \
               [f'RB{i}' for i in range(1, 4)] + [f'TE{i}' for i in range(1, 4)]
top_skill_df = top_skill_df[[col for col in column_order if col in top_skill_df.columns]+['team']]
nan_counts = top_skill_df.isna().sum(axis=1)
#%%Monday Night Football
home_name = "DET"
away_name = "DAL"
dc = top_skill_df.copy()[top_skill_df['team'].isin([home_name,away_name])].reset_index(drop=False)
indices = []
for t in dc['team']:
    for col in ['QB1','WR1','WR2','WR3','WR4','RB1','RB2','RB3','TE1','TE2','TE3']:
        indices.append(dc.loc[dc['team']==t,col].values[0])
stadium = roof_type[home_name]
psw = most_recent_psw.reset_index().copy()
psw = psw[psw['player_id'].isin(indices)]

temp_game = most_recent_tsw.reset_index().copy()
temp_game = temp_game[temp_game['team'].isin([home_name,away_name])]
team_2, team_1 = away_name, home_name #second team is home team
temp_game.set_index('team',inplace=True,drop=True)
dc.set_index('team',inplace=True,drop=True)
#stadium = roof_type[team_1]
team1_players = []
team2_players = []
temp_player=psw.copy()
df = pd.DataFrame()
for ind in indices:
    if ind not in temp_player['player_id'].unique():
        temp_player.loc[len(temp_player)+1,'player_id']=ind
        temp_player.loc[len(temp_player),'player_name']="Placeholder"

for col in dc.columns:
    if col in ['game_id','game_date']:
        continue
    try:
        team1_players.append(Player(temp_player.loc[temp_player['player_id']==dc.loc[team_1,col],'player_name'].values[0],
                  col,
                  temp_player.loc[temp_player['player_id']==dc.loc[team_1,col],'player_id'].values[0],
                  temp_player.loc[temp_player['player_id']==dc.loc[team_1,col],temp_player.columns[5:]]))
    except:
        print(col)
        pass
for col in dc.columns:
    if col in ['game_id','game_date']:
        continue
    try:
        team2_players.append(Player(temp_player.loc[temp_player['player_id']==dc.loc[team_2,col],'player_name'].values[0],
                  col,
                  temp_player.loc[temp_player['player_id']==dc.loc[team_2,col],'player_id'].values[0],
                  temp_player.loc[temp_player['player_id']==dc.loc[team_2,col],temp_player.columns[5:]]))
    except:
        print(col)
        pass
# Teams

# Game
home_score, away_score = [], []
player_df = pd.DataFrame()
#%%
for game_number in range(4000):
    away = Team(team_2, team2_players, "away", temp_game.loc[team_2,temp_game.columns[3:]])
    home = Team(team_1, team1_players, "home", temp_game.loc[team_1,temp_game.columns[3:]])
    g = Game(away, home, stadium)
    try:
        t1 = time.time()
        g.simulate_game()
        t2 = time.time()
        print(t2-t1)
    except Exception as e:
        print("Something has broken: ", str(e))
        traceback.print_exc()
        break
    home_score.append(home.score)
    away_score.append(away.score)
    differential = np.array(home_score)-np.array(away_score)
    mean = differential.mean()
    med = np.quantile(differential,0.50)
    total = np.array(home_score)+np.array(away_score)
    df_ind = len(df)
    df.loc[df_ind,'home'] = home.score
    df.loc[df_ind,'away'] = away.score
    if game_number %5 == 0:
        print("\n\n\n",sum(df['home']>df['away'])/max(1,sum(df['home']!=df['away'])), 1.96*((0.25/len(df)))**0.5)
    df.loc[df_ind,'home_team'] = home.name
    df.loc[df_ind,'away_team'] = away.name
    for player in away.players + home.players:
        #print(f"{player.name} Stats: {player.stats}")
        new_len = len(player_df)
        player_df.loc[new_len,'name'] = player.name
        player_df.loc[new_len,'position'] = player.position
        player_df.loc[new_len,'player_id'] = player.player_id
        player_df.loc[new_len,'game_number'] = len(df)
        
        for stat in player.stats:
            #print(stat, player.stats[stat])
            player_df.loc[new_len, stat] = player.stats[stat]
            pass
        
    for team in [away, home]:
        #print(f"{player.name} Stats: {player.stats}")
        new_len = len(player_df)
        player_df.loc[new_len,'name'] = team.name + " Defense"
        player_df.loc[new_len,'position'] = "DST"
        player_df.loc[new_len,'player_id'] = "NA"
        player_df.loc[new_len,'game_number'] = len(df)
        
        for stat in ['sack','interception','pa','fumble_recovered', 'defensive_touchdown']:
            #print(stat, player.stats[stat])
            player_df.loc[new_len, stat] = team.stats[stat]
            pass

        new_len = len(player_df)
        player_df.loc[new_len,'name'] = team.name + " Kicker"
        player_df.loc[new_len,'position'] = "K"
        player_df.loc[new_len,'player_id'] = "NA"
        player_df.loc[new_len,'game_number'] = len(df)
       
        for stat in ['pat_made','fg 0-39','fg 40-49','fg 50-59']:
            #print(stat, player.stats[stat])
            player_df.loc[new_len, stat] = team.stats[stat]
            pass
        
    player_df['fantasy_points'] = player_df['passing_yards']*0.04+(player_df['passing_yards']>300)*3+player_df['rushing_yards']/10+(player_df['rushing_yards']>100)*3+player_df['receiving_yards']*0.1+(player_df['receiving_yards']>100)*3+player_df['passing_touchdowns']*4+player_df['receiving_touchdowns']*6+player_df['rushing_touchdowns']*6+player_df['interceptions']*(-1)+player_df['receptions']*1+player_df['fumble_lost']*(-1)
    mask = player_df['position'] == 'DST'
    pa = player_df.loc[mask, 'pa']
    conditions = [
    pa == 0,
    pa < 7,
    pa < 14,
    pa < 21,
    pa < 28,
    pa < 35,
    pa > 34
    ]
    choices = [
        10,
        7,
        4,
        1,
        0,
        -1,
        -4
    ]
    player_df.loc[mask,'fantasy_points'] = player_df.loc[mask,'defensive_touchdown']*6+player_df.loc[mask,'sack']+player_df.loc[mask,'interception']*2+np.select(conditions,choices)+player_df.loc[mask,'fumble_recovered']*2
    mask = player_df['position'] == 'K'
    player_df.loc[mask,'fantasy_points'] = player_df.loc[mask,'pat_made']+player_df.loc[mask,'fg 0-39']*3+player_df.loc[mask,'fg 40-49']*4+player_df.loc[mask,'fg 50-59']*5
    tote = np.quantile(total,0.5)
    pivot = pd.pivot_table(player_df, values = [col for col in player_df.columns if col not in ['name','position','player_id']], index = ['name','position','player_id'], aggfunc="mean")
    med_piv = pd.pivot_table(player_df, values = [col for col in player_df.columns if col not in ['name','position','player_id']], index = ['name','position','player_id'], aggfunc="median")
    if game_number % 100 == 0:
        fantasy_matrix = player_df.pivot_table(
            index="game_number",        # <-- replace with your simulation/game identifier
            columns="name",   # <-- replace with your player identifier
            values="fantasy_points",
            aggfunc="first"        # one value per sim_id/player_id
        )
        fantasy_matrix.to_csv("C:/Users/steve/OneDrive/Desktop/simulations.csv")
        #sal_dict = {"B.Mayfield":10600, "N.Collins":11400, "B.Irving":11200, "M.Evans":10800, "E.Egbuka":9800, "C.Stroud":9400, "N.Chubb":7000, "J.Higgins":6400,"C.Otton":5000,"D.Culp":200,"D.Ogunbowale":1400,"D.Pierce":2400,"D.Schultz":4400,"H.Bryant":200,"J.Noel":1200,"J.Watson":1000,"P.Durham":600,"R.White":3200,"S.Shepard":3000,"S.Tucker":400,"T.Johnson":200}
        # correlation matrix across players
        corr = fantasy_matrix.corr()
#%%Power Rankings
try:
    df = pd.read_csv("C:/Users/steve/OneDrive/Desktop/Game Results/week 3.csv").iloc[:,1:]
except:
    df = pd.DataFrame(columns= ["home_team","away_team"])
for home_name in roof_type:
    print(home_name)
    if home_name in ["LAR","OAK"]:
        continue
    stadium = roof_type[home_name]
    for away_name in roof_type:
        if away_name in ["LAR","OAK"]:
            continue
        if home_name == away_name:
            continue
        
        dc = top_skill_df.copy()[top_skill_df['team'].isin([home_name,away_name])].reset_index(drop=False)
        indices = []
        for t in dc['team']:
            for col in ['QB1','WR1','WR2','WR3','WR4','RB1','RB2','RB3','TE1','TE2','TE3']:
                indices.append(dc.loc[dc['team']==t,col].values[0])
        
        psw = most_recent_psw.reset_index().copy()
        psw = psw[psw['player_id'].isin(indices)]
        
        temp_game = most_recent_tsw.reset_index().copy()
        temp_game = temp_game[temp_game['team'].isin([home_name,away_name])]
        team_2, team_1 = away_name, home_name #second team is home team
        temp_game.set_index('team',inplace=True,drop=True)
        dc.set_index('team',inplace=True,drop=True)
        #stadium = roof_type[team_1]
        team1_players = []
        team2_players = []
        temp_player=psw.copy()
        for ind in indices:
            if ind not in temp_player['player_id'].unique():
                temp_player.loc[len(temp_player)+1,'player_id']=ind
                temp_player.loc[len(temp_player),'player_name']="Placeholder"
        
        for col in dc.columns:
            if col in ['game_id','game_date']:
                continue
            try:
                team1_players.append(Player(temp_player.loc[temp_player['player_id']==dc.loc[team_1,col],'player_name'].values[0],
                          col,
                          temp_player.loc[temp_player['player_id']==dc.loc[team_1,col],'player_id'].values[0],
                          temp_player.loc[temp_player['player_id']==dc.loc[team_1,col],temp_player.columns[5:]]))
            except:
                print(col)
                pass
        for col in dc.columns:
            if col in ['game_id','game_date']:
                continue
            try:
                team2_players.append(Player(temp_player.loc[temp_player['player_id']==dc.loc[team_2,col],'player_name'].values[0],
                          col,
                          temp_player.loc[temp_player['player_id']==dc.loc[team_2,col],'player_id'].values[0],
                          temp_player.loc[temp_player['player_id']==dc.loc[team_2,col],temp_player.columns[5:]]))
            except:
                print(col)
                pass
        # Teams
        
        # Game
        home_score, away_score = [], []
        #player_df = pd.DataFrame()
        for game_number in range(50):
            if len(df.loc[(df['home_team']==home_name) & (df['away_team']==away_name),:])>49:
                break
            away = Team(team_2, team2_players, "away", temp_game.loc[team_2,temp_game.columns[3:]])
            home = Team(team_1, team1_players, "home", temp_game.loc[team_1,temp_game.columns[3:]])
            g = Game(away, home, stadium)
            try:
                t1 = time.time()
                g.simulate_game()
                t2 = time.time()
                print(t2-t1)
            except Exception as e:
                print("Something has broken: ", str(e))
                break
            home_score.append(home.score)
            away_score.append(away.score)
            differential = np.array(home_score)-np.array(away_score)
            mean = differential.mean()
            total = np.array(home_score)+np.array(away_score)
            df_ind = len(df)
            df.loc[df_ind,'home'] = home.score
            df.loc[df_ind,'away'] = away.score
            print("\n\n\n",sum(df['home']>df['away'])/max(1,sum(df['home']!=df['away'])))
            df.loc[df_ind,'home_team'] = home.name
            df.loc[df_ind,'away_team'] = away.name
            df.to_csv("C:/Users/steve/OneDrive/Desktop/Game Results/week 3.csv")
        #     for player in away.players + home.players:
        #         #print(f"{player.name} Stats: {player.stats}")
        #         new_len = len(player_df)
        #         player_df.loc[new_len,'name'] = player.name
        #         player_df.loc[new_len,'position'] = player.position
        #         player_df.loc[new_len,'player_id'] = player.player_id
        #         player_df.loc[new_len,'game_number'] = len(df)
        #         for stat in player.stats:
        #             #print(stat, player.stats[stat])
        #             player_df.loc[new_len, stat] = player.stats[stat]
        #             pass
        # pivot = pd.pivot_table(player_df, values = [col for col in player_df.columns if col not in ['name','position','player_id']], index = ['name','position','player_id'], aggfunc=["mean","std"])
        # t = (schedules['sim_spread']-schedules['result'])**2
        # t = t[~np.isnan(t)]
        # t = t/len(t)
        # t = np.sqrt(t.sum())
        # actual = np.sqrt(sum((schedules['result']-schedules['spread_line'])**2/len(schedules)))
        # win_pick = (schedules['sim_spread']>0)*(schedules['result']>0)
        # win_pick = win_pick.sum()/(schedules['sim_spread']>0).sum()
        # win_pick = (schedules['spread_line']>0)*(schedules['result']>0)
        # win_pick = win_pick.sum()/(schedules['spread_line']>0).sum()
        
        # Example: print QB stats
        
        # df['total'] = df['home']+df['away']
        # df['MIN Points'] = df['away']
        # df['CHI Points'] = df['home']
        # df['home_margin'] = df['home']-df['away']
        # df['home_margin'].quantile(0.5)
        # df['home_win'] = df['home']>df['away']
        # plt.plot(df['home'],df['away'],'k.')
    df['home_margin'] = df['home']-df['away']
    pivot = pd.pivot_table(df, columns = ["away_team"], index = ["home_team"], values = "home_margin", aggfunc = "mean")
    means = pd.DataFrame((pivot.mean(axis=1)-pivot.mean(axis=0))*0.5)
    old_df = pd.read_csv("C:/Users/steve/OneDrive/Desktop/Game Results/week 1.csv").iloc[:,1:]
    old_df['home_margin'] = old_df['home']-old_df['away']
    pivot = pd.pivot_table(old_df, columns = ["away_team"], index = ["home_team"], values = "home_margin", aggfunc = "mean")
    old_means = (pivot.mean(axis=1)-pivot.mean(axis=0))*0.5
    means['Week/Week Change'] = means[0]-old_means
#%%
from scipy.stats import gaussian_kde
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import axes3d
import math
from great_tables import GT
means = means.rename({0:'Spread'})
df = means.reset_index()
df.columns = ["Team", "Spread vs rest of NFL"]

gt = (
    GT(df)
    .tab_header(
        title="Spread Comparison",
        subtitle="Home team spread vs rest of NFL"
    )
)

with open("C:/Users/steve/OneDrive/Desktop/spread_table.html", "w", encoding="utf-8") as f:
    f.write(gt.as_raw_html())
#%%
# Normalize vector
def normalize(X):
    return X / (1e-16 + np.sqrt((np.array(X) ** 2).sum(axis=-1)))[..., np.newaxis]

# get data
fig, ax = plt.subplots()
x = df[(df['home_team']=='BUF') & (df['away_team'].isin(['TEN', 'CHI', 'DAL']))]['home']
y = df[(df['home_team']=='BUF') & (df['away_team'].isin(['TEN', 'CHI', 'DAL']))]['away']
#x = player_df[player_df['name']=='C.Williams']['passing_yards']
#y = player_df[player_df['name']=='J.Jefferson']['receiving_yards']
# Create a grid over which to evaluate the KDE
xmin, xmax = 0, 45
ymin, ymax = 0, 45
xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

# Calculate the kernel density estimate
positions = np.vstack([xx.ravel(), yy.ravel()])
values = np.vstack([x, y])
kernel = gaussian_kde(values)
z = np.reshape(kernel(positions).T, xx.shape)

cs = plt.contour(xx, yy, z, levels=10, colors='None')
plt.contourf(xx, yy, z, levels=10, cmap="GnBu")

# Get contour segments
segs = cs.allsegs
# number of contour levels
num_levels = len(cs.allsegs)

# LightSource direction 225deg
rads = math.radians(45)
ls = np.array([math.cos(rads), math.sin(rads)])

# Loop through levels
for lvl in range(num_levels):
    for el in range(len(cs.allsegs[lvl])):

    # Get segment from level=lvl and element=el
        xn = cs.allsegs[lvl][el][:, 0]
        yn = cs.allsegs[lvl][el][:, 1]
        points = np.array([xn, yn]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-2], points[1:-1],points[2:]], axis=1)
    
        # Compute normal vector from segments
        dy = segments[:, 1, 1] - segments[:, 0, 1]
        dx = segments[:, 1, 0] - segments[:, 0, 0]
        N = normalize(np.column_stack([-dy,dx]))
    
        # Compute cosine of angle theta between normal and lightsource
        costheta = -np.dot(N, ls)
    
        # LineWidths based on cos(theta)
        lwidths = np.abs(0.7 * costheta)
    
        lc = LineCollection(segments, linewidths=lwidths, cmap="Greys", norm=plt.Normalize(-1, 1))
        
        # Color based on cosine
        lc.set_array(costheta)
    
        ax.add_collection(lc)
plt.plot([0, 45], [0, 45], 'k--', alpha=0.5)  # 45-degree line
plt.xlabel('Buffalo Points')
plt.ylabel('Bottom-Three Points')
