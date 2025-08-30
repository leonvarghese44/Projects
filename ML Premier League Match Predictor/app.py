# My Premier League Match Predictor Project - The Backend Server

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from thefuzz import process
from flask import Flask, request, jsonify
from flask_cors import CORS

# First, I need to create my web server application.
app = Flask(__name__)
# This line allows my HTML page to talk to this Python server.
CORS(app)

# --- Global variables ---
# I'll store the trained model and other important things here.
# This way, the model only has to train once when I start the server.
best_model = None
label_encoder = None
features_columns = None
team_stats = {}
teams = []
data = None # This will hold my full dataset for H2H lookups

# --- My Helper Functions ---
def get_points(result, is_home):
    if result == 'H': return 3 if is_home else 0
    if result == 'A': return 0 if is_home else 3
    if result == 'D': return 1
    return 0

def find_best_match(name, choices):
    best_match = process.extractOne(name, choices)
    return best_match[0] if best_match else None

# This is the main function that does all the heavy lifting.
def load_and_prepare_data_and_model():
    # I need to use 'global' so I can modify the variables I created outside.
    global best_model, label_encoder, features_columns, team_stats, teams, data

    print("--- Server is starting: Loading data and training model... ---")
    
    # My list of data files.
    filenames = [
        'E0_2223.csv',
        'E0_2324.csv',
        'E0_2425.csv',
        'E0_2526.csv'
    ]
    all_seasons_data = [pd.read_csv(f, parse_dates=['Date'], dayfirst=True) for f in filenames]
    master_df = pd.concat(all_seasons_data, ignore_index=True)
    master_df.sort_values(by='Date', inplace=True)

    data = master_df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'B365H', 'B365D', 'B365A']]
    data.dropna(inplace=True)

    teams = data['HomeTeam'].unique()

    # The same feature engineering logic to calculate form and H2H.
    for team in teams:
        home_games = data[data['HomeTeam'] == team]
        away_games = data[data['AwayTeam'] == team]
        all_games = pd.concat([home_games, away_games]).sort_values('Date')
        all_games['GoalsScored'] = all_games.apply(lambda row: row['FTHG'] if row['HomeTeam'] == team else row['FTAG'], axis=1)
        all_games['GoalsConceded'] = all_games.apply(lambda row: row['FTAG'] if row['HomeTeam'] == team else row['FTHG'], axis=1)
        all_games['Points'] = all_games.apply(lambda row: get_points(row['FTR'], row['HomeTeam'] == team), axis=1)
        all_games['AvgGoalsScored_Last5'] = all_games['GoalsScored'].shift(1).rolling(window=5, min_periods=1).mean()
        all_games['AvgGoalsConceded_Last5'] = all_games['GoalsConceded'].shift(1).rolling(window=5, min_periods=1).mean()
        all_games['Points_Last5'] = all_games['Points'].shift(1).rolling(window=5, min_periods=1).sum()
        team_stats[team] = all_games[['Date', 'AvgGoalsScored_Last5', 'AvgGoalsConceded_Last5', 'Points_Last5']]

    home_stats_list, away_stats_list = [], []
    for index, row in data.iterrows():
        home_form = team_stats[row['HomeTeam']][team_stats[row['HomeTeam']]['Date'] < row['Date']].tail(1)
        home_stats_list.append(home_form.add_prefix('Home_'))
        away_form = team_stats[row['AwayTeam']][team_stats[row['AwayTeam']]['Date'] < row['Date']].tail(1)
        away_stats_list.append(away_form.add_prefix('Away_'))

    data = pd.concat([data.reset_index(drop=True), pd.concat(home_stats_list).reset_index(drop=True), pd.concat(away_stats_list).reset_index(drop=True)], axis=1)

    h2h_stats = []
    for index, row in data.iterrows():
        past_matches = data[((data['HomeTeam'] == row['HomeTeam']) & (data['AwayTeam'] == row['AwayTeam'])) & (data['Date'] < row['Date'])]
        h2h_stats.append(((past_matches['FTR'] == 'H').sum() / len(past_matches)) if len(past_matches) > 0 else 0.5)
    data['H2H_Home_Win_Ratio'] = h2h_stats

    data.drop(columns=['Date', 'Home_Date', 'Away_Date'], inplace=True)
    data.dropna(inplace=True)

    label_encoder = LabelEncoder()
    target = label_encoder.fit_transform(data['FTR'])
    features = data.drop(columns=['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR'])
    features_columns = features.columns

    # Using the best parameters we found earlier.
    best_params = {'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 100}
    best_model = xgb.XGBClassifier(objective='multi:softmax', num_class=3, seed=42, **best_params)
    best_model.fit(features, target)

    print("--- Model training complete. Server is now ready for predictions. ---")

# This is the API endpoint that my HTML page will call.
@app.route('/predict', methods=['POST'])
def predict():
    # Get the data the user sent from the UI.
    json_data = request.get_json()
    home_team_input = json_data['home_team']
    away_team_input = json_data['away_team']
    home_odds = float(json_data['home_odds'])
    draw_odds = float(json_data['draw_odds'])
    away_odds = float(json_data['away_odds'])

    # Find the correct team names.
    home_team_name = find_best_match(home_team_input, teams)
    away_team_name = find_best_match(away_team_input, teams)

    if not home_team_name or not away_team_name:
        return jsonify({'error': 'Could not find one of the teams.'}), 400

    # Build the input for the model, just like before.
    home_latest_form = team_stats[home_team_name].tail(1).drop(columns=['Date']).add_prefix('Home_')
    away_latest_form = team_stats[away_team_name].tail(1).drop(columns=['Date']).add_prefix('Away_')
    
    past_matches_sim = data[((data['HomeTeam'] == home_team_name) & (data['AwayTeam'] == away_team_name))]
    sim_h2h_win_ratio = ((past_matches_sim['FTR'] == 'H').sum() / len(past_matches_sim)) if len(past_matches_sim) > 0 else 0.5
    
    odds_df = pd.DataFrame({'B365H': [home_odds], 'B365D': [draw_odds], 'B365A': [away_odds]})
    h2h_df = pd.DataFrame({'H2H_Home_Win_Ratio': [sim_h2h_win_ratio]})
    
    prediction_input = pd.concat([odds_df, home_latest_form.reset_index(drop=True), away_latest_form.reset_index(drop=True), h2h_df], axis=1)
    prediction_input = prediction_input[features_columns]

    # Get the probabilities and send them back to the UI.
    match_probabilities = best_model.predict_proba(prediction_input)[0]
    classes = list(label_encoder.classes_)
    
    # --- FIXED THE ERROR HERE ---
    # I'm converting the special numpy numbers to regular python numbers.
    result = {
        'home_team_corrected': home_team_name,
        'away_team_corrected': away_team_name,
        'probabilities': {
            'home_win': float(match_probabilities[classes.index('H')]),
            'draw': float(match_probabilities[classes.index('D')]),
            'away_win': float(match_probabilities[classes.index('A')])
        }
    }
    return jsonify(result)

# This is the final part that actually starts the server.
if __name__ == '__main__':
    # I'll prepare everything once, right at the start.
    load_and_prepare_data_and_model()
    # Now, I'll start my web server.
    app.run(port=5000, debug=False)

