# Premier League Predictor - Backend Server

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from thefuzz import process
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests # New library for making API calls

# Initialize the Flask web server.
app = Flask(__name__)
# Enable Cross-Origin Resource Sharing (CORS) to allow the UI to communicate with the server.
CORS(app)

# Global variables to store the model and data once loaded.
best_model = None
label_encoder = None
features_columns = None
team_stats = {}
teams = []
data = None
model_accuracy = 0.0 # Stores the calculated accuracy of the model.

# --- Function to get live odds from The Odds API ---
def get_live_odds(home_team, away_team):
    """Fetches live betting odds from The Odds API."""
    # --- inserted the API key here ---
    API_KEY = "29981667af634cde0f98c8f45ece5306" 
    # --------------------------------------------------------------------

    if API_KEY == "YOUR_API_KEY_HERE":
        print("WARNING: Live odds fetching is disabled. Please add your API key from the-odds-api.com")
        return None

    SPORT = 'soccer_epl'
    REGIONS = 'uk'
    MARKETS = 'h2h'
    
    url = f'https://api.the-odds-api.com/v4/sports/{SPORT}/odds/?apiKey={API_KEY}&regions={REGIONS}&markets={MARKETS}'
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        odds_json = response.json()
        
        best_match = None
        highest_score = 0
        
        for game in odds_json:
            home_score = process.extractOne(home_team, [game['home_team']])[1]
            away_score = process.extractOne(away_team, [game['away_team']])[1]
            total_score = (home_score + away_score) / 2
            
            if total_score > highest_score:
                highest_score = total_score
                best_match = game

        if best_match and highest_score > 80:
            print(f"Found best match in odds feed: {best_match['home_team']} vs {best_match['away_team']} (Score: {highest_score})")
            if len(best_match['bookmakers']) > 0:
                prices = best_match['bookmakers'][0]['markets'][0]['outcomes']
                odds = {}
                for outcome in prices:
                    if outcome['name'] == best_match['home_team']:
                        odds['home'] = outcome['price']
                    elif outcome['name'] == best_match['away_team']:
                        odds['away'] = outcome['price']
                    else:
                        odds['draw'] = outcome['price']

                if 'home' in odds and 'away' in odds and 'draw' in odds:
                    print(f"Live odds found: H={odds['home']}, D={odds['draw']}, A={odds['away']}")
                    return odds
        
        print(f"Match for {home_team} vs {away_team} not found in the live odds feed.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching live odds: {e}")
        return None

def get_points(result, is_home):
    """Calculates points awarded for a given match result."""
    if result == 'H': return 3 if is_home else 0
    if result == 'A': return 0 if is_home else 3
    if result == 'D': return 1
    return 0

def find_best_match(name, choices):
    """Finds the closest team name match to handle potential user typos."""
    best_match = process.extractOne(name, choices)
    return best_match[0] if best_match else None

def load_and_prepare_data_and_model():
    """Loads all data from live URLs, engineers features, and trains the final model."""
    global best_model, label_encoder, features_columns, team_stats, teams, data, model_accuracy

    print("--- Server is starting: Fetching LIVE data and training model... ---")
    
    urls = [
        'https://www.football-data.co.uk/mmz4281/2223/E0.csv',
        'https://www.football-data.co.uk/mmz4281/2324/E0.csv',
        'https://www.football-data.co.uk/mmz4281/2425/E0.csv',
        'https://www.football-data.co.uk/mmz4281/2526/E0.csv'
    ]
    
    all_seasons_data = []
    for url in urls:
        try:
            df = pd.read_csv(url)
            df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
            all_seasons_data.append(df)
            print(f"Successfully fetched data from {url}")
        except Exception as e:
            print(f"Error fetching data from {url}. Error: {e}")
    
    master_df = pd.concat(all_seasons_data, ignore_index=True)
    
    allowed_teams = [
        "Arsenal", "Tottenham", "Liverpool", "Chelsea", "Nott'm Forest",
        "Man City", "Sunderland", "Everton", "Bournemouth", "Brentford",
        "Burnley", "Leeds United", "Fulham", "Crystal Palace", "Newcastle",
        "Man United", "Aston Villa", "Brighton", "Wolves", "West Ham"
    ]
    master_df = master_df[
        master_df['HomeTeam'].isin(allowed_teams) &
        master_df['AwayTeam'].isin(allowed_teams)
    ].copy()
    print(f"--- Data filtered for {len(allowed_teams)} specific teams. ---")

    master_df.sort_values(by='Date', inplace=True)

    data = master_df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'B365H', 'B365D', 'B365A']]
    data.dropna(inplace=True)

    teams = data['HomeTeam'].unique()

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

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    best_params = {'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 100}
    test_model = xgb.XGBClassifier(objective='multi:softmax', num_class=3, seed=42, **best_params)
    test_model.fit(X_train, y_train)
    predictions = test_model.predict(X_test)
    model_accuracy = accuracy_score(y_test, predictions)
    print(f"--- Model Accuracy calculated: {model_accuracy:.2%} ---")
    
    best_model = xgb.XGBClassifier(objective='multi:softmax', num_class=3, seed=42, **best_params)
    best_model.fit(features, target)

    print("--- Model training complete. Server is now ready for predictions. ---")

@app.route('/teams', methods=['GET'])
def get_teams():
    return jsonify({'teams': sorted(list(teams))})

@app.route('/status', methods=['GET'])
def status():
    return jsonify({'model_accuracy': model_accuracy})

@app.route('/predict', methods=['POST'])
def predict():
    json_data = request.get_json()
    home_team_input = json_data['home_team']
    away_team_input = json_data['away_team']
    
    home_team_name = find_best_match(home_team_input, teams)
    away_team_name = find_best_match(away_team_input, teams)

    if not home_team_name or not away_team_name:
        return jsonify({'error': 'Could not find one of the teams.'}), 400

    # --- MODIFIED: Added a fallback for odds ---
    odds_source = "Live API"
    live_odds = get_live_odds(home_team_name, away_team_name)
    
    if not live_odds:
        odds_source = "Historical Average"
        print(f"Falling back to historical odds for {home_team_name} vs {away_team_name}")
        # Find the last match between these two teams in our data to get typical odds.
        last_fixture = data[
            (data['HomeTeam'] == home_team_name) & 
            (data['AwayTeam'] == away_team_name)
        ].tail(1)
        
        if not last_fixture.empty:
            home_odds = last_fixture['B365H'].values[0]
            draw_odds = last_fixture['B365D'].values[0]
            away_odds = last_fixture['B365A'].values[0]
        else:
            # If they've never played, we can't even use historical odds.
            return jsonify({'error': 'No live or historical odds found for this fixture.'}), 404
    else:
        home_odds = live_odds['home']
        draw_odds = live_odds['draw']
        away_odds = live_odds['away']

    home_latest_form = team_stats[home_team_name].tail(1).drop(columns=['Date']).add_prefix('Home_')
    away_latest_form = team_stats[away_team_name].tail(1).drop(columns=['Date']).add_prefix('Away_')
    
    past_matches_sim = data[((data['HomeTeam'] == home_team_name) & (data['AwayTeam'] == away_team_name))]
    sim_h2h_win_ratio = ((past_matches_sim['FTR'] == 'H').sum() / len(past_matches_sim)) if len(past_matches_sim) > 0 else 0.5
    
    odds_df = pd.DataFrame({'B365H': [home_odds], 'B365D': [draw_odds], 'B365A': [away_odds]})
    h2h_df = pd.DataFrame({'H2H_Home_Win_Ratio': [sim_h2h_win_ratio]})
    
    prediction_input = pd.concat([odds_df, home_latest_form.reset_index(drop=True), away_latest_form.reset_index(drop=True), h2h_df], axis=1)
    prediction_input = prediction_input[features_columns]

    match_probabilities = best_model.predict_proba(prediction_input)[0]
    classes = list(label_encoder.classes_)
    
    result = {
        'home_team_corrected': home_team_name,
        'away_team_corrected': away_team_name,
        'probabilities': {
            'home_win': float(match_probabilities[classes.index('H')]),
            'draw': float(match_probabilities[classes.index('D')]),
            'away_win': float(match_probabilities[classes.index('A')])
        },
        'odds_source': odds_source # Let the front-end know where the odds came from.
    }
    return jsonify(result)

if __name__ == '__main__':
    load_and_prepare_data_and_model()
    app.run(port=5000, debug=False)

