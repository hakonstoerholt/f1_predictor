import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from datetime import datetime

# --- Load Data ---
features_df = pd.read_csv('features.csv')

# --- Prepare Data for Modeling ---
# Fill missing values with 0
features_df.fillna(0, inplace=True)

# Separate current season for prediction
current_year = datetime.now().year
train_df = features_df[features_df['year'] < current_year]
predict_df = features_df[features_df['year'] == current_year]

# Define features and target
features = [
    'cumulative_points',
    'avg_qualifying_position',
    'recent_form',
    'points_gap',
    'team_avg_points'
]
target = 'is_champion'

X_train = train_df[features]
y_train = train_df[target]

# --- Model Training ---
if not X_train.empty:
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    # --- Save Model ---
    joblib.dump(model, 'f1_champion_predictor.joblib')
    print("Model trained and saved.")

    # --- Predict Championship Winner for Current Season ---
    if not predict_df.empty:
        X_predict = predict_df[features]

        # Get the latest standings for each driver in the current season
        latest_standings = predict_df.loc[predict_df.groupby('driver_number')['date_start'].idxmax()]
        X_latest = latest_standings[features]

        # Predict probabilities
        probabilities = model.predict_proba(X_latest)[:, 1]
        latest_standings['champion_probability'] = probabilities

        # Get the predicted champion
        predicted_champion = latest_standings.loc[latest_standings['champion_probability'].idxmax()]

        print("\nPredicted Championship Winner:")
        print(predicted_champion[['driver_number', 'champion_probability']])
    else:
        print("No data available for the current season to make a prediction.")
else:
    print("Training data is empty. Cannot train model.")
