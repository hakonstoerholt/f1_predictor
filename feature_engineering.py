import pandas as pd
import numpy as np

# --- Load Data ---
sessions_df = pd.read_csv('sessions.csv')
results_df = pd.read_csv('results.csv')
drivers_df = pd.read_csv('drivers.csv')

# --- Pre-processing ---
sessions_df['date_start'] = pd.to_datetime(sessions_df['date_start'])
sessions_df['year'] = sessions_df['date_start'].dt.year

# --- Merge DataFrames ---
results_merged_df = pd.merge(results_df, sessions_df[['session_key', 'session_name', 'meeting_key', 'year', 'date_start']], on='session_key')

# Create a driver to team mapping for each year
driver_team_map = drivers_df.groupby(['year', 'driver_number'])['team_name'].first().reset_index()
results_merged_df = pd.merge(results_merged_df, driver_team_map, on=['year', 'driver_number'], how='left')


# --- Feature Engineering ---
def create_features(df):
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values(by=['date_start', 'driver_number'])

    # Calculate points for each race
    points_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
    race_results = df[df['session_name'] == 'Race'].copy()
    race_results['points'] = race_results['position'].map(points_map).fillna(0)

    # Calculate cumulative points
    race_results['cumulative_points'] = race_results.groupby(['year', 'driver_number'])['points'].cumsum().shift(1).fillna(0)

    # Qualifying performance
    qualifying_results = df[df['session_name'] == 'Qualifying'].copy()
    qualifying_results['qualifying_position'] = qualifying_results['position']
    qualifying_avg = qualifying_results.groupby(['year', 'driver_number'])['qualifying_position'].expanding().mean().reset_index()
    qualifying_avg.rename(columns={'qualifying_position': 'avg_qualifying_position'}, inplace=True)

    if 'level_2' in qualifying_avg.columns:
        qualifying_avg.drop(columns=['level_2'], inplace=True)

    # Merge features
    race_results = pd.merge(race_results, qualifying_avg, on=['year', 'driver_number'], how='left')

    # Recent form (last 5 races)
    race_results['recent_form'] = race_results.groupby(['year', 'driver_number'])['points'].rolling(window=5, min_periods=1).mean().reset_index(level=[0,1], drop=True).shift(1)

    # Points gap to leader
    leader_points = race_results.groupby('date_start')['cumulative_points'].max()
    race_results = race_results.merge(leader_points.rename('leader_points'), on='date_start')
    race_results['points_gap'] = race_results['leader_points'] - race_results['cumulative_points']

    # Team performance (teammate's average points)
    team_points = race_results.groupby(['date_start', 'team_name'])['cumulative_points'].mean().rename('team_avg_points').reset_index()
    race_results = race_results.merge(team_points, on=['date_start', 'team_name'])

    # Determine the champion for each season
    champions = race_results.loc[race_results.groupby('year')['cumulative_points'].idxmax()]
    champion_map = champions.set_index('year')['driver_number'].to_dict()
    race_results['is_champion'] = race_results.apply(lambda row: 1 if champion_map.get(row['year']) == row['driver_number'] else 0, axis=1)

    return race_results

features_df = create_features(results_merged_df)

# --- Save Processed Data ---
features_df.to_csv('features.csv', index=False)

print("Feature engineering complete. Features saved to features.csv")
