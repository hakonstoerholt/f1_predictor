import requests
import pandas as pd
from datetime import datetime
import time

# --- Configuration ---
# Fetch data for the past 4 years, plus the current year
current_year = datetime.now().year
start_year = current_year - 4

# --- Helper function to fetch data ---
def fetch_data(endpoint, params=None):
    base_url = "https://api.openf1.org/v1/"
    response = requests.get(f"{base_url}{endpoint}", params=params)
    response.raise_for_status()
    return response.json()

# --- Initialize DataFrames ---
all_meetings_df = pd.DataFrame()
all_sessions_df = pd.DataFrame()
all_drivers_df = pd.DataFrame()
all_results_df = pd.DataFrame()

# --- Fetch Data for Each Year ---
for year in range(start_year, current_year + 1):
    print(f"Fetching data for {year}...")

    # --- Meetings ---
    meetings_params = {"year": year}
    meetings_data = fetch_data("meetings", params=meetings_params)
    meetings_df = pd.DataFrame(meetings_data)

    if meetings_df.empty:
        print(f"No data available for {year}. Skipping.")
        continue

    all_meetings_df = pd.concat([all_meetings_df, meetings_df], ignore_index=True)

    # --- Sessions ---
    sessions_data = []
    for meeting_key in meetings_df['meeting_key']:
        sessions_params = {"meeting_key": meeting_key}
        sessions_data.extend(fetch_data("sessions", params=sessions_params))
        time.sleep(1)
    sessions_df = pd.DataFrame(sessions_data)
    all_sessions_df = pd.concat([all_sessions_df, sessions_df], ignore_index=True)

    # --- Drivers (get once per season) ---
    if not sessions_df.empty:
        first_session_key = sessions_df['session_key'].iloc[0]
        drivers_params = {"session_key": first_session_key}
        drivers_data = fetch_data("drivers", params=drivers_params)
        drivers_df = pd.DataFrame(drivers_data)
        all_drivers_df = pd.concat([all_drivers_df, drivers_df], ignore_index=True)

    # --- Session Results (Race and Qualifying) ---
    results_data = []
    race_qualifying_sessions = sessions_df[sessions_df['session_name'].isin(['Race', 'Qualifying'])]
    for session_key in race_qualifying_sessions['session_key']:
        results_params = {"session_key": session_key}
        results_data.extend(fetch_data("session_result", params=results_params))
        time.sleep(1)
    results_df = pd.DataFrame(results_data)
    all_results_df = pd.concat([all_results_df, results_df], ignore_index=True)

# --- Save to CSV ---
all_meetings_df.to_csv('meetings.csv', index=False)
all_sessions_df.to_csv('sessions.csv', index=False)
all_drivers_df.to_csv('drivers.csv', index=False)
all_results_df.to_csv('results.csv', index=False)

print("\nData fetched and saved to CSV files.")
