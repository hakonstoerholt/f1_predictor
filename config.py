# Configuration settings for F1 Win Predictor

# Data Collection Settings
DATA_YEARS = [2022, 2023, 2024]  # Years to collect data from
MAX_RACES_PER_YEAR = None  # Limit races per year (None for all)
CACHE_DIR = "~/.fastf1_cache"  # FastF1 cache directory

# Feature Engineering Settings
EARLY_LAPS_THRESHOLD = 10  # Number of laps to use for early race features
MIN_LAPS_FOR_ANALYSIS = 5  # Minimum laps for meaningful analysis

# Model Training Settings
TEST_SIZE = 0.2  # Proportion of data for testing
RANDOM_STATE = 42  # For reproducibility
CV_FOLDS = 5  # Cross-validation folds
HYPERPARAMETER_TUNING = True  # Enable hyperparameter tuning

# Model Selection
DEFAULT_MODELS = [
    'Random Forest',
    'XGBoost', 
    'Gradient Boosting',
    'Neural Network'
]

# Performance Thresholds
MIN_ACCURACY = 0.6  # Minimum acceptable accuracy
MIN_F1_SCORE = 0.5  # Minimum acceptable F1 score

# Visualization Settings
PLOT_STYLE = 'plotly_white'  # Default plot style
FIGURE_SIZE = (12, 8)  # Default figure size
TOP_N_FEATURES = 15  # Number of top features to display

# File Paths
MODELS_DIR = "models/"  # Directory to save trained models
PREDICTIONS_DIR = "predictions/"  # Directory to save predictions
RESULTS_DIR = "results/"  # Directory to save results

# API Settings (for future enhancements)
WEATHER_API_KEY = None  # Weather API key for enhanced weather data
UPDATE_INTERVAL = 3600  # Update interval in seconds for real-time features

# Track Characteristics (can be expanded)
TRACK_DIFFICULTY = {
    'Monaco': 10,
    'Singapore': 9,
    'Baku': 8,
    'Suzuka': 7,
    'Interlagos': 7,
    'Spa-Francorchamps': 6,
    'Silverstone': 5,
    'Hungaroring': 5,
    'Monza': 3,
    'Sochi': 2
}

# Points System
POINTS_SYSTEM = {
    1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
    6: 8, 7: 6, 8: 4, 9: 2, 10: 1
}

# Feature Groups for Analysis
FEATURE_GROUPS = {
    'lap_performance': ['AvgLapTime', 'StdDevLapTime', 'ConsistencyScore'],
    'position_data': ['AvgPosition', 'QualiPosition', 'PositionGainedLap1'],
    'championship_context': ['PointsBeforeRace', 'ChampionshipPositionBeforeRace'],
    'weather': ['Weather_TrackTemp_mean', 'Weather_AirTemp_mean', 'IsRainyRace'],
    'track_specific': ['TrackDifficulty', 'OvertakingDifficulty']
}
