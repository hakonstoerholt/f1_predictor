import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load Data ---
standings_df = pd.read_csv('championship_standings.csv')

# --- Load Model ---
model = joblib.load('f1_champion_predictor.joblib')

# --- Feature Engineering ---
# Recreate the target variable
final_standings = standings_df.loc[standings_df.groupby('meeting_key')['championship_points'].idxmax()]
champion_driver_number = final_standings.loc[final_standings['championship_points'].idxmax()]['driver_number']
standings_df['is_champion'] = (standings_df['driver_number'] == champion_driver_number).astype(int)

# --- Prepare Data for Evaluation ---
features = ['championship_points']
target = 'is_champion'
X = standings_df[features]
y = standings_df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Generate Predictions ---
y_pred = model.predict(X_test)

# --- Classification Report ---
print("Classification Report:")
print(classification_report(y_test, y_pred))

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Champion', 'Champion'], yticklabels=['Not Champion', 'Champion'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
print("\nConfusion matrix saved to confusion_matrix.png")
