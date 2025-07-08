# Formula 1 Win Predictor

This repository contains an experimental workflow to predict the winner of a Formula 1 race. It analyzes early laps and contextual information, trains various machine-learning models and outputs the driver most likely to win.

## Installation

1. **Enable Git LFS**
   The project includes large files managed with Git Large File Storage. Ensure LFS is enabled so that notebook files are downloaded correctly:

   ```bash
   git lfs install
   git lfs pull
   ```

2. **Install dependencies**
   Use the requirements file to install Python packages:

   ```bash
   pip install -r requirements.txt
   ```

After setup, open `f1_win_predictor.ipynb` in Jupyter to explore the model. Configuration values are defined in `config.py`.
