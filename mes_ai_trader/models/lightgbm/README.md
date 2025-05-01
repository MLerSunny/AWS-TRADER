# LightGBM Price Movement Classifier

This module trains a LightGBM classifier to predict the direction of price movement (up/down) in the next 30 seconds for the Micro E-mini S&P 500 futures contract (MES).

## Features

- Pulls 30 days of feature data from the Feast feature store
- Trains a binary classifier (up/down) for 30-second price movement
- Generates and logs SHAP values for model interpretation
- Saves the trained model for inference

## Data Sources

The model uses the following feature groups from the feature store:

- RSI-14 features
- ATR-14 features
- VWAP features
- Order book imbalance features
- Margin features

## Model Architecture

The model is a gradient boosting decision tree (GBDT) classifier with the following parameters:

- Learning rate: 0.05
- Maximum tree depth: Unlimited (-1)
- Number of leaves: 31
- Feature fraction: 90%
- Bagging fraction: 80%
- Early stopping: 50 rounds
- Maximum boost rounds: 500

## Usage

### Training the Model

```bash
cd mes_ai_trader/models/lightgbm
python train_lgb.py
```

### Loading the Model for Inference

```python
import pickle

# Load the model
with open('lgbm_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Make predictions
predictions = model.predict(X_test)
```

## Output Files

- `lgbm_model.pkl`: Serialized LightGBM model
- `../../docs/shap_last.png`: SHAP values visualization for feature importance

## Requirements

- LightGBM
- Feast
- SHAP
- Matplotlib
- Pandas
- NumPy
- Scikit-learn
