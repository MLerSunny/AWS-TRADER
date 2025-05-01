# Temporal Fusion Transformer (TFT) for Return Prediction

This module trains a Temporal Fusion Transformer model to predict 10%, 50%, and 90% quantiles of MES future returns for the next 30 seconds.

## Features

- Implements PyTorch Forecasting's TFT model for quantile prediction
- Pulls feature data from the Feast feature store
- Utilizes 60 seconds of context to predict 30 seconds ahead
- Computes probability distribution of returns via quantile prediction
- Uses attention mechanisms to identify significant features

## Model Architecture

The TFT model is a state-of-the-art architecture for time series forecasting that combines:

- LSTM encoders for processing sequential data
- Attention layers for capturing dependencies
- Quantile regression for predicting distributions
- Skip connections and gating mechanisms for better feature selection

Key parameters:
- Hidden size: 64
- Attention heads: 4
- Dropout: 0.1
- Learning rate: 0.001
- Batch size: 64
- Max epochs: 50
- Quantiles: 10%, 50%, 90%

## Data Processing

- Features are pulled from the feature store at 1-second intervals
- 60 seconds of historical data is used for each prediction
- Data is partitioned 80/20 for training and validation
- Group normalization is applied to account for different scales

## Usage

### Training the Model

```bash
cd mes_ai_trader/models/tft
python train_tft.py
```

### Loading the Model for Inference

```python
import torch
from pytorch_forecasting.models import TemporalFusionTransformer

# Load the model
state_dict = torch.load('tft_model.pt')
model = TemporalFusionTransformer.load_from_checkpoint('path_to_checkpoint')
model.load_state_dict(state_dict)

# Make predictions
predictions = model.predict(data, mode="quantiles")
```

## Output Files

- `tft_model.pt`: Saved PyTorch model state dict
- `checkpoints/`: Directory containing training checkpoints

## Requirements

- PyTorch Lightning
- PyTorch Forecasting
- Feast
- Pandas
- NumPy
- Matplotlib 