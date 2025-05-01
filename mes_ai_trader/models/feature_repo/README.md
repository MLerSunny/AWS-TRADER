# MES AI Trader Feature Repository

This directory contains the [Feast](https://feast.dev/) feature store configuration for the MES AI Trader project.

## Overview

The feature store contains technical indicators and market data features for the Micro E-mini S&P 500 futures contract (MES). These features are used for model training and online inference.

## Feature Sets

| Feature Set | Description | Features |
|-------------|-------------|----------|
| RSI-14 | Relative Strength Index (14-period) | RSI value, trend direction, overbought/oversold flags |
| ATR-14 | Average True Range (14-period) | ATR value, normalized ATR, historical percentile |
| VWAP | Volume-Weighted Average Price | VWAP value, price-to-VWAP ratio, trend direction |
| Order Book Imbalance | Market microstructure metrics | Imbalance ratio, bid/ask volumes, MA of imbalance |
| Margin | Margin requirements | Initial/maintenance margins, 1-day change, margin-to-price ratio |

## Configuration

- **Online Store**: AWS DynamoDB
- **Offline Store**: AWS S3 at `s3://mes-artifacts/features/`
- **Registry**: Stored at `s3://mes-artifacts/features/registry.db`

## Usage

### Initialize or Update Feature Registry

```bash
cd models/feature_repo
feast apply
```

### Generate Sample Data (Development)

```bash
cd models/feature_repo
python data_generator.py
```

### Load Features to Online Store

```bash
feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")
```

### Retrieve Features for Training

```python
from feast import FeatureStore

# Initialize the store
store = FeatureStore(repo_path="models/feature_repo")

# Get training data
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "rsi_features:rsi_14",
        "atr_features:atr_14",
        "vwap_features:vwap",
        "order_book_features:imbalance",
        "margin_features:initial_margin",
    ],
).to_df()
```

### Retrieve Features for Online Inference

```python
from feast import FeatureStore
import pandas as pd

# Initialize the store
store = FeatureStore(repo_path="models/feature_repo")

# Create an entity DataFrame
entity_df = pd.DataFrame(
    {
        "market_id": ["MES"],
        "event_timestamp": [pd.Timestamp.now()],
    }
)

# Get online features
features = store.get_online_features(
    entity_rows=[{"market_id": "MES"}],
    features=[
        "rsi_features:rsi_14",
        "atr_features:atr_14",
        "vwap_features:vwap",
        "order_book_features:imbalance",
        "margin_features:initial_margin",
    ],
).to_dict()
```

## Extending

To add new feature views:

1. Define the data source in `feature_views.py`
2. Define the feature view in the same file
3. Run `feast apply` to update the registry 