from datetime import timedelta

from feast import (
    Feature,
    FeatureView,
    FileSource,
    ValueType,
)

from entities import market

# Define sources for each feature set (offline store)

# Source for RSI-14 data
rsi_source = FileSource(
    path="s3://mes-artifacts/features/rsi_features.parquet",
    event_timestamp_column="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Source for ATR-14 data
atr_source = FileSource(
    path="s3://mes-artifacts/features/atr_features.parquet",
    event_timestamp_column="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Source for VWAP data
vwap_source = FileSource(
    path="s3://mes-artifacts/features/vwap_features.parquet",
    event_timestamp_column="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Source for order book imbalance data
ob_imbalance_source = FileSource(
    path="s3://mes-artifacts/features/ob_imbalance_features.parquet",
    event_timestamp_column="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Source for margin data
margin_source = FileSource(
    path="s3://mes-artifacts/features/margin_features.parquet",
    event_timestamp_column="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Define feature views

# RSI-14 Feature View
rsi_view = FeatureView(
    name="rsi_features",
    entities=[market],
    ttl=timedelta(days=1),
    schema=[
        Feature(name="rsi_14", dtype=ValueType.FLOAT),
        Feature(name="rsi_14_trend", dtype=ValueType.INT32),  # 1 for uptrend, -1 for downtrend, 0 for neutral
        Feature(name="rsi_14_overbought", dtype=ValueType.BOOL),  # True if RSI > 70
        Feature(name="rsi_14_oversold", dtype=ValueType.BOOL),  # True if RSI < 30
    ],
    online=True,
    source=rsi_source,
    tags={"team": "trading", "category": "momentum"},
)

# ATR-14 Feature View
atr_view = FeatureView(
    name="atr_features",
    entities=[market],
    ttl=timedelta(days=1),
    schema=[
        Feature(name="atr_14", dtype=ValueType.FLOAT),
        Feature(name="atr_14_normalized", dtype=ValueType.FLOAT),  # ATR as percentage of price
        Feature(name="atr_14_percentile", dtype=ValueType.FLOAT),  # Percentile rank of current ATR vs historical
    ],
    online=True,
    source=atr_source,
    tags={"team": "trading", "category": "volatility"},
)

# VWAP Feature View
vwap_view = FeatureView(
    name="vwap_features",
    entities=[market],
    ttl=timedelta(days=1),
    schema=[
        Feature(name="vwap", dtype=ValueType.FLOAT),
        Feature(name="price_to_vwap", dtype=ValueType.FLOAT),  # Current price / VWAP
        Feature(name="vwap_trend", dtype=ValueType.INT32),  # 1 for uptrend, -1 for downtrend, 0 for neutral
    ],
    online=True,
    source=vwap_source,
    tags={"team": "trading", "category": "price"},
)

# Order Book Imbalance Feature View
ob_imbalance_view = FeatureView(
    name="order_book_features",
    entities=[market],
    ttl=timedelta(days=1),
    schema=[
        Feature(name="imbalance", dtype=ValueType.FLOAT),  # Range [-1, 1] where +1 is 100% bid pressure
        Feature(name="bid_volume", dtype=ValueType.FLOAT),
        Feature(name="ask_volume", dtype=ValueType.FLOAT),
        Feature(name="imbalance_ma", dtype=ValueType.FLOAT),  # Moving average of imbalance
    ],
    online=True,
    source=ob_imbalance_source,
    tags={"team": "trading", "category": "orderflow"},
)

# Margin Feature View
margin_view = FeatureView(
    name="margin_features",
    entities=[market],
    ttl=timedelta(days=1),
    schema=[
        Feature(name="initial_margin", dtype=ValueType.FLOAT),
        Feature(name="maintenance_margin", dtype=ValueType.FLOAT),
        Feature(name="margin_change_1d", dtype=ValueType.FLOAT),  # 1-day change in margin requirements
        Feature(name="margin_to_price_ratio", dtype=ValueType.FLOAT),  # Margin as % of contract value
    ],
    online=True,
    source=margin_source,
    tags={"team": "trading", "category": "risk"},
) 