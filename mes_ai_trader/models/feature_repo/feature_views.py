from datetime import timedelta

from feast import (
    Entity, Feature, FeatureView, 
    FileSource, S3Source, ValueType,
)
from feast.types import Float32, String

# Define entities
market = Entity(
    name="market_id",
    description="Market identifier",
    value_type=ValueType.STRING,
    join_keys=["market_id"],
)

symbol = Entity(
    name="symbol",
    description="Trading symbol identifier",
    value_type=ValueType.STRING,
    join_keys=["symbol"],
)

account = Entity(
    name="account_id", 
    description="Trading account identifier",
    value_type=ValueType.STRING,
    join_keys=["account_id"],
)

# Feature views for technical indicators
rsi_source = FileSource(
    path="sample_data/rsi_features.parquet",
    event_timestamp_column="event_timestamp",
    created_timestamp_column="created_timestamp",
)

rsi_view = FeatureView(
    name="rsi_features",
    entities=[market],
    ttl=timedelta(days=3),
    schema=[
        Feature(name="rsi_14", dtype=Float32),
        Feature(name="rsi_14_trend", dtype=Float32),
        Feature(name="rsi_14_overbought", dtype=Float32),
        Feature(name="rsi_14_oversold", dtype=Float32),
    ],
    online=True,
    source=rsi_source,
    tags={"category": "technical_indicators"},
)

atr_source = FileSource(
    path="sample_data/atr_features.parquet",
    event_timestamp_column="event_timestamp",
    created_timestamp_column="created_timestamp",
)

atr_view = FeatureView(
    name="atr_features",
    entities=[market],
    ttl=timedelta(days=3),
    schema=[
        Feature(name="atr_14", dtype=Float32),
        Feature(name="atr_14_normalized", dtype=Float32),
        Feature(name="atr_14_percentile", dtype=Float32),
    ],
    online=True,
    source=atr_source,
    tags={"category": "technical_indicators"},
)

vwap_source = FileSource(
    path="sample_data/vwap_features.parquet",
    event_timestamp_column="event_timestamp",
    created_timestamp_column="created_timestamp",
)

vwap_view = FeatureView(
    name="vwap_features",
    entities=[market],
    ttl=timedelta(days=3),
    schema=[
        Feature(name="vwap", dtype=Float32),
        Feature(name="price_to_vwap", dtype=Float32),
        Feature(name="vwap_trend", dtype=Float32),
    ],
    online=True,
    source=vwap_source,
    tags={"category": "technical_indicators"},
)

order_book_source = FileSource(
    path="sample_data/ob_imbalance_features.parquet",
    event_timestamp_column="event_timestamp",
    created_timestamp_column="created_timestamp",
)

order_book_view = FeatureView(
    name="order_book_features",
    entities=[market],
    ttl=timedelta(days=3),
    schema=[
        Feature(name="bid_volume", dtype=Float32),
        Feature(name="ask_volume", dtype=Float32),
        Feature(name="imbalance", dtype=Float32),
        Feature(name="imbalance_ma", dtype=Float32),
    ],
    online=True,
    source=order_book_source,
    tags={"category": "market_microstructure"},
)

# Feature view for margin data - using S3 source for hourly updates
margin_source = S3Source(
    bucket="mes-ai-trader-feature-store",
    path="feature_store/margin_features/",
    event_timestamp_column="event_timestamp",
    created_timestamp_column="created_timestamp"
)

margin_view = FeatureView(
    name="margin_features",
    entities=[symbol],
    ttl=timedelta(days=30),  # Margin requirements change infrequently
    schema=[
        Feature(name="initial_margin", dtype=Float32),
        Feature(name="maintenance_margin", dtype=Float32),
        Feature(name="settlement_price", dtype=Float32),
        Feature(name="effective_date", dtype=String),
        Feature(name="margin_ratio", dtype=Float32),
        Feature(name="margin_to_price_ratio", dtype=Float32),
        Feature(name="source", dtype=String),
    ],
    online=True,
    source=margin_source,
    tags={"category": "risk_parameters"},
    description="Margin requirements from CME or broker API",
)

# Feature view for account data
account_source = FileSource(
    path="sample_data/account_features.parquet",
    event_timestamp_column="event_timestamp",
)

account_view = FeatureView(
    name="account_features",
    entities=[account],
    ttl=timedelta(days=1),
    schema=[
        Feature(name="equity", dtype=Float32),
        Feature(name="margin_used", dtype=Float32),
        Feature(name="leverage", dtype=Float32),
        Feature(name="risk_limit_pct", dtype=Float32),
    ],
    online=True,
    source=account_source,
    tags={"category": "account"},
) 