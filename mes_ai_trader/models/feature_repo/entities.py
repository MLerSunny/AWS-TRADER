from feast import Entity, ValueType

# Define a market entity that will be used to organize features
market = Entity(
    name="market",
    description="Market identifier (e.g., MES for Micro E-mini S&P 500)",
    value_type=ValueType.STRING,
    join_keys=["market_id"],
)

# Define a timestamp entity that will be used to organize features over time
timestamp = Entity(
    name="event_timestamp",
    description="Timestamp for market events",
    value_type=ValueType.UNIX_TIMESTAMP,
    join_keys=["event_timestamp"],
) 