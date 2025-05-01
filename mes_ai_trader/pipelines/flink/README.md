# MES Trader Flink Jobs

This directory contains Apache Flink jobs for processing market data streams.

## OHLCV Job

The `ohlcv.scala` job reads tick data from the `mes-raw-ticks` Kinesis stream, computes 1-second OHLCV (Open, High, Low, Close, Volume) bars with order book imbalance metrics, and writes the results to the `mes-bars-1s` Kinesis stream.

### Features

- 1-second tumbling windows for OHLCV bar computation
- Order book imbalance calculation to measure buying/selling pressure
- Efficient JSON serialization/deserialization
- Checkpoint-based fault tolerance
- Exactly-once processing semantics

### Data Models

#### Input: MesTick

```json
{
  "symbol": "MES",
  "price": 4500.75,
  "size": 1.0,
  "timestamp": 1625097600000,
  "type": "trade",
  "raw": {
    "bp": 4500.50,  // Bid price (if available)
    "ap": 4501.00,  // Ask price (if available)
    "bs": 5.0,      // Bid size (if available)
    "as": 3.0       // Ask size (if available)
  }
}
```

#### Output: OHLCVBar

```json
{
  "symbol": "MES",
  "timestamp": 1625097600000,
  "open": 4500.25,
  "high": 4501.00,
  "low": 4500.00,
  "close": 4500.75,
  "volume": 12.0,
  "numTicks": 8,
  "imbalance": 0.25,  // Range [-1, 1] where positive values indicate bullish pressure
  "bidVolume": 35.0,
  "askVolume": 21.0
}
```

### Building

Build the job using SBT:

```bash
cd mes_ai_trader/pipelines/flink
sbt clean assembly
```

This will produce a JAR file in the `target/scala-2.12/` directory.

### Running

Deploy the JAR to your Flink cluster:

```bash
flink run -c com.mesaitrader.flink.OHLCVJob \
  target/scala-2.12/mes-trader-flink-jobs-0.1.0.jar
```

### Configuration

The job uses the following configuration:

- AWS Region: us-east-1 (can be modified in the code)
- Input Kinesis Stream: mes-raw-ticks
- Output Kinesis Stream: mes-bars-1s
- Window Size: 1 second
