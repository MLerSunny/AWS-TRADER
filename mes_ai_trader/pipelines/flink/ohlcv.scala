package com.mesaitrader.flink

import org.apache.flink.api.common.serialization.{DeserializationSchema, SerializationSchema}
import org.apache.flink.api.common.typeinfo.TypeInformation
import org.apache.flink.streaming.api.scala._
import org.apache.flink.streaming.api.windowing.assigners.TumblingProcessingTimeWindows
import org.apache.flink.streaming.api.windowing.time.Time
import org.apache.flink.streaming.connectors.kinesis.FlinkKinesisConsumer
import org.apache.flink.streaming.connectors.kinesis.FlinkKinesisProducer
import org.apache.flink.streaming.connectors.kinesis.config.{AWSConfigConstants, ConsumerConfigConstants}

import java.util.Properties
import java.nio.charset.StandardCharsets
import java.sql.Timestamp
import scala.collection.JavaConverters._

import com.fasterxml.jackson.databind.{DeserializationFeature, ObjectMapper}
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import com.fasterxml.jackson.annotation.JsonProperty
import com.fasterxml.jackson.databind.node.ObjectNode

/**
 * Data model for a Micro ES tick from Tradovate
 */
case class MesTick(
  @JsonProperty("symbol") symbol: String,
  @JsonProperty("price") price: Double,
  @JsonProperty("size") size: Double,
  @JsonProperty("timestamp") timestamp: Long,
  @JsonProperty("type") tickType: String,
  @JsonProperty("raw") raw: Map[String, Any]
) {
  // Extract bid/ask information from raw data if available
  def getBidPrice: Option[Double] = raw.get("bp").map(_.toString.toDouble)
  def getAskPrice: Option[Double] = raw.get("ap").map(_.toString.toDouble)
  def getBidSize: Option[Double] = raw.get("bs").map(_.toString.toDouble)
  def getAskSize: Option[Double] = raw.get("as").map(_.toString.toDouble)
}

/**
 * OHLCV + Order book imbalance data model
 */
case class OHLCVBar(
  @JsonProperty("symbol") symbol: String,
  @JsonProperty("timestamp") timestamp: Long,
  @JsonProperty("open") open: Double,
  @JsonProperty("high") high: Double,
  @JsonProperty("low") low: Double,
  @JsonProperty("close") close: Double,
  @JsonProperty("volume") volume: Double,
  @JsonProperty("numTicks") numTicks: Int,
  @JsonProperty("imbalance") imbalance: Double,
  @JsonProperty("bidVolume") bidVolume: Double,
  @JsonProperty("askVolume") askVolume: Double
)

/**
 * JSON Serialization/Deserialization for Kinesis
 */
class JsonDeserializationSchema extends DeserializationSchema[MesTick] {
  private val objectMapper = new ObjectMapper()
    .registerModule(DefaultScalaModule)
    .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false)
    
  override def deserialize(message: Array[Byte]): MesTick = {
    objectMapper.readValue(message, classOf[MesTick])
  }
  
  override def isEndOfStream(nextElement: MesTick): Boolean = false
  
  override def getProducedType: TypeInformation[MesTick] = 
    TypeInformation.of(classOf[MesTick])
}

class JsonSerializationSchema extends SerializationSchema[OHLCVBar] {
  private val objectMapper = new ObjectMapper()
    .registerModule(DefaultScalaModule)
    
  override def serialize(element: OHLCVBar): Array[Byte] = {
    objectMapper.writeValueAsBytes(element)
  }
}

/**
 * Main Flink job for computing OHLCV + order book imbalance from tick data
 */
object OHLCVJob {
  
  def main(args: Array[String]): Unit = {
    // Set up the streaming execution environment
    val env = StreamExecutionEnvironment.getExecutionEnvironment
    
    // Configure for checkpointing
    env.enableCheckpointing(60000) // Checkpoint every 60 seconds
    
    // Configure Kinesis consumer
    val inputProperties = new Properties()
    inputProperties.setProperty(AWSConfigConstants.AWS_REGION, "us-east-1")
    inputProperties.setProperty(ConsumerConfigConstants.STREAM_INITIAL_POSITION, "LATEST")
    
    // Configure Kinesis producer
    val outputProperties = new Properties()
    outputProperties.setProperty(AWSConfigConstants.AWS_REGION, "us-east-1")
    
    // Source: Kinesis consumer reading from mes-raw-ticks
    val consumer = new FlinkKinesisConsumer[MesTick](
      "mes-raw-ticks",
      new JsonDeserializationSchema(),
      inputProperties
    )
    
    // Sink: Kinesis producer writing to mes-bars-1s
    val producer = new FlinkKinesisProducer[OHLCVBar](
      new JsonSerializationSchema(),
      outputProperties
    )
    producer.setDefaultStream("mes-bars-1s")
    producer.setDefaultPartition("0") // Use a single partition for simplicity
    
    // Data processing
    val barStream = env
      .addSource(consumer)
      .name("Kinesis Tick Source")
      .keyBy(_.symbol) // Group by symbol (MES)
      .window(TumblingProcessingTimeWindows.of(Time.seconds(1))) // 1-second window
      .aggregate(new OHLCVAggregator())
      .name("OHLCV Aggregation")
    
    // Write the result to Kinesis
    barStream
      .addSink(producer)
      .name("Kinesis Bar Sink")
    
    // Execute the Flink job
    env.execute("MES OHLCV 1-Second Bar Generation")
  }
}

/**
 * Aggregation function to compute OHLCV + order book imbalance
 */
class OHLCVAggregator extends AggregateFunction[MesTick, OHLCVAccumulator, OHLCVBar] {
  // Initialize an empty accumulator
  override def createAccumulator(): OHLCVAccumulator = OHLCVAccumulator()
  
  // Add a tick to the accumulator
  override def add(tick: MesTick, acc: OHLCVAccumulator): OHLCVAccumulator = {
    // First tick in the window
    if (acc.count == 0) {
      acc.open = tick.price
      acc.high = tick.price
      acc.low = tick.price
      acc.symbol = tick.symbol
      acc.timestamp = tick.timestamp
    } else {
      // Update high and low
      acc.high = Math.max(acc.high, tick.price)
      acc.low = Math.min(acc.low, tick.price)
      // Keep the latest timestamp in the window
      acc.timestamp = Math.max(acc.timestamp, tick.timestamp)
    }
    
    // Always update close price with the latest tick
    acc.close = tick.price
    
    // Add to volume
    acc.volume += tick.size
    
    // Count ticks
    acc.count += 1
    
    // Update bid/ask accumulation for order book imbalance
    if (tick.getBidSize.isDefined && tick.getBidPrice.isDefined) {
      acc.bidVolume += tick.getBidSize.get
    }
    
    if (tick.getAskSize.isDefined && tick.getAskPrice.isDefined) {
      acc.askVolume += tick.getAskSize.get
    }
    
    acc
  }
  
  // Extract the result from the accumulator
  override def getResult(acc: OHLCVAccumulator): OHLCVBar = {
    // Calculate order book imbalance (difference between bid and ask volumes)
    val totalVolume = acc.bidVolume + acc.askVolume
    val imbalance = if (totalVolume > 0) {
      // Normalize to range [-1, 1] where:
      // -1 = 100% ask pressure (bearish)
      // +1 = 100% bid pressure (bullish)
      (acc.bidVolume - acc.askVolume) / totalVolume
    } else 0.0
    
    OHLCVBar(
      symbol = acc.symbol,
      timestamp = acc.timestamp,
      open = acc.open,
      high = acc.high,
      low = acc.low,
      close = acc.close,
      volume = acc.volume,
      numTicks = acc.count,
      imbalance = imbalance,
      bidVolume = acc.bidVolume,
      askVolume = acc.askVolume
    )
  }
  
  // Merge two accumulators
  override def merge(a: OHLCVAccumulator, b: OHLCVAccumulator): OHLCVAccumulator = {
    val merged = OHLCVAccumulator()
    
    // Use values from the non-empty accumulator if one is empty
    if (a.count == 0) return b
    if (b.count == 0) return a
    
    // Merge OHLCV data
    merged.symbol = a.symbol // Assuming both have the same symbol
    merged.open = a.open // Use the first accumulator's open
    merged.high = Math.max(a.high, b.high)
    merged.low = Math.min(a.low, b.low)
    merged.close = b.close // Use the second accumulator's close (assuming chronological order)
    merged.volume = a.volume + b.volume
    merged.count = a.count + b.count
    
    // Merge order book data
    merged.bidVolume = a.bidVolume + b.bidVolume
    merged.askVolume = a.askVolume + b.askVolume
    
    // Use the latest timestamp
    merged.timestamp = Math.max(a.timestamp, b.timestamp)
    
    merged
  }
}

/**
 * Accumulator for OHLCV calculation
 */
case class OHLCVAccumulator(
  var symbol: String = "",
  var timestamp: Long = 0L,
  var open: Double = 0.0,
  var high: Double = Double.MinValue,
  var low: Double = Double.MaxValue,
  var close: Double = 0.0,
  var volume: Double = 0.0,
  var count: Int = 0,
  var bidVolume: Double = 0.0,
  var askVolume: Double = 0.0
) 