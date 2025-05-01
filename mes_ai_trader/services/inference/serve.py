#!/usr/bin/env python
"""
FastAPI Inference Service for MES AI Trader

This service provides REST API endpoints to get predictions from
LightGBM, Temporal Fusion Transformer, and PPO models.
"""
import os
import logging
import time
import sys
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from functools import lru_cache
import asyncio
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query, Request, Security
from fastapi.security.api_key import APIKeyHeader, APIKeyQuery
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from feast import FeatureStore
from stable_baselines3 import PPO
from prometheus_client import Counter, Histogram, Summary, start_http_server
import opentelemetry.trace as trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import config
from utils.circuit_breaker import CircuitBreaker, CircuitOpenError
from utils.model_monitor import ModelMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize tracing if enabled
if config.ENABLE_TRACING:
    trace.set_tracer_provider(TracerProvider())
    jaeger_exporter = JaegerExporter(
        agent_host_name=config.JAEGER_AGENT_HOST,
        agent_port=config.JAEGER_AGENT_PORT,
    )
    trace.get_tracer_provider().add_span_processor(
        BatchSpanProcessor(jaeger_exporter)
    )
    
# Initialize FastAPI app
app = FastAPI(
    title="MES AI Trader Inference API",
    description="API for making predictions using LightGBM, TFT, and PPO models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.API_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup tracing
if config.ENABLE_TRACING:
    FastAPIInstrumentor.instrument_app(app, tracer_provider=trace.get_tracer_provider())

# Setup security
API_KEY_NAME = config.API_KEY_HEADER
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
api_key_query = APIKeyQuery(name=API_KEY_NAME, auto_error=False)

# Prometheus metrics
if config.ENABLE_METRICS:
    # Track predictions
    PREDICTION_COUNTER = Counter(
        "predictions_total", 
        "Total number of predictions",
        ["model", "status"]
    )
    
    # Track prediction latency
    PREDICTION_LATENCY = Histogram(
        "prediction_latency_seconds",
        "Time spent making predictions",
        ["model"]
    )
    
    # Track cache hit rate
    CACHE_HIT_COUNTER = Counter(
        "cache_hits_total",
        "Total number of cache hits",
        ["model"]
    )
    
    CACHE_MISS_COUNTER = Counter(
        "cache_misses_total",
        "Total number of cache misses",
        ["model"]
    )
    
    # Track feature fetch latency
    FEATURE_FETCH_TIME = Histogram(
        "feature_fetch_seconds",
        "Time spent fetching features",
        ["source"]
    )

# Global model storage
lgbm_model = None
tft_model = None
ppo_model = None
model_load_time = {}

# Cache for predictions
prediction_cache = {
    "lgbm": {"data": None, "timestamp": None},
    "tft": {"data": None, "timestamp": None},
    "ppo": {"data": None, "timestamp": None}
}

# Model monitors
model_monitors = {}

# Response models
class PredictionResponse(BaseModel):
    prediction: Any
    std_dev: float
    timestamp: str
    features_used: List[str]
    model_info: Dict[str, Any]
    cached: bool = False

# Initialize circuit breakers
feature_store_cb = CircuitBreaker(
    name="feature_store",
    fallback_function=lambda: {
        "error": "Feature store unavailable, using default features"
    }
)

# Security Functions
def get_api_key(
    api_key_header: str = Security(api_key_header),
    api_key_query: str = Security(api_key_query),
):
    """Get API key from header or query parameters."""
    if not config.API_AUTH_ENABLED:
        return True
        
    if api_key_header:
        return api_key_header
    if api_key_query:
        return api_key_query
    
    raise HTTPException(
        status_code=403, detail="Could not validate credentials"
    )

def verify_api_key(api_key: str = Depends(get_api_key)):
    """Verify that the API key is valid."""
    if not config.API_AUTH_ENABLED or api_key is True:
        return True
        
    # In production, this would validate against a database of keys
    # or an external authentication service
    valid_api_keys = ["test-key", "production-key"]  # Replace with actual key validation
    
    if api_key not in valid_api_keys:
        raise HTTPException(
            status_code=403, detail="Invalid API key"
        )
    return True

# Load models at startup to avoid loading on each request
@app.on_event("startup")
async def setup():
    """Initialize models and services at startup with retry mechanism."""
    global lgbm_model, tft_model, ppo_model, model_load_time, model_monitors
    
    # Start metrics server if enabled
    if config.ENABLE_METRICS:
        start_http_server(config.METRICS_PORT)
        logger.info(f"Metrics server started on port {config.METRICS_PORT}")
    
    # Use background tasks to load models concurrently
    lgbm_task = asyncio.create_task(load_lgbm_model())
    tft_task = asyncio.create_task(load_tft_model())
    ppo_task = asyncio.create_task(load_ppo_model())
    
    # Wait for all tasks to complete
    await asyncio.gather(lgbm_task, tft_task, ppo_task, return_exceptions=True)
    
    # Initialize model monitors
    for model_name in ["lgbm", "tft", "ppo"]:
        if model_name == "lgbm" and lgbm_model:
            feature_names = lgbm_model.feature_name()
            model_monitors[model_name] = ModelMonitor(
                model_name=model_name,
                feature_names=feature_names
            )
        elif model_name == "tft" and tft_model:
            # Approximate feature names for TFT
            feature_names = [f"feature_{i}" for i in range(50)]  # Adjust based on actual model
            model_monitors[model_name] = ModelMonitor(
                model_name=model_name,
                feature_names=feature_names
            )
        elif model_name == "ppo" and ppo_model:
            # Approximate feature names for PPO
            feature_names = [f"feature_{i}" for i in range(50)]  # Adjust based on actual model
            model_monitors[model_name] = ModelMonitor(
                model_name=model_name,
                feature_names=feature_names
            )
    
    # Log results
    models_loaded = [
        m for m, loaded in 
        [("LGBM", lgbm_model is not None), 
         ("TFT", tft_model is not None), 
         ("PPO", ppo_model is not None)] 
        if loaded
    ]
    
    if models_loaded:
        logger.info(f"Successfully loaded models: {', '.join(models_loaded)}")
    else:
        logger.warning("Failed to load any models at startup")

@app.on_event("shutdown")
async def cleanup():
    """Clean up resources at shutdown."""
    # Save any pending model monitor data
    for monitor in model_monitors.values():
        monitor.compute_window_metrics()

async def load_lgbm_model():
    """Load LightGBM model with retries."""
    global lgbm_model, model_load_time
    
    for attempt in range(config.MODEL_LOAD_RETRIES):
        try:
            logger.info(f"Loading LightGBM model from {config.LGBM_MODEL_PATH} (attempt {attempt+1}/{config.MODEL_LOAD_RETRIES})")
            lgbm_model = lgb.Booster(model_file=config.LGBM_MODEL_PATH)
            model_load_time["lgbm"] = datetime.now()
            logger.info("LightGBM model loaded successfully")
            return
        except Exception as e:
            logger.error(f"Error loading LightGBM model (attempt {attempt+1}): {e}")
            if attempt < config.MODEL_LOAD_RETRIES - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    logger.error(f"Failed to load LightGBM model after {config.MODEL_LOAD_RETRIES} attempts")

async def load_tft_model():
    """Load TFT model with retries."""
    global tft_model, model_load_time
    
    for attempt in range(config.MODEL_LOAD_RETRIES):
        try:
            logger.info(f"Loading TFT model from {config.TFT_MODEL_PATH} (attempt {attempt+1}/{config.MODEL_LOAD_RETRIES})")
            tft_model = torch.load(config.TFT_MODEL_PATH, map_location=torch.device('cpu'))
            model_load_time["tft"] = datetime.now()
            logger.info("TFT model loaded successfully")
            return
        except Exception as e:
            logger.error(f"Error loading TFT model (attempt {attempt+1}): {e}")
            if attempt < config.MODEL_LOAD_RETRIES - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    logger.error(f"Failed to load TFT model after {config.MODEL_LOAD_RETRIES} attempts")

async def load_ppo_model():
    """Load PPO model with retries."""
    global ppo_model, model_load_time
    
    for attempt in range(config.MODEL_LOAD_RETRIES):
        try:
            logger.info(f"Loading PPO model from {config.PPO_MODEL_PATH} (attempt {attempt+1}/{config.MODEL_LOAD_RETRIES})")
            ppo_model = PPO.load(config.PPO_MODEL_PATH)
            model_load_time["ppo"] = datetime.now()
            logger.info("PPO model loaded successfully")
            return
        except Exception as e:
            logger.error(f"Error loading PPO model (attempt {attempt+1}): {e}")
            if attempt < config.MODEL_LOAD_RETRIES - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    logger.error(f"Failed to load PPO model after {config.MODEL_LOAD_RETRIES} attempts")

def is_cache_valid(model_name):
    """Check if the cache for a given model is valid."""
    cache_entry = prediction_cache.get(model_name)
    if not cache_entry or not cache_entry["timestamp"]:
        return False
    
    # Check if cache is still within TTL
    cache_age = (datetime.now() - cache_entry["timestamp"]).total_seconds()
    return cache_age < config.CACHE_TTL_SECONDS

@lru_cache(maxsize=1)
def get_feature_store():
    """Get and cache the feature store connection."""
    try:
        return FeatureStore(repo_path=config.FEATURE_REPO_PATH)
    except Exception as e:
        logger.error(f"Error initializing feature store: {e}")
        raise

@feature_store_cb
async def get_online_features():
    """
    Fetch the latest features from the online feature store with circuit breaker.
    
    Returns:
        pd.DataFrame: Latest feature values
    """
    start_time = time.time()
    
    try:
        feature_store = get_feature_store()
        
        # Define features to fetch
        features = [
            "rsi_features:rsi_14",
            "rsi_features:rsi_14_trend",
            "atr_features:atr_14",
            "atr_features:atr_14_normalized",
            "vwap_features:vwap",
            "vwap_features:price_to_vwap",
            "vwap_features:vwap_trend",
            "order_book_features:imbalance",
            "order_book_features:bid_volume",
            "order_book_features:ask_volume",
            "order_book_features:imbalance_ma",
            "margin_features:initial_margin",
            "margin_features:margin_to_price_ratio"
        ]
        
        # Get latest feature values
        online_features = feature_store.get_online_features(
            entity_rows=[{"market_id": "MES"}],
            features=features
        ).to_dict()
        
        # Convert to DataFrame
        feature_df = pd.DataFrame(online_features, index=[0])
        
        # Handle NaN values
        feature_df = feature_df.fillna(method='ffill').fillna(0)
        
        # Record metrics
        if config.ENABLE_METRICS:
            FEATURE_FETCH_TIME.labels(source="online").observe(time.time() - start_time)
        
        return feature_df
    except Exception as e:
        logger.error(f"Error fetching online features: {e}")
        
        # Record metrics
        if config.ENABLE_METRICS:
            FEATURE_FETCH_TIME.labels(source="online").observe(time.time() - start_time)
        
        # Fallback to default features
        return get_default_features()

def get_default_features():
    """
    Generate default features when feature store is unavailable.
    
    Returns:
        pd.DataFrame: Default feature values
    """
    # Create a dataframe with default values
    default_features = {
        "rsi_features:rsi_14": [50.0],  # Neutral RSI
        "rsi_features:rsi_14_trend": [0.0],  # No trend
        "atr_features:atr_14": [2.5],  # Average ATR
        "atr_features:atr_14_normalized": [0.5],  # Mid-range
        "vwap_features:vwap": [4500.0],  # Typical MES price
        "vwap_features:price_to_vwap": [1.0],  # At VWAP
        "vwap_features:vwap_trend": [0.0],  # No trend
        "order_book_features:imbalance": [0.0],  # Balanced
        "order_book_features:bid_volume": [100],  # Average volume
        "order_book_features:ask_volume": [100],  # Average volume
        "order_book_features:imbalance_ma": [0.0],  # Balanced
        "margin_features:initial_margin": [5000],  # Default margin
        "margin_features:margin_to_price_ratio": [1.1]  # Default ratio
    }
    
    return pd.DataFrame(default_features)

# Middleware for request timing and metrics
@app.middleware("http")
async def add_metrics(request: Request, call_next):
    """Add metrics middleware for tracking request duration and status."""
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    return response

@app.get("/health")
async def health_check():
    """Health check endpoint that also reports model load status."""
    models_status = {
        "lgbm": {"loaded": lgbm_model is not None, "load_time": model_load_time.get("lgbm")},
        "tft": {"loaded": tft_model is not None, "load_time": model_load_time.get("tft")},
        "ppo": {"loaded": ppo_model is not None, "load_time": model_load_time.get("ppo")}
    }
    
    # Check feature store
    try:
        feature_store = get_feature_store()
        feature_store_status = "ok"
    except Exception:
        feature_store_status = "error"
    
    # Add circuit breaker status
    circuit_breakers = {
        "feature_store": feature_store_cb.state.value
    }
    
    all_loaded = all(status["loaded"] for status in models_status.values())
    
    # Check if any circuit breakers are open
    any_circuit_open = any(state == "open" for state in circuit_breakers.values())
    
    # Determine overall status
    if all_loaded and not any_circuit_open and feature_store_status == "ok":
        status = "healthy"
    elif all_loaded:
        status = "degraded"
    else:
        status = "unhealthy"
    
    return {
        "status": status,
        "models": models_status,
        "feature_store": feature_store_status,
        "circuit_breakers": circuit_breakers,
        "cache_ttl_seconds": config.CACHE_TTL_SECONDS
    }

@app.get("/metrics/drift")
async def get_drift_metrics(
    model: str = Query(..., description="Model name to get drift metrics for"),
    authenticated: bool = Depends(verify_api_key)
):
    """Get model drift metrics."""
    if model not in model_monitors:
        raise HTTPException(status_code=404, detail=f"Model monitor not found for {model}")
    
    monitor = model_monitors[model]
    
    # Get latest metrics
    latest_metrics = monitor.metrics_df.sort_values("timestamp").tail(10)
    
    if len(latest_metrics) == 0:
        return {"model": model, "metrics": [], "message": "No metrics available yet"}
    
    return {
        "model": model,
        "metrics": latest_metrics.to_dict(orient="records"),
        "drift_status": latest_metrics["drift_status"].iloc[-1] if "drift_status" in latest_metrics.columns else "unknown",
        "retraining_needed": monitor.retraining_needed,
        "retraining_reason": monitor.retraining_reason
    }

@app.get("/reload_models")
async def reload_models(
    background_tasks: BackgroundTasks,
    authenticated: bool = Depends(verify_api_key)
):
    """
    Endpoint to trigger model reloading in the background.
    Useful for refreshing models after updates without service restart.
    """
    logger.info("Model reload requested")
    background_tasks.add_task(load_models)
    return {"message": "Model reload initiated in background"}

@app.get("/lgbm", response_model=PredictionResponse)
async def lgbm_predict(
    force_refresh: bool = Query(False, description="Force refresh predictions instead of using cache"),
    authenticated: bool = Depends(verify_api_key)
):
    """
    Get prediction from LightGBM model.
    
    Returns:
        PredictionResponse: Prediction and metadata
    """
    model_name = "lgbm"
    
    # Start prediction timer
    start_time = time.time()
    
    try:
        global lgbm_model
        
        # Check if model is loaded
        if not lgbm_model:
            await load_lgbm_model()
            if not lgbm_model:
                raise HTTPException(status_code=503, detail="LightGBM model not available")
        
        # Check cache first if not forcing refresh
        if not force_refresh and is_cache_valid(model_name):
            cached_response = prediction_cache[model_name]["data"]
            cached_response.cached = True
            logger.info("Returning cached LightGBM prediction")
            
            # Record metrics
            if config.ENABLE_METRICS:
                CACHE_HIT_COUNTER.labels(model=model_name).inc()
                PREDICTION_COUNTER.labels(model=model_name, status="success").inc()
            
            return cached_response
        
        # Record cache miss
        if config.ENABLE_METRICS:
            CACHE_MISS_COUNTER.labels(model=model_name).inc()
        
        # Get features
        features = await get_online_features()
        feature_names = features.columns.tolist()
        
        # Make prediction
        # Handle NaN values before prediction
        features = features.fillna(0)  # Replace NaNs with zeros
        
        # For classification we get probabilities
        raw_pred = lgbm_model.predict(features)
        
        # For binary classification, format the prediction
        if len(raw_pred.shape) == 1 or raw_pred.shape[1] == 1:
            # Single value prediction or binary class with single output
            prediction = float(raw_pred[0])
            std_dev = 0.0  # Not directly available for single predictions
        else:
            # Multi-class with probabilities
            prediction = raw_pred[0].tolist()
            std_dev = float(np.std(raw_pred[0]))
        
        # Measure prediction time
        prediction_time = time.time() - start_time
        
        # Record metrics
        if config.ENABLE_METRICS:
            PREDICTION_LATENCY.labels(model=model_name).observe(prediction_time)
            PREDICTION_COUNTER.labels(model=model_name, status="success").inc()
        
        # Create response
        response = PredictionResponse(
            prediction=prediction,
            std_dev=std_dev,
            timestamp=datetime.now().isoformat(),
            features_used=feature_names,
            model_info={
                "type": "LightGBM",
                "path": config.LGBM_MODEL_PATH,
                "num_features": len(feature_names),
                "prediction_time_ms": round(prediction_time * 1000, 2)
            },
            cached=False
        )
        
        # Update cache
        prediction_cache[model_name] = {
            "data": response,
            "timestamp": datetime.now()
        }
        
        # Record prediction in model monitor
        if model_name in model_monitors:
            model_monitors[model_name].record_prediction(
                features=features,
                prediction=prediction
            )
        
        return response
    except CircuitOpenError as e:
        logger.error(f"Circuit breaker open for LightGBM prediction: {e}")
        
        if config.ENABLE_METRICS:
            PREDICTION_COUNTER.labels(model=model_name, status="circuit_open").inc()
        
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Error making LightGBM prediction: {e}")
        
        if config.ENABLE_METRICS:
            PREDICTION_COUNTER.labels(model=model_name, status="error").inc()
        
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tft", response_model=PredictionResponse)
async def tft_predict(force_refresh: bool = Query(False, description="Force refresh predictions instead of using cache")):
    """
    Get prediction from Temporal Fusion Transformer model.
    
    Returns:
        PredictionResponse: Prediction with quantiles and metadata
    """
    global tft_model
    
    # Check if model is loaded
    if not tft_model:
        await load_tft_model()
        if not tft_model:
            raise HTTPException(status_code=503, detail="TFT model not available")
    
    # Check cache first if not forcing refresh
    if not force_refresh and is_cache_valid("tft"):
        cached_response = prediction_cache["tft"]["data"]
        cached_response.cached = True
        logger.info("Returning cached TFT prediction")
        return cached_response
    
    try:
        start_time = time.time()
        
        # Get features
        features = get_online_features()
        feature_names = features.columns.tolist()
        
        # Prepare input for TFT model
        # TFT typically expects a time-series format with specific structure
        # This is a simplified version and should be adjusted based on actual TFT implementation
        
        # Set model to evaluation mode
        tft_model.eval()
        
        # Handle NaN values and convert to correct format
        features = features.fillna(0)
        
        # Convert features to tensor format expected by the model
        input_tensor = torch.tensor(features.values, dtype=torch.float32)
        
        # Make prediction
        with torch.no_grad():
            output = tft_model(input_tensor)
        
        # TFT typically outputs quantile predictions
        # Assuming output has shape [batch_size, horizon, quantiles]
        quantiles = [0.1, 0.5, 0.9]  # 10%, 50% (median), 90% percentiles
        
        # Format the prediction for response
        # Using median as the main prediction
        prediction = {
            "quantiles": quantiles,
            "values": output[0].numpy().tolist()  # First batch, all horizons and quantiles
        }
        
        # Calculate std_dev using the quantiles
        # Approximation: (q90 - q10) / 2.56 is roughly equivalent to standard deviation
        # if the distribution is normal
        q10 = output[0, 0, 0].item()  # First batch, first horizon, first quantile
        q90 = output[0, 0, 2].item()  # First batch, first horizon, last quantile
        std_dev = float((q90 - q10) / 2.56)
        
        # Measure prediction time
        prediction_time = time.time() - start_time
        
        # Create response
        response = PredictionResponse(
            prediction=prediction,
            std_dev=std_dev,
            timestamp=datetime.now().isoformat(),
            features_used=feature_names,
            model_info={
                "type": "Temporal Fusion Transformer",
                "path": config.TFT_MODEL_PATH,
                "quantiles": quantiles,
                "prediction_time_ms": round(prediction_time * 1000, 2)
            },
            cached=False
        )
        
        # Update cache
        prediction_cache["tft"] = {
            "data": response,
            "timestamp": datetime.now()
        }
        
        return response
    except Exception as e:
        logger.error(f"Error making TFT prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ppo", response_model=PredictionResponse)
async def ppo_predict(force_refresh: bool = Query(False, description="Force refresh predictions instead of using cache")):
    """
    Get prediction from PPO reinforcement learning model.
    
    Returns:
        PredictionResponse: Action recommendation and metadata
    """
    global ppo_model
    
    # Check if model is loaded
    if not ppo_model:
        await load_ppo_model()
        if not ppo_model:
            raise HTTPException(status_code=503, detail="PPO model not available")
    
    # Check cache first if not forcing refresh
    if not force_refresh and is_cache_valid("ppo"):
        cached_response = prediction_cache["ppo"]["data"]
        cached_response.cached = True
        logger.info("Returning cached PPO prediction")
        return cached_response
    
    try:
        start_time = time.time()
        
        # Get features
        features = get_online_features()
        feature_names = features.columns.tolist()
        
        # For PPO, we need to format the observation as expected by the model
        # Based on the MESEnv in train_ppo.py
        
        # Simplified observation creation - in production this would need to
        # include all the state information required by the model
        window_size = 60  # Same as in train_ppo.py
        
        # Create synthetic price for this example (would come from actual data in production)
        current_price = 4500.0  # Example price
        
        # Create a simplified observation vector
        # In production, this would need to match exactly what the model expects
        observation = np.zeros((window_size, len(feature_names) + 5), dtype=np.float32)
        
        # Fill in available features
        # Handle NaN values
        feature_values = features.fillna(0).values[0]
        for i in range(window_size):
            observation[i, :len(feature_names)] = feature_values
        
        # Add price, position, equity, unrealized PnL, balance placeholders
        # These would come from the actual trading state in production
        observation[:, len(feature_names)] = current_price
        
        # Convert to tensor and get model prediction
        obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        
        # Get action from PPO model with error handling
        try:
            action, _ = ppo_model.predict(observation, deterministic=True)
        except Exception as e:
            logger.error(f"Error in PPO prediction: {e}")
            # Fallback to a safe action (HOLD)
            action = 0
        
        # Get action probabilities for uncertainty estimation
        try:
            action_probs = ppo_model.policy.get_distribution(obs_tensor).distribution.probs.detach().numpy()
        except Exception as e:
            logger.error(f"Error getting action probabilities: {e}")
            # Fallback to uniform distribution
            action_probs = np.ones((1, 3)) / 3  # Assuming 3 actions
        
        # Map action to trading decision
        action_mapping = {0: "HOLD", 1: "BUY", 2: "SELL"}
        decision = action_mapping.get(int(action), "HOLD")  # Default to HOLD if action is invalid
        
        # Calculate standard deviation of action probabilities as uncertainty measure
        std_dev = float(np.std(action_probs))
        
        # Measure prediction time
        prediction_time = time.time() - start_time
        
        # Prepare response
        prediction = {
            "action": int(action),
            "decision": decision,
            "action_probabilities": action_probs[0].tolist()
        }
        
        response = PredictionResponse(
            prediction=prediction,
            std_dev=std_dev,
            timestamp=datetime.now().isoformat(),
            features_used=feature_names,
            model_info={
                "type": "PPO",
                "path": config.PPO_MODEL_PATH,
                "action_mapping": action_mapping,
                "prediction_time_ms": round(prediction_time * 1000, 2)
            },
            cached=False
        )
        
        # Update cache
        prediction_cache["ppo"] = {
            "data": response,
            "timestamp": datetime.now()
        }
        
        return response
    except Exception as e:
        logger.error(f"Error making PPO prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 