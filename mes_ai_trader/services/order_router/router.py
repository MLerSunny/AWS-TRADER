"""
Order Router Service for MES AI Trader

This service routes order execution requests to Tradovate's REST API
and implements OCO (One-Cancels-Other) orders with trailing stops.
"""
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any, Union

import boto3
import requests
import psycopg2
from fastapi import FastAPI, HTTPException, Depends, Security, Request, status
from fastapi.security.api_key import APIKeyHeader, APIKeyQuery
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field, validator
import asyncio
import aiopg
from prometheus_client import Counter, Histogram, start_http_server
import time
import jwt
from pathlib import Path

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize AWS clients
secrets_client = boto3.client("secretsmanager", region_name=config.AWS_REGION)
ssm_client = boto3.client("ssm", region_name=config.AWS_REGION)

# Initialize FastAPI app
app = FastAPI(
    title="MES AI Trader Order Router",
    description="""
    Service for routing orders to Tradovate's REST API.
    
    ## Features
    
    * Execute market, limit, and stop orders
    * Implement OCO (One-Cancels-Other) orders
    * Create trailing stops based on ATR
    * Record order history in database
    
    ## Authentication
    
    API calls require an API key that can be provided via:
    * Header: `X-API-Key`
    * Query parameter: `api_key`
    """,
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

# Setup security
API_KEY_NAME = config.API_KEY_HEADER
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
api_key_query = APIKeyQuery(name=API_KEY_NAME, auto_error=False)

# Prometheus metrics
if config.ENABLE_METRICS:
    # Track order executions
    ORDER_COUNTER = Counter(
        "order_executions_total", 
        "Total number of order executions",
        ["action", "symbol", "status"]
    )
    
    # Track API latency
    REQUEST_TIME = Histogram(
        "request_processing_seconds",
        "Time spent processing request",
        ["endpoint"]
    )

# Global database connection pool
db_pool = None

# Initialize DB pool at startup
@app.on_event("startup")
async def setup():
    """Initialize services at application startup."""
    global db_pool
    
    # Start metrics server if enabled
    if config.ENABLE_METRICS:
        start_http_server(config.METRICS_PORT)
        logger.info(f"Metrics server started on port {config.METRICS_PORT}")
    
    # Initialize DB pool
    try:
        dsn = f"host={config.DB_HOST} port={config.DB_PORT} dbname={config.DB_NAME} user={config.DB_USER} password={config.DB_PASSWORD}"
        db_pool = await aiopg.create_pool(
            dsn, 
            minsize=config.DB_MIN_CONNECTIONS, 
            maxsize=config.DB_MAX_CONNECTIONS
        )
        logger.info("Database connection pool initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database connection pool: {e}")
        # Don't raise an exception here to allow the app to start even with DB issues

# Clean up DB pool at shutdown
@app.on_event("shutdown")
async def close_db_pool():
    """Close database connection pool at application shutdown."""
    global db_pool
    if db_pool:
        db_pool.close()
        await db_pool.wait_closed()
        logger.info("Database connection pool closed")

# Models
class OrderRequest(BaseModel):
    """Order execution request model."""

    symbol: str = Field(..., description="Trading symbol (e.g., 'MES')")
    action: str = Field(..., description="Order action: 'Buy' or 'Sell'")
    quantity: int = Field(..., gt=0, description="Order quantity")
    order_type: str = Field(
        "Market", description="Order type: 'Market', 'Limit', etc."
    )
    limit_price: Optional[float] = Field(None, description="Limit price if applicable")
    current_atr: Optional[float] = Field(
        None, description="Current ATR value for trailing stop"
    )
    account_id: Optional[int] = Field(None, description="Tradovate account ID")
    contract_id: Optional[int] = Field(None, description="Tradovate contract ID")

    @validator("action")
    def validate_action(cls, v):
        """Validate the action field."""
        if v.lower() not in ["buy", "sell"]:
            raise ValueError("Action must be 'Buy' or 'Sell'")
        return v.capitalize()

    @validator("order_type")
    def validate_order_type(cls, v):
        """Validate the order_type field."""
        valid_types = ["market", "limit", "stop", "stoplimit"]
        if v.lower() not in valid_types:
            raise ValueError(f"Order type must be one of {valid_types}")
        return v.capitalize()


class OrderResponse(BaseModel):
    """Order execution response model."""

    success: bool
    order_id: Optional[str] = None
    stop_id: Optional[str] = None
    message: str
    details: Dict = {}

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

def create_access_token(data: dict):
    """Create a JWT access token."""
    to_encode = data.copy()
    expires = datetime.utcnow() + timedelta(minutes=config.JWT_EXPIRATION_MINUTES)
    to_encode.update({"exp": expires})
    encoded_jwt = jwt.encode(
        to_encode, config.JWT_SECRET, algorithm=config.JWT_ALGORITHM
    )
    return encoded_jwt

# Helper Functions
def get_tradovate_credentials() -> Dict:
    """
    Retrieve Tradovate credentials from AWS Secrets Manager.
    
    Returns:
        Dict: Credentials including username, password, and app_id/secret
    """
    try:
        logger.info(f"Retrieving Tradovate credentials from Secrets Manager: {config.TRADOVATE_SECRETS_NAME}")
        response = secrets_client.get_secret_value(SecretId=config.TRADOVATE_SECRETS_NAME)
        
        if "SecretString" in response:
            secret = json.loads(response["SecretString"])
            required_keys = ["username", "password", "app_id", "app_secret"]
            
            if not all(key in secret for key in required_keys):
                logger.error("Missing required keys in Tradovate credentials")
                raise ValueError(f"Secret must contain: {required_keys}")
                
            return secret
        else:
            logger.error("Secret value is not a string")
            raise ValueError("Secret value must be a string")
            
    except Exception as e:
        logger.error(f"Error retrieving Tradovate credentials: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve broker credentials"
        )

def get_trading_mode() -> str:
    """
    Get the current trading mode from AWS Parameter Store.
    
    Returns:
        str: Trading mode ('live', 'shadow', etc.)
    """
    try:
        response = ssm_client.get_parameter(
            Name=config.TRADING_MODE_PARAM_PATH,
            WithDecryption=False
        )
        mode = response['Parameter']['Value']
        logger.info(f"Current trading mode: {mode}")
        return mode
    except Exception as e:
        logger.warning(f"Error retrieving trading mode from Parameter Store: {e}")
        logger.info("Defaulting to shadow mode")
        return "shadow"

async def get_db_connection():
    """
    Get the database connection pool.
    
    Returns:
        pool: PostgreSQL database connection pool or None if unavailable
    """
    global db_pool
    if db_pool is None:
        logger.warning("Database pool not initialized, trying to initialize now")
        try:
            dsn = f"host={config.DB_HOST} port={config.DB_PORT} dbname={config.DB_NAME} user={config.DB_USER} password={config.DB_PASSWORD}"
            db_pool = await aiopg.create_pool(
                dsn, 
                minsize=config.DB_MIN_CONNECTIONS, 
                maxsize=config.DB_MAX_CONNECTIONS
            )
            logger.info("Database connection pool initialized on-demand")
        except Exception as e:
            logger.error(f"Failed to initialize database pool on-demand: {e}")
            return None
    return db_pool

async def log_order_to_db(order_details: Dict) -> bool:
    """
    Log order details to the PostgreSQL database.
    
    Args:
        order_details: Dictionary containing order information
        
    Returns:
        bool: True if successful, False otherwise
    """
    pool = await get_db_connection()
    if not pool:
        logger.error("Could not connect to database to log order")
        return False
    
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                # Check if orders table exists, create if not
                await cursor.execute("""
                    CREATE TABLE IF NOT EXISTS orders (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP WITH TIME ZONE,
                        symbol VARCHAR(10),
                        action VARCHAR(10),
                        quantity INTEGER,
                        order_type VARCHAR(20),
                        price FLOAT,
                        trading_mode VARCHAR(10),
                        order_id VARCHAR(50),
                        stop_id VARCHAR(50),
                        atr_value FLOAT,
                        order_details JSONB
                    )
                """)
                
                # Insert order into database
                await cursor.execute("""
                    INSERT INTO orders (
                        timestamp, symbol, action, quantity, order_type, price, 
                        trading_mode, order_id, stop_id, atr_value, order_details
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    datetime.now(),
                    order_details.get('symbol', ''),
                    order_details.get('action', ''),
                    order_details.get('quantity', 0),
                    order_details.get('order_type', ''),
                    order_details.get('limit_price', 0.0),
                    order_details.get('trading_mode', 'shadow'),
                    order_details.get('order_id', ''),
                    order_details.get('stop_id', ''),
                    order_details.get('atr_used', 0.0),
                    json.dumps(order_details)
                ))
                
                logger.info(f"Order logged to database: {order_details.get('symbol')} {order_details.get('action')}")
                return True
    except Exception as e:
        logger.error(f"Error logging order to database: {e}")
        return False


class TradovateClient:
    """Client for interacting with Tradovate's REST API."""

    # Class variable to store client instances for reuse
    _instances = {}
    
    @classmethod
    async def get_client(cls, credentials: Dict):
        """
        Get or create a client instance for the given credentials.
        
        Args:
            credentials: Dict containing API credentials
            
        Returns:
            TradovateClient: Authenticated client instance
        """
        username = credentials["username"]
        
        # Return existing instance if available and still authenticated
        if username in cls._instances and cls._instances[username].is_authenticated():
            return cls._instances[username]
            
        # Create new instance
        client = cls(credentials)
        if await client.authenticate():
            cls._instances[username] = client
            return client
        
        raise ValueError("Failed to authenticate client")

    def __init__(self, credentials: Dict):
        """
        Initialize the Tradovate client.
        
        Args:
            credentials: Dict containing Tradovate API credentials
        """
        self.base_url = config.TRADOVATE_API_URL
        self.username = credentials["username"]
        self.password = credentials["password"]
        self.app_id = credentials["app_id"]
        self.app_secret = credentials["app_secret"]
        self.access_token = None
        self.md_access_token = None
        self.session = requests.Session()
        self.accounts = []
        self.user_id = None
        self.authenticated = False
        self.token_expiry = None
        self.max_retries = config.MAX_RETRIES
        
        # Configure session for connection pooling
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=10,
            max_retries=config.MAX_RETRIES
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)

    def is_authenticated(self) -> bool:
        """
        Check if the client is authenticated and token is valid.
        
        Returns:
            bool: True if authenticated with valid token
        """
        # Check if we have a token and if it's still valid
        if not self.authenticated or not self.access_token:
            return False
            
        # Check token expiration if available
        if self.token_expiry and datetime.now() > self.token_expiry:
            return False
            
        return True

    async def authenticate(self) -> bool:
        """
        Authenticate with Tradovate API.
        
        Returns:
            bool: True if authentication is successful
        """
        if self.is_authenticated():
            return True

        url = f"{self.base_url}/auth/accessTokenRequest"
        
        payload = {
            "name": self.username,
            "password": self.password,
            "appId": self.app_id,
            "appVersion": "1.0",
            "cid": self.app_secret,
            "sec": "true",
        }
        
        # Implement retry logic
        for attempt in range(self.max_retries):
            try:
                response = self.session.post(url, json=payload, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if "accessToken" in data:
                    self.access_token = data["accessToken"]
                    self.user_id = data.get("userId")
                    self.authenticated = True
                    
                    # Set token expiration (default: 4 hours)
                    expiry_hours = 4
                    if "expirationTime" in data:
                        # If API provides expiration time, use it
                        expiry_timestamp = data["expirationTime"]
                        self.token_expiry = datetime.fromtimestamp(expiry_timestamp / 1000)
                    else:
                        # Otherwise set default expiration
                        self.token_expiry = datetime.now() + timedelta(hours=expiry_hours)
                    
                    # Set auth headers for future requests
                    self.session.headers.update(
                        {"Authorization": f"Bearer {self.access_token}"}
                    )
                    
                    # Get account information
                    await self.get_accounts()
                    
                    return True
                else:
                    logger.error(f"Authentication failed: {data}")
                    # Wait before retry
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Authentication request failed (attempt {attempt+1}/{self.max_retries}): {e}")
                # Wait before retry
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
        
        # If we get here, all attempts failed
        return False

    async def get_accounts(self) -> List[Dict]:
        """
        Get Tradovate accounts for the authenticated user.
        
        Returns:
            List[Dict]: List of account objects
        """
        url = f"{self.base_url}/account/list"
        
        try:
            # Implement retry with backoff
            for attempt in range(self.max_retries):
                try:
                    response = self.session.get(url, timeout=10)
                    response.raise_for_status()
                    self.accounts = response.json()
                    return self.accounts
                except requests.exceptions.RequestException as e:
                    if attempt < self.max_retries - 1:
                        logger.warning(f"Failed to get accounts (attempt {attempt+1}/{self.max_retries}): {e}")
                        await asyncio.sleep(2 ** attempt)
                    else:
                        raise e
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get accounts after {self.max_retries} attempts: {e}")
            return []

    async def get_account_id(self, account_id: Optional[int] = None) -> int:
        """
        Get a valid account ID.
        
        Args:
            account_id: Optional specific account ID to use
            
        Returns:
            int: Valid account ID
        """
        if account_id:
            return account_id
            
        if not self.accounts:
            await self.get_accounts()
            
        if self.accounts:
            # Use first active account
            for account in self.accounts:
                if account.get("active", False):
                    return account.get("id")
            # If no active account, use the first one
            return self.accounts[0].get("id")
            
        raise ValueError("No valid trading account found")

    async def get_contract_id(self, symbol: str, contract_id: Optional[int] = None) -> int:
        """
        Get contract ID for the specified symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'MES')
            contract_id: Optional specific contract ID to use
            
        Returns:
            int: Contract ID
        """
        if contract_id:
            return contract_id
            
        # Find contract ID by symbol
        url = f"{self.base_url}/contract/find"
        payload = {"name": symbol}
        
        # Implement retry logic
        for attempt in range(self.max_retries):
            try:
                response = self.session.post(url, json=payload, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if "id" in data:
                    return data["id"]
                else:
                    raise ValueError(f"Contract not found for symbol: {symbol}")
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to find contract (attempt {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise ValueError(f"Failed to find contract for symbol: {symbol}")

    async def place_order(
        self, 
        account_id: int, 
        contract_id: int, 
        action: str, 
        quantity: int, 
        order_type: str = "Market",
        limit_price: Optional[float] = None
    ) -> Dict:
        """
        Place an order with Tradovate.
        
        Args:
            account_id: Tradovate account ID
            contract_id: Tradovate contract ID
            action: Order action ('Buy' or 'Sell')
            quantity: Order quantity
            order_type: Order type ('Market', 'Limit', etc.)
            limit_price: Limit price (required for Limit orders)
            
        Returns:
            Dict: Order response
        """
        url = f"{self.base_url}/order/placeOrder"
        
        payload = {
            "accountId": account_id,
            "contractId": contract_id,
            "action": action,
            "orderQty": quantity,
            "orderType": order_type,
            "timeInForce": "GTC",  # Good Till Canceled
        }
        
        # Add price for limit orders
        if order_type.lower() == "limit" and limit_price is not None:
            payload["price"] = limit_price
        
        # Implement retry logic
        for attempt in range(self.max_retries):    
            try:
                response = self.session.post(url, json=payload, timeout=10)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to place order (attempt {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise ValueError(f"Failed to place order: {str(e)}")

    async def place_oco_trailing_stop(
        self,
        account_id: int,
        contract_id: int,
        action: str,
        quantity: int,
        atr_value: float
    ) -> Dict:
        """
        Place a trailing stop order based on ATR.
        
        Args:
            account_id: Tradovate account ID
            contract_id: Tradovate contract ID
            action: Opposite of the entry action ('Buy' for a Sell entry, 'Sell' for a Buy entry)
            quantity: Order quantity
            atr_value: Current ATR value
            
        Returns:
            Dict: Order response
        """
        url = f"{self.base_url}/order/placeOrder"
        
        # Calculate trailing stop distance as 1 × ATR
        trail_amount = atr_value
        
        payload = {
            "accountId": account_id,
            "contractId": contract_id,
            "action": action,
            "orderQty": quantity,
            "orderType": "Trail",  # Trailing stop
            "timeInForce": "GTC",  # Good Till Canceled
            "trailAmount": trail_amount,
        }
        
        # Implement retry logic
        for attempt in range(self.max_retries):
            try:
                response = self.session.post(url, json=payload, timeout=10)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to place trailing stop (attempt {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise ValueError(f"Failed to place trailing stop: {str(e)}")


# Create a dependency for TradovateClient
async def get_tradovate_client():
    """Dependency for getting an authenticated Tradovate client."""
    credentials = get_tradovate_credentials()
    try:
        client = await TradovateClient.get_client(credentials)
        return client
    except ValueError as e:
        raise HTTPException(status_code=401, detail=f"Failed to authenticate with Tradovate: {str(e)}")


# Health check endpoint with enhanced monitoring
@app.get(
    "/health", 
    summary="Service health check",
    description="Returns the health status of the service and its dependencies.",
    response_model_exclude_none=True,
    responses={
        200: {
            "description": "Service health information",
            "content": {
                "application/json": {
                    "example": {
                        "status": "ok",
                        "timestamp": "2023-09-01T12:30:45Z",
                        "version": "1.0.0",
                        "dependencies": {
                            "database": "ok",
                            "aws_ssm": "ok"
                        }
                    }
                }
            }
        }
    }
)
async def health_check():
    """Health check endpoint with database and dependency status."""
    health_status = {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "dependencies": {}
    }
    
    # Check database
    try:
        pool = await get_db_connection()
        if pool:
            async with pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute("SELECT 1")
                    health_status["dependencies"]["database"] = "ok"
        else:
            health_status["dependencies"]["database"] = "unavailable"
            health_status["status"] = "degraded"
    except Exception:
        health_status["dependencies"]["database"] = "error"
        health_status["status"] = "degraded"
    
    # Check AWS services
    try:
        get_trading_mode()
        health_status["dependencies"]["aws_ssm"] = "ok"
    except Exception:
        health_status["dependencies"]["aws_ssm"] = "error"
        health_status["status"] = "degraded"
    
    return health_status

# Middleware for request timing and tracking
@app.middleware("http")
async def add_metrics(request: Request, call_next):
    """Add metrics middleware for tracking request duration and status."""
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    # Record metrics if enabled
    if config.ENABLE_METRICS:
        REQUEST_TIME.labels(endpoint=request.url.path).observe(duration)
    
    return response

# Main order execution endpoint with security
@app.post(
    "/execute", 
    response_model=OrderResponse,
    summary="Execute a trading order",
    description="""
    Executes a trading order through Tradovate.
    
    The order can be a market order, limit order, or stop order.
    When market orders are executed, an OCO (One-Cancels-Other) trailing stop
    is automatically created based on ATR (Average True Range).
    
    All orders are logged to the database for record-keeping.
    """,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "description": "Order executed successfully",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "order_id": "12345",
                        "stop_id": "67890",
                        "message": "Order executed successfully",
                        "details": {
                            "symbol": "MES",
                            "action": "Buy",
                            "quantity": 1,
                            "order_type": "Market",
                            "execution_time": "2023-09-01T12:30:45Z"
                        }
                    }
                }
            }
        },
        status.HTTP_400_BAD_REQUEST: {
            "description": "Invalid order request"
        },
        status.HTTP_403_FORBIDDEN: {
            "description": "Authentication failed"
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Internal server error or broker API error"
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "Service temporarily unavailable"
        }
    }
)
async def execute_order(
    order: OrderRequest, 
    authenticated: bool = Depends(verify_api_key),
    tradovate_client: TradovateClient = Depends(get_tradovate_client)
):
    """
    Execute an order through Tradovate.
    
    Args:
        order: Order execution request
        authenticated: Security dependency to verify API key
        tradovate_client: Authenticated Tradovate client
        
    Returns:
        OrderResponse: Order execution result
    """
    try:
        # Get current trading mode
        trading_mode = get_trading_mode()
        
        # Get account ID
        account_id = await tradovate_client.get_account_id(order.account_id)
        
        # Get contract ID
        contract_id = await tradovate_client.get_contract_id(order.symbol, order.contract_id)
        
        # Use provided ATR or default
        atr_value = order.current_atr if order.current_atr is not None else config.DEFAULT_ATR
        
        order_response = {}
        stop_response = {}
        
        # Only execute real orders in live mode
        if trading_mode.lower() == "live":
            # Place the main order
            order_response = await tradovate_client.place_order(
                account_id=account_id,
                contract_id=contract_id,
                action=order.action,
                quantity=order.quantity,
                order_type=order.order_type,
                limit_price=order.limit_price
            )
            
            # Determine opposite action for the stop order
            stop_action = "Sell" if order.action == "Buy" else "Buy"
            
            # Place the trailing stop order
            stop_response = await tradovate_client.place_oco_trailing_stop(
                account_id=account_id,
                contract_id=contract_id,
                action=stop_action,
                quantity=order.quantity,
                atr_value=atr_value
            )
            
            message = "Order executed successfully with trailing stop"
        else:
            # In shadow mode, just generate IDs for logging
            current_time = int(datetime.now().timestamp())
            order_response = {
                "id": f"shadow-{current_time}",
                "status": "Simulated",
                "orderType": order.order_type,
                "action": order.action,
                "orderQty": order.quantity,
            }
            
            stop_action = "Sell" if order.action == "Buy" else "Buy"
            stop_response = {
                "id": f"shadow-stop-{current_time}",
                "status": "Simulated",
                "orderType": "Trail",
                "action": stop_action,
                "orderQty": order.quantity,
                "trailAmount": atr_value,
            }
            
            message = f"Order simulated in {trading_mode} mode (no actual execution)"
            logger.info(f"SHADOW ORDER: {order.action} {order.quantity} {order.symbol} at {order.limit_price or 'market price'}")
        
        # Prepare response details
        details = {
            "main_order": order_response,
            "stop_order": stop_response,
            "timestamp": datetime.now().isoformat(),
            "atr_used": atr_value,
            "symbol": order.symbol,
            "action": order.action,
            "quantity": order.quantity,
            "order_type": order.order_type,
            "limit_price": order.limit_price,
            "trading_mode": trading_mode
        }
        
        # Log order to database regardless of mode
        await log_order_to_db(details)
        
        return OrderResponse(
            success=True,
            order_id=order_response.get("orderId", str(order_response.get("id", ""))),
            stop_id=stop_response.get("orderId", str(stop_response.get("id", ""))),
            message=message,
            details=details
        )
    except ValueError as e:
        logger.error(f"Value error during order execution: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error executing order: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to execute order: {str(e)}")


# Custom OpenAPI schema
def custom_openapi():
    """Generate a custom OpenAPI schema with security information."""
    if app.openapi_schema:
        return app.openapi_schema
        
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # Add security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "APIKeyHeader": {
            "type": "apiKey",
            "in": "header",
            "name": API_KEY_NAME
        },
        "APIKeyQuery": {
            "type": "apiKey",
            "in": "query",
            "name": API_KEY_NAME
        }
    }
    
    # Apply security globally
    openapi_schema["security"] = [
        {"APIKeyHeader": []},
        {"APIKeyQuery": []}
    ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema
    
app.openapi = custom_openapi


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 