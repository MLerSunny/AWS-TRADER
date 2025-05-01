# Trading Mode Feature for MES AI Trader

The MES AI Trader system supports multiple trading modes to facilitate safe development, testing, and deployment.

## Overview

The trading mode setting controls whether the system executes actual trades or simulates them:

- **shadow**: Orders are simulated and logged to the database but not sent to the broker (default)
- **live**: Orders are fully executed through the broker API

## Configuration

Trading mode is configured via AWS Parameter Store:

```
/mes-ai-trader/trading-mode
```

Default value: `shadow`

## Usage

### Checking the Current Mode

```bash
# Using AWS CLI
aws ssm get-parameter --name /mes-ai-trader/trading-mode

# Response
{
    "Parameter": {
        "Name": "/mes-ai-trader/trading-mode",
        "Type": "String",
        "Value": "shadow",
        "Version": 1,
        "LastModifiedDate": "2023-09-30T12:34:56.789Z",
        "ARN": "arn:aws:ssm:us-east-1:123456789012:parameter/mes-ai-trader/trading-mode",
        "DataType": "text"
    }
}
```

### Changing the Mode

```bash
# Set to shadow mode (safe testing)
aws ssm put-parameter --name /mes-ai-trader/trading-mode --value shadow --overwrite

# Set to live mode (real trading)
aws ssm put-parameter --name /mes-ai-trader/trading-mode --value live --overwrite
```

## Implementation

The Order Router service checks the trading mode parameter before executing any trade:

1. In **shadow** mode:
   - No actual REST calls are made to the broker API
   - Simulated order details are generated
   - Orders are logged to the database with `trading_mode = 'shadow'`
   - Responses indicate simulated status

2. In **live** mode:
   - REST calls are made to execute real trades
   - Orders are logged to the database with `trading_mode = 'live'`
   - Real order IDs are returned

## Security Considerations

- Always verify the trading mode before deploying to production
- Implement a change approval process for switching to live mode
- Consider adding IP restrictions to Parameter Store updates
- Regularly audit the trading mode parameter changes

## Database Logging

All orders, whether shadow or live, are logged to the PostgreSQL database in the `orders` table with the following schema:

```sql
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
```

This allows for consistent monitoring, backtesting, and analysis across modes. 