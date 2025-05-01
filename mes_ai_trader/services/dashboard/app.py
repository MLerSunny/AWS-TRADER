#!/usr/bin/env python
"""
MES AI Trader Dashboard

This Streamlit dashboard provides visualization and interaction capabilities
for the MES AI Trader system, including equity curves, SHAP explanations,
and a Claude-3 powered Q&A assistant.
"""
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

import boto3
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import psycopg2
import shap
import streamlit as st
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
DB_HOST = os.environ.get("DB_HOST", "db")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "postgres")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "postgres")
BEDROCK_REGION = os.environ.get("AWS_REGION", "us-east-1")
CLAUDE_MODEL_ID = os.environ.get("CLAUDE_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")
SHAP_STORE_PATH = os.environ.get("SHAP_STORE_PATH", "./data/shap_values")


# Set page config
st.set_page_config(
    page_title="MES AI Trader Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_db_connection():
    """
    Establish a connection to the PostgreSQL database.
    
    Returns:
        conn: PostgreSQL database connection
    """
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
        )
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        st.error(f"Failed to connect to database: {e}")
        return None


def get_today_equity_data() -> pd.DataFrame:
    """
    Fetch today's equity curve data from the database.
    
    Returns:
        pd.DataFrame: DataFrame with timestamp and equity data
    """
    today = datetime.now().date()
    start_of_day = datetime.combine(today, datetime.min.time())
    
    conn = get_db_connection()
    if not conn:
        # Return dummy data if connection fails
        return pd.DataFrame({
            "timestamp": pd.date_range(start=start_of_day, periods=10, freq="1h"),
            "equity": [10000 + i * 50 + np.random.normal(0, 20) for i in range(10)]
        })
    
    try:
        query = """
        SELECT timestamp, equity 
        FROM trading_account_equity 
        WHERE timestamp >= %s 
        ORDER BY timestamp
        """
        
        df = pd.read_sql_query(query, conn, params=(start_of_day,))
        
        if df.empty:
            # Return dummy data if no data for today
            return pd.DataFrame({
                "timestamp": pd.date_range(start=start_of_day, periods=10, freq="1h"),
                "equity": [10000 + i * 50 + np.random.normal(0, 20) for i in range(10)]
            })
        
        return df
    except Exception as e:
        logger.error(f"Error fetching equity data: {e}")
        st.error(f"Failed to fetch equity data: {e}")
        # Return dummy data on error
        return pd.DataFrame({
            "timestamp": pd.date_range(start=start_of_day, periods=10, freq="1h"),
            "equity": [10000 + i * 50 + np.random.normal(0, 20) for i in range(10)]
        })
    finally:
        conn.close()


def get_last_trade() -> Dict:
    """
    Fetch the last trade from the database.
    
    Returns:
        Dict: Last trade details
    """
    conn = get_db_connection()
    if not conn:
        # Return dummy data if connection fails
        return {
            "id": 1,
            "symbol": "MES",
            "action": "BUY",
            "quantity": 1,
            "entry_price": 4550.25,
            "exit_price": 4562.75,
            "entry_time": datetime.now() - timedelta(hours=2),
            "exit_time": datetime.now() - timedelta(hours=1),
            "pnl": 12.5,
            "model": "lgbm"
        }
    
    try:
        query = """
        SELECT id, symbol, action, quantity, entry_price, exit_price, 
               entry_time, exit_time, pnl, model
        FROM trades 
        ORDER BY exit_time DESC 
        LIMIT 1
        """
        
        with conn.cursor() as cur:
            cur.execute(query)
            result = cur.fetchone()
            
            if not result:
                # Return dummy data if no trades
                return {
                    "id": 1,
                    "symbol": "MES",
                    "action": "BUY",
                    "quantity": 1,
                    "entry_price": 4550.25,
                    "exit_price": 4562.75,
                    "entry_time": datetime.now() - timedelta(hours=2),
                    "exit_time": datetime.now() - timedelta(hours=1),
                    "pnl": 12.5,
                    "model": "lgbm"
                }
            
            columns = ["id", "symbol", "action", "quantity", "entry_price", "exit_price", 
                      "entry_time", "exit_time", "pnl", "model"]
            return dict(zip(columns, result))
            
    except Exception as e:
        logger.error(f"Error fetching last trade: {e}")
        st.error(f"Failed to fetch last trade: {e}")
        # Return dummy data on error
        return {
            "id": 1,
            "symbol": "MES",
            "action": "BUY",
            "quantity": 1,
            "entry_price": 4550.25,
            "exit_price": 4562.75,
            "entry_time": datetime.now() - timedelta(hours=2),
            "exit_time": datetime.now() - timedelta(hours=1),
            "pnl": 12.5,
            "model": "lgbm"
        }
    finally:
        conn.close()


def load_shap_values(trade_id: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[str]]]:
    """
    Load SHAP values for a specific trade.
    
    Args:
        trade_id: ID of the trade
        
    Returns:
        Tuple containing SHAP values, base value, and feature names
    """
    try:
        # Check if SHAP data exists for this trade
        file_path = f"{SHAP_STORE_PATH}/trade_{trade_id}_shap.npz"
        
        if not os.path.exists(file_path):
            # If no SHAP data for this specific trade, use sample data
            sample_files = [f for f in os.listdir(SHAP_STORE_PATH) if f.endswith("_shap.npz")]
            if sample_files:
                file_path = os.path.join(SHAP_STORE_PATH, sample_files[0])
            else:
                return None, None, None
        
        data = np.load(file_path, allow_pickle=True)
        shap_values = data['shap_values']
        base_value = data['base_value']
        feature_names = data['feature_names'].tolist()
        
        return shap_values, base_value, feature_names
        
    except Exception as e:
        logger.error(f"Error loading SHAP values: {e}")
        return None, None, None


def plot_equity_curve(equity_data: pd.DataFrame) -> go.Figure:
    """
    Create a plotly figure for the equity curve.
    
    Args:
        equity_data: DataFrame with timestamp and equity data
        
    Returns:
        go.Figure: Plotly figure object
    """
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=equity_data["timestamp"],
            y=equity_data["equity"],
            mode="lines",
            name="Equity",
            line=dict(color="green", width=2),
            fill="tozeroy",
            fillcolor="rgba(0, 255, 0, 0.1)",
        )
    )
    
    # Calculate daily change
    if len(equity_data) > 1:
        start_equity = equity_data["equity"].iloc[0]
        current_equity = equity_data["equity"].iloc[-1]
        daily_change = current_equity - start_equity
        daily_pct = (daily_change / start_equity) * 100
        
        # Add annotation for daily change
        fig.add_annotation(
            x=equity_data["timestamp"].iloc[-1],
            y=current_equity,
            text=f"Daily: ${daily_change:.2f} ({daily_pct:.2f}%)",
            showarrow=True,
            arrowhead=1,
        )
    
    fig.update_layout(
        title="Today's Equity Curve",
        xaxis_title="Time",
        yaxis_title="Equity ($)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    
    return fig


def plot_shap_waterfall(shap_values: np.ndarray, base_value: float, feature_names: List[str]) -> go.Figure:
    """
    Create a plotly figure for the SHAP waterfall chart.
    
    Args:
        shap_values: SHAP values for the trade
        base_value: Base value for SHAP explanation
        feature_names: Feature names corresponding to SHAP values
        
    Returns:
        go.Figure: Plotly figure object
    """
    # Use only the first row of SHAP values if it's 2D
    if shap_values.ndim > 1:
        shap_values = shap_values[0]
    
    # Sort features by absolute SHAP value
    indices = np.argsort(np.abs(shap_values))[::-1]
    sorted_shap = shap_values[indices]
    sorted_features = [feature_names[i] for i in indices]
    
    # Limit to top 10 features for clarity
    sorted_shap = sorted_shap[:10]
    sorted_features = sorted_features[:10]
    
    # Calculate cumulative values for waterfall
    cumulative = np.cumsum(sorted_shap)
    total_shap = np.sum(shap_values)
    
    # Create figure
    fig = go.Figure()
    
    # Add base value
    fig.add_trace(
        go.Bar(
            x=["Base value"],
            y=[base_value],
            marker=dict(color="rgba(200, 200, 200, 0.8)"),
            name="Base value",
        )
    )
    
    # Add individual feature contributions
    for i, (feature, shap_val) in enumerate(zip(sorted_features, sorted_shap)):
        color = "green" if shap_val > 0 else "red"
        fig.add_trace(
            go.Bar(
                x=[feature],
                y=[shap_val],
                marker=dict(color=color),
                name=feature,
            )
        )
    
    # Add final prediction
    fig.add_trace(
        go.Bar(
            x=["Final prediction"],
            y=[base_value + total_shap],
            marker=dict(color="blue"),
            name="Final prediction",
        )
    )
    
    fig.update_layout(
        title="SHAP Waterfall for Last Trade",
        xaxis_title="Features",
        yaxis_title="Impact on Prediction",
        barmode="relative",
        waterfallgroupgap=0.2,
    )
    
    return fig


def get_claude_response(prompt: str) -> str:
    """
    Get a response from Claude-3 using AWS Bedrock.
    
    Args:
        prompt: User's question or prompt
        
    Returns:
        str: Claude's response
    """
    try:
        # Initialize Bedrock client
        bedrock_runtime = boto3.client(
            service_name="bedrock-runtime",
            region_name=BEDROCK_REGION,
        )
        
        # Format the prompt for Claude
        formatted_prompt = f"""
        Human: You are an AI assistant for a MES futures trading system. Answer the following question:

        {prompt}

        Please provide helpful, accurate information based on your knowledge of futures trading,
        quantitative finance, and algorithmic trading systems.

        Assistant:
        """
        
        # Define the request parameters
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "temperature": 0.5,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": formatted_prompt
                        }
                    ]
                }
            ]
        }
        
        response = bedrock_runtime.invoke_model(
            modelId=CLAUDE_MODEL_ID,
            body=json.dumps(request_body)
        )
        
        # Parse the response
        response_body = json.loads(response.get('body').read())
        return response_body.get('content')[0].get('text')
        
    except Exception as e:
        logger.error(f"Error getting Claude response: {e}")
        return f"Sorry, I encountered an error: {str(e)}"


def main():
    """Main dashboard application."""
    # Set header
    st.title("MES AI Trader Dashboard")
    
    # Create sidebar for Claude-3 Q&A
    st.sidebar.header("Ask Claude-3 Anything")
    user_question = st.sidebar.text_area("Your Question:", height=100, 
                                        placeholder="Ask about trading strategies, market conditions, or system functionality...")
    
    if st.sidebar.button("Get Answer"):
        with st.sidebar.spinner("Claude-3 is thinking..."):
            if user_question:
                claude_response = get_claude_response(user_question)
                st.sidebar.markdown("### Claude's Answer:")
                st.sidebar.markdown(claude_response)
            else:
                st.sidebar.warning("Please enter a question.")
    
    # Create two columns for the main content
    col1, col2 = st.columns(2)
    
    # Today's equity curve in first column
    with col1:
        st.header("Today's Performance")
        equity_data = get_today_equity_data()
        equity_fig = plot_equity_curve(equity_data)
        st.plotly_chart(equity_fig, use_container_width=True)
    
    # SHAP waterfall for last trade in second column
    with col2:
        st.header("Last Trade Analysis")
        
        # Get last trade info
        last_trade = get_last_trade()
        
        # Display trade info in an expander
        with st.expander("Trade Details", expanded=True):
            st.write(f"**Symbol:** {last_trade['symbol']}")
            st.write(f"**Action:** {last_trade['action']}")
            st.write(f"**Quantity:** {last_trade['quantity']}")
            st.write(f"**Entry Price:** ${last_trade['entry_price']:.2f}")
            st.write(f"**Exit Price:** ${last_trade['exit_price']:.2f}")
            st.write(f"**P&L:** ${last_trade['pnl']:.2f}")
            st.write(f"**Model:** {last_trade['model']}")
        
        # Load and display SHAP values
        shap_values, base_value, feature_names = load_shap_values(last_trade['id'])
        
        if shap_values is not None and base_value is not None and feature_names is not None:
            shap_fig = plot_shap_waterfall(shap_values, base_value, feature_names)
            st.plotly_chart(shap_fig, use_container_width=True)
        else:
            st.warning("SHAP data not available for this trade.")


if __name__ == "__main__":
    main() 