#!/usr/bin/env python
"""
Proximal Policy Optimization (PPO) Training Script for MES Trader

This script trains a PPO agent to trade MES futures using a custom Gym environment.
The reward function is defined as the change in equity divided by Value at Risk (VaR).
"""
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Union, Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from feast import FeatureStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
MARKET_ID = "MES"  # Micro E-mini S&P 500
FEATURE_REPO_PATH = "../feature_repo"
MODEL_OUTPUT_PATH = "ppo.zip"
TRAIN_DAYS = 30  # Use 30 days of data for training
WINDOW_SIZE = 60  # Use 60 seconds of history for observation
VAR_WINDOW = 100  # Window for calculating VaR
VAR_CONFIDENCE = 0.95  # Confidence level for VaR calculation
COMMISSION_PCT = 0.0002  # 0.02% commission per trade
SLIPPAGE_PCT = 0.0001  # 0.01% slippage per trade
LEVERAGE = 5  # 5x leverage
INITIAL_BALANCE = 10000  # $10,000 initial balance

# Directory for saving models
MODEL_DIR = "checkpoints"
os.makedirs(MODEL_DIR, exist_ok=True)


class MESEnv(gym.Env):
    """
    Custom Gym environment for trading Micro E-mini S&P 500 (MES) futures.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        window_size: int = WINDOW_SIZE,
        var_window: int = VAR_WINDOW,
        var_confidence: float = VAR_CONFIDENCE,
        commission_pct: float = COMMISSION_PCT,
        slippage_pct: float = SLIPPAGE_PCT,
        leverage: float = LEVERAGE,
        initial_balance: float = INITIAL_BALANCE,
    ):
        """
        Initialize the MES trading environment.
        
        Args:
            data: Dataframe containing market data
            window_size: Number of periods to include in observation
            var_window: Window size for VaR calculation
            var_confidence: Confidence level for VaR calculation
            commission_pct: Commission percentage per trade
            slippage_pct: Slippage percentage per trade
            leverage: Leverage multiplier
            initial_balance: Initial account balance
        """
        super(MESEnv, self).__init__()
        
        self.data = data
        self.window_size = window_size
        self.var_window = var_window
        self.var_confidence = var_confidence
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.leverage = leverage
        self.initial_balance = initial_balance
        
        # Calculate the number of features
        self.feature_columns = [col for col in self.data.columns if col.startswith(('rsi', 'atr', 'vwap', 'order_book', 'margin'))]
        self.feature_count = len(self.feature_columns) + 5  # price, position, equity, unrealized PnL, balance
        
        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        
        # Define observation space (normalized market data for window_size periods)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.window_size, self.feature_count), dtype=np.float32
        )
        
        # Initialize the environment
        self.reset()
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment for a new episode.
        
        Args:
            seed: Random seed
            options: Optional configuration
        
        Returns:
            observation: Initial observation
            info: Additional information
        """
        super().reset(seed=seed)
        
        # Reset environment state
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0  # 0 = no position, positive = long, negative = short
        self.entry_price = 0
        self.equity_history = [self.initial_balance]
        self.returns_history = []
        self.done = False
        
        # Get initial observation
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def step(self, action):
        """
        Execute one trading step.
        
        Args:
            action: Action to take (0: Hold, 1: Buy, 2: Sell)
        
        Returns:
            observation: New observation
            reward: Reward for the action
            done: Whether the episode is done
            truncated: Whether the episode was truncated
            info: Additional information
        """
        # Check if episode is done
        if self.done:
            return self._get_observation(), 0, self.done, False, {}
        
        # Process the action
        self._take_action(action)
        
        # Move to the next step
        self.current_step += 1
        
        # Check if we've reached the end of the data
        if self.current_step >= len(self.data) - 1:
            self.done = True
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Get the new observation
        observation = self._get_observation()
        
        # Create info dictionary
        info = {
            "balance": self.balance,
            "position": self.position,
            "equity": self.equity_history[-1],
            "returns": self.returns_history[-1] if self.returns_history else 0,
        }
        
        return observation, reward, self.done, False, info
    
    def _get_observation(self):
        """
        Get the current observation.
        
        Returns:
            np.array: Current observation
        """
        # Extract the window of data
        start_idx = self.current_step - self.window_size
        end_idx = self.current_step
        
        # Get market data features
        market_data = self.data.iloc[start_idx:end_idx][self.feature_columns].values
        
        # Get price data
        price_data = self.data.iloc[start_idx:end_idx]['price'].values.reshape(-1, 1)
        
        # Prepare position and account info
        position = np.full((self.window_size, 1), self.position)
        equity = np.full((self.window_size, 1), self.equity_history[-1])
        
        # Calculate unrealized PnL if there's a position
        unrealized_pnl = np.zeros((self.window_size, 1))
        if self.position != 0:
            current_price = self.data.iloc[self.current_step]['price']
            unrealized_pnl_value = (current_price - self.entry_price) * self.position * self.leverage
            unrealized_pnl = np.full((self.window_size, 1), unrealized_pnl_value)
        
        # Include balance
        balance = np.full((self.window_size, 1), self.balance)
        
        # Combine all features
        observation = np.concatenate(
            [market_data, price_data, position, equity, unrealized_pnl, balance], axis=1
        )
        
        return observation.astype(np.float32)
    
    def _take_action(self, action):
        """
        Process and execute the trading action.
        
        Args:
            action: Action to take (0: Hold, 1: Buy, 2: Sell)
        """
        current_price = self.data.iloc[self.current_step]['price']
        
        # Calculate position size based on current equity and leverage
        equity = self.balance + self._calculate_unrealized_pnl()
        position_value = equity * self.leverage
        position_size = position_value / current_price
        
        if action == 0:  # Hold
            pass
        elif action == 1:  # Buy
            if self.position <= 0:  # If no position or short, close and go long
                # Close existing short position if any
                if self.position < 0:
                    self._close_position(current_price)
                
                # Open long position
                self.position = position_size
                self.entry_price = current_price * (1 + self.slippage_pct)  # Include slippage
                self.balance -= position_size * self.entry_price * self.commission_pct  # Deduct commission
        elif action == 2:  # Sell
            if self.position >= 0:  # If no position or long, close and go short
                # Close existing long position if any
                if self.position > 0:
                    self._close_position(current_price)
                
                # Open short position
                self.position = -position_size
                self.entry_price = current_price * (1 - self.slippage_pct)  # Include slippage
                self.balance -= abs(position_size) * self.entry_price * self.commission_pct  # Deduct commission
        
        # Update equity history
        equity = self.balance + self._calculate_unrealized_pnl()
        self.equity_history.append(equity)
        
        # Calculate returns
        if len(self.equity_history) > 1:
            returns = (self.equity_history[-1] - self.equity_history[-2]) / self.equity_history[-2]
            self.returns_history.append(returns)
    
    def _close_position(self, current_price):
        """
        Close the current position.
        
        Args:
            current_price: Current market price
        """
        if self.position == 0:
            return
        
        # Calculate PnL
        exit_price = current_price
        if self.position > 0:  # Long position
            exit_price *= (1 - self.slippage_pct)  # Include slippage
        else:  # Short position
            exit_price *= (1 + self.slippage_pct)  # Include slippage
        
        pnl = (exit_price - self.entry_price) * self.position
        
        # Update balance
        self.balance += pnl
        self.balance -= abs(self.position) * exit_price * self.commission_pct  # Deduct commission
        
        # Reset position
        self.position = 0
        self.entry_price = 0
    
    def _calculate_unrealized_pnl(self):
        """
        Calculate unrealized profit and loss.
        
        Returns:
            float: Unrealized PnL
        """
        if self.position == 0:
            return 0
        
        current_price = self.data.iloc[self.current_step]['price']
        unrealized_pnl = (current_price - self.entry_price) * self.position
        
        return unrealized_pnl
    
    def _calculate_var(self):
        """
        Calculate Value at Risk (VaR).
        
        Returns:
            float: Value at Risk
        """
        # Get the returns history for VaR window
        if len(self.returns_history) < self.var_window:
            return 0.01 * self.equity_history[-1]  # Default to 1% of equity if not enough history
        
        returns = self.returns_history[-self.var_window:]
        
        # Calculate VaR as the percentile of negative returns
        var = np.percentile(returns, (1 - self.var_confidence) * 100) * self.equity_history[-1]
        
        # Ensure VaR is positive (it's a loss measure)
        var = abs(var)
        
        # Minimum VaR to avoid division by zero
        min_var = 0.001 * self.equity_history[-1]  # 0.1% of equity
        
        return max(var, min_var)
    
    def _calculate_reward(self):
        """
        Calculate reward as the change in equity divided by VaR.
        
        Returns:
            float: Reward value
        """
        if len(self.equity_history) < 2:
            return 0
        
        # Calculate change in equity
        delta_equity = self.equity_history[-1] - self.equity_history[-2]
        
        # Calculate VaR
        var = self._calculate_var()
        
        # Calculate reward (delta_equity / VaR)
        reward = delta_equity / var
        
        return reward


def fetch_data():
    """
    Fetch data from the feature store.
    
    Returns:
        pd.DataFrame: Market data
    """
    logger.info("Initializing feature store connection...")
    store = FeatureStore(repo_path=FEATURE_REPO_PATH)
    
    # Calculate date range for feature fetching
    end_date = datetime.now()
    start_date = end_date - timedelta(days=TRAIN_DAYS)
    
    logger.info(f"Fetching features from {start_date} to {end_date}...")
    
    # Create an entity DataFrame with timestamps for the past 30 days
    timestamps = pd.date_range(start=start_date, end=end_date, freq="1s")
    entities = pd.DataFrame({
        "market_id": [MARKET_ID] * len(timestamps),
        "event_timestamp": timestamps
    })
    
    # Define the features to fetch
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
    
    # Get historical features
    feature_data = store.get_historical_features(
        entity_df=entities,
        features=features
    ).to_df()
    
    logger.info(f"Retrieved {len(feature_data)} feature records")
    
    # Generate synthetic price data for demonstration
    # In a real application, this would be replaced with actual price data
    price_data = generate_synthetic_prices(feature_data.index, feature_data)
    
    # Combine features with price data
    data = pd.concat([feature_data, price_data], axis=1)
    
    # Drop NaN values
    data = data.dropna()
    
    return data


def generate_synthetic_prices(index, feature_data):
    """
    Generate synthetic price data based on feature information.
    In a real scenario, this would be replaced with actual price data.
    
    Args:
        index: DataFrame index to match
        feature_data: Feature data to align with
    
    Returns:
        pd.DataFrame: Synthetic price data
    """
    n = len(index)
    base_price = 4500  # Base price for MES
    
    # Use the order book imbalance as a signal for price trend
    if 'order_book_features:imbalance' in feature_data.columns:
        imbalance = feature_data['order_book_features:imbalance'].fillna(0)
    else:
        imbalance = np.random.normal(0, 0.1, n)
    
    # Generate a random walk with drift influenced by imbalance
    daily_volatility = 0.01
    drift = imbalance * 0.5  # Scale imbalance effect
    price_changes = drift + np.random.normal(0, daily_volatility, n)
    
    # Calculate price series
    price_series = base_price * (1 + np.cumsum(price_changes))
    
    # Create price dataframe
    price_df = pd.DataFrame(index=index)
    price_df['price'] = price_series
    
    return price_df


def train_ppo_agent(data):
    """
    Train a PPO agent on the MES trading environment.
    
    Args:
        data: Market data
    
    Returns:
        PPO: Trained PPO agent
    """
    logger.info("Creating and training PPO agent...")
    
    # Create the environment
    def make_env():
        env = MESEnv(data)
        env = Monitor(env)
        return env
    
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
    
    # Define checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=MODEL_DIR,
        name_prefix="ppo_mes"
    )
    
    # Create the PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./tensorboard/"
    )
    
    # Train the agent
    logger.info("Starting training...")
    model.learn(
        total_timesteps=1000000,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # Evaluate the agent
    logger.info("Evaluating the trained agent...")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    logger.info(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    return model


def save_model(model):
    """
    Save the trained PPO model.
    
    Args:
        model: Trained PPO model
    """
    logger.info(f"Saving model to {MODEL_OUTPUT_PATH}")
    model.save(MODEL_OUTPUT_PATH)


def main():
    """Main execution function."""
    try:
        logger.info("Starting PPO agent training process...")
        
        # Fetch data
        data = fetch_data()
        
        # Train the PPO agent
        model = train_ppo_agent(data)
        
        # Save the model
        save_model(model)
        
        logger.info("Training process completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during model training: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main() 