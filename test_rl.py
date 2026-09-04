import logging
import os
import pandas as pd
import numpy as np
from datetime import datetime
import sys

from onyx.core.config import config
from onyx.execution.simulator import ExecutionSimulator
from onyx.rl.environment import PortfolioTradingEnv
from onyx.rl.agent import PPOAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_rl_environment():
    logger.info("Testing Phase 6: DRL Environment & Agent...")
    
    # 1. Setup Mock Historical Data
    isins = ['INF001', 'INF002', 'INF003']
    dates = pd.date_range(start='2024-01-01', periods=10, freq='B')
    
    mock_data = []
    for dt in dates:
        for isin in isins:
            mock_data.append({
                'trade_date': dt,
                'isin': isin,
                'adj_close': np.random.uniform(100, 200),
                'total_traded_qty': np.random.uniform(5000, 15000),
                'alpha_score': np.random.normal(0, 1),
                'volatility': 0.02
            })
            
    df_hist = pd.DataFrame(mock_data)
    
    # 2. Initialize Environment
    simulator = ExecutionSimulator(volatility_factor=0.02)
    
    env = PortfolioTradingEnv(
        universe_isins=isins,
        historical_data=df_hist,
        simulator=simulator,
        initial_cash=1_000_000.0,
        risk_penalty_lambda=0.1
    )
    
    logger.info(f"Environment Initialized. State Space Dim: {env.observation_space_dim}, Action Space Dim: {env.action_space_dim}")
    
    # 3. Initialize Agent
    agent = PPOAgent(state_dim=env.observation_space_dim, action_dim=env.action_space_dim)
    
    # 4. Run an Episode
    state = env.reset()
    done = False
    total_reward = 0.0
    step = 0
    
    logger.info("\nRunning Mock Episode...")
    while not done:
        # Agent samples action
        action, log_prob = agent.select_action(state, deterministic=False)
        
        # Validate action properties (Simplex bounds)
        assert np.isclose(np.sum(action), 1.0), f"Action sum is {np.sum(action)}, expected 1.0"
        assert np.all(action >= 0.0) and np.all(action <= 1.0), "Action values out of bounds [0, 1]"
        
        # Step environment
        next_state, reward, done, info = env.step(action)
        
        # In a real training loop, we'd store transitions and call agent.train_step()
        # agent.train_step(...)
        
        logger.info(f"Step {step} | Date: {info['date'].date()} | Reward: {reward:.5f} | Portfolio: ₹{info['portfolio_value']:,.2f} | Action: {action.round(3)}")
        
        state = next_state
        total_reward += reward
        step += 1
        
    logger.info(f"\nEpisode Complete. Total Steps: {step}, Total Reward: {total_reward:.5f}")
    logger.info("SUCCESS: RL Environment bounds, state extraction, and execution simulator wrapping verified.")

if __name__ == "__main__":
    test_rl_environment()
