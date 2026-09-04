import logging
import os
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import time

from onyx.core.config import config, ensure_directories
from onyx.execution.simulator import ExecutionSimulator
from onyx.rl.environment import PortfolioTradingEnv
from onyx.rl.agent import PPOAgent, TORCH_AVAILABLE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_rl_agent(episodes=500):
    if not TORCH_AVAILABLE:
        logger.error("PyTorch is not installed. Please run 'pkg install python-torch' first.")
        sys.exit(1)
        
    logger.info("Starting Local DRL Training Loop (PPO)...")
    
    # 1. Setup Mock Historical Data (3 Assets, 250 Days)
    isins = ['INF001', 'INF002', 'INF003']
    dates = pd.date_range(start='2024-01-01', periods=250, freq='B')
    
    # Generate somewhat realistic trending data
    np.random.seed(42)
    p1 = np.linspace(100, 200, 250) + np.random.normal(0, 2, 250)
    p2 = np.linspace(300, 400, 250) + np.random.normal(0, 5, 250)
    p3 = np.linspace(50, 80, 250) + np.random.normal(0, 1, 250)
    
    mock_data = []
    for i, dt in enumerate(dates):
        mock_data.append({'trade_date': dt, 'isin': 'INF001', 'adj_close': p1[i], 'total_traded_qty': 150000, 'alpha_score': np.random.normal(0, 1), 'volatility': 0.02})
        mock_data.append({'trade_date': dt, 'isin': 'INF002', 'adj_close': p2[i], 'total_traded_qty': 200000, 'alpha_score': np.random.normal(0, 1), 'volatility': 0.02})
        mock_data.append({'trade_date': dt, 'isin': 'INF003', 'adj_close': p3[i], 'total_traded_qty': 500000, 'alpha_score': np.random.normal(0, 1), 'volatility': 0.02})
            
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
    
    # 3. Initialize Agent
    agent = PPOAgent(state_dim=env.observation_space_dim, action_dim=env.action_space_dim, lr=3e-4)
    
    # 4. Training Loop
    start_time = time.time()
    best_reward = -np.inf
    
    logger.info(f"Training for {episodes} episodes...")
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        
        # Buffers for PPO Update
        states, actions, log_probs, rewards, next_states, dones = [], [], [], [], [], []
        episode_reward = 0.0
        
        while not done:
            action, log_prob = agent.select_action(state, deterministic=False)
            
            next_state, reward, done, info = env.step(action)
            
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            
            state = next_state
            episode_reward += reward
            
        # Update Agent
        actor_loss, critic_loss = agent.train_step(states, actions, log_probs, rewards, next_states, dones)
        
        if episode_reward > best_reward:
            best_reward = episode_reward
            
        if (episode + 1) % 50 == 0:
            elapsed = time.time() - start_time
            logger.info(f"Episode {episode+1}/{episodes} | Reward: {episode_reward:.2f} | Best: {best_reward:.2f} | A_Loss: {actor_loss:.4f} | C_Loss: {critic_loss:.4f} | Time: {elapsed:.1f}s")
            
    logger.info(f"Training Complete! Total Time: {time.time() - start_time:.1f}s")
    
    # 5. Out-of-Sample (Deterministic) Evaluation
    logger.info("Evaluating deterministic policy...")
    state = env.reset()
    done = False
    final_value = 0.0
    while not done:
        action, _ = agent.select_action(state, deterministic=True)
        next_state, reward, done, info = env.step(action)
        state = next_state
        final_value = info['portfolio_value']
        
    logger.info(f"Final DRL Portfolio Value: ₹{final_value:,.2f}")

if __name__ == "__main__":
    train_rl_agent(episodes=300)
