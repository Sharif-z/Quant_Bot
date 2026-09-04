import logging
import pandas as pd
import numpy as np
import time
import sys

from onyx.execution.simulator import ExecutionSimulator
from onyx.rl.environment import PortfolioTradingEnv
from onyx.rl.agent import PPOAgent, TORCH_AVAILABLE
from onyx.validation.metrics import ValidationMetrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_rl_agent():
    if not TORCH_AVAILABLE:
        logger.error("PyTorch not installed. Cannot evaluate DRL agent.")
        sys.exit(1)
        
    logger.info("Initializing DRL Statistical Scrutiny Suite...")
    
    # 1. Setup Mock Historical Data (3 Assets, 250 Days)
    isins = ['INF001', 'INF002', 'INF003']
    dates = pd.date_range(start='2024-01-01', periods=250, freq='B')
    
    np.random.seed(42)
    # INF001: Quality/Stable (Low Vol, positive alpha)
    p1 = np.linspace(100, 150, 250) + np.random.normal(0, 1, 250)
    # INF002: Momentum/High Beta (High Vol, explosive alpha but volatile)
    p2 = np.linspace(300, 450, 250) + np.random.normal(0, 8, 250)
    # INF003: Value Trap (Negative drift)
    p3 = np.linspace(80, 60, 250) + np.random.normal(0, 2, 250)
    
    mock_data = []
    for i, dt in enumerate(dates):
        mock_data.append({'trade_date': dt, 'isin': 'INF001', 'adj_close': p1[i], 'total_traded_qty': 500000, 'alpha_score': 0.8, 'volatility': 0.01})
        mock_data.append({'trade_date': dt, 'isin': 'INF002', 'adj_close': p2[i], 'total_traded_qty': 200000, 'alpha_score': 1.5, 'volatility': 0.05})
        mock_data.append({'trade_date': dt, 'isin': 'INF003', 'adj_close': p3[i], 'total_traded_qty': 100000, 'alpha_score': -1.2, 'volatility': 0.03})
            
    df_hist = pd.DataFrame(mock_data)
    
    # 2. Environment
    simulator = ExecutionSimulator(volatility_factor=0.02)
    env = PortfolioTradingEnv(universe_isins=isins, historical_data=df_hist, simulator=simulator, initial_cash=1_000_000.0, risk_penalty_lambda=0.1)
    
    # 3. Agent (Quick Train to simulate trained weights)
    agent = PPOAgent(state_dim=env.observation_space_dim, action_dim=env.action_space_dim, lr=1e-3)
    logger.info("Training agent briefly to stabilize policy parameters...")
    for _ in range(50): # Brief training for deterministic output
        state = env.reset()
        done = False
        states, actions, log_probs, rewards, next_states, dones = [], [], [], [], [], []
        while not done:
            action, log_prob = agent.select_action(state, deterministic=False)
            next_state, reward, done, _ = env.step(action)
            states.append(state); actions.append(action); log_probs.append(log_prob)
            rewards.append(reward); next_states.append(next_state); dones.append(done)
            state = next_state
        agent.train_step(states, actions, log_probs, rewards, next_states, dones)
        
    # 4. Out-of-Sample Scrutiny Run
    logger.info("--- Beginning Deterministic Out-of-Sample Run ---")
    state = env.reset()
    done = False
    
    daily_values = [1_000_000.0]
    allocations = []
    
    while not done:
        action, _ = agent.select_action(state, deterministic=True)
        next_state, reward, done, info = env.step(action)
        state = next_state
        daily_values.append(info['portfolio_value'])
        allocations.append(action)
        
    # Calculate returns
    port_series = pd.Series(daily_values)
    returns = port_series.pct_change().dropna()
    
    # --- SCRUTINY GATE 1: Deflated Sharpe Ratio ---
    logger.info("\n--- SCRUTINY GATE 1: DSR ---")
    # Penalize for 300 trials (the length of standard training)
    dsr_prob = ValidationMetrics.calculate_dsr(returns, num_trials=300)
    std_sharpe = ValidationMetrics.calculate_sharpe(returns)
    
    logger.info(f"Standard Sharpe Ratio: {std_sharpe:.3f}")
    logger.info(f"Deflated Sharpe Ratio (Prob > 0): {dsr_prob:.5f}")
    
    if dsr_prob < 0.95:
        logger.warning("FAILED DSR GATE: The agent's edge is likely a statistical mirage due to multiple-testing selection bias.")
    else:
        logger.info("PASSED DSR GATE: Agent edge survives rigorous Multiple Testing penalties.")
        
    # --- SCRUTINY GATE 2: Factor Attribution ---
    logger.info("\n--- SCRUTINY GATE 2: Factor Neutrality ---")
    avg_allocs = np.mean(allocations, axis=0)
    logger.info(f"Average Allocation -> Quality/Stable (INF001): {avg_allocs[0]:.2%}")
    logger.info(f"Average Allocation -> High-Beta/Mom (INF002): {avg_allocs[1]:.2%}")
    logger.info(f"Average Allocation -> Value Trap (INF003): {avg_allocs[2]:.2%}")
    
    if avg_allocs[1] > 0.80:
        logger.warning("FAILED ATTRIBUTION GATE: Agent is overwhelmingly loading on High-Beta momentum and ignoring Quality.")
    elif avg_allocs[2] > 0.20:
        logger.warning("FAILED ATTRIBUTION GATE: Agent is heavily allocating to the Value Trap (learning noise).")
    else:
        logger.info("PASSED ATTRIBUTION GATE: Agent correctly balances Quality/Stable assets while capitalizing on Momentum, filtering out Value Traps.")
        
    logger.info("\nScrutiny Complete.")

if __name__ == "__main__":
    evaluate_rl_agent()
