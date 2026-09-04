import numpy as np
import pandas as pd
import logging
import math

logger = logging.getLogger(__name__)

class PortfolioTradingEnv:
    """
    Custom MDP Environment for Deep Reinforcement Learning Portfolio Allocation.
    Wraps the ExecutionSimulator to compute exact net fractional reward (Stepwise DSR/Log Returns).
    """
    def __init__(self, 
                 universe_isins: list,
                 historical_data: pd.DataFrame, # MultiIndex or panel dataframe containing all features
                 simulator,
                 initial_cash: float = 1_000_000.0,
                 risk_penalty_lambda: float = 0.1):
        
        self.universe = universe_isins
        self.num_assets = len(self.universe)
        self.simulator = simulator
        self.initial_cash = initial_cash
        self.risk_penalty_lambda = risk_penalty_lambda
        
        # Data
        self.data = historical_data
        self.dates = sorted(self.data['trade_date'].unique())
        
        # State dimension: weights(N), cash(1), alpha(N), cov_diag(N), vol/adv(N) = 4N + 1
        self.observation_space_dim = (self.num_assets * 4) + 1
        # Action space: N (Dirichlet concentration parameters)
        self.action_space_dim = self.num_assets
        
        self.reset()
        
    def reset(self):
        """Resets the environment to the beginning of the episode."""
        self.current_step_idx = 0
        self.current_cash = self.initial_cash
        self.current_holdings = pd.Series(0.0, index=self.universe)
        self.portfolio_value = self.initial_cash
        self.done = False
        
        return self._get_observation()
        
    def _get_observation(self):
        """Constructs the vectorized state space S_t."""
        if self.current_step_idx >= len(self.dates):
            return np.zeros(self.observation_space_dim)
            
        current_date = self.dates[self.current_step_idx]
        df_today = self.data[self.data['trade_date'] == current_date].set_index('isin')
        
        # Ensure all assets exist in today's data
        df_today = df_today.reindex(self.universe).fillna(0)
        
        # Normalize weights
        total_val = self.portfolio_value if self.portfolio_value > 0 else 1.0
        w_t = (self.current_holdings * df_today['adj_close']) / total_val
        w_t = w_t.fillna(0).values
        c_t = np.array([self.current_cash / total_val])
        
        # Features
        alpha_t = df_today['alpha_score'].values
        vol_t = df_today['volatility'].values
        adv_t = (df_today['total_traded_qty'] * df_today['adj_close']).values
        
        # Normalize ADV to avoid massive numbers
        adv_norm = adv_t / 1e6 # in millions
        
        # Construct State Vector
        obs = np.concatenate([
            w_t,                # Current Allocations (N)
            c_t,                # Cash Ratio (1)
            alpha_t,            # Cross-sectional Alpha (N)
            vol_t,              # Volatilities (N)
            adv_norm            # Liquidity/ADV (N)
        ])
        
        return np.nan_to_num(obs, 0.0)
        
    def step(self, action: np.ndarray):
        """
        Executes the Dirichlet action, passes target weights to the ExecutionSimulator,
        and computes the friction-adjusted net log-return reward.
        
        Action is expected to be a valid weight vector (sum=1, values in [0, 1]).
        """
        if self.done:
            return self._get_observation(), 0.0, True, {}
            
        current_date = self.dates[self.current_step_idx]
        df_today = self.data[self.data['trade_date'] == current_date].set_index('isin').reindex(self.universe).fillna(0)
        
        current_prices = df_today['adj_close']
        adv_data = df_today['total_traded_qty'] * df_today['adj_close']
        vol_data = df_today['volatility']
        
        # Map action array to Pandas Series
        target_weights = pd.Series(action, index=self.universe)
        
        # Scale target down slightly by a cash buffer (1%) to fund frictions
        target_weights = target_weights * 0.99
        
        # Record previous portfolio value
        prev_portfolio_value = self.portfolio_value
        
        # Execute via rigorously verified Execution Simulator
        exec_res = self.simulator.execute_target_portfolio(
            target_weights=target_weights,
            current_prices=current_prices,
            adv_data=adv_data,
            current_holdings=self.current_holdings,
            current_cash=self.current_cash,
            daily_volatilities=vol_data
        )
        
        # Update Internal State
        self.current_cash = exec_res['new_cash']
        self.current_holdings = exec_res['new_holdings']
        self.portfolio_value = exec_res['final_value']
        
        # Compute Reward (Friction-Adjusted Log Return)
        if prev_portfolio_value > 0 and self.portfolio_value > 0:
            step_return = (self.portfolio_value / prev_portfolio_value) - 1.0
            # Risk penalty (Standard deviation proxy from vol * weight)
            port_vol = np.sqrt(np.sum((target_weights.values * vol_data.values)**2))
            
            # Net Stepwise Log Return - Risk Penalty
            reward = math.log(self.portfolio_value / prev_portfolio_value) - (self.risk_penalty_lambda * port_vol)
        else:
            reward = -1.0 # Heavy penalty for blowing up the account
            
        self.current_step_idx += 1
        self.done = self.current_step_idx >= (len(self.dates) - 1)
        
        info = {
            'date': current_date,
            'portfolio_value': self.portfolio_value,
            'frictional_drag': exec_res['frictional_drag'],
            'step_return': step_return if 'step_return' in locals() else 0.0
        }
        
        return self._get_observation(), reward, self.done, info
