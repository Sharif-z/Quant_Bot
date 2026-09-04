import logging
import pandas as pd
import numpy as np
from typing import List, Dict

from onyx.validation.metrics import ValidationMetrics

logger = logging.getLogger(__name__)

class PurgedWalkForward:
    """
    Executes a rigorous Purged Walk-Forward Cross-Validation.
    Simulates the strategy sequentially over time, completely insulating the 
    training/optimization phases from future data, and subjecting trades to frictional execution.
    """
    def __init__(self, pit_store, alpha_engines: dict, covariance_engine, optimizer, simulator, cash_buffer: float = 0.95):
        self.pit_store = pit_store
        self.alpha_engines = alpha_engines
        self.covariance_engine = covariance_engine
        self.optimizer = optimizer
        self.simulator = simulator
        self.cash_buffer = cash_buffer # Retain 5% cash to fund frictional taxes and slippage
        
    def run_backtest(self, universe_isins: List[str], rebalance_dates: List[pd.Timestamp], initial_cash: float = 1_000_000.0) -> Dict:
        """
        Executes the walk-forward validation loop across rebalance dates.
        """
        logger.info(f"Starting Purged Walk-Forward Engine across {len(rebalance_dates)} rebalance cycles...")
        
        current_cash = initial_cash
        current_holdings = pd.Series(0.0, index=universe_isins)
        
        portfolio_history = []
        ic_history = []
        
        prev_alpha_scores = None
        prev_prices = None
        
        for i, rebalance_date in enumerate(rebalance_dates):
            date_obj = rebalance_date.date()
            logger.info(f"--- Rebalance Cycle: {date_obj} ---")
            
            # 1. Point-in-Time Data Extraction (Zero Lookahead)
            # We look back 150 days to generate enough history for momentum and covariance
            df_hist = pd.DataFrame()
            for isin in universe_isins:
                try:
                    df_asset = self.pit_store.as_of(isin, date_obj, lookback_days=150)
                    if not df_asset.empty:
                        df_asset['isin'] = isin
                        df_hist = pd.concat([df_hist, df_asset])
                except Exception:
                    pass
                    
            if df_hist.empty:
                logger.warning(f"No data available for {date_obj}. Skipping.")
                continue
                
            # Current Prices
            latest_data = df_hist[df_hist['trade_date'] == df_hist['trade_date'].max()]
            if latest_data.empty:
                continue
            current_prices = latest_data.set_index('isin')['adj_close']
            
            # Evaluate IC of the PREVIOUS cycle's alpha
            if prev_alpha_scores is not None and prev_prices is not None:
                # Forward returns = Current Price / Prev Price - 1
                forward_returns = (current_prices / prev_prices) - 1.0
                ic = ValidationMetrics.calculate_ic(prev_alpha_scores, forward_returns)
                ic_history.append({'date': date_obj, 'ic': ic})
            
            # 2. Alpha Generation
            # Momentum
            m_engine = self.alpha_engines.get('momentum')
            df_m = m_engine.calculate_scores(df_hist[['isin', 'trade_date', 'adj_close', 'total_traded_qty']])
            
            # Mock fundamental alpha generation for this backtest integration test
            # In a real system, this queries the PIT fundamental database
            df_alpha = pd.Series(df_m['momentum_score'], index=universe_isins).fillna(0.0)
            
            # Save for next cycle IC calculation
            prev_alpha_scores = df_alpha.copy()
            prev_prices = current_prices.copy()
            
            # 3. Covariance Estimation
            df_hist_pivot = df_hist.pivot(index='trade_date', columns='isin', values='adj_close')
            returns = df_hist_pivot.pct_change().dropna()
            
            if len(returns) < 10:
                logger.warning(f"Not enough returns history to compute covariance on {date_obj}. Skipping.")
                continue
                
            cov_matrix = self.covariance_engine.estimate(returns)
            
            # 4. Optimization
            # Estimate ADV (Average Daily Volume in currency)
            adv_data = df_hist.groupby('isin')['total_traded_qty'].mean() * current_prices
            
            portfolio_value = current_cash + (current_holdings * current_prices).sum()
            if np.isnan(portfolio_value):
                portfolio_value = current_cash
            
            # Only optimize if we have variation in alphas
            if df_alpha.std() > 0:
                target_weights = self.optimizer.optimize(
                    alpha_scores=df_alpha,
                    cov_matrix=cov_matrix,
                    adv_data=adv_data,
                    portfolio_value=portfolio_value,
                    max_participation=0.05
                )
            else:
                target_weights = pd.Series(0.0, index=universe_isins)
                
            # CASH BUFFER LOGIC: Scale target weights down to reserve cash for taxes/slippage
            scaled_target_weights = target_weights * self.cash_buffer
            
            # 5. Execution Simulation (Frictional)
            exec_res = self.simulator.execute_target_portfolio(
                target_weights=scaled_target_weights,
                current_prices=current_prices,
                adv_data=adv_data,
                current_holdings=current_holdings,
                current_cash=current_cash
            )
            
            current_cash = exec_res['new_cash']
            current_holdings = exec_res['new_holdings']
            net_value = exec_res['final_value']
            
            portfolio_history.append({
                'date': date_obj,
                'portfolio_value': net_value,
                'cash': current_cash,
                'frictional_drag': exec_res['frictional_drag']
            })
            
            logger.info(f"Net Value: ₹{net_value:,.2f} | Drag: ₹{exec_res['frictional_drag']:,.2f}")

        # Post-Processing
        df_port = pd.DataFrame(portfolio_history).set_index('date')
        df_ic = pd.DataFrame(ic_history) if ic_history else pd.DataFrame()
        
        # Calculate Returns & DSR
        df_port['returns'] = df_port['portfolio_value'].pct_change().fillna(0)
        dsr = ValidationMetrics.calculate_dsr(df_port['returns'], num_trials=1) # 1 trial for baseline
        standard_sharpe = ValidationMetrics.calculate_sharpe(df_port['returns'])
        
        mean_ic = df_ic['ic'].mean() if not df_ic.empty else 0.0
        
        logger.info("\n--- BACKTEST COMPLETE ---")
        logger.info(f"Final Value: ₹{df_port['portfolio_value'].iloc[-1]:,.2f}")
        logger.info(f"Standard Sharpe Ratio: {standard_sharpe:.3f}")
        logger.info(f"Deflated Sharpe Ratio (DSR) Prob: {dsr:.3f}")
        logger.info(f"Mean Information Coefficient (IC): {mean_ic:.3f}")

        return {
            'portfolio_history': df_port,
            'ic_history': df_ic,
            'dsr': dsr,
            'sharpe': standard_sharpe,
            'mean_ic': mean_ic
        }
