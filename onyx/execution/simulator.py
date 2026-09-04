import numpy as np
import pandas as pd
import logging
from onyx.execution.accounting import FrictionCalculator

logger = logging.getLogger(__name__)

class ExecutionSimulator:
    """
    Simulates portfolio rebalancing transitions using exact delta generation,
    Square Root Law of market impact, and rigorous frictional accounting.
    """
    def __init__(self, volatility_factor: float = 0.02):
        """
        :param volatility_factor: Default daily volatility estimate (2%) if not provided per asset.
        """
        self.accounting = FrictionCalculator()
        self.volatility_factor = volatility_factor

    def compute_market_impact(self, order_qty: float, adv: float, daily_volatility: float) -> float:
        """
        Square Root Law of Market Impact.
        Returns the proportional price slippage (e.g., 0.005 for 0.5% slippage).
        """
        if adv <= 0 or order_qty == 0:
            return 0.0
        
        # Proportional impact = c * sigma * sqrt(Q / ADV)
        # Using theoretical constant c = 1.0 
        c = 1.0 
        participation_rate = abs(order_qty) / adv
        
        # Cap participation to avoid math errors or ridiculous theoretical slippage
        participation_rate = min(participation_rate, 1.0)
        
        impact = c * daily_volatility * np.sqrt(participation_rate)
        return impact

    def execute_target_portfolio(self, 
                               target_weights: pd.Series, 
                               current_prices: pd.Series,
                               adv_data: pd.Series,
                               current_holdings: pd.Series = None,
                               current_cash: float = 1000000.0,
                               daily_volatilities: pd.Series = None) -> dict:
        """
        Simulates transitioning from current holdings to target weights, executing trades
        sequentially (sells then buys), deducting slippage and taxes from the cash balance.
        """
        assets = target_weights.index
        
        if current_holdings is None:
            current_holdings = pd.Series(0.0, index=assets)
        else:
            # Reindex to ensure alignment
            current_holdings = current_holdings.reindex(assets).fillna(0.0)
            
        if daily_volatilities is None:
            daily_volatilities = pd.Series(self.volatility_factor, index=assets)
            
        # 1. Calculate Initial Portfolio Value (M2M)
        portfolio_value = current_cash + (current_holdings * current_prices).sum()
        
        # 2. Calculate Target Holdings
        target_values = target_weights * portfolio_value
        target_shares = (target_values / current_prices).fillna(0) # Fractional shares allowed for theoretical test
        
        # 3. Calculate Delta (Trade List)
        trades_shares = target_shares - current_holdings
        
        # 4. Separate Sells and Buys (Process Sells first to free cash)
        sells = trades_shares[trades_shares < 0]
        buys = trades_shares[trades_shares > 0]
        
        execution_log = []
        new_cash = current_cash
        new_holdings = current_holdings.copy()
        
        def process_trade(asset, qty):
            nonlocal new_cash, new_holdings
            
            is_buy = qty > 0
            abs_qty = abs(qty)
            price = current_prices[asset]
            adv = adv_data.get(asset, 1e6)
            vol = daily_volatilities.get(asset, self.volatility_factor)
            
            # Market Impact Slippage
            impact_pct = self.compute_market_impact(abs_qty, adv, vol)
            
            # Executed Price (Worse than current price)
            if is_buy:
                exec_price = price * (1 + impact_pct)
            else:
                exec_price = price * (1 - impact_pct)
                
            trade_value = abs_qty * exec_price
            
            # Frictions (Taxes)
            frictions = self.accounting.calculate_trade_costs(trade_value, is_buy=is_buy, is_delivery=True)
            total_cost = frictions['total_friction']
            
            net_cash_impact = -trade_value - total_cost if is_buy else trade_value - total_cost
            
            # Update State
            new_cash += net_cash_impact
            new_holdings[asset] += qty
            
            execution_log.append({
                'asset': asset,
                'type': 'BUY' if is_buy else 'SELL',
                'qty': abs_qty,
                'target_price': price,
                'exec_price': exec_price,
                'slippage_pct': impact_pct,
                'trade_value': trade_value,
                'frictions': total_cost,
                'net_cash_impact': net_cash_impact
            })
            
        # Execute Sells
        for asset, qty in sells.items():
            process_trade(asset, qty)
            
        # Execute Buys
        for asset, qty in buys.items():
            process_trade(asset, qty)
            
        final_portfolio_value = new_cash + (new_holdings * current_prices).sum()
        
        # In a real scenario, executing target weights perfectly might result in negative cash
        # due to slippage and taxes reducing overall portfolio value during the transition.
        # A robust system typically scales down buys if cash is insufficient.
        if new_cash < -0.01:
            logger.warning(f"Portfolio transition resulted in negative cash balance: {new_cash:.2f}")
            
        df_log = pd.DataFrame(execution_log) if execution_log else pd.DataFrame()
        
        return {
            'initial_value': portfolio_value,
            'final_value': final_portfolio_value,
            'frictional_drag': portfolio_value - final_portfolio_value,
            'new_cash': new_cash,
            'new_holdings': new_holdings,
            'execution_log': df_log
        }
