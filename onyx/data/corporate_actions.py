import logging
import pandas as pd
from datetime import datetime
from sqlalchemy.orm import Session
from onyx.data.database import CorporateAction, UDiffMarketData

logger = logging.getLogger(__name__)

class CorporateActionEngine:
    """
    Handles the backward-adjustment of historical prices for corporate actions
    to generate continuous return series (Adjusted Research Price).
    """
    def __init__(self, db_session: Session):
        self.db = db_session

    def get_adjustment_factors(self, isin: str, as_of_date: datetime.date) -> pd.DataFrame:
        """
        Retrieves all corporate actions for an ISIN that occurred AFTER the historical date 
        up to the as_of_date, to calculate the cumulative adjustment factor (\\gamma).
        """
        # Fetch actions
        actions = self.db.query(CorporateAction).filter(
            CorporateAction.isin == isin,
            CorporateAction.ex_date <= as_of_date
        ).order_by(CorporateAction.ex_date.desc()).all()
        
        if not actions:
            return pd.DataFrame()
            
        data = []
        for action in actions:
            gamma = 1.0
            if action.action_type == 'SPLIT':
                # A new shares for B old shares. To adjust historical prices backwards, 
                # price must decrease. We use multiplier: gamma = B / A
                if action.ratio_a and action.ratio_a > 0:
                    gamma = action.ratio_b / action.ratio_a
            elif action.action_type == 'BONUS':
                # A new shares for every B held -> gamma = B / (A + B)
                if (action.ratio_a + action.ratio_b) > 0:
                    gamma = action.ratio_b / (action.ratio_a + action.ratio_b)
            elif action.action_type == 'DIVIDEND':
                # We need the pre-ex-date close price to calculate exact gamma = (P_cum - D) / P_cum
                # For this simplified engine, we will calculate it dynamically during the apply phase
                # if we have the P_cum. Here we just store the raw action.
                pass
                
            data.append({
                'ex_date': action.ex_date,
                'action_type': action.action_type,
                'gamma': gamma,
                'dividend_amount': action.dividend_amount
            })
            
        return pd.DataFrame(data)

    def apply_adjustments(self, df_prices: pd.DataFrame, actions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes a raw price dataframe (sorted chronologically) and applies the cumulative gamma factor.
        Requires columns: 'trade_date', 'close_price', 'open_price', 'high_price', 'low_price'
        """
        if actions_df.empty:
            df_prices['adj_close'] = df_prices['close_price']
            df_prices['adj_open'] = df_prices['open_price']
            df_prices['adj_high'] = df_prices['high_price']
            df_prices['adj_low'] = df_prices['low_price']
            return df_prices

        # Create copies for adjusted prices
        df_adj = df_prices.copy()
        
        # We process backward in time from the most recent ex_date
        # For a date t, the price is multiplied by the product of all gammas for ex_dates > t
        
        df_adj['cum_gamma'] = 1.0
        
        for _, action in actions_df.iterrows():
            ex_date = pd.to_datetime(action['ex_date']).date()
            action_type = action['action_type']
            
            mask_before_ex = df_adj['trade_date'] < ex_date
            
            if action_type in ['SPLIT', 'BONUS']:
                df_adj.loc[mask_before_ex, 'cum_gamma'] *= action['gamma']
            elif action_type == 'DIVIDEND':
                # Exact calculation requires P_cum (close price on day before ex_date)
                # Find the closest trade_date < ex_date
                pre_ex_data = df_adj[mask_before_ex].sort_values('trade_date', ascending=False)
                if not pre_ex_data.empty:
                    p_cum = pre_ex_data.iloc[0]['close_price']
                    if p_cum > 0:
                        div_gamma = (p_cum - action['dividend_amount']) / p_cum
                        df_adj.loc[mask_before_ex, 'cum_gamma'] *= div_gamma
                        
        # Apply cumulative gamma to all price columns
        df_adj['adj_close'] = df_adj['close_price'] * df_adj['cum_gamma']
        df_adj['adj_open'] = df_adj['open_price'] * df_adj['cum_gamma']
        df_adj['adj_high'] = df_adj['high_price'] * df_adj['cum_gamma']
        df_adj['adj_low'] = df_adj['low_price'] * df_adj['cum_gamma']
        
        return df_adj
