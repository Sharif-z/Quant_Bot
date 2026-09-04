import logging
import pandas as pd
from datetime import datetime
from sqlalchemy.orm import Session
from onyx.data.database import UDiffMarketData, SecurityMasterHistory
from onyx.data.corporate_actions import CorporateActionEngine

logger = logging.getLogger(__name__)

class PointInTimeStore:
    """
    The core feature engine guaranteeing zero look-ahead bias.
    Provides the dual price series: Raw Execution Price and Adjusted Research Price.
    """
    def __init__(self, db_session: Session):
        self.db = db_session
        self.ca_engine = CorporateActionEngine(db_session)

    def get_investable_universe(self, as_of_date: datetime.date) -> list:
        """
        Returns the list of ISINs that were actively listed and tradable exactly on as_of_date.
        """
        # A stock is in the universe if listed on/before as_of_date 
        # AND (not delisted OR delisted after as_of_date)
        universe = self.db.query(SecurityMasterHistory.isin).filter(
            SecurityMasterHistory.listing_date <= as_of_date,
            (SecurityMasterHistory.delisting_date == None) | (SecurityMasterHistory.delisting_date > as_of_date),
            SecurityMasterHistory.is_active == True
        ).all()
        
        return [row[0] for row in universe]

    def as_of(self, isin: str, target_date: datetime.date, lookback_days: int = 252) -> pd.DataFrame:
        """
        Returns the point-in-time historical data for a given ISIN up to the target_date.
        Includes both raw prices and backward-adjusted prices.
        """
        # 1. Fetch raw data strictly <= target_date
        raw_data = self.db.query(
            UDiffMarketData.trade_date,
            UDiffMarketData.open_price,
            UDiffMarketData.high_price,
            UDiffMarketData.low_price,
            UDiffMarketData.close_price,
            UDiffMarketData.total_traded_qty,
            UDiffMarketData.total_traded_value
        ).filter(
            UDiffMarketData.isin == isin,
            UDiffMarketData.trade_date <= target_date
        ).order_by(UDiffMarketData.trade_date.desc()).limit(lookback_days).all()
        
        if not raw_data:
            return pd.DataFrame()
            
        # Convert to DataFrame and sort chronologically (oldest to newest)
        df = pd.DataFrame(raw_data).sort_values('trade_date').reset_index(drop=True)
        
        # 2. Fetch corporate actions up to target_date
        actions_df = self.ca_engine.get_adjustment_factors(isin, target_date)
        
        # 3. Apply backward adjustments
        df_dual = self.ca_engine.apply_adjustments(df, actions_df)
        
        return df_dual
