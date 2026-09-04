import pandas as pd
import numpy as np

class MomentumEngine:
    """
    Momentum & Trend Alpha Engine.
    Evaluates relative strength, moving average crossovers, and volume confirmation.
    Operates on time-series panel data for multiple assets.
    """
    def __init__(self, roc_period: int = 126, fast_ma: int = 20, slow_ma: int = 50):
        # 126 days is approx 6 months
        self.roc_period = roc_period
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma

    def calculate_scores(self, df_prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates Momentum scores on a panel dataset.
        Required columns in df_prices:
        - 'isin'
        - 'trade_date'
        - 'adj_close'
        - 'total_traded_qty'
        
        Returns cross-sectional scores for the latest date per ISIN.
        """
        df = df_prices.sort_values(['isin', 'trade_date']).copy()
        
        # 1. Rate of Change (Relative Strength)
        df['roc'] = df.groupby('isin')['adj_close'].pct_change(periods=self.roc_period)
        
        # 2. Moving Average Crossover (Trend)
        df['ma_fast'] = df.groupby('isin')['adj_close'].transform(lambda x: x.rolling(self.fast_ma).mean())
        df['ma_slow'] = df.groupby('isin')['adj_close'].transform(lambda x: x.rolling(self.slow_ma).mean())
        df['trend_strength'] = (df['ma_fast'] - df['ma_slow']) / df['ma_slow'].replace(0, np.nan)
        
        # 3. Volume Confirmation
        # Check if recent volume (e.g., 5-day avg) is higher than long-term volume (e.g., 20-day avg)
        df['vol_5'] = df.groupby('isin')['total_traded_qty'].transform(lambda x: x.rolling(5).mean())
        df['vol_20'] = df.groupby('isin')['total_traded_qty'].transform(lambda x: x.rolling(20).mean())
        df['volume_surge'] = df['vol_5'] / df['vol_20'].replace(0, np.nan)
        
        # Extract the latest row for each ISIN to form the cross-section
        df_latest = df.groupby('isin').tail(1).copy()
        
        # Cross-sectional z-scoring
        for col in ['roc', 'trend_strength', 'volume_surge']:
            mean_val = df_latest[col].mean()
            std_val = df_latest[col].std()
            if pd.isna(std_val) or std_val == 0:
                std_val = 1e-6
            df_latest[f'{col}_z'] = (df_latest[col] - mean_val) / std_val
            
        # Composite Momentum Score
        # We give higher weight to ROC and Trend, with Volume acting as a multiplier/confirmation
        df_latest['momentum_score'] = (df_latest['roc_z'].fillna(0) + df_latest['trend_strength_z'].fillna(0)) * df_latest['volume_surge'].clip(lower=0.5, upper=2.0).fillna(1.0)
        
        return df_latest.set_index('isin')[['momentum_score', 'roc', 'trend_strength']]
