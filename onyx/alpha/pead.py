import pandas as pd
import numpy as np

class PEADEngine:
    """
    Post-Earnings Announcement Drift (PEAD) Alpha Engine - Version 1.
    Calculates Standardized Unexpected Earnings (SUE) based on seasonal earnings drift.
    """
    def __init__(self):
        pass

    def calculate_scores(self, df_earnings: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the SUE score cross-sectionally.
        Required columns in df_earnings:
        - 'isin'
        - 'eps_current' (EPS for the most recently announced quarter)
        - 'eps_seasonal_lag' (EPS for the exact same quarter last year, t-4)
        - 'eps_std_dev' (Historical standard deviation of seasonal EPS differences over last 8 quarters)
        """
        df = df_earnings.copy()
        
        # 1. Calculate Seasonal Earnings Surprise
        df['earnings_surprise'] = df['eps_current'] - df['eps_seasonal_lag']
        
        # 2. Standardized Unexpected Earnings (SUE)
        # Scaled by historical variability to normalize the surprise magnitude
        df['sue'] = df['earnings_surprise'] / df['eps_std_dev'].replace(0, np.nan)
        
        # Cross-sectional ranking/z-score
        mean_sue = df['sue'].mean()
        std_sue = df['sue'].std()
        if pd.isna(std_sue) or std_sue == 0:
            std_sue = 1e-6
        df['pead_score'] = (df['sue'] - mean_sue) / std_sue
        
        # Fill NaN for stocks missing earnings data with neutral 0 score
        df['pead_score'] = df['pead_score'].fillna(0)
        
        return df[['isin', 'pead_score', 'sue']]
