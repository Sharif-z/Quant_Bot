import pandas as pd
import numpy as np
from onyx.alpha.quality import QualityEngine

class ValueEngine:
    """
    Value Alpha Engine.
    Evaluates fundamental discounts (Book-to-Market, Earnings Yield).
    Dynamically intersects with Quality metrics to avoid value traps.
    """
    def __init__(self, quality_engine: QualityEngine = None):
        self.quality_engine = quality_engine or QualityEngine()

    def calculate_scores(self, df_fundamentals: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates Value scores cross-sectionally.
        Required columns in df_fundamentals:
        - 'book_value'
        - 'market_cap'
        - 'earnings_ttm'
        Plus all columns required by QualityEngine (if not already provided in 'is_investable' form).
        """
        df = df_fundamentals.copy()
        
        # 1. Book-to-Market (B/M)
        df['bm_ratio'] = df['book_value'] / df['market_cap'].replace(0, np.nan)
        
        # 2. Earnings Yield (E/P)
        df['earnings_yield'] = df['earnings_ttm'] / df['market_cap'].replace(0, np.nan)
        
        # Cross-sectional z-scores
        for col in ['bm_ratio', 'earnings_yield']:
            mean_val = df[col].mean()
            std_val = df[col].std()
            if pd.isna(std_val) or std_val == 0:
                std_val = 1e-6
            df[f'{col}_z'] = (df[col] - mean_val) / std_val
            
        # Composite Value Score
        df['value_score'] = df['bm_ratio_z'].fillna(0) + df['earnings_yield_z'].fillna(0)
        
        # Intersect with Quality to avoid value traps
        # If 'is_investable' is not present, calculate it
        if 'is_investable' not in df.columns:
            quality_results = self.quality_engine.calculate_scores(df)
            df['is_investable'] = quality_results['is_investable']
            df['quality_score'] = quality_results['quality_score']
            
        # Penalize value score heavily if it is a value trap (i.e., high value but low quality/junk)
        df['qarp_score'] = np.where(
            df['is_investable'],
            df['value_score'] + df['quality_score'],  # Quality At a Reasonable Price
            -999.0  # Trap avoidance
        )
        
        return df[['value_score', 'qarp_score', 'is_investable']]
