import pandas as pd
import numpy as np

class QualityEngine:
    """
    Quality Minus Junk (QMJ) Alpha Engine.
    Evaluates firms on Gross Profitability, ROIC, and Balance Sheet Safety.
    Operates strictly as an exclusionary filter to purge "junk" equities.
    """
    def __init__(self, junk_percentile: float = 0.20):
        """
        :param junk_percentile: The bottom Nth percentile to classify as 'junk'.
        """
        self.junk_percentile = junk_percentile

    def calculate_scores(self, df_fundamentals: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates Quality scores cross-sectionally.
        Required columns in df_fundamentals:
        - 'gross_profit'
        - 'total_assets'
        - 'net_income'
        - 'invested_capital'
        - 'total_debt'
        - 'total_equity'
        """
        df = df_fundamentals.copy()
        
        # 1. Gross Profitability to Assets (GPA)
        # Higher is better
        df['gpa'] = df['gross_profit'] / df['total_assets'].replace(0, np.nan)
        
        # 2. Return on Invested Capital (ROIC)
        # Higher is better
        df['roic'] = df['net_income'] / df['invested_capital'].replace(0, np.nan)
        
        # 3. Balance Sheet Safety (Debt-to-Equity inverse)
        # Lower D/E is better, so we use inverse or negative
        df['debt_to_equity'] = df['total_debt'] / df['total_equity'].replace(0, np.nan)
        df['safety'] = -df['debt_to_equity']
        
        # Cross-sectional z-scores for each metric
        for col in ['gpa', 'roic', 'safety']:
            mean_val = df[col].mean()
            std_val = df[col].std()
            if pd.isna(std_val) or std_val == 0:
                std_val = 1e-6
            df[f'{col}_z'] = (df[col] - mean_val) / std_val
            
        # Composite Quality Score
        df['quality_score'] = df['gpa_z'].fillna(0) + df['roic_z'].fillna(0) + df['safety_z'].fillna(0)
        
        # Determine Junk Threshold
        junk_threshold = df['quality_score'].quantile(self.junk_percentile)
        
        # True if NOT junk
        df['is_investable'] = df['quality_score'] >= junk_threshold
        
        return df[['quality_score', 'is_investable']]
