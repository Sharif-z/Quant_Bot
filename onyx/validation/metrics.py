import numpy as np
import pandas as pd
import scipy.stats as stats
import logging

logger = logging.getLogger(__name__)

class ValidationMetrics:
    """
    Computes rigorous statistical metrics for algorithmic strategies.
    Includes Information Coefficient (IC) and Deflated Sharpe Ratio (DSR).
    """
    
    @staticmethod
    def calculate_ic(alpha_scores: pd.Series, forward_returns: pd.Series) -> float:
        """
        Calculates the Information Coefficient (IC) for a single cross-section.
        IC is the Spearman rank correlation between predicted alpha and actual forward returns.
        """
        df = pd.concat([alpha_scores, forward_returns], axis=1).dropna()
        if len(df) < 2:
            return 0.0
            
        ic, _ = stats.spearmanr(df.iloc[:, 0], df.iloc[:, 1])
        return ic if not np.isnan(ic) else 0.0

    @staticmethod
    def calculate_sharpe(returns: pd.Series, risk_free_rate: float = 0.05, periods_per_year: int = 252) -> float:
        """Standard Annualized Sharpe Ratio."""
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
            
        excess_returns = returns - (risk_free_rate / periods_per_year)
        ann_return = excess_returns.mean() * periods_per_year
        ann_vol = excess_returns.std() * np.sqrt(periods_per_year)
        
        return ann_return / ann_vol

    @staticmethod
    def calculate_dsr(returns: pd.Series, num_trials: int = 100, expected_mean_sr: float = 0.0, risk_free_rate: float = 0.05) -> float:
        """
        Deflated Sharpe Ratio (DSR).
        Adjusts the Sharpe Ratio for selection bias under Multiple Testing, non-normal returns (Skew/Kurtosis).
        Returns the probability that the true Sharpe Ratio is strictly greater than 0.
        """
        if len(returns) < 3 or returns.std() == 0:
            return 0.0
            
        # Calculate standard stats
        sr = ValidationMetrics.calculate_sharpe(returns, risk_free_rate)
        n = len(returns)
        skew = returns.skew()
        kurtosis = returns.kurtosis()
        
        # Expected maximum Sharpe Ratio among `num_trials` independent strategies
        if num_trials > 1:
            emc = 0.5772156649
            max_expected_sr = expected_mean_sr + np.sqrt(2 * np.log(num_trials)) * (
                1 - emc / (2 * np.log(num_trials))
            )
        else:
            max_expected_sr = expected_mean_sr
        
        # Adjusted standard error of the Sharpe Ratio (accounting for skew/kurtosis)
        sr_var = (1 - (skew * sr) + ((kurtosis - 1) / 4) * (sr ** 2)) / (n - 1)
        sr_std = np.sqrt(sr_var) if sr_var > 0 else 1e-6
        
        # Calculate the Z-score of the deflated Sharpe
        z_score = (sr - max_expected_sr) / sr_std
        
        # Probability (CDF)
        dsr_prob = stats.norm.cdf(z_score)
        
        return dsr_prob
