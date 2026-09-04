import numpy as np
import pandas as pd
import scipy.stats as stats
import logging

logger = logging.getLogger(__name__)

class DriftMonitor:
    """
    Phase 12: Model Drift Detection.
    Monitors live data for Population Stability Index (PSI) drift and tracks rolling Information Coefficient (IC) decay.
    """
    
    @staticmethod
    def calculate_psi(expected_array: np.ndarray, actual_array: np.ndarray, buckets: int = 10) -> float:
        """
        Calculates Population Stability Index (PSI) to detect distributional shifts.
        
        :param expected_array: Array of feature values from the Training set (Reference).
        :param actual_array: Array of feature values from the Live set (Current).
        :param buckets: Number of quantiles to bin the data into.
        :return: A float representing the PSI score. 
                 PSI < 0.1 = No shift. 0.1 < PSI < 0.25 = Slight shift. PSI > 0.25 = Major shift.
        """
        if len(expected_array) == 0 or len(actual_array) == 0:
            return 0.0
            
        # Define the quantile breakpoints based on the expected (training) distribution
        breakpoints = np.percentile(expected_array, np.linspace(0, 100, buckets + 1))
        # Ensure breakpoints are strictly unique to avoid binning errors
        breakpoints = np.unique(breakpoints)
        # Extend boundaries slightly to catch outliers
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf
        
        # Calculate frequencies in each bin
        expected_counts, _ = np.histogram(expected_array, bins=breakpoints)
        actual_counts, _ = np.histogram(actual_array, bins=breakpoints)
        
        # Convert to percentages and avoid divide-by-zero
        expected_pct = expected_counts / len(expected_array)
        actual_pct = actual_counts / len(actual_array)
        
        expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
        actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)
        
        # Calculate PSI
        psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
        return np.sum(psi_values)

    @staticmethod
    def check_kill_switch(historical_predictions: pd.Series, historical_returns: pd.Series, 
                          training_features: pd.DataFrame, live_features: pd.DataFrame,
                          psi_threshold: float = 0.25) -> bool:
        """
        Checks the Strategy Kill Switch conditions:
        1. Does the Rolling IC drop below 0 (model predictions are backwards)?
        2. Does the average PSI across all features exceed the critical threshold (0.25)?
        
        :return: True if the Strategy Kill Switch should be triggered (Halt Trading).
        """
        kill_strategy = False
        
        # 1. Rolling IC Check
        if len(historical_predictions) > 10 and len(historical_returns) > 10:
            df = pd.concat([historical_predictions, historical_returns], axis=1).dropna()
            if len(df) > 5:
                rolling_ic, _ = stats.spearmanr(df.iloc[:, 0], df.iloc[:, 1])
                if rolling_ic < 0:
                    logger.error(f"STRATEGY KILL TRIGGERED: Rolling IC is {rolling_ic:.4f} (Model predictive edge has collapsed below 0).")
                    kill_strategy = True
                else:
                    logger.info(f"Drift Monitor: Rolling IC is healthy at {rolling_ic:.4f}")
        
        # 2. PSI Feature Drift Check
        if training_features is not None and live_features is not None:
            total_psi = 0.0
            common_cols = set(training_features.columns).intersection(live_features.columns)
            
            for col in common_cols:
                train_data = training_features[col].dropna().values
                live_data = live_features[col].dropna().values
                if len(train_data) > 0 and len(live_data) > 0:
                    psi = DriftMonitor.calculate_psi(train_data, live_data)
                    total_psi += psi
                    if psi > psi_threshold:
                        logger.warning(f"Major Feature Drift detected in {col} (PSI: {psi:.4f})")
                        
            avg_psi = total_psi / max(len(common_cols), 1)
            logger.info(f"Drift Monitor: Average Feature PSI is {avg_psi:.4f}")
            
            if avg_psi > psi_threshold:
                logger.error(f"STRATEGY KILL TRIGGERED: Average PSI {avg_psi:.4f} > {psi_threshold}. Massive Market Regime Shift Detected.")
                kill_strategy = True
                
        return kill_strategy
