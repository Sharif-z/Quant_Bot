import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class CovarianceEstimator:
    """
    Handles robust covariance matrix estimation for portfolio optimization.
    Implements Ledoit-Wolf Shrinkage (Pure Numpy) and Random Matrix Theory (RMT) filtering.
    """
    def __init__(self, method: str = 'ledoit_wolf'):
        """
        :param method: 'ledoit_wolf' or 'rmt' (Random Matrix Theory)
        """
        self.method = method

    def estimate(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Estimates the covariance matrix given a historical returns dataframe.
        Columns should be asset identifiers (ISINs), rows should be dates.
        """
        if self.method == 'ledoit_wolf':
            return self._ledoit_wolf(returns)
        elif self.method == 'rmt':
            return self._rmt_filter(returns)
        else:
            raise ValueError(f"Unknown covariance method: {self.method}")

    def _ledoit_wolf(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Pure Numpy Implementation of Ledoit-Wolf shrinkage to a scaled identity matrix.
        Eliminates the scikit-learn dependency for Android/Termux environments.
        """
        returns_clean = returns.dropna(axis=1, how='all').fillna(0.0)
        X = returns_clean.values
        T, N = X.shape
        
        if N == 0 or T == 0:
            return pd.DataFrame()
            
        # Sample covariance
        S = np.cov(X, rowvar=False)
        if N == 1:
            S = np.array([[S]])
            
        # Target matrix (Scaled Identity)
        m = np.trace(S) / N
        F = m * np.eye(N)
        
        # Estimate shrinkage intensity (delta)
        # Using a simplified constant shrinkage or a basic variance of sample covariance
        X_centered = X - X.mean(axis=0)
        # Empirical variance of the covariance estimator
        var_S = np.sum((X_centered[:, :, None] * X_centered[:, None, :]) ** 2, axis=0) / T - S**2
        
        # Calculate optimal delta
        pi = np.sum(var_S)
        gamma = np.linalg.norm(S - F, 'fro')**2
        
        if gamma == 0:
            delta = 1.0
        else:
            delta = max(0.0, min(1.0, pi / (T * gamma)))
            
        shrunk_cov = (1 - delta) * S + delta * F
        
        return pd.DataFrame(shrunk_cov, index=returns_clean.columns, columns=returns_clean.columns)

    def _rmt_filter(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Random Matrix Theory eigenvalue filtering.
        Clips eigenvalues falling within the Marchenko-Pastur theoretical noise interval.
        """
        returns_clean = returns.dropna(axis=1, how='all').fillna(0.0)
        
        T, N = returns_clean.shape
        if N == 0 or T == 0:
            return pd.DataFrame()
            
        q = T / N
        
        # 1. Standardize returns
        std_returns = returns_clean / returns_clean.std()
        std_returns = std_returns.fillna(0)
        
        # 2. Compute sample correlation matrix
        corr = std_returns.corr().values
        
        # 3. Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(corr)
        
        # 4. Marchenko-Pastur bounds
        # Variance of the standardized returns is 1
        var = 1.0
        lambda_plus = var * (1 + (1/q)**0.5)**2
        
        # 5. Filter noise eigenvalues (clipping to the mean of noise eigenvalues to preserve trace)
        eigenvalues_filtered = eigenvalues.copy()
        noise_eigenvalues = eigenvalues[eigenvalues <= lambda_plus]
        if len(noise_eigenvalues) > 0:
            avg_noise = noise_eigenvalues.mean()
            eigenvalues_filtered[eigenvalues <= lambda_plus] = avg_noise
            
        # 6. Reconstruct correlation matrix
        corr_filtered = eigenvectors @ np.diag(eigenvalues_filtered) @ eigenvectors.T
        
        # Remove negative diagonals if any due to numerical issues
        np.fill_diagonal(corr_filtered, 1.0)
        
        # 7. Convert back to covariance matrix
        vols = np.diag(returns_clean.std().values)
        cov_filtered = vols @ corr_filtered @ vols
        
        return pd.DataFrame(cov_filtered, index=returns_clean.columns, columns=returns_clean.columns)
