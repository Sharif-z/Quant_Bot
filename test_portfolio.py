import logging
import pandas as pd
import numpy as np
import sys
from onyx.portfolio.covariance import CovarianceEstimator
from onyx.portfolio.optimizer import ConvexOptimizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_portfolio():
    try:
        logger.info("Testing Portfolio Construction Engine...")
        
        # 1. Mock Returns Data for Covariance Estimation
        # 3 Assets over 100 days
        np.random.seed(42)
        returns = pd.DataFrame({
            'SEC1': np.random.normal(0.001, 0.02, 100),
            'SEC2': np.random.normal(-0.001, 0.03, 100),
            'SEC3': np.random.normal(0.002, 0.015, 100) # Low risk, high return
        })
        
        # Covariance - Ledoit Wolf
        cov_engine_lw = CovarianceEstimator(method='ledoit_wolf')
        cov_lw = cov_engine_lw.estimate(returns)
        logger.info("\nLedoit-Wolf Shrunk Covariance Matrix:")
        logger.info(cov_lw.to_string())
        
        # Covariance - RMT
        cov_engine_rmt = CovarianceEstimator(method='rmt')
        cov_rmt = cov_engine_rmt.estimate(returns)
        logger.info("\nRMT Filtered Covariance Matrix:")
        logger.info(cov_rmt.to_string())
        
        # 2. Mock Alpha Scores
        alphas = pd.Series({'SEC1': 0.05, 'SEC2': -0.10, 'SEC3': 0.15})
        
        # 3. Optimizer Setup
        # Max weight 50%
        optimizer = ConvexOptimizer(lambda_risk=2.0, lambda_turnover=0.05, max_weight=0.50)
        
        # Optimize without current weights (initial allocation)
        logger.info("\nRunning Initial Optimization...")
        weights = optimizer.optimize(alphas, cov_lw)
        logger.info("Optimal Weights:")
        logger.info(weights.to_string())
        
        # Ensure budget constraint
        assert np.isclose(weights.sum(), 1.0), "Budget constraint failed."
        # Ensure max weight constraint
        assert weights.max() <= 0.50 + 1e-5, "Max weight constraint failed."
        # Ensure SEC3 gets high allocation due to high alpha and low risk
        assert weights['SEC3'] > weights['SEC2'], "Alpha weighting failed."
        
        logger.info("\nSUCCESS: Portfolio Construction stack verified successfully.")
        
    except Exception as e:
        logger.error(f"Portfolio Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_portfolio()
