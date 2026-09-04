import numpy as np
import pandas as pd
import logging
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

class ConvexOptimizer:
    """
    Deterministic Convex Portfolio Optimizer.
    Balances expected alpha against portfolio variance (risk) and turnover (friction).
    """
    def __init__(self, lambda_risk: float = 2.0, lambda_turnover: float = 0.05, max_weight: float = 0.10):
        """
        :param lambda_risk: Penalty parameter for portfolio variance.
        :param lambda_turnover: Penalty parameter for L1 turnover (trading friction).
        :param max_weight: Maximum allowed weight for a single asset (10% default).
        """
        self.lambda_risk = lambda_risk
        self.lambda_turnover = lambda_turnover
        self.max_weight = max_weight

    def optimize(self, alpha_scores: pd.Series, cov_matrix: pd.DataFrame, 
                 current_weights: pd.Series = None,
                 sector_map: pd.Series = None, max_sector_weight: float = 0.25,
                 adv_data: pd.Series = None, portfolio_value: float = 10000000.0, max_participation: float = 0.05) -> pd.Series:
        """
        Solves the convex optimization problem:
        Minimize: - (w^T * alpha) + lambda_risk * (w^T * Sigma * w) + lambda_turnover * sum(|w - w_current|)
        
        Subject to:
        1. sum(w) = 1.0 (Fully invested)
        2. 0 <= w_i <= max_weight (Long-only, single asset cap)
        3. Sector limits: sum(w_sector) <= max_sector_weight
        4. Liquidity cap: w_i * portfolio_value <= max_participation * ADV_i
        """
        n_assets = len(alpha_scores)
        asset_names = alpha_scores.index
        
        # Align inputs
        cov = cov_matrix.loc[asset_names, asset_names].values
        alphas = alpha_scores.values
        
        if current_weights is None:
            w_current = np.zeros(n_assets)
        else:
            w_current = current_weights.reindex(asset_names).fillna(0.0).values
            
        # Objective Function
        def objective(w):
            port_alpha = w.dot(alphas)
            port_var = w.T @ cov @ w
            turnover = np.sum(np.abs(w - w_current))
            return -(port_alpha) + (self.lambda_risk * port_var) + (self.lambda_turnover * turnover)
            
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0} # Budget
        ]
        
        # Add Sector Constraints
        if sector_map is not None:
            # Align sector map
            sectors = sector_map.reindex(asset_names)
            unique_sectors = sectors.dropna().unique()
            for sector in unique_sectors:
                # Create a binary mask for the sector
                sector_mask = (sectors == sector).astype(float).values
                # fun(w) >= 0 -> max_sector_weight - sum(w * mask) >= 0
                constraints.append({
                    'type': 'ineq', 
                    'fun': lambda w, mask=sector_mask: max_sector_weight - np.sum(w * mask)
                })
        
        # Compute dynamic upper bounds for liquidity
        upper_bounds = np.full(n_assets, self.max_weight)
        if adv_data is not None:
            adv = adv_data.reindex(asset_names).fillna(0.0).values
            # max_w_liquidity = (max_participation * ADV) / portfolio_value
            liquidity_caps = (max_participation * adv) / portfolio_value
            # The bound is the tighter of max_weight and the liquidity cap
            upper_bounds = np.minimum(upper_bounds, liquidity_caps)
            
        bounds = tuple((0.0, ub) for ub in upper_bounds)
        
        # Initial guess (equal weight)
        w0 = np.ones(n_assets) / n_assets
        
        logger.info(f"Running SLSQP Optimizer for {n_assets} assets...")
        result = minimize(
            objective, 
            w0, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints,
            options={'ftol': 1e-7, 'disp': False}
        )
        
        if not result.success:
            logger.warning(f"Optimization failed to converge: {result.message}")
            # Fallback to equal weight of positive alphas if optimization fails
            positive_alphas = np.where(alphas > 0, alphas, 0)
            if np.sum(positive_alphas) > 0:
                w_fallback = positive_alphas / np.sum(positive_alphas)
                return pd.Series(np.clip(w_fallback, 0, self.max_weight), index=asset_names)
                
        # Clean weights (remove tiny allocations due to float precision)
        optimal_weights = result.x
        optimal_weights[optimal_weights < 1e-4] = 0.0
        optimal_weights = optimal_weights / np.sum(optimal_weights) # Re-normalize
        
        return pd.Series(optimal_weights, index=asset_names)
