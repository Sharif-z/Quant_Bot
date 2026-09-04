import numpy as np
import pandas as pd
import scipy.stats as stats
import logging

logger = logging.getLogger(__name__)

class StatisticalGate:
    """
    Phase 10: The Statistical Gate.
    A rigorous interrogation suite to validate ML Models (XGBoost) and prevent overfitting/momentum-chasing.
    """
    
    @staticmethod
    def calculate_factor_attribution(portfolio_returns: np.ndarray, factor_returns: pd.DataFrame) -> dict:
        """
        1. Factor-Neutral Attribution.
        Runs a multi-variate OLS regression: R_port = alpha + B1(Mkt) + B2(Size) + B3(Value) + B4(Mom)
        Using numpy.linalg.lstsq to remain extremely lightweight.
        
        :param portfolio_returns: 1D array of portfolio returns over time.
        :param factor_returns: DataFrame where each column is a factor return series (e.g. 'Mkt', 'SMB', 'HML', 'WML').
        :return: Dictionary containing residual alpha, factor betas, and R-squared.
        """
        if len(portfolio_returns) != len(factor_returns):
            raise ValueError("Portfolio returns and factor returns must have the same length.")
            
        # Prepare independent variables (X) with an intercept term for Alpha
        X = factor_returns.values
        X_with_intercept = np.hstack([np.ones((len(X), 1)), X])
        y = portfolio_returns
        
        # OLS Regression: Solve X * Beta = y
        # beta contains [alpha, b1, b2, ... bn]
        beta, residuals, rank, s = np.linalg.lstsq(X_with_intercept, y, rcond=None)
        
        # Calculate R-squared
        y_mean = np.mean(y)
        total_sum_of_squares = np.sum((y - y_mean)**2)
        if len(residuals) > 0 and total_sum_of_squares > 0:
            r_squared = 1 - (residuals[0] / total_sum_of_squares)
        else:
            r_squared = np.nan
            
        results = {
            "residual_alpha": beta[0],
            "r_squared": r_squared
        }
        
        for i, col in enumerate(factor_returns.columns):
            results[f"beta_{col}"] = beta[i+1]
            
        return results

    @staticmethod
    def calculate_ic_metrics(ml_predictions: pd.Series, forward_returns: pd.Series) -> dict:
        """
        2. Information Coefficient (IC) & ICIR.
        Measures pure predictive ranking ability using Spearman Rank Correlation.
        
        :param ml_predictions: Cross-sectional ML probabilities or scores.
        :param forward_returns: Actual forward returns for the same assets.
        :return: Dictionary with IC and p-value.
        """
        # Align data
        df = pd.concat([ml_predictions, forward_returns], axis=1).dropna()
        if len(df) < 5:
            return {"IC": np.nan, "p_value": np.nan}
            
        ic, p_value = stats.spearmanr(df.iloc[:, 0], df.iloc[:, 1])
        
        return {
            "IC": ic,
            "p_value": p_value
        }

    @staticmethod
    def calculate_deflated_sharpe_ratio(actual_sharpe: float, num_trials: int, 
                                        expected_correlation: float = 0.5, 
                                        sample_length_years: float = 1.0) -> float:
        """
        3. Deflated Sharpe Ratio (DSR).
        Penalizes the Sharpe ratio for multiple-testing bias (hyperparameter tuning).
        Uses a simplified probabilistic estimation of the expected maximum Sharpe ratio.
        
        :param actual_sharpe: The annualized Sharpe ratio achieved by the model.
        :param num_trials: The number of hyperparameter combinations tested (e.g. ML threshold, trailing stop %).
        :param expected_correlation: Average correlation among the tested strategies.
        :param sample_length_years: Length of the backtest in years.
        :return: The estimated p-value (probability that the Sharpe is just luck). 
                 Lower is better (e.g. < 0.05 is statistically significant).
        """
        # Approximation of the expected maximum Sharpe ratio under the null hypothesis
        # Formula: Expected Max SR ~ sqrt(2 * ln(num_trials)) 
        # (Modified for correlation)
        effective_trials = num_trials * (1 - expected_correlation) + 1
        
        if effective_trials <= 1:
            expected_max_sr = 0.0
        else:
            # Euler-Mascheroni constant approximation for extreme value theory
            expected_max_sr = np.sqrt(2 * np.log(effective_trials)) 
            
        # Standard error of Sharpe Ratio ~ sqrt((1 + SR^2/2) / N) where N is number of periods
        # Assuming roughly 252 trading days per year
        n_periods = max(int(252 * sample_length_years), 1)
        sr_std_error = np.sqrt((1 + (expected_max_sr**2) / 2) / n_periods)
        
        # Calculate Z-score of our actual Sharpe vs the Expected Max Sharpe
        if sr_std_error > 0:
            z_score = (actual_sharpe - expected_max_sr) / sr_std_error
        else:
            z_score = 0.0
            
        # Calculate p-value (1 - CDF of normal distribution)
        p_val_dsr = 1.0 - stats.norm.cdf(z_score)
        
        return {
            "expected_max_sharpe": expected_max_sr,
            "deflated_p_value": p_val_dsr,
            "statistically_significant": p_val_dsr < 0.05
        }

def run_statistical_gate_demo():
    """Demonstration of the Statistical Gate rejecting a fake Momentum-chasing model."""
    logger.info("--- Phase 10: Executing The Statistical Gate ---")
    gate = StatisticalGate()
    
    # 1. Fake Factor Neutrality Test
    logger.info("\n1. Factor-Neutral Attribution Test")
    # Simulate an ML model that just disguised Momentum as Alpha
    np.random.seed(42)
    market_returns = np.random.normal(0.0005, 0.01, 252)
    momentum_returns = np.random.normal(0.001, 0.015, 252)
    value_returns = np.random.normal(0.0002, 0.01, 252)
    
    # The ML portfolio is literally just 0.9 * Momentum + 0.1 * Market + Noise (NO REAL ALPHA)
    ml_portfolio = 0.9 * momentum_returns + 0.1 * market_returns + np.random.normal(0, 0.001, 252)
    
    factors = pd.DataFrame({
        "Mkt": market_returns,
        "Value": value_returns,
        "Momentum": momentum_returns
    })
    
    attr = gate.calculate_factor_attribution(ml_portfolio, factors)
    logger.info(f"Residual Alpha: {attr['residual_alpha']:.6f}")
    logger.info(f"Momentum Beta : {attr['beta_Momentum']:.2f}")
    if attr['residual_alpha'] < 0.0001 and attr['beta_Momentum'] > 0.5:
        logger.error("GATE FAILED: XGBoost model is chasing momentum! No residual alpha found.")
    
    # 2. Fake IC Test
    logger.info("\n2. Information Coefficient (IC) Test")
    ml_preds = pd.Series(np.random.uniform(0, 1, 100))
    # True edge model: Returns are somewhat correlated to predictions
    actual_returns = ml_preds * 0.05 + np.random.normal(0, 0.02, 100)
    ic_metrics = gate.calculate_ic_metrics(ml_preds, actual_returns)
    logger.info(f"Spearman IC: {ic_metrics['IC']:.4f}, p-value: {ic_metrics['p_value']:.4e}")
    if ic_metrics['IC'] > 0.05 and ic_metrics['p_value'] < 0.05:
        logger.info("GATE PASSED: Model shows statistically significant predictive ranking power.")
        
    # 3. Fake DSR Test
    logger.info("\n3. Deflated Sharpe Ratio (DSR) Test")
    # A model got a Sharpe of 2.1, but we tested 300 hyperparameter combinations!
    actual_sr = 2.1
    trials = 300
    dsr_metrics = gate.calculate_deflated_sharpe_ratio(actual_sr, trials)
    logger.info(f"Actual Sharpe: {actual_sr}")
    logger.info(f"Expected Max Sharpe from 300 trials (Noise): {dsr_metrics['expected_max_sharpe']:.2f}")
    logger.info(f"Deflated P-Value: {dsr_metrics['deflated_p_value']:.4f}")
    if not dsr_metrics['statistically_significant']:
        logger.error("GATE FAILED: Sharpe Ratio is mathematically indistinguishable from hyperparameter overfitting noise.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    run_statistical_gate_demo()
