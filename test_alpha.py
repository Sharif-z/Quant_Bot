import logging
import pandas as pd
import numpy as np
import sys
from onyx.alpha.quality import QualityEngine
from onyx.alpha.value import ValueEngine
from onyx.alpha.momentum import MomentumEngine
from onyx.alpha.pead import PEADEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_alpha_engines():
    logger.info("Testing Baseline Alpha Engines...")
    try:
        # Mock Fundamental Data
        df_fundamentals = pd.DataFrame([
            {'isin': 'SEC1', 'gross_profit': 500, 'total_assets': 1000, 'net_income': 100, 'invested_capital': 800, 'total_debt': 100, 'total_equity': 900, 'book_value': 900, 'market_cap': 2000, 'earnings_ttm': 150},
            {'isin': 'SEC2', 'gross_profit': 10, 'total_assets': 1000, 'net_income': -50, 'invested_capital': 800, 'total_debt': 800, 'total_equity': 200, 'book_value': 200, 'market_cap': 100, 'earnings_ttm': -50}, # Junk / Value Trap
            {'isin': 'SEC3', 'gross_profit': 800, 'total_assets': 1000, 'net_income': 200, 'invested_capital': 800, 'total_debt': 50, 'total_equity': 950, 'book_value': 950, 'market_cap': 1000, 'earnings_ttm': 250}, # High Quality Value
        ])
        
        # Test Quality
        q_engine = QualityEngine(junk_percentile=0.33)
        df_q = q_engine.calculate_scores(df_fundamentals)
        
        logger.info("\nQuality Scores:")
        df_fundamentals_q = pd.concat([df_fundamentals['isin'], df_q], axis=1)
        logger.info(df_fundamentals_q.to_string())
        
        # SEC2 should be marked as junk (is_investable = False)
        assert df_fundamentals_q[df_fundamentals_q['isin'] == 'SEC2']['is_investable'].values[0] == False
        
        # Test Value
        v_engine = ValueEngine(quality_engine=q_engine)
        df_v = v_engine.calculate_scores(df_fundamentals)
        
        logger.info("\nValue Scores:")
        df_fundamentals_v = pd.concat([df_fundamentals['isin'], df_v], axis=1)
        logger.info(df_fundamentals_v.to_string())
        
        # SEC2 should have a heavily penalized qarp_score (-999.0) because it's a value trap
        assert df_fundamentals_v[df_fundamentals_v['isin'] == 'SEC2']['qarp_score'].values[0] == -999.0
        
        # Mock Momentum Data
        dates = pd.date_range(start='2024-01-01', periods=150, freq='D')
        df_prices = pd.DataFrame({
            'isin': ['SEC1']*150 + ['SEC2']*150,
            'trade_date': list(dates) + list(dates),
            'adj_close': np.concatenate([np.linspace(100, 150, 150), np.linspace(50, 20, 150)]), # SEC1 goes up, SEC2 goes down
            'total_traded_qty': np.ones(300) * 1000
        })
        
        m_engine = MomentumEngine(roc_period=126, fast_ma=20, slow_ma=50)
        df_m = m_engine.calculate_scores(df_prices)
        
        logger.info("\nMomentum Scores:")
        logger.info(df_m.to_string())
        
        # SEC1 should have higher momentum than SEC2
        assert df_m.loc['SEC1', 'momentum_score'] > df_m.loc['SEC2', 'momentum_score']
        
        # Mock PEAD Data
        df_earnings = pd.DataFrame([
            {'isin': 'SEC1', 'eps_current': 5.0, 'eps_seasonal_lag': 3.0, 'eps_std_dev': 0.5}, # Positive Surprise
            {'isin': 'SEC2', 'eps_current': 1.0, 'eps_seasonal_lag': 4.0, 'eps_std_dev': 1.0}, # Negative Surprise
        ])
        
        p_engine = PEADEngine()
        df_p = p_engine.calculate_scores(df_earnings)
        
        logger.info("\nPEAD Scores:")
        logger.info(df_p.to_string())
        
        # SEC1 should have positive SUE, SEC2 negative
        assert df_p[df_p['isin'] == 'SEC1']['sue'].values[0] > 0
        assert df_p[df_p['isin'] == 'SEC2']['sue'].values[0] < 0
        
        logger.info("\nSUCCESS: All Alpha Engines executed correctly.")
        
    except Exception as e:
        logger.error(f"Alpha engine test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_alpha_engines()
