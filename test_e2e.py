import logging
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys

# Phase 1 Imports
from onyx.core.config import config, ensure_directories
from onyx.data.database import init_db, SessionLocal, CorporateAction, SecurityMasterHistory
from onyx.data.udiff_ingestion import UDiffIngestionEngine
from onyx.data.point_in_time import PointInTimeStore

# Phase 2 Imports
from onyx.alpha.quality import QualityEngine
from onyx.alpha.value import ValueEngine
from onyx.alpha.momentum import MomentumEngine
from onyx.alpha.pead import PEADEngine

# Phase 4 Imports
from onyx.portfolio.covariance import CovarianceEstimator
from onyx.portfolio.optimizer import ConvexOptimizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_end_to_end_test():
    logger.info("Starting Master End-to-End Test (Phases 1, 2, & 4)...")
    ensure_directories()
    
    # 1. Clean Slate DB
    if os.path.exists('onyx_trading.db'):
        os.remove('onyx_trading.db')
    init_db()
    db = SessionLocal()
    
    try:
        # --- PHASE 1: DATA ENGINEERING ---
        logger.info("\n=== PHASE 1: DATA INGESTION & POINT-IN-TIME ===")
        # Add 3 securities to master
        isins = ['INF001', 'INF002', 'INF003']
        symbols = ['ALPHA', 'BETA', 'GAMMA']
        for i, isin in enumerate(isins):
            db.add(SecurityMasterHistory(isin=isin, symbol=symbols[i], listing_date=datetime(2023, 1, 1).date(), is_active=True))
        
        # Generate 150 days of fake UDiFF price data for 3 assets
        dates = pd.date_range(start='2024-01-01', periods=150, freq='B')
        np.random.seed(42)
        
        # Price paths
        # INF001: Strong uptrend
        p1 = np.linspace(100, 200, 150) + np.random.normal(0, 2, 150)
        # INF002: Downtrend
        p2 = np.linspace(300, 150, 150) + np.random.normal(0, 5, 150)
        # INF003: Sideways / slightly up
        p3 = np.linspace(50, 60, 150) + np.random.normal(0, 1, 150)
        
        ingestion_engine = UDiffIngestionEngine(db)
        udiff_file = os.path.join(config.udiff_dir, "temp_udiff_e2e.csv")
        
        logger.info(f"Ingesting 150 days of historical data for 3 equities...")
        for i, dt in enumerate(dates):
            df_fake = pd.DataFrame([
                {'ISIN': 'INF001', 'SYMBOL': 'ALPHA', 'SERIES': 'EQ', 'OPEN': p1[i]-1, 'HIGH': p1[i]+1, 'LOW': p1[i]-1, 'CLOSE': p1[i], 'LAST': p1[i], 'TOTTRDQTY': 150000, 'TOTTRDVAL': p1[i] * 150000, 'TOTALTRADES': 5000},
                {'ISIN': 'INF002', 'SYMBOL': 'BETA', 'SERIES': 'EQ', 'OPEN': p2[i]-1, 'HIGH': p2[i]+1, 'LOW': p2[i]-1, 'CLOSE': p2[i], 'LAST': p2[i], 'TOTTRDQTY': 200000, 'TOTTRDVAL': p2[i] * 200000, 'TOTALTRADES': 6000},
                {'ISIN': 'INF003', 'SYMBOL': 'GAMMA', 'SERIES': 'EQ', 'OPEN': p3[i]-1, 'HIGH': p3[i]+1, 'LOW': p3[i]-1, 'CLOSE': p3[i], 'LAST': p3[i], 'TOTTRDQTY': 500000, 'TOTTRDVAL': p3[i] * 500000, 'TOTALTRADES': 10000}
            ])
            df_fake.to_csv(udiff_file, index=False)
            ingestion_engine.ingest_file(udiff_file, dt.date())
            
        # Add Corporate Action: 2:1 Stock Split on INF001 on day 100
        split_date = dates[100].date()
        logger.info(f"Injecting a 2:1 Stock Split on {split_date} for INF001...")
        db.add(CorporateAction(isin="INF001", ex_date=split_date, action_type="SPLIT", ratio_a=2.0, ratio_b=1.0))
        db.commit()
        
        # Query Point-in-Time Store
        final_date = dates[-1].date()
        pit_store = PointInTimeStore(db)
        
        logger.info(f"Extracting Point-In-Time adjusted history as of {final_date}...")
        df_hist = pd.concat([
            pit_store.as_of("INF001", final_date, lookback_days=150).assign(isin="INF001"),
            pit_store.as_of("INF002", final_date, lookback_days=150).assign(isin="INF002"),
            pit_store.as_of("INF003", final_date, lookback_days=150).assign(isin="INF003")
        ])
        
        # --- PHASE 2: ALPHA ENGINES ---
        logger.info("\n=== PHASE 2: CROSS-SECTIONAL ALPHA GENERATION ===")
        
        # 1. Momentum (Reads directly from Phase 1 history)
        m_engine = MomentumEngine(roc_period=100, fast_ma=10, slow_ma=30)
        df_hist_momentum = df_hist[['isin', 'trade_date', 'adj_close', 'total_traded_qty']].copy()
        df_m = m_engine.calculate_scores(df_hist_momentum)
        
        # 2. Mock Fundamentals for Quality & Value
        df_fundamentals = pd.DataFrame([
            {'isin': 'INF001', 'gross_profit': 1500, 'total_assets': 3000, 'net_income': 400, 'invested_capital': 2500, 'total_debt': 500, 'total_equity': 2000, 'book_value': 2100, 'market_cap': 5000, 'earnings_ttm': 450},
            {'isin': 'INF002', 'gross_profit': 100, 'total_assets': 3000, 'net_income': -50, 'invested_capital': 2500, 'total_debt': 2800, 'total_equity': 200, 'book_value': 100, 'market_cap': 200, 'earnings_ttm': -100}, # Junk
            {'isin': 'INF003', 'gross_profit': 800, 'total_assets': 2000, 'net_income': 200, 'invested_capital': 1500, 'total_debt': 100, 'total_equity': 1900, 'book_value': 1950, 'market_cap': 1000, 'earnings_ttm': 250}
        ])
        
        q_engine = QualityEngine()
        df_q = q_engine.calculate_scores(df_fundamentals)
        
        v_engine = ValueEngine(quality_engine=q_engine)
        df_v = v_engine.calculate_scores(df_fundamentals)
        
        # 3. Aggregate Alphas
        # Simple equal weight of Momentum and Value (QARP)
        df_alpha = pd.DataFrame(index=isins)
        df_alpha['momentum'] = df_m['momentum_score']
        
        df_v['isin'] = df_fundamentals['isin']
        df_alpha['value'] = df_v.set_index('isin')['qarp_score']
        
        df_alpha['total_alpha'] = df_alpha['momentum'] + df_alpha['value']
        
        logger.info(f"\nFinal Consolidated Alpha Scores:\n{df_alpha.to_string()}")
        
        # --- PHASE 4: PORTFOLIO CONSTRUCTION ---
        logger.info("\n=== PHASE 4: DETERMINISTIC PORTFOLIO OPTIMIZATION ===")
        
        # Format returns for covariance estimation
        df_hist_pivot = df_hist.pivot(index='trade_date', columns='isin', values='adj_close')
        returns = df_hist_pivot.pct_change().dropna()
        
        # Calculate Pure Numpy Ledoit-Wolf Covariance
        cov_engine = CovarianceEstimator(method='ledoit_wolf')
        cov_matrix = cov_engine.estimate(returns)
        
        logger.info(f"\nLedoit-Wolf Covariance Matrix:\n{cov_matrix.to_string()}")
        
        # Setup Convex Optimizer
        # ADV for liquidity constraints
        adv_data = df_hist.groupby('isin')['total_traded_qty'].mean() * 100 # Approx 100 rupees per share
        
        optimizer = ConvexOptimizer(lambda_risk=2.0, max_weight=0.50)
        
        logger.info("\nRunning SLSQP Optimization...")
        optimal_weights = optimizer.optimize(
            alpha_scores=df_alpha['total_alpha'], 
            cov_matrix=cov_matrix,
            adv_data=adv_data,
            portfolio_value=1_000_000.0, # 10 Lakh test
            max_participation=0.05
        )
        
        logger.info(f"\nFinal Optimal Portfolio Weights:\n{optimal_weights.to_string()}")
        
        # Verification
        assert np.isclose(optimal_weights.sum(), 1.0), "Optimizer failed budget constraint."
        assert optimal_weights['INF002'] == 0.0, "Optimizer failed to exclude junk/value trap."
        
        logger.info("\nSUCCESS: End-to-End Pipeline (Phase 1 -> Phase 2 -> Phase 4) executed flawlessly.")
        
    except Exception as e:
        logger.error(f"End-to-End Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        db.close()
        if os.path.exists(os.path.join(config.udiff_dir, "temp_udiff_e2e.csv")):
            os.remove(os.path.join(config.udiff_dir, "temp_udiff_e2e.csv"))

if __name__ == "__main__":
    run_end_to_end_test()
