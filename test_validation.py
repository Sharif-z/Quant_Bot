import logging
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys

from onyx.core.config import config, ensure_directories
from onyx.data.database import init_db, SessionLocal, SecurityMasterHistory
from onyx.data.udiff_ingestion import UDiffIngestionEngine
from onyx.data.point_in_time import PointInTimeStore

from onyx.alpha.momentum import MomentumEngine
from onyx.portfolio.covariance import CovarianceEstimator
from onyx.portfolio.optimizer import ConvexOptimizer
from onyx.execution.simulator import ExecutionSimulator
from onyx.validation.walkforward import PurgedWalkForward

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_walkforward_engine():
    logger.info("Starting Walk-Forward Validation Test...")
    ensure_directories()
    
    # 1. DB Init
    if os.path.exists('onyx_trading.db'):
        os.remove('onyx_trading.db')
    init_db()
    db = SessionLocal()
    
    try:
        # Mock Data Injection (3 Assets, 250 days of data to allow multiple rebalances)
        isins = ['INF001', 'INF002', 'INF003']
        symbols = ['ALPHA', 'BETA', 'GAMMA']
        for i, isin in enumerate(isins):
            db.add(SecurityMasterHistory(isin=isin, symbol=symbols[i], listing_date=datetime(2023, 1, 1).date(), is_active=True))
            
        dates = pd.date_range(start='2024-01-01', periods=250, freq='B')
        np.random.seed(42)
        
        # Price paths (Trending up to generate some alpha)
        p1 = np.linspace(100, 200, 250) + np.random.normal(0, 2, 250)
        p2 = np.linspace(300, 400, 250) + np.random.normal(0, 5, 250)
        p3 = np.linspace(50, 80, 250) + np.random.normal(0, 1, 250)
        
        ingestion = UDiffIngestionEngine(db)
        udiff_file = os.path.join(config.udiff_dir, "temp_udiff_wf.csv")
        
        logger.info(f"Ingesting 250 days of historical data...")
        for i, dt in enumerate(dates):
            df_fake = pd.DataFrame([
                {'ISIN': 'INF001', 'SYMBOL': 'ALPHA', 'SERIES': 'EQ', 'OPEN': p1[i]-1, 'HIGH': p1[i]+1, 'LOW': p1[i]-1, 'CLOSE': p1[i], 'LAST': p1[i], 'TOTTRDQTY': 150000, 'TOTTRDVAL': p1[i] * 150000, 'TOTALTRADES': 5000},
                {'ISIN': 'INF002', 'SYMBOL': 'BETA', 'SERIES': 'EQ', 'OPEN': p2[i]-1, 'HIGH': p2[i]+1, 'LOW': p2[i]-1, 'CLOSE': p2[i], 'LAST': p2[i], 'TOTTRDQTY': 200000, 'TOTTRDVAL': p2[i] * 200000, 'TOTALTRADES': 6000},
                {'ISIN': 'INF003', 'SYMBOL': 'GAMMA', 'SERIES': 'EQ', 'OPEN': p3[i]-1, 'HIGH': p3[i]+1, 'LOW': p3[i]-1, 'CLOSE': p3[i], 'LAST': p3[i], 'TOTTRDQTY': 500000, 'TOTTRDVAL': p3[i] * 500000, 'TOTALTRADES': 10000}
            ])
            df_fake.to_csv(udiff_file, index=False)
            ingestion.ingest_file(udiff_file, dt.date())
            
        # 2. Setup Engines
        pit_store = PointInTimeStore(db)
        alpha_engines = {'momentum': MomentumEngine(roc_period=50, fast_ma=10, slow_ma=30)}
        cov_engine = CovarianceEstimator(method='ledoit_wolf')
        optimizer = ConvexOptimizer(lambda_risk=2.0, max_weight=0.50)
        simulator = ExecutionSimulator(volatility_factor=0.015)
        
        # 3. Setup Walk-Forward backtester
        wf = PurgedWalkForward(
            pit_store=pit_store,
            alpha_engines=alpha_engines,
            covariance_engine=cov_engine,
            optimizer=optimizer,
            simulator=simulator,
            cash_buffer=0.98 # Keep 2% cash buffer to fund friction
        )
        
        # Define rebalance dates (Every 30 business days, starting from day 150 to allow history)
        rebalance_dates = [dates[150], dates[180], dates[210], dates[240]]
        
        res = wf.run_backtest(
            universe_isins=isins,
            rebalance_dates=rebalance_dates,
            initial_cash=1_000_000.0
        )
        
        logger.info("\nSUCCESS: Walk-Forward Engine executed completely.")
        
    except Exception as e:
        logger.error(f"Walk-Forward Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        db.close()
        if os.path.exists(os.path.join(config.udiff_dir, "temp_udiff_wf.csv")):
            os.remove(os.path.join(config.udiff_dir, "temp_udiff_wf.csv"))

if __name__ == "__main__":
    test_walkforward_engine()
