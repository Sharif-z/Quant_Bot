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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_integration_test():
    logger.info("Starting Phase 1 & 2 Integration Test...")
    ensure_directories()
    
    # 1. Initialize DB (Clean slate)
    if os.path.exists('onyx_trading.db'):
        os.remove('onyx_trading.db')
    init_db()
    
    db = SessionLocal()
    
    try:
        # --- PHASE 1: DATA ENGINEERING ---
        logger.info("--- Testing Phase 1 (Data Generation) ---")
        
        # Add security to master
        sec = SecurityMasterHistory(isin="INF001", symbol="MOCKCO", listing_date=datetime(2023, 1, 1).date(), is_active=True)
        db.add(sec)
        
        # Generate 150 days of fake UDiFF price data
        # We start price at 100 and drift it up to 200 over 150 days
        prices = np.linspace(100, 200, 150)
        dates = pd.date_range(start='2024-01-01', periods=150, freq='B') # Business days
        
        ingestion_engine = UDiffIngestionEngine(db)
        udiff_file = os.path.join(config.udiff_dir, "temp_udiff.csv")
        
        logger.info(f"Ingesting 150 days of historical data for INF001...")
        for i, dt in enumerate(dates):
            p = prices[i]
            df_fake = pd.DataFrame([{
                'ISIN': 'INF001', 'SYMBOL': 'MOCKCO', 'SERIES': 'EQ',
                'OPEN': p-1, 'HIGH': p+2, 'LOW': p-2, 'CLOSE': p, 'LAST': p,
                'TOTTRDQTY': 10000, 'TOTTRDVAL': p * 10000, 'TOTALTRADES': 500
            }])
            df_fake.to_csv(udiff_file, index=False)
            ingestion_engine.ingest_file(udiff_file, dt.date())
            
        # Add a Corporate Action: 2:1 Stock Split on day 100 (approx mid-May 2024)
        split_date = dates[100].date()
        logger.info(f"Injecting a 2:1 Stock Split on {split_date}...")
        split = CorporateAction(
            isin="INF001", ex_date=split_date, action_type="SPLIT", ratio_a=2.0, ratio_b=1.0
        )
        db.add(split)
        db.commit()
        
        # Query Point-in-Time Store on the final day
        final_date = dates[-1].date()
        pit_store = PointInTimeStore(db)
        
        logger.info(f"Extracting Point-In-Time adjusted price series as of {final_date}...")
        df_hist = pit_store.as_of("INF001", final_date, lookback_days=150)
        
        # Verify length and that adjusted prices exist
        assert len(df_hist) == 150, f"Expected 150 rows, got {len(df_hist)}"
        assert 'adj_close' in df_hist.columns, "adj_close column missing"
        
        # We need to adapt the PointInTimeStore output to the format MomentumEngine expects
        # (isin, trade_date, adj_close, total_traded_qty)
        df_hist['isin'] = 'INF001'
        df_hist_momentum = df_hist[['isin', 'trade_date', 'adj_close', 'total_traded_qty']].copy()
        
        # --- PHASE 2: ALPHA ENGINES ---
        logger.info("--- Testing Phase 2 (Alpha Generation) ---")
        
        # 1. Momentum Engine (Consuming Phase 1 Data)
        logger.info("Running Momentum Engine on Phase 1 Adjusted Prices...")
        m_engine = MomentumEngine(roc_period=100, fast_ma=10, slow_ma=30)
        df_m = m_engine.calculate_scores(df_hist_momentum)
        
        logger.info("Momentum Scores:")
        logger.info(df_m.to_string())
        assert 'momentum_score' in df_m.columns, "Momentum score calculation failed."
        
        # 2. Mock Fundamentals for Quality & Value
        logger.info("Running Quality & Value Engines...")
        df_fundamentals = pd.DataFrame([{
            'isin': 'INF001', 'gross_profit': 1500, 'total_assets': 3000, 
            'net_income': 400, 'invested_capital': 2500, 'total_debt': 500, 
            'total_equity': 2000, 'book_value': 2100, 'market_cap': 5000, 'earnings_ttm': 450
        }])
        
        q_engine = QualityEngine()
        df_q = q_engine.calculate_scores(df_fundamentals)
        assert df_q['is_investable'].values[0] == True, "Quality engine failed."
        
        v_engine = ValueEngine(quality_engine=q_engine)
        df_v = v_engine.calculate_scores(df_fundamentals)
        assert 'qarp_score' in df_v.columns, "Value engine failed."
        
        # 3. PEAD Engine
        logger.info("Running PEAD Engine...")
        df_earnings = pd.DataFrame([{
            'isin': 'INF001', 'eps_current': 12.0, 'eps_seasonal_lag': 10.0, 'eps_std_dev': 1.5
        }])
        p_engine = PEADEngine()
        df_p = p_engine.calculate_scores(df_earnings)
        assert df_p['sue'].values[0] > 0, "PEAD engine failed."
        
        logger.info("SUCCESS: Phase 1 & 2 Integration Test completed with zero errors.")
        
    except Exception as e:
        logger.error(f"Integration Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        db.close()
        # Cleanup temp file
        if os.path.exists(os.path.join(config.udiff_dir, "temp_udiff.csv")):
            os.remove(os.path.join(config.udiff_dir, "temp_udiff.csv"))

if __name__ == "__main__":
    run_integration_test()
