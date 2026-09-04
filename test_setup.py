import logging
import os
import pandas as pd
from datetime import datetime
from onyx.core.config import config, ensure_directories
from onyx.data.database import init_db, SessionLocal, CorporateAction, SecurityMasterHistory
from onyx.data.udiff_ingestion import UDiffIngestionEngine
from onyx.data.point_in_time import PointInTimeStore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_phase1():
    logger.info("Starting Phase 1 verification test...")
    ensure_directories()
    
    # 1. Initialize DB
    if os.path.exists('onyx_trading.db'):
        os.remove('onyx_trading.db')
    init_db()
    
    db = SessionLocal()
    
    try:
        # 2. Add a security to the master history
        logger.info("Adding security INF000000001 to master...")
        sec = SecurityMasterHistory(
            isin="INF000000001",
            symbol="TESTCORP",
            listing_date=datetime(2025, 1, 1).date(),
            is_active=True
        )
        db.add(sec)
        db.commit()
        
        # 3. Create a fake UDiFF CSV for 2025-01-10
        udiff_file = os.path.join(config.udiff_dir, "test_udiff.csv")
        df_fake = pd.DataFrame([{
            'ISIN': 'INF000000001',
            'SYMBOL': 'TESTCORP',
            'SERIES': 'EQ',
            'OPEN': 100.0,
            'HIGH': 105.0,
            'LOW': 98.0,
            'CLOSE': 102.0,
            'LAST': 102.0,
            'TOTTRDQTY': 5000,
            'TOTTRDVAL': 505000.0,
            'TOTALTRADES': 120
        }])
        df_fake.to_csv(udiff_file, index=False)
        
        # 4. Ingest the UDiFF file
        logger.info("Ingesting UDiFF file...")
        ingestion_engine = UDiffIngestionEngine(db)
        success = ingestion_engine.ingest_file(udiff_file, datetime(2025, 1, 10).date())
        assert success, "UDiFF Ingestion failed."
        
        # 5. Add a corporate action (2:1 Stock Split on 2025-01-12)
        logger.info("Injecting a 2:1 Stock Split corporate action...")
        split = CorporateAction(
            isin="INF000000001",
            ex_date=datetime(2025, 1, 12).date(),
            action_type="SPLIT",
            ratio_a=2.0,  # 2 new shares
            ratio_b=1.0   # 1 old share
        )
        db.add(split)
        db.commit()
        
        # 6. Query Point-In-Time Store on 2025-01-15 (after split)
        # Expected: RAW close is 102.0, ADJ close is 51.0
        logger.info("Querying Point-in-Time Store as of 2025-01-15...")
        pit_store = PointInTimeStore(db)
        
        universe = pit_store.get_investable_universe(datetime(2025, 1, 15).date())
        assert "INF000000001" in universe, "Security not in investable universe."
        
        df_hist = pit_store.as_of("INF000000001", datetime(2025, 1, 15).date())
        
        if not df_hist.empty:
            raw_close = df_hist.iloc[0]['close_price']
            adj_close = df_hist.iloc[0]['adj_close']
            
            logger.info(f"Raw Close Price: {raw_close}")
            logger.info(f"Adjusted Close Price: {adj_close}")
            
            assert raw_close == 102.0, "Raw close price mismatch."
            assert adj_close == 51.0, "Adjusted close price mismatch. Corporate Action engine failed."
            logger.info("SUCCESS: Phase 1 Data Engineering stack verified successfully.")
        else:
            logger.error("Historical data query returned empty.")
            
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()
        
if __name__ == "__main__":
    test_phase1()
