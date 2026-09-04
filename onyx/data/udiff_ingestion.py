import os
import pandas as pd
import logging
from datetime import datetime
from sqlalchemy.orm import Session
from onyx.data.database import UDiffMarketData
from onyx.core.config import config

logger = logging.getLogger(__name__)

class UDiffIngestionEngine:
    """
    Handles the parsing and ingestion of NSE UDiFF format end-of-day data.
    """
    def __init__(self, db_session: Session):
        self.db = db_session

    def validate_and_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the Data Quality Firewall rules.
        - Removes rows with missing critical prices.
        - Filters to only 'EQ' (Equity) series to ignore bonds/warrants unless specified.
        - Ensures ISIN is present.
        """
        # Ensure required columns exist
        required_cols = ['ISIN', 'SYMBOL', 'SERIES', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'TOTTRDQTY', 'TOTTRDVAL', 'TOTALTRADES']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in UDiFF file: {missing_cols}")
        
        # Filter for Equity series
        # In a real system, we might want BE (Book Entry), SM (SME), but EQ is standard cash market.
        df_filtered = df[df['SERIES'].isin(['EQ', 'BE'])].copy()
        
        # Drop rows with NaN in critical price columns
        df_filtered.dropna(subset=['OPEN', 'HIGH', 'LOW', 'CLOSE', 'ISIN'], inplace=True)
        
        # Ensure prices are strictly positive
        df_filtered = df_filtered[(df_filtered['OPEN'] > 0) & (df_filtered['CLOSE'] > 0)]
        
        return df_filtered

    def ingest_file(self, file_path: str, trade_date: datetime.date):
        """
        Ingests a specific UDiFF CSV file into the point-in-time store.
        """
        logger.info(f"Starting ingestion for {file_path} for date {trade_date}")
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False
            
        try:
            df = pd.read_csv(file_path)
            
            # Clean column names (strip whitespace)
            df.columns = df.columns.str.strip()
            
            # Apply firewall
            df_clean = self.validate_and_filter(df)
            
            # Convert to list of dictionaries for bulk insert
            records = []
            for _, row in df_clean.iterrows():
                record = UDiffMarketData(
                    trade_date=trade_date,
                    isin=row['ISIN'].strip(),
                    symbol=row['SYMBOL'].strip(),
                    series=row['SERIES'].strip(),
                    open_price=float(row['OPEN']),
                    high_price=float(row['HIGH']),
                    low_price=float(row['LOW']),
                    close_price=float(row['CLOSE']),
                    last_traded_price=float(row.get('LAST', row['CLOSE'])),
                    total_traded_qty=int(row['TOTTRDQTY']),
                    total_traded_value=float(row['TOTTRDVAL']),
                    total_trades=int(row.get('TOTALTRADES', 0))
                )
                records.append(record)
                
            # Bulk save
            self.db.bulk_save_objects(records)
            self.db.commit()
            logger.info(f"Successfully ingested {len(records)} records for {trade_date}")
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to ingest UDiFF file {file_path}: {str(e)}")
            return False
