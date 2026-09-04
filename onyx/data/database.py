import logging
from sqlalchemy import create_engine, Column, String, Integer, Float, Date, Boolean, MetaData, Table, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from onyx.core.config import config

logger = logging.getLogger(__name__)

Base = declarative_base()

class UDiffMarketData(Base):
    """
    SQLAlchemy model representing the UDiFF end-of-day cash market bhavcopy data.
    This stores the raw unadjusted prices to preserve the exact historical state.
    """
    __tablename__ = 'udiff_market_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_date = Column(Date, nullable=False, index=True)
    isin = Column(String(12), nullable=False, index=True)
    symbol = Column(String(50), nullable=False)
    series = Column(String(10), nullable=False)
    
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    last_traded_price = Column(Float)
    
    total_traded_qty = Column(Integer, nullable=False)
    total_traded_value = Column(Float, nullable=False)
    total_trades = Column(Integer, nullable=False)

class CorporateAction(Base):
    """
    SQLAlchemy model representing a corporate action (split, bonus, dividend).
    """
    __tablename__ = 'corporate_actions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    isin = Column(String(12), nullable=False, index=True)
    ex_date = Column(Date, nullable=False, index=True)
    action_type = Column(String(20), nullable=False)  # SPLIT, BONUS, DIVIDEND
    
    # Mathematical parameters for the adjustment formula
    ratio_a = Column(Float)  # e.g., 'A' new shares
    ratio_b = Column(Float)  # e.g., for every 'B' old shares
    dividend_amount = Column(Float)

class SecurityMasterHistory(Base):
    """
    Tracks the point-in-time state of the investable universe.
    """
    __tablename__ = 'security_master_history'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    isin = Column(String(12), nullable=False, index=True)
    symbol = Column(String(50), nullable=False)
    listing_date = Column(Date, nullable=False)
    delisting_date = Column(Date, nullable=True)
    is_active = Column(Boolean, default=True)

# Database Setup
engine = create_engine(config.db_connection_string, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Initializes the database schema."""
    logger.info("Initializing database schema...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database schema initialized.")

def get_session():
    """Dependency for providing a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
