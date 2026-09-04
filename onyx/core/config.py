import os
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class TransactionCosts:
    """
    Transaction cost parameters for the Indian market, based on SEBI 2026 guidelines.
    """
    stt_delivery: float = 0.001  # 0.1% on Buy & Sell for delivery
    stt_intraday: float = 0.00025  # 0.025% on Sell only for intraday
    stt_futures: float = 0.0002  # 0.02% on Sell only for futures
    exchange_transaction_charge: float = 0.0000325  # ~0.00325%
    stamp_duty_delivery: float = 0.00015  # 0.015% on Buy
    stamp_duty_intraday: float = 0.00003  # 0.003% on Buy
    sebi_turnover_fee: float = 0.000001  # 0.0001% (10 per crore)
    gst_rate: float = 0.18  # 18% on Brokerage + Exchange + SEBI fees

@dataclass
class SystemConfig:
    """
    Global system configuration settings.
    """
    # SQLite is used for the research/paper mode.
    # In live mode, this should be a connection string to PostgreSQL or similar.
    db_connection_string: str = os.getenv("ONYX_DB_URL", "sqlite:///onyx_trading.db")
    
    # Portfolio constraints
    initial_capital: float = 1_000_0000.0  # 1 Crore paper portfolio
    max_ops: int = 10  # 10 Orders per second threshold
    
    # Paths for data storage
    data_dir: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data_storage")
    udiff_dir: str = os.path.join(data_dir, "udiff")

    costs: TransactionCosts = field(default_factory=TransactionCosts)

# Global config instance
config = SystemConfig()

def ensure_directories():
    """Ensure required data directories exist."""
    os.makedirs(config.data_dir, exist_ok=True)
    os.makedirs(config.udiff_dir, exist_ok=True)

# Initialize directories on load
ensure_directories()
