import logging
import time
from datetime import datetime
from jugaad_data.nse import NSELive
import yfinance as yf
import pandas as pd

logger = logging.getLogger(__name__)

class DataPipeline:
    """
    Live and Historical Data Pipeline bypassing Akamai protections.
    Uses jugaad_data for live NSE quotes and yfinance for historical EOD data.
    """
    def __init__(self):
        self.n = NSELive()
        
    def fetch_live_quote(self, symbol: str) -> dict:
        """
        Fetches live price data for a single NSE symbol.
        Returns a dictionary with 'price' and 'volume'.
        """
        clean_symbol = symbol.replace('.NS', '')
        for attempt in range(3):
            try:
                data = self.n.stock_quote(clean_symbol)
                # Parse the JSON response for price and volume
                if data and 'priceInfo' in data:
                    last_price = data['priceInfo'].get('lastPrice', 0.0)
                    volume = data['preOpenMarket'].get('totalTradedVolume', 0) if 'preOpenMarket' in data else 0
                    return {
                        'symbol': clean_symbol,
                        'price': float(last_price),
                        'volume': int(volume),
                        'timestamp': datetime.now()
                    }
                else:
                    logger.warning(f"Unexpected response format for {symbol}: {data}")
            except Exception as e:
                logger.error(f"Attempt {attempt+1}: Failed to fetch equity quote for {symbol}: {e}")
                time.sleep(2)
                
        logger.error(f"Failed to fetch live quote for {symbol} after 3 attempts.")
        return None

    def fetch_historical_data(self, symbol: str, days: int = 250) -> pd.DataFrame:
        """
        Fetches historical EOD data via yfinance.
        """
        yf_symbol = symbol if symbol.endswith('.NS') else f"{symbol}.NS"
        try:
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(period=f"{days}d")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch historical data for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_universe_live(self, universe: list) -> dict:
        """
        Fetches live quotes for an entire universe of symbols.
        """
        results = {}
        for symbol in universe:
            quote = self.fetch_live_quote(symbol)
            if quote:
                results[symbol] = quote
            # Sleep slightly to avoid spamming the NSE server
            time.sleep(0.5)
        return results
