import os
import requests
import logging
from datetime import datetime
import json
import sqlite3

logger = logging.getLogger(__name__)

# Attempt to load .env file if dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logger.warning("python-dotenv not installed. Falling back to raw os.environ.")

class NewsWaterfallScraper:
    """
    Phase 13: Event-Driven NLP Waterfall.
    Rationed API calling structure to maximize free tiers for NLP Sentiment data.
    """
    def __init__(self, db_path: str = "onyx_state.db"):
        self.currents_key = os.getenv("CURRENTS_API_KEY")
        self.indian_api_key = os.getenv("INDIAN_API_KEY")
        self.marketaux_key = os.getenv("MARKETAUX_API_KEY")
        self.newsapi_key = os.getenv("NEWSAPI_ORG_KEY")
        self.db_path = db_path
        
        # Ensure the DB schema for news exists
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS news_sentiment (
                date TEXT,
                ticker TEXT,
                news_volume INTEGER,
                marketaux_sentiment REAL,
                surgical_flag INTEGER,
                PRIMARY KEY (date, ticker)
            )
        """)
        conn.commit()
        conn.close()

    def layer1_macro_sweep(self, universe: list) -> dict:
        """
        Layer 1: Currents API + NewsAPI.
        Pulls broad Indian business news and counts occurrences of our tickers.
        """
        logger.info("Executing Layer 1: Macro Sweep (Currents & NewsAPI)")
        news_volume_map = {ticker: 0 for ticker in universe}
        
        # NewsAPI Call
        if self.newsapi_key:
            try:
                url = f"https://newsapi.org/v2/top-headlines?country=in&category=business&apiKey={self.newsapi_key}"
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    articles = response.json().get('articles', [])
                    for article in articles:
                        text = (article.get('title') or "") + " " + (article.get('description') or "")
                        for ticker in universe:
                            if ticker in text or ticker.replace("BANK", " BANK") in text:
                                news_volume_map[ticker] += 1
            except Exception as e:
                logger.error(f"NewsAPI request failed: {e}")
                
        # Currents API Call
        if self.currents_key:
            try:
                url = f"https://api.currentsapi.services/v1/latest-news?country=IN&category=business&apiKey={self.currents_key}"
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    news = response.json().get('news', [])
                    for article in news:
                        text = (article.get('title') or "") + " " + (article.get('description') or "")
                        for ticker in universe:
                            if ticker in text or ticker.replace("BANK", " BANK") in text:
                                news_volume_map[ticker] += 1
            except Exception as e:
                logger.error(f"Currents API request failed: {e}")

        return news_volume_map

    def layer2_targeted_sentiment(self, targets: list) -> dict:
        """
        Layer 2: Marketaux API.
        Extracts pre-calculated AI sentiment scores for the top target stocks.
        """
        logger.info(f"Executing Layer 2: Targeted Sentiment for {len(targets)} stocks (Marketaux)")
        sentiment_map = {ticker: 0.0 for ticker in targets}
        
        if not self.marketaux_key or not targets:
            return sentiment_map
            
        try:
            # We can request multiple symbols in one go to save API limits
            symbols_str = ",".join([f"{t}.NS" for t in targets]) 
            url = f"https://api.marketaux.com/v1/news/all?symbols={symbols_str}&filter_entities=true&api_token={self.marketaux_key}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json().get('data', [])
                # Aggregate sentiment per ticker
                for item in data:
                    for entity in item.get('entities', []):
                        ticker = entity.get('symbol', '').replace(".NS", "")
                        score = entity.get('sentiment_score', 0.0)
                        if ticker in sentiment_map:
                            # Averaging or keeping latest; we will just sum it for simplicity
                            sentiment_map[ticker] += score
            else:
                logger.warning(f"Marketaux API returned status code {response.status_code}")
        except Exception as e:
            logger.error(f"Marketaux API request failed: {e}")
            
        return sentiment_map

    def layer3_surgical_strike(self, high_conviction_targets: list) -> dict:
        """
        Layer 3: IndianAPI.in
        Highly rationed API call exclusively to verify >90% conviction trades.
        """
        logger.info(f"Executing Layer 3: Surgical Strike for {len(high_conviction_targets)} stocks (IndianAPI)")
        surgical_flags = {ticker: 0 for ticker in high_conviction_targets}
        
        if not self.indian_api_key or not high_conviction_targets:
            return surgical_flags
            
        for ticker in high_conviction_targets:
            try:
                # E.g. pulling corporate actions
                url = f"https://indianapi.in/api/v1/corporate-actions?symbol={ticker}"
                headers = {"Authorization": f"Bearer {self.indian_api_key}"}
                response = requests.get(url, headers=headers, timeout=5)
                
                if response.status_code == 200:
                    data = response.json()
                    # Just flag if recent announcements exist (mock logic)
                    if len(data) > 0:
                        surgical_flags[ticker] = 1
            except Exception as e:
                logger.error(f"IndianAPI request failed for {ticker}: {e}")
                
        return surgical_flags

    def execute_waterfall(self, universe: list, optimizer_targets: list, high_conviction_targets: list):
        """
        Executes the 3-Layer Waterfall and stores the result in SQLite.
        """
        today_str = datetime.now().strftime("%Y-%m-%d")
        
        # 1. Macro Sweep
        vol_map = self.layer1_macro_sweep(universe)
        
        # 2. Targeted Sentiment
        sent_map = self.layer2_targeted_sentiment(optimizer_targets)
        
        # 3. Surgical Strike
        surg_map = self.layer3_surgical_strike(high_conviction_targets)
        
        # 4. Save to DB
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for ticker in universe:
            v = vol_map.get(ticker, 0)
            s = sent_map.get(ticker, 0.0)
            sf = surg_map.get(ticker, 0)
            
            # Using REPLACE to handle upserts gracefully
            cursor.execute("""
                INSERT OR REPLACE INTO news_sentiment 
                (date, ticker, news_volume, marketaux_sentiment, surgical_flag)
                VALUES (?, ?, ?, ?, ?)
            """, (today_str, ticker, v, s, sf))
            
        conn.commit()
        conn.close()
        logger.info("Waterfall Execution Complete. Data committed to SQLite.")
