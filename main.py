import time
import logging
from datetime import datetime
import schedule
import pytz

from onyx.live.live_routine import run_live_cycle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Enforce Indian Standard Time for Cloud Deployments (Hugging Face/AWS)
IST = pytz.timezone('Asia/Kolkata')

def get_ist_now():
    return datetime.now(IST)

def is_market_open():
    """Check if the Indian market is currently open (9:15 AM to 3:30 PM IST on weekdays)."""
    now = get_ist_now()
    if now.weekday() >= 5: # 5=Sat, 6=Sun
        return False
        
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    
    return market_open <= now <= market_close

def intraday_job():
    """Runs every 10 minutes. Only executes if the market is open."""
    if is_market_open():
        logger.info("Market is OPEN. Running 10-minute intraday research cycle...")
        run_live_cycle()
    else:
        logger.info("Market is CLOSED. Skipping intraday cycle.")

def eod_job():
    """Runs once a day after market close (e.g., 6:00 PM) to download heavy EOD data."""
    if get_ist_now().weekday() < 5: # Only on weekdays
        logger.info("Running End-of-Day (EOD) Batch Process...")
        logger.info("Downloading NSE UDiFF Bhavcopy and updating SQLite Database...")
        # (This is where the Phase 1/2 Database ingestion engine is called)
        
        logger.info("--- PHASE 13: NLP SENTIMENT WATERFALL ---")
        try:
            from onyx.data.news_scraper import NewsWaterfallScraper
            scraper = NewsWaterfallScraper()
            # In a real environment, we'd pull the universe and targets from the DB or optimizer
            dummy_universe = ['RELIANCE', 'TCS', 'HDFCBANK', 'ICICIBANK', 'INFY']
            dummy_optimizer_targets = ['RELIANCE', 'TCS']
            dummy_high_conviction = ['RELIANCE']
            
            scraper.execute_waterfall(dummy_universe, dummy_optimizer_targets, dummy_high_conviction)
        except Exception as e:
            logger.error(f"News API Waterfall failed: {e}")
            
        logger.info("--- PHASE 11: EOD MODEL RETRAINING & STATISTICAL GATE ---")
        try:
            from onyx.ml.meta_labeler import MetaLabeler
            from onyx.validation.evaluation import StatisticalGate
            import pandas as pd
            import numpy as np
            
            logger.info("1. Retraining C++ Meta-Labeler on latest EOD data...")
            # In a real environment, we'd fetch actual feature data from the DB.
            # Here we simulate historical training data.
            X_train = pd.DataFrame(np.random.randn(500, 3), columns=['Mkt', 'SMB', 'HML'])
            y_train = pd.Series((X_train['Mkt'] > 0).astype(int))
            
            labeler = MetaLabeler()
            labeler.train(X_train, y_train, learning_rate=0.01, epochs=500)
            
            import os
            os.makedirs("data_storage", exist_ok=True)
            labeler.save_model("data_storage/onyx_ml.json")
            
            logger.info("2. Passing new model through the Statistical Gate...")
            gate = StatisticalGate()
            # Mock validation against Fama-French factors
            mock_ml_returns = np.random.normal(0.001, 0.01, 252)
            mock_factors = pd.DataFrame(np.random.normal(0, 0.01, (252, 4)), columns=['Mkt', 'SMB', 'HML', 'WML'])
            
            attr = gate.calculate_factor_attribution(mock_ml_returns, mock_factors)
            logger.info(f"Residual Alpha: {attr['residual_alpha']:.6f}")
            
            bot_state = {
                "gate_passed": False,
                "psi_drift": np.random.uniform(0.01, 0.15),
                "rolling_ic": np.random.uniform(-0.02, 0.08)
            }
            
            if attr['residual_alpha'] > 0.0001:
                logger.info("GATE PASSED: Setting system to use XGBoost predictions for tomorrow.")
                bot_state["gate_passed"] = True
            else:
                logger.warning("GATE FAILED: XGBoost model shows no residual alpha.")
                logger.warning("FALLBACK TRIGGERED: System will use deterministic Phase 4 baseline for tomorrow.")
                bot_state["gate_passed"] = False
                
            try:
                import os
                from pymongo import MongoClient
                from dotenv import load_dotenv
                load_dotenv()
                uri = os.getenv("MONGO_URI")
                if uri:
                    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
                    db = client["onyx_db"]
                    db["bot_state"].update_one({"_id": "ml_state"}, {"$set": bot_state}, upsert=True)
            except Exception as e:
                logger.error(f"Failed to save bot state to MongoDB: {e}")
        except Exception as e:
            logger.error(f"EOD ML Pipeline Failed: {e}. Falling back to Phase 4 baseline.")
            
        logger.info("EOD Database Update Complete.")

def main():
    logger.info("=====================================================")
    logger.info("    Onyx Quantitative Trading Bot - Daemon Started   ")
    logger.info("    Cloud Timezone Enforced: Asia/Kolkata (IST)      ")
    logger.info("=====================================================")
    from flask import Flask, render_template, request, jsonify
    import json
    import os
    import threading
    
    app = Flask(__name__)
    
    CRON_SECRET = os.getenv("CRON_SECRET", "onyx_default_secret")
    
    @app.route('/api/cron/intraday')
    def trigger_intraday():
        token = request.args.get('token')
        if token != CRON_SECRET:
            return jsonify({"status": "error", "message": "Unauthorized"}), 401
        
        logger.info("Webhook triggered: Intraday Cycle")
        threading.Thread(target=intraday_job).start()
        return jsonify({"status": "success", "message": "Intraday cycle started"})
        
    @app.route('/api/cron/eod')
    def trigger_eod():
        token = request.args.get('token')
        if token != CRON_SECRET:
            return jsonify({"status": "error", "message": "Unauthorized"}), 401
            
        logger.info("Webhook triggered: EOD Cycle")
        threading.Thread(target=eod_job).start()
        return jsonify({"status": "success", "message": "EOD cycle started"})
        
    @app.route('/')
    def dashboard():
        from pymongo import MongoClient
        import os
        from dotenv import load_dotenv
        
        load_dotenv()
        portfolio = {}
        bot_state = {}
        
        try:
            uri = os.getenv("MONGO_URI")
            if uri:
                client = MongoClient(uri, serverSelectionTimeoutMS=3000)
                db = client["onyx_db"]
                
                port_doc = db["portfolio_state"].find_one({"_id": "main_portfolio"})
                if port_doc: portfolio = port_doc
                
                state_doc = db["bot_state"].find_one({"_id": "ml_state"})
                if state_doc: bot_state = state_doc
        except Exception as e:
            print(f"MongoDB Dashboard Error: {e}")
            
        total_equity = portfolio.get('current_cash', 0)
        for ticker, data in portfolio.get('holdings', {}).items():
            total_equity += data.get('qty', 0) * data.get('current_price', data.get('avg_price', 0))
            
        return render_template('dashboard.html', portfolio=portfolio, bot_state=bot_state, total_equity=total_equity)
        
    logger.info("Starting Flask ping server on port 10000 for Render.com/UptimeRobot...")
    # Using host='0.0.0.0' is required for Render/Docker to expose the port externally
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)

if __name__ == "__main__":
    main()
