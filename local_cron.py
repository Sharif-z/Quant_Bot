import time
import requests
import schedule
from datetime import datetime
import pytz
import logging
import os
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load secret token
load_dotenv()
CRON_SECRET = os.getenv("CRON_SECRET", "onyx_default_secret")

# This is your live production bot on Render!
RENDER_URL = os.getenv("RENDER_URL", "https://quant-bot-ntcm.onrender.com") 

IST = pytz.timezone('Asia/Kolkata')

def trigger_intraday():
    logger.info("Pinging Remote Bot: INTRADAY CYCLE")
    try:
        res = requests.get(f"{RENDER_URL}/api/cron/intraday", params={"token": CRON_SECRET}, timeout=10)
        if res.status_code == 200:
            logger.info("✅ Remote Server Acknowledged Intraday Trigger.")
        else:
            logger.error(f"❌ Failed to trigger. Status: {res.status_code} - {res.text}")
    except Exception as e:
        logger.error(f"❌ Connection Error: {e}")

def trigger_eod():
    logger.info("Pinging Remote Bot: EOD BATCH PROCESS")
    try:
        res = requests.get(f"{RENDER_URL}/api/cron/eod", params={"token": CRON_SECRET}, timeout=10)
        if res.status_code == 200:
            logger.info("✅ Remote Server Acknowledged EOD Trigger.")
        else:
            logger.error(f"❌ Failed to trigger EOD. Status: {res.status_code} - {res.text}")
    except Exception as e:
        logger.error(f"❌ Connection Error: {e}")

def main():
    logger.info("=====================================================")
    logger.info("      ONYX LOCAL TRIGGER ENGINE (TERMUX)             ")
    logger.info(f"      Target: {RENDER_URL}                           ")
    logger.info("=====================================================")
    
    # Intraday triggers every 10 minutes
    schedule.every(10).minutes.do(trigger_intraday)
    
    # EOD triggers at 6:00 PM IST
    # Assuming the system running this script is in IST (your phone)
    schedule.every().day.at("18:00").do(trigger_eod)
    
    # Run once on boot to verify connection
    logger.info("Sending initial test ping to remote server...")
    trigger_intraday()
    
    logger.info("Engine running. Leave this script running in Termux from 8:30 AM to 6:30 PM.")
    
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()
