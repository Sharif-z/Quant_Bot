import json
import os
import logging
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class PaperPortfolio:
    """
    MongoDB-backed state manager for the fictional ₹1 Crore portfolio.
    Bypasses ephemeral storage limitations of Render.
    """
    def __init__(self, db_name: str = "onyx_db", coll_name: str = "portfolio_state"):
        self.uri = os.getenv("MONGO_URI")
        if not self.uri:
            logger.error("MONGO_URI not found in environment. Bot cannot persist data!")
            
        self.client = MongoClient(self.uri, serverSelectionTimeoutMS=5000)
        self.db = self.client[db_name]
        self.collection = self.db[coll_name]
        self.doc_id = "main_portfolio"
        
        self.state = {
            "_id": self.doc_id,
            "initial_cash": 10_000_000.0,
            "current_cash": 10_000_000.0,
            "holdings": {},  # {"RELIANCE": {"qty": 100, "avg_price": 2500.0, "peak_price": 2600.0, "current_price": 2550.0}}
            "trade_history": [],
            "daily_mtm": {}, # {"YYYY-MM-DD": 10500000}
            "last_updated": None
        }
        self.load_state()
        
    def load_state(self):
        """Loads the portfolio state from MongoDB if it exists."""
        if not self.uri: return
        
        try:
            doc = self.collection.find_one({"_id": self.doc_id})
            if doc:
                self.state = doc
                logger.info("Loaded existing portfolio state from MongoDB.")
            else:
                self.save_state()
                logger.info("Initialized new MongoDB portfolio with ₹1 Crore.")
        except Exception as e:
            logger.error(f"Failed to load portfolio from MongoDB: {e}")
            
    def save_state(self):
        """Persists the current state to MongoDB."""
        if not self.uri: return
        self.state["last_updated"] = datetime.now().isoformat()
        try:
            self.collection.update_one({"_id": self.doc_id}, {"$set": self.state}, upsert=True)
        except Exception as e:
            logger.error(f"Failed to save portfolio to MongoDB: {e}")
            
    def get_total_value(self, current_prices: dict) -> float:
        """
        Calculates the Mark-to-Market (MTM) value of the portfolio based on current live prices.
        Also updates the current_price inside the holdings dictionary for the UI.
        """
        value = self.state["current_cash"]
        for symbol, data in self.state["holdings"].items():
            if symbol in current_prices:
                data["current_price"] = current_prices[symbol]
                value += data["qty"] * current_prices[symbol]
            else:
                logger.warning(f"Missing live price for {symbol}, using average buy price for valuation.")
                data["current_price"] = data.get("current_price", data["avg_price"])
                value += data["qty"] * data["current_price"]
        return value
        
    def execute_trade(self, symbol: str, qty: int, price: float, transaction_cost: float = 0.001):
        """
        Executes a paper trade (buy/sell) and updates holdings and cash.
        Positive qty = Buy, Negative qty = Sell.
        """
        trade_value = abs(qty) * price
        cost = trade_value * transaction_cost
        realized_pnl = 0.0
        
        if qty > 0: # Buy
            if self.state["current_cash"] < (trade_value + cost):
                logger.warning(f"Insufficient funds to buy {qty} of {symbol}. Cash: {self.state['current_cash']:.2f}")
                return False
                
            self.state["current_cash"] -= (trade_value + cost)
            
            if symbol not in self.state["holdings"]:
                self.state["holdings"][symbol] = {"qty": 0, "avg_price": 0.0, "peak_price": 0.0, "current_price": price}
                
            holding = self.state["holdings"][symbol]
            new_qty = holding["qty"] + qty
            new_avg_price = ((holding["qty"] * holding["avg_price"]) + trade_value) / new_qty
            
            holding["qty"] = new_qty
            holding["avg_price"] = new_avg_price
            holding["peak_price"] = max(holding.get("peak_price", 0.0), price)
            holding["current_price"] = price
            
        elif qty < 0: # Sell
            if symbol not in self.state["holdings"] or self.state["holdings"][symbol]["qty"] < abs(qty):
                logger.warning(f"Insufficient quantity to sell {abs(qty)} of {symbol}.")
                return False
                
            holding = self.state["holdings"][symbol]
            
            # Phase 17: Calculate EXACT Realized PnL for the UI
            gross_pnl = (price - holding["avg_price"]) * abs(qty)
            realized_pnl = gross_pnl - cost
            
            self.state["current_cash"] += (trade_value - cost)
            holding["qty"] -= abs(qty)
            
            # Clean up if holding goes to zero
            if holding["qty"] == 0:
                del self.state["holdings"][symbol]
                
        # Phase 14 & 17: Log trade history with exact Realized PnL
        if "trade_history" not in self.state:
            self.state["trade_history"] = []
            
        trade_record = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "action": "BUY" if qty > 0 else "SELL",
            "qty": abs(qty),
            "price": price,
            "value": trade_value,
            "realized_pnl": realized_pnl if qty < 0 else 0.0, # Track real PnL here!
            "frictional_cost": cost
        }
        self.state["trade_history"].append(trade_record)
        
        # Keep MongoDB document size manageable
        if len(self.state["trade_history"]) > 200:
            self.state["trade_history"] = self.state["trade_history"][-200:]
                
        self.save_state()
        logger.info(f"Executed paper trade: {'BUY' if qty > 0 else 'SELL'} {abs(qty)} {symbol} @ ₹{price:.2f}")
        return True

    def check_trailing_stops(self, current_prices: dict, stop_loss_pct: float = 0.15) -> list:
        stopped_out = []
        for symbol, data in list(self.state["holdings"].items()):
            if symbol in current_prices:
                current_price = current_prices[symbol]
                data["current_price"] = current_price
                
                if current_price > data.get("peak_price", 0.0):
                    data["peak_price"] = current_price
                    self.save_state()
                    
                peak = data.get("peak_price", current_price)
                if peak > 0 and (peak - current_price) / peak >= stop_loss_pct:
                    logger.warning(f"!!! TRAILING STOP HIT for {symbol} !!! Peak: {peak:.2f}, Current: {current_price:.2f} (Drop >= {stop_loss_pct:.1%})")
                    stopped_out.append(symbol)
                    
        return stopped_out
        
    def record_daily_mtm(self, mtm_value: float):
        """Phase 17: Records the daily total portfolio value for the Calendar Widget."""
        today = datetime.now().strftime("%Y-%m-%d")
        if "daily_mtm" not in self.state:
            self.state["daily_mtm"] = {}
        self.state["daily_mtm"][today] = mtm_value
        self.save_state()
