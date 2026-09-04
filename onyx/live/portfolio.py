import json
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class PaperPortfolio:
    """
    Local state manager for the fictional ₹1 Crore portfolio.
    Persists data to a local JSON file to survive restarts between the 10-minute cron loops.
    """
    def __init__(self, file_path: str = "data_storage/portfolio.json"):
        self.file_path = file_path
        self.state = {
            "initial_cash": 10_000_000.0,
            "current_cash": 10_000_000.0,
            "holdings": {},  # Format: {"RELIANCE": {"qty": 100, "avg_price": 2500.0}}
            "trade_history": [],
            "last_updated": None
        }
        self.load_state()
        
    def load_state(self):
        """Loads the portfolio state from disk if it exists."""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r') as f:
                    self.state = json.load(f)
                logger.info("Loaded existing paper portfolio state.")
            except Exception as e:
                logger.error(f"Failed to load portfolio state: {e}")
        else:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            self.save_state()
            logger.info("Initialized new paper portfolio with ₹1 Crore.")
            
    def save_state(self):
        """Persists the current state to disk."""
        self.state["last_updated"] = datetime.now().isoformat()
        try:
            with open(self.file_path, 'w') as f:
                json.dump(self.state, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save portfolio state: {e}")
            
    def get_total_value(self, current_prices: dict) -> float:
        """
        Calculates the Mark-to-Market (MTM) value of the portfolio based on current live prices.
        current_prices format: {"RELIANCE": 2550.0, ...}
        """
        value = self.state["current_cash"]
        for symbol, data in self.state["holdings"].items():
            if symbol in current_prices:
                value += data["qty"] * current_prices[symbol]
            else:
                logger.warning(f"Missing live price for {symbol}, using average buy price for valuation.")
                value += data["qty"] * data["avg_price"]
        return value
        
    def execute_trade(self, symbol: str, qty: int, price: float, transaction_cost: float = 0.001):
        """
        Executes a paper trade (buy/sell) and updates holdings and cash.
        Positive qty = Buy, Negative qty = Sell.
        transaction_cost accounts for the 0.1% STT + Brokerage frictional drag.
        """
        trade_value = abs(qty) * price
        cost = trade_value * transaction_cost
        
        if qty > 0: # Buy
            if self.state["current_cash"] < (trade_value + cost):
                logger.warning(f"Insufficient funds to buy {qty} of {symbol}. Cash: {self.state['current_cash']:.2f}")
                return False
                
            self.state["current_cash"] -= (trade_value + cost)
            
            if symbol not in self.state["holdings"]:
                self.state["holdings"][symbol] = {"qty": 0, "avg_price": 0.0, "peak_price": 0.0}
                
            holding = self.state["holdings"][symbol]
            new_qty = holding["qty"] + qty
            new_avg_price = ((holding["qty"] * holding["avg_price"]) + trade_value) / new_qty
            
            holding["qty"] = new_qty
            holding["avg_price"] = new_avg_price
            holding["peak_price"] = max(holding.get("peak_price", 0.0), price)
            
        elif qty < 0: # Sell
            if symbol not in self.state["holdings"] or self.state["holdings"][symbol]["qty"] < abs(qty):
                logger.warning(f"Insufficient quantity to sell {abs(qty)} of {symbol}.")
                return False
                
            self.state["current_cash"] += (trade_value - cost)
            self.state["holdings"][symbol]["qty"] -= abs(qty)
            
            # Clean up if holding goes to zero
            if self.state["holdings"][symbol]["qty"] == 0:
                del self.state["holdings"][symbol]
                
        # Phase 14: Log trade history
        if "trade_history" not in self.state:
            self.state["trade_history"] = []
            
        trade_record = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "action": "BUY" if qty > 0 else "SELL",
            "qty": abs(qty),
            "price": price,
            "value": trade_value
        }
        self.state["trade_history"].append(trade_record)
        
        # Enforce size limit on visual ledger
        if len(self.state["trade_history"]) > 100:
            self.state["trade_history"] = self.state["trade_history"][-100:]
                
        self.save_state()
        logger.info(f"Executed paper trade: {'BUY' if qty > 0 else 'SELL'} {abs(qty)} {symbol} @ ₹{price:.2f}")
        return True

    def check_trailing_stops(self, current_prices: dict, stop_loss_pct: float = 0.15) -> list:
        """
        Hard Risk Overlay: Checks if any holding has dropped by more than `stop_loss_pct` 
        from its tracked `peak_price`.
        Returns a list of symbols that hit the trailing stop.
        """
        stopped_out = []
        for symbol, data in list(self.state["holdings"].items()):
            if symbol in current_prices:
                current_price = current_prices[symbol]
                
                # Update peak price if the stock made a new high
                if current_price > data.get("peak_price", 0.0):
                    data["peak_price"] = current_price
                    self.save_state()
                    
                # Check for Trailing Stop breach
                peak = data.get("peak_price", current_price)
                if peak > 0 and (peak - current_price) / peak >= stop_loss_pct:
                    logger.warning(f"!!! TRAILING STOP HIT for {symbol} !!! Peak: {peak:.2f}, Current: {current_price:.2f} (Drop >= {stop_loss_pct:.1%})")
                    stopped_out.append(symbol)
                    
        return stopped_out
