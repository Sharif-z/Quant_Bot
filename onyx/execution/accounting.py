from onyx.core.config import config
import logging

logger = logging.getLogger(__name__)

class FrictionCalculator:
    """
    Calculates exact Indian market frictions (taxes, exchange charges, stamp duty)
    based on the SEBI framework defined in config.py.
    """
    def __init__(self):
        self.costs = config.costs

    def calculate_trade_costs(self, trade_value: float, is_buy: bool, is_delivery: bool = True) -> dict:
        """
        Calculates all Indian market frictions for a given trade.
        Returns a dictionary of exact costs.
        """
        # Absolute trade value
        tv = abs(trade_value)
        
        # 1. Securities Transaction Tax (STT)
        if is_delivery:
            stt = tv * self.costs.stt_delivery
        else:
            stt = tv * self.costs.stt_intraday if not is_buy else 0.0
            
        # 2. Exchange Transaction Charge
        exc_charge = tv * self.costs.exchange_transaction_charge
        
        # 3. SEBI Turnover Fee
        sebi_fee = tv * self.costs.sebi_turnover_fee
        
        # 4. Stamp Duty (Only on Buys)
        stamp_duty = 0.0
        if is_buy:
            stamp_duty = tv * (self.costs.stamp_duty_delivery if is_delivery else self.costs.stamp_duty_intraday)
            
        # 5. GST (18% on Exchange Charges + SEBI Fees, assuming 0 brokerage)
        gst = (exc_charge + sebi_fee) * self.costs.gst_rate
        
        total_friction = stt + exc_charge + sebi_fee + stamp_duty + gst
        
        return {
            'trade_value': tv,
            'is_buy': is_buy,
            'stt': stt,
            'exchange_charge': exc_charge,
            'sebi_fee': sebi_fee,
            'stamp_duty': stamp_duty,
            'gst': gst,
            'total_friction': total_friction
        }
