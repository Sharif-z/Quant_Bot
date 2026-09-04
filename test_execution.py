import logging
import pandas as pd
import numpy as np
import sys
from onyx.execution.accounting import FrictionCalculator
from onyx.execution.simulator import ExecutionSimulator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_execution_engine():
    try:
        logger.info("Testing Phase 5: Execution Engine & Frictional Accounting...")
        
        # 1. Test Frictional Accounting
        acc = FrictionCalculator()
        trade_val = 100000.0 # 1 Lakh Rupees
        
        # Buy Delivery
        costs = acc.calculate_trade_costs(trade_val, is_buy=True, is_delivery=True)
        # Expected:
        # STT = 0.1% of 100k = 100
        # Exc = 0.00325% = 3.25
        # SEBI = 0.0001% = 0.1
        # Stamp = 0.015% = 15
        # GST = 18% of (3.25 + 0.1) = 0.603
        
        assert np.isclose(costs['stt'], 100.0), f"STT failed: {costs['stt']}"
        assert np.isclose(costs['exchange_charge'], 3.25), f"Exchange failed: {costs['exchange_charge']}"
        assert np.isclose(costs['stamp_duty'], 15.0), f"Stamp failed: {costs['stamp_duty']}"
        
        logger.info(f"Frictional Check Passed. Total friction on 1L Buy: ₹{costs['total_friction']:.3f}")
        
        # 2. Test Execution Simulator
        sim = ExecutionSimulator(volatility_factor=0.02) # 2% daily vol
        
        # Scenario: Start with 10 Lakh cash, transition to 50% SEC1, 50% SEC2
        target_weights = pd.Series({'SEC1': 0.5, 'SEC2': 0.5})
        current_prices = pd.Series({'SEC1': 100.0, 'SEC2': 200.0})
        
        # ADV: SEC1 is highly liquid, SEC2 is illiquid
        adv_data = pd.Series({'SEC1': 10_000_000, 'SEC2': 10_000}) 
        
        logger.info("\nExecuting Target Portfolio...")
        res = sim.execute_target_portfolio(
            target_weights=target_weights,
            current_prices=current_prices,
            adv_data=adv_data,
            current_cash=10_00_000.0
        )
        
        log = res['execution_log']
        logger.info("\nExecution Log:")
        logger.info(log[['asset', 'type', 'qty', 'target_price', 'exec_price', 'slippage_pct', 'frictions', 'net_cash_impact']].to_string())
        
        logger.info(f"\nInitial Value: ₹{res['initial_value']:,.2f}")
        logger.info(f"Final Value: ₹{res['final_value']:,.2f}")
        logger.info(f"Frictional Drag (Slippage + Taxes): ₹{res['frictional_drag']:,.2f}")
        logger.info(f"Remaining Cash: ₹{res['new_cash']:,.2f}")
        
        # SEC2 should have much higher slippage than SEC1 because of low ADV
        sec1_slip = log[log['asset'] == 'SEC1']['slippage_pct'].values[0]
        sec2_slip = log[log['asset'] == 'SEC2']['slippage_pct'].values[0]
        assert sec2_slip > sec1_slip, "Square Root Law failed to penalize illiquid asset."
        
        logger.info("\nSUCCESS: Execution Simulator strictly verified.")
        
    except Exception as e:
        logger.error(f"Execution Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_execution_engine()
