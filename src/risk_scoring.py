"""
Step 8: Inventory Risk Engine & Financial Impact Calculator

Calculates Safety Stock, Reorder Points (ROP), Economic Order Quantity (EOQ),
Stockout Risk Probabilities, and Financial Revenue at Risk / Locked Capital.
"""

import os
import sys
import logging
import time
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

class InventoryRiskEngine:
    def __init__(
        self,
        lead_time_days: int = 7,
        service_level_z: float = 1.65,  # 95% Service Level
        order_cost_s: float = 50.0,
        holding_cost_rate: float = 0.20,
        data_dir: str = PROCESSED_DATA_DIR,
        reports_dir: str = REPORTS_DIR
    ):
        self.L = lead_time_days
        self.Z = service_level_z
        self.S = order_cost_s
        self.H_rate = holding_cost_rate
        self.data_dir = data_dir
        self.reports_dir = reports_dir

    def calculate_risk_metrics(self) -> pd.DataFrame:
        """Calculates SKU/Store level Safety Stock, ROP, EOQ, Risk Level, and Financial Impact."""
        path = os.path.join(self.data_dir, "test_predictions.parquet")
        logger.info(f"Loading predictions for Risk Engine from '{path}'...")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Predictions parquet not found at '{path}'. Run ML forecasting engine first.")
        preds_df = pd.read_parquet(path)
        
        logger.info("Computing SKU-Store demand aggregations and risk metrics...")
        
        group = preds_df.groupby(["store_id", "item_id"])
        
        risk_df = group.agg(
            avg_daily_forecast=("predicted_quantity", "mean"),
            std_daily_forecast=("predicted_quantity", "std"),
            unit_price=("price_base", "mean"),
            total_actual_sales=("quantity", "sum"),
            total_predicted_sales=("predicted_quantity", "sum")
        ).reset_index()
        
        risk_df["avg_daily_forecast"] = np.maximum(risk_df["avg_daily_forecast"], 0.0)
        risk_df["std_daily_forecast"] = risk_df["std_daily_forecast"].fillna(0.1)
        
        # 1. Safety Stock (SS = Z * std * sqrt(L))
        risk_df["safety_stock"] = np.ceil(self.Z * risk_df["std_daily_forecast"] * np.sqrt(self.L))
        
        # 2. Reorder Point (ROP = d * L + SS)
        risk_df["reorder_point"] = np.ceil((risk_df["avg_daily_forecast"] * self.L) + risk_df["safety_stock"])
        
        # 3. Economic Order Quantity (EOQ = sqrt(2 * D * S / H))
        annual_demand = np.maximum(risk_df["avg_daily_forecast"] * 365, 0.0)
        holding_cost = np.maximum(risk_df["unit_price"] * self.H_rate, 0.5)
        risk_df["eoq"] = np.ceil(np.sqrt((2 * annual_demand * self.S) / holding_cost))
        
        # 4. Simulated Current Stock
        np.random.seed(42)
        sim_multiplier = np.random.uniform(0.2, 2.2, size=len(risk_df))
        risk_df["current_stock"] = np.ceil(risk_df["reorder_point"] * sim_multiplier)
        
        # 5. Days of Supply
        risk_df["days_of_supply"] = risk_df["current_stock"] / np.maximum(risk_df["avg_daily_forecast"], 0.01)
        
        # 6. Risk Level Categorization
        conditions = [
            (risk_df["days_of_supply"] <= 3.0),
            (risk_df["days_of_supply"] > 3.0) & (risk_df["days_of_supply"] <= 7.0),
            (risk_df["days_of_supply"] > 7.0) & (risk_df["days_of_supply"] <= 14.0),
            (risk_df["days_of_supply"] > 14.0)
        ]
        choices = ["CRITICAL STOCKOUT", "MEDIUM RISK (REORDER)", "BALANCED INVENTORY", "HIGH OVERSTOCK"]
        risk_df["risk_level"] = np.select(conditions, choices, default="BALANCED INVENTORY")
        
        quadrant_choices = ["Reorder now 🚨", "Watch / volatile ⚠️", "Healthy ✅", "Markdown / clear 🏷️"]
        risk_df["quadrant"] = np.select(conditions, quadrant_choices, default="Healthy ✅")
        
        action_choices = [
            "Raise a replenishment order before stock runs out.",
            "Investigate — demand is erratic; review manually.",
            "No action needed; leave as is.",
            "Promote or discount to free up capital."
        ]
        risk_df["recommended_action"] = np.select(conditions, action_choices, default="No action needed; leave as is.")
        
        # 7. Risk Scores (0 - 100)
        risk_df["stockout_score"] = np.clip((14.0 - risk_df["days_of_supply"]) * 7.5, 0.0, 100.0).round(1)
        risk_df["overstock_score"] = np.clip((risk_df["days_of_supply"] - 7.0) * 7.5, 0.0, 100.0).round(1)
        
        # 8. Financial Risk Calculation
        unmet_days = np.maximum(7.0 - risk_df["days_of_supply"], 0)
        risk_df["revenue_at_risk"] = unmet_days * risk_df["avg_daily_forecast"] * risk_df["unit_price"]
        
        excess_units = np.maximum(risk_df["current_stock"] - risk_df["reorder_point"], 0)
        risk_df["locked_capital"] = np.where(risk_df["days_of_supply"] > 14.0, excess_units * risk_df["unit_price"], 0.0)
        
        out_parquet = os.path.join(self.data_dir, "inventory_risk_report.parquet")
        out_csv = os.path.join(self.data_dir, "inventory_risk_report.csv")
        
        risk_df.to_parquet(out_parquet, index=False)
        risk_df.head(50000).to_csv(out_csv, index=False)
        
        logger.info(f"Inventory Risk Analysis completed. Output shape: {risk_df.shape}")
        return risk_df

    def summarize_risk_report(self, risk_df: pd.DataFrame) -> dict:
        """Generates executive summary statistics of inventory risk and financial impact."""
        summary = {
            "total_skus_evaluated": len(risk_df),
            "critical_stockout_skus": int((risk_df["risk_level"] == "CRITICAL STOCKOUT").sum()),
            "reorder_needed_skus": int((risk_df["risk_level"] == "MEDIUM RISK (REORDER)").sum()),
            "overstock_skus": int((risk_df["risk_level"] == "HIGH OVERSTOCK").sum()),
            "total_revenue_at_risk": float(risk_df["revenue_at_risk"].sum()),
            "total_locked_capital": float(risk_df["locked_capital"].sum())
        }
        
        logger.info("\n=========================================================")
        logger.info("EXECUTIVE INVENTORY RISK & FINANCIAL IMPACT REPORT:")
        logger.info(f"  - Total Product SKUs Evaluated: {summary['total_skus_evaluated']:,}")
        logger.info(f"  - Critical Stockout Warning SKUs: {summary['critical_stockout_skus']:,}")
        logger.info(f"  - Reorder Point Reached SKUs   : {summary['reorder_needed_skus']:,}")
        logger.info(f"  - Excess Overstock SKUs         : {summary['overstock_skus']:,}")
        logger.info(f"  - Financial Lost Revenue at Risk: ${summary['total_revenue_at_risk']:,.2f}")
        logger.info(f"  - Financial Working Capital Locked: ${summary['total_locked_capital']:,.2f}")
        logger.info("=========================================================")
        
        return summary

def main():
    start_time = time.time()
    logger.info("Starting Step 8: Inventory Risk Engine & Financial Impact Calculator...")
    
    engine = InventoryRiskEngine()
    risk_df = engine.calculate_risk_metrics()
    summary = engine.summarize_risk_report(risk_df)
    
    elapsed = time.time() - start_time
    logger.info(f"\n=========================================================")
    logger.info(f"SUCCESS: Risk Engine completed in {elapsed:.2f} seconds.")
    logger.info(f"Saved Inventory Risk Report Artifacts:")
    logger.info("  - Parquet: 'data/processed/inventory_risk_report.parquet'")
    logger.info("  - CSV    : 'data/processed/inventory_risk_report.csv'")
    logger.info(f"=========================================================")

if __name__ == "__main__":
    main()
