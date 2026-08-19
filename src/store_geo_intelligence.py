"""
Store & Geographic Intelligence Engine: Store Ranking, Efficiency per Area, City Breakdown & Opportunity Detection
"""

import os
import sys
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

class StoreGeoEngine:
    def __init__(self, data_dir: str = PROCESSED_DATA_DIR):
        self.data_dir = data_dir

    def calculate_store_efficiency(self, stores_df: pd.DataFrame, sales_df: pd.DataFrame) -> pd.DataFrame:
        """Calculates Store Ranking, Revenue per sq. ft. floor area, and Units per sq. ft. area."""
        store_sales = sales_df.groupby("store_id").agg(
            total_revenue=("sum_total", "sum"),
            total_units=("quantity", "sum"),
            avg_price=("price_base", "mean"),
            active_days=("date", "nunique")
        ).reset_index()
        
        merged = stores_df.merge(store_sales, on="store_id", how="left")
        merged["area"] = merged["area"].replace(0, 1000)
        
        merged["rev_per_sqft"] = (merged["total_revenue"] / merged["area"]).round(2)
        merged["units_per_sqft"] = (merged["total_units"] / merged["area"]).round(2)
        merged["daily_revenue"] = (merged["total_revenue"] / np.maximum(merged["active_days"], 1)).round(2)
        
        # Sort by total revenue
        merged = merged.sort_values(by="total_revenue", ascending=False).reset_index(drop=True)
        merged["store_rank"] = merged.index + 1
        return merged

    def calculate_geographic_performance(self, stores_df: pd.DataFrame, sales_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregates revenue, units, store count, and growth score by City / Geographic Location."""
        merged_sales = sales_df.merge(stores_df[["store_id", "city"]], on="store_id", how="left")
        merged_sales["city"] = merged_sales["city"].fillna("Metro Region")
        
        geo = merged_sales.groupby("city").agg(
            total_revenue=("sum_total", "sum"),
            total_units=("quantity", "sum"),
            avg_price=("price_base", "mean"),
            transaction_count=("quantity", "count")
        ).reset_index()
        
        geo_stores = stores_df.groupby("city")["store_id"].count().reset_index().rename(columns={"store_id": "active_stores"})
        geo = geo.merge(geo_stores, on="city", how="left")
        
        geo["rev_per_store"] = (geo["total_revenue"] / np.maximum(geo["active_stores"], 1)).round(2)
        geo["geo_opportunity_score"] = np.clip(
            (geo["total_revenue"] / (geo["total_revenue"].max() + 1e-5) * 60) +
            (geo["active_stores"] * 10),
            45, 95
        ).round(1)
        
        return geo.sort_values("total_revenue", ascending=False).reset_index(drop=True)

    def detect_store_opportunities(self, stores_df: pd.DataFrame, sales_df: pd.DataFrame, catalog_df: pd.DataFrame) -> list[dict]:
        """Identifies specific store-level demand vs availability opportunity insights."""
        opportunities = []
        if catalog_df is not None and "dept_name" in catalog_df.columns:
            merged = sales_df.merge(catalog_df[["item_id", "dept_name"]], on="item_id", how="left")
            merged["dept_name"] = merged["dept_name"].fillna("General Category")
            dept_store = merged.groupby(["store_id", "dept_name"])["sum_total"].sum().reset_index()
            
            for store_id in stores_df["store_id"].unique():
                s_data = dept_store[dept_store["store_id"] == store_id]
                if not s_data.empty:
                    top_dept = s_data.sort_values("sum_total", ascending=False).iloc[0]["dept_name"]
                    top_rev = s_data.sort_values("sum_total", ascending=False).iloc[0]["sum_total"]
                    opportunities.append({
                        "store_id": int(store_id),
                        "opportunity_type": "HIGH DEMAND CATEGORY LEADER",
                        "department": str(top_dept),
                        "revenue": f"${top_rev:,.2f}",
                        "recommendation": f"Store {store_id} shows exceptionally strong demand for '{top_dept}'. Expand store floor space and ensure high safety stock for this category."
                    })
        return opportunities
