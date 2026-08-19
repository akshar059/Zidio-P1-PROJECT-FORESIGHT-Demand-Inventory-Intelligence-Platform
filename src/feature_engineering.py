"""
Step 5: Feature Engineering Module

Generates time-series features (lags, rolling statistics, temporal calendar encodings,
and price ratio dynamics) for ML & Time-Series demand forecasting.
Saves feature matrix to data/processed/feature_matrix.parquet.
"""

import os
import sys
import logging
import time
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Base paths resolved dynamically
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

class FeatureEngineer:
    def __init__(self, data_dir: str = PROCESSED_DATA_DIR):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def load_cleaned_datasets(self) -> dict[str, pd.DataFrame]:
        """Loads cleaned parquet datasets required for feature engineering."""
        logger.info("Loading cleaned datasets for feature engineering...")
        sales_path = os.path.join(self.data_dir, "cleaned_sales.parquet")
        catalog_path = os.path.join(self.data_dir, "cleaned_catalog.parquet")
        stores_path = os.path.join(self.data_dir, "cleaned_stores.parquet")
        price_path = os.path.join(self.data_dir, "cleaned_price_history.parquet")

        if not os.path.exists(sales_path):
            raise FileNotFoundError(f"Cleaned sales dataset not found at '{sales_path}'. Run data cleaning first.")

        sales = pd.read_parquet(sales_path)
        catalog = pd.read_parquet(catalog_path) if os.path.exists(catalog_path) else None
        stores = pd.read_parquet(stores_path) if os.path.exists(stores_path) else None
        price_hist = pd.read_parquet(price_path) if os.path.exists(price_path) else None

        return {
            "sales": sales,
            "catalog": catalog,
            "stores": stores,
            "price_history": price_hist
        }

    def generate_feature_matrix(self) -> pd.DataFrame:
        """Constructs time-series lag, rolling statistics, calendar encodings, and price features."""
        logger.info("Starting Feature Engineering Pipeline...")
        datasets = self.load_cleaned_datasets()
        sales = datasets["sales"]
        catalog = datasets["catalog"]

        sales["date"] = pd.to_datetime(sales["date"])

        logger.info("Aggregating daily sales per (store_id, item_id)...")
        daily = sales.groupby(["date", "store_id", "item_id"]).agg(
            quantity=("quantity", "sum"),
            price_base=("price_base", "mean")
        ).reset_index()

        daily = daily.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)

        if catalog is not None:
            cat_cols = [c for c in ["item_id", "dept_name", "class_name", "subclass_name"] if c in catalog.columns]
            daily = daily.merge(catalog[cat_cols], on="item_id", how="left")
            if "dept_name" in daily.columns:
                daily["dept_name"] = daily["dept_name"].fillna("Unknown")
            if "class_name" in daily.columns:
                daily["class_name"] = daily["class_name"].fillna("Unknown")

        # 1. Temporal / Calendar Features
        logger.info("Computing calendar & seasonal temporal features...")
        daily["day_of_week"] = daily["date"].dt.dayofweek.astype(np.int8)
        daily["day_of_month"] = daily["date"].dt.day.astype(np.int8)
        daily["month"] = daily["date"].dt.month.astype(np.int8)
        daily["year"] = daily["date"].dt.year.astype(np.int16)
        daily["quarter"] = daily["date"].dt.quarter.astype(np.int8)
        daily["is_weekend"] = (daily["day_of_week"] >= 5).astype(np.int8)

        series_group = daily.groupby(["store_id", "item_id"])["quantity"]

        # 2. Lag Features
        logger.info("Generating lag features (lag_1, lag_7, lag_14, lag_28)...")
        daily["lag_1"] = series_group.shift(1).fillna(0).astype(np.float32)
        daily["lag_7"] = series_group.shift(7).fillna(0).astype(np.float32)
        daily["lag_14"] = series_group.shift(14).fillna(0).astype(np.float32)
        daily["lag_28"] = series_group.shift(28).fillna(0).astype(np.float32)

        # 3. Rolling Window Statistics
        logger.info("Generating rolling window statistics...")
        lag1_series = daily.groupby(["store_id", "item_id"])["lag_1"]
        daily["rolling_mean_7"] = lag1_series.transform(lambda x: x.rolling(7, min_periods=1).mean()).fillna(0).astype(np.float32)
        daily["rolling_std_7"] = lag1_series.transform(lambda x: x.rolling(7, min_periods=1).std()).fillna(0).astype(np.float32)
        daily["rolling_mean_28"] = lag1_series.transform(lambda x: x.rolling(28, min_periods=1).mean()).fillna(0).astype(np.float32)
        daily["rolling_std_28"] = lag1_series.transform(lambda x: x.rolling(28, min_periods=1).std()).fillna(0).astype(np.float32)

        # 4. Price Features
        logger.info("Computing price ratio dynamics...")
        price_group = daily.groupby(["store_id", "item_id"])["price_base"]
        price_30d = price_group.transform(lambda x: x.rolling(30, min_periods=1).mean()).fillna(daily["price_base"])
        daily["price_ratio_30d"] = (daily["price_base"] / (price_30d + 1e-5)).astype(np.float32)

        output_path = os.path.join(self.data_dir, "feature_matrix.parquet")
        daily.to_parquet(output_path, index=False)
        logger.info(f"Feature matrix successfully created and saved to '{output_path}'. Shape: {daily.shape}")
        return daily

def main():
    start_time = time.time()
    logger.info("Starting Step 5: Feature Engineering Phase...")
    
    engineer = FeatureEngineer()
    feature_df = engineer.generate_feature_matrix()
    
    elapsed = time.time() - start_time
    logger.info(f"\n=========================================================")
    logger.info(f"SUCCESS: Feature Engineering completed in {elapsed:.2f} seconds.")
    logger.info(f"Feature matrix saved: 'data/processed/feature_matrix.parquet'")
    logger.info(f"Matrix shape: {feature_df.shape}")
    logger.info(f"=========================================================")

if __name__ == "__main__":
    main()
