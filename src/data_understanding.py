"""
Step 1: Understand the Business and Datasets

This module analyzes the retail demand forecasting domain and explores the 8 raw CSV datasets:
- actual_matrix.csv
- catalog.csv
- discounts_history.csv
- markdowns.csv
- online.csv
- price_history.csv
- sales.csv
- stores.csv
"""

import os
import sys
import logging
import time
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
RAW_DATA_DIR = os.path.join(BASE_DIR, "DATASET")

def explain_business_context():
    """Outputs executive overview of business model, objectives, and domain logic."""
    overview = """
================================================================================
                    STEP 1: BUSINESS & DATASET UNDERSTANDING
================================================================================

1. BUSINESS CONTEXT & OBJECTIVE:
   - Organization: Omnichannel Retailer with offline physical stores and online e-commerce operations.
   - Core Business Goal: Optimize inventory management and demand forecasting to:
       a) Prevent stockouts (lost revenue and degraded customer satisfaction).
       b) Minimize overstocking & dead stock markdown losses (trapped capital, holding costs).
       c) Enhance dynamic reorder points (ROP), safety stock calculations, and economic order quantity (EOQ).

2. DATASET ROSTER & DOMAIN ROLES:
   - stores.csv          : Physical store master data (division, format, city, floor area).
   - catalog.csv         : Product catalog & hierarchy (dept, class, subclass, weight, fatness).
   - sales.csv           : In-store POS transactions (date, item, store, quantity, price, sum).
   - online.csv          : E-commerce online sales transactions.
   - price_history.csv   : Base item pricing changes over time per store.
   - discounts_history.csv: Promotional discount event history and discounted prices.
   - markdowns.csv       : Clearance markdown price and quantity logs.
   - actual_matrix.csv   : Active item-store matrix calendar showing item availability dates.
================================================================================
"""
    logger.info(overview)
    return overview

def inspect_datasets(data_dir=RAW_DATA_DIR):
    """Explores dataset schemas, dimensions, missing values, and data types."""
    logger.info("Starting dataset inspection across raw CSV files...")
    if not os.path.exists(data_dir):
        logger.error(f"Raw data directory missing at: {data_dir}")
        raise FileNotFoundError(f"Raw data directory not found at '{data_dir}'")

    dataset_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])
    
    summaries = {}
    for filename in dataset_files:
        filepath = os.path.join(data_dir, filename)
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        
        full_df = pd.read_csv(filepath)
        
        summary = {
            "filename": filename,
            "size_mb": round(file_size_mb, 2),
            "rows": len(full_df),
            "columns": len(full_df.columns),
            "col_names": list(full_df.columns),
            "missing_counts": full_df.isnull().sum().to_dict(),
            "dtypes": {k: str(v) for k, v in full_df.dtypes.to_dict().items()}
        }
        summaries[filename] = summary
        
        logger.info(f"Dataset: {filename:<25} | Size: {summary['size_mb']:>7.2f} MB | Rows: {summary['rows']:>9,d} | Cols: {summary['columns']}")
    
    return summaries

def main():
    start_time = time.time()
    logger.info("Starting Step 1: Business & Dataset Understanding...")
    explain_business_context()
    summaries = inspect_datasets()
    elapsed = time.time() - start_time
    logger.info(f"\n=========================================================")
    logger.info(f"SUCCESS: Step 1 completed in {elapsed:.2f} seconds.")
    logger.info(f"Inspected {len(summaries)} raw CSV datasets.")
    logger.info(f"=========================================================")

if __name__ == "__main__":
    main()
