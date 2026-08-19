"""
Step 2: Data Collection & Ingestion Module

This module handles robust ingestion, verification, schema validation,
and memory optimization of raw retail datasets from storage.
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
RAW_DATA_DIR = os.path.join(BASE_DIR, "DATASET")

EXPECTED_FILES = [
    "actual_matrix.csv",
    "catalog.csv",
    "discounts_history.csv",
    "markdowns.csv",
    "online.csv",
    "price_history.csv",
    "sales.csv",
    "stores.csv",
]

def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcasts numerical columns to save memory without precision loss."""
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == "int64":
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif col_type == "float64":
            df[col] = pd.to_numeric(df[col], downcast="float")
    return df

class DataCollector:
    def __init__(self, raw_data_dir: str = RAW_DATA_DIR):
        self.raw_data_dir = raw_data_dir

    def verify_files(self) -> bool:
        """Verifies all expected dataset files exist in raw storage directory."""
        missing = []
        for filename in EXPECTED_FILES:
            filepath = os.path.join(self.raw_data_dir, filename)
            base_no_ext = filename.replace(".csv", "")
            part1 = os.path.join(self.raw_data_dir, f"{base_no_ext}_part1.csv.gz")
            gz_single = os.path.join(self.raw_data_dir, f"{filename}.gz")
            
            if not (os.path.exists(filepath) or os.path.exists(part1) or os.path.exists(gz_single)):
                missing.append(filename)
        
        if missing:
            logger.error(f"Missing raw dataset files in '{self.raw_data_dir}': {missing}")
            raise FileNotFoundError(f"Missing raw files: {missing}")
        
        logger.info(f"Verification successful: All {len(EXPECTED_FILES)} raw dataset files present.")
        return True

    def load_dataset(self, filename: str, optimize_mem: bool = True) -> pd.DataFrame:
        """Loads a single CSV dataset with error handling and optional memory downcasting."""
        filepath = os.path.join(self.raw_data_dir, filename)
        logger.info(f"Ingesting raw dataset: {filename}...")
        
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
        else:
            base_no_ext = filename.replace(".csv", "")
            part1 = os.path.join(self.raw_data_dir, f"{base_no_ext}_part1.csv.gz")
            part2 = os.path.join(self.raw_data_dir, f"{base_no_ext}_part2.csv.gz")
            gz_single = os.path.join(self.raw_data_dir, f"{filename}.gz")
            
            if os.path.exists(part1) and os.path.exists(part2):
                logger.info(f"  Reading split compressed parts: {part1} and {part2}...")
                df1 = pd.read_csv(part1, compression="gzip")
                df2 = pd.read_csv(part2, compression="gzip")
                df = pd.concat([df1, df2], ignore_index=True)
            elif os.path.exists(gz_single):
                df = pd.read_csv(gz_single, compression="gzip")
            else:
                raise FileNotFoundError(f"Cannot find {filename} or compressed parts in {self.raw_data_dir}")
                
        initial_mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        
        if optimize_mem:
            df = optimize_dtypes(df)
            final_mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
            logger.info(f"  Loaded {filename}: shape={df.shape}, Initial RAM={initial_mem_mb:.2f}MB, Optimized RAM={final_mem_mb:.2f}MB")
        else:
            logger.info(f"  Loaded {filename}: shape={df.shape}, RAM={initial_mem_mb:.2f}MB")
            
        return df

    def collect_all_datasets(self) -> dict[str, pd.DataFrame]:
        """Ingests and validates all raw datasets into a dictionary of DataFrames."""
        self.verify_files()
        datasets = {}
        for filename in EXPECTED_FILES:
            name_key = filename.replace(".csv", "")
            datasets[name_key] = self.load_dataset(filename)
        
        logger.info("Successfully ingested and collected all 8 datasets.")
        return datasets

def main():
    start_time = time.time()
    logger.info("Starting Step 2: Data Collection & Ingestion...")
    collector = DataCollector()
    collected_data = collector.collect_all_datasets()
    elapsed = time.time() - start_time
    logger.info(f"\n=========================================================")
    logger.info(f"SUCCESS: Step 2 completed in {elapsed:.2f} seconds.")
    logger.info(f"Ingested all {len(collected_data)} datasets successfully.")
    logger.info(f"=========================================================")

if __name__ == "__main__":
    main()
