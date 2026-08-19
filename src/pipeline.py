"""
PROJECT FORESIGHT: Master End-to-End Data Pipeline
Executes full data ingestion, cleaning, feature engineering, baseline evaluation,
ML model training, product clustering, risk scoring, and dashboard precomputation with a single command.

Usage:
    python src/pipeline.py
"""

import os
import sys
import logging
import time
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_collection import DataCollector
from src.data_cleaning import DataCleaner
from src.feature_engineering import FeatureEngineer
from src.baseline_model import BaselineForecaster
from src.ml_forecasting import MLForecastingEngine
from src.product_clustering import ProductClusteringEngine
from src.risk_scoring import InventoryRiskEngine
from src.precompute_dashboard_data import precompute_all


def run_full_pipeline():
    start_time = time.time()
    logger.info("==========================================================================")
    logger.info("🚀 STARTING PROJECT FORESIGHT END-TO-END DATA PIPELINE RUN")
    logger.info("==========================================================================")

    # Step 1: Ingestion / Synthetic Data Collection
    logger.info("Step 1/8: Ingesting & Verifying Raw Dataset Extracts...")
    collector = DataCollector()
    raw_data = collector.collect_all_datasets()

    # Step 2: Data Cleaning & Preprocessing
    logger.info("Step 2/8: Running Data Cleaning, Validation & Schema Normalization...")
    cleaner = DataCleaner()
    cleaned_data = cleaner.clean_all(raw_data)
    cleaner.save_processed_datasets(cleaned_data)

    # Step 3: Feature Engineering
    logger.info("Step 3/8: Engineering Lags, Rolling Statistics & Promo Signal Features...")
    engineer = FeatureEngineer()
    engineer.generate_feature_matrix()

    # Step 4: Baseline Seasonal-Naive Evaluation
    logger.info("Step 4/8: Computing Seasonal-Naive Baseline Benchmark (WAPE Bar)...")
    evaluator = BaselineForecaster()
    evaluator.run_all_baselines()

    # Step 5: Machine Learning Forecasting (LightGBM, XGBoost, CatBoost, Prophet)
    logger.info("Step 5/8: Training Multi-Model Demand Forecaster & Backtesting...")
    forecaster = MLForecastingEngine()
    forecaster.run_all_models()

    # Step 6: Product Portfolio Segmentation (K-Means Clustering)
    logger.info("Step 6/8: Executing K-Means Product Clustering & Health Scoring...")
    sales_path = os.path.join(BASE_DIR, "data", "processed", "daily_sales_summary.parquet")
    sales_df = pd.read_parquet(sales_path) if os.path.exists(sales_path) else pd.DataFrame()
    clusterer = ProductClusteringEngine()
    if not sales_df.empty:
        clusterer.cluster_products(sales_df)

    # Step 7: Inventory Risk Scoring & 4-Quadrant Decisioning Grid
    logger.info("Step 7/9: Scoring Stockout & Overstock Risks with Action Recommendations...")
    risk_engine = InventoryRiskEngine()
    risk_engine.calculate_risk_metrics()

    # Step 8: Fundamental Statistical Hypothesis Testing Suite
    logger.info("Step 8/9: Running 27+ Fundamental Statistical Hypothesis & Diagnostic Tests...")
    from src.statistical_tests import StatisticalTestEngine
    stat_engine = StatisticalTestEngine()
    stat_engine.run_all_tests()

    # Step 9: Precompute Dashboard Parquet Cache
    logger.info("Step 9/9: Precomputing High-Speed Dashboard Parquet Artifacts...")
    precompute_all()

    elapsed = time.time() - start_time
    logger.info("==========================================================================")
    logger.info(f"✅ PIPELINE SUCCESSFULLY COMPLETED IN {elapsed:.2f} SECONDS")
    logger.info("==========================================================================")


if __name__ == "__main__":
    run_full_pipeline()
