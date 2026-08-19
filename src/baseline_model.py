"""
Step 6: Baseline Forecasting Module

Computes statistical baseline forecasting benchmarks:
1. Last Value Baseline
2. 7-Day Moving Average Baseline
3. 28-Day Moving Average Baseline
4. Seasonal Naive (Lag 7) Baseline

Evaluates performance using WAPE, MAE, RMSE, MAPE (%), Bias (%), and R2.
Saves benchmark report to reports/baseline_leaderboard.csv.
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

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculates WAPE, MAE, RMSE, MAPE (%), Bias (%), and R2."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.clip(np.array(y_pred, dtype=float), 0, None)
    
    abs_errors = np.abs(y_true - y_pred)
    sum_true = np.sum(y_true)
    
    wape = np.sum(abs_errors) / sum_true if sum_true > 0 else np.nan
    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    denom = np.where(y_true == 0, 1.0, y_true)
    mape = np.mean(abs_errors / denom) * 100
    bias = (np.sum(y_pred - y_true) / sum_true) * 100 if sum_true > 0 else np.nan
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    
    return {
        "WAPE": round(float(wape), 4),
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "MAPE (%)": round(float(mape), 2),
        "Bias (%)": round(float(bias), 2),
        "R2": round(float(r2), 4)
    }

class BaselineForecaster:
    def __init__(self, data_dir: str = PROCESSED_DATA_DIR, reports_dir: str = REPORTS_DIR):
        self.data_dir = data_dir
        self.reports_dir = reports_dir
        os.makedirs(self.reports_dir, exist_ok=True)

    def load_feature_matrix(self) -> pd.DataFrame:
        """Loads feature matrix dataset."""
        path = os.path.join(self.data_dir, "feature_matrix.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Feature matrix not found at '{path}'. Please run Step 5 (feature_engineering.py) first.")
        df = pd.read_parquet(path)
        return df

    def run_all_baselines(self, test_days: int = 28) -> pd.DataFrame:
        """Evaluates statistical baseline models against ground truth on test period."""
        logger.info("Starting Baseline Forecasting Evaluation...")
        df = self.load_feature_matrix()

        max_date = df["date"].max()
        split_date = max_date - pd.Timedelta(days=test_days)

        train = df[df["date"] <= split_date].copy()
        test = df[df["date"] > split_date].copy()
        y_test = test["quantity"].values

        results = []

        # 1. Last Value (Lag 1)
        preds_last = test["lag_1"].values
        m_last = compute_metrics(y_test, preds_last)
        m_last["Model"] = "Last Value Baseline"
        results.append(m_last)

        # 2. Moving Average 7D
        preds_ma7 = test["rolling_mean_7"].values
        m_ma7 = compute_metrics(y_test, preds_ma7)
        m_ma7["Model"] = "Moving Avg (7D)"
        results.append(m_ma7)

        # 3. Moving Average 28D
        preds_ma28 = test["rolling_mean_28"].values
        m_ma28 = compute_metrics(y_test, preds_ma28)
        m_ma28["Model"] = "Moving Avg (28D)"
        results.append(m_ma28)

        # 4. Seasonal Naive (Lag 7)
        preds_sn7 = test["lag_7"].values
        m_sn7 = compute_metrics(y_test, preds_sn7)
        m_sn7["Model"] = "Seasonal Naive (7D)"
        results.append(m_sn7)

        baseline_df = pd.DataFrame(results)[["Model", "WAPE", "MAE", "RMSE", "MAPE (%)", "Bias (%)", "R2"]]
        baseline_df = baseline_df.sort_values(by="WAPE", ascending=True).reset_index(drop=True)

        out_path = os.path.join(self.reports_dir, "baseline_leaderboard.csv")
        baseline_df.to_csv(out_path, index=False)
        logger.info(f"Baseline Leaderboard saved to '{out_path}'.")
        return baseline_df

def main():
    start_time = time.time()
    logger.info("Starting Step 6: Baseline Model Evaluation Phase...")
    
    forecaster = BaselineForecaster()
    leaderboard = forecaster.run_all_baselines()
    
    elapsed = time.time() - start_time
    logger.info(f"\n=========================================================")
    logger.info(f"SUCCESS: Baseline Model Evaluation completed in {elapsed:.2f} seconds.")
    logger.info("\n📊 BASELINE MODEL LEADERBOARD:")
    logger.info("\n" + leaderboard.to_string(index=False))
    logger.info(f"\nSaved Artifact: 'reports/baseline_leaderboard.csv'")
    logger.info(f"=========================================================")

if __name__ == "__main__":
    main()
