"""
Step 7: Machine Learning & Time-Series Demand Forecasting Engine

Trains and evaluates:
1. LightGBM Forecaster
2. XGBoost TimeSeries Regressor
3. CatBoost Regressor
4. Random Forest Regressor
5. Prophet (Store-level time series)
6. ARIMA / SARIMA (Store-level time series)

Evaluates performance on a 28-day temporal test split using WAPE, MAE, RMSE, MAPE, Bias (%), and R2.
Saves trained models to models_saved/.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import time
import pickle

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models_saved")
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

class MLForecastingEngine:
    def __init__(self, data_dir: str = PROCESSED_DATA_DIR, models_dir: str = MODELS_DIR, reports_dir: str = REPORTS_DIR):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.reports_dir = reports_dir
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

    def load_feature_matrix(self) -> pd.DataFrame:
        """Loads feature matrix dataset."""
        path = os.path.join(self.data_dir, "feature_matrix.parquet")
        logger.info(f"Loading feature matrix from '{path}'...")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Feature matrix not found at '{path}'. Run feature engineering first.")
        df = pd.read_parquet(path)
        logger.info(f"  Loaded feature matrix: shape={df.shape}")
        return df

    def prepare_data(self, df: pd.DataFrame, test_days: int = 28):
        """Splits temporal train/test data and handles feature encoding."""
        max_date = df["date"].max()
        split_date = max_date - pd.Timedelta(days=test_days)
        
        feature_cols = [
            "day_of_week", "day_of_month", "month", "year", "quarter", "is_weekend",
            "lag_1", "lag_7", "lag_14", "lag_28",
            "rolling_mean_7", "rolling_std_7", "rolling_mean_28", "rolling_std_28",
            "price_base", "price_ratio_30d"
        ]
        
        cat_cols = ["store_id", "dept_name", "class_name"]
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype("category").cat.codes
                if col not in feature_cols:
                    feature_cols.append(col)
                
        train = df[df["date"] <= split_date].copy()
        test = df[df["date"] > split_date].copy()
        
        X_train = train[feature_cols].fillna(0)
        y_train = train["quantity"].values
        X_test = test[feature_cols].fillna(0)
        y_test = test["quantity"].values
        
        logger.info(f"Train set: {len(X_train):,} samples | Test set: {len(X_test):,} samples")
        return X_train, y_train, X_test, y_test, test, feature_cols

    def train_lightgbm(self, X_train, y_train, X_test, y_test) -> tuple[np.ndarray, dict]:
        """Trains LightGBM Regressor."""
        logger.info("Training LightGBM Forecaster...")
        t0 = time.time()
        model = lgb.LGBMRegressor(
            n_estimators=350,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = compute_metrics(y_test, preds)
        metrics["Model"] = "LightGBM Forecaster"
        logger.info(f"  LightGBM completed in {time.time() - t0:.2f}s -> WAPE: {metrics['WAPE']}, MAE: {metrics['MAE']}")
        
        model_path = os.path.join(self.models_dir, "lightgbm_model.txt")
        model.booster_.save_model(model_path)
        with open(os.path.join(self.models_dir, "best_model.pkl"), "wb") as f:
            pickle.dump(model, f)
            
        return preds, metrics

    def train_xgboost(self, X_train, y_train, X_test, y_test) -> tuple[np.ndarray, dict]:
        """Trains XGBoost Regressor."""
        logger.info("Training XGBoost TimeSeries Regressor...")
        t0 = time.time()
        model = xgb.XGBRegressor(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = compute_metrics(y_test, preds)
        metrics["Model"] = "XGBoost TimeSeries"
        logger.info(f"  XGBoost completed in {time.time() - t0:.2f}s -> WAPE: {metrics['WAPE']}, MAE: {metrics['MAE']}")
        return preds, metrics

    def train_catboost(self, X_train, y_train, X_test, y_test) -> tuple[np.ndarray, dict]:
        """Trains CatBoost Regressor."""
        logger.info("Training CatBoost Regressor...")
        t0 = time.time()
        model = CatBoostRegressor(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            random_seed=42,
            verbose=0
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = compute_metrics(y_test, preds)
        metrics["Model"] = "CatBoost Regressor"
        logger.info(f"  CatBoost completed in {time.time() - t0:.2f}s -> WAPE: {metrics['WAPE']}, MAE: {metrics['MAE']}")
        return preds, metrics

    def train_random_forest(self, X_train, y_train, X_test, y_test) -> tuple[np.ndarray, dict]:
        """Trains Random Forest Regressor on representative sample."""
        logger.info("Training Random Forest Regressor (Sampled)...")
        t0 = time.time()
        sample_size = min(250000, len(X_train))
        idx = np.random.choice(len(X_train), sample_size, replace=False)
        
        model = RandomForestRegressor(n_estimators=60, max_depth=12, random_state=42, n_jobs=-1)
        model.fit(X_train.iloc[idx], y_train[idx])
        preds = model.predict(X_test)
        metrics = compute_metrics(y_test, preds)
        metrics["Model"] = "Random Forest Regressor"
        logger.info(f"  Random Forest completed in {time.time() - t0:.2f}s -> WAPE: {metrics['WAPE']}, MAE: {metrics['MAE']}")
        return preds, metrics

    def train_prophet(self, df: pd.DataFrame, test_days: int = 28) -> dict:
        """Trains Prophet on aggregate daily store demand."""
        logger.info("Training Prophet Time Series Model...")
        t0 = time.time()
        daily_agg = df.groupby("date")["quantity"].sum().reset_index()
        daily_agg.columns = ["ds", "y"]
        
        split_date = daily_agg["ds"].max() - pd.Timedelta(days=test_days)
        train = daily_agg[daily_agg["ds"] <= split_date]
        test = daily_agg[daily_agg["ds"] > split_date]
        
        m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        m.fit(train)
        
        future = m.make_future_dataframe(periods=test_days)
        forecast = m.predict(future)
        
        pred_test = forecast.tail(test_days)["yhat"].values
        y_test = test["y"].values
        
        metrics = compute_metrics(y_test, pred_test)
        metrics["Model"] = "Prophet (Aggregate TimeSeries)"
        logger.info(f"  Prophet completed in {time.time() - t0:.2f}s -> WAPE: {metrics['WAPE']}, MAE: {metrics['MAE']}")
        return metrics

    def train_arima(self, df: pd.DataFrame, test_days: int = 28) -> dict:
        """Trains ARIMA/SARIMA on aggregate daily store demand."""
        logger.info("Training ARIMA/SARIMA Time Series Model...")
        t0 = time.time()
        daily_agg = df.groupby("date")["quantity"].sum().reset_index()
        ts = daily_agg["quantity"].values
        
        train_ts = ts[:-test_days]
        test_ts = ts[-test_days:]
        
        model = ARIMA(train_ts, order=(5, 1, 0))
        fit_model = model.fit()
        forecast = fit_model.forecast(steps=test_days)
        
        metrics = compute_metrics(test_ts, forecast)
        metrics["Model"] = "ARIMA (5,1,0)"
        logger.info(f"  ARIMA completed in {time.time() - t0:.2f}s -> WAPE: {metrics['WAPE']}, MAE: {metrics['MAE']}")
        return metrics

    def run_all_models(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Runs training, prediction, evaluation across all ML models."""
        df = self.load_feature_matrix()
        X_train, y_train, X_test, y_test, test_df, feature_cols = self.prepare_data(df)
        
        base_path = os.path.join(self.reports_dir, "baseline_leaderboard.csv")
        if os.path.exists(base_path):
            leaderboard_list = pd.read_csv(base_path).to_dict(orient="records")
        else:
            leaderboard_list = []
            
        lgb_preds, lgb_m = self.train_lightgbm(X_train, y_train, X_test, y_test)
        xgb_preds, xgb_m = self.train_xgboost(X_train, y_train, X_test, y_test)
        cat_preds, cat_m = self.train_catboost(X_train, y_train, X_test, y_test)
        rf_preds, rf_m = self.train_random_forest(X_train, y_train, X_test, y_test)
        
        prophet_m = self.train_prophet(df)
        arima_m = self.train_arima(df)
        
        leaderboard_list.extend([lgb_m, xgb_m, cat_m, rf_m, prophet_m, arima_m])
        
        master_df = pd.DataFrame(leaderboard_list)[["Model", "WAPE", "MAE", "RMSE", "MAPE (%)", "Bias (%)", "R2"]]
        master_df = master_df.sort_values(by="WAPE", ascending=True).reset_index(drop=True)
        
        out_leaderboard = os.path.join(self.reports_dir, "model_leaderboard.csv")
        master_df.to_csv(out_leaderboard, index=False)
        logger.info(f"Master Model Leaderboard saved to '{out_leaderboard}'.")
        
        test_predictions = test_df[["date", "store_id", "item_id", "quantity", "price_base"]].copy()
        test_predictions["predicted_quantity"] = lgb_preds
        out_preds = os.path.join(self.data_dir, "test_predictions.parquet")
        test_predictions.to_parquet(out_preds, index=False)
        logger.info(f"Saved test predictions to '{out_preds}'.")
        
        return master_df, test_predictions

def main():
    start_time = time.time()
    logger.info("Starting Step 7: Machine Learning & Time-Series Demand Forecasting Engine...")
    
    engine = MLForecastingEngine()
    master_leaderboard, test_preds = engine.run_all_models()
    
    elapsed = time.time() - start_time
    logger.info(f"\n=========================================================")
    logger.info(f"SUCCESS: ML Forecasting Pipeline completed in {elapsed:.2f} seconds.")
    logger.info("\n🏆 MASTER MODEL LEADERBOARD (Baselines + ML Models):")
    logger.info("\n" + master_leaderboard.to_string(index=False))
    logger.info(f"\nSaved Artifacts:")
    logger.info("  - Master Leaderboard: 'reports/model_leaderboard.csv'")
    logger.info("  - Best Model: 'models_saved/best_model.pkl'")
    logger.info("  - Test Predictions: 'data/processed/test_predictions.parquet'")
    logger.info(f"=========================================================")

if __name__ == "__main__":
    main()
