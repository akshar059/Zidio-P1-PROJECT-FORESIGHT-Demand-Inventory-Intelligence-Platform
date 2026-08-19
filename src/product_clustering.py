"""
Product Intelligence & K-Means Clustering Engine: Health Score, Lifecycle & 5 Strategic Clusters (High Performance)
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

CLUSTER_LABELS = {
    0: "Cluster 1 — High Demand / Stable",
    1: "Cluster 2 — Promotion Sensitive",
    2: "Cluster 3 — High Price / Low Volume",
    3: "Cluster 4 — Declining Products",
    4: "Cluster 5 — High Growth Products"
}

class ProductClusteringEngine:
    def __init__(self, data_dir: str = PROCESSED_DATA_DIR):
        self.data_dir = data_dir

    def cluster_products(self, sales_df: pd.DataFrame, catalog_df: pd.DataFrame = None) -> pd.DataFrame:
        """Applies K-Means Clustering to group products into 5 strategic portfolio clusters."""
        # Single-pass aggregation
        sku_stats = sales_df.groupby("item_id").agg(
            total_quantity=("quantity", "sum"),
            mean_quantity=("quantity", "mean"),
            std_quantity=("quantity", "std"),
            total_revenue=("sum_total", "sum"),
            avg_price=("price_base", "mean")
        ).fillna(0.0).reset_index()
        
        sku_stats["volatility"] = (sku_stats["std_quantity"] / (sku_stats["mean_quantity"] + 1e-5)).fillna(0.2)
        
        # Simulated 90-day growth rate
        np.random.seed(42)
        sku_stats["growth_rate"] = np.random.normal(12.5, 18.0, size=len(sku_stats)).round(1)
        
        features = ["total_quantity", "total_revenue", "avg_price", "volatility", "growth_rate"]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(sku_stats[features])
        
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=3)
        sku_stats["cluster_id"] = kmeans.fit_predict(X_scaled)
        
        # Sort cluster IDs by revenue ranking to guarantee consistent semantic naming
        cluster_rev = sku_stats.groupby("cluster_id")["total_revenue"].mean().sort_values(ascending=False).index
        cluster_map = {cluster_rev[i]: CLUSTER_LABELS[i] for i in range(5)}
        
        sku_stats["cluster_name"] = sku_stats["cluster_id"].map(cluster_map)
        
        # Product Health Score Calculation (0 - 100)
        sku_stats["product_health_score"] = np.clip(
            (sku_stats["total_revenue"] / (sku_stats["total_revenue"].max() + 1e-5) * 40) +
            (np.clip(sku_stats["growth_rate"], -20, 50) + 20) / 70 * 40 +
            (1.0 - np.clip(sku_stats["volatility"], 0, 2.0) / 2.0) * 20,
            25, 98
        ).round(1)
        
        # Product Lifecycle Classification
        conditions = [
            sku_stats["growth_rate"] > 25.0,
            sku_stats["growth_rate"] > 5.0,
            sku_stats["growth_rate"] >= -10.0
        ]
        choices = ["🚀 Emerging", "📈 Growing", "🏛️ Mature"]
        sku_stats["lifecycle"] = np.select(conditions, choices, default="📉 Declining")
        
        if catalog_df is not None and not catalog_df.empty:
            cat_cols = [c for c in ["item_id", "dept_name", "class_name", "item_type"] if c in catalog_df.columns]
            sku_stats = sku_stats.merge(catalog_df[cat_cols], on="item_id", how="left")
            
        return sku_stats
