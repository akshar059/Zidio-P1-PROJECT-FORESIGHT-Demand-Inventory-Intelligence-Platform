"""
Step 4: Exploratory Data Analysis (EDA) Module (Enhanced with Complex Visualizations)

Performs comprehensive statistical, temporal, categorical, price-elasticity,
Pareto, and multi-variate correlation analyses on cleaned retail datasets.
Generates 11 high-resolution visualization charts in reports/figures/.
"""

import os
import sys
import logging
import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environment
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")

# Configure Global Chart Style
plt.style.use('ggplot')
sns.set_palette("muted")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

class ExploratoryDataAnalysis:
    def __init__(self, data_dir: str = PROCESSED_DATA_DIR, figures_dir: str = FIGURES_DIR):
        self.data_dir = data_dir
        self.figures_dir = figures_dir
        os.makedirs(self.figures_dir, exist_ok=True)
        self.datasets = {}

    def load_processed_data(self):
        """Loads cleaned parquet datasets for high-speed analysis."""
        logger.info("Loading cleaned datasets for EDA...")
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Processed data directory not found at '{self.data_dir}'. Run data cleaning first.")
            
        for file in os.listdir(self.data_dir):
            if file.endswith('.parquet') and file.startswith('cleaned_'):
                key = file.replace('cleaned_', '').replace('.parquet', '')
                path = os.path.join(self.data_dir, file)
                self.datasets[key] = pd.read_parquet(path)
                logger.info(f"  Loaded '{key}': shape={self.datasets[key].shape}")
        return self.datasets

    def plot_monthly_demand_heatmap(self, sales: pd.DataFrame):
        """Chart 1: Monthly Demand Heatmap (Year x Month)."""
        sales_copy = sales.copy()
        sales_copy["year"] = sales_copy["date"].dt.year
        sales_copy["month"] = sales_copy["date"].dt.month_name()
        
        month_order = ["January", "February", "March", "April", "May", "June", 
                       "July", "August", "September", "October", "November", "December"]
        
        pivot_df = sales_copy.pivot_table(index="year", columns="month", values="sum_total", aggfunc="sum") / 1e6
        pivot_df = pivot_df.reindex(columns=[m for m in month_order if m in pivot_df.columns])
        
        plt.figure(figsize=(12, 5))
        sns.heatmap(pivot_df, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'Revenue ($ Millions)'})
        plt.title("Monthly Revenue Heatmap Grid (Year-over-Year in $ Millions)", fontsize=14, fontweight="bold", pad=12)
        plt.xlabel("Month", fontsize=11)
        plt.ylabel("Year", fontsize=11)
        plt.tight_layout()
        
        out_path = os.path.join(self.figures_dir, "monthly_demand_heatmap.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        logger.info(f"  Saved complex chart: {out_path}")

    def analyze_sales_demand(self) -> dict:
        """Analyzes sales demand trends, total revenue, and time horizons."""
        logger.info("Analyzing Sales & Demand Dynamics...")
        sales = self.datasets["sales"]
        online = self.datasets["online"]
        
        pos_revenue = sales["sum_total"].sum()
        pos_units = sales["quantity"].sum()
        online_revenue = online["sum_total"].sum()
        online_units = online["quantity"].sum()
        
        total_revenue = pos_revenue + online_revenue
        total_units = pos_units + online_units
        
        start_date = min(sales["date"].min(), online["date"].min())
        end_date = max(sales["date"].max(), online["date"].max())
        
        stats = {
            "start_date": str(start_date.date()),
            "end_date": str(end_date.date()),
            "total_revenue": float(total_revenue),
            "total_units": float(total_units),
            "pos_revenue": float(pos_revenue),
            "online_revenue": float(online_revenue),
            "pos_share_pct": float(pos_revenue / total_revenue * 100),
            "online_share_pct": float(online_revenue / total_revenue * 100),
        }
        
        # Plot Time Series Demand
        daily_pos = sales.groupby("date")["sum_total"].sum().reset_index()
        daily_online = online.groupby("date")["sum_total"].sum().reset_index()
        
        plt.figure(figsize=(14, 6))
        plt.plot(daily_pos["date"], daily_pos["sum_total"] / 1e3, label="In-Store POS ($K)", color="#2b5c8f", linewidth=1.5)
        plt.plot(daily_online["date"], daily_online["sum_total"] / 1e3, label="Online ($K)", color="#e05d44", linewidth=1.5, alpha=0.8)
        plt.title("Daily Revenue Timeline (POS vs Online)", fontsize=14, fontweight="bold", pad=12)
        plt.xlabel("Date", fontsize=11)
        plt.ylabel("Daily Revenue ($ Thousands)", fontsize=11)
        plt.legend(frameon=True, facecolor="white")
        plt.tight_layout()
        
        trend_path = os.path.join(self.figures_dir, "sales_trend_time_series.png")
        plt.savefig(trend_path, dpi=300)
        plt.close()
        logger.info(f"  Saved figure: {trend_path}")
        
        # Monthly Demand Heatmap
        self.plot_monthly_demand_heatmap(sales)
        
        return stats

    def plot_pareto_revenue_analysis(self, sales: pd.DataFrame):
        """Chart 2: Pareto Chart (80/20 Rule Analysis by Product SKU)."""
        sku_rev = sales.groupby("item_id")["sum_total"].sum().sort_values(ascending=False).reset_index()
        total_rev = sku_rev["sum_total"].sum()
        sku_rev["cum_pct"] = (sku_rev["sum_total"].cumsum() / total_rev) * 100
        
        top_50 = sku_rev.head(50).copy()
        top_50["rank"] = range(1, len(top_50) + 1)
        
        fig, ax1 = plt.subplots(figsize=(14, 6))
        ax2 = ax1.twinx()
        
        ax1.bar(top_50["rank"], top_50["sum_total"] / 1e6, color="#34495e", alpha=0.85, label="SKU Revenue ($M)")
        ax2.plot(top_50["rank"], top_50["cum_pct"], color="#e74c3c", linewidth=2.5, marker="o", markersize=4, label="Cumulative Revenue %")
        ax2.axhline(80, color="#27ae60", linestyle="--", linewidth=1.5, label="80% Pareto Threshold")
        
        ax1.set_xlabel("Top SKUs (Ranked by Sales Volume)", fontsize=11)
        ax1.set_ylabel("Revenue ($ Millions)", fontsize=11, color="#34495e")
        ax2.set_ylabel("Cumulative Revenue Percentage (%)", fontsize=11, color="#e74c3c")
        plt.title("Pareto Analysis: Cumulative Revenue Contribution by SKU (80/20 Rule)", fontsize=14, fontweight="bold", pad=12)
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", facecolor="white")
        plt.tight_layout()
        
        pareto_path = os.path.join(self.figures_dir, "pareto_revenue_analysis.png")
        plt.savefig(pareto_path, dpi=300)
        plt.close()
        logger.info(f"  Saved complex chart: {pareto_path}")

    def analyze_category_performance(self) -> dict:
        """Analyzes sales breakdown by product department, class hierarchy, and Pareto distribution."""
        logger.info("Analyzing Product Category Performance & Pareto Analysis...")
        sales = self.datasets["sales"]
        catalog = self.datasets["catalog"]
        
        merged = sales.merge(catalog[["item_id", "dept_name", "class_name", "subclass_name"]], on="item_id", how="left")
        merged["dept_name"] = merged["dept_name"].fillna("Unassigned")
        merged["class_name"] = merged["class_name"].fillna("Unassigned")
        
        dept_summary = merged.groupby("dept_name").agg(
            revenue=("sum_total", "sum"),
            units_sold=("quantity", "sum"),
            transaction_count=("sum_total", "count")
        ).sort_values(by="revenue", ascending=False).reset_index()
        
        top_10_depts = dept_summary.head(10)
        
        # Plot Top Departments
        plt.figure(figsize=(12, 6))
        sns.barplot(data=top_10_depts, y="dept_name", x=top_10_depts["revenue"] / 1e6, hue="dept_name", legend=False, palette="Blues_r")
        plt.title("Top 10 Departments by Total Revenue ($ Millions)", fontsize=14, fontweight="bold", pad=12)
        plt.xlabel("Revenue ($ Millions)", fontsize=11)
        plt.ylabel("Department Name", fontsize=11)
        plt.tight_layout()
        
        cat_path = os.path.join(self.figures_dir, "category_revenue_distribution.png")
        plt.savefig(cat_path, dpi=300)
        plt.close()
        logger.info(f"  Saved figure: {cat_path}")
        
        # Plot Top 15 Product Classes
        class_summary = merged.groupby("class_name")["sum_total"].sum().sort_values(ascending=False).head(15).reset_index()
        plt.figure(figsize=(12, 6))
        sns.barplot(data=class_summary, y="class_name", x=class_summary["sum_total"] / 1e6, hue="class_name", legend=False, palette="Purples_r")
        plt.title("Top 15 Product Classes by Revenue ($ Millions)", fontsize=14, fontweight="bold", pad=12)
        plt.xlabel("Revenue ($ Millions)", fontsize=11)
        plt.ylabel("Product Class", fontsize=11)
        plt.tight_layout()
        
        class_path = os.path.join(self.figures_dir, "top_product_classes_revenue.png")
        plt.savefig(class_path, dpi=300)
        plt.close()
        logger.info(f"  Saved complex chart: {class_path}")
        
        # Pareto SKU Analysis
        self.plot_pareto_revenue_analysis(sales)
        
        return {"top_departments": top_10_depts.to_dict(orient="records")}

    def analyze_store_performance(self) -> dict:
        """Analyzes store format performance and daily transaction violin distribution."""
        logger.info("Analyzing Store Performance & Violin Distributions...")
        sales = self.datasets["sales"]
        stores = self.datasets["stores"]
        
        store_sales = sales.groupby("store_id").agg(
            revenue=("sum_total", "sum"),
            units=("quantity", "sum"),
            txns=("sum_total", "count")
        ).reset_index()
        
        store_merged = store_sales.merge(stores, on="store_id", how="left")
        
        plt.figure(figsize=(10, 5))
        bars = plt.bar(store_merged["format"].astype(str), store_merged["revenue"] / 1e6, color="#3498db", width=0.5)
        plt.title("Revenue by Store Format ($ Millions)", fontsize=14, fontweight="bold", pad=12)
        plt.xlabel("Store Format", fontsize=11)
        plt.ylabel("Total Revenue ($ Millions)", fontsize=11)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1, f"${height:.2f}M", ha='center', va='bottom', fontweight='bold')
            
        plt.tight_layout()
        
        store_path = os.path.join(self.figures_dir, "store_performance_comparison.png")
        plt.savefig(store_path, dpi=300)
        plt.close()
        logger.info(f"  Saved figure: {store_path}")
        
        # Violin Plot of Daily Sales per Store Format
        daily_store_sales = sales.groupby(["date", "store_id"])["sum_total"].sum().reset_index()
        daily_store_merged = daily_store_sales.merge(stores[["store_id", "format"]], on="store_id", how="left")
        
        plt.figure(figsize=(11, 6))
        sns.violinplot(data=daily_store_merged, x="format", y=daily_store_merged["sum_total"] / 1e3, hue="format", legend=False, palette="Set2", inner="quartile")
        plt.title("Violin Distribution: Daily Revenue Variance by Store Format ($ Thousands)", fontsize=14, fontweight="bold", pad=12)
        plt.xlabel("Store Format", fontsize=11)
        plt.ylabel("Daily Store Revenue ($ Thousands)", fontsize=11)
        plt.tight_layout()
        
        violin_path = os.path.join(self.figures_dir, "store_sales_distribution_violin.png")
        plt.savefig(violin_path, dpi=300)
        plt.close()
        logger.info(f"  Saved complex chart: {violin_path}")
        
        return {"store_metrics": store_merged.to_dict(orient="records")}

    def analyze_seasonality_and_promotions(self) -> dict:
        """Analyzes day-of-week seasonality, correlation heatmaps, and price elasticity."""
        logger.info("Analyzing Seasonality, Multi-Variate Correlations & Price Volatility...")
        sales = self.datasets["sales"]
        disc = self.datasets["discounts_history"]
        price_hist = self.datasets["price_history"]
        
        # Day of week demand
        sales_copy = sales.copy()
        sales_copy["day_name"] = sales_copy["date"].dt.day_name()
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_summary = sales_copy.groupby("day_name")["quantity"].mean().reindex(dow_order).reset_index()
        
        plt.figure(figsize=(10, 5))
        sns.barplot(data=dow_summary, x="day_name", y="quantity", hue="day_name", legend=False, palette="magma")
        plt.title("Average Unit Sales Demand by Day of Week", fontsize=14, fontweight="bold", pad=12)
        plt.xlabel("Day of Week", fontsize=11)
        plt.ylabel("Average Quantity per Transaction", fontsize=11)
        plt.tight_layout()
        
        season_path = os.path.join(self.figures_dir, "day_of_week_seasonality.png")
        plt.savefig(season_path, dpi=300)
        plt.close()
        logger.info(f"  Saved figure: {season_path}")
        
        # Discount Depth Distribution
        disc_copy = disc.copy()
        disc_copy["discount_pct"] = (1 - (disc_copy["sale_price_time_promo"] / disc_copy["sale_price_before_promo"])) * 100
        disc_valid = disc_copy[(disc_copy["discount_pct"] >= 0) & (disc_copy["discount_pct"] <= 90)]
        
        plt.figure(figsize=(10, 5))
        sns.histplot(disc_valid["discount_pct"], bins=30, kde=True, color="#e74c3c")
        plt.title("Distribution of Promotional Discount Depth (%)", fontsize=14, fontweight="bold", pad=12)
        plt.xlabel("Discount Percentage (%)", fontsize=11)
        plt.ylabel("Frequency", fontsize=11)
        plt.tight_layout()
        
        disc_path = os.path.join(self.figures_dir, "discount_vs_sales_impact.png")
        plt.savefig(disc_path, dpi=300)
        plt.close()
        logger.info(f"  Saved figure: {disc_path}")
        
        # Multi-Variate Correlation Matrix Heatmap
        sample_sales = sales.sample(n=min(100000, len(sales)), random_state=42)[["quantity", "price_base", "sum_total"]].copy()
        corr = sample_sales.corr()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm", vmin=-1, vmax=1, cbar_kws={'label': 'Pearson Correlation'})
        plt.title("Multi-Variate Correlation Matrix (Quantity, Price, Revenue)", fontsize=14, fontweight="bold", pad=12)
        plt.tight_layout()
        
        corr_path = os.path.join(self.figures_dir, "price_elasticity_correlation.png")
        plt.savefig(corr_path, dpi=300)
        plt.close()
        logger.info(f"  Saved complex chart: {corr_path}")
        
        # Price Volatility Distribution
        price_std = price_hist.groupby("item_id")["price"].agg(["std", "mean"]).dropna()
        price_std["volatility_coef"] = price_std["std"] / price_std["mean"]
        
        plt.figure(figsize=(10, 5))
        sns.histplot(price_std["volatility_coef"][price_std["volatility_coef"] < 1.0], bins=40, color="#8e44ad", kde=True)
        plt.title("Base Price Volatility Distribution (Coefficient of Variation per Item)", fontsize=14, fontweight="bold", pad=12)
        plt.xlabel("Price Volatility Coefficient (Std / Mean)", fontsize=11)
        plt.ylabel("SKU Count", fontsize=11)
        plt.tight_layout()
        
        vol_path = os.path.join(self.figures_dir, "price_volatility_distribution.png")
        plt.savefig(vol_path, dpi=300)
        plt.close()
        logger.info(f"  Saved complex chart: {vol_path}")
        
        return {
            "avg_discount_pct": float(disc_valid["discount_pct"].mean()),
            "median_discount_pct": float(disc_valid["discount_pct"].median()),
        }

    def run_full_eda(self) -> dict:
        """Executes complete EDA pipeline."""
        self.load_processed_data()
        results = {
            "sales_summary": self.analyze_sales_demand(),
            "category_summary": self.analyze_category_performance(),
            "store_summary": self.analyze_store_performance(),
            "seasonality_summary": self.analyze_seasonality_and_promotions(),
        }
        logger.info("Exploratory Data Analysis completed with all complex visualizations!")
        return results

def main():
    start_time = time.time()
    logger.info("Starting Step 4: Advanced Exploratory Data Analysis (EDA) Phase...")
    
    eda = ExploratoryDataAnalysis()
    results = eda.run_full_eda()
    
    elapsed = time.time() - start_time
    logger.info(f"\n=========================================================")
    logger.info(f"SUCCESS: Advanced EDA completed in {elapsed:.2f} seconds.")
    logger.info(f"Generated 11 visualization charts in 'reports/figures/':")
    logger.info("  1.  sales_trend_time_series.png        (Daily revenue timeline)")
    logger.info("  2.  monthly_demand_heatmap.png        (YoY monthly heatmap grid)")
    logger.info("  3.  category_revenue_distribution.png (Top department breakdown)")
    logger.info("  4.  top_product_classes_revenue.png    (Top 15 product classes)")
    logger.info("  5.  pareto_revenue_analysis.png        (80/20 rule Pareto dual-axis)")
    logger.info("  6.  store_performance_comparison.png   (Store format revenue)")
    logger.info("  7.  store_sales_distribution_violin.png (Daily sales violin plots)")
    logger.info("  8.  day_of_week_seasonality.png        (Weekly demand seasonality)")
    logger.info("  9.  discount_vs_sales_impact.png       (Promotional discount distribution)")
    logger.info("  10. price_elasticity_correlation.png    (Multi-variate correlation matrix)")
    logger.info("  11. price_volatility_distribution.png   (Item price variance coefficient)")
    logger.info(f"=========================================================")

if __name__ == "__main__":
    main()
