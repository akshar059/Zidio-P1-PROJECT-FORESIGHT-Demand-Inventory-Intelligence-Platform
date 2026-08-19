"""
Step 3: Data Cleaning & English Translation Module

This module performs end-to-end data cleaning and Russian-to-English translation across all 8 datasets:
1. Removes redundant index columns (e.g., Unnamed: 0).
2. Translates Russian Cyrillic terms (catalog departments, classes, subclasses, item types) to English labels.
3. Standardizes data labels and attributes.
4. Parses date columns to ISO datetime format.
5. Strips whitespace from string identifiers and text attributes.
6. Handles missing values cleanly (catalog metadata, discount codes).
7. Cleans/filters transaction anomalies (negative quantities, negative prices).
8. Removes duplicate records.
9. Exports cleaned and processed dataset artifacts as both Parquet and CSV files.
"""

import os
import sys
import json
import logging
import time
import re
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
TRANSLATION_MAP_PATH = os.path.join(PROCESSED_DATA_DIR, "translation_mapping.json")

# Comprehensive Built-in English Translation Dictionary
DEFAULT_TRANSLATION_MAP = {
    # Departments & Categories
    "Молочные продукты": "Dairy Products",
    "Напитки": "Beverages",
    "Хлебобулочные изделия": "Bakery & Bread",
    "Мясо, птица, колбасы": "Meat, Poultry & Sausages",
    "Овощи, фрукты, ягоды, грибы": "Fruits & Vegetables",
    "Замороженная продукция": "Frozen Foods",
    "Кондитерские изделия": "Confectionery & Sweets",
    "Бакалея": "Grocery & Staples",
    "Рыба и морепродукты": "Fish & Seafood",
    "Косметика и гигиена": "Cosmetics & Hygiene",
    "Бытовая химия": "Household Chemicals",
    "Детские товары": "Baby Products",
    "Товары для дома": "Home & Household Goods",
    "Товары для животных": "Pet Supplies",
    "Алкогольные напитки": "Alcoholic Beverages",
    "Безалкогольные напитки": "Non-Alcoholic Beverages",
    "Табачные изделия": "Tobacco Products",
    "Готовая кулинария": "Ready-to-Eat & Culinary",
    "Здоровое питание": "Healthy Living & Organic",
    "Книги, пресса, канцелярия": "Books, Media & Stationery",
    "Автотовары, электроника": "Auto & Electronics",
    
    # Common Product Types & Terms
    "Молоко": "Milk",
    "Сыр": "Cheese",
    "Масло": "Butter & Oil",
    "Творог": "Cottage Cheese / Quark",
    "Сметано-творожные": "Sour Cream & Quark",
    "Йогурты": "Yogurts",
    "Хлеб": "Bread",
    "Булочки": "Buns & Rolls",
    "Печенье, Крекеры": "Cookies & Crackers",
    "Конфеты Шоколадные": "Chocolate Candies",
    "Соки, Нектары": "Juices & Nectars",
    "Вода": "Water",
    "Газированная": "Carbonated",
    "Пиво": "Beer",
    "Вино": "Wine",
    "Водка": "Vodka",
    "Коньяк": "Cognac / Brandy",
    "Курица": "Chicken",
    "Говядина": "Beef",
    "Свинина": "Pork",
    "Индейка": "Turkey",
    "Колбаса": "Sausage",
    "Сосиски": "Frankfurters / Hot Dogs",
    "Пельмени": "Dumplings / Pelmeni",
    "Яблоки": "Apples",
    "Бананы": "Bananas",
    "Томаты": "Tomatoes",
    "Огурцы": "Cucumbers",
    "Картофель": "Potatoes",
    "Лук": "Onions",
    "Морковь": "Carrots",
    "Яйцо": "Eggs",
    "Чай": "Tea",
    "Кофе": "Coffee",
    "Сахар": "Sugar",
    "Соль": "Salt",
    "Мука": "Flour",
    "Макаронные изделия": "Pasta",
    "Крупы": "Grains & Cereals",
    "Консервы": "Canned Goods",
    "Шампуни": "Shampoos",
    "Мыло": "Soap",
    "Зубные пасты": "Toothpastes",
    "Стиральные порошки": "Laundry Detergents",
    "Бумажные изделия": "Paper Products",
    "Корм для кошек": "Cat Food",
    "Корм для собак": "Dog Food"
}

class DataCleaner:
    def __init__(self, output_dir: str = PROCESSED_DATA_DIR):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.translation_map = DEFAULT_TRANSLATION_MAP.copy()
        self._load_json_translation_map()

    def _load_json_translation_map(self):
        """Loads extended Russian-to-English translation dictionary from JSON if available."""
        if os.path.exists(TRANSLATION_MAP_PATH):
            try:
                with open(TRANSLATION_MAP_PATH, "r", encoding="utf-8") as f:
                    json_map = json.load(f)
                    self.translation_map.update(json_map)
                    logger.info(f"Loaded {len(json_map)} custom translations from '{TRANSLATION_MAP_PATH}'.")
            except Exception as e:
                logger.warning(f"Could not load translation map from '{TRANSLATION_MAP_PATH}': {e}")

    def _translate_cyrillic_series(self, series: pd.Series) -> pd.Series:
        """Translates Cyrillic Russian text strings in a Series to English labels."""
        def translate_val(val):
            if pd.isna(val):
                return val
            val_str = str(val).strip()
            if val_str in self.translation_map:
                return self.translation_map[val_str]
            
            # If word contains Cyrillic and is not in map, fallback to clean English title case
            if re.search(r'[\u0400-\u04FF]', val_str):
                # Try word-by-word replacement if possible
                words = val_str.split()
                translated_words = [self.translation_map.get(w, w) for w in words]
                res = " ".join(translated_words)
                # If still has Cyrillic, transliterate or label cleanly
                if re.search(r'[\u0400-\u04FF]', res):
                    return f"Category ({val_str})"
                return res
            return val_str

        return series.apply(translate_val)

    def _remove_unnamed_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drops unnamed index columns."""
        unnamed = [col for col in df.columns if col.startswith("Unnamed:")]
        if unnamed:
            df = df.drop(columns=unnamed)
        return df

    def _strip_string_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Strips leading and trailing whitespace from object/string columns."""
        for col in df.select_dtypes(include=["object", "string"]).columns:
            df[col] = df[col].astype(str).str.strip()
        return df

    def clean_stores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans stores metadata dataset and labels store attributes."""
        logger.info("Cleaning stores dataset...")
        df = self._remove_unnamed_cols(df)
        df = self._strip_string_cols(df)
        df["store_id"] = df["store_id"].astype(int)
        df["area"] = df["area"].astype(int)
        df = df.drop_duplicates(subset=["store_id"])
        logger.info(f"  Stores cleaned: shape={df.shape}")
        return df

    def clean_catalog(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans product catalog dataset, translates Russian text to English, and labels missing attributes."""
        logger.info("Cleaning catalog dataset & translating Russian terms to English labels...")
        df = self._remove_unnamed_cols(df)
        df = self._strip_string_cols(df)

        df["item_type"] = df["item_type"].fillna("Unknown SKU Type")
        
        # Apply Russian to English translation across text columns
        text_cols = ["dept_name", "class_name", "subclass_name", "item_type"]
        for col in text_cols:
            if col in df.columns:
                df[col] = self._translate_cyrillic_series(df[col])

        df["weight_volume_missing"] = df["weight_volume"].isnull().astype(int)
        df["weight_netto_missing"] = df["weight_netto"].isnull().astype(int)
        df["fatness_missing"] = df["fatness"].isnull().astype(int)

        df["weight_volume"] = df["weight_volume"].fillna(-1.0)
        df["weight_netto"] = df["weight_netto"].fillna(-1.0)
        df["fatness"] = df["fatness"].fillna(-1.0)

        df = df.drop_duplicates(subset=["item_id"])
        logger.info(f"  Catalog cleaned & translated: shape={df.shape}")
        return df

    def clean_sales(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans offline POS sales transactions and labels transaction fields."""
        logger.info("Cleaning sales dataset...")
        df = self._remove_unnamed_cols(df)
        df = self._strip_string_cols(df)

        df["date"] = pd.to_datetime(df["date"])
        initial_count = len(df)

        valid_mask = (df["quantity"] > 0) & (df["price_base"] > 0) & (df["sum_total"] > 0)
        df = df[valid_mask].copy()

        df = df.drop_duplicates()
        removed_count = initial_count - len(df)
        logger.info(f"  Sales cleaned: shape={df.shape} (Removed {removed_count:,} anomalous/duplicate rows)")
        return df

    def clean_online(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans e-commerce online sales transactions."""
        logger.info("Cleaning online sales dataset...")
        df = self._remove_unnamed_cols(df)
        df = self._strip_string_cols(df)

        df["date"] = pd.to_datetime(df["date"])
        initial_count = len(df)

        valid_mask = (df["quantity"] > 0) & (df["price_base"] > 0) & (df["sum_total"] > 0)
        df = df[valid_mask].copy()

        df = df.drop_duplicates()
        removed_count = initial_count - len(df)
        logger.info(f"  Online sales cleaned: shape={df.shape} (Removed {removed_count:,} anomalous/duplicate rows)")
        return df

    def clean_price_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans base price history dataset."""
        logger.info("Cleaning price_history dataset...")
        df = self._remove_unnamed_cols(df)
        df = self._strip_string_cols(df)

        df["date"] = pd.to_datetime(df["date"])
        initial_count = len(df)

        df = df[df["price"] > 0].copy()
        df = df.drop_duplicates(subset=["date", "item_id", "store_id"])
        removed_count = initial_count - len(df)
        logger.info(f"  Price history cleaned: shape={df.shape} (Removed {removed_count:,} invalid/duplicate rows)")
        return df

    def clean_discounts_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans promotional discounts dataset and labels promo codes."""
        logger.info("Cleaning discounts_history dataset...")
        df = self._remove_unnamed_cols(df)
        df = self._strip_string_cols(df)

        df["date"] = pd.to_datetime(df["date"])
        initial_count = len(df)

        df["promo_type_code"] = df["promo_type_code"].fillna(-1).astype(int)
        df["number_disc_day"] = df["number_disc_day"].fillna(1).astype(int)

        valid_mask = (df["sale_price_before_promo"] > 0) & (df["sale_price_time_promo"] > 0)
        df = df[valid_mask].copy()

        df = df.drop_duplicates()
        removed_count = initial_count - len(df)
        logger.info(f"  Discounts history cleaned: shape={df.shape} (Removed {removed_count:,} anomalous/duplicate rows)")
        return df

    def clean_markdowns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans markdowns dataset."""
        logger.info("Cleaning markdowns dataset...")
        df = self._remove_unnamed_cols(df)
        df = self._strip_string_cols(df)

        df["date"] = pd.to_datetime(df["date"])
        initial_count = len(df)

        valid_mask = (df["quantity"] > 0) & (df["price"] > 0) & (df["normal_price"] > 0)
        df = df[valid_mask].copy()

        df = df.drop_duplicates()
        removed_count = initial_count - len(df)
        logger.info(f"  Markdowns cleaned: shape={df.shape} (Removed {removed_count:,} anomalous/duplicate rows)")
        return df

    def clean_actual_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans actual item-store availability matrix."""
        logger.info("Cleaning actual_matrix dataset...")
        df = self._remove_unnamed_cols(df)
        df = self._strip_string_cols(df)

        df["date"] = pd.to_datetime(df["date"])
        df = df.drop_duplicates(subset=["item_id", "date", "store_id"])
        logger.info(f"  Actual matrix cleaned: shape={df.shape}")
        return df

    def clean_all(self, raw_datasets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Runs cleaning and translation pipelines across all 8 datasets."""
        cleaned = {}
        cleaned["stores"] = self.clean_stores(raw_datasets["stores"])
        cleaned["catalog"] = self.clean_catalog(raw_datasets["catalog"])
        cleaned["sales"] = self.clean_sales(raw_datasets["sales"])
        cleaned["online"] = self.clean_online(raw_datasets["online"])
        cleaned["price_history"] = self.clean_price_history(raw_datasets["price_history"])
        cleaned["discounts_history"] = self.clean_discounts_history(raw_datasets["discounts_history"])
        cleaned["markdowns"] = self.clean_markdowns(raw_datasets["markdowns"])
        cleaned["actual_matrix"] = self.clean_actual_matrix(raw_datasets["actual_matrix"])
        
        logger.info("Completed cleaning and English translation for all 8 datasets.")
        return cleaned

    def save_processed_datasets(self, cleaned_datasets: dict[str, pd.DataFrame]):
        """Saves cleaned DataFrames into destination processed folder in Parquet & CSV formats."""
        logger.info(f"Saving cleaned and processed files to '{self.output_dir}'...")
        
        for name, df in cleaned_datasets.items():
            csv_path = os.path.join(self.output_dir, f"cleaned_{name}.csv")
            parquet_path = os.path.join(self.output_dir, f"cleaned_{name}.parquet")
            
            df.to_csv(csv_path, index=False)
            df.to_parquet(parquet_path, index=False)
            
            csv_size_mb = os.path.getsize(csv_path) / (1024 * 1024)
            parquet_size_mb = os.path.getsize(parquet_path) / (1024 * 1024)
            
            logger.info(f"Saved 'cleaned_{name}': CSV ({csv_size_mb:.2f} MB), Parquet ({parquet_size_mb:.2f} MB)")
            
        logger.info("All cleaned and processed files saved successfully!")

def main():
    start_time = time.time()
    logger.info("Starting Step 3: Data Cleaning, English Translation & Processing Pipeline...")
    from src.data_collection import DataCollector
    collector = DataCollector()
    raw_data = collector.collect_all_datasets()
    
    cleaner = DataCleaner()
    cleaned_data = cleaner.clean_all(raw_data)
    cleaner.save_processed_datasets(cleaned_data)
    
    elapsed = time.time() - start_time
    logger.info(f"\n=========================================================")
    logger.info(f"SUCCESS: Step 3 completed in {elapsed:.2f} seconds.")
    logger.info(f"Cleaned artifacts with English labels saved to 'data/processed/'.")
    logger.info(f"=========================================================")

if __name__ == "__main__":
    main()
