import os
import glob
from typing import Tuple, Dict, Optional
import pandas as pd

COLUMN_ALIASES = {
    'name': ['restaurant name', 'name', 'restaurant_name', 'title', 'rest_name'],
    'cuisines': ['cuisines', 'cuisine', 'cuisine_type', 'cuisines_offered'],
    'rate': ['rate', 'rating', 'aggregate_rating', 'stars', 'user_rating'],
    'approx_cost': ['approx_cost(for two people)', 'average_cost_for_two', 'price', 'cost', 'avg_cost', 'price_range'],
    'location': ['location', 'city', 'locality', 'address', 'neighborhood'],
    'rest_type': ['rest_type', 'restaurant_type', 'type'],
    'online_order': ['online_order', 'online', 'has_online_order'],
    'book_table': ['book_table', 'table_booking', 'has_table_booking']
}


def find_dataset(data_dir: str = 'data') -> Optional[str]:
    """
    Search for the main restaurant dataset in data directory.
    Prioritizes full datasets (e.g. zomato.csv) over sample datasets.
    """
    if not os.path.exists(data_dir):
        return None

    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    if not csv_files:
        return None

    # Priority 1: Exact matches for standard Zomato full datasets
    for f in csv_files:
        base = os.path.basename(f).lower()
        if base == 'zomato.csv' or base == 'zomato_restaurants.csv':
            return f

    # Priority 2: Any non-sample CSV
    for f in csv_files:
        base = os.path.basename(f).lower()
        if not base.startswith('sample'):
            return f

    # Priority 3: Fallback to sample
    return csv_files[0]


def detect_column_mapping(df: pd.DataFrame) -> Dict[str, str]:
    """
    Dynamically maps raw dataset column headers to canonical feature names.
    """
    mapping = {}
    lower_cols = {col.strip().lower(): col for col in df.columns}

    for canonical, variations in COLUMN_ALIASES.items():
        for var in variations:
            if var in lower_cols:
                mapping[canonical] = lower_cols[var]
                break

    return mapping


def load_raw_dataset(file_path: Optional[str] = None, data_dir: str = 'data') -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Loads raw restaurant dataset, validates columns, and normalizes column mapping.
    
    Returns:
        Tuple of (DataFrame with canonical column names, Mapping dictionary)
    """
    target_path = file_path or find_dataset(data_dir)

    if not target_path or not os.path.exists(target_path):
        raise FileNotFoundError(
            f"No restaurant dataset found. Please place your Zomato CSV dataset into the '{data_dir}/' directory."
        )

    print(f"[*] Loading dataset from: {target_path}")

    # Load with fallback encoding and low_memory handling
    try:
        df = pd.read_csv(target_path, low_memory=False, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(target_path, low_memory=False, encoding='latin-1')

    if df.empty:
        raise ValueError(f"The loaded dataset file '{target_path}' is empty.")

    col_map = detect_column_mapping(df)

    if 'name' not in col_map:
        # Fallback to any column containing name
        for col in df.columns:
            if 'name' in col.lower():
                col_map['name'] = col
                break
        if 'name' not in col_map:
            raise ValueError("Dataset must contain a column for Restaurant Name.")

    if 'cuisines' not in col_map:
        for col in df.columns:
            if 'cuis' in col.lower() or 'food' in col.lower():
                col_map['cuisines'] = col
                break

    # Standardize DataFrame column names using mapping
    rename_dict = {orig: canon for canon, orig in col_map.items()}
    df_standard = df.rename(columns=rename_dict).copy()

    print(f"[+] Loaded {len(df_standard)} records with {len(df_standard.columns)} columns.")
    print(f"[+] Column mapping: {col_map}")

    return df_standard, col_map
