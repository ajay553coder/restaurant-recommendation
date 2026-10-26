import re
from typing import Tuple, Dict, Any, List, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Top 25 representative cuisines in Indian/Zomato restaurant domain
TOP_CUISINES = [
    'North Indian', 'Chinese', 'South Indian', 'Fast Food', 'Biryani',
    'Desserts', 'Beverages', 'Continental', 'Cafe', 'Street Food',
    'Bakery', 'Italian', 'Andhra', 'Ice Cream', 'Pizza',
    'Burger', 'Rolls', 'Seafood', 'Mughlai', 'Kerala',
    'Juices', 'Momos', 'Mithai', 'Kebab', 'Arabian'
]

# Top 25 locations
TOP_LOCATIONS = [
    'Whitefield', 'BTM', 'Electronic City', 'HSR', 'Marathahalli',
    'Indiranagar', 'JP Nagar', 'Bannerghatta Road', 'Jayanagar', 'Bellandur',
    'Sarjapur Road', 'New BEL Road', 'Banashankari', 'Koramangala 5th Block',
    'Kalyan Nagar', 'Malleshwaram', 'Rajajinagar', 'Basavanagudi', 'Banaswadi',
    'Frazer Town', 'Vijayawada', 'Hyderabad', 'Guntur', 'Rajahmundry', 'Eluru'
]

# Top 10 restaurant types
TOP_REST_TYPES = [
    'Quick Bytes', 'Casual Dining', 'Cafe', 'Delivery', 'Dessert Parlor',
    'Takeaway', 'Bakery', 'Sweet Shop', 'Pub', 'Fine Dining'
]


def clean_rating(val: Any) -> float:
    """
    Parses raw rating strings into numeric floats out of 5.0.
    Returns np.nan for unrated/missing values ('NEW', '-', NaN).
    Never treats missing ratings as genuine low ratings.
    """
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip()
    if not val_str or val_str.lower() in ['new', '-', 'none', 'nan']:
        return np.nan
    if '/' in val_str:
        val_str = val_str.split('/')[0].strip()
    try:
        score = float(val_str)
        return score if 1.0 <= score <= 5.0 else np.nan
    except ValueError:
        return np.nan


def clean_cost(val: Any) -> float:
    """
    Parses cost/price strings (e.g. '1,200', '₹500') into numeric floats.
    Returns np.nan for missing/invalid prices. Never converts unknown prices to zero.
    """
    if pd.isna(val):
        return np.nan
    val_str = str(val).replace(',', '').replace('₹', '').strip()
    match = re.search(r'\d+(\.\d+)?', val_str)
    if match:
        try:
            cost = float(match.group(0))
            return cost if cost > 0 else np.nan
        except ValueError:
            return np.nan
    return np.nan


def assign_rating_category(rating: float) -> Optional[str]:
    """
    Assigns transparent rating categories based on empirical distribution:
      - 'Low': rating < 3.5
      - 'Medium': 3.5 <= rating < 4.0
      - 'High': rating >= 4.0
    Returns None if rating is missing/NaN.
    """
    if pd.isna(rating):
        return None
    if rating < 3.5:
        return 'Low'
    elif rating < 4.0:
        return 'Medium'
    else:
        return 'High'


def clean_text_field(val: Any, default: str = 'Not Specified') -> str:
    """Cleans categorical string fields."""
    if pd.isna(val):
        return default
    s = str(val).strip()
    return s if s else default


def preprocess_dataset(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Full data preprocessing pipeline:
    1. Records initial dataset dimensions and missing value statistics.
    2. Removes duplicate restaurant records (restaurants listed across multiple categories).
    3. Converts rating and price fields to numeric without falsifying missing data.
    4. Imputes missing price with median cost to preserve record availability.
    5. Normalizes text fields (cuisines, location, rest_type).
    6. Extracts binary flags for online order and table booking.
    7. Encodes top cuisines as multi-hot binary indicators.
    8. Creates documented rating categories ('Low', 'Medium', 'High') for Decision Tree.
    9. Compiles a comprehensive preprocessing report.
    """
    report = {
        'original_row_count': int(len(df_raw)),
        'original_col_count': int(len(df_raw.columns)),
        'missing_values_before': {k: int(v) for k, v in df_raw.isnull().sum().to_dict().items()}
    }

    df = df_raw.copy()

    # Step 1: Deduplication
    # Zomato raw data lists the same restaurant repeatedly across multiple search sections.
    subset_keys = ['name']
    if 'location' in df.columns:
        subset_keys.append('location')
    rows_before_dedup = len(df)
    df = df.drop_duplicates(subset=subset_keys, keep='first').reset_index(drop=True)
    report['duplicates_removed'] = int(rows_before_dedup - len(df))

    # Step 2: Rating conversion
    if 'rate' in df.columns:
        df['rating_num'] = df['rate'].apply(clean_rating)
    else:
        df['rating_num'] = np.nan

    # Step 3: Cost conversion
    if 'approx_cost' in df.columns:
        df['cost_num'] = df['approx_cost'].apply(clean_cost)
    else:
        df['cost_num'] = np.nan

    # Median cost imputation for missing costs (preserves valid restaurants, does NOT set to 0)
    median_cost = df['cost_num'].median()
    if pd.isna(median_cost) or median_cost <= 0:
        median_cost = 400.0
    report['median_cost_imputed'] = float(median_cost)
    df['cost_num_imputed'] = df['cost_num'].fillna(median_cost)

    # Step 4: Rating categories for Decision Tree (unrated rows remain None/NaN)
    df['rating_category'] = df['rating_num'].apply(assign_rating_category)

    # Step 5: Clean categorical fields
    df['cuisines_clean'] = df['cuisines'].apply(lambda x: clean_text_field(x, 'Multi-Cuisine')) if 'cuisines' in df.columns else 'Multi-Cuisine'
    df['location_clean'] = df['location'].apply(lambda x: clean_text_field(x, 'Unknown')) if 'location' in df.columns else 'Unknown'
    df['rest_type_clean'] = df['rest_type'].apply(lambda x: clean_text_field(x, 'Casual Dining')) if 'rest_type' in df.columns else 'Casual Dining'

    # Step 6: Binary indicators
    def parse_bool_flag(val):
        if pd.isna(val): return 0
        s = str(val).strip().lower()
        return 1 if s in ['yes', '1', 'true', 't'] else 0

    df['has_online_order'] = df['online_order'].apply(parse_bool_flag) if 'online_order' in df.columns else 0
    df['has_table_booking'] = df['book_table'].apply(parse_bool_flag) if 'book_table' in df.columns else 0

    # Step 7: Multi-hot encoding for top cuisines
    for cuisine in TOP_CUISINES:
        col_name = f"cuisine_{cuisine.lower().replace(' ', '_')}"
        df[col_name] = df['cuisines_clean'].apply(lambda x: 1 if cuisine.lower() in str(x).lower() else 0)

    # Step 8: Final record summary
    report['final_row_count'] = int(len(df))
    report['valid_ratings_count'] = int(df['rating_num'].notna().sum())
    report['unrated_count'] = int(df['rating_num'].isna().sum())
    report['valid_cost_count'] = int(df['cost_num'].notna().sum())
    report['rating_category_distribution'] = {k: int(v) for k, v in df['rating_category'].value_counts().to_dict().items()}
    report['final_feature_list'] = list(df.columns)

    print(f"[+] Preprocessing complete: {report['original_row_count']} -> {report['final_row_count']} unique records.")
    return df, report


class FeaturePipeline:
    """
    Scales and encodes structured features for KNN and K-Means algorithms.
    Uses pure structured features: NO NLP, NO TF-IDF.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
        self.fitted = False

    def _extract_feature_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)

        # Numerical: Cost for two (scaled)
        features['cost_scaled'] = df['cost_num_imputed']

        # Numerical: Rating (with median imputation for vector space continuity)
        median_rating = df['rating_num'].median()
        if pd.isna(median_rating):
            median_rating = 3.7
        features['rating_scaled'] = df['rating_num'].fillna(median_rating)

        # Binary attributes
        features['has_online_order'] = df['has_online_order']
        features['has_table_booking'] = df['has_table_booking']

        # Multi-hot cuisines
        for cuisine in TOP_CUISINES:
            col_name = f"cuisine_{cuisine.lower().replace(' ', '_')}"
            features[col_name] = df[col_name] if col_name in df.columns else 0

        # One-hot top locations
        for loc in TOP_LOCATIONS:
            col_name = f"loc_{loc.lower().replace(' ', '_')}"
            features[col_name] = df['location_clean'].apply(lambda x: 1 if loc.lower() in str(x).lower() else 0)

        # One-hot top restaurant types
        for rt in TOP_REST_TYPES:
            col_name = f"type_{rt.lower().replace(' ', '_')}"
            features[col_name] = df['rest_type_clean'].apply(lambda x: 1 if rt.lower() in str(x).lower() else 0)

        return features

    def fit(self, df: pd.DataFrame):
        feat_df = self._extract_feature_dataframe(df)
        self.feature_names = list(feat_df.columns)
        # Fit scaler on numerical columns
        num_cols = ['cost_scaled', 'rating_scaled']
        self.scaler.fit(feat_df[num_cols])
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("FeaturePipeline must be fitted before transform.")
        feat_df = self._extract_feature_dataframe(df)
        num_cols = ['cost_scaled', 'rating_scaled']
        feat_df[num_cols] = self.scaler.transform(feat_df[num_cols])
        return feat_df.values.astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def transform_query(self, cuisine: Optional[str], budget: Optional[float], min_rating: Optional[float], location: Optional[str]) -> np.ndarray:
        """
        Converts user query preference parameters into a structured query vector
        matching the exact feature dimensions.
        """
        query_dict = {}

        # Budget
        b_val = budget if (budget and budget > 0) else 500.0
        r_val = min_rating if (min_rating and min_rating > 0) else 3.7

        num_df = pd.DataFrame([{'cost_scaled': b_val, 'rating_scaled': r_val}])
        num_scaled = self.scaler.transform(num_df)[0]
        query_dict['cost_scaled'] = num_scaled[0]
        query_dict['rating_scaled'] = num_scaled[1]

        query_dict['has_online_order'] = 1
        query_dict['has_table_booking'] = 0

        # Cuisines multi-hot (weight=2.0 for higher preference alignment)
        c_lower = str(cuisine).lower() if cuisine and cuisine != 'All' else ''
        for c in TOP_CUISINES:
            col_name = f"cuisine_{c.lower().replace(' ', '_')}"
            query_dict[col_name] = 2.0 if (c_lower and c.lower() in c_lower) else 0.0

        # Location one-hot (weight=2.0 for location alignment)
        loc_lower = str(location).lower() if location and location != 'All' else ''
        for loc in TOP_LOCATIONS:
            col_name = f"loc_{loc.lower().replace(' ', '_')}"
            query_dict[col_name] = 2.0 if (loc_lower and loc.lower() in loc_lower) else 0.0

        # Rest types
        for rt in TOP_REST_TYPES:
            col_name = f"type_{rt.lower().replace(' ', '_')}"
            query_dict[col_name] = 0.0

        # Ensure correct column ordering matching feature_names
        query_vector = np.array([query_dict.get(fn, 0.0) for fn in self.feature_names], dtype=np.float32).reshape(1, -1)
        return query_vector
