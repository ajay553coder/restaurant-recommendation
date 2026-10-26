from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from src.preprocessing import TOP_CUISINES, TOP_LOCATIONS, TOP_REST_TYPES


class DecisionTreeRatingClassifier:
    """
    Algorithm C: Decision Tree Classification for Rating Category Prediction.
    
    Prediction Task:
    Predicts the rating category of a restaurant ('Low', 'Medium', 'High') from its
    operational and business attributes:
      - Budget / Cost for two
      - Online ordering availability
      - Table booking availability
      - Primary restaurant types
      - Cuisine categories
      - Neighborhood / Location
      
    CRITICAL METHODOLOGY NOTE:
    The original rating ('rate', 'rating_num') and all rating-derived attributes
    are strictly excluded from the feature inputs to prevent data leakage.
    Preprocessing (numerical scaling) is fitted strictly on the training partition.
    """
    def __init__(self, max_depth: int = 5, min_samples_leaf: int = 20, random_state: int = 42):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.dt_model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )
        self.lr_model = LogisticRegression(
            max_iter=1000,
            random_state=random_state
        )
        self.scaler = StandardScaler()
        self.feature_names = []
        self.classes_ = ['Low', 'Medium', 'High']
        self.evaluation_results = {}

    def _build_feature_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts independent features for rating classification.
        STRICTLY EXCLUDES rating_num, rate, or any target proxies.
        """
        X = pd.DataFrame(index=df.index)

        # Operational Numerical Attributes
        X['cost_num'] = df['cost_num_imputed']

        # Booking & Service Options
        X['has_online_order'] = df['has_online_order']
        X['has_table_booking'] = df['has_table_booking']

        # Multi-hot cuisines
        for cuisine in TOP_CUISINES:
            col_name = f"cuisine_{cuisine.lower().replace(' ', '_')}"
            X[col_name] = df[col_name] if col_name in df.columns else 0

        # One-hot top locations
        for loc in TOP_LOCATIONS:
            col_name = f"loc_{loc.lower().replace(' ', '_')}"
            X[col_name] = df['location_clean'].apply(lambda x: 1 if loc.lower() in str(x).lower() else 0)

        # One-hot top restaurant types
        for rt in TOP_REST_TYPES:
            col_name = f"type_{rt.lower().replace(' ', '_')}"
            X[col_name] = df['rest_type_clean'].apply(lambda x: 1 if rt.lower() in str(x).lower() else 0)

        return X

    def fit_and_evaluate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Splits labeled data into Train (80%) and Test (20%) stratified sets.
        Fits feature scaler on Train only, fits Decision Tree and Logistic Regression,
        and computes comprehensive evaluation metrics.
        """
        # Filter strictly to labeled records (restaurants with verified ratings)
        labeled_mask = df['rating_category'].notna()
        df_labeled = df[labeled_mask].copy().reset_index(drop=True)

        print(f"[*] Preparing Decision Tree dataset: {len(df_labeled)} labeled samples.")
        print(f"[*] Class distribution: {dict(df_labeled['rating_category'].value_counts())}")

        X_raw = self._build_feature_dataframe(df_labeled)
        y = np.array(df_labeled['rating_category'].values, dtype=str)
        self.feature_names = list(X_raw.columns)
        X_arr = X_raw.values.astype(np.float32)

        # Train/Test Split (80/20 Stratified) using pure NumPy arrays
        X_train_arr, X_test_arr, y_train, y_test = train_test_split(
            X_arr, y, test_size=0.20, random_state=self.random_state, stratify=y
        )

        # Prevent Data Leakage: Fit Scaler strictly on X_train (Column 0: cost_num)
        cost_train = X_train_arr[:, [0]]
        cost_test = X_test_arr[:, [0]]

        self.scaler.fit(cost_train)
        X_train_arr[:, 0] = self.scaler.transform(cost_train).flatten()
        X_test_arr[:, 0] = self.scaler.transform(cost_test).flatten()

        # 1. Train Decision Tree
        print(f"[*] Training Decision Tree Classifier (max_depth={self.max_depth})...")
        self.dt_model.fit(X_train_arr, y_train)
        dt_preds = self.dt_model.predict(X_test_arr)

        # 2. Train Comparison Model (Logistic Regression from Syllabus)
        print("[*] Training Logistic Regression Comparison Model...")
        self.lr_model.fit(X_train_arr, y_train)
        lr_preds = self.lr_model.predict(X_test_arr)

        # Compute Metrics for Decision Tree
        dt_acc = float(accuracy_score(y_test, dt_preds))
        dt_prec = float(precision_score(y_test, dt_preds, average='weighted', zero_division=0))
        dt_rec = float(recall_score(y_test, dt_preds, average='weighted', zero_division=0))
        dt_f1 = float(f1_score(y_test, dt_preds, average='weighted', zero_division=0))
        dt_cm = confusion_matrix(y_test, dt_preds, labels=self.classes_).tolist()

        # Compute Metrics for Logistic Regression
        lr_acc = float(accuracy_score(y_test, lr_preds))
        lr_prec = float(precision_score(y_test, lr_preds, average='weighted', zero_division=0))
        lr_rec = float(recall_score(y_test, lr_preds, average='weighted', zero_division=0))
        lr_f1 = float(f1_score(y_test, lr_preds, average='weighted', zero_division=0))
        lr_cm = confusion_matrix(y_test, lr_preds, labels=self.classes_).tolist()

        # Extract Top Feature Importances from Decision Tree
        importances = self.dt_model.feature_importances_
        sorted_feat_indices = np.argsort(importances)[::-1]
        top_features = []
        for idx in sorted_feat_indices[:8]:
            if importances[idx] > 0.001:
                top_features.append({
                    'feature': self.feature_names[idx],
                    'importance': round(float(importances[idx]), 4)
                })

        self.evaluation_results = {
            'train_samples': int(len(y_train)),
            'test_samples': int(len(y_test)),
            'classes': self.classes_,
            'decision_tree': {
                'algorithm': 'Decision Tree Classifier',
                'max_depth': self.max_depth,
                'accuracy': round(dt_acc, 4),
                'precision_weighted': round(dt_prec, 4),
                'recall_weighted': round(dt_rec, 4),
                'f1_weighted': round(dt_f1, 4),
                'confusion_matrix': dt_cm,
                'top_features': top_features
            },
            'logistic_regression': {
                'algorithm': 'Logistic Regression (Syllabus Comparison)',
                'accuracy': round(lr_acc, 4),
                'precision_weighted': round(lr_prec, 4),
                'recall_weighted': round(lr_rec, 4),
                'f1_weighted': round(lr_f1, 4),
                'confusion_matrix': lr_cm
            }
        }

        print(f"[+] Evaluation Complete:")
        print(f"    Decision Tree Accuracy: {dt_acc * 100:.2f}% | F1: {dt_f1:.4f}")
        print(f"    Logistic Regression Accuracy: {lr_acc * 100:.2f}% | F1: {lr_f1:.4f}")

        return self.evaluation_results

    def predict_category(self, restaurant_row: pd.Series) -> str:
        """
        Predicts rating category for a restaurant from its attributes.
        """
        feat_df = self._build_feature_dataframe(pd.DataFrame([restaurant_row]))
        feat_arr = feat_df.values.astype(np.float32)
        feat_arr[:, 0] = self.scaler.transform(feat_arr[:, [0]]).flatten()
        pred = self.dt_model.predict(feat_arr)[0]
        return str(pred)
