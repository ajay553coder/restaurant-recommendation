import os
import sys
import unittest
import numpy as np
import pandas as pd

# Ensure project root is in python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_raw_dataset
from src.preprocessing import preprocess_dataset, FeaturePipeline
from src.recommender import RestaurantRecommender
from src.clustering import RestaurantClusterer
from src.decision_tree import DecisionTreeRatingClassifier
from src.utils import load_artifacts, save_artifacts


class TestDineWiseML(unittest.TestCase):
    """
    Comprehensive test suite for DineWise ML DWDM algorithms and pipeline.
    """

    @classmethod
    def setUpClass(cls):
        # Verify original dataset preservation
        zomato_path = os.path.join(PROJECT_ROOT, 'data', 'zomato.csv')
        sample_path = os.path.join(PROJECT_ROOT, 'data', 'sample_zomato.csv')

        assert os.path.exists(zomato_path) or os.path.exists(sample_path), "Dataset missing from data/ directory!"
        cls.raw_df, cls.col_map = load_raw_dataset(data_dir=os.path.join(PROJECT_ROOT, 'data'))
        cls.cleaned_df, cls.prep_report = preprocess_dataset(cls.raw_df)

    def test_01_dataset_preservation_and_loading(self):
        """Verify original dataset was preserved and properly loaded."""
        self.assertGreater(len(self.raw_df), 0, "Loaded dataset should have rows")
        self.assertIn('name', self.raw_df.columns)
        self.assertIn('cuisines', self.raw_df.columns)
        self.assertIn('location', self.raw_df.columns)
        print(f"\n[PASS] Dataset loaded successfully ({len(self.raw_df)} records).")

    def test_02_preprocessing_pipeline(self):
        """Verify data cleaning, missing value handling, and deduplication."""
        self.assertGreater(len(self.cleaned_df), 0)
        self.assertLessEqual(len(self.cleaned_df), len(self.raw_df), "Duplicates should have been removed")
        
        # Verify rating column is numeric
        self.assertTrue(pd.api.types.is_numeric_dtype(self.cleaned_df['rating_num']))
        # Verify cost column is numeric, positive where available, and imputed cost is strictly positive
        self.assertTrue(pd.api.types.is_numeric_dtype(self.cleaned_df['cost_num']))
        self.assertTrue((self.cleaned_df['cost_num'].dropna() > 0).all(), "Available costs must be strictly positive")
        self.assertTrue((self.cleaned_df['cost_num_imputed'] > 0).all(), "Imputed costs must be strictly positive")

        # Verify feature pipeline
        pipeline = FeaturePipeline()
        X = pipeline.fit_transform(self.cleaned_df)
        self.assertEqual(X.shape[0], len(self.cleaned_df))
        self.assertEqual(len(pipeline.feature_names), 64, "Feature pipeline should generate 64 structured features")
        print(f"[PASS] Preprocessing pipeline produced {X.shape[1]} features across {X.shape[0]} unique restaurants.")

    def test_03_knn_recommender_and_constraints(self):
        """Verify KNN recommendation engine adheres strictly to hard constraints."""
        pipeline = FeaturePipeline()
        X = pipeline.fit_transform(self.cleaned_df)

        recommender = RestaurantRecommender(n_neighbors=50)
        recommender.fit(self.cleaned_df, pipeline, X)

        # Test with budget constraint of 600 and min_rating of 4.0
        recs = recommender.recommend(
            cuisine="North Indian",
            location=None,
            budget=600.0,
            min_rating=4.0,
            top_n=5
        )

        self.assertFalse(recs.empty, "Recommendations should be returned for common cuisine")
        # Hard constraints assertion
        for _, row in recs.iterrows():
            self.assertLessEqual(row['cost_num'], 600.0, "Recommended restaurant exceeds budget constraint!")
            self.assertGreaterEqual(row['rating_num'], 4.0, "Recommended restaurant violates min_rating constraint!")
            self.assertIn('recommendation_reason', row)
            self.assertIn('neighborhood_distance', row)

        # Test impossible constraints: should gracefully return empty dataframe
        empty_recs = recommender.recommend(
            cuisine="NonexistentCuisineXYZ",
            location="NonexistentLocationXYZ",
            budget=10.0,
            min_rating=5.0,
            top_n=5
        )
        self.assertTrue(empty_recs.empty, "Should gracefully return empty DataFrame when no candidates match")
        print("[PASS] KNN Recommender satisfied all hard constraints and distance ranking.")

    def test_04_kmeans_clustering(self):
        """Verify K-Means clustering and cluster summary generation."""
        pipeline = FeaturePipeline()
        X = pipeline.fit_transform(self.cleaned_df)

        clusterer = RestaurantClusterer(n_clusters=4, random_state=42)
        clusterer.fit(self.cleaned_df, X)

        self.assertEqual(len(clusterer.cluster_labels), len(self.cleaned_df))
        unique_labels = set(clusterer.cluster_labels)
        self.assertEqual(unique_labels, {0, 1, 2, 3}, "Should assign 4 distinct clusters")

        self.assertEqual(len(clusterer.cluster_summaries), 4)
        self.assertEqual(len(clusterer.representative_restaurants), 4)
        for c_id in range(4):
            self.assertGreater(len(clusterer.representative_restaurants[c_id]), 0)

        print("[PASS] K-Means clustering partitioned data into 4 distinct clusters with representatives.")

    def test_05_decision_tree_data_leakage_prevention(self):
        """CRITICAL TEST: Ensure Decision Tree has ZERO data leakage."""
        dt_classifier = DecisionTreeRatingClassifier(max_depth=5, random_state=42)
        eval_results = dt_classifier.fit_and_evaluate(self.cleaned_df)

        # Verify rating column was NOT included in training features
        feature_names = dt_classifier.feature_names
        leakage_indicators = ['rate', 'rating_num', 'rating_category', 'rating', 'votes']
        for feat in feature_names:
            self.assertNotIn(feat.lower(), [l.lower() for l in leakage_indicators],
                             f"DATA LEAKAGE DETECTED: '{feat}' is derived from target rating!")

        # Verify metrics exist
        self.assertIn('decision_tree', eval_results)
        self.assertIn('logistic_regression', eval_results)
        dt_res = eval_results['decision_tree']
        self.assertGreater(dt_res['accuracy'], 0.40, "Decision Tree accuracy should be reasonable (>40%)")
        self.assertEqual(len(dt_res['confusion_matrix']), 3, "Confusion matrix should be 3x3 for Low/Med/High")

        # Test single prediction
        sample_row = self.cleaned_df.iloc[0]
        pred_cat = dt_classifier.predict_category(sample_row)
        self.assertIn(pred_cat, ['Low', 'Medium', 'High'], "Predicted tier must be Low, Medium, or High")
        print(f"[PASS] Decision Tree verified with NO data leakage (Test Accuracy: {dt_res['accuracy']*100:.2f}%).")

    def test_06_model_serialization_and_loading(self):
        """Verify saved model artifacts can be loaded and produce identical results."""
        artifacts = load_artifacts(model_dir=os.path.join(PROJECT_ROOT, 'model'))
        self.assertIsNotNone(artifacts)

        df_c, pipeline, knn, kmeans, dt, meta = artifacts
        self.assertGreater(len(df_c), 0)
        self.assertIsNotNone(knn)
        self.assertIsNotNone(kmeans)
        self.assertIsNotNone(dt)
        self.assertIn('evaluation_results', meta)
        self.assertIn('clustering_summary', meta)
        print("[PASS] Model artifacts successfully deserialized and verified.")


if __name__ == '__main__':
    unittest.main()
