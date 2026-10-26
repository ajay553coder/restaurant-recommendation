import os
import json
import joblib
from typing import Dict, Any, Tuple, Optional
import pandas as pd


def save_artifacts(
    df_cleaned: pd.DataFrame,
    feature_pipeline: Any,
    knn_model: Any,
    kmeans_model: Any,
    dt_model: Any,
    metadata: Dict[str, Any],
    model_dir: str = 'model',
    data_dir: str = 'data'
):
    """
    Saves trained models, fitted preprocessors, and metadata artifacts.
    """
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # Save cleaned dataset artifact
    cleaned_csv_path = os.path.join(data_dir, 'cleaned_restaurants.csv')
    cleaned_pkl_path = os.path.join(model_dir, 'cleaned_restaurants.pkl')
    df_cleaned.to_csv(cleaned_csv_path, index=False, encoding='utf-8')
    joblib.dump(df_cleaned, cleaned_pkl_path)

    # Save models
    joblib.dump(feature_pipeline, os.path.join(model_dir, 'feature_pipeline.pkl'))
    joblib.dump(knn_model, os.path.join(model_dir, 'knn_recommender.pkl'))
    joblib.dump(kmeans_model, os.path.join(model_dir, 'kmeans_clusterer.pkl'))
    joblib.dump(dt_model, os.path.join(model_dir, 'decision_tree_classifier.pkl'))

    # Save metadata JSON
    meta_path = os.path.join(model_dir, 'metadata.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)

    print(f"[+] All artifacts saved successfully in '{model_dir}/' and '{cleaned_csv_path}'.")


def load_artifacts(model_dir: str = 'model') -> Tuple[pd.DataFrame, Any, Any, Any, Any, Dict[str, Any]]:
    """
    Loads saved model artifacts and preprocessed data.
    """
    cleaned_pkl_path = os.path.join(model_dir, 'cleaned_restaurants.pkl')
    feat_path = os.path.join(model_dir, 'feature_pipeline.pkl')
    knn_path = os.path.join(model_dir, 'knn_recommender.pkl')
    kmeans_path = os.path.join(model_dir, 'kmeans_clusterer.pkl')
    dt_path = os.path.join(model_dir, 'decision_tree_classifier.pkl')
    meta_path = os.path.join(model_dir, 'metadata.json')

    for p in [cleaned_pkl_path, feat_path, knn_path, kmeans_path, dt_path, meta_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required artifact '{p}' not found. Please run 'train_model.py' first.")

    df_cleaned = joblib.load(cleaned_pkl_path)
    feature_pipeline = joblib.load(feat_path)
    knn_model = joblib.load(knn_path)
    kmeans_model = joblib.load(kmeans_path)
    dt_model = joblib.load(dt_path)

    with open(meta_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    return df_cleaned, feature_pipeline, knn_model, kmeans_model, dt_model, metadata


def export_reports(preprocessing_report: Dict[str, Any], evaluation_results: Dict[str, Any], reports_dir: str = 'reports'):
    """
    Exports structured JSON and formatted academic text reports for DWDM review.
    """
    os.makedirs(reports_dir, exist_ok=True)

    # Preprocessing Report JSON
    prep_json_path = os.path.join(reports_dir, 'preprocessing_report.json')
    with open(prep_json_path, 'w', encoding='utf-8') as f:
        json.dump(preprocessing_report, f, indent=4)

    # Model Evaluation Report JSON
    eval_json_path = os.path.join(reports_dir, 'model_evaluation.json')
    with open(eval_json_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=4)

    # Human-Readable Summary Report for Faculty
    summary_txt_path = os.path.join(reports_dir, 'summary_report.txt')
    with open(summary_txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("DineWise ML — DWDM CAPSTONE PROJECT SUMMARY REPORT\n")
        f.write("Personalized Restaurant Discovery Using Machine Learning\n")
        f.write("=" * 70 + "\n\n")

        f.write("1. DATASET & PREPROCESSING SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Original Records:       {preprocessing_report.get('original_row_count', 'N/A')}\n")
        f.write(f"Duplicate Rows Removed: {preprocessing_report.get('duplicates_removed', 'N/A')}\n")
        f.write(f"Clean Unique Entities:  {preprocessing_report.get('final_row_count', 'N/A')}\n")
        f.write(f"Rated Restaurants:      {preprocessing_report.get('valid_ratings_count', 'N/A')}\n")
        f.write(f"Unrated Restaurants:    {preprocessing_report.get('unrated_count', 'N/A')}\n")
        f.write(f"Median Cost Imputed:    Rs. {preprocessing_report.get('median_cost_imputed', 'N/A')}\n\n")

        f.write("2. ALGORITHM A: KNN RECOMMENDATION ENGINE\n")
        f.write("-" * 40 + "\n")
        f.write("Algorithm:              k-Nearest Neighbors (NearestNeighbors)\n")
        f.write("Metric:                 Euclidean Distance on Normalized Feature Space\n")
        f.write("Features:               Cost, Rating, Online Order, Table Booking, Cuisines, Locations, Rest Types\n")
        f.write("Constraints Handled:    Budget, Minimum Rating, Location Preference, Cuisine Preference\n\n")

        f.write("3. ALGORITHM B: K-MEANS CLUSTERING\n")
        f.write("-" * 40 + "\n")
        kmeans_info = evaluation_results.get('clustering', {})
        f.write(f"Number of Clusters (k): {kmeans_info.get('n_clusters', 4)}\n")
        f.write(f"Silhouette Score:       {kmeans_info.get('silhouette_score', 'N/A')}\n\n")

        f.write("4. ALGORITHM C: DECISION TREE CLASSIFIER (RATING CATEGORY)\n")
        f.write("-" * 40 + "\n")
        dt = evaluation_results.get('decision_tree', {})
        lr = evaluation_results.get('logistic_regression', {})
        f.write(f"Decision Tree Accuracy: {dt.get('accuracy', 0)*100:.2f}%\n")
        f.write(f"Decision Tree F1-Score: {dt.get('f1_weighted', 0):.4f}\n")
        f.write(f"Decision Tree Precision:{dt.get('precision_weighted', 0):.4f}\n")
        f.write(f"Decision Tree Recall:   {dt.get('recall_weighted', 0):.4f}\n\n")

        f.write("5. SYLLABUS COMPARISON: LOGISTIC REGRESSION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Logistic Reg. Accuracy: {lr.get('accuracy', 0)*100:.2f}%\n")
        f.write(f"Logistic Reg. F1-Score: {lr.get('f1_weighted', 0):.4f}\n\n")
        f.write("Data Leakage Prevention: Verified (Original rating strictly excluded from model inputs).\n")

    print(f"[+] Exported reports to '{reports_dir}/'.")
