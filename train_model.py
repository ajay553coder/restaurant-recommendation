import os
import sys
import time

# Ensure UTF-8 output encoding for Windows command line
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure project root directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_raw_dataset
from src.preprocessing import preprocess_dataset, FeaturePipeline
from src.recommender import RestaurantRecommender
from src.clustering import RestaurantClusterer
from src.decision_tree import DecisionTreeRatingClassifier
from src.utils import save_artifacts, export_reports


def run_training_pipeline():
    print("=" * 70)
    print("🍽️  DineWise ML — Model Training & Evaluation Pipeline")
    print("    Personalized Restaurant Discovery Using Machine Learning")
    print("=" * 70)
    start_time = time.time()

    # Step 1: Load Raw Dataset
    print("\n[Step 1/6] Loading Raw Dataset...")
    df_raw, col_map = load_raw_dataset(data_dir='data')

    # Step 2: Data Preprocessing
    print("\n[Step 2/6] Executing Data Preprocessing Pipeline...")
    df_clean, prep_report = preprocess_dataset(df_raw)

    # Step 3: Feature Engineering & Scaling
    print("\n[Step 3/6] Fitting Structured Feature Pipeline (No NLP, No TF-IDF)...")
    feature_pipeline = FeaturePipeline()
    feature_matrix = feature_pipeline.fit_transform(df_clean)
    print(f"[+] Feature matrix shape: {feature_matrix.shape} ({len(feature_pipeline.feature_names)} features)")

    # Step 4: Algorithm A — KNN Recommendation Engine
    print("\n[Step 4/6] Training Algorithm A: KNN Recommender...")
    knn_recommender = RestaurantRecommender(n_neighbors=50)
    knn_recommender.fit(df_clean, feature_pipeline, feature_matrix)

    # Step 5: Algorithm B — K-Means Clustering
    print("\n[Step 5/6] Training Algorithm B: K-Means Clustering...")
    kmeans_clusterer = RestaurantClusterer(n_clusters=4, random_state=42)
    kmeans_clusterer.fit(df_clean, feature_matrix)
    clustering_summary = kmeans_clusterer.get_summary_dict()

    # Step 6: Algorithm C — Decision Tree Rating Classification (plus Logistic Regression Comparison)
    print("\n[Step 6/6] Training Algorithm C: Decision Tree Rating Classifier...")
    dt_classifier = DecisionTreeRatingClassifier(max_depth=5, min_samples_leaf=20, random_state=42)
    eval_results = dt_classifier.fit_and_evaluate(df_clean)
    eval_results['clustering'] = clustering_summary

    # Metadata Compilation
    metadata = {
        'project_name': 'DineWise ML',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_unique_restaurants': int(len(df_clean)),
        'total_features': int(feature_matrix.shape[1]),
        'feature_names': feature_pipeline.feature_names,
        'algorithms_trained': [
            'Algorithm A: k-Nearest Neighbors (NearestNeighbors)',
            'Algorithm B: K-Means Clustering',
            'Algorithm C: Decision Tree Classifier',
            'Comparison: Logistic Regression'
        ],
        'clustering_summary': clustering_summary,
        'evaluation_results': eval_results
    }

    # Save artifacts & export reports
    print("\n[*] Saving model artifacts and evaluation reports...")
    save_artifacts(
        df_cleaned=df_clean,
        feature_pipeline=feature_pipeline,
        knn_model=knn_recommender,
        kmeans_model=kmeans_clusterer,
        dt_model=dt_classifier,
        metadata=metadata,
        model_dir='model',
        data_dir='data'
    )

    export_reports(prep_report, eval_results, reports_dir='reports')

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"🎉 Pipeline Completed Successfully in {elapsed:.2f} seconds!")
    print(f"    - Cleaned Dataset:  data/cleaned_restaurants.csv")
    print(f"    - Model Artifacts:  model/")
    print(f"    - Academic Reports: reports/")
    print(f"    - Launch App:       streamlit run app.py")
    print("=" * 70)


if __name__ == '__main__':
    run_training_pipeline()
