from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import pairwise_distances


class RestaurantClusterer:
    """
    Algorithm B: K-Means Clustering.
    Partitions restaurants into coherent market segments based on structured attributes
    (cost, rating, online order, table booking, and cuisine profiles).
    """
    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.pca = PCA(n_components=2, random_state=random_state)
        self.cluster_labels = None
        self.cluster_summaries = {}
        self.representative_restaurants = {}
        self.silhouette = 0.0

    def fit(self, df: pd.DataFrame, feature_matrix: np.ndarray) -> 'RestaurantClusterer':
        """
        Fits KMeans clustering on structured feature matrix and computes segment statistics.
        """
        print(f"[*] Training K-Means clustering (k={self.n_clusters}, random_state={self.random_state})...")
        self.cluster_labels = self.kmeans.fit_predict(feature_matrix)

        # Compute Silhouette Score on sample for efficiency & evaluation
        sample_size = min(3000, len(feature_matrix))
        np.random.seed(self.random_state)
        sample_indices = np.random.choice(len(feature_matrix), size=sample_size, replace=False)
        self.silhouette = float(round(silhouette_score(feature_matrix[sample_indices], self.cluster_labels[sample_indices]), 4))

        # Project features to 2D for plotting (clearly distinguishing projected axes from raw features)
        pca_coords = self.pca.fit_transform(feature_matrix)

        df['cluster_label'] = self.cluster_labels
        df['pca_dim_1'] = np.round(pca_coords[:, 0], 3)
        df['pca_dim_2'] = np.round(pca_coords[:, 1], 3)

        # Build feature summaries and find representative restaurants for each cluster
        centroids = self.kmeans.cluster_centers_

        for c_id in range(self.n_clusters):
            c_mask = (self.cluster_labels == c_id)
            c_df = df[c_mask]
            c_features = feature_matrix[c_mask]

            # Distances to centroid to find representative (medoid-like) records
            centroid = centroids[c_id].reshape(1, -1)
            dist_to_centroid = pairwise_distances(c_features, centroid).flatten()
            closest_indices = np.argsort(dist_to_centroid)[:3]
            rep_records = c_df.iloc[closest_indices][['name', 'location_clean', 'cuisines_clean', 'cost_num', 'rating_num']].to_dict(orient='records')

            # Clean cuisine distribution
            cuisine_counts = {}
            for row_cuisines in c_df['cuisines_clean']:
                for c in str(row_cuisines).split(','):
                    c_clean = c.strip().title()
                    if c_clean and c_clean != 'Multi-Cuisine':
                        cuisine_counts[c_clean] = cuisine_counts.get(c_clean, 0) + 1

            top_cuisines = [k for k, _ in sorted(cuisine_counts.items(), key=lambda x: x[1], reverse=True)[:3]]

            # Segment metrics
            avg_cost = round(float(c_df['cost_num'].mean()), 1) if c_df['cost_num'].notna().any() else 0.0
            avg_rating = round(float(c_df['rating_num'].mean()), 2) if c_df['rating_num'].notna().any() else 0.0

            self.cluster_summaries[c_id] = {
                'cluster_id': c_id,
                'size': int(len(c_df)),
                'percentage': round(float(len(c_df) / len(df) * 100), 1),
                'avg_cost': avg_cost,
                'avg_rating': avg_rating,
                'top_cuisines': top_cuisines,
                'top_locations': list(c_df['location_clean'].value_counts().head(3).index)
            }
            self.representative_restaurants[c_id] = rep_records

        print(f"[+] K-Means clustering completed. Silhouette score: {self.silhouette:.4f}")
        return self

    def get_summary_dict(self) -> Dict[str, Any]:
        """Returns serializable clustering summary dictionary."""
        return {
            'n_clusters': self.n_clusters,
            'silhouette_score': self.silhouette,
            'cluster_summaries': self.cluster_summaries,
            'representative_restaurants': self.representative_restaurants
        }
