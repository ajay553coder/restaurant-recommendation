from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from src.preprocessing import FeaturePipeline


class RestaurantRecommender:
    """
    Algorithm A: K-Nearest Neighbors (KNN) Recommendation Engine.
    Recommends restaurants by projecting structured attributes (cuisine, budget,
    rating, location, rest_type) into a multi-dimensional metric space and finding
    nearest neighbors using Euclidean distance.
    """
    def __init__(self, n_neighbors: int = 50):
        self.n_neighbors = n_neighbors
        self.knn_model = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean', algorithm='auto')
        self.feature_matrix = None
        self.df_restaurants = None
        self.feature_pipeline = None

    def fit(self, df: pd.DataFrame, feature_pipeline: FeaturePipeline, feature_matrix: np.ndarray):
        """
        Fits NearestNeighbors model on structured restaurant feature matrix.
        """
        self.df_restaurants = df.reset_index(drop=True)
        self.feature_pipeline = feature_pipeline
        self.feature_matrix = feature_matrix
        self.knn_model.fit(self.feature_matrix)
        print(f"[+] Fitted KNN Recommender with {len(df)} restaurants across {feature_matrix.shape[1]} structured features.")
        return self

    def generate_explanation(self, restaurant: pd.Series, cuisine: Optional[str], budget: Optional[float], min_rating: Optional[float], distance: float) -> str:
        """
        Builds a concise, transparent explanation for why a restaurant is recommended.
        """
        reasons = []

        # Cuisine
        rest_cuisine = str(restaurant.get('cuisines_clean', restaurant.get('cuisines', '')))
        if cuisine and cuisine != "All" and cuisine.lower() in rest_cuisine.lower():
            reasons.append(f"Offers your preferred {cuisine} cuisine")

        # Budget
        rest_cost = restaurant.get('cost_num')
        if budget and pd.notna(rest_cost) and rest_cost > 0 and rest_cost <= budget:
            reasons.append(f"Comfortably within your ₹{int(budget)} budget (₹{int(rest_cost)} for two)")

        # Rating
        rest_rating = restaurant.get('rating_num')
        if pd.notna(rest_rating) and rest_rating >= 4.0:
            reasons.append(f"High customer rating of ⭐{rest_rating:.1f}/5.0")
        elif min_rating and pd.notna(rest_rating) and rest_rating >= min_rating:
            reasons.append(f"Meets your minimum rating requirement (⭐{rest_rating:.1f})")

        # Location
        rest_loc = str(restaurant.get('location_clean', restaurant.get('location', '')))
        if rest_loc and rest_loc != 'Unknown':
            reasons.append(f"Located in {rest_loc}")

        if not reasons:
            reasons.append("Close mathematical match to your preferred dining profile")

        return ". ".join(reasons) + "."

    def recommend(
        self,
        cuisine: Optional[str] = None,
        location: Optional[str] = None,
        budget: Optional[float] = None,
        min_rating: Optional[float] = None,
        top_n: int = 5
    ) -> pd.DataFrame:
        """
        Finds top N nearest-neighbor restaurants matching user profile and hard constraints.
        
        Hard Constraints:
        - Budget: restaurant cost must be <= budget
        - Min Rating: restaurant rating must be >= min_rating
        - Location: restaurant location must match if explicitly specified
        """
        if self.df_restaurants is None or self.feature_matrix is None:
            raise RuntimeError("Recommender must be fitted before calling recommend().")

        # Apply Hard Constraints mask
        candidates_mask = pd.Series(True, index=self.df_restaurants.index)

        # Budget constraint
        if budget is not None and budget > 0:
            candidates_mask &= (self.df_restaurants['cost_num_imputed'] <= budget)

        # Rating constraint
        if min_rating is not None and min_rating > 0:
            # Drop unrated or below min_rating
            candidates_mask &= (self.df_restaurants['rating_num'] >= min_rating)

        # Location constraint
        if location and location != "All" and location.strip():
            loc_lower = location.strip().lower()
            candidates_mask &= self.df_restaurants['location_clean'].str.lower().str.contains(loc_lower, na=False)

        # Cuisine constraint
        if cuisine and cuisine != "All" and cuisine.strip():
            c_lower = cuisine.strip().lower()
            candidates_mask &= self.df_restaurants['cuisines_clean'].str.lower().str.contains(c_lower, na=False)

        eligible_indices = np.where(candidates_mask.values)[0]

        empty_cols = ['name', 'location_clean', 'cuisines_clean', 'cost_num', 'rating_num', 'neighborhood_distance', 'recommendation_reason']
        if len(eligible_indices) == 0:
            return pd.DataFrame(columns=empty_cols)

        # Build query feature vector from user preference parameters
        query_vec = self.feature_pipeline.transform_query(
            cuisine=cuisine,
            budget=budget,
            min_rating=min_rating,
            location=location
        )

        # Compute Euclidean distance to all candidate restaurants meeting hard constraints
        candidate_vectors = self.feature_matrix[eligible_indices]
        distances = np.linalg.norm(candidate_vectors - query_vec, axis=1)

        # Rank candidates by nearest neighbor distance (ascending)
        top_k = min(top_n, len(distances))
        best_order = np.argsort(distances)[:top_k]

        matched_indices = eligible_indices[best_order]
        matched_distances = distances[best_order]

        # Extract recommended restaurants
        recommendations = self.df_restaurants.iloc[matched_indices].copy().reset_index(drop=True)
        recommendations['neighborhood_distance'] = [round(float(d), 3) for d in matched_distances]

        # Generate human-readable explanations
        explanations = []
        for i, row in recommendations.iterrows():
            exp = self.generate_explanation(
                restaurant=row,
                cuisine=cuisine,
                budget=budget,
                min_rating=min_rating,
                distance=matched_distances[i]
            )
            explanations.append(exp)

        recommendations['recommendation_reason'] = explanations
        return recommendations
