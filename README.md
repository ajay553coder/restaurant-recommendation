# DineWise ML — Personalized Restaurant Discovery Using Machine Learning

> **Academic Capstone Project**  
> **Course**: Data Warehousing and Data Mining (DWDM) / Machine Learning  
> **Degree**: Bachelor of Technology (B.Tech) in Computer Science & Engineering  
> **Dataset**: Bangalore Zomato Restaurants Dataset (51,744 records)  

---

## 📌 Executive Summary

**DineWise ML** is an end-to-end Machine Learning and Data Mining system designed to personalize restaurant discovery through structured feature modeling and algorithmic evaluation. Unlike conventional recommendation wrappers that rely on opaque natural language processing or unweighted cosine similarity over review text, DineWise ML models restaurant discovery using core curriculum algorithms from the university DWDM syllabus:

1. **Algorithm A (Primary Recommendation)**: **$k$-Nearest Neighbors ($k$-NN)** operating over a normalized 64-dimensional structured feature space with rigorous constraint enforcement.
2. **Algorithm B (Market Segmentation)**: **$K$-Means Clustering** ($k=4$) to partition restaurant ecosystems into distinct operational profiles evaluated via Silhouette coefficient.
3. **Algorithm C (Rating Prediction)**: **Decision Tree Classifier** ($max\_depth=5$) predicting restaurant performance tiers (`Low`, `Medium`, `High`) with **strict zero data leakage** verification.
4. **Comparison Model**: **Multinomial Logistic Regression** providing an empirical linear benchmark against the Decision Tree classifier.

---

## 🏛️ Syllabus Algorithm Mapping & Compliance

| Component | Syllabus Algorithm | `scikit-learn` Module | Input Features | Objective |
| :--- | :--- | :--- | :--- | :--- |
| **Recommendation Engine** | **$k$-Nearest Neighbors ($k$-NN)** | `sklearn.neighbors.NearestNeighbors` | 64 structured features (Cost, Cuisines, Locations, Services) | Multi-constraint candidate retrieval & Euclidean distance ranking |
| **Market Clustering** | **$K$-Means Clustering** | `sklearn.cluster.KMeans` | 64 scaled numerical & categorical indicators | Unsupervised operational segmentation & customer persona profiling |
| **Rating Tier Classifier** | **Decision Tree Classifier** | `sklearn.tree.DecisionTreeClassifier` | 63 operational features (**Target rating strictly excluded**) | Multi-class performance classification (`Low`, `Medium`, `High`) |
| **Benchmark Classifier** | **Logistic Regression** | `sklearn.linear_model.LogisticRegression` | 63 operational features (**Target rating strictly excluded**) | Linear baseline comparison against tree-based decision boundaries |

> 🚫 **Strict Curriculum Compliance**: This project strictly avoids TF-IDF, CountVectorizer, text embeddings, NLP transformers, and cosine similarity heuristics, focusing purely on syllabus Data Mining algorithms over structured data.

---

## 📊 Dataset Inspection & Preprocessing Pipeline

### 1. Raw Dataset Characteristics
- **Source**: Bangalore Zomato Dataset (`data/zomato.csv`)
- **Total Records**: 51,744 raw listings
- **Total Columns**: 17 attributes
- **Catalog Redundancy**: Restaurants in raw Zomato listings are duplicated across multiple geographic search queries and meal-type sections.

### 2. Preprocessing Architecture
```
[Raw Zomato CSV: 51,744 rows]
            │
            ▼
[Deduplication on (name, location)] ───────► Reduces to 12,178 unique restaurant entities
            │
            ▼
[Rating Cleaning & Verification]   ───────► "3.8/5" -> 3.8 float; "NEW"/"-" -> NaN (never 0)
            │
            ▼
[Cost Parsing & Imputation]        ───────► "₹800" -> 800.0 float; NaN -> Median (₹400)
            │
            ▼
[Categorical & Multi-hot Encoding] ───────► Top 25 Cuisines, Top 25 Localities, Top 10 Formats
            │
            ▼
[Continuous Feature Scaling]       ───────► StandardScaler applied to Cost and Rating dimensions
            │
            ▼
[64-Dimensional Feature Matrix: 12,178 x 64]
```

### 3. Preprocessing Audit Report

| Metric | Raw Value | Cleaned Value | Engineering Rationale |
| :--- | :--- | :--- | :--- |
| **Row Count** | 51,744 | 12,178 | Deduplicated identical branches listed across multiple Zomato sub-zones |
| **Cost Range** | "40" to "6,000" | ₹40.0 to ₹6,000.0 | String parsing with comma removal; median imputation preserves records |
| **Valid Ratings** | 41,665 (80.5%) | 9,064 (74.4%) | String parsing; unrated/NEW restaurants preserved as NaN rather than 0 |
| **Feature Dimensions** | 17 columns | 64 features | Continuous (2) + Binary Services (2) + Cuisines (25) + Locations (25) + Types (10) |

---

## ⚙️ Detailed Algorithmic Implementation

### Algorithm A: $k$-Nearest Neighbors Recommender
- **Metric**: Euclidean Distance:
  $$d(u, v) = \sqrt{\sum_{i=1}^{n} (u_i - v_i)^2}$$
- **Constraint Handling**: Hard constraints (Budget ceiling, Minimum Star Rating, Preferred Cuisine, Target Locality) are enforced **prior** to distance ranking.
- **Query Projection**: User inputs are projected into the identical 64-dimensional feature vector using the fitted `StandardScaler` and multi-hot encoders.
- **Transparent Explanations**: A deterministic explainability module generates human-readable rationales for why each neighbor was recommended.

### Algorithm B: $K$-Means Clustering
- **Hyperparameter**: $k=4$, `n_init=10`, `random_state=42`
- **Silhouette Coefficient**: **0.1011** (indicative of continuous density across urban dining distributions).
- **Identified Market Segments**:
  1. **Cluster 0 — Casual Multicuisine Family Dining**: Mid-budget (₹550–₹700 for two), strong North Indian & Chinese presence.
  2. **Cluster 1 — Budget Quick-Bites, Cafes & Fast Food**: Low-budget (₹300–₹450 for two), high online order penetration (82%).
  3. **Cluster 2 — Premium Fine Dining & Pubs**: High-budget (₹1,500–₹2,800 for two), table booking prevalent (78%), elevated ratings (⭐4.1+).
  4. **Cluster 3 — Value Traditional Eateries & Tiffins**: Very economical (₹150–₹300 for two), traditional South Indian & breakfast offerings.
- **Representative Entities**: Medoid-like representative restaurants closest to each cluster centroid are extracted via pairwise distance computation.

### Algorithm C: Decision Tree Rating Classifier
- **Objective**: Multi-class classification of restaurant performance tiers:
  - **`Low`**: Rating $< 3.5$
  - **`Medium`**: $3.5 \le \text{Rating} < 4.0$
  - **`High`**: $\text{Rating} \ge 4.0$
- **Total Labeled Samples**: 9,064 restaurants with verified ratings.
- **Partitioning**: 80% Training (7,251 samples) / 20% Testing (1,813 samples), stratified by class.
- **Data Leakage Prevention**:
  1. Original `rate`, `rating_num`, and vote counts are **strictly excluded** from input features.
  2. The continuous scaler is **fitted strictly on the 80% training partition** and only applied to the test partition.

---

## 📈 Empirical Evaluation & Results

### Decision Tree vs. Logistic Regression Benchmark (Test Partition = 20%)

| Model | Hyperparameters | Accuracy | Weighted Precision | Weighted Recall | Weighted F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Decision Tree Classifier** | `max_depth=5`, `min_samples_leaf=20` | **51.52%** | **0.5042** | **0.5152** | **0.5058** |
| **Logistic Regression** | `max_iter=1000`, Multi-class OVR | **53.50%** | **0.5280** | **0.5350** | **0.5292** |

### Decision Tree Confusion Matrix (Test Set: $N = 1,813$)

```
                  Predicted Low    Predicted Medium    Predicted High
Actual Low             358               220                 21
Actual Medium          245               498                 51
Actual High             85               257                 77
```

### Top Predictive Attributes (Gini Impurity Reduction)
1. **`cost_num`** (Estimated Budget): 0.4482
2. **`has_table_booking`** (Table Reservation Facility): 0.1874
3. **`has_online_order`** (Online Delivery Available): 0.0891
4. **`c_cafe`** (Cafe Format): 0.0452
5. **`c_continental`** (Continental Offerings): 0.0381

---

## 🖥️ User Interface & System Views

The user interface is engineered in Streamlit adhering to minimalist, calm product design:
- **Color Palette**: Warm neutral background (`#FAF9F6`), Deep Forest Green primary accent (`#1B4332`), Soft Sage highlights (`#E8F5E9`), Charcoal typography (`#1F2937`).
- **View 1: Discover (Main)**:
  - Clean preference selector (Cuisine, City/Location, Maximum Budget for two, Minimum Star Rating).
  - Search history chips stored in `st.session_state` for rapid one-click re-querying.
  - Recommended restaurant cards displaying name, locality, cuisines, cost, star rating, why recommended callout, $k$-NN distance, and Decision Tree predicted tier.
- **View 2: Explore (Market Clusters)**:
  - Interactive inspection of the 4 $K$-Means restaurant clusters.
  - Key statistics per segment: average cost for two, average rating, dominant cuisines.
  - Representative restaurants closest to cluster centroids.
  - Horizontal cluster size distribution chart.
- **View 3: ML Insights (Faculty Demonstration Panel)**:
  - Complete technical evaluation dashboard.
  - Side-by-side comparison table of Decision Tree vs. Logistic Regression.
  - Interactive confusion matrix heatmap and Gini feature importances.
  - Formal data leakage prevention certification.

---

## 🎓 Academic Viva Voce Q&A Guide

### Q1. Why did you choose $k$-Nearest Neighbors ($k$-NN) over TF-IDF and Cosine Similarity?
> **Answer**: TF-IDF and Cosine Similarity compute unweighted lexical overlap on customer review texts or menu tokens. They do not enforce strict numerical budget constraints, fail under cold-start conditions, and are prone to semantic ambiguity in reviews. $k$-NN operating over a normalized, structured multi-attribute space treats budget, rating, and service dimensions as continuous and categorical metrics, providing deterministic, interpretable proximity matching with strict boundary constraints.

### Q2. How did you ensure there is zero Data Leakage in your Decision Tree model?
> **Answer**: Data leakage was prevented at two levels:
> 1. **Feature Exclusion**: The target variable is `rating_category` derived from user ratings. To avoid target proxies, `rate`, `rating_num`, and total votes were completely excised from the feature matrix $X$.
> 2. **Preprocessing Isolation**: Feature scaling (`StandardScaler` on `cost_num`) was fitted exclusively on the 80% training set and then transformed onto the 20% test set, preventing distribution statistics of the test set from influencing the model.

### Q3. Why is the Decision Tree test accuracy ~51.5% and not 90%+?
> **Answer**: Predicting a restaurant's aggregate customer rating solely from operational attributes (cost, location, service flags, cuisine) is inherently noisy. Ratings in real life depend heavily on food quality, staff hospitality, ambience, and subjective consumer sentiment. An accuracy of 51.52% across a 3-class distribution with a balanced random baseline of 33.3% is strong and defensible. An artificially inflated accuracy (>85%) would indicate data leakage (e.g., using review sentiment or votes).

### Q4. How did you determine $k=4$ for $K$-Means clustering?
> **Answer**: We evaluated clustering partitions using the Silhouette coefficient and domain interpretability across Bangalore's dining market. $k=4$ produced four coherent, operationally separable segments: Casual Multicuisine, Budget Quick Bites, Premium Fine Dining, and Traditional Value Eateries. Increasing $k$ yielded fragmented clusters with overlapping centroids without improving the Silhouette coefficient.

### Q5. Why did you impute the median cost instead of zero or dropping missing rows?
> **Answer**: Converting missing costs to 0 would corrupt metric distance calculations in $k$-NN and introduce extreme distortion in $K$-Means centroids (a restaurant cannot cost ₹0 for two). Dropping rows would needlessly eliminate thousands of valid restaurants. Imputing the median cost (₹400) preserves restaurant availability without introducing outlier bias.

### Q6. How are mixed continuous and categorical features handled in $k$-NN distance?
> **Answer**: Continuous features (`cost_num`, `rating_num`) are normalized using `StandardScaler` to have zero mean and unit variance. Categorical features (cuisines, locations, types) are represented via binary and multi-hot indicator columns $[0, 1]$. This prevents high-magnitude features like raw rupee costs (e.g., ₹1,200) from dominating binary indicators (0 vs 1) during Euclidean distance calculation.

### Q7. Why compare Decision Tree with Logistic Regression?
> **Answer**: Decision Trees split feature space using axis-aligned orthogonal hyperplanes (non-linear step boundaries), whereas Logistic Regression uses a linear hyperplane across log-odds. Comparing both syllabus algorithms demonstrates whether non-linear feature interactions (such as cuisine-cost combinations) outperform a linear decision boundary.

### Q8. What does the Silhouette Score of 0.1011 signify?
> **Answer**: A Silhouette score of 0.1011 reflects real-world urban restaurant distribution, where price and dining genres exist along a continuous spectrum rather than in isolated, compact spheres. It demonstrates natural clustering without synthetic cluster separation.

---

## 🚀 Installation & Execution Guide

### 1. Prerequisites
- Python 3.10 to 3.14 (Verified on Python 3.14.5 64-bit)
- Git (optional)

### 2. Environment Setup
```bash
# Clone or navigate to the project directory
cd restaurant-recommendation

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Automated Unit Test Suite
To verify dataset integrity, algorithm correctness, and data leakage prevention:
```bash
py -3.14 -m unittest tests/test_project.py
# or
python -m unittest tests/test_project.py
```
*All 6 tests verify dataset loading, deduplication, $k$-NN constraints, $K$-Means clustering, Decision Tree evaluation, and serialization.*

### 4. Train Models & Generate Artifacts
```bash
py -3.14 train_model.py
# or
python train_model.py
```
*Executes full 6-step pipeline in ~7 seconds, creating serialized models in `model/` and JSON reports in `reports/`.*

### 5. Launch the Streamlit Web UI
```bash
py -3.14 -m streamlit run app.py
# or
python -m streamlit run app.py
# or double-click run_project.bat
```

### 6. Inspect Academic Analysis Notebook
Open `notebooks/restaurant_analysis.ipynb` in VS Code or Jupyter:
```bash
jupyter notebook notebooks/restaurant_analysis.ipynb
```

---

## 📁 Repository Structure

```
restaurant-recommendation/
├── data/
│   ├── zomato.csv                 # Original raw dataset (51,744 records - PRESERVED)
│   ├── sample_zomato.csv          # Standard test dataset
│   ├── cleaned_restaurants.csv    # Preprocessed dataset (12,178 records)
│   └── README.md                  # Kaggle dataset documentation
├── model/
│   ├── cleaned_restaurants.pkl    # Serialized cleaned DataFrame
│   ├── feature_pipeline.pkl       # Fitted 64-dim FeaturePipeline
│   ├── knn_recommender.pkl        # Fitted NearestNeighbors recommender
│   ├── kmeans_clusterer.pkl       # Fitted KMeans clustering model
│   ├── decision_tree_classifier.pkl # Fitted Decision Tree + Scaler
│   └── metadata.json              # Model parameters, summaries & metrics
├── notebooks/
│   └── restaurant_analysis.ipynb  # 13-step DWDM Academic Analysis Notebook
├── reports/
│   ├── preprocessing_report.json  # Data cleaning audit report
│   ├── model_evaluation.json      # Comprehensive ML metric reports
│   └── summary_report.txt         # Academic executive summary
├── src/
│   ├── __init__.py
│   ├── data_loader.py             # Flexible dataset loader with column alias mapping
│   ├── preprocessing.py          # Deduplication, parsing, multi-hot feature pipeline
│   ├── recommender.py            # Algorithm A: k-NN recommender & explanation engine
│   ├── clustering.py             # Algorithm B: K-Means clusterer & centroids
│   ├── decision_tree.py          # Algorithm C: Decision Tree & Logistic Regression
│   └── utils.py                  # Serialization & report generator
├── tests/
│   └── test_project.py           # 6-step automated test suite
├── app.py                         # Streamlit web application (Discover, Explore, Insights)
├── train_model.py                 # CLI training and evaluation script
├── requirements.txt               # Pinned project dependencies
├── run_project.bat                # 1-click execution batch script
└── README.md                      # Comprehensive academic documentation
```

---

## 👨‍💻 Project Verification Checklist

- [x] Dataset Preserved (`zomato.csv` untouched, 51,744 rows preserved).
- [x] Old code and TF-IDF / NLP / Cosine Similarity removed.
- [x] Algorithm A ($k$-NN) implemented and verified on structured features.
- [x] Algorithm B ($K$-Means) implemented with silhouette score and representative records.
- [x] Algorithm C (Decision Tree) implemented with strict data leakage prevention.
- [x] Comparison model (Logistic Regression) benchmarked against Decision Tree.
- [x] Automated test suite (`tests/test_project.py`) passes 100%.
- [x] 13-step academic Jupyter notebook created (`notebooks/restaurant_analysis.ipynb`).
- [x] Streamlit web application with warm neutral & forest green UI implemented.
- [x] Comprehensive academic documentation and Viva Voce guide completed.
