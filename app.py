import os
import sys
import base64
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Ensure project root directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import load_artifacts
from src.preprocessing import TOP_CUISINES, TOP_LOCATIONS

# -----------------------------------------------------------------------------
# Streamlit App Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DineWise ML — Personalized Restaurant Discovery",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------------------------
# Base64 Image Caching Helpers
# -----------------------------------------------------------------------------
@st.cache_data
def get_base64_image(image_path: str) -> str:
    """Reads and encodes local image asset to base64 data URI for reliable, fast rendering."""
    if not os.path.exists(image_path):
        return ""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def get_cuisine_image_path(cuisines_str: str) -> str:
    """
    Maps restaurant cuisine string to authentic royalty-free cuisine category image asset.
    Uses fallback dining image if cuisine does not match specific category.
    """
    s = str(cuisines_str).lower()
    if any(k in s for k in ['biryani', 'mughlai', 'hyderabad', 'kebab']):
        target = 'assets/cuisines/biryani.jpg'
    elif any(k in s for k in ['cafe', 'coffee', 'tea', 'beverage']):
        target = 'assets/cuisines/cafe.jpg'
    elif any(k in s for k in ['chinese', 'momo', 'asian', 'thai', 'noodle']):
        target = 'assets/cuisines/chinese.jpg'
    elif any(k in s for k in ['dessert', 'ice cream', 'bakery', 'mithai', 'sweet', 'cake']):
        target = 'assets/cuisines/desserts.jpg'
    elif any(k in s for k in ['fast food', 'burger', 'pizza', 'roll', 'street food', 'sandwich']):
        target = 'assets/cuisines/fast_food.jpg'
    elif any(k in s for k in ['south indian', 'andhra', 'kerala', 'chettinad', 'dosa', 'idli']):
        target = 'assets/cuisines/south_indian.jpg'
    elif any(k in s for k in ['north indian', 'punjabi', 'roti', 'tandoori', 'dal']):
        target = 'assets/cuisines/north_indian.jpg'
    else:
        target = 'assets/cuisines/dining.jpg'

    return target if os.path.exists(target) else 'assets/hero_dining.jpg'


# -----------------------------------------------------------------------------
# Premium Styling: Warm Cream, Forest Green, Charcoal & Subtle Neutral Borders
# -----------------------------------------------------------------------------
st.markdown("""
    <style>
        /* Warm cream background & refined typography */
        :root {
            --bg-cream: #FAF8F5;
            --primary-forest: #1B4332;
            --primary-hover: #2D6A4F;
            --accent-sage: #E8F5E9;
            --text-charcoal: #1F2937;
            --text-muted: #525C68;
            --border-neutral: #E8E5DF;
            --card-white: #FFFFFF;
        }

        /* Top Page Padding & Container Limits */
        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 3.5rem !important;
            max-width: 1200px !important;
            margin: 0 auto !important;
        }

        .stApp {
            background-color: var(--bg-cream);
            color: var(--text-charcoal);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        }

        /* Top Brand Navigation Header */
        .header-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 4px 0 16px 0;
            border-bottom: 1px solid var(--border-neutral);
            margin-bottom: 24px;
        }
        .brand-cluster {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .brand-icon-box {
            width: 42px;
            height: 42px;
            background-color: var(--primary-forest);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 2px 6px rgba(27, 67, 50, 0.15);
        }
        .brand-title {
            font-size: 1.5rem;
            font-weight: 800;
            color: var(--primary-forest);
            letter-spacing: -0.02em;
            margin: 0;
            line-height: 1.15;
        }
        .brand-tagline {
            font-size: 0.88rem;
            color: var(--text-muted);
            margin: 2px 0 0 0;
            font-weight: 500;
        }

        /* Segmented Control Overrides: No Radio Dots! */
        [data-testid="stSegmentedControl"] {
            background-color: #EDEAE3 !important;
            border-radius: 24px !important;
            padding: 4px !important;
            border: 1px solid var(--border-neutral) !important;
            gap: 2px !important;
        }
        [data-testid="stSegmentedControl"] button {
            border-radius: 20px !important;
            font-weight: 600 !important;
            font-size: 0.88rem !important;
            padding: 6px 18px !important;
            border: none !important;
            color: var(--text-muted) !important;
            background-color: transparent !important;
            transition: all 0.18s ease-in-out !important;
        }
        [data-testid="stSegmentedControl"] button[aria-pressed="true"],
        [data-testid="stSegmentedControl"] button[data-checked="true"] {
            background-color: var(--primary-forest) !important;
            color: #FFFFFF !important;
            box-shadow: 0 2px 8px rgba(27, 67, 50, 0.25) !important;
        }

        /* Hero Section */
        .hero-banner {
            background-color: var(--card-white);
            border: 1px solid var(--border-neutral);
            border-radius: 16px;
            padding: 24px 28px;
            margin-bottom: 24px;
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.02);
        }
        .hero-badge {
            display: inline-block;
            background-color: var(--accent-sage);
            color: var(--primary-forest);
            border: 1px solid #C8E6C9;
            padding: 3px 10px;
            border-radius: 16px;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }
        .hero-heading {
            font-size: 1.85rem;
            font-weight: 800;
            color: var(--primary-forest);
            letter-spacing: -0.025em;
            line-height: 1.25;
            margin: 0 0 8px 0;
        }
        .hero-subtext {
            font-size: 1.02rem;
            color: var(--text-muted);
            line-height: 1.5;
            margin: 0 0 16px 0;
            max-width: 580px;
        }
        .hero-pills {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }
        .hero-pill-item {
            background-color: #F4F1EA;
            color: var(--text-charcoal);
            padding: 4px 11px;
            border-radius: 6px;
            font-size: 0.82rem;
            font-weight: 600;
        }
        .hero-img {
            width: 100%;
            height: 195px;
            object-fit: cover;
            border-radius: 12px;
            border: 1px solid var(--border-neutral);
            box-shadow: 0 4px 14px rgba(27, 67, 50, 0.07);
        }

        /* Filter Panel */
        .filter-panel-header {
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--primary-forest);
            margin-bottom: 14px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        /* Primary Search Button in Forest Green */
        div.stButton > button[kind="primary"],
        div.stButton > button {
            background-color: var(--primary-forest) !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 700 !important;
            font-size: 0.95rem !important;
            padding: 10px 22px !important;
            transition: all 0.18s ease-in-out !important;
            box-shadow: 0 2px 6px rgba(27, 67, 50, 0.18) !important;
        }
        div.stButton > button:hover {
            background-color: var(--primary-hover) !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 12px rgba(27, 67, 50, 0.25) !important;
        }

        /* Restaurant Result Cards in Grid */
        .restaurant-card {
            background-color: var(--card-white);
            border: 1px solid var(--border-neutral);
            border-radius: 14px;
            overflow: hidden;
            margin-bottom: 22px;
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.03);
            transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
            display: flex;
            flex-direction: column;
            height: 100%;
        }
        .restaurant-card:hover {
            transform: translateY(-3px);
            border-color: var(--primary-hover);
            box-shadow: 0 8px 24px rgba(27, 67, 50, 0.09);
        }
        .card-img-container {
            width: 100%;
            height: 160px;
            position: relative;
            overflow: hidden;
            background-color: #EDEAE3;
        }
        .card-img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            transition: transform 0.35s ease;
        }
        .restaurant-card:hover .card-img {
            transform: scale(1.03);
        }
        .card-rating-badge {
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(4px);
            padding: 3px 9px;
            border-radius: 6px;
            font-weight: 700;
            font-size: 0.85rem;
            color: var(--primary-forest);
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.12);
        }
        .card-body {
            padding: 16px 16px 14px 16px;
            display: flex;
            flex-direction: column;
            flex-grow: 1;
        }
        .card-title {
            font-size: 1.18rem;
            font-weight: 700;
            color: #111827;
            margin: 0 0 6px 0;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            line-height: 1.25;
        }
        .card-meta-row {
            font-size: 0.88rem;
            color: var(--text-muted);
            margin-bottom: 8px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            line-height: 1.4;
        }
        .card-badges-row {
            display: flex;
            gap: 6px;
            margin-bottom: 10px;
            flex-wrap: wrap;
        }
        .card-tag {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 5px;
            font-size: 0.80rem;
            font-weight: 600;
        }
        .tag-price {
            background-color: #FEF3C7;
            color: #92400E;
            border: 1px solid #FDE68A;
        }
        .tag-tier {
            background-color: #F1F5F9;
            color: #334155;
            border: 1px solid #E2E8F0;
        }
        .card-explanation {
            background-color: #FAF8F5;
            border-left: 3px solid var(--primary-hover);
            padding: 8px 10px;
            border-radius: 0 6px 6px 0;
            font-size: 0.82rem;
            color: #374151;
            line-height: 1.42;
            margin-top: auto;
            min-height: 52px;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }
        .explanation-label {
            font-weight: 700;
            color: var(--primary-forest);
            margin-right: 4px;
        }

        /* Empty State */
        .empty-state-box {
            background: var(--card-white);
            border: 1px dashed var(--border-neutral);
            border-radius: 14px;
            padding: 36px 20px;
            text-align: center;
            margin: 20px 0;
        }
        .empty-state-icon {
            font-size: 2.2rem;
            margin-bottom: 8px;
        }
        .empty-state-title {
            font-size: 1.2rem;
            font-weight: 700;
            color: var(--primary-forest);
            margin-bottom: 6px;
        }
        .empty-state-desc {
            color: var(--text-muted);
            font-size: 0.95rem;
            max-width: 500px;
            margin: 0 auto 12px auto;
        }

        /* Recent Searches Bar */
        .history-wrap {
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
            margin-bottom: 16px;
        }
        .history-label {
            font-size: 0.86rem;
            font-weight: 700;
            color: var(--text-muted);
        }
    </style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Data & Artifact Loader
# -----------------------------------------------------------------------------
@st.cache_resource
def get_cached_artifacts():
    """Cached loader for pre-trained model artifacts and dataset."""
    try:
        return load_artifacts(model_dir='model'), None
    except Exception as e:
        return None, str(e)


def init_session():
    """Initializes session state variables for search history and filters."""
    if 'recent_searches' not in st.session_state:
        st.session_state.recent_searches = []
    if 'filter_cuisine' not in st.session_state:
        st.session_state.filter_cuisine = 'All'
    if 'filter_location' not in st.session_state:
        st.session_state.filter_location = 'All'
    if 'filter_budget' not in st.session_state:
        st.session_state.filter_budget = 800
    if 'filter_min_rating' not in st.session_state:
        st.session_state.filter_min_rating = 3.5


# -----------------------------------------------------------------------------
# 1. Discover Page (Main Restaurant Discovery View)
# -----------------------------------------------------------------------------
def render_discover_page(df_cleaned, knn_model, dt_model):
    """
    Main restaurant discovery experience:
    - Balanced Hero section with genuine dining imagery
    - Responsive 2-row filter panel
    - High-density 3-column card grid with authentic cuisine category photos
    """

    # --- Hero Section ---
    col_hero_text, col_hero_visual = st.columns([1.25, 0.85], gap="large", vertical_alignment="center")

    with col_hero_text:
        st.markdown("""
            <div style="padding: 4px 0;">
                <span class="hero-badge">AI-Powered Curation</span>
                <h2 class="hero-heading">Find Your Next Favorite Restaurant</h2>
                <p class="hero-subtext">Discover places that match your cravings, budget, and preferences with algorithmic precision.</p>
                <div class="hero-pills">
                    <span class="hero-pill-item">✓ 12,000+ Verified Spots</span>
                    <span class="hero-pill-item">✓ 25 Regional Cuisines</span>
                    <span class="hero-pill-item">✓ Strict Budget Guarantee</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col_hero_visual:
        hero_b64 = get_base64_image("assets/hero_dining.jpg")
        if hero_b64:
            st.markdown(f"""
                <div>
                    <img src="data:image/jpeg;base64,{hero_b64}" alt="Curated Dining" class="hero-img" />
                </div>
            """, unsafe_allow_html=True)
        else:
            st.image("assets/hero_dining.jpg", use_container_width=True)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # --- Search History Chips (Unobtrusive) ---
    if st.session_state.recent_searches:
        st.markdown("<span style='font-size: 0.85rem; font-weight: 700; color: #525C68;'>Recent Searches:</span>", unsafe_allow_html=True)
        h_cols = st.columns(min(len(st.session_state.recent_searches) + 1, 6))
        for idx, item in enumerate(st.session_state.recent_searches[:5]):
            with h_cols[idx]:
                if st.button(f"↺ {item['cuisine']} in {item['location']}", key=f"hist_{idx}"):
                    st.session_state.filter_cuisine = item['cuisine']
                    st.session_state.filter_location = item['location']
                    st.session_state.filter_budget = item['budget']
                    st.session_state.filter_min_rating = item['min_rating']
                    st.rerun()
        with h_cols[-1]:
            if st.button("Clear", key="clear_hist"):
                st.session_state.recent_searches = []
                st.rerun()
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # --- Search & Filters Panel (Responsive Grid) ---
    with st.container(border=True):
        st.markdown('<div class="filter-panel-header"><span>⚙️</span> Refine Your Preferences</div>', unsafe_allow_html=True)

        # Row 1: Cuisines, Locations, Number of results
        f_row1_c1, f_row1_c2, f_row1_c3 = st.columns([1.3, 1.3, 0.8])

        cuisine_options = ['All'] + TOP_CUISINES
        default_c_idx = cuisine_options.index(st.session_state.filter_cuisine) if st.session_state.filter_cuisine in cuisine_options else 0

        location_options = ['All'] + sorted(TOP_LOCATIONS)
        default_l_idx = location_options.index(st.session_state.filter_location) if st.session_state.filter_location in location_options else 0

        with f_row1_c1:
            sel_cuisine = st.selectbox("🍽️ Cuisine Preference", options=cuisine_options, index=default_c_idx)

        with f_row1_c2:
            sel_location = st.selectbox("📍 Location / Area", options=location_options, index=default_l_idx)

        with f_row1_c3:
            top_n = st.selectbox("🔢 Number of Places", options=[3, 6, 9, 12], index=1)

        # Row 2: Budget slider, Minimum Rating slider, Primary Action Button
        f_row2_c1, f_row2_c2, f_row2_c3 = st.columns([1.3, 1.3, 0.8], vertical_alignment="bottom")

        with f_row2_c1:
            sel_budget = st.slider("💰 Max Budget for Two (₹)", min_value=150, max_value=3000, value=st.session_state.filter_budget, step=50)

        with f_row2_c2:
            sel_min_rating = st.slider("⭐ Minimum Star Rating", min_value=0.0, max_value=5.0, value=st.session_state.filter_min_rating, step=0.1)

        with f_row2_c3:
            search_btn = st.button("Find Restaurants", type="primary", use_container_width=True)

    if search_btn:
        new_search = {
            'cuisine': sel_cuisine,
            'location': sel_location,
            'budget': sel_budget,
            'min_rating': sel_min_rating
        }
        if new_search not in st.session_state.recent_searches:
            st.session_state.recent_searches.insert(0, new_search)
            st.session_state.recent_searches = st.session_state.recent_searches[:5]

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # --- Algorithmic Execution: KNN Recommender ---
    with st.spinner("Finding nearest matching restaurants..."):
        recs = knn_model.recommend(
            cuisine=sel_cuisine if sel_cuisine != 'All' else None,
            location=sel_location if sel_location != 'All' else None,
            budget=float(sel_budget),
            min_rating=float(sel_min_rating) if sel_min_rating > 0 else None,
            top_n=top_n
        )

    # --- Recommended Results Display (3-Column Grid) ---
    st.markdown("### Recommended for You")

    if recs.empty:
        st.markdown("""
            <div class="empty-state-box">
                <div class="empty-state-icon">🔍</div>
                <div class="empty-state-title">No Matching Restaurants Found</div>
                <div class="empty-state-desc">
                    We could not find any restaurants satisfying all your constraints simultaneously.
                </div>
                <div style="font-size: 0.88rem; color: #1B4332; font-weight: 600;">
                    💡 Tip: Try expanding your budget ceiling or lowering your minimum star rating filter.
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"<p style='color: #525C68; margin-top: -8px; margin-bottom: 20px; font-size: 0.95rem;'>Showing top {len(recs)} dining spots matching your exact criteria, ranked by nearest feature similarity.</p>", unsafe_allow_html=True)

        grid_cols = st.columns(3)

        for idx, (_, row) in enumerate(recs.iterrows()):
            col_target = grid_cols[idx % 3]

            name = str(row.get('name', 'Restaurant'))
            location = str(row.get('location_clean', 'Unknown'))
            cuisines = str(row.get('cuisines_clean', 'Multi-Cuisine'))
            cost = row.get('cost_num')
            rating = row.get('rating_num')
            reason = str(row.get('recommendation_reason', 'Matches your dining profile.'))

            rating_text = f"⭐ {rating:.1f}" if pd.notna(rating) and rating > 0 else "⭐ Unrated"
            cost_text = f"₹{int(cost)} for two" if pd.notna(cost) and cost > 0 else "Price unlisted"

            # Predict category using Decision Tree
            try:
                pred_category = dt_model.predict_category(row)
            except Exception:
                pred_category = "Standard"

            # Select category image asset
            img_path = get_cuisine_image_path(cuisines)
            img_b64 = get_base64_image(img_path)

            with col_target:
                card_html = f"""
                    <div class="restaurant-card">
                        <div class="card-img-container">
                            <img src="data:image/jpeg;base64,{img_b64}" alt="{cuisines}" class="card-img" />
                            <div class="card-rating-badge">{rating_text}</div>
                        </div>
                        <div class="card-body">
                            <h4 class="card-title" title="{name}">{name}</h4>
                            <div class="card-meta-row">📍 {location}</div>
                            <div class="card-meta-row">🍲 {cuisines}</div>
                            <div class="card-badges-row">
                                <span class="card-tag tag-price">{cost_text}</span>
                                <span class="card-tag tag-tier">{pred_category} Tier</span>
                            </div>
                            <div class="card-explanation">
                                <span class="explanation-label">Why Recommended:</span>{reason}
                            </div>
                        </div>
                    </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. Explore Page (K-Means Restaurant Market Clusters)
# -----------------------------------------------------------------------------
def render_explore_page(metadata):
    """Secondary Exploration Page: K-Means Restaurant Clusters."""
    st.markdown("## Restaurant Market Segments (K-Means)")
    st.markdown("<p style='color: #525C68; margin-top: -10px; margin-bottom: 24px;'>Explore natural groupings of restaurants partitioned by the K-Means clustering algorithm using pricing, ratings, and dining types.</p>", unsafe_allow_html=True)

    clustering_data = metadata.get('clustering_summary', {})
    summaries = clustering_data.get('cluster_summaries', {})
    reps = clustering_data.get('representative_restaurants', {})
    silhouette = clustering_data.get('silhouette_score', 'N/A')

    # Top Overview Metrics
    c_m1, c_m2, c_m3 = st.columns(3)
    c_m1.metric("Total Segments", f"{len(summaries)}")
    c_m2.metric("Clustering Model", "K-Means (k=4)")
    c_m3.metric("Silhouette Score", f"{silhouette}")

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    cluster_names = {
        0: "Cluster 0: Casual Multicuisine Family Dining",
        1: "Cluster 1: Budget Quick-Bites, Cafes & Fast Food",
        2: "Cluster 2: Premium Fine Dining & Pubs",
        3: "Cluster 3: Value Traditional Eateries & Tiffins"
    }

    selected_c_id = st.segmented_control(
        "Market Segment",
        options=list(range(len(summaries))),
        format_func=lambda x: cluster_names.get(x, f"Cluster {x}"),
        default=0,
        label_visibility="collapsed"
    )
    if selected_c_id is None:
        selected_c_id = 0

    info = summaries.get(str(selected_c_id), summaries.get(selected_c_id, {}))
    rep_list = reps.get(str(selected_c_id), reps.get(selected_c_id, []))

    if info:
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        with st.container(border=True):
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Restaurants in Cluster", f"{info.get('size', 0):,}", f"{info.get('percentage', 0)}% of total")
            col_b.metric("Average Cost for Two", f"₹{int(info.get('avg_cost', 0))}")
            col_c.metric("Average Rating", f"⭐ {info.get('avg_rating', 0)}")
            top_c = ", ".join(info.get('top_cuisines', []))
            col_d.metric("Dominant Cuisines", top_c if top_c else "Various")

        # Representative Restaurants in this segment
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        st.markdown("##### Representative Restaurants in this Segment (Closest to Centroid)")

        r_cols = st.columns(len(rep_list) if rep_list else 1)
        for i, r in enumerate(rep_list):
            with r_cols[i]:
                cuis = r.get('cuisines_clean', 'Multi-Cuisine')
                img_path = get_cuisine_image_path(cuis)
                img_b64 = get_base64_image(img_path)

                r_cost = r.get('cost_num')
                cost_disp = f"₹{int(r_cost)}" if pd.notna(r_cost) else "N/A"
                r_rate = r.get('rating_num')
                rate_disp = f"⭐ {r_rate:.1f}" if pd.notna(r_rate) else "⭐ Unrated"

                st.markdown(f"""
                    <div class="restaurant-card">
                        <div class="card-img-container" style="height: 130px;">
                            <img src="data:image/jpeg;base64,{img_b64}" class="card-img" />
                            <div class="card-rating-badge">{rate_disp}</div>
                        </div>
                        <div class="card-body">
                            <h4 class="card-title">{r.get('name', 'Restaurant')}</h4>
                            <div class="card-meta-row">📍 {r.get('location_clean', 'Unknown')}</div>
                            <div class="card-meta-row">🍲 {cuis}</div>
                            <div class="card-badges-row">
                                <span class="card-tag tag-price">{cost_disp} for two</span>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

    # Cluster Size Distribution Chart
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Cluster Size Distribution")

    fig, ax = plt.subplots(figsize=(8, 2.7))
    fig.patch.set_facecolor('#FAF8F5')
    ax.set_facecolor('#FAF8F5')

    c_labels = [f"Cluster {i}" for i in range(len(summaries))]
    c_sizes = [summaries.get(str(i), {}).get('size', 0) for i in range(len(summaries))]

    bars = ax.barh(c_labels, c_sizes, color=['#1B4332', '#2D6A4F', '#40916C', '#52B788'], edgecolor='#1B4332')
    ax.set_xlabel("Number of Restaurants", fontsize=9, color='#1F2937')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x', linestyle='--', alpha=0.35, color='#B0BEC5')

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 70, bar.get_y() + bar.get_height()/2, f"{width:,}", va='center', fontsize=9, fontweight='bold', color='#1B4332')

    plt.tight_layout()
    st.pyplot(fig)


# -----------------------------------------------------------------------------
# 3. ML Insights Page (Faculty Demonstration Panel)
# -----------------------------------------------------------------------------
def render_ml_insights_page(metadata):
    """Technical Demonstration Page for Faculty Evaluation."""
    st.markdown("## DWDM Machine Learning Insights")
    st.markdown("<p style='color: #525C68; margin-top: -10px; margin-bottom: 24px;'>Comprehensive technical evaluation of the algorithms implemented according to the DWDM syllabus.</p>", unsafe_allow_html=True)

    tab_dt, tab_knn, tab_kmeans = st.tabs([
        "Algorithm C: Decision Tree & Logistic Regression",
        "Algorithm A: KNN Recommender",
        "Algorithm B: K-Means Clustering"
    ])

    eval_data = metadata.get('evaluation_results', {})
    dt_metrics = eval_data.get('decision_tree', {})
    lr_metrics = eval_data.get('logistic_regression', {})

    with tab_dt:
        st.markdown("#### Rating Category Prediction Task")
        st.write("""
        **Objective**: Predict whether a restaurant will achieve **Low** (<3.5), **Medium** (3.5 - 3.9), or **High** (≥4.0) rating 
        purely from its operational attributes (budget, online ordering, table reservations, cuisine offerings, location, and restaurant format).
        """)

        st.info("🛡️ **Data Leakage Prevention Verification**: Original customer rating (`rate`, `rating_num`) was strictly excluded from feature inputs. Preprocessor was fitted strictly on the 80% training partition.")

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown("##### Model Evaluation Metrics (Test Partition = 20%)")
            metrics_df = pd.DataFrame({
                'Metric': ['Accuracy', 'Weighted Precision', 'Weighted Recall', 'Weighted F1-Score'],
                'Decision Tree (max_depth=5)': [
                    f"{dt_metrics.get('accuracy', 0)*100:.2f}%",
                    f"{dt_metrics.get('precision_weighted', 0):.4f}",
                    f"{dt_metrics.get('recall_weighted', 0):.4f}",
                    f"{dt_metrics.get('f1_weighted', 0):.4f}"
                ],
                'Logistic Regression (Comparison)': [
                    f"{lr_metrics.get('accuracy', 0)*100:.2f}%",
                    f"{lr_metrics.get('precision_weighted', 0):.4f}",
                    f"{lr_metrics.get('recall_weighted', 0):.4f}",
                    f"{lr_metrics.get('f1_weighted', 0):.4f}"
                ]
            })
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        with col_t2:
            st.markdown("##### Decision Tree Confusion Matrix")
            classes = ['Low', 'Medium', 'High']
            cm = dt_metrics.get('confusion_matrix', [[0, 0, 0], [0, 0, 0], [0, 0, 0]])
            cm_df = pd.DataFrame(cm, index=[f"Actual {c}" for c in classes], columns=[f"Pred {c}" for c in classes])
            st.dataframe(cm_df, use_container_width=True)

        # Feature Importances
        st.markdown("##### Top Predictive Attributes in Decision Tree")
        top_feats = dt_metrics.get('top_features', [])
        if top_feats:
            tf_df = pd.DataFrame(top_feats)
            tf_df.columns = ['Feature Name', 'Gini Importance']
            st.dataframe(tf_df, use_container_width=True, hide_index=True)

    with tab_knn:
        st.markdown("#### Algorithm A: k-Nearest Neighbors Recommender")
        st.write("""
        - **Algorithm**: `sklearn.neighbors.NearestNeighbors`
        - **Metric**: Euclidean distance over normalized continuous and categorical space.
        - **Feature Representation**:
          1. Continuous: Cost for two (StandardScaler), Rating (StandardScaler)
          2. Binary: Online ordering (1/0), Table booking (1/0)
          3. Multi-hot: Top 25 regional cuisines
          4. One-hot: Top 25 locations and top 10 restaurant formats
        - **Query Transformation**: User search parameters are projected into the identical 64-dimensional feature vector.
        - **Hard Constraints**: Strict filtering on budget ceiling and minimum star rating to guarantee constraint adherence.
        """)

    with tab_kmeans:
        st.markdown("#### Algorithm B: K-Means Restaurant Clustering")
        st.write(f"""
        - **Algorithm**: `sklearn.cluster.KMeans` (k=4, random_state=42)
        - **Silhouette Score**: `{metadata.get('clustering_summary', {}).get('silhouette_score', 'N/A')}`
        - **Feature Space**: 64-dimensional structured attribute space.
        - **Centroid Convergence**: Successfully converged across all 12,178 unique restaurant records.
        """)


# -----------------------------------------------------------------------------
# Main Application Flow
# -----------------------------------------------------------------------------
def main():
    init_session()

    # Load Model Artifacts
    artifacts, load_err = get_cached_artifacts()

    # Top Compact Header Bar (Brand on left, Clean Segmented Control on right)
    col_brand, col_nav = st.columns([1.5, 1.4], vertical_alignment="center")

    with col_brand:
        st.markdown("""
            <div class="brand-cluster">
                <div class="brand-icon-box">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#FAF8F5" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M18 2v8a4 4 0 0 1-4 4H4a2 2 0 0 1-2-2V2"/>
                        <path d="M6 2v20"/>
                        <path d="M18 14v8"/>
                        <path d="M10 2v6"/>
                    </svg>
                </div>
                <div>
                    <h1 class="brand-title">DineWise ML</h1>
                    <p class="brand-tagline">Discover restaurants made for your taste.</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col_nav:
        # Sleek modern segmented control: Discover (default), Explore, ML Insights (NO radio dots!)
        page_view = st.segmented_control(
            "Navigation",
            options=["Discover", "Explore", "ML Insights"],
            default="Discover",
            label_visibility="collapsed"
        )
        if not page_view:
            page_view = "Discover"

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    if load_err or artifacts is None:
        st.error("⚠️ Model artifacts not loaded.")
        st.info(f"Details: {load_err}")
        st.warning("Please run `python train_model.py` in your terminal to train models on your dataset.")
        return

    df_cleaned, feature_pipeline, knn_model, kmeans_model, dt_model, metadata = artifacts

    # View Routing
    if page_view == "Discover":
        render_discover_page(df_cleaned, knn_model, dt_model)
    elif page_view == "Explore":
        render_explore_page(metadata)
    else:
        render_ml_insights_page(metadata)


if __name__ == '__main__':
    main()
