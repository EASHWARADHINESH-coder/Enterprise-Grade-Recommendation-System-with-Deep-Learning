"""
Content-Based Recommendation (NLP)
==================================
Enterprise-Grade Recommendation System with Deep Learning

Recommends items from their text, not from who bought them. That distinction is
what makes this module the answer to the item cold-start problem: a product
listed this morning has a full description and tag set, and therefore a full
content vector, before a single customer has ever clicked it.

Pipeline:
    description + tags + category + brand
        -> TF-IDF vectorisation
        -> cosine similarity
        -> item-item neighbours / user taste profiles

Run:
    python src/content_based_nlp.py
"""

import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data_loader import load_all
from preprocessing import save_pickle


# =========================================================
# PATHS
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)


# =========================================================
# SETTINGS
# =========================================================
MAX_TFIDF_FEATURES = 5000
RELEVANCE_RATING_THRESHOLD = 4
COLD_START_MIN_INTERACTIONS = 3

# Centre of the price band each segment gravitates to. Used to give a brand-new
# user a price-appropriate first recommendation rather than the most expensive
# item in their chosen category.
SEGMENT_PRICE_PERCENTILE = {
    "Budget Shopper": 0.15,
    "Value Seeker": 0.35,
    "Mainstream Buyer": 0.55,
    "Premium Buyer": 0.78,
    "Luxury Enthusiast": 0.93,
}


# =========================================================
# TEXT ASSEMBLY
# =========================================================
def build_item_text(items_df):
    """
    Assemble the text document that represents each item.

    The description alone would under-weight the structured attributes, so
    category, subcategory, brand and tags are appended. Tags are un-hyphenated
    ("noise-cancelling" -> "noise cancelling") so the vectoriser can match on
    the individual words as well as the phrase.
    """
    items_copy = items_df.copy()

    tag_text = (
        items_copy["content_tags"].fillna("")
        .str.replace("|", " ", regex=False)
        .str.replace("-", " ", regex=False)
    )

    items_copy["content_text"] = (
        items_copy["description"].fillna("") + " "
        + items_copy["category"].fillna("") + " "
        + items_copy["subcategory"].fillna("") + " "
        + items_copy["brand"].fillna("") + " "
        + tag_text
    ).str.strip()

    return items_copy


# =========================================================
# TF-IDF FEATURES
# =========================================================
def create_tfidf_features(items_df, max_features=MAX_TFIDF_FEATURES):
    """
    Fit the TF-IDF vectoriser over the catalogue.

    `min_df=2` drops terms appearing in a single product - almost always a model
    code, which contributes nothing but noise to similarity. Bigrams are enabled
    so multi-word attributes ("stainless steel") survive as single features
    rather than being split into two weak unigrams.
    """
    items_with_text = build_item_text(items_df)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )

    tfidf_matrix = vectorizer.fit_transform(items_with_text["content_text"])

    save_pickle(vectorizer, "tfidf_vectorizer.pkl")
    save_pickle(tfidf_matrix, "tfidf_matrix.pkl")

    items_with_text[["item_id", "title", "category", "description", "content_text"]].to_csv(
        os.path.join(PROCESSED_DATA_PATH, "item_text_metadata.csv"), index=False
    )

    return items_with_text, vectorizer, tfidf_matrix


# =========================================================
# ITEM-ITEM CONTENT SIMILARITY
# =========================================================
def create_content_similarity_matrix(items_df, tfidf_matrix):
    """Dense item x item cosine similarity over the TF-IDF vectors."""
    similarity = cosine_similarity(tfidf_matrix, tfidf_matrix).astype(np.float32)

    similarity_df = pd.DataFrame(
        similarity,
        index=items_df["item_id"].values,
        columns=items_df["item_id"].values,
    )

    save_pickle(similarity_df, "content_similarity.pkl")
    return similarity_df


def recommend_similar_items(item_id, items_df, similarity_df, n_top=10):
    """
    Find the N most content-similar items.

    Powers the "customers also viewed" surface and the /similar-items endpoint.
    Works for any catalogued item, including ones with zero sales history.
    """
    if item_id not in similarity_df.index:
        raise ValueError("item_id " + str(item_id) + " not found in the content similarity matrix.")

    scores = (
        similarity_df.loc[item_id]
        .drop(item_id, errors="ignore")
        .sort_values(ascending=False)
        .head(n_top)
    )

    columns = [c for c in ["item_id", "title", "category", "subcategory", "brand",
                           "price", "content_tags"] if c in items_df.columns]

    recommendations = items_df.loc[items_df["item_id"].isin(scores.index), columns].copy()
    recommendations["item_id"] = pd.Categorical(
        recommendations["item_id"], categories=scores.index, ordered=True
    )
    recommendations = recommendations.sort_values("item_id").reset_index(drop=True)
    recommendations["similarity_score"] = scores.to_numpy()
    recommendations["item_id"] = recommendations["item_id"].astype(int)

    return recommendations


# =========================================================
# USER TASTE PROFILE
# =========================================================
def build_user_content_profile(user_id, interactions_df, items_df, tfidf_matrix,
                               min_rating=RELEVANCE_RATING_THRESHOLD):
    """
    Represent a user as the centroid of the items they liked.

    "Liked" means rated at or above the threshold OR purchased. Including
    purchases materially widens coverage, because a large share of purchases
    never receive a star rating.

    Returns (profile_vector, liked_item_ids). The profile is None when the user
    has no positive signal at all - the caller's cue to fall back to cold-start
    handling rather than fabricate a taste vector.
    """
    user_rows = interactions_df.loc[interactions_df["user_id"] == user_id]

    if user_rows.empty:
        return None, []

    liked_mask = (user_rows["rating"] >= min_rating) | (user_rows["purchase"] == 1)
    liked_item_ids = user_rows.loc[liked_mask.fillna(False), "item_id"].unique().tolist()

    if not liked_item_ids:
        return None, []

    position_lookup = pd.Series(np.arange(len(items_df)), index=items_df["item_id"].values)
    liked_positions = [position_lookup[i] for i in liked_item_ids if i in position_lookup.index]

    if not liked_positions:
        return None, []

    profile = np.asarray(tfidf_matrix[liked_positions].mean(axis=0))
    return profile, liked_item_ids


def recommend_for_existing_user(user_id, interactions_df, items_df, tfidf_matrix, n_top=10):
    """Rank the catalogue by cosine similarity to the user's taste profile."""
    profile, liked_item_ids = build_user_content_profile(
        user_id, interactions_df, items_df, tfidf_matrix
    )

    if profile is None:
        return pd.DataFrame(columns=["item_id", "title", "category", "price", "similarity_score"])

    scores = cosine_similarity(profile, tfidf_matrix).flatten()

    columns = [c for c in ["item_id", "title", "category", "subcategory", "brand", "price"]
               if c in items_df.columns]

    recommendation_df = items_df[columns].copy()
    recommendation_df["similarity_score"] = scores

    # Never recommend something the user has already engaged with.
    seen = interactions_df.loc[interactions_df["user_id"] == user_id, "item_id"].unique()
    recommendation_df = recommendation_df.loc[~recommendation_df["item_id"].isin(seen)]

    return recommendation_df.sort_values("similarity_score", ascending=False).head(n_top)


# =========================================================
# COLD-START HANDLING
# =========================================================
def get_cold_start_items(interactions_df, items_df, min_interactions=COLD_START_MIN_INTERACTIONS):
    """
    Identify catalogue items with too little history for collaborative methods.

    Reindexing against the full catalogue is essential - an item with *zero*
    interactions never appears in the interaction log at all, and is precisely
    the case that must not be missed.
    """
    counts = (
        interactions_df.groupby("item_id").size()
        .reindex(items_df["item_id"].values, fill_value=0)
        .reset_index()
    )
    counts.columns = ["item_id", "interaction_count"]

    return counts.loc[counts["interaction_count"] < min_interactions].reset_index(drop=True)


def recommend_for_cold_start_item(cold_item_id, items_df, similarity_df, n_top=10):
    """
    Recommend neighbours for an item with no meaningful interaction history.

    Collaborative signals are unavailable by definition here, so the text
    representation carries the whole decision.
    """
    return recommend_similar_items(cold_item_id, items_df, similarity_df, n_top)


def recommend_for_new_user_by_profile(preferred_category, items_df, user_segment=None, n_top=10):
    """
    First-touch recommendations for a user with no interaction history.

    Uses only what registration provides: the category they said they care
    about and the segment they fall into. The segment sets the target price
    band, so a Budget Shopper is not greeted with the most expensive item in
    the category - which is the failure mode of a naive popularity fallback.
    """
    candidates = items_df.loc[
        items_df["category"].str.lower() == str(preferred_category).lower()
    ].copy()

    if candidates.empty:
        candidates = items_df.copy()

    if user_segment in SEGMENT_PRICE_PERCENTILE and "price_percentile" in candidates.columns:
        target = SEGMENT_PRICE_PERCENTILE[user_segment]
        # Rank by closeness to the segment's price band, then by intrinsic quality.
        candidates["price_fit"] = 1.0 - (candidates["price_percentile"] - target).abs()
        quality = candidates["base_quality"] if "base_quality" in candidates.columns else 0.5
        candidates["cold_start_score"] = 0.6 * candidates["price_fit"] + 0.4 * quality
    else:
        candidates["cold_start_score"] = (
            candidates["base_quality"] if "base_quality" in candidates.columns else 0.5
        )

    columns = [c for c in ["item_id", "title", "category", "subcategory", "brand",
                           "price", "cold_start_score"] if c in candidates.columns]

    return candidates.sort_values("cold_start_score", ascending=False)[columns].head(n_top)


# =========================================================
# EXPLAINABILITY SUPPORT
# =========================================================
def top_shared_terms(item_id_a, item_id_b, items_df, vectorizer, tfidf_matrix, n_terms=5):
    """
    Return the TF-IDF terms that drive similarity between two items.

    This is what turns "similarity 0.83" into an explanation a human accepts:
    the shared vocabulary is the actual reason the two items scored close.
    """
    position_lookup = pd.Series(np.arange(len(items_df)), index=items_df["item_id"].values)

    if item_id_a not in position_lookup.index or item_id_b not in position_lookup.index:
        return []

    vec_a = tfidf_matrix[position_lookup[item_id_a]].toarray().flatten()
    vec_b = tfidf_matrix[position_lookup[item_id_b]].toarray().flatten()

    contribution = vec_a * vec_b
    if not contribution.any():
        return []

    feature_names = np.asarray(vectorizer.get_feature_names_out())
    top_positions = np.argsort(-contribution)[:n_terms]

    return [str(feature_names[p]) for p in top_positions if contribution[p] > 0]


# =========================================================
# PIPELINE ENTRY POINT
# =========================================================
def main():
    users_df, items_df, interactions_df = load_all()

    print("Building content-based (TF-IDF) artifacts ...")

    items_with_text, vectorizer, tfidf_matrix = create_tfidf_features(items_df)
    print("  TF-IDF matrix       :", tfidf_matrix.shape)
    print("  vocabulary size     : {:,}".format(len(vectorizer.get_feature_names_out())))

    similarity_df = create_content_similarity_matrix(items_with_text, tfidf_matrix)
    print("  content similarity  :", similarity_df.shape)

    # ---- Demo 1: similar items ------------------------------------------
    sample_item_id = int(items_df["item_id"].iloc[0])
    base = items_df.loc[items_df["item_id"] == sample_item_id].iloc[0]
    print("\nItems similar to #{} - {} ({}):".format(
        sample_item_id, base["title"], base["category"]))
    print(recommend_similar_items(sample_item_id, items_with_text, similarity_df, 5)
          .to_string(index=False))

    # ---- Demo 2: existing user ------------------------------------------
    active_user = int(interactions_df["user_id"].value_counts().index[0])
    print("\nContent recommendations for active user {}:".format(active_user))
    print(recommend_for_existing_user(active_user, interactions_df, items_with_text,
                                      tfidf_matrix, 5).to_string(index=False))

    # ---- Demo 3: cold-start items ---------------------------------------
    cold_items = get_cold_start_items(interactions_df, items_df)
    print("\nCold-start items detected:", len(cold_items))

    if not cold_items.empty:
        cold_id = int(cold_items["item_id"].iloc[0])
        print("Neighbours for cold-start item {}:".format(cold_id))
        print(recommend_for_cold_start_item(cold_id, items_with_text, similarity_df, 5)
              .to_string(index=False))

    # ---- Demo 4: cold-start user ----------------------------------------
    print("\nFirst-touch recommendations for a new Budget Shopper who likes Electronics:")
    print(recommend_for_new_user_by_profile("Electronics", items_df, "Budget Shopper", 5)
          .to_string(index=False))

    print("\nContent-based artifacts saved to", PROCESSED_DATA_PATH)


if __name__ == "__main__":
    main()
