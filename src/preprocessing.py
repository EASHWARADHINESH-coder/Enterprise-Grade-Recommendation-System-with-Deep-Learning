"""
Preprocessing & Feature Engineering
===================================
Enterprise-Grade Recommendation System with Deep Learning

Turns the raw interaction log into the structures every recommender needs:

    user_item_matrix              explicit ratings           (CF, SVD)
    normalized_user_item_matrix   mean-centred ratings       (SVD)
    implicit_feedback_matrix      cart/purchase intent       (implicit CF, NCF)
    item_popularity_features      exposure & long-tail flags (re-ranking)
    item_content_features         price/category/brand       (cold-start items)
    user_profile_features         segment/age/gender         (cold-start users)

Matrices are stored as float32. At 6,000 x 2,500 the explicit matrix alone is
15 million cells; float64 would double the memory for no modelling benefit.

Run:
    python src/preprocessing.py
"""

import os
import pickle

import numpy as np
import pandas as pd

from data_loader import load_all


# =========================================================
# PATHS
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)


# =========================================================
# SETTINGS
# =========================================================
# An item at or below this interaction percentile counts as long-tail.
LONG_TAIL_QUANTILE = 0.80

# A rating at or above this counts as a positive signal for ranking metrics.
RELEVANCE_RATING_THRESHOLD = 4


# =========================================================
# EXPLICIT RATING MATRIX
# =========================================================
def create_user_item_matrix(interactions_df):
    """
    Build the user x item explicit rating matrix.

    Only rated interactions contribute. Unrated cells become 0, which the
    downstream models treat as "no explicit signal" rather than "rated zero" -
    the rating scale starts at 1, so 0 is unambiguous.
    """
    rated = interactions_df.dropna(subset=["rating"])

    return rated.pivot_table(
        index="user_id",
        columns="item_id",
        values="rating",
        aggfunc="mean",
        fill_value=0,
    ).astype(np.float32)


# =========================================================
# MEAN-CENTRED RATING MATRIX
# =========================================================
def create_normalized_user_item_matrix(interactions_df):
    """
    Mean-centre each user's ratings before factorisation.

    Without this, a generous rater and a harsh rater who rank items identically
    look like different users to the model. Centring removes that per-user
    offset so the factors capture relative preference.

    Uses a vectorised groupby-transform; the row-wise .apply equivalent is
    orders of magnitude slower on a log of this size.
    """
    rated = interactions_df.dropna(subset=["rating"]).copy()

    user_mean = rated.groupby("user_id")["rating"].transform("mean")
    rated["normalized_rating"] = rated["rating"] - user_mean

    return rated.pivot_table(
        index="user_id",
        columns="item_id",
        values="normalized_rating",
        aggfunc="mean",
        fill_value=0,
    ).astype(np.float32)


# =========================================================
# IMPLICIT FEEDBACK MATRIX
# =========================================================
def create_implicit_feedback_matrix(interactions_df):
    """
    Build the binary intent matrix from the engagement funnel.

    A cell is 1 when the user added the item to cart or purchased it. This is a
    genuine behavioural signal, not a rating threshold applied after the fact -
    which is why it stays available for the ~31% of interactions that carry no
    star rating at all.
    """
    interactions_copy = interactions_df.copy()

    if "implicit_feedback" not in interactions_copy.columns:
        interactions_copy["implicit_feedback"] = (
            (interactions_copy["add_to_cart"] == 1) | (interactions_copy["purchase"] == 1)
        ).astype(int)

    return interactions_copy.pivot_table(
        index="user_id",
        columns="item_id",
        values="implicit_feedback",
        aggfunc="max",
        fill_value=0,
    ).astype(np.float32)


# =========================================================
# ITEM POPULARITY / LONG-TAIL FEATURES
# =========================================================
def create_item_popularity_features(interactions_df, items_df=None):
    """
    Compute per-item exposure statistics used by the popularity re-ranker.

    When `items_df` is supplied, items with zero interactions are included with
    counts of 0. That matters: cold-start items are exactly the ones missing
    from the interaction log, and silently dropping them would make the
    cold-start handling untestable.
    """
    item_stats = interactions_df.groupby("item_id").agg(
        interaction_count=("item_id", "count"),
        average_rating=("rating", "mean"),
        purchase_count=("purchase", "sum"),
        total_revenue=("revenue", "sum"),
        avg_view_time=("view_time_seconds", "mean"),
    )

    if items_df is not None:
        item_stats = item_stats.reindex(items_df["item_id"].values)

    for column in ["interaction_count", "purchase_count", "total_revenue",
                   "avg_view_time", "average_rating"]:
        item_stats[column] = item_stats[column].fillna(0.0)

    item_stats = item_stats.reset_index()
    if "index" in item_stats.columns:
        item_stats = item_stats.rename(columns={"index": "item_id"})

    max_count = item_stats["interaction_count"].max()
    item_stats["popularity_ratio"] = (
        item_stats["interaction_count"] / max_count if max_count > 0 else 0.0
    )

    # Conversion rate is the commercially meaningful quality signal: an item
    # with few views but a high buy-through rate deserves promotion.
    item_stats["conversion_rate"] = np.where(
        item_stats["interaction_count"] > 0,
        item_stats["purchase_count"] / item_stats["interaction_count"],
        0.0,
    )

    threshold = item_stats["interaction_count"].quantile(LONG_TAIL_QUANTILE)
    item_stats["is_long_tail"] = (item_stats["interaction_count"] <= threshold).astype(int)

    return item_stats


# =========================================================
# ITEM CONTENT FEATURES
# =========================================================
def create_item_content_features(items_df):
    """
    Build a numeric item feature matrix for cold-start scoring.

    This is the structured counterpart to the TF-IDF text representation: price
    and quality are scaled to [0, 1], and category / subcategory / brand are
    one-hot encoded. A brand-new item has all of these on day one, which is
    what makes it recommendable before it has any interaction history.
    """
    items_copy = items_df.copy()

    numeric_cols = [c for c in ["price", "base_quality", "price_percentile"]
                    if c in items_copy.columns]
    categorical_cols = [c for c in ["category", "subcategory", "brand"]
                        if c in items_copy.columns]

    for col in numeric_cols:
        min_val = items_copy[col].min()
        max_val = items_copy[col].max()
        if max_val > min_val:
            items_copy[col] = (items_copy[col] - min_val) / (max_val - min_val)
        else:
            items_copy[col] = 0.0

    cat_df = pd.get_dummies(items_copy[categorical_cols], drop_first=False).astype(np.float32)
    num_df = items_copy[numeric_cols].astype(np.float32)

    feature_df = pd.concat([items_copy[["item_id"]], num_df, cat_df], axis=1)
    return feature_df.set_index("item_id")


# =========================================================
# USER PROFILE FEATURES
# =========================================================
def create_user_profile_features(users_df):
    """
    Build a numeric user feature matrix for cold-start scoring.

    `user_segment` is the field that carries commercial meaning here - it is
    what lets the system make a sensible first recommendation to a user who has
    never clicked anything, purely from who they signed up as.
    """
    users_copy = users_df.copy()

    if "age" in users_copy.columns:
        min_age = users_copy["age"].min()
        max_age = users_copy["age"].max()
        if max_age > min_age:
            users_copy["age"] = (users_copy["age"] - min_age) / (max_age - min_age)
        else:
            users_copy["age"] = 0.0

    categorical_cols = [c for c in ["user_segment", "preferred_category", "gender", "location"]
                        if c in users_copy.columns]

    profile_df = pd.get_dummies(
        users_copy[["user_id", "age"] + categorical_cols],
        columns=categorical_cols,
        drop_first=False,
    )

    return profile_df.set_index("user_id").astype(np.float32)


# =========================================================
# USER ENGAGEMENT FEATURES
# =========================================================
def create_user_engagement_features(interactions_df):
    """
    Aggregate behavioural features per user.

    Used by the governance notebook to check that recommendation quality does
    not collapse for low-activity users, and by the dashboard to describe who a
    selected user actually is.
    """
    engagement = interactions_df.groupby("user_id").agg(
        interaction_count=("item_id", "count"),
        distinct_items=("item_id", "nunique"),
        purchase_count=("purchase", "sum"),
        total_spend=("revenue", "sum"),
        avg_rating_given=("rating", "mean"),
        avg_view_time=("view_time_seconds", "mean"),
        first_seen=("timestamp", "min"),
        last_seen=("timestamp", "max"),
    )

    engagement["conversion_rate"] = np.where(
        engagement["interaction_count"] > 0,
        engagement["purchase_count"] / engagement["interaction_count"],
        0.0,
    )
    engagement["avg_rating_given"] = engagement["avg_rating_given"].fillna(0.0)

    return engagement


# =========================================================
# RELEVANCE LABELS  (used by the ranking metrics)
# =========================================================
def mark_relevant_interactions(interactions_df):
    """
    Flag which interactions count as a "hit" for Precision@K / Recall@K.

    An interaction is relevant if the user rated it at or above the threshold,
    OR if they purchased it. Including purchases matters because a bought-but-
    unrated item is unambiguous evidence of relevance, and dropping it would
    understate every model's recall.
    """
    interactions_copy = interactions_df.copy()

    rated_high = interactions_copy["rating"] >= RELEVANCE_RATING_THRESHOLD
    purchased = interactions_copy["purchase"] == 1

    interactions_copy["is_relevant"] = (rated_high.fillna(False) | purchased).astype(int)
    return interactions_copy


# =========================================================
# PERSISTENCE
# =========================================================
def save_pickle(obj, filename):
    with open(os.path.join(PROCESSED_DATA_PATH, filename), "wb") as f:
        pickle.dump(obj, f)


def load_pickle(filename):
    path = os.path.join(PROCESSED_DATA_PATH, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Artifact not found: " + path + "\n"
            "Run the pipeline first:  python src/preprocessing.py"
        )
    with open(path, "rb") as f:
        return pickle.load(f)


# =========================================================
# PIPELINE ENTRY POINT
# =========================================================
def main():
    users_df, items_df, interactions_df = load_all()

    print("Building preprocessing artifacts ...")

    user_item_matrix = create_user_item_matrix(interactions_df)
    save_pickle(user_item_matrix, "user_item_matrix.pkl")
    print("  user_item_matrix            :", user_item_matrix.shape)

    normalized_matrix = create_normalized_user_item_matrix(interactions_df)
    save_pickle(normalized_matrix, "normalized_user_item_matrix.pkl")
    print("  normalized_user_item_matrix :", normalized_matrix.shape)

    implicit_matrix = create_implicit_feedback_matrix(interactions_df)
    save_pickle(implicit_matrix, "implicit_feedback_matrix.pkl")
    print("  implicit_feedback_matrix    :", implicit_matrix.shape)

    item_popularity = create_item_popularity_features(interactions_df, items_df)
    save_pickle(item_popularity, "item_popularity_features.pkl")
    print("  item_popularity_features    :", item_popularity.shape)

    item_content = create_item_content_features(items_df)
    save_pickle(item_content, "item_content_features.pkl")
    print("  item_content_features       :", item_content.shape)

    user_profile = create_user_profile_features(users_df)
    save_pickle(user_profile, "user_profile_features.pkl")
    print("  user_profile_features       :", user_profile.shape)

    user_engagement = create_user_engagement_features(interactions_df)
    save_pickle(user_engagement, "user_engagement_features.pkl")
    print("  user_engagement_features    :", user_engagement.shape)

    sparsity = 1 - (len(interactions_df) / (len(users_df) * len(items_df)))
    print("\nInteraction matrix sparsity : {:.4%}".format(sparsity))
    print("Cold-start items (<3 events):",
          int((item_popularity["interaction_count"] < 3).sum()))
    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()
