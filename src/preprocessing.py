"""Preprocessing & Feature Engineering - builds the matrices models consume."""

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
LONG_TAIL_QUANTILE = 0.80
RELEVANCE_RATING_THRESHOLD = 4


# =========================================================
# EXPLICIT RATING MATRIX
# =========================================================
def create_user_item_matrix(interactions_df: pd.DataFrame) -> pd.DataFrame:
    """Build the user x item explicit rating matrix."""
    # Ratings start at 1, so a 0 unambiguously means "no explicit signal".
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
def create_normalized_user_item_matrix(interactions_df: pd.DataFrame) -> pd.DataFrame:
    """Mean-centre each user's ratings so factors capture relative preference."""
    rated = interactions_df.dropna(subset=["rating"]).copy()

    # transform broadcasts the group mean in one pass; .apply is far slower.
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
def create_implicit_feedback_matrix(interactions_df: pd.DataFrame) -> pd.DataFrame:
    """Build the binary cart/purchase intent matrix."""
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
def create_item_popularity_features(interactions_df: pd.DataFrame,
                                    items_df: pd.DataFrame = None) -> pd.DataFrame:
    """Per-item exposure stats used by the popularity re-ranker."""
    item_stats = interactions_df.groupby("item_id").agg(
        interaction_count=("item_id", "count"),
        average_rating=("rating", "mean"),
        purchase_count=("purchase", "sum"),
        total_revenue=("revenue", "sum"),
        avg_view_time=("view_time_seconds", "mean"),
    )

    # Reindex keeps zero-interaction items, which never appear in the log
    # and are exactly the cold-start cases that must not be dropped.
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
def create_item_content_features(items_df: pd.DataFrame) -> pd.DataFrame:
    """Numeric item features for cold-start scoring."""
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
def create_user_profile_features(users_df: pd.DataFrame) -> pd.DataFrame:
    """Numeric user features for cold-start scoring."""
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
def create_user_engagement_features(interactions_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate behavioural features per user."""
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
# RELEVANCE LABELS
# =========================================================
def mark_relevant_interactions(interactions_df: pd.DataFrame) -> pd.DataFrame:
    """Flag which interactions count as a hit for the ranking metrics."""
    interactions_copy = interactions_df.copy()

    # Purchases count even when unrated - dropping them understates recall.
    rated_high = interactions_copy["rating"] >= RELEVANCE_RATING_THRESHOLD
    purchased = interactions_copy["purchase"] == 1

    interactions_copy["is_relevant"] = (rated_high.fillna(False) | purchased).astype(int)
    return interactions_copy


# =========================================================
# PERSISTENCE
# =========================================================
def save_pickle(obj, filename: str) -> None:
    """Write an artifact to data/processed."""
    with open(os.path.join(PROCESSED_DATA_PATH, filename), "wb") as f:
        pickle.dump(obj, f)


def load_pickle(filename: str):
    """Read an artifact from data/processed."""
    path = os.path.join(PROCESSED_DATA_PATH, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Artifact not found: " + path + "\n"
            "Run the pipeline first:  python src/preprocessing.py"
        )
    with open(path, "rb") as f:
        return pickle.load(f)


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    """Build and save every preprocessing artifact."""
    users_df, items_df, interactions_df = load_all()

    print("Building preprocessing artifacts ...")

    artifacts = [
        ("user_item_matrix", create_user_item_matrix(interactions_df)),
        ("normalized_user_item_matrix", create_normalized_user_item_matrix(interactions_df)),
        ("implicit_feedback_matrix", create_implicit_feedback_matrix(interactions_df)),
        ("item_popularity_features", create_item_popularity_features(interactions_df, items_df)),
        ("item_content_features", create_item_content_features(items_df)),
        ("user_profile_features", create_user_profile_features(users_df)),
        ("user_engagement_features", create_user_engagement_features(interactions_df)),
    ]

    for name, obj in artifacts:
        save_pickle(obj, name + ".pkl")
        print("  {:<28}: {}".format(name, obj.shape))

    item_popularity = artifacts[3][1]
    sparsity = 1 - (len(interactions_df) / (len(users_df) * len(items_df)))

    print("\nInteraction matrix sparsity : {:.4%}".format(sparsity))
    print("Cold-start items (<3 events):",
          int((item_popularity["interaction_count"] < 3).sum()))
    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()
