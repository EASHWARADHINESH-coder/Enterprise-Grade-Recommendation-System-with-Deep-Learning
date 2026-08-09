"""Hybrid Recommendation System - fuses four signals into one ranked list."""

import os
import sys

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from collaborative_filtering import (
    build_popularity_lookup,
    min_max_normalize,
    rerank_with_popularity_balance,
)
from content_based_nlp import recommend_for_new_user_by_profile, recommend_similar_items
from data_loader import load_all
from ncf_recommender import load_ncf_model, score_all_items
from preprocessing import load_pickle


# =========================================================
# SETTINGS
# =========================================================
# Chosen by measurement, not intuition. The original svd+content+ncf blend
# scored NDCG@10 0.0947 - worse than item-CF alone (0.1245). Sweep result:
# item_cf .45 / svd .10 / content .10 / ncf .35 -> 0.1293.
HYBRID_WEIGHTS = {
    "item_cf": 0.45,
    "collaborative": 0.10,
    "content": 0.10,
    "ncf": 0.35,
}

CANDIDATE_POOL_SIZE = 200
COLD_START_MIN_INTERACTIONS = 3
RELEVANCE_RATING_THRESHOLD = 4

STRATEGY_HYBRID = "hybrid_fusion"
STRATEGY_COLD_USER = "cold_start_user_profile"
STRATEGY_UNKNOWN_USER = "global_popularity_fallback"


# =========================================================
# SERVICE CLASS
# =========================================================
class HybridRecommender:
    """Loaded recommendation service shared by the API and the dashboard."""

    def __init__(self, users_df, items_df, interactions_df, user_item_matrix,
                 predicted_ratings, popularity_lookup, content_similarity,
                 tfidf_matrix, tfidf_vectorizer, ncf_model, user_to_index,
                 item_to_index, item_similarity=None, weights=None,
                 feedback_type="implicit"):

        self.users_df = users_df
        self.items_df = items_df
        self.interactions_df = interactions_df

        self.user_item_matrix = user_item_matrix
        self.predicted_ratings = predicted_ratings
        self.popularity_lookup = popularity_lookup
        self.item_similarity = item_similarity

        self.content_similarity = content_similarity
        self.tfidf_matrix = tfidf_matrix
        self.tfidf_vectorizer = tfidf_vectorizer

        self.ncf_model = ncf_model
        self.user_to_index = user_to_index
        self.item_to_index = item_to_index

        self.weights = dict(HYBRID_WEIGHTS) if weights is None else weights
        self.feedback_type = feedback_type

        self.items_indexed = items_df.set_index("item_id")

        # Precompute per-user history once; filtering the full log per
        # request is the biggest avoidable cost in the serving path.
        self.user_seen = interactions_df.groupby("user_id")["item_id"].apply(set).to_dict()

    # =====================================================
    # CONSTRUCTION
    # =====================================================
    @classmethod
    def load(cls, feedback_type: str = "implicit") -> "HybridRecommender":
        """Load every artifact from data/processed."""
        # Defaults to implicit: the explicit model predicts ratings well
        # (RMSE 0.91) but scored NDCG@10 = 0.0000 for ranking.
        users_df, items_df, interactions_df = load_all()

        user_item_matrix = load_pickle("user_item_matrix.pkl")
        predicted_ratings = load_pickle("predicted_ratings.pkl")
        item_popularity = load_pickle("item_popularity_features.pkl")
        content_similarity = load_pickle("content_similarity.pkl")
        item_similarity = load_pickle("item_similarity.pkl")
        tfidf_matrix = load_pickle("tfidf_matrix.pkl")
        tfidf_vectorizer = load_pickle("tfidf_vectorizer.pkl")

        # NCF is optional but its absence must be loud: a previous revision
        # swallowed this and served with the deep signal silently at zero.
        try:
            ncf_model = load_ncf_model(feedback_type=feedback_type)
            user_to_index = load_pickle("ncf_user_to_index.pkl")
            item_to_index = load_pickle("ncf_item_to_index.pkl")
        except FileNotFoundError as exc:
            print("WARNING: NCF artifacts unavailable ({}).".format(exc))
            print("         Running on collaborative + content signals only.")
            print("         Train with:  python models/ncf_recommender.py")
            ncf_model, user_to_index, item_to_index = None, {}, {}

        return cls(
            users_df=users_df,
            items_df=items_df,
            interactions_df=interactions_df,
            user_item_matrix=user_item_matrix,
            predicted_ratings=predicted_ratings,
            popularity_lookup=build_popularity_lookup(item_popularity),
            content_similarity=content_similarity,
            item_similarity=item_similarity,
            tfidf_matrix=tfidf_matrix,
            tfidf_vectorizer=tfidf_vectorizer,
            ncf_model=ncf_model,
            user_to_index=user_to_index,
            item_to_index=item_to_index,
            feedback_type=feedback_type,
        )

    # =====================================================
    # USER STATE
    # =====================================================
    def is_known_user(self, user_id: int) -> bool:
        """Is this user in the users table?"""
        return bool((self.users_df["user_id"] == user_id).any())

    def seen_items(self, user_id: int) -> set:
        """Items this user has already interacted with."""
        return self.user_seen.get(user_id, set())

    def user_interaction_count(self, user_id: int) -> int:
        """How many distinct items this user has touched."""
        return len(self.seen_items(user_id))

    def is_cold_start_user(self, user_id: int) -> bool:
        """Too little history to personalise from."""
        # A threshold, not a has-any-history check: one click produces
        # confidently wrong recommendations.
        return self.user_interaction_count(user_id) < COLD_START_MIN_INTERACTIONS

    # =====================================================
    # INDIVIDUAL SIGNALS
    # =====================================================
    def item_cf_scores(self, user_id: int) -> pd.Series:
        """Item-based CF scores - the strongest single signal (NDCG 0.1245)."""
        if self.item_similarity is None or user_id not in self.user_item_matrix.index:
            return pd.Series(dtype=float)

        user_ratings = self.user_item_matrix.loc[user_id]
        rated = user_ratings[user_ratings > 0]

        if rated.empty:
            return pd.Series(dtype=float)

        scores = pd.Series(
            self.item_similarity[rated.index].to_numpy() @ rated.to_numpy(),
            index=self.item_similarity.index,
        )
        scores = scores.drop(list(self.seen_items(user_id)), errors="ignore")

        return min_max_normalize(scores)

    def collaborative_scores(self, user_id: int) -> pd.Series:
        """Latent-factor (SVD) scores over unseen items."""
        if user_id not in self.predicted_ratings.index:
            return pd.Series(dtype=float)

        scores = self.predicted_ratings.loc[user_id].drop(
            list(self.seen_items(user_id)), errors="ignore"
        )
        return min_max_normalize(scores)

    def content_scores(self, user_id: int) -> pd.Series:
        """Content similarity between the user's taste profile and each item."""
        user_rows = self.interactions_df.loc[self.interactions_df["user_id"] == user_id]
        if user_rows.empty:
            return pd.Series(dtype=float)

        liked_mask = (
            (user_rows["rating"] >= RELEVANCE_RATING_THRESHOLD)
            | (user_rows["purchase"] == 1)
        ).fillna(False)

        liked_items = [i for i in user_rows.loc[liked_mask, "item_id"].unique()
                       if i in self.content_similarity.index]

        if not liked_items:
            return pd.Series(dtype=float)

        # Mean of liked items' similarity rows - equivalent to a TF-IDF
        # centroid but avoids a sparse matmul per request.
        profile_similarity = self.content_similarity.loc[liked_items].mean(axis=0)
        profile_similarity = profile_similarity.drop(
            list(self.seen_items(user_id)), errors="ignore"
        )

        return min_max_normalize(profile_similarity)

    def ncf_scores(self, user_id: int) -> pd.Series:
        """Deep-model scores over unseen items."""
        if self.ncf_model is None or user_id not in self.user_to_index:
            return pd.Series(dtype=float)

        scores = score_all_items(
            self.ncf_model,
            user_id,
            self.user_to_index,
            self.item_to_index,
            exclude_item_ids=list(self.seen_items(user_id)),
            feedback_type=self.feedback_type,
        )

        return min_max_normalize(scores)

    # =====================================================
    # SCORE FUSION
    # =====================================================
    def fuse(self, item_cf, collaborative, content, ncf) -> pd.DataFrame:
        """Weighted linear fusion, then popularity-balanced re-ranking."""
        available = {
            "item_cf": item_cf,
            "collaborative": collaborative,
            "content": content,
            "ncf": ncf,
        }
        active = {name: s for name, s in available.items() if not s.empty}

        if not active:
            return pd.DataFrame()

        # Renormalise over active signals, otherwise a user the NCF has never
        # seen gets scores scaled by the surviving weight mass and they stop
        # being comparable across users.
        weight_mass = sum(self.weights[name] for name in active)
        effective_weights = {name: self.weights[name] / weight_mass for name in active}

        all_item_ids = sorted(set().union(*(set(s.index) for s in active.values())))
        fused = pd.DataFrame(index=all_item_ids)

        for name, series in available.items():
            fused[name + "_score"] = series.reindex(fused.index).fillna(0.0)

        fused["hybrid_raw_score"] = sum(
            effective_weights[name] * fused[name + "_score"] for name in active
        )

        reranked = rerank_with_popularity_balance(
            fused["hybrid_raw_score"],
            self.popularity_lookup,
            candidate_pool=CANDIDATE_POOL_SIZE,
        )

        fused = fused.loc[reranked.index].join(
            reranked[["adjusted_score", "interaction_count", "is_long_tail",
                      "popularity_penalty", "long_tail_boost"]]
        )

        fused["active_signals"] = ", ".join(sorted(active))
        return fused.sort_values("adjusted_score", ascending=False)

    # =====================================================
    # COLD-START FALLBACK
    # =====================================================
    def cold_start_recommendations(self, user_id: int, top_n: int = 10) -> pd.DataFrame:
        """Two-tier fallback: registration profile, else global popularity."""
        user_row = self.users_df.loc[self.users_df["user_id"] == user_id]

        if user_row.empty:
            ranked = rerank_with_popularity_balance(
                self.popularity_lookup["popularity_ratio"],
                self.popularity_lookup,
                candidate_pool=CANDIDATE_POOL_SIZE,
            ).head(top_n)

            result = self.items_df.loc[self.items_df["item_id"].isin(ranked.index)].copy()
            result["hybrid_score"] = ranked["adjusted_score"].reindex(
                result["item_id"]).to_numpy()
            result["strategy"] = STRATEGY_UNKNOWN_USER
            return result.sort_values("hybrid_score", ascending=False)

        profile = user_row.iloc[0]
        result = recommend_for_new_user_by_profile(
            preferred_category=profile["preferred_category"],
            items_df=self.items_df,
            user_segment=profile["user_segment"],
            n_top=top_n,
        ).copy()

        result = result.rename(columns={"cold_start_score": "hybrid_score"})
        result["strategy"] = STRATEGY_COLD_USER
        return result

    # =====================================================
    # PUBLIC API
    # =====================================================
    def recommend(self, user_id: int, top_n: int = 10) -> pd.DataFrame:
        """Top-N recommendations with per-signal attribution and a strategy label."""
        if self.is_cold_start_user(user_id):
            return self.cold_start_recommendations(user_id, top_n)

        fused = self.fuse(
            self.item_cf_scores(user_id),
            self.collaborative_scores(user_id),
            self.content_scores(user_id),
            self.ncf_scores(user_id),
        )

        if fused.empty:
            return self.cold_start_recommendations(user_id, top_n)

        top = fused.head(top_n).copy()
        top.index.name = "item_id"
        top = top.reset_index()

        catalogue_columns = [
            c for c in ["item_id", "title", "category", "subcategory", "brand",
                        "price", "content_tags"]
            if c in self.items_df.columns
        ]
        top = top.merge(self.items_df[catalogue_columns], on="item_id", how="left")

        top = top.rename(columns={"adjusted_score": "hybrid_score"})
        # Always label the strategy so a caller can tell personalised from fallback.
        top["strategy"] = STRATEGY_HYBRID

        return top

    def similar_items(self, item_id: int, top_n: int = 10) -> pd.DataFrame:
        """Content-similar items. Works for cold items too."""
        return recommend_similar_items(
            item_id=item_id,
            items_df=self.items_df,
            similarity_df=self.content_similarity,
            n_top=top_n,
        )

    def item_details(self, item_id: int):
        """Catalogue row for one item, or None."""
        if item_id not in self.items_indexed.index:
            return None
        return self.items_indexed.loc[item_id]

    def user_details(self, user_id: int):
        """Profile row for one user, or None."""
        row = self.users_df.loc[self.users_df["user_id"] == user_id]
        if row.empty:
            return None
        return row.iloc[0]


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    """Demonstrate warm, cold-start and unknown-user paths."""
    print("Loading hybrid recommender ...")
    recommender = HybridRecommender.load()

    display_columns = ["item_id", "title", "category", "price", "item_cf_score",
                       "collaborative_score", "content_score", "ncf_score", "hybrid_score"]

    active_user = int(recommender.interactions_df["user_id"].value_counts().index[0])
    profile = recommender.user_details(active_user)

    print("\nWarm user {}: {}, prefers {}, {} interactions".format(
        active_user, profile["user_segment"], profile["preferred_category"],
        recommender.user_interaction_count(active_user)))

    warm = recommender.recommend(active_user, top_n=5)
    print("  strategy:", warm["strategy"].iloc[0])
    print(warm[[c for c in display_columns if c in warm.columns]].to_string(index=False))

    counts = recommender.users_df["user_id"].map(recommender.user_interaction_count)
    cold_user = int(recommender.users_df.loc[
        counts < COLD_START_MIN_INTERACTIONS, "user_id"].iloc[0])
    cold_profile = recommender.user_details(cold_user)

    print("\nCold-start user {}: {}, prefers {}, {} interactions".format(
        cold_user, cold_profile["user_segment"], cold_profile["preferred_category"],
        recommender.user_interaction_count(cold_user)))

    cold = recommender.recommend(cold_user, top_n=5)
    print("  strategy:", cold["strategy"].iloc[0])
    cold_columns = [c for c in ["item_id", "title", "category", "price", "hybrid_score"]
                    if c in cold.columns]
    print(cold[cold_columns].to_string(index=False))

    unknown = recommender.recommend(999999, top_n=3)
    print("\nUnknown user 999999 -> strategy:", unknown["strategy"].iloc[0])

    sample_item = int(recommender.items_df["item_id"].iloc[0])
    details = recommender.item_details(sample_item)
    print("\nItems similar to #{} ({}):".format(sample_item, details["title"]))
    print(recommender.similar_items(sample_item, 5)[
        ["item_id", "title", "category", "price", "similarity_score"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
