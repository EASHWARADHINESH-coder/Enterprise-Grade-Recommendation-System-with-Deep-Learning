"""
Recommendation Evaluation (CRITICAL)
====================================
Enterprise-Grade Recommendation System with Deep Learning

Benchmarks every model on the same users, the same split, and the same metrics.

The two constraints the business case imposes are both structural, and both are
enforced here rather than assumed:

  TIME-BASED SPLIT     the final 90 days are held out. Training happens only on
                       what came before.
  NO INTERACTION LEAK  no model may see any test-period interaction, and the
                       candidate set excludes everything the user touched during
                       training. Without that second rule a model scores points
                       for "predicting" purchases it was shown.

Models compared:
    Popularity            non-personalised baseline
    Item-based CF         classical collaborative filtering
    SVD                   matrix factorisation
    Content (TF-IDF)      NLP-based
    NCF                   deep learning
    Hybrid                score fusion of the above

Run:
    python notebooks/recommendation_evaluation.py
"""

import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))
sys.path.append(os.path.join(BASE_DIR, "models"))

from baseline_recommenders import create_predicted_rating_matrix, create_svd_model
from collaborative_filtering import (
    build_popularity_lookup,
    min_max_normalize,
    rerank_with_popularity_balance,
)
from data_loader import load_all
from evaluation_metrics import (
    catalogue_coverage,
    evaluate_ranking,
    intra_list_diversity,
    long_tail_share,
    novelty,
)
from hybrid_recommender import HYBRID_WEIGHTS
from ncf_recommender import load_ncf_model, score_all_items, time_based_split
from preprocessing import (
    create_implicit_feedback_matrix,
    create_item_popularity_features,
    create_normalized_user_item_matrix,
    create_user_item_matrix,
    load_pickle,
)


# =========================================================
# PATHS & SETTINGS
# =========================================================
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed")

K_VALUES = [5, 10, 20]
TOP_N = max(K_VALUES)
TEST_PERIOD_DAYS = 90
RELEVANCE_RATING_THRESHOLD = 4

# Cap the evaluation cohort. Scoring the full catalogue for every one of ~5,800
# users across six models is slow and adds nothing: the metric standard errors
# are already small at this sample size.
MAX_EVAL_USERS = 800
RANDOM_SEED = 42

# Which NCF variant to rank with. This must be the implicit model, and the
# reason is the single most important modelling lesson in this project.
#
# The explicit model minimises MSE against observed star ratings. It is a good
# rating predictor (RMSE 0.91) but it has never once been shown an item a user
# did not interact with, so scoring the unseen catalogue is entirely out of
# distribution. What it falls back on is the global item bias, which is the
# same for everybody - measured here, it returned the same ~14 items to every
# user (catalogue coverage 0.006) and scored NDCG@10 = 0.0000.
#
# The implicit model is trained with sampled negatives drawn from the catalogue
# the user never touched, so discriminating "would engage" from "would not" IS
# its training objective. Same architecture, same data, same 800-user cohort:
#
#     explicit  NDCG@10 = 0.0000   coverage 0.006
#     implicit  NDCG@10 = 0.0576   coverage 0.013
#
# Accuracy on a rating-prediction metric says nothing about ranking quality.
NCF_RANKING_MODE = "implicit"


# =========================================================
# GROUND TRUTH
# =========================================================
def build_ground_truth(test_df):
    """
    Extract what each user actually found relevant during the test period.

    Relevance is a high rating OR a purchase. Purchases must be included: a
    bought-but-unrated item is unambiguous evidence of relevance, and dropping
    it would understate every model's recall by roughly the share of purchases
    that go unrated.

    Also returns graded relevance (the rating itself) so NDCG can distinguish a
    5-star hit from a 4-star one.
    """
    relevant_by_user = {}
    graded_by_user = {}

    is_relevant = (
        (test_df["rating"] >= RELEVANCE_RATING_THRESHOLD).fillna(False)
        | (test_df["purchase"] == 1)
    )
    relevant_rows = test_df.loc[is_relevant]

    for user_id, group in relevant_rows.groupby("user_id"):
        relevant_by_user[int(user_id)] = set(group["item_id"].tolist())

        # A purchase without a rating still counts; give it the threshold value
        # so it contributes real gain without outranking an explicit 5-star.
        grades = {}
        for item_id, rating in zip(group["item_id"], group["rating"]):
            grades[int(item_id)] = (
                float(rating) if pd.notna(rating) else float(RELEVANCE_RATING_THRESHOLD)
            )
        graded_by_user[int(user_id)] = grades

    return relevant_by_user, graded_by_user


# =========================================================
# TRAIN-ONLY MODEL ARTIFACTS
# =========================================================
def build_train_artifacts(train_df, items_df):
    """
    Rebuild every model from the training period alone.

    This is the step that actually enforces no-leakage. Reusing the artifacts in
    data/processed would be far quicker, but those were fitted on the full
    history including the test window, so every metric computed against them
    would be inflated. The models must be refitted on train-only data.
    """
    print("  rebuilding models on the training period only ...")

    user_item_matrix = create_user_item_matrix(train_df)
    normalized_matrix = create_normalized_user_item_matrix(train_df)
    implicit_matrix = create_implicit_feedback_matrix(train_df)
    item_popularity = create_item_popularity_features(train_df, items_df)

    popularity_lookup = build_popularity_lookup(item_popularity)

    svd, user_factors, item_factors = create_svd_model(normalized_matrix)
    predicted_ratings = create_predicted_rating_matrix(
        normalized_matrix, user_factors, item_factors
    )

    item_similarity = pd.DataFrame(
        cosine_similarity(user_item_matrix.T).astype(np.float32),
        index=user_item_matrix.columns,
        columns=user_item_matrix.columns,
    )

    popularity_ranking = (
        item_popularity.sort_values("interaction_count", ascending=False)["item_id"].tolist()
    )

    return {
        "user_item_matrix": user_item_matrix,
        "implicit_matrix": implicit_matrix,
        "item_popularity": item_popularity,
        "popularity_lookup": popularity_lookup,
        "predicted_ratings": predicted_ratings,
        "item_similarity": item_similarity,
        "popularity_ranking": popularity_ranking,
    }


# =========================================================
# MODEL SCORERS
# =========================================================
def recommend_popularity(user_id, artifacts, seen, top_n=TOP_N):
    """Non-personalised baseline: most-interacted items the user has not seen."""
    return [i for i in artifacts["popularity_ranking"] if i not in seen][:top_n]


def recommend_item_cf(user_id, artifacts, seen, top_n=TOP_N):
    """Classical item-based collaborative filtering."""
    matrix = artifacts["user_item_matrix"]
    similarity = artifacts["item_similarity"]

    if user_id not in matrix.index:
        return []

    user_ratings = matrix.loc[user_id]
    rated = user_ratings[user_ratings > 0]

    if rated.empty:
        return []

    scores = pd.Series(
        similarity[rated.index].to_numpy() @ rated.to_numpy(),
        index=similarity.index,
    )
    scores = scores.drop(list(seen), errors="ignore")

    return scores.nlargest(top_n).index.tolist()


def recommend_svd_model(user_id, artifacts, seen, top_n=TOP_N):
    """Matrix factorisation."""
    predicted = artifacts["predicted_ratings"]

    if user_id not in predicted.index:
        return []

    scores = predicted.loc[user_id].drop(list(seen), errors="ignore")
    return scores.nlargest(top_n).index.tolist()


def get_liked_items(user_id, train_df, valid_index):
    """Items the user rated highly or purchased, during the training period."""
    user_rows = train_df.loc[train_df["user_id"] == user_id]

    if user_rows.empty:
        return []

    liked_mask = (
        (user_rows["rating"] >= RELEVANCE_RATING_THRESHOLD).fillna(False)
        | (user_rows["purchase"] == 1)
    )
    return [i for i in user_rows.loc[liked_mask, "item_id"].unique() if i in valid_index]


def recommend_content(user_id, content_similarity, train_df, seen, top_n=TOP_N):
    """
    Content-based TF-IDF recommendations.

    The content similarity matrix is derived purely from item metadata, so it
    carries no interaction leakage. Only the user's liked-item set is restricted
    to the training period.
    """
    liked = get_liked_items(user_id, train_df, content_similarity.index)

    if not liked:
        return []

    scores = content_similarity.loc[liked].mean(axis=0).drop(list(seen), errors="ignore")
    return scores.nlargest(top_n).index.tolist()


def recommend_ncf_model(user_id, ncf_model, user_to_index, item_to_index, seen, top_n=TOP_N):
    """Deep learning recommendations."""
    if ncf_model is None:
        return []

    scores = score_all_items(
        ncf_model, user_id, user_to_index, item_to_index,
        exclude_item_ids=list(seen), feedback_type=NCF_RANKING_MODE,
    )

    if scores.empty:
        return []
    return scores.head(top_n).index.tolist()


def recommend_hybrid_eval(user_id, artifacts, content_similarity, train_df,
                          ncf_model, user_to_index, item_to_index, seen, top_n=TOP_N):
    """
    Hybrid fusion using train-period artifacts only.

    Mirrors HybridRecommender.fuse, but rebuilt against the training-period
    models so the benchmark stays leak-free. The weights are identical, so the
    comparison against the individual signals is fair.
    """
    # ---- Item-based CF (the strongest single signal) ----
    item_cf = pd.Series(dtype=float)
    matrix = artifacts["user_item_matrix"]
    if user_id in matrix.index:
        user_ratings = matrix.loc[user_id]
        rated = user_ratings[user_ratings > 0]
        if not rated.empty:
            similarity = artifacts["item_similarity"]
            raw = pd.Series(
                similarity[rated.index].to_numpy() @ rated.to_numpy(),
                index=similarity.index,
            ).drop(list(seen), errors="ignore")
            item_cf = min_max_normalize(raw)

    # ---- Latent-factor collaborative (SVD) ----
    predicted = artifacts["predicted_ratings"]
    if user_id in predicted.index:
        collaborative = min_max_normalize(
            predicted.loc[user_id].drop(list(seen), errors="ignore")
        )
    else:
        collaborative = pd.Series(dtype=float)

    # ---- Content ----
    content = pd.Series(dtype=float)
    liked = get_liked_items(user_id, train_df, content_similarity.index)
    if liked:
        content = min_max_normalize(
            content_similarity.loc[liked].mean(axis=0).drop(list(seen), errors="ignore")
        )

    # ---- Deep learning ----
    ncf = pd.Series(dtype=float)
    if ncf_model is not None:
        raw = score_all_items(
            ncf_model, user_id, user_to_index, item_to_index,
            exclude_item_ids=list(seen), feedback_type="explicit",
        )
        if not raw.empty:
            ncf = min_max_normalize(raw)

    available = {
        "item_cf": item_cf,
        "collaborative": collaborative,
        "content": content,
        "ncf": ncf,
    }
    active = {name: s for name, s in available.items() if not s.empty}

    if not active:
        return []

    weight_mass = sum(HYBRID_WEIGHTS[name] for name in active)
    all_items = sorted(set().union(*(set(s.index) for s in active.values())))

    fused = pd.Series(0.0, index=all_items)
    for name, series in active.items():
        weight = HYBRID_WEIGHTS[name] / weight_mass
        fused = fused.add(weight * series.reindex(all_items).fillna(0.0), fill_value=0.0)

    reranked = rerank_with_popularity_balance(fused, artifacts["popularity_lookup"])
    return reranked.head(top_n).index.tolist()


# =========================================================
# EVALUATION DRIVER
# =========================================================
def main():
    print("=" * 68)
    print("RECOMMENDATION EVALUATION - time-based split, no interaction leakage")
    print("=" * 68)

    users_df, items_df, interactions_df = load_all()

    # ---- Split -----------------------------------------------------------
    train_df, test_df, cutoff = time_based_split(interactions_df, TEST_PERIOD_DAYS)

    print("\nSPLIT")
    print("  cutoff date        : {}".format(cutoff.date()))
    print("  train interactions : {:,}  ({} to {})".format(
        len(train_df), train_df["timestamp"].min().date(), train_df["timestamp"].max().date()))
    print("  test interactions  : {:,}  ({} to {})".format(
        len(test_df), test_df["timestamp"].min().date(), test_df["timestamp"].max().date()))

    # Leakage assertions rather than assumptions.
    assert test_df["timestamp"].min() > cutoff, "LEAKAGE: test row at or before cutoff"
    assert train_df["timestamp"].max() <= cutoff, "LEAKAGE: train row after cutoff"
    print("  leakage check      : PASSED (train and test strictly separated in time)")

    # ---- Ground truth ----------------------------------------------------
    relevant_by_user, graded_by_user = build_ground_truth(test_df)
    print("\nGROUND TRUTH")
    print("  users with >=1 relevant test item : {:,}".format(len(relevant_by_user)))
    print("  mean relevant items per user      : {:.1f}".format(
        np.mean([len(v) for v in relevant_by_user.values()])))

    # ---- Rebuild models on train only ------------------------------------
    print("\nMODEL PREPARATION")
    artifacts = build_train_artifacts(train_df, items_df)

    # Content similarity is metadata-only, so the stored matrix is leak-free.
    content_similarity = load_pickle("content_similarity.pkl")

    # The saved NCF was trained on the training period only
    # (see ncf_recommender.train_and_save), so it is safe to reuse here.
    try:
        ncf_model = load_ncf_model(NCF_RANKING_MODE)
        user_to_index = load_pickle("ncf_user_to_index.pkl")
        item_to_index = load_pickle("ncf_item_to_index.pkl")
        print("  NCF loaded ({} variant, trained on training period only)".format(
            NCF_RANKING_MODE))
    except FileNotFoundError:
        ncf_model, user_to_index, item_to_index = None, {}, {}
        print("  WARNING: NCF unavailable - omitted from the benchmark")

    # ---- Evaluation cohort -----------------------------------------------
    rng = np.random.default_rng(RANDOM_SEED)
    candidate_users = sorted(relevant_by_user.keys())

    if len(candidate_users) > MAX_EVAL_USERS:
        eval_users = sorted(
            rng.choice(candidate_users, size=MAX_EVAL_USERS, replace=False).tolist()
        )
    else:
        eval_users = candidate_users

    print("  evaluation cohort  : {:,} users".format(len(eval_users)))

    # Each user's training-period history - the candidate exclusion set.
    train_seen = train_df.groupby("user_id")["item_id"].apply(set).to_dict()

    # ---- Generate recommendations ----------------------------------------
    models = {
        "Popularity": lambda u, s: recommend_popularity(u, artifacts, s),
        "Item-based CF": lambda u, s: recommend_item_cf(u, artifacts, s),
        "SVD": lambda u, s: recommend_svd_model(u, artifacts, s),
        "Content (TF-IDF)": lambda u, s: recommend_content(u, content_similarity, train_df, s),
        "NCF (Deep Learning)": lambda u, s: recommend_ncf_model(
            u, ncf_model, user_to_index, item_to_index, s),
        "Hybrid": lambda u, s: recommend_hybrid_eval(
            u, artifacts, content_similarity, train_df,
            ncf_model, user_to_index, item_to_index, s),
    }

    print("\nGENERATING RECOMMENDATIONS")
    recommendations_by_model = {}
    timings = {}

    for model_name, recommend_fn in models.items():
        start = time.time()
        per_user = {}

        for user_id in eval_users:
            seen = train_seen.get(user_id, set())
            per_user[user_id] = recommend_fn(user_id, seen)

        elapsed = time.time() - start
        recommendations_by_model[model_name] = per_user
        timings[model_name] = elapsed

        produced = sum(1 for r in per_user.values() if r)
        print("  {:<22} {:>6.1f}s  ({:,}/{:,} users served)".format(
            model_name, elapsed, produced, len(eval_users)))

    # ---- Ranking metrics --------------------------------------------------
    print("\n" + "=" * 68)
    print("RANKING METRICS")
    print("=" * 68)

    all_results = []
    for model_name, per_user in recommendations_by_model.items():
        result = evaluate_ranking(per_user, relevant_by_user, K_VALUES, graded_by_user)
        if result.empty:
            continue
        result.insert(0, "Model", model_name)
        all_results.append(result)

    results_df = pd.concat(all_results, ignore_index=True)

    for k in K_VALUES:
        print("\n--- K = {} ---".format(k))
        subset = results_df.loc[results_df["K"] == k].sort_values("NDCG@K", ascending=False)
        display = subset[["Model", "Precision@K", "Recall@K", "MAP@K", "NDCG@K", "HitRate@K"]]
        print(display.to_string(index=False, float_format=lambda v: "{:.4f}".format(v)))

    # ---- Beyond-accuracy metrics -----------------------------------------
    print("\n" + "=" * 68)
    print("BEYOND-ACCURACY METRICS  (at K=10)")
    print("=" * 68)

    popularity_counts = train_df.groupby("item_id").size()
    long_tail_items = set(
        artifacts["item_popularity"].loc[
            artifacts["item_popularity"]["is_long_tail"] == 1, "item_id"
        ].tolist()
    )

    diversity_rows = []

    for model_name, per_user in recommendations_by_model.items():
        top10 = [r[:10] for r in per_user.values() if r]

        if not top10:
            continue

        sample = top10[:150]     # intra-list diversity is O(k^2) per user
        diversity_rows.append({
            "Model": model_name,
            "Coverage": catalogue_coverage(top10, len(items_df)),
            "LongTailShare": long_tail_share(top10, long_tail_items),
            "Novelty": novelty(top10, popularity_counts),
            "IntraListDiversity": float(np.mean([
                intra_list_diversity(r, content_similarity) for r in sample
            ])),
            "Seconds": timings[model_name],
        })

    diversity_df = pd.DataFrame(diversity_rows).sort_values("Coverage", ascending=False)
    print(diversity_df.to_string(index=False, float_format=lambda v: "{:.4f}".format(v)))

    # ---- Persist ----------------------------------------------------------
    results_path = os.path.join(PROCESSED_DATA_PATH, "recommendation_evaluation_results.csv")
    diversity_path = os.path.join(PROCESSED_DATA_PATH, "beyond_accuracy_results.csv")

    results_df.to_csv(results_path, index=False)
    diversity_df.to_csv(diversity_path, index=False)

    print("\nSaved:")
    print(" ", results_path)
    print(" ", diversity_path)

    # ---- Headline ---------------------------------------------------------
    at10 = results_df.loc[results_df["K"] == 10].sort_values("NDCG@K", ascending=False)
    best = at10.iloc[0]
    baseline = at10.loc[at10["Model"] == "Popularity"].iloc[0]

    print("\n" + "=" * 68)
    print("HEADLINE")
    print("=" * 68)
    print("  Best model by NDCG@10 : {} ({:.4f})".format(best["Model"], best["NDCG@K"]))
    print("  Popularity baseline   : {:.4f}".format(baseline["NDCG@K"]))

    if baseline["NDCG@K"] > 0:
        lift = (best["NDCG@K"] - baseline["NDCG@K"]) / baseline["NDCG@K"]
        print("  Lift over baseline    : {:+.1%}".format(lift))


if __name__ == "__main__":
    main()
