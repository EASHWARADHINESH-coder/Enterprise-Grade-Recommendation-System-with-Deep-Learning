"""Explainability Layer - why an item was recommended."""

import os
import sys

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from content_based_nlp import top_shared_terms
from preprocessing import load_pickle


# =========================================================
# SETTINGS
# =========================================================
RELEVANCE_RATING_THRESHOLD = 4

SIGNAL_LABELS = {
    "item_cf_score": "item-based collaborative filtering",
    "collaborative_score": "latent-factor collaborative filtering",
    "content_score": "content similarity",
    "ncf_score": "the deep learning model",
}

SIGNAL_MEANING = {
    "item_cf_score": "it is bought alongside products you have already rated highly",
    "collaborative_score": "customers with similar purchase histories rated this highly",
    "content_score": "it closely matches the attributes of items you already bought",
    "ncf_score": "the neural model predicts a strong preference from your interaction pattern",
}

SIGNAL_WEIGHT_KEYS = {
    "item_cf_score": "item_cf",
    "collaborative_score": "collaborative",
    "content_score": "content",
    "ncf_score": "ncf",
}


# =========================================================
# SIGNAL ATTRIBUTION
# =========================================================
def attribute_signals(recommendation_row: pd.Series, weights: dict) -> pd.DataFrame:
    """Decompose a hybrid score into each signal's actual contribution."""
    # Report weight x score, not the raw score: a high score on a low-weight
    # signal contributes little to the ranking.
    rows = []

    for column, weight_key in SIGNAL_WEIGHT_KEYS.items():
        score = float(recommendation_row.get(column, 0.0) or 0.0)
        weight = float(weights.get(weight_key, 0.0))
        rows.append({
            "signal": SIGNAL_LABELS[column],
            "raw_score": score,
            "weight": weight,
            "contribution": score * weight,
        })

    frame = pd.DataFrame(rows)
    total = frame["contribution"].sum()
    frame["contribution_share"] = frame["contribution"] / total if total > 0 else 0.0

    return frame.sort_values("contribution", ascending=False).reset_index(drop=True)


def dominant_signal(recommendation_row: pd.Series, weights: dict):
    """Return the (signal label, share) that drove this recommendation."""
    attribution = attribute_signals(recommendation_row, weights)

    if attribution.empty or attribution["contribution"].sum() == 0:
        return "popularity fallback", 0.0

    top = attribution.iloc[0]
    return str(top["signal"]), float(top["contribution_share"])


# =========================================================
# 1. WHY THIS ITEM
# =========================================================
def explain_recommendation(recommender, user_id: int, recommendation_row: pd.Series) -> str:
    """Build a human-readable justification for one recommended item."""
    # Every clause is conditional on a measured quantity - never a template.
    item_id = int(recommendation_row["item_id"])
    strategy = recommendation_row.get("strategy", "")

    item = recommender.item_details(item_id)
    user = recommender.user_details(user_id)

    if item is None:
        return "Recommended by the hybrid ranking model."

    if strategy == "cold_start_user_profile":
        segment = user["user_segment"] if user is not None else "your"
        return (
            f"You are new here, so this is based on your registration profile rather than "
            f"purchase history: it sits in {item['category']}, the category you selected, "
            f"and is priced at INR {item['price']:,.0f}, which fits the typical "
            f"{segment} budget. Recommendations will become personalised once you have "
            f"browsed a few items."
        )

    if strategy == "global_popularity_fallback":
        return (
            f"We have no profile for this account yet, so this shows our best-performing "
            f"{item['category']} item across all customers, adjusted so that popular "
            f"products do not crowd out everything else."
        )

    reasons = []

    signal_label, share = dominant_signal(recommendation_row, recommender.weights)
    for column, meaning in SIGNAL_MEANING.items():
        if SIGNAL_LABELS.get(column) == signal_label:
            reasons.append(f"{meaning} (driving {share:.0%} of this recommendation)")
            break

    for column, meaning in SIGNAL_MEANING.items():
        if SIGNAL_LABELS.get(column) == signal_label:
            continue
        if float(recommendation_row.get(column, 0.0) or 0.0) > 0.65:
            reasons.append(meaning)

    if user is not None and item["category"] == user["preferred_category"]:
        reasons.append(f"it is in {item['category']}, your stated category of interest")

    # Long-tail promotion is disclosed, not hidden.
    if recommendation_row.get("is_long_tail", 0) == 1:
        reasons.append("it is an under-exposed product we are surfacing for variety")

    if not reasons:
        return "Recommended by the combined collaborative, content, and deep learning ranking."

    return "Recommended because " + "; ".join(reasons) + "."


# =========================================================
# 2. SIMILAR ITEMS / USERS EVIDENCE
# =========================================================
def explain_via_similar_items(recommender, user_id: int, item_id: int,
                              n_evidence: int = 3) -> pd.DataFrame:
    """Show which of the user's own purchases most resemble this suggestion."""
    seen = recommender.seen_items(user_id)

    if not seen or item_id not in recommender.content_similarity.index:
        return pd.DataFrame(columns=["item_id", "title", "category", "similarity"])

    known = [i for i in seen if i in recommender.content_similarity.columns]
    if not known:
        return pd.DataFrame(columns=["item_id", "title", "category", "similarity"])

    similarities = recommender.content_similarity.loc[item_id, known].sort_values(ascending=False)
    top = similarities.head(n_evidence)

    evidence = []
    for evidence_item_id, similarity in top.items():
        details = recommender.item_details(int(evidence_item_id))
        if details is None:
            continue
        evidence.append({
            "item_id": int(evidence_item_id),
            "title": details["title"],
            "category": details["category"],
            "similarity": round(float(similarity), 4),
        })

    return pd.DataFrame(evidence)


def explain_via_similar_users(recommender, user_id: int, item_id: int,
                              n_neighbours: int = 50) -> dict:
    """How many similar customers rated this item well."""
    # 50 neighbours, not 5: at 99% sparsity a small neighbourhood almost
    # never overlaps on a specific item and the evidence comes back empty.
    empty = {"neighbours_considered": 0, "neighbours_who_liked": 0, "mean_rating": None}

    if user_id not in recommender.user_item_matrix.index:
        return empty
    if item_id not in recommender.user_item_matrix.columns:
        return empty

    try:
        user_similarity = load_pickle("user_similarity.pkl")
    except FileNotFoundError:
        return empty

    if user_id not in user_similarity.index:
        return empty

    neighbours = (
        user_similarity.loc[user_id]
        .drop(user_id, errors="ignore")
        .nlargest(n_neighbours)
    )

    neighbour_ratings = recommender.user_item_matrix.loc[neighbours.index, item_id]
    rated = neighbour_ratings[neighbour_ratings > 0]
    liked = rated[rated >= RELEVANCE_RATING_THRESHOLD]

    return {
        "neighbours_considered": int(len(neighbours)),
        "neighbours_who_rated": int(len(rated)),
        "neighbours_who_liked": int(len(liked)),
        "mean_rating": round(float(rated.mean()), 2) if len(rated) else None,
        "mean_similarity": round(float(neighbours.mean()), 4),
    }


def render_similar_user_evidence(evidence: dict) -> str:
    """Phrase the collaborative evidence, including when there is none."""
    # Empty evidence is expected: the re-ranker promotes long-tail items,
    # which are long-tail precisely because few people interacted with them.
    considered = evidence.get("neighbours_considered", 0)
    rated = evidence.get("neighbours_who_rated", 0)
    liked = evidence.get("neighbours_who_liked", 0)
    mean_rating = evidence.get("mean_rating")

    if considered == 0:
        return "No comparable customers available for this account yet."

    if rated == 0:
        return (
            f"None of your {considered} closest customer matches have rated this item - "
            f"it is a low-exposure product surfaced for catalogue variety, so the "
            f"recommendation rests on content and model signals rather than crowd evidence."
        )

    return (
        f"{liked} of the {rated} closest customer matches who rated this item scored it "
        f"{RELEVANCE_RATING_THRESHOLD} or above (mean {mean_rating}), out of {considered} "
        f"neighbours considered."
    )


# =========================================================
# 3. CONTENT SIMILARITY JUSTIFICATION
# =========================================================
def explain_content_similarity(recommender, source_item_id: int,
                               recommended_item_id: int, n_terms: int = 5) -> dict:
    """Expose the TF-IDF terms and attributes linking two items."""
    source = recommender.item_details(source_item_id)
    recommended = recommender.item_details(recommended_item_id)

    if source is None or recommended is None:
        return {"shared_terms": [], "similarity": None, "shared_attributes": []}

    similarity = None
    if (source_item_id in recommender.content_similarity.index
            and recommended_item_id in recommender.content_similarity.columns):
        similarity = round(
            float(recommender.content_similarity.loc[source_item_id, recommended_item_id]), 4
        )

    shared_terms = top_shared_terms(
        source_item_id,
        recommended_item_id,
        recommender.items_df,
        recommender.tfidf_vectorizer,
        recommender.tfidf_matrix,
        n_terms=n_terms,
    )

    shared_attributes = []
    for field in ("category", "subcategory", "brand"):
        if field in source and field in recommended and source[field] == recommended[field]:
            shared_attributes.append(f"{field.replace('_', ' ')}: {source[field]}")

    source_tags = set(str(source.get("content_tags", "")).split("|"))
    recommended_tags = set(str(recommended.get("content_tags", "")).split("|"))
    shared_tags = sorted(t for t in (source_tags & recommended_tags) if t)

    return {
        "similarity": similarity,
        "shared_terms": shared_terms,
        "shared_attributes": shared_attributes,
        "shared_tags": shared_tags,
    }


def explain_similar_item(recommender, source_item_id: int, recommended_item_id: int) -> str:
    """Render the content-similarity justification as a sentence."""
    detail = explain_content_similarity(recommender, source_item_id, recommended_item_id)
    source = recommender.item_details(source_item_id)

    if source is None or detail["similarity"] is None:
        return "Similar based on product description and attributes."

    parts = [f"Content similarity to '{source['title']}' is {detail['similarity']:.2f}"]

    if detail["shared_attributes"]:
        parts.append("shared " + ", ".join(detail["shared_attributes"]))

    if detail["shared_tags"]:
        parts.append("common attributes: " + ", ".join(detail["shared_tags"][:4]))

    if detail["shared_terms"]:
        parts.append("matching description terms: " + ", ".join(detail["shared_terms"][:4]))

    return "; ".join(parts) + "."


# =========================================================
# FULL EXPLANATION BUNDLE
# =========================================================
def build_full_explanation(recommender, user_id: int, recommendation_row: pd.Series) -> dict:
    """Assemble all three mandated explanation types as structured data."""
    item_id = int(recommendation_row["item_id"])

    evidence_items = explain_via_similar_items(recommender, user_id, item_id)
    content_detail = {}

    if not evidence_items.empty:
        content_detail = explain_content_similarity(
            recommender, int(evidence_items.iloc[0]["item_id"]), item_id
        )

    return {
        "item_id": item_id,
        "summary": explain_recommendation(recommender, user_id, recommendation_row),
        "signal_attribution": attribute_signals(
            recommendation_row, recommender.weights).to_dict("records"),
        "similar_items_evidence": evidence_items.to_dict("records"),
        "similar_users_evidence": explain_via_similar_users(recommender, user_id, item_id),
        "content_justification": content_detail,
    }


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    """Demonstrate the three explanation types for warm and cold users."""
    from hybrid_recommender import HybridRecommender

    print("Loading recommender for explainability demo ...")
    recommender = HybridRecommender.load()

    warm_user = int(recommender.interactions_df["user_id"].value_counts().index[0])
    profile = recommender.user_details(warm_user)

    print("\n" + "=" * 70)
    print("USER {} - {}, prefers {}".format(
        warm_user, profile["user_segment"], profile["preferred_category"]))
    print("=" * 70)

    recommendations = recommender.recommend(warm_user, top_n=3)

    for _, row in recommendations.iterrows():
        print("\n--- {}  (INR {:,.0f}, {}) ---".format(
            row["title"], row["price"], row["category"]))

        print("\n[1] WHY THIS ITEM")
        print("   ", explain_recommendation(recommender, warm_user, row))

        print("\n[2] SIGNAL ATTRIBUTION")
        for _, signal in attribute_signals(row, recommender.weights).iterrows():
            bar = "#" * int(signal["contribution_share"] * 30)
            print("    {:<38} {:>6.1%} {}".format(
                signal["signal"], signal["contribution_share"], bar))

        print("\n[3] YOUR SIMILAR PURCHASES")
        evidence = explain_via_similar_items(recommender, warm_user, int(row["item_id"]))
        if evidence.empty:
            print("    (no comparable prior purchases)")
        else:
            for _, item in evidence.iterrows():
                print("    {:.3f}  {} ({})".format(
                    item["similarity"], item["title"], item["category"]))

        print("\n[4] SIMILAR-USER EVIDENCE")
        neighbours = explain_via_similar_users(recommender, warm_user, int(row["item_id"]))
        print("   ", render_similar_user_evidence(neighbours))

        print("\n[5] CONTENT JUSTIFICATION")
        if not evidence.empty:
            print("   ", explain_similar_item(
                recommender, int(evidence.iloc[0]["item_id"]), int(row["item_id"])))

    counts = recommender.users_df["user_id"].map(recommender.user_interaction_count)
    cold_user = int(recommender.users_df.loc[counts < 3, "user_id"].iloc[0])
    cold_profile = recommender.user_details(cold_user)

    print("\n" + "=" * 70)
    print("COLD-START USER {} - {}, prefers {}".format(
        cold_user, cold_profile["user_segment"], cold_profile["preferred_category"]))
    print("=" * 70)

    for _, row in recommender.recommend(cold_user, top_n=2).iterrows():
        print("\n--- {} ---".format(row["title"]))
        print("   ", explain_recommendation(recommender, cold_user, row))


if __name__ == "__main__":
    main()
