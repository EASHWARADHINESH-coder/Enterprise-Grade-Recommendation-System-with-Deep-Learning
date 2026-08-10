"""Application Layer - FastAPI Recommendation Service (transport only)."""

import os
import sys
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))
sys.path.append(os.path.join(BASE_DIR, "models"))

from explainability import (
    attribute_signals,
    build_full_explanation,
    explain_recommendation,
    explain_similar_item,
    explain_via_similar_items,
    explain_via_similar_users,
    render_similar_user_evidence,
)
from hybrid_recommender import HybridRecommender


# =========================================================
# SETTINGS
# =========================================================
SERVICE_TITLE = "Enterprise-Grade Recommendation System with Deep Learning"
SERVICE_VERSION = "3.0.0"
DOMAIN = "E-Commerce / Retail"

MAX_TOP_N = 50
MAX_BATCH_USERS = 100

# Populated at startup by the lifespan handler.
recommender = None


# =========================================================
# RESPONSE MODELS
# Pydantic also validates output: a null where a price is expected
# fails here rather than reaching the storefront.
# =========================================================
class SignalContribution(BaseModel):
    signal: str
    raw_score: float
    weight: float
    contribution: float
    contribution_share: float


class RecommendationItem(BaseModel):
    item_id: int
    title: str
    category: str
    brand: str | None = None
    price: float
    score: float = Field(..., description="Final hybrid score after re-ranking")
    explanation: str
    is_long_tail: bool | None = None
    signal_breakdown: list[SignalContribution] | None = None


class RecommendationResponse(BaseModel):
    user_id: int
    strategy: str = Field(..., description="hybrid_fusion | cold_start_user_profile | global_popularity_fallback")
    top_n: int
    user_segment: str | None = None
    interaction_count: int
    recommendations: list[RecommendationItem]

    # Business context: what this recommendation set is worth if it converts.
    total_basket_value: float = Field(..., description="Sum of recommended item prices, INR")
    average_price: float


class SimilarItem(BaseModel):
    item_id: int
    title: str
    category: str
    brand: str | None = None
    price: float
    similarity_score: float
    explanation: str


class SimilarItemsResponse(BaseModel):
    item_id: int
    source_title: str
    source_category: str
    top_n: int
    similar_items: list[SimilarItem]


class UserProfileResponse(BaseModel):
    user_id: int
    age: int
    gender: str
    location: str
    user_segment: str
    preferred_category: str
    interaction_count: int
    is_cold_start: bool


class ItemDetailResponse(BaseModel):
    item_id: int
    title: str
    category: str
    subcategory: str | None = None
    brand: str | None = None
    price: float
    description: str
    content_tags: list[str]
    interaction_count: int
    is_long_tail: bool


class BatchRequest(BaseModel):
    user_ids: list[int] = Field(..., min_length=1, max_length=MAX_BATCH_USERS)
    top_n: int = Field(10, ge=1, le=MAX_TOP_N)


class BatchResponse(BaseModel):
    requested: int
    served: int
    results: dict[str, list[int]]


# =========================================================
# HELPERS
# =========================================================
def require_ready():
    """Guard every route: a half-loaded service must fail loudly, not silently."""
    if recommender is None:
        raise HTTPException(
            status_code=503,
            detail="Recommendation service is still initialising or failed to load artifacts.",
        )


def safe_float(value, default=0.0):
    return default if value is None or pd.isna(value) else float(value)


def safe_str(value, default=None):
    return default if value is None or pd.isna(value) else str(value)


def build_recommendation_items(user_id, frame, include_breakdown):
    """Convert the recommender's DataFrame into validated response objects."""
    items = []

    for _, row in frame.iterrows():
        breakdown = None
        if include_breakdown and "item_cf_score" in row.index:
            attribution = attribute_signals(row, recommender.weights)
            breakdown = [
                SignalContribution(
                    signal=str(entry["signal"]),
                    raw_score=round(float(entry["raw_score"]), 4),
                    weight=float(entry["weight"]),
                    contribution=round(float(entry["contribution"]), 4),
                    contribution_share=round(float(entry["contribution_share"]), 4),
                )
                for entry in attribution.to_dict("records")
            ]

        items.append(
            RecommendationItem(
                item_id=int(row["item_id"]),
                title=safe_str(row.get("title"), "Unknown"),
                category=safe_str(row.get("category"), "Unknown"),
                brand=safe_str(row.get("brand")),
                price=safe_float(row.get("price")),
                score=round(safe_float(row.get("hybrid_score")), 6),
                explanation=explain_recommendation(recommender, user_id, row),
                is_long_tail=(
                    bool(row["is_long_tail"]) if "is_long_tail" in row.index
                    and pd.notna(row["is_long_tail"]) else None
                ),
                signal_breakdown=breakdown,
            )
        )

    return items


# =========================================================
# LIFESPAN
# =========================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all model artifacts once at startup."""
    global recommender

    print("Loading recommendation artifacts ...")
    try:
        recommender = HybridRecommender.load()
        print("  users        : {:,}".format(len(recommender.users_df)))
        print("  items        : {:,}".format(len(recommender.items_df)))
        print("  interactions : {:,}".format(len(recommender.interactions_df)))
        print("  NCF loaded   : {}".format(recommender.ncf_model is not None))
        print("Service ready.")
    except FileNotFoundError as exc:
        # Do not start pretending to be healthy. A recommender missing its
        # artifacts should refuse requests, not serve degraded results silently.
        print("FAILED to load artifacts: {}".format(exc))
        print("Run the pipeline first - see README Quickstart.")
        recommender = None

    yield

    print("Shutting down recommendation service.")


app = FastAPI(
    title=SERVICE_TITLE,
    description=(
        "Hybrid recommendation service for an e-commerce catalogue. "
        "Combines item-based collaborative filtering, latent-factor CF, "
        "TF-IDF content similarity, and a PyTorch Neural Collaborative "
        "Filtering model, with explainable output and cold-start fallback."
    ),
    version=SERVICE_VERSION,
    lifespan=lifespan,
)


# =========================================================
# ROOT / HEALTH
# =========================================================
@app.get("/")
def root():
    return {
        "service": SERVICE_TITLE,
        "domain": DOMAIN,
        "version": SERVICE_VERSION,
        "status": "ready" if recommender is not None else "unavailable",
        "endpoints": [
            "/recommend/{user_id}",
            "/similar-items/{item_id}",
            "/users/{user_id}",
            "/items/{item_id}",
            "/recommend/batch",
            "/docs",
        ],
    }


@app.get("/health")
def health():
    if recommender is None:
        raise HTTPException(status_code=503, detail="Artifacts not loaded.")

    return {
        "status": "healthy",
        "users": len(recommender.users_df),
        "items": len(recommender.items_df),
        "ncf_available": recommender.ncf_model is not None,
        "fusion_weights": recommender.weights,
    }


# =========================================================
# RECOMMEND
# =========================================================
@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
def recommend(
    user_id: int,
    top_n: int = Query(10, ge=1, le=MAX_TOP_N, description="Number of items to return"),
    explain: bool = Query(True, description="Include per-signal score attribution"),
):
    """Personalised recommendations for a customer."""
    require_ready()

    if user_id < 1:
        raise HTTPException(status_code=422, detail="user_id must be a positive integer.")

    frame = recommender.recommend(user_id, top_n=top_n)

    if frame.empty:
        raise HTTPException(
            status_code=404,
            detail="No recommendations could be generated for user {}.".format(user_id),
        )

    items = build_recommendation_items(user_id, frame, include_breakdown=explain)
    profile = recommender.user_details(user_id)
    prices = [item.price for item in items]

    return RecommendationResponse(
        user_id=user_id,
        strategy=str(frame["strategy"].iloc[0]),
        top_n=len(items),
        user_segment=safe_str(profile["user_segment"]) if profile is not None else None,
        interaction_count=recommender.user_interaction_count(user_id),
        recommendations=items,
        total_basket_value=round(sum(prices), 2),
        average_price=round(sum(prices) / len(prices), 2) if prices else 0.0,
    )


# =========================================================
# SIMILAR ITEMS
# =========================================================
@app.get("/similar-items/{item_id}", response_model=SimilarItemsResponse)
def similar_items(
    item_id: int,
    top_n: int = Query(10, ge=1, le=MAX_TOP_N),
):
    """Content-similar products."""
    require_ready()

    source = recommender.item_details(item_id)
    if source is None:
        raise HTTPException(status_code=404, detail="item_id {} not found.".format(item_id))

    frame = recommender.similar_items(item_id, top_n=top_n)

    results = [
        SimilarItem(
            item_id=int(row["item_id"]),
            title=safe_str(row.get("title"), "Unknown"),
            category=safe_str(row.get("category"), "Unknown"),
            brand=safe_str(row.get("brand")),
            price=safe_float(row.get("price")),
            similarity_score=round(safe_float(row.get("similarity_score")), 4),
            explanation=explain_similar_item(recommender, item_id, int(row["item_id"])),
        )
        for _, row in frame.iterrows()
    ]

    return SimilarItemsResponse(
        item_id=item_id,
        source_title=safe_str(source["title"], "Unknown"),
        source_category=safe_str(source["category"], "Unknown"),
        top_n=len(results),
        similar_items=results,
    )


# =========================================================
# USER PROFILE
# =========================================================
@app.get("/users/{user_id}", response_model=UserProfileResponse)
def get_user(user_id: int):
    """Customer profile plus the engagement state that drives recommendation strategy."""
    require_ready()

    profile = recommender.user_details(user_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="user_id {} not found.".format(user_id))

    return UserProfileResponse(
        user_id=user_id,
        age=int(profile["age"]),
        gender=safe_str(profile["gender"], "Unknown"),
        location=safe_str(profile["location"], "Unknown"),
        user_segment=safe_str(profile["user_segment"], "Unknown"),
        preferred_category=safe_str(profile["preferred_category"], "Unknown"),
        interaction_count=recommender.user_interaction_count(user_id),
        is_cold_start=recommender.is_cold_start_user(user_id),
    )


# =========================================================
# ITEM DETAIL
# =========================================================
@app.get("/items/{item_id}", response_model=ItemDetailResponse)
def get_item(item_id: int):
    """Product detail, including whether it sits in the long tail."""
    require_ready()

    item = recommender.item_details(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="item_id {} not found.".format(item_id))

    interaction_count = 0
    is_long_tail = True
    if item_id in recommender.popularity_lookup.index:
        row = recommender.popularity_lookup.loc[item_id]
        interaction_count = int(row["interaction_count"])
        is_long_tail = bool(row["is_long_tail"])

    tags = [t for t in str(item.get("content_tags", "")).split("|") if t]

    return ItemDetailResponse(
        item_id=item_id,
        title=safe_str(item["title"], "Unknown"),
        category=safe_str(item["category"], "Unknown"),
        subcategory=safe_str(item.get("subcategory")),
        brand=safe_str(item.get("brand")),
        price=safe_float(item.get("price")),
        description=safe_str(item.get("description"), ""),
        content_tags=tags,
        interaction_count=interaction_count,
        is_long_tail=is_long_tail,
    )


# =========================================================
# EXPLANATION DETAIL
# =========================================================
@app.get("/explain/{user_id}/{item_id}")
def explain(user_id: int, item_id: int):
    """Full three-part explanation for one recommendation."""
    require_ready()

    if recommender.item_details(item_id) is None:
        raise HTTPException(status_code=404, detail="item_id {} not found.".format(item_id))

    frame = recommender.recommend(user_id, top_n=50)
    match = frame.loc[frame["item_id"] == item_id]

    if match.empty:
        # The item is not in this user's current top 50; still explain what
        # evidence exists rather than returning nothing useful.
        neighbours = explain_via_similar_users(recommender, user_id, item_id)
        return {
            "user_id": user_id,
            "item_id": item_id,
            "in_current_recommendations": False,
            "similar_items_evidence": explain_via_similar_items(
                recommender, user_id, item_id).to_dict("records"),
            "similar_users_evidence": neighbours,
            "similar_users_summary": render_similar_user_evidence(neighbours),
        }

    explanation = build_full_explanation(recommender, user_id, match.iloc[0])
    explanation["in_current_recommendations"] = True
    explanation["similar_users_summary"] = render_similar_user_evidence(
        explanation["similar_users_evidence"]
    )
    return explanation


# =========================================================
# BATCH PREDICTION
# =========================================================
@app.post("/recommend/batch", response_model=BatchResponse)
def recommend_batch(request: BatchRequest):
    """Score many customers in one call."""
    require_ready()

    results = {}
    for user_id in request.user_ids:
        frame = recommender.recommend(user_id, top_n=request.top_n)
        results[str(user_id)] = (
            [] if frame.empty else [int(i) for i in frame["item_id"].tolist()]
        )

    return BatchResponse(
        requested=len(request.user_ids),
        served=sum(1 for v in results.values() if v),
        results=results,
    )


if __name__ == "__main__":
    import uvicorn

    # PORT env var wins so the service can start when 8000 is already taken.
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="127.0.0.1", port=port)
