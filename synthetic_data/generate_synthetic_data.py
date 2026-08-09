"""Synthetic Data Generation - creates the e-commerce dataset."""

import os
from datetime import timedelta

import numpy as np
import pandas as pd
from faker import Faker


# =========================================================
# PATHS
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw")

os.makedirs(RAW_DATA_PATH, exist_ok=True)

USERS_FILE = os.path.join(RAW_DATA_PATH, "users.csv")
ITEMS_FILE = os.path.join(RAW_DATA_PATH, "items.csv")
INTERACTIONS_FILE = os.path.join(RAW_DATA_PATH, "interactions.csv")


# =========================================================
# REPRODUCIBILITY
# =========================================================
RANDOM_SEED = 42

rng = np.random.default_rng(RANDOM_SEED)
fake = Faker("en_IN")
Faker.seed(RANDOM_SEED)

REFERENCE_DATE = pd.Timestamp("2026-06-30")


# =========================================================
# DATA VOLUME  (spec minimums: 5,000 / 2,000 / 100,000)
# =========================================================
NUM_USERS = 6000
NUM_ITEMS = 2500
NUM_INTERACTIONS = 150000


# =========================================================
# REALISM CONTROLS
# =========================================================

# Shifted power law: exposure(rank) = 1 / (rank + offset) ** alpha.
# The offset stops the head saturating - with a pure 1/rank law the top
# item's exposure exceeds the user base and it loses all signal.
ZIPF_ALPHA = 1.25
ZIPF_RANK_OFFSET = 25

# User activity is lognormal: a few power users, a long tail of light users.
USER_ACTIVITY_LOG_MEAN = 2.85
USER_ACTIVITY_LOG_SIGMA = 0.95
MIN_INTERACTIONS_PER_ACTIVE_USER = 3
MAX_INTERACTIONS_PER_ACTIVE_USER = 220

# Deliberate cold-start carve-outs.
COLD_START_USER_FRACTION = 0.08   # users with 0-2 interactions
COLD_START_ITEM_FRACTION = 0.10   # recently launched items with <3 interactions

# Probability a user picks from their preferred category rather than exploring.
CATEGORY_AFFINITY_STRENGTH = 0.62

# How strongly a user's segment pulls them toward their price band.
PRICE_AFFINITY_STRENGTH = 0.55

# Share of browse-only interactions that still receive an explicit star rating.
# Buyers rate far more often than browsers, so effective coverage lands higher.
RATING_COVERAGE = 0.65

# Tuned so ratings are J-shaped (mean ~3.9, skewed high). A symmetric
# distribution centred on 3 is the classic tell-tale of synthetic data.
RATING_INTERCEPT = 2.35
RATING_SLOPE = 3.15
RATING_NOISE_SD = 0.60

# Observation window for the interaction log (supports a time-based split).
HISTORY_DAYS = 540          # 18 months
RECENCY_BIAS = 1.6          # >1 concentrates interactions toward recent dates


# =========================================================
# E-COMMERCE CATALOGUE TAXONOMY
# Price bands differ per category so price is a real latent signal.
# =========================================================
CATALOGUE = {
    "Electronics": {
        "subcategories": [
            "Smartphones", "Laptops", "Headphones", "Smart Watches",
            "Cameras", "Tablets", "Gaming Consoles", "Speakers",
        ],
        "brands": [
            "Zentro", "Nexovia", "Aurelis", "Kravix", "Optiwave",
            "Lumitek", "Byteforge", "Volante",
        ],
        "price_range": (1999, 149999),
        "tags": [
            "wireless", "bluetooth", "noise-cancelling", "fast-charging",
            "high-resolution", "portable", "waterproof", "long-battery",
            "touchscreen", "voice-assistant", "gaming", "professional",
        ],
    },
    "Fashion": {
        "subcategories": [
            "Men's Clothing", "Women's Clothing", "Footwear", "Handbags",
            "Watches", "Sunglasses", "Ethnic Wear", "Activewear",
        ],
        "brands": [
            "Marlowe", "Vestira", "Urbanknot", "Silkroute", "Denimworks",
            "Cassia", "Northloom", "Verdant",
        ],
        "price_range": (299, 24999),
        "tags": [
            "cotton", "breathable", "slim-fit", "handcrafted", "seasonal",
            "casual", "formal", "sustainable", "leather", "printed",
            "lightweight", "everyday",
        ],
    },
    "Home & Kitchen": {
        "subcategories": [
            "Cookware", "Small Appliances", "Furniture", "Bedding",
            "Storage", "Lighting", "Dinnerware", "Home Decor",
        ],
        "brands": [
            "Hearthline", "Casaform", "Coppernest", "Brightloom",
            "Terracot", "Nordhaus", "Claypeak", "Everhome",
        ],
        "price_range": (199, 79999),
        "tags": [
            "non-stick", "stainless-steel", "space-saving", "dishwasher-safe",
            "energy-efficient", "modular", "handcrafted", "minimalist",
            "durable", "eco-friendly", "compact", "premium-finish",
        ],
    },
    "Beauty & Personal Care": {
        "subcategories": [
            "Skincare", "Haircare", "Fragrances", "Makeup",
            "Grooming Tools", "Bath & Body", "Men's Grooming", "Nail Care",
        ],
        "brands": [
            "Auralux", "Petalworks", "Dermivia", "Bloomessence",
            "Purecraft", "Velora", "Herbatone", "Glowmark",
        ],
        "price_range": (149, 12999),
        "tags": [
            "paraben-free", "dermatologist-tested", "organic", "hydrating",
            "anti-ageing", "sensitive-skin", "vegan", "cruelty-free",
            "long-lasting", "fragrance-free", "spf", "natural-extracts",
        ],
    },
    "Sports & Outdoors": {
        "subcategories": [
            "Fitness Equipment", "Cycling", "Camping Gear", "Team Sports",
            "Running", "Yoga", "Swimming", "Adventure Gear",
        ],
        "brands": [
            "Peakstride", "Ironvale", "Trailmark", "Kinetiq",
            "Summitworks", "Flexcore", "Rovena", "Altitude",
        ],
        "price_range": (399, 89999),
        "tags": [
            "durable", "weather-resistant", "lightweight", "adjustable",
            "anti-slip", "high-impact", "professional", "beginner-friendly",
            "compact", "shock-absorbing", "breathable", "ergonomic",
        ],
    },
    "Books & Media": {
        "subcategories": [
            "Fiction", "Non-Fiction", "Academic", "Children's Books",
            "Comics", "Audiobooks", "Magazines", "Self-Help",
        ],
        "brands": [
            "Inkhouse", "Paperloom", "Quillstone", "Verbatim",
            "Chapterworks", "Foliopress", "Margin", "Bindery",
        ],
        "price_range": (99, 4999),
        "tags": [
            "bestseller", "award-winning", "illustrated", "hardcover",
            "paperback", "beginner-friendly", "reference", "translated",
            "collectible", "annotated", "series", "critically-acclaimed",
        ],
    },
    "Toys & Games": {
        "subcategories": [
            "Board Games", "Building Blocks", "Puzzles", "Remote Control",
            "Educational Toys", "Dolls", "Outdoor Play", "Card Games",
        ],
        "brands": [
            "Playnest", "Brickworks", "Tinkerly", "Wondertoy",
            "Puzzlecraft", "Juniora", "Rompbox", "Cleverkid",
        ],
        "price_range": (149, 19999),
        "tags": [
            "educational", "age-3-plus", "age-8-plus", "family",
            "multiplayer", "battery-operated", "wooden", "non-toxic",
            "creative", "problem-solving", "collectible", "outdoor",
        ],
    },
    "Grocery & Gourmet": {
        "subcategories": [
            "Beverages", "Snacks", "Organic Foods", "Spices",
            "Baking", "Breakfast", "Dry Fruits", "Condiments",
        ],
        "brands": [
            "Harvestly", "Spicehaus", "Grainmill", "Orchardly",
            "Purefarm", "Roastcraft", "Nutriva", "Fieldnote",
        ],
        "price_range": (49, 4999),
        "tags": [
            "organic", "gluten-free", "no-preservatives", "cold-pressed",
            "high-protein", "sugar-free", "artisanal", "single-origin",
            "farm-sourced", "wholegrain", "vegan", "resealable",
        ],
    },
    "Automotive": {
        "subcategories": [
            "Car Accessories", "Bike Accessories", "Car Care", "Tools",
            "Tyres", "Electronics & GPS", "Safety", "Lubricants",
        ],
        "brands": [
            "Torqline", "Axlepro", "Roadgrip", "Motorix",
            "Gearhaus", "Drivemax", "Pistonworks", "Autoloom",
        ],
        "price_range": (199, 59999),
        "tags": [
            "universal-fit", "heavy-duty", "weatherproof", "easy-install",
            "certified", "anti-rust", "high-performance", "compact",
            "long-life", "impact-resistant", "all-terrain", "oem-grade",
        ],
    },
    "Health & Wellness": {
        "subcategories": [
            "Supplements", "Medical Devices", "Ayurveda", "Fitness Nutrition",
            "Personal Safety", "Vision Care", "Orthopaedic", "Wellness Tech",
        ],
        "brands": [
            "Vitalume", "Curaleaf", "Medipeak", "Herbwell",
            "Nutrify", "Wellspring", "Bioform", "Restora",
        ],
        "price_range": (199, 34999),
        "tags": [
            "clinically-tested", "sugar-free", "ayurvedic", "immunity",
            "doctor-recommended", "iso-certified", "plant-based", "sports-nutrition",
            "daily-use", "portable", "rechargeable", "hypoallergenic",
        ],
    },
}

CATEGORIES = list(CATALOGUE.keys())


# =========================================================
# USER SEGMENTS
# =========================================================
USER_SEGMENTS = {
    "Budget Shopper": {
        "share": 0.26, "price_percentile": 0.15,
        "purchase_propensity": 0.22, "rating_bias": -0.15, "avg_basket_size": 1.4,
    },
    "Value Seeker": {
        "share": 0.28, "price_percentile": 0.35,
        "purchase_propensity": 0.28, "rating_bias": 0.00, "avg_basket_size": 1.7,
    },
    "Mainstream Buyer": {
        "share": 0.24, "price_percentile": 0.55,
        "purchase_propensity": 0.33, "rating_bias": 0.10, "avg_basket_size": 2.0,
    },
    "Premium Buyer": {
        "share": 0.15, "price_percentile": 0.78,
        "purchase_propensity": 0.41, "rating_bias": 0.20, "avg_basket_size": 2.3,
    },
    "Luxury Enthusiast": {
        "share": 0.07, "price_percentile": 0.93,
        "purchase_propensity": 0.48, "rating_bias": 0.25, "avg_basket_size": 2.6,
    },
}

GENDERS = ["Male", "Female", "Other"]
GENDER_WEIGHTS = [0.48, 0.49, 0.03]

LOCATIONS = [
    "Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai", "Kolkata",
    "Pune", "Ahmedabad", "Jaipur", "Kochi", "Coimbatore", "Lucknow",
    "Indore", "Chandigarh", "Bhubaneswar", "Nagpur",
]


# =========================================================
# 1. USERS
# =========================================================
def generate_users(num_users=NUM_USERS):
    """Build the users table."""
    segment_names = list(USER_SEGMENTS.keys())
    segment_shares = np.array([USER_SEGMENTS[s]["share"] for s in segment_names])
    segment_shares = segment_shares / segment_shares.sum()

    segments = rng.choice(segment_names, size=num_users, p=segment_shares)

    # Age correlates with segment: premium/luxury buyers skew older, because a
    # flat 18-70 uniform age would carry no signal at all.
    segment_age_centre = {
        "Budget Shopper": 27,
        "Value Seeker": 32,
        "Mainstream Buyer": 37,
        "Premium Buyer": 43,
        "Luxury Enthusiast": 47,
    }
    ages = np.array([
        int(np.clip(rng.normal(segment_age_centre[s], 9), 18, 72))
        for s in segments
    ])

    genders = rng.choice(GENDERS, size=num_users, p=GENDER_WEIGHTS)
    locations = rng.choice(LOCATIONS, size=num_users)
    preferred_categories = rng.choice(CATEGORIES, size=num_users)

    signup_offsets = rng.integers(0, HISTORY_DAYS + 365, size=num_users)
    signup_dates = [
        (REFERENCE_DATE - timedelta(days=int(offset))).strftime("%Y-%m-%d")
        for offset in signup_offsets
    ]

    users_df = pd.DataFrame({
        "user_id": np.arange(1, num_users + 1),
        "name": [fake.name() for _ in range(num_users)],
        "age": ages,
        "gender": genders,
        "location": locations,
        "user_segment": segments,
        "preferred_category": preferred_categories,
        "signup_date": signup_dates,
    })

    return users_df


# =========================================================
# 2. ITEMS
# =========================================================
TITLE_ADJECTIVES = [
    "Pro", "Ultra", "Classic", "Prime", "Signature", "Everyday", "Elite",
    "Compact", "Max", "Essential", "Studio", "Advanced", "Lite", "Heritage",
]

USE_CASES = [
    "daily use", "professional use", "travel", "home use", "gifting",
    "outdoor use", "long-term use", "everyday convenience",
]

AUDIENCES = [
    "first-time buyers", "enthusiasts", "professionals", "families",
    "students", "frequent travellers", "value-conscious shoppers",
    "gift shoppers",
]


def generate_items(num_items=NUM_ITEMS):
    """Build the items table."""
    records = []

    # Uneven distribution across categories, since real catalogues are not
    # balanced across departments.
    category_weights = rng.dirichlet(np.ones(len(CATEGORIES)) * 6.0)
    category_assignments = rng.choice(CATEGORIES, size=num_items, p=category_weights)

    for idx, category in enumerate(category_assignments, start=1):
        spec = CATALOGUE[category]
        subcategory = str(rng.choice(spec["subcategories"]))
        brand = str(rng.choice(spec["brands"]))

        low, high = spec["price_range"]
        # Log-normal inside the band: most items cluster low, a few are premium.
        log_low, log_high = np.log(low), np.log(high)
        mu = log_low + 0.35 * (log_high - log_low)
        sigma = 0.45 * (log_high - log_low) / 3.0
        price = round(float(np.clip(np.exp(rng.normal(mu, sigma)), low, high)), 2)

        tags = list(rng.choice(spec["tags"], size=rng.integers(3, 6), replace=False))
        adjective = str(rng.choice(TITLE_ADJECTIVES))
        model_code = str(rng.integers(100, 999))

        title = brand + " " + adjective + " " + subcategory.rstrip("s") + " " + model_code

        use_case = str(rng.choice(USE_CASES))
        audience = str(rng.choice(AUDIENCES))
        feature_a, feature_b = tags[0], tags[1]

        description = (
            brand + " " + adjective + " " + subcategory + " built for " + use_case + ". "
            "This " + category + " product features " + feature_a.replace("-", " ") +
            " and " + feature_b.replace("-", " ") + " construction. "
            "Part of the " + brand + " " + subcategory + " range, it is designed for " +
            audience + " who want reliable " + subcategory.lower() + " performance. "
            "Key attributes: " + ", ".join(t.replace("-", " ") for t in tags) + "."
        )

        # Intrinsic quality: drives ratings independently of personal fit, so a
        # genuinely good product is rated well across segments.
        base_quality = round(float(np.clip(rng.beta(6, 2.5), 0.05, 0.99)), 4)

        records.append({
            "item_id": idx,
            "title": title,
            "category": category,
            "subcategory": subcategory,
            "brand": brand,
            "description": description,
            "price": price,
            "content_tags": "|".join(tags),
            "base_quality": base_quality,
        })

    items_df = pd.DataFrame(records)

    # ---- Popularity exposure (Zipf) --------------------------------------
    ranks = np.arange(1, num_items + 1)
    exposure = 1.0 / np.power(ranks + ZIPF_RANK_OFFSET, ZIPF_ALPHA)
    rng.shuffle(exposure)                       # decorrelate popularity from item_id
    items_df["exposure_weight"] = exposure

    # ---- Cold-start items -------------------------------------------------
    num_cold_items = int(num_items * COLD_START_ITEM_FRACTION)
    cold_positions = rng.choice(num_items, size=num_cold_items, replace=False)

    is_cold = np.zeros(num_items, dtype=bool)
    is_cold[cold_positions] = True
    items_df["is_cold_start_item"] = is_cold.astype(int)
    items_df.loc[is_cold, "exposure_weight"] *= 0.002

    # Cold-start items launched in the last 30 days; the rest spread out.
    launch_offsets = np.where(
        is_cold,
        rng.integers(0, 30, size=num_items),
        rng.integers(30, HISTORY_DAYS + 200, size=num_items),
    )
    items_df["launch_date"] = [
        (REFERENCE_DATE - timedelta(days=int(offset))).strftime("%Y-%m-%d")
        for offset in launch_offsets
    ]
    items_df["launch_days_ago"] = launch_offsets

    # Price percentile is used for segment price-affinity matching.
    items_df["price_percentile"] = items_df["price"].rank(pct=True)

    return items_df


# =========================================================
# 3. INTERACTIONS
# =========================================================
def draw_user_activity(num_users, target_interactions):
    """Draw per-user interaction counts from a log-normal distribution."""
    raw = rng.lognormal(USER_ACTIVITY_LOG_MEAN, USER_ACTIVITY_LOG_SIGMA, size=num_users)
    counts = np.clip(
        np.round(raw),
        MIN_INTERACTIONS_PER_ACTIVE_USER,
        MAX_INTERACTIONS_PER_ACTIVE_USER,
    ).astype(int)

    # Scale to land close to the requested interaction budget.
    scale = target_interactions / counts.sum()
    counts = np.clip(
        np.round(counts * scale),
        MIN_INTERACTIONS_PER_ACTIVE_USER,
        MAX_INTERACTIONS_PER_ACTIVE_USER,
    ).astype(int)

    return counts


def weighted_sample_without_replacement(log_weights, k):
    """Gumbel top-k trick."""
    keys = log_weights + rng.gumbel(size=log_weights.shape[0])
    return np.argpartition(-keys, k - 1)[:k]


def generate_interactions(users_df, items_df, target_interactions=NUM_INTERACTIONS):
    """Build the interactions table as a realistic e-commerce engagement funnel."""
    num_users = len(users_df)
    num_items = len(items_df)

    item_ids = items_df["item_id"].to_numpy()
    item_categories = items_df["category"].to_numpy()
    item_price_pct = items_df["price_percentile"].to_numpy()
    item_quality = items_df["base_quality"].to_numpy()
    item_prices = items_df["price"].to_numpy()
    item_launch_days_ago = items_df["launch_days_ago"].to_numpy()

    log_exposure = np.log(items_df["exposure_weight"].to_numpy())

    # ---- Per-user interaction budget -------------------------------------
    activity = draw_user_activity(num_users, target_interactions)

    num_cold_users = int(num_users * COLD_START_USER_FRACTION)
    cold_user_positions = rng.choice(num_users, size=num_cold_users, replace=False)
    activity[cold_user_positions] = rng.integers(0, 3, size=num_cold_users)

    user_ids = users_df["user_id"].to_numpy()
    user_pref_category = users_df["preferred_category"].to_numpy()
    user_segments = users_df["user_segment"].to_numpy()

    segment_price_pct = np.array([USER_SEGMENTS[s]["price_percentile"] for s in user_segments])
    segment_purchase_prop = np.array([USER_SEGMENTS[s]["purchase_propensity"] for s in user_segments])
    segment_rating_bias = np.array([USER_SEGMENTS[s]["rating_bias"] for s in user_segments])
    segment_basket = np.array([USER_SEGMENTS[s]["avg_basket_size"] for s in user_segments])

    all_user_idx = []
    all_item_pos = []
    all_affinity = []

    for u in range(num_users):
        k = int(activity[u])
        if k <= 0:
            continue
        k = min(k, num_items)

        # --- Preference weighting -----------------------------------------
        # Start from raw popularity exposure, then tilt toward the categories
        # and price band this specific user cares about.
        category_match = (item_categories == user_pref_category[u]).astype(float)
        category_multiplier = 1.0 + CATEGORY_AFFINITY_STRENGTH * 8.0 * category_match

        price_distance = np.abs(item_price_pct - segment_price_pct[u])
        price_multiplier = 1.0 + PRICE_AFFINITY_STRENGTH * 4.0 * np.exp(
            -(price_distance ** 2) / 0.06
        )

        log_weights = log_exposure + np.log(category_multiplier) + np.log(price_multiplier)
        chosen = weighted_sample_without_replacement(log_weights, k)

        # --- Affinity for the chosen pairs --------------------------------
        # This single latent score drives dwell time, cart, purchase and rating,
        # so the four signals stay mutually consistent (a user does not rate 5
        # stars something they bounced off in two seconds).
        affinity = (
            0.42 * category_match[chosen]
            + 0.28 * np.exp(-(price_distance[chosen] ** 2) / 0.06)
            + 0.30 * item_quality[chosen]
        )
        affinity = np.clip(affinity + rng.normal(0, 0.11, size=k), 0.0, 1.0)

        all_user_idx.append(np.full(k, u))
        all_item_pos.append(chosen)
        all_affinity.append(affinity)

    user_idx = np.concatenate(all_user_idx)
    item_pos = np.concatenate(all_item_pos)
    affinity = np.concatenate(all_affinity)
    n = len(user_idx)

    # ---- Funnel stage 1: click / dwell time -------------------------------
    click = np.ones(n, dtype=int)
    view_time = np.clip(rng.lognormal(np.log(12 + 190 * affinity), 0.55), 3, 1800).round(1)

    # ---- Funnel stage 2: add to cart --------------------------------------
    cart_prob = 1.0 / (1.0 + np.exp(-(-2.4 + 4.6 * affinity)))
    add_to_cart = (rng.random(n) < cart_prob).astype(int)

    # ---- Funnel stage 3: purchase (only reachable from cart) --------------
    purchase_prob = segment_purchase_prop[user_idx] * (0.35 + 1.25 * affinity)
    purchase = ((add_to_cart == 1) & (rng.random(n) < purchase_prob)).astype(int)

    quantity = np.where(
        purchase == 1,
        np.maximum(1, rng.poisson(segment_basket[user_idx])),
        0,
    )
    revenue = np.round(quantity * item_prices[item_pos], 2)

    # ---- Explicit feedback: star ratings ----------------------------------
    # Buyers are far more likely to leave a rating than browsers, which is why
    # ratings are dense on purchases and sparse elsewhere - as in reality.
    rating_prob = np.where(purchase == 1, 0.88, RATING_COVERAGE)
    has_rating = rng.random(n) < rating_prob

    raw_rating = (
        RATING_INTERCEPT
        + RATING_SLOPE * affinity
        + segment_rating_bias[user_idx]
        + rng.normal(0, RATING_NOISE_SD, size=n)
    )
    rating = np.where(has_rating, np.clip(np.round(raw_rating), 1, 5), np.nan)

    # ---- Timestamps --------------------------------------------------------
    # Beta(RECENCY_BIAS, 1) skews toward 1, concentrating activity in recent
    # months, which is what makes a time-based train/test split meaningful.
    recency_fraction = rng.beta(RECENCY_BIAS, 1.0, size=n)
    days_ago = (HISTORY_DAYS * (1.0 - recency_fraction)).astype(int)

    # An interaction can never predate the item's launch.
    days_ago = np.maximum(np.minimum(days_ago, item_launch_days_ago[item_pos]), 0)

    seconds_offset = rng.integers(0, 86400, size=n)
    timestamps = (
        pd.to_datetime(REFERENCE_DATE)
        - pd.to_timedelta(days_ago, unit="D")
        + pd.to_timedelta(seconds_offset, unit="s")
    )

    interactions_df = pd.DataFrame({
        "user_id": user_ids[user_idx],
        "item_id": item_ids[item_pos],
        "click": click,
        "view_time_seconds": view_time,
        "add_to_cart": add_to_cart,
        "purchase": purchase,
        "quantity": quantity,
        "revenue": revenue,
        "rating": rating,
        "timestamp": timestamps,
    })

    # Implicit feedback: the model-facing binary target. A user showed genuine
    # intent if they carted or bought.
    interactions_df["implicit_feedback"] = (
        (interactions_df["add_to_cart"] == 1) | (interactions_df["purchase"] == 1)
    ).astype(int)

    interactions_df = interactions_df.sort_values(["timestamp", "user_id"]).reset_index(drop=True)
    interactions_df["timestamp"] = interactions_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    return interactions_df


# =========================================================
# 4. VALIDATION SUMMARY
# =========================================================
def print_validation_summary(users_df, items_df, interactions_df):
    """Print the checks that prove the engineered properties actually landed."""
    num_users = len(users_df)
    num_items = len(items_df)
    num_interactions = len(interactions_df)

    sparsity = 1 - (num_interactions / (num_users * num_items))

    item_counts = interactions_df.groupby("item_id").size().reindex(
        items_df["item_id"], fill_value=0
    )
    user_counts = interactions_df.groupby("user_id").size().reindex(
        users_df["user_id"], fill_value=0
    )

    sorted_counts = item_counts.sort_values(ascending=False)
    head_size = max(1, int(num_items * 0.10))
    head_share = sorted_counts.head(head_size).sum() / num_interactions

    long_tail_items = int((item_counts <= item_counts.quantile(0.80)).sum())
    cold_items = int((item_counts < 3).sum())
    cold_users = int((user_counts < 3).sum())

    print("\n" + "=" * 62)
    print("SYNTHETIC DATA VALIDATION SUMMARY")
    print("=" * 62)

    print("\nVOLUME  (spec minimum: 5,000 / 2,000 / 100,000)")
    print("  Users                     : {:,}    {}".format(
        num_users, "PASS" if num_users >= 5000 else "FAIL"))
    print("  Items                     : {:,}    {}".format(
        num_items, "PASS" if num_items >= 2000 else "FAIL"))
    print("  Interactions              : {:,}   {}".format(
        num_interactions, "PASS" if num_interactions >= 100000 else "FAIL"))

    print("\nSPARSITY")
    print("  Possible user-item cells  : {:,}".format(num_users * num_items))
    print("  Observed interactions     : {:,}".format(num_interactions))
    print("  Matrix sparsity           : {:.4%}".format(sparsity))

    print("\nPOPULARITY BIAS  (engineered via Zipf alpha={})".format(ZIPF_ALPHA))
    print("  Share of interactions held by top 10% of items : {:.1%}".format(head_share))
    print("  Most-interacted item count                     : {:,}".format(sorted_counts.iloc[0]))
    print("  Median item interaction count                  : {:.0f}".format(item_counts.median()))

    print("\nLONG TAIL")
    print("  Items at or below the 80th percentile : {:,} ({:.1%} of catalogue)".format(
        long_tail_items, long_tail_items / num_items))

    print("\nCOLD START")
    print("  Users with < 3 interactions : {:,} ({:.1%})".format(cold_users, cold_users / num_users))
    print("  Items with < 3 interactions : {:,} ({:.1%})".format(cold_items, cold_items / num_items))

    print("\nUSER ACTIVITY")
    print("  Mean interactions per user   : {:.1f}".format(user_counts.mean()))
    print("  Median interactions per user : {:.0f}".format(user_counts.median()))
    print("  Max interactions per user    : {:,}".format(user_counts.max()))

    print("\nENGAGEMENT FUNNEL")
    total_clicks = int(interactions_df["click"].sum())
    total_cart = int(interactions_df["add_to_cart"].sum())
    total_purchase = int(interactions_df["purchase"].sum())
    print("  Clicks                : {:,}".format(total_clicks))
    print("  Add to cart           : {:,} ({:.1%} of clicks)".format(
        total_cart, total_cart / total_clicks))
    print("  Purchases             : {:,} ({:.1%} of clicks)".format(
        total_purchase, total_purchase / total_clicks))
    print("  Total revenue         : INR {:,.0f}".format(interactions_df["revenue"].sum()))

    rated = interactions_df["rating"].notna().sum()
    print("\nEXPLICIT FEEDBACK")
    print("  Rows carrying a rating : {:,} ({:.1%})".format(rated, rated / num_interactions))
    print("  Mean rating            : {:.2f}".format(interactions_df["rating"].mean()))
    print("  Rating distribution    :")
    for value, count in interactions_df["rating"].value_counts().sort_index().items():
        print("      {} star : {:>7,}  ({:.1%})".format(int(value), count, count / rated))

    print("\nTEMPORAL COVERAGE")
    print("  Earliest interaction : {}".format(interactions_df["timestamp"].min()))
    print("  Latest interaction   : {}".format(interactions_df["timestamp"].max()))

    print("\nPRICE BY CATEGORY (median)")
    medians = items_df.groupby("category")["price"].median().sort_values(ascending=False)
    for category, median_price in medians.items():
        print("  {:<26} INR {:>10,.0f}".format(category, median_price))

    print("\n" + "=" * 62)


# =========================================================
# MAIN
# =========================================================
def main():
    print("Generating synthetic e-commerce data (seed={}) ...".format(RANDOM_SEED))

    users_df = generate_users()
    print("  users        :", users_df.shape)

    items_df = generate_items()
    print("  items        :", items_df.shape)

    interactions_df = generate_interactions(users_df, items_df)
    print("  interactions :", interactions_df.shape)

    # Drop the internal helper column before persisting.
    items_out = items_df.drop(columns=["launch_days_ago"])

    users_df.to_csv(USERS_FILE, index=False)
    items_out.to_csv(ITEMS_FILE, index=False)
    interactions_df.to_csv(INTERACTIONS_FILE, index=False)

    print("\nWritten:")
    print(" ", USERS_FILE)
    print(" ", ITEMS_FILE)
    print(" ", INTERACTIONS_FILE)

    print_validation_summary(users_df, items_out, interactions_df)


if __name__ == "__main__":
    main()
