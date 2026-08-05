"""
Synthetic Data Validation & Advanced EDA
========================================
Enterprise-Grade Recommendation System with Deep Learning

Validates that the generated data actually behaves like real e-commerce data.

This is not decorative. The generator deliberately engineers popularity bias,
a long tail, sparsity, and cold-start cohorts; if those properties did not
materialise, every downstream model would be trained on a flat random dataset
and every metric reported later would be meaningless.

Produces the three plots the business case marks as mandatory:
    1. User activity distribution
    2. Item popularity distribution
    3. Interaction matrix sparsity

plus long-tail, funnel, price, and segment analysis.

Run:
    python notebooks/eda_validation.py
"""

import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")     # write files without needing a display

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))

from data_loader import load_all

FIGURES_PATH = os.path.join(BASE_DIR, "reports", "figures")
os.makedirs(FIGURES_PATH, exist_ok=True)

sns.set_theme(style="whitegrid")


def save_figure(name):
    path = os.path.join(FIGURES_PATH, name)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print("    saved:", name)


def main():
    users_df, items_df, interactions_df = load_all()

    num_users = len(users_df)
    num_items = len(items_df)
    num_interactions = len(interactions_df)

    print("=" * 66)
    print("SYNTHETIC DATA VALIDATION & ADVANCED EDA")
    print("=" * 66)
    print("\nUsers        : {:,}".format(num_users))
    print("Items        : {:,}".format(num_items))
    print("Interactions : {:,}".format(num_interactions))

    # Reindex against the full tables so zero-activity users and zero-sale
    # items are counted rather than silently dropped.
    user_activity = interactions_df.groupby("user_id").size().reindex(
        users_df["user_id"], fill_value=0)
    item_popularity = interactions_df.groupby("item_id").size().reindex(
        items_df["item_id"], fill_value=0)

    # =====================================================
    # 1. USER ACTIVITY DISTRIBUTION  (mandatory plot)
    # =====================================================
    print("\n[1] USER ACTIVITY DISTRIBUTION")
    print("    mean   : {:.1f}".format(user_activity.mean()))
    print("    median : {:.0f}".format(user_activity.median()))
    print("    max    : {:,}".format(user_activity.max()))
    print("    users with 0 interactions : {:,}".format(int((user_activity == 0).sum())))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    sns.histplot(user_activity, bins=60, kde=False, ax=axes[0], color="#4C72B0")
    axes[0].set_title("User Activity Distribution")
    axes[0].set_xlabel("Interactions per user")
    axes[0].set_ylabel("Number of users")

    # Log scale exposes the power-law shape that a linear axis hides.
    sns.histplot(user_activity[user_activity > 0], bins=60, log_scale=(True, True),
                 ax=axes[1], color="#4C72B0")
    axes[1].set_title("User Activity (log-log) - power law check")
    axes[1].set_xlabel("Interactions per user (log)")
    axes[1].set_ylabel("Number of users (log)")

    save_figure("01_user_activity_distribution.png")

    # =====================================================
    # 2. ITEM POPULARITY DISTRIBUTION  (mandatory plot)
    # =====================================================
    sorted_popularity = item_popularity.sort_values(ascending=False)
    head_size = max(1, int(num_items * 0.10))
    head_share = sorted_popularity.head(head_size).sum() / num_interactions

    print("\n[2] ITEM POPULARITY DISTRIBUTION")
    print("    top 10% of items hold {:.1%} of all interactions".format(head_share))
    print("    most popular item : {:,} interactions".format(sorted_popularity.iloc[0]))
    print("    median item       : {:.0f} interactions".format(item_popularity.median()))
    print("    items with 0      : {:,}".format(int((item_popularity == 0).sum())))

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

    sns.histplot(item_popularity, bins=60, ax=axes[0], color="#DD8452")
    axes[0].set_title("Item Popularity Distribution")
    axes[0].set_xlabel("Interactions per item")
    axes[0].set_ylabel("Number of items")

    axes[1].plot(range(1, len(sorted_popularity) + 1), sorted_popularity.values,
                 color="#DD8452")
    axes[1].set_title("Long Tail - rank vs demand")
    axes[1].set_xlabel("Item rank by popularity")
    axes[1].set_ylabel("Interactions")

    # Lorenz curve: the standard way to show concentration of demand.
    cumulative = np.cumsum(sorted_popularity.values) / num_interactions
    item_fraction = np.arange(1, len(sorted_popularity) + 1) / num_items

    axes[2].plot(item_fraction, cumulative, color="#C44E52", label="Observed")
    axes[2].plot([0, 1], [0, 1], "--", color="grey", label="Perfectly uniform")
    axes[2].fill_between(item_fraction, cumulative, item_fraction, alpha=0.2, color="#C44E52")
    axes[2].set_title("Demand Concentration (Lorenz curve)")
    axes[2].set_xlabel("Fraction of catalogue")
    axes[2].set_ylabel("Fraction of interactions")
    axes[2].legend()

    save_figure("02_item_popularity_distribution.png")

    # Gini coefficient quantifies the concentration in a single number.
    #
    # The curve plotted above is sorted DESCENDING, which is the conventional
    # "top X% of the catalogue drives Y% of demand" view. The Gini integral,
    # however, is defined against the ASCENDING Lorenz curve - the one that sits
    # below the diagonal. Reusing the descending curve integrates the area above
    # the diagonal instead and returns the value with the sign flipped.
    ascending = np.sort(item_popularity.values)
    lorenz = np.cumsum(ascending) / ascending.sum()
    population = np.arange(1, len(ascending) + 1) / len(ascending)

    gini = 1 - 2 * np.trapezoid(lorenz, population)
    print("    Gini coefficient of demand: {:.3f}  (0 = uniform, 1 = one item takes all)".format(gini))

    # =====================================================
    # 3. INTERACTION MATRIX SPARSITY  (mandatory plot)
    # =====================================================
    sparsity = 1 - (num_interactions / (num_users * num_items))

    print("\n[3] INTERACTION MATRIX SPARSITY")
    print("    possible cells : {:,}".format(num_users * num_items))
    print("    observed       : {:,}".format(num_interactions))
    print("    sparsity       : {:.4%}".format(sparsity))

    # Sample the most active users and most popular items - a random sample of a
    # 99% sparse matrix renders as an empty rectangle and shows nothing at all.
    top_users = user_activity.nlargest(60).index
    top_items = item_popularity.nlargest(60).index

    sample = interactions_df.loc[
        interactions_df["user_id"].isin(top_users)
        & interactions_df["item_id"].isin(top_items)
    ]

    matrix = sample.pivot_table(index="user_id", columns="item_id",
                                values="rating", aggfunc="mean")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(matrix.notna(), cmap="Blues", cbar_kws={"label": "Interaction present"},
                ax=axes[0])
    axes[0].set_title("Interaction Sparsity - 60 most active users x 60 most popular items")
    axes[0].set_xlabel("Items")
    axes[0].set_ylabel("Users")

    sns.heatmap(matrix, cmap="RdYlGn", vmin=1, vmax=5,
                cbar_kws={"label": "Rating"}, ax=axes[1])
    axes[1].set_title("Rating Values (dense corner of the matrix)")
    axes[1].set_xlabel("Items")
    axes[1].set_ylabel("Users")

    save_figure("03_interaction_matrix_sparsity.png")

    observed_density = matrix.notna().to_numpy().mean()
    print("    density in the densest 60x60 corner : {:.1%}".format(observed_density))
    print("    (even here it is mostly empty - this is what makes CF hard)")

    # =====================================================
    # 4. COLD START
    # =====================================================
    cold_users = int((user_activity < 3).sum())
    cold_items = int((item_popularity < 3).sum())

    print("\n[4] COLD START")
    print("    users with < 3 interactions : {:,} ({:.1%})".format(
        cold_users, cold_users / num_users))
    print("    items with < 3 interactions : {:,} ({:.1%})".format(
        cold_items, cold_items / num_items))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    user_bands = pd.cut(user_activity, [-1, 0, 2, 10, 40, 1e9],
                        labels=["0", "1-2", "3-10", "11-40", "40+"])
    user_bands.value_counts().sort_index().plot(kind="bar", ax=axes[0], color="#55A868")
    axes[0].set_title("Users by activity band")
    axes[0].set_xlabel("Interactions")
    axes[0].set_ylabel("Users")
    axes[0].tick_params(axis="x", rotation=0)

    item_bands = pd.cut(item_popularity, [-1, 0, 2, 10, 100, 1e9],
                        labels=["0", "1-2", "3-10", "11-100", "100+"])
    item_bands.value_counts().sort_index().plot(kind="bar", ax=axes[1], color="#8172B3")
    axes[1].set_title("Items by demand band")
    axes[1].set_xlabel("Interactions")
    axes[1].set_ylabel("Items")
    axes[1].tick_params(axis="x", rotation=0)

    save_figure("04_cold_start_distribution.png")

    # =====================================================
    # 5. ENGAGEMENT FUNNEL
    # =====================================================
    clicks = int(interactions_df["click"].sum())
    carts = int(interactions_df["add_to_cart"].sum())
    purchases = int(interactions_df["purchase"].sum())

    print("\n[5] ENGAGEMENT FUNNEL")
    print("    clicks       : {:,}".format(clicks))
    print("    add to cart  : {:,} ({:.1%} of clicks)".format(carts, carts / clicks))
    print("    purchases    : {:,} ({:.1%} of clicks)".format(purchases, purchases / clicks))
    print("    revenue      : INR {:,.0f}".format(interactions_df["revenue"].sum()))

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

    axes[0].barh(["Click", "Add to cart", "Purchase"], [clicks, carts, purchases],
                 color=["#4C72B0", "#DD8452", "#55A868"])
    axes[0].set_title("Engagement Funnel")
    axes[0].set_xlabel("Events")
    for i, v in enumerate([clicks, carts, purchases]):
        axes[0].text(v, i, " {:,}".format(v), va="center")

    sns.histplot(interactions_df["view_time_seconds"], bins=60, log_scale=(True, False),
                 ax=axes[1], color="#4C72B0")
    axes[1].set_title("Dwell Time Distribution")
    axes[1].set_xlabel("Seconds on product page (log)")

    rating_counts = interactions_df["rating"].value_counts().sort_index()
    axes[2].bar(rating_counts.index.astype(int), rating_counts.values, color="#C44E52")
    axes[2].set_title("Rating Distribution (mean {:.2f})".format(
        interactions_df["rating"].mean()))
    axes[2].set_xlabel("Stars")
    axes[2].set_ylabel("Ratings")

    save_figure("05_engagement_funnel.png")

    print("\n    Realism note: real marketplace ratings are J-shaped, skewed high.")
    print("    Observed mean {:.2f} with {:.0%} at 4-5 stars.".format(
        interactions_df["rating"].mean(),
        rating_counts.loc[[4.0, 5.0]].sum() / rating_counts.sum()))

    # =====================================================
    # 6. LATENT STRUCTURE - segment vs price
    # =====================================================
    print("\n[6] LATENT STRUCTURE (what the models must recover)")

    purchases_df = interactions_df.loc[interactions_df["purchase"] == 1].merge(
        items_df[["item_id", "price", "category"]], on="item_id"
    ).merge(
        users_df[["user_id", "user_segment", "preferred_category"]], on="user_id"
    )

    segment_price = purchases_df.groupby("user_segment")["price"].median().sort_values()
    print("\n    Median purchase price by segment:")
    for segment, price in segment_price.items():
        print("      {:<20} INR {:>9,.0f}".format(segment, price))

    category_match = (
        purchases_df["category"] == purchases_df["preferred_category"]
    ).mean()
    print("\n    Purchases inside the customer's declared category: {:.1%}".format(
        category_match))
    print("    (random would be ~10% with 10 categories - so preference is real)")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    segment_order = segment_price.index.tolist()
    sns.boxplot(data=purchases_df, x="user_segment", y="price", order=segment_order,
                ax=axes[0], showfliers=False, palette="viridis", hue="user_segment",
                legend=False)
    axes[0].set_yscale("log")
    axes[0].set_title("Purchase Price by Customer Segment")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Price (INR, log)")
    axes[0].tick_params(axis="x", rotation=30)

    category_price = items_df.groupby("category")["price"].median().sort_values()
    category_price.plot(kind="barh", ax=axes[1], color="#4C72B0")
    axes[1].set_xscale("log")
    axes[1].set_title("Median Price by Category")
    axes[1].set_xlabel("Price (INR, log)")
    axes[1].set_ylabel("")

    save_figure("06_latent_structure.png")

    # =====================================================
    # 7. TEMPORAL COVERAGE
    # =====================================================
    print("\n[7] TEMPORAL COVERAGE")
    print("    from {} to {}".format(
        interactions_df["timestamp"].min().date(),
        interactions_df["timestamp"].max().date()))

    monthly = interactions_df.set_index("timestamp").resample("ME").size()

    plt.figure(figsize=(12, 4))
    monthly.plot(color="#4C72B0", marker="o")
    plt.title("Interaction Volume Over Time (recency-weighted by design)")
    plt.xlabel("")
    plt.ylabel("Interactions per month")
    save_figure("07_temporal_coverage.png")

    last_90 = interactions_df.loc[
        interactions_df["timestamp"] > interactions_df["timestamp"].max() - pd.Timedelta(days=90)
    ]
    print("    last 90 days hold {:.1%} of all interactions".format(
        len(last_90) / num_interactions))
    print("    (this is the held-out test period for the time-based split)")

    # =====================================================
    # VERDICT
    # =====================================================
    print("\n" + "=" * 66)
    print("VALIDATION VERDICT")
    print("=" * 66)

    checks = [
        ("Volume >= 5,000 users", num_users >= 5000),
        ("Volume >= 2,000 items", num_items >= 2000),
        ("Volume >= 100,000 interactions", num_interactions >= 100000),
        ("Sparsity > 95% (realistic)", sparsity > 0.95),
        ("Popularity bias: top 10% hold > 40%", head_share > 0.40),
        ("Long tail present (Gini > 0.5)", gini > 0.5),
        ("Cold-start users present", cold_users > 0),
        ("Cold-start items present", cold_items > 0),
        ("Ratings skew high (mean > 3.5)", interactions_df["rating"].mean() > 3.5),
        ("Category preference is real (> 30%)", category_match > 0.30),
    ]

    for label, passed in checks:
        print("  [{}]  {}".format("PASS" if passed else "FAIL", label))

    if all(passed for _, passed in checks):
        print("\n  All checks passed - the dataset exhibits the properties the")
        print("  business case requires, and the models are being trained on")
        print("  something that behaves like a real platform.")

    print("\nFigures written to:", FIGURES_PATH)


if __name__ == "__main__":
    main()
