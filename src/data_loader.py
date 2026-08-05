"""
Data Loading Layer
==================
Enterprise-Grade Recommendation System with Deep Learning

Single entry point for reading the three raw tables. Paths are computed
relative to this file, so the project runs from any folder or drive without
editing a hardcoded path.
"""

import os

import pandas as pd


# =========================================================
# PATHS
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw")

USERS_FILE = os.path.join(RAW_DATA_PATH, "users.csv")
ITEMS_FILE = os.path.join(RAW_DATA_PATH, "items.csv")
INTERACTIONS_FILE = os.path.join(RAW_DATA_PATH, "interactions.csv")


def check_file(path, name):
    """Fail with an actionable message rather than a bare pandas error."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            name + " not found at " + path + "\n"
            "Generate it first:  python synthetic_data/generate_synthetic_data.py"
        )


def load_users():
    """Load the users table (user_id, age, gender, location, user_segment, ...)."""
    check_file(USERS_FILE, "users.csv")
    return pd.read_csv(USERS_FILE)


def load_items():
    """Load the catalogue (item_id, category, title, description, price, content_tags, ...)."""
    check_file(ITEMS_FILE, "items.csv")
    return pd.read_csv(ITEMS_FILE)


def load_interactions():
    """
    Load the interaction log.

    `timestamp` is parsed to datetime here so that every downstream consumer
    gets a real datetime and the time-based train/test split cannot silently
    degrade into a string comparison.
    """
    check_file(INTERACTIONS_FILE, "interactions.csv")
    return pd.read_csv(INTERACTIONS_FILE, parse_dates=["timestamp"])


def load_all():
    """Convenience loader returning (users, items, interactions)."""
    return load_users(), load_items(), load_interactions()


if __name__ == "__main__":
    users_df, items_df, interactions_df = load_all()

    print("Users        :", users_df.shape)
    print("Items        :", items_df.shape)
    print("Interactions :", interactions_df.shape)

    print("\nUsers sample:")
    print(users_df.head())

    print("\nItems sample:")
    print(items_df[["item_id", "title", "category", "price"]].head())

    print("\nInteractions sample:")
    print(interactions_df.head())
