"""Data Loading Layer - reads the three raw tables."""

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


# =========================================================
# LOADERS
# =========================================================
def check_file(path: str, name: str) -> None:
    """Raise an actionable error if a raw file is missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            name + " not found at " + path + "\n"
            "Generate it first:  python synthetic_data/generate_synthetic_data.py"
        )


def load_users() -> pd.DataFrame:
    """Load the users table."""
    check_file(USERS_FILE, "users.csv")
    return pd.read_csv(USERS_FILE)


def load_items() -> pd.DataFrame:
    """Load the item catalogue."""
    check_file(ITEMS_FILE, "items.csv")
    return pd.read_csv(ITEMS_FILE)


def load_interactions() -> pd.DataFrame:
    """Load the interaction log."""
    check_file(INTERACTIONS_FILE, "interactions.csv")
    # parse_dates matters: as strings the time-based split compares
    # lexicographically and silently produces a wrong cutoff.
    return pd.read_csv(INTERACTIONS_FILE, parse_dates=["timestamp"])


def load_all():
    """Return (users, items, interactions)."""
    return load_users(), load_items(), load_interactions()


# =========================================================
# MAIN
# =========================================================
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
