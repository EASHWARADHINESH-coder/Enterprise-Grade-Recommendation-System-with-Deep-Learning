"""Neural Collaborative Filtering (PyTorch) - the deep learning model."""

import copy
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))

from data_loader import load_all
from preprocessing import load_pickle, save_pickle


# =========================================================
# PATHS
# =========================================================
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed")


# =========================================================
# SETTINGS
# =========================================================
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

EMBEDDING_DIM = 32
HIDDEN_DIMS = [128, 64, 32]
DROPOUT = 0.25
BATCH_SIZE = 1024
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 4
VALID_RATIO = 0.2

# Negatives sampled per observed positive (ratio from the NCF paper).
NUM_NEGATIVES = 4

TEST_PERIOD_DAYS = 90

# Implicit model overrides: defaults peak at epoch 1 then overfit.
# Quality is capacity-bound at ~0.858 either way, but these give a
# real convergence curve over ~20 epochs.
IMPLICIT_WEIGHT_DECAY = 1e-3
IMPLICIT_DROPOUT = 0.40


def get_device() -> str:
    """Return cuda if available, else cpu."""
    return "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# ID ENCODING
# =========================================================
def encode_ids(interactions_df: pd.DataFrame, tag: str = "ncf"):
    """Map sparse IDs onto contiguous embedding indices and persist the maps."""
    encoded = interactions_df.copy()

    unique_users = sorted(encoded["user_id"].unique())
    unique_items = sorted(encoded["item_id"].unique())

    user_to_index = {uid: idx for idx, uid in enumerate(unique_users)}
    item_to_index = {iid: idx for idx, iid in enumerate(unique_items)}
    index_to_user = {idx: uid for uid, idx in user_to_index.items()}
    index_to_item = {idx: iid for iid, idx in item_to_index.items()}

    encoded["user_idx"] = encoded["user_id"].map(user_to_index)
    encoded["item_idx"] = encoded["item_id"].map(item_to_index)

    # Maps must persist: serving has to reproduce the same index or the
    # embedding row returned belongs to someone else.
    save_pickle(user_to_index, tag + "_user_to_index.pkl")
    save_pickle(item_to_index, tag + "_item_to_index.pkl")
    save_pickle(index_to_user, tag + "_index_to_user.pkl")
    save_pickle(index_to_item, tag + "_index_to_item.pkl")

    return encoded, user_to_index, item_to_index, index_to_user, index_to_item


# =========================================================
# SPLITTING
# =========================================================
def time_based_split(interactions_df: pd.DataFrame, test_days: int = TEST_PERIOD_DAYS):
    """Hold out the most recent test_days as the test period."""
    # A random split leaks the future and inflates every metric.
    interactions_copy = interactions_df.copy()
    interactions_copy["timestamp"] = pd.to_datetime(interactions_copy["timestamp"])

    cutoff = interactions_copy["timestamp"].max() - pd.Timedelta(days=test_days)

    train_df = interactions_copy.loc[interactions_copy["timestamp"] <= cutoff].copy()
    test_df = interactions_copy.loc[interactions_copy["timestamp"] > cutoff].copy()

    return train_df, test_df, cutoff


def train_valid_split(interactions_df: pd.DataFrame, valid_ratio: float = VALID_RATIO,
                      random_state: int = RANDOM_SEED):
    """Split the training period into train and validation folds."""
    shuffled = interactions_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    split_index = int(len(shuffled) * (1 - valid_ratio))
    return shuffled.iloc[:split_index].copy(), shuffled.iloc[split_index:].copy()


# =========================================================
# NEGATIVE SAMPLING
# =========================================================
def build_implicit_training_frame(encoded_df: pd.DataFrame, num_items: int,
                                  num_negatives: int = NUM_NEGATIVES,
                                  seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Build the implicit training set with sampled negatives."""
    # Without negatives the model is asked "given they clicked, did they buy?"
    # - a conversion problem. Retrieval needs contrast against the catalogue.
    # Measured: 53% accuracy / NDCG 0.0000 without, 86% / 0.0576 with.
    rng = np.random.default_rng(seed)

    positives = encoded_df.loc[encoded_df["implicit_feedback"] == 1, ["user_idx", "item_idx"]]

    # Exclude everything touched, including viewed-not-carted: those are
    # ambiguous, not confirmed negatives.
    seen_by_user = encoded_df.groupby("user_idx")["item_idx"].apply(set)

    negative_users = []
    negative_items = []

    for user_idx, group in positives.groupby("user_idx"):
        required = len(group) * num_negatives
        seen = seen_by_user.get(user_idx, set())

        candidates = rng.integers(0, num_items, size=int(required * 1.5) + 16)
        sampled = [int(c) for c in candidates if c not in seen][:required]

        negative_users.extend([user_idx] * len(sampled))
        negative_items.extend(sampled)

    positive_frame = pd.DataFrame({
        "user_idx": positives["user_idx"].to_numpy(),
        "item_idx": positives["item_idx"].to_numpy(),
        "label": 1.0,
    })
    negative_frame = pd.DataFrame({
        "user_idx": negative_users,
        "item_idx": negative_items,
        "label": 0.0,
    })

    combined = pd.concat([positive_frame, negative_frame], ignore_index=True)
    return combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)


# =========================================================
# DATASET
# =========================================================
class InteractionsDataset(Dataset):
    """Wraps encoded interactions as (user_idx, item_idx, label) tensors."""

    def __init__(self, df: pd.DataFrame, feedback_type: str = "explicit"):
        if "label" in df.columns:
            labels = df["label"].to_numpy(dtype=np.float32)
        elif feedback_type == "explicit":
            df = df.dropna(subset=["rating"])
            labels = df["rating"].to_numpy(dtype=np.float32)
        elif feedback_type == "implicit":
            labels = df["implicit_feedback"].to_numpy(dtype=np.float32)
        else:
            raise ValueError("feedback_type must be 'explicit' or 'implicit'.")

        self.users = torch.tensor(df["user_idx"].to_numpy(), dtype=torch.long)
        self.items = torch.tensor(df["item_idx"].to_numpy(), dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]


# =========================================================
# MODEL
# =========================================================
class NeuralCollaborativeFiltering(nn.Module):
    """Two embedding tables feeding an MLP over their concatenation."""

    def __init__(self, num_users: int, num_items: int, embedding_dim: int = EMBEDDING_DIM,
                 hidden_dims: list = None, dropout: float = DROPOUT):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = HIDDEN_DIMS

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        layers = []
        input_dim = embedding_dim * 2
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)
        self.output_layer = nn.Linear(input_dim, 1)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Small embedding init, Xavier for the MLP."""
        # Large random embeddings make the first epochs pure noise.
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, user_indices, item_indices):
        """Score a batch of (user, item) pairs."""
        # Concatenation, not element-wise product: a product hard-codes a
        # multiplicative interaction and reduces this to matrix factorisation.
        user_vec = self.user_embedding(user_indices)
        item_vec = self.item_embedding(item_indices)

        x = torch.cat([user_vec, item_vec], dim=1)
        x = self.mlp(x)
        return self.output_layer(x).squeeze(1)


# =========================================================
# EARLY STOPPING
# =========================================================
class EarlyStopping:
    """Stop when validation loss stops improving, restoring the best weights."""

    def __init__(self, patience: int = PATIENCE, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state = None
        self.best_epoch = 0

    def step(self, valid_loss: float, model: nn.Module, epoch: int) -> bool:
        """Record the epoch and report whether training should stop."""
        # Restoring matters as much as stopping: otherwise you keep the
        # weights from the last, worse epoch.
        if valid_loss < self.best_loss - self.min_delta:
            self.best_loss = valid_loss
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            return False

        self.counter += 1
        return self.counter >= self.patience


# =========================================================
# TRAINING
# =========================================================
def train_ncf_model(model: nn.Module, train_loader: DataLoader, valid_loader: DataLoader,
                    feedback_type: str = "explicit", learning_rate: float = LEARNING_RATE,
                    weight_decay: float = WEIGHT_DECAY, epochs: int = EPOCHS,
                    patience: int = PATIENCE, device: str = None, verbose: bool = True):
    """Train with Adam, weight decay and early stopping."""
    if device is None:
        device = get_device()

    model = model.to(device)

    if feedback_type == "explicit":
        criterion = nn.MSELoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)
    early_stopper = EarlyStopping(patience=patience)

    history = {"train_loss": [], "valid_loss": [], "epoch_seconds": []}

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        model.train()
        train_losses = []
        for users, items, labels in train_loader:
            users, items, labels = users.to(device), items.to(device), labels.to(device)

            optimizer.zero_grad()
            loss = criterion(model(users, items), labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # eval() disables dropout; no_grad() skips gradient tracking.
        model.eval()
        valid_losses = []
        with torch.no_grad():
            for users, items, labels in valid_loader:
                users, items, labels = users.to(device), items.to(device), labels.to(device)
                valid_losses.append(criterion(model(users, items), labels).item())

        avg_train = float(np.mean(train_losses))
        avg_valid = float(np.mean(valid_losses))
        elapsed = time.time() - epoch_start

        history["train_loss"].append(avg_train)
        history["valid_loss"].append(avg_valid)
        history["epoch_seconds"].append(elapsed)

        if verbose:
            print("  epoch {:02d} | train {:.4f} | valid {:.4f} | {:.1f}s".format(
                epoch, avg_train, avg_valid, elapsed))

        if early_stopper.step(avg_valid, model, epoch):
            if verbose:
                print("  early stopping at epoch {} (best was epoch {})".format(
                    epoch, early_stopper.best_epoch))
            break

    if early_stopper.best_state is not None:
        model.load_state_dict(early_stopper.best_state)

    history["best_epoch"] = early_stopper.best_epoch
    history["best_valid_loss"] = early_stopper.best_loss

    return model, history


# =========================================================
# EVALUATION
# =========================================================
def evaluate_explicit_model(model, data_loader, device: str = None) -> dict:
    """Prediction-error metrics for the explicit model."""
    if device is None:
        device = get_device()

    model.eval()
    model.to(device)

    preds, trues = [], []
    with torch.no_grad():
        for users, items, labels in data_loader:
            outputs = model(users.to(device), items.to(device)).cpu().numpy()
            preds.extend(outputs.tolist())
            trues.extend(labels.numpy().tolist())

    preds, trues = np.array(preds), np.array(trues)
    mse = float(np.mean((preds - trues) ** 2))

    return {
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(np.mean(np.abs(preds - trues))),
    }


def evaluate_implicit_model(model, data_loader, threshold: float = 0.5,
                            device: str = None) -> dict:
    """Classification metrics for the implicit model."""
    if device is None:
        device = get_device()

    model.eval()
    model.to(device)

    probs, trues = [], []
    with torch.no_grad():
        for users, items, labels in data_loader:
            logits = model(users.to(device), items.to(device))
            probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            trues.extend(labels.numpy().tolist())

    probs, trues = np.array(probs), np.array(trues)
    preds = (probs >= threshold).astype(int)

    true_positive = int(((preds == 1) & (trues == 1)).sum())
    predicted_positive = int((preds == 1).sum())
    actual_positive = int((trues == 1).sum())

    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / actual_positive if actual_positive else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "Accuracy": float((preds == trues).mean()),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
    }


# =========================================================
# INFERENCE
# =========================================================
def score_all_items(model, user_id: int, user_to_index: dict, item_to_index: dict,
                    exclude_item_ids: list = None, feedback_type: str = "explicit",
                    device: str = None) -> pd.Series:
    """Score every candidate item for one user in a single batched pass."""
    if device is None:
        device = get_device()

    if user_id not in user_to_index:
        return pd.Series(dtype=float)

    model.eval()
    model.to(device)

    exclude = set(exclude_item_ids or [])
    candidate_ids = [iid for iid in item_to_index if iid not in exclude]

    if not candidate_ids:
        return pd.Series(dtype=float)

    # One batched pass; per-item calls would be thousands of tiny forwards.
    item_tensor = torch.tensor([item_to_index[i] for i in candidate_ids],
                               dtype=torch.long, device=device)
    user_tensor = torch.full((len(candidate_ids),), user_to_index[user_id],
                             dtype=torch.long, device=device)

    with torch.no_grad():
        scores = model(user_tensor, item_tensor)
        if feedback_type == "implicit":
            scores = torch.sigmoid(scores)
        scores = scores.cpu().numpy()

    return pd.Series(scores, index=candidate_ids).sort_values(ascending=False)


def recommend_ncf(model, user_id: int, interactions_df: pd.DataFrame, items_df: pd.DataFrame,
                  user_to_index: dict, item_to_index: dict, top_n: int = 10,
                  feedback_type: str = "explicit", device: str = None) -> pd.DataFrame:
    """Top-N NCF recommendations joined back to the catalogue."""
    seen = interactions_df.loc[interactions_df["user_id"] == user_id, "item_id"].unique().tolist()

    scores = score_all_items(model, user_id, user_to_index, item_to_index,
                             exclude_item_ids=seen, feedback_type=feedback_type,
                             device=device)

    if scores.empty:
        return pd.DataFrame(columns=["item_id", "predicted_score", "title", "category", "price"])

    top = scores.head(top_n)
    columns = [c for c in ["item_id", "title", "category", "subcategory", "brand", "price"]
               if c in items_df.columns]

    result = pd.DataFrame({"item_id": top.index, "predicted_score": top.to_numpy()})
    return result.merge(items_df[columns], on="item_id", how="left")


# =========================================================
# PERSISTENCE
# =========================================================
def save_ncf_artifacts(model, history: dict, feedback_type: str, tag: str = "ncf") -> None:
    """Persist weights, history and the architecture spec."""
    # state_dict, not the model object: pickling the class means any later
    # refactor of this file silently breaks loading.
    torch.save(model.state_dict(),
               os.path.join(PROCESSED_DATA_PATH, tag + "_model_" + feedback_type + ".pt"))
    save_pickle(history, tag + "_history_" + feedback_type + ".pkl")

    # Read architecture off the model, not the constants, so an overridden
    # model reloads as what it actually is.
    dropout_layers = [m.p for m in model.mlp if isinstance(m, nn.Dropout)]
    hidden_dims = [m.out_features for m in model.mlp if isinstance(m, nn.Linear)]

    save_pickle(
        {
            "num_users": model.user_embedding.num_embeddings,
            "num_items": model.item_embedding.num_embeddings,
            "embedding_dim": model.user_embedding.embedding_dim,
            "hidden_dims": hidden_dims,
            "dropout": dropout_layers[0] if dropout_layers else DROPOUT,
            "feedback_type": feedback_type,
        },
        tag + "_architecture_" + feedback_type + ".pkl",
    )


def load_ncf_model(feedback_type: str = "explicit", tag: str = "ncf") -> nn.Module:
    """Rebuild the model from its saved spec and load the weights."""
    arch = load_pickle(tag + "_architecture_" + feedback_type + ".pkl")
    weights_path = os.path.join(PROCESSED_DATA_PATH,
                                tag + "_model_" + feedback_type + ".pt")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            "NCF weights not found: " + weights_path + "\n"
            "Train first:  python models/ncf_recommender.py"
        )

    model = NeuralCollaborativeFiltering(
        num_users=arch["num_users"],
        num_items=arch["num_items"],
        embedding_dim=arch["embedding_dim"],
        hidden_dims=arch["hidden_dims"],
        dropout=arch["dropout"],
    )
    model.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu")))
    model.eval()

    return model


# =========================================================
# TRAINING ENTRY POINT
# =========================================================
def train_and_save(feedback_type: str = "explicit", tag: str = "ncf"):
    """Train one variant end to end and save it."""
    users_df, items_df, interactions_df = load_all()

    print("\nTraining NCF ({} feedback) on {}".format(feedback_type, get_device()))

    encoded, user_to_index, item_to_index, _, _ = encode_ids(interactions_df, tag=tag)

    train_period, _, cutoff = time_based_split(encoded)
    print("  training period ends {} ({:,} interactions)".format(
        cutoff.date(), len(train_period)))

    if feedback_type == "implicit":
        # Sample negatives before the split so both folds share the ratio.
        sampled = build_implicit_training_frame(train_period, num_items=len(item_to_index))
        positives = int(sampled["label"].sum())
        print("  negative sampling: {:,} positives + {:,} negatives (1:{})".format(
            positives, len(sampled) - positives, NUM_NEGATIVES))
        train_df, valid_df = train_valid_split(sampled)
        dropout = IMPLICIT_DROPOUT
        weight_decay = IMPLICIT_WEIGHT_DECAY
    else:
        train_df, valid_df = train_valid_split(train_period)
        dropout = DROPOUT
        weight_decay = WEIGHT_DECAY

    train_dataset = InteractionsDataset(train_df, feedback_type)
    valid_dataset = InteractionsDataset(valid_df, feedback_type)
    print("  train {:,} | valid {:,}".format(len(train_dataset), len(valid_dataset)))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = NeuralCollaborativeFiltering(
        num_users=len(user_to_index),
        num_items=len(item_to_index),
        dropout=dropout,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print("  parameters: {:,}".format(total_params))

    start = time.time()
    model, history = train_ncf_model(model, train_loader, valid_loader,
                                     feedback_type=feedback_type,
                                     weight_decay=weight_decay)
    training_seconds = time.time() - start

    history["training_seconds"] = training_seconds
    history["total_parameters"] = total_params

    save_ncf_artifacts(model, history, feedback_type, tag=tag)

    if feedback_type == "explicit":
        metrics = evaluate_explicit_model(model, valid_loader)
    else:
        metrics = evaluate_implicit_model(model, valid_loader)

    print("\n  training time: {:.1f}s over {} epochs".format(
        training_seconds, len(history["train_loss"])))
    print("  validation metrics:", {k: round(v, 4) for k, v in metrics.items()})

    return model, history, metrics, user_to_index, item_to_index, items_df, interactions_df


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    """Train both variants and show sample recommendations."""
    for feedback_type in ("explicit", "implicit"):
        result = train_and_save(feedback_type)
        model, history, metrics, user_to_index, item_to_index, items_df, interactions_df = result

        sample_user = int(interactions_df["user_id"].value_counts().index[0])
        recs = recommend_ncf(model, sample_user, interactions_df, items_df,
                             user_to_index, item_to_index, top_n=5,
                             feedback_type=feedback_type)
        print("\n  Top-5 for user {}:".format(sample_user))
        print(recs.to_string(index=False))

    print("\nNCF artifacts saved to", PROCESSED_DATA_PATH)


if __name__ == "__main__":
    main()
