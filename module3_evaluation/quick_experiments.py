"""
quick_experiments.py - Fast experiment utilities for final presentation prep
===========================================================================

Usage:
    python -m module3_evaluation.quick_experiments
"""

import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
RANDOM_SEED = 42


def compute_metrics(y_true, y_pred):
    """Compute basic binary metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def load_splits():
    """Load train, val, and test splits."""
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    return train_df, val_df, test_df


def run_source_held_out_baseline():
    """Train a fresh baseline while holding out each source from training."""
    train_df, val_df, test_df = load_splits()
    train_pool = pd.concat([train_df, val_df], ignore_index=True)
    rows = []

    for source in sorted(test_df["source"].unique()):
        held_out_test = test_df[test_df["source"] == source].copy()
        train_subset = train_pool[train_pool["source"] != source].copy()

        if train_subset.empty or held_out_test.empty:
            continue

        tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words="english")
        x_train_tfidf = tfidf.fit_transform(train_subset["text"])
        x_test_tfidf = tfidf.transform(held_out_test["text"])

        x_train = hstack([x_train_tfidf, train_subset[["perplexity", "burstiness"]].values])
        x_test = hstack([x_test_tfidf, held_out_test[["perplexity", "burstiness"]].values])

        model = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_SEED)
        model.fit(x_train, train_subset["label"].values)
        preds = model.predict(x_test)

        rows.append({
            "held_out_source": source,
            "train_samples": int(len(train_subset)),
            "test_samples": int(len(held_out_test)),
            **compute_metrics(held_out_test["label"].values, preds),
        })

    result_df = pd.DataFrame(rows).sort_values("held_out_source").reset_index(drop=True)
    result_path = os.path.join(RESULTS_DIR, "source_held_out_baseline.csv")
    result_df.to_csv(result_path, index=False)
    print(f"Saved source-held-out baseline results: {result_path}")
    return result_df


def plot_source_held_out_baseline(result_df: pd.DataFrame):
    """Create a PPT-friendly chart for source-held-out F1."""
    if result_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(result_df["held_out_source"], result_df["f1"], color="#f97316")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 Score")
    ax.set_xlabel("Held-Out Source")
    ax.set_title("Source-Held-Out Baseline Performance")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=15)
    plt.tight_layout()

    save_path = os.path.join(RESULTS_DIR, "source_held_out_baseline.png")
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved source-held-out chart: {save_path}")
    return save_path


def plot_quick_challenge_chart():
    """Create a comparison chart from quick human-edited AI results."""
    quick_path = os.path.join(RESULTS_DIR, "quick_challenge_experiment.json")
    if not os.path.exists(quick_path):
        print("quick_challenge_experiment.json not found; skipping challenge chart.")
        return None

    with open(quick_path, encoding="utf-8") as f:
        payload = json.load(f)

    labels = ["Clean subset", "Human-edited AI"]
    baseline = [
        payload.get("baseline_subset", {}).get("f1", 0.0),
        payload.get("baseline_human_edited_ai", {}).get("f1", 0.0),
    ]
    hybrid = [
        payload.get("hybrid_subset", {}).get("f1", 0.0),
        payload.get("hybrid_human_edited_ai", {}).get("f1", 0.0),
    ]

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width / 2 for i in x], baseline, width=width, label="Baseline", color="#2563eb")
    ax.bar([i + width / 2 for i in x], hybrid, width=width, label="Hybrid", color="#16a34a")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 Score")
    ax.set_title("Robustness on Human-Edited AI Text")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    save_path = os.path.join(RESULTS_DIR, "quick_challenge_chart.png")
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved quick challenge chart: {save_path}")
    return save_path


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_df = run_source_held_out_baseline()
    plot_source_held_out_baseline(result_df)
    plot_quick_challenge_chart()


if __name__ == "__main__":
    main()
