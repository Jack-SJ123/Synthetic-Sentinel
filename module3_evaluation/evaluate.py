"""
evaluate.py — Synthetic Sentinel Evaluation & Analysis
========================================================
Phase 3: Compute metrics, generate confusion matrix, probability distributions,
burstiness analysis, error analysis, and baseline comparison.

Usage:
    python evaluate.py
    python evaluate.py --model-path models/best_roberta.pt
"""

import os
import json
import argparse
import warnings
import random
import re

import numpy as np
import pandas as pd
import joblib

import torch
from torch.utils.data import DataLoader

from transformers import RobertaTokenizerFast

from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from module2_modeling.train_model import HybridDetector, TextDataset

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
DATASET_METADATA_PATH = os.path.join(DATA_DIR, "dataset_metadata.json")
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# Styling
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "font.family": "sans-serif",
    "font.size": 12,
})

HUMAN_COLOR = "#58a6ff"
AI_COLOR = "#f85149"
ACCENT = "#3fb950"


def load_model_and_data(model_path: str, max_len: int = 256):
    """Load the best hybrid model and test data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    tokenizer_path = os.path.join(MODEL_DIR, "tokenizer")
    if os.path.exists(tokenizer_path):
        tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path)
    else:
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    # Load model
    model = HybridDetector()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load test data
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    test_dataset = TextDataset(
        test_df["text"].tolist(), test_df["label"].tolist(),
        test_df["perplexity"].tolist(), test_df["burstiness"].tolist(),
        tokenizer, max_len,
    )
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    return model, tokenizer, test_loader, test_df, device


def get_predictions(model, loader, device):
    """Run inference and collect predictions + probabilities."""
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            perplexity = batch["perplexity"].to(device)
            burstiness = batch["burstiness"].to(device)
            labels = batch["label"]

            logits = model(input_ids, attention_mask, perplexity, burstiness)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # P(AI)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def predict_hybrid_dataframe(model, df: pd.DataFrame, tokenizer, device, max_len: int = 256):
    """Run hybrid inference on an arbitrary dataframe with standard columns."""
    dataset = TextDataset(
        df["text"].tolist(),
        df["label"].tolist(),
        df["perplexity"].tolist(),
        df["burstiness"].tolist(),
        tokenizer,
        max_len,
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    return get_predictions(model, loader, device)


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """Compute standard binary metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def plot_confusion_matrix(labels, preds, save_path):
    """Generate and save a confusion matrix heatmap."""
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Human", "AI"], yticklabels=["Human", "AI"],
        ax=ax, linewidths=0.5, linecolor="#30363d",
        annot_kws={"size": 18, "weight": "bold"},
    )
    ax.set_xlabel("Predicted Label", fontsize=14, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=14, fontweight="bold")
    ax.set_title("Confusion Matrix — Hybrid Detector", fontsize=16, fontweight="bold", pad=15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved confusion matrix: {save_path}")


def plot_probability_distribution(labels, probs, save_path):
    """Histogram of predicted probabilities split by true label."""
    fig, ax = plt.subplots(figsize=(10, 6))

    human_probs = probs[labels == 0]
    ai_probs = probs[labels == 1]

    ax.hist(human_probs, bins=40, alpha=0.7, color=HUMAN_COLOR, label="Human", edgecolor="#0d1117")
    ax.hist(ai_probs, bins=40, alpha=0.7, color=AI_COLOR, label="AI-Generated", edgecolor="#0d1117")

    ax.axvline(x=0.5, color=ACCENT, linestyle="--", linewidth=2, label="Decision Threshold")
    ax.set_xlabel("Predicted Probability (AI)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count", fontsize=14, fontweight="bold")
    ax.set_title("Prediction Probability Distribution", fontsize=16, fontweight="bold", pad=15)
    ax.legend(fontsize=12, facecolor="#161b22", edgecolor="#30363d")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved probability distribution: {save_path}")


def plot_burstiness_analysis(test_df, probs, save_path):
    """Scatter plot: burstiness vs predicted probability, colored by true label."""
    fig, ax = plt.subplots(figsize=(10, 6))

    human_mask = test_df["label"] == 0
    ai_mask = test_df["label"] == 1

    ax.scatter(
        test_df.loc[human_mask, "burstiness"], probs[human_mask],
        alpha=0.5, s=30, color=HUMAN_COLOR, label="Human", edgecolors="none",
    )
    ax.scatter(
        test_df.loc[ai_mask, "burstiness"], probs[ai_mask],
        alpha=0.5, s=30, color=AI_COLOR, label="AI-Generated", edgecolors="none",
    )

    ax.axhline(y=0.5, color=ACCENT, linestyle="--", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Burstiness (Sentence Length Std Dev)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Predicted Probability (AI)", fontsize=14, fontweight="bold")
    ax.set_title("Burstiness vs Prediction Probability", fontsize=16, fontweight="bold", pad=15)
    ax.legend(fontsize=12, facecolor="#161b22", edgecolor="#30363d")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved burstiness analysis: {save_path}")


def plot_training_curves(save_path):
    """Plot training/validation loss and F1 curves from history."""
    history_path = os.path.join(MODEL_DIR, "training_history.json")
    if not os.path.exists(history_path):
        print("  [!] No training history found; skipping training curves.")
        return

    with open(history_path) as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curves
    axes[0].plot(epochs, history["train_loss"], marker="o", color=HUMAN_COLOR,
                 linewidth=2, markersize=8, label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], marker="s", color=AI_COLOR,
                 linewidth=2, markersize=8, label="Val Loss")
    axes[0].set_xlabel("Epoch", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Loss", fontsize=13, fontweight="bold")
    axes[0].set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    axes[0].legend(facecolor="#161b22", edgecolor="#30363d")
    axes[0].grid(True, alpha=0.3)

    # F1 / Accuracy curves
    axes[1].plot(epochs, history["val_f1"], marker="o", color=ACCENT,
                 linewidth=2, markersize=8, label="Val F1")
    axes[1].plot(epochs, history["val_acc"], marker="s", color="#d2a8ff",
                 linewidth=2, markersize=8, label="Val Accuracy")
    axes[1].set_xlabel("Epoch", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Score", fontsize=13, fontweight="bold")
    axes[1].set_title("Validation F1 & Accuracy", fontsize=14, fontweight="bold")
    axes[1].legend(facecolor="#161b22", edgecolor="#30363d")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.05)

    plt.suptitle("Training Curves — Hybrid Detector", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved training curves: {save_path}")


def plot_roc_curve(labels, probs, save_path):
    """Plot ROC curve with AUC."""
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color=ACCENT, linewidth=2.5, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#8b949e", linestyle="--", linewidth=1, alpha=0.7)
    ax.fill_between(fpr, tpr, alpha=0.15, color=ACCENT)
    ax.set_xlabel("False Positive Rate", fontsize=14, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=14, fontweight="bold")
    ax.set_title("ROC Curve — Hybrid Detector", fontsize=16, fontweight="bold", pad=15)
    ax.legend(fontsize=13, facecolor="#161b22", edgecolor="#30363d")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved ROC curve: {save_path}")


# ---------------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------------

def evaluate_baseline(test_df: pd.DataFrame) -> dict:
    """Evaluate the Logistic Regression baseline on the test set."""
    lr_path = os.path.join(MODEL_DIR, "baseline_lr.pkl")
    tfidf_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

    if not os.path.exists(lr_path) or not os.path.exists(tfidf_path):
        print("  [!] Baseline model not found; skipping.")
        return {}

    lr = joblib.load(lr_path)
    tfidf = joblib.load(tfidf_path)

    from scipy.sparse import hstack
    X_tfidf = tfidf.transform(test_df["text"])
    X_stats = test_df[["perplexity", "burstiness"]].values
    X = hstack([X_tfidf, X_stats])

    y_true = test_df["label"].values
    y_pred = lr.predict(X)
    y_prob = lr.predict_proba(X)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def predict_baseline(df: pd.DataFrame):
    """Run baseline inference on an arbitrary dataframe."""
    lr_path = os.path.join(MODEL_DIR, "baseline_lr.pkl")
    tfidf_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

    if not os.path.exists(lr_path) or not os.path.exists(tfidf_path):
        return None

    lr = joblib.load(lr_path)
    tfidf = joblib.load(tfidf_path)

    from scipy.sparse import hstack

    X_tfidf = tfidf.transform(df["text"])
    X_stats = df[["perplexity", "burstiness"]].values
    X = hstack([X_tfidf, X_stats])

    y_pred = lr.predict(X)
    y_prob = lr.predict_proba(X)[:, 1]
    return y_pred, y_prob


def edit_text_to_humanize(text: str) -> str:
    """Create a deterministic human-edited variant of an AI-like text."""
    replacements = [
        (r"\bHowever,\b", "Honestly,"),
        (r"\bTherefore,\b", "So,"),
        (r"\bIn conclusion,\b", "Basically,"),
        (r"\bIt is important to note that\b", "One thing people forget is"),
        (r"\bMoreover,\b", "Also,"),
        (r"\bAdditionally,\b", "And honestly,"),
        (r"\bOrganizations\b", "A lot of teams"),
        (r"\bindividuals\b", "people"),
    ]
    edited = text
    for pattern, repl in replacements:
        edited = re.sub(pattern, repl, edited)

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", edited) if s.strip()]
    if not sentences:
        return text

    prefix_options = [
        "Honestly, ",
        "I mean, ",
        "To be fair, ",
        "From what I've seen, ",
    ]
    suffix_options = [
        " It is not perfect, though.",
        " At least that is how it feels in practice.",
        " That part matters more than people admit.",
        " It sounds simple, but it really is not.",
    ]

    sentences[0] = random.choice(prefix_options) + sentences[0][:1].lower() + sentences[0][1:]
    if len(sentences) > 1:
        sentences.insert(1, random.choice([
            "People do not always say it this directly.",
            "That is where the conversation gets messy.",
            "In real life, it usually feels less neat than that.",
        ]))
    edited = " ".join(sentences) + random.choice(suffix_options)
    edited = edited.replace(" do not ", " don't ")
    edited = edited.replace(" is not ", " isn't ")
    edited = edited.replace(" are not ", " aren't ")
    return edited


def build_human_edited_ai_challenge(test_df: pd.DataFrame) -> pd.DataFrame:
    """Create a challenge set by humanizing AI-labeled examples from the test split."""
    ai_df = test_df[test_df["label"] == 1].copy().reset_index(drop=True)
    challenge_df = ai_df.copy()
    challenge_df["original_text"] = challenge_df["text"]
    challenge_df["text"] = challenge_df["text"].apply(edit_text_to_humanize)
    challenge_df["burstiness"] = challenge_df["text"].apply(
        lambda text: float(np.std([len(s.split()) for s in re.split(r"[.!?]+", text) if s.strip()]))
        if len([s for s in re.split(r"[.!?]+", text) if s.strip()]) >= 2 else 0.0
    )
    challenge_df["source"] = challenge_df["source"].astype(str) + "_edited"
    return challenge_df


def evaluate_by_source(test_df: pd.DataFrame, labels, preds, probs, save_path: str) -> pd.DataFrame:
    """Compute source-wise metrics for the hybrid model."""
    rows = []
    working = test_df.copy()
    working["label"] = labels
    working["pred"] = preds
    working["prob"] = probs

    for source, group in working.groupby("source"):
        rows.append({
            "source": source,
            "samples": int(len(group)),
            **compute_metrics(group["label"].values, group["pred"].values, group["prob"].values),
        })

    source_df = pd.DataFrame(rows).sort_values("source").reset_index(drop=True)
    source_df.to_csv(save_path, index=False)
    print(f"  -> Saved source-wise metrics: {save_path}")
    return source_df


def evaluate_human_edited_ai_challenge(model, tokenizer, device, test_df: pd.DataFrame, max_len: int, save_dir: str):
    """Evaluate how the models behave on human-edited AI text."""
    challenge_df = build_human_edited_ai_challenge(test_df)
    labels, preds, probs = predict_hybrid_dataframe(model, challenge_df, tokenizer, device, max_len=max_len)
    hybrid_metrics = compute_metrics(labels, preds, probs)

    challenge_df["hybrid_pred"] = preds
    challenge_df["hybrid_prob_ai"] = probs

    baseline_out = predict_baseline(challenge_df)
    baseline_metrics = {}
    if baseline_out is not None:
        baseline_preds, baseline_probs = baseline_out
        challenge_df["baseline_pred"] = baseline_preds
        challenge_df["baseline_prob_ai"] = baseline_probs
        baseline_metrics = compute_metrics(labels, baseline_preds, baseline_probs)

    preview_path = os.path.join(save_dir, "human_edited_ai_samples.csv")
    preview_columns = [
        "source", "label", "original_text", "text", "hybrid_pred", "hybrid_prob_ai",
    ]
    if "baseline_pred" in challenge_df.columns:
        preview_columns.extend(["baseline_pred", "baseline_prob_ai"])
    challenge_df[preview_columns].head(100).to_csv(preview_path, index=False)
    print(f"  -> Saved human-edited AI samples: {preview_path}")

    summary = {
        "hybrid_metrics": hybrid_metrics,
        "baseline_metrics": baseline_metrics,
        "samples": int(len(challenge_df)),
    }
    summary_path = os.path.join(save_dir, "human_edited_ai_results.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  -> Saved human-edited AI summary: {summary_path}")
    return summary


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

def error_analysis(test_df: pd.DataFrame, labels, preds, probs, save_path):
    """Identify and save misclassified samples for review."""
    errors = test_df.copy()
    errors["predicted"] = preds
    errors["prob_ai"] = probs
    errors["correct"] = labels == preds

    misclassified = errors[~errors["correct"]].sort_values("prob_ai", ascending=False)

    # Save top errors
    misclassified.head(50).to_csv(save_path, index=False)
    print(f"  -> Saved error analysis ({len(misclassified)} errors): {save_path}")

    # Summary
    fp = ((labels == 0) & (preds == 1)).sum()  # Humans flagged as AI
    fn = ((labels == 1) & (preds == 0)).sum()  # AI missed as human
    print(f"  -> False Positives (human -> AI): {fp}")
    print(f"  -> False Negatives (AI -> human): {fn}")

    return len(misclassified), fp, fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Synthetic Sentinel — Evaluation")
    parser.add_argument("--model-path", default=os.path.join(MODEL_DIR, "best_roberta.pt"))
    parser.add_argument("--max-len", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("EVALUATION: Synthetic Sentinel — Hybrid Detector")
    print("=" * 60)

    if os.path.exists(DATASET_METADATA_PATH):
        with open(DATASET_METADATA_PATH, encoding="utf-8") as f:
            dataset_metadata = json.load(f)
        synthetic_fraction = dataset_metadata.get("synthetic_fraction", 0.0)
        if synthetic_fraction > 0:
            print(
                f"[!] Dataset warning: {synthetic_fraction * 100:.1f}% of the current dataset is "
                "synthetic fallback data. Treat metrics as synthetic-benchmark results."
            )

    # Load model & data
    print("\nLoading model and test data ...")
    model, tokenizer, test_loader, test_df, device = load_model_and_data(args.model_path, args.max_len)

    # Predictions
    print("Running inference ...")
    labels, preds, probs = get_predictions(model, test_loader, device)

    # --- Metrics ---
    print("\n--- Test Set Metrics ---")
    metrics = compute_metrics(labels, preds, probs)
    for k, v in metrics.items():
        print(f"  {k:>12s}: {v:.4f}")

    print("\n" + classification_report(labels, preds, target_names=["Human", "AI-Generated"]))

    # --- Visualizations ---
    print("Generating visualizations ...")
    plot_confusion_matrix(labels, preds, os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plot_probability_distribution(labels, probs, os.path.join(RESULTS_DIR, "prob_dist.png"))
    plot_burstiness_analysis(test_df, probs, os.path.join(RESULTS_DIR, "burstiness.png"))
    plot_training_curves(os.path.join(RESULTS_DIR, "training_curves.png"))
    plot_roc_curve(labels, probs, os.path.join(RESULTS_DIR, "roc_curve.png"))

    print("\n--- Source-wise Evaluation ---")
    source_metrics_df = evaluate_by_source(
        test_df, labels, preds, probs,
        os.path.join(RESULTS_DIR, "source_metrics.csv"),
    )
    print(source_metrics_df.to_string(index=False))

    # --- Baseline comparison ---
    print("\n--- Baseline Comparison ---")
    baseline_metrics = evaluate_baseline(test_df)
    if baseline_metrics:
        comparison = pd.DataFrame({
            "Logistic Regression": baseline_metrics,
            "Hybrid (RoBERTa)": metrics,
        })
        print(comparison.to_string())
        comparison.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"))
        print(f"  -> Saved comparison: {os.path.join(RESULTS_DIR, 'model_comparison.csv')}")

    # --- Error analysis ---
    print("\n--- Error Analysis ---")
    total_errors, fp, fn = error_analysis(
        test_df, labels, preds, probs,
        os.path.join(RESULTS_DIR, "misclassified_samples.csv"),
    )

    print("\n--- Human-Edited AI Challenge ---")
    human_edited_ai_results = evaluate_human_edited_ai_challenge(
        model, tokenizer, device, test_df, args.max_len, RESULTS_DIR
    )
    print(json.dumps(human_edited_ai_results, indent=2))

    # --- Save all metrics ---
    all_results = {
        "hybrid_metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "source_metrics_path": os.path.join(RESULTS_DIR, "source_metrics.csv"),
        "human_edited_ai_results": human_edited_ai_results,
        "total_test_samples": len(labels),
        "total_errors": total_errors,
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }
    with open(os.path.join(RESULTS_DIR, "evaluation_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[OK] Evaluation complete! All results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
