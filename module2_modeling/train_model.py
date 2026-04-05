"""
train_model.py — Synthetic Sentinel Model Training
=====================================================
Phase 2: Baseline (Logistic Regression), RoBERTa fine-tuning, and Hybrid model.

Usage:
    python train_model.py                   # Train all models
    python train_model.py --baseline-only   # Only train Logistic Regression
    python train_model.py --epochs 5 --lr 5e-5  # Custom hyperparams
"""

import os
import json
import argparse
import warnings

import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import (
    RobertaTokenizerFast,
    RobertaModel,
    get_linear_schedule_with_warmup,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Baseline: Logistic Regression on TF-IDF + statistical features
# ---------------------------------------------------------------------------

def train_baseline(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    """Train a Logistic Regression baseline using TF-IDF + perplexity + burstiness."""
    print("\n" + "=" * 60)
    print("BASELINE: Logistic Regression (TF-IDF + Statistical Features)")
    print("=" * 60)

    # TF-IDF
    print("  Fitting TF-IDF vectorizer ...")
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words="english")
    X_train_tfidf = tfidf.fit_transform(train_df["text"])
    X_val_tfidf = tfidf.transform(val_df["text"])

    # Statistical features
    train_stats = train_df[["perplexity", "burstiness"]].values
    val_stats = val_df[["perplexity", "burstiness"]].values

    # Combine
    from scipy.sparse import hstack
    X_train = hstack([X_train_tfidf, train_stats])
    X_val = hstack([X_val_tfidf, val_stats])

    y_train = train_df["label"].values
    y_val = val_df["label"].values

    # Train
    print("  Training Logistic Regression ...")
    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_SEED)
    lr.fit(X_train, y_train)

    # Evaluate
    y_pred = lr.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    print(f"  -> Validation Accuracy: {acc:.4f}")
    print(f"  -> Validation F1-Score: {f1:.4f}")

    # Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(lr, os.path.join(MODEL_DIR, "baseline_lr.pkl"))
    joblib.dump(tfidf, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    print(f"  -> Saved to {MODEL_DIR}/baseline_lr.pkl")

    return {"model": "LogisticRegression", "accuracy": acc, "f1": f1}


# ---------------------------------------------------------------------------
# PyTorch Dataset for RoBERTa
# ---------------------------------------------------------------------------

class TextDataset(Dataset):
    """Custom dataset for RoBERTa training."""

    def __init__(self, texts, labels, perplexities, burstiness_vals, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.perplexities = perplexities
        self.burstiness_vals = burstiness_vals
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "perplexity": torch.tensor(self.perplexities[idx], dtype=torch.float),
            "burstiness": torch.tensor(self.burstiness_vals[idx], dtype=torch.float),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Hybrid Model: RoBERTa [CLS] + statistical features -> MLP classifier
# ---------------------------------------------------------------------------

class HybridDetector(nn.Module):
    """
    Combines RoBERTa's [CLS] embedding with perplexity & burstiness features.
    Architecture:
        RoBERTa -> [CLS] (768-dim) + [perplexity, burstiness] (2-dim)
        -> Linear(770, 256) -> ReLU -> Dropout -> Linear(256, 2)
    """

    def __init__(self, roberta_model_name: str = "roberta-base", dropout: float = 0.3):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(roberta_model_name)
        hidden_size = self.roberta.config.hidden_size  # 768 for roberta-base

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + 2, 256),  # +2 for perplexity & burstiness
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, input_ids, attention_mask, perplexity, burstiness):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token

        # Normalize features in-model (Simple approach: log(1+x) / factor)
        # This keeps PPL (0-100) and Burst (0-10) in a 0-1 range roughly.
        perp_scaled = torch.log1p(perplexity) / 5.0
        burst_scaled = burstiness / 5.0

        # Concatenate statistical features
        stats = torch.stack([perp_scaled, burst_scaled], dim=1)  # (batch, 2)
        combined = torch.cat([cls_output, stats], dim=1)      # (batch, 770)

        logits = self.classifier(combined)
        return logits


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_hybrid(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    epochs: int = 3,
    lr: float = 2e-5,
    batch_size: int = 16,
    max_len: int = 256,
    weight_decay: float = 0.01,
    freeze_roberta: bool = False,
) -> dict:
    """Train the Hybrid RoBERTa detector."""
    print("\n" + "=" * 60)
    print("HYBRID MODEL: RoBERTa + Statistical Features")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    # Datasets
    train_dataset = TextDataset(
        train_df["text"].tolist(), train_df["label"].tolist(),
        train_df["perplexity"].tolist(), train_df["burstiness"].tolist(),
        tokenizer, max_len,
    )
    val_dataset = TextDataset(
        val_df["text"].tolist(), val_df["label"].tolist(),
        val_df["perplexity"].tolist(), val_df["burstiness"].tolist(),
        tokenizer, max_len,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    model = HybridDetector().to(device)

    if freeze_roberta:
        print("  Freezing RoBERTa encoder; training classifier head only.")
        for param in model.roberta.parameters():
            param.requires_grad = False
    else:
        print("  Fine-tuning RoBERTa encoder and classifier head.")

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    total_steps = max(len(train_loader) * epochs, 1)
    warmup_steps = max(int(total_steps * 0.1), 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    print(f"  Scheduler: linear warmup ({warmup_steps} warmup steps / {total_steps} total steps)")

    # Training
    best_f1 = 0.0
    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": []}

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"  Epoch {epoch+1}/{epochs} [Train]"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            perplexity = batch["perplexity"].to(device)
            burstiness = batch["burstiness"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask, perplexity, burstiness)
            loss = criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)

        # --- Validate ---
        model.eval()
        val_preds, val_labels = [], []
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"  Epoch {epoch+1}/{epochs} [Val]  "):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                perplexity = batch["perplexity"].to(device)
                burstiness = batch["burstiness"].to(device)
                labels = batch["label"].to(device)

                logits = model(input_ids, attention_mask, perplexity, burstiness)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_f1 = f1_score(val_labels, val_preds)
        val_acc = accuracy_score(val_labels, val_preds)

        history["val_loss"].append(avg_val_loss)
        history["val_f1"].append(val_f1)
        history["val_acc"].append(val_acc)

        print(f"  Epoch {epoch+1}: train_loss={avg_train_loss:.4f}  "
              f"val_loss={avg_val_loss:.4f}  val_f1={val_f1:.4f}  val_acc={val_acc:.4f}")

        # Save best
        if val_f1 > best_f1:
            best_f1 = val_f1
            os.makedirs(MODEL_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_roberta.pt"))
            print(f"  -> New best model saved (F1={best_f1:.4f})")

    # Save training history
    with open(os.path.join(MODEL_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Save tokenizer for inference
    tokenizer.save_pretrained(os.path.join(MODEL_DIR, "tokenizer"))

    print(f"\n  [OK] Best validation F1: {best_f1:.4f}")
    return {"model": "HybridDetector", "best_f1": best_f1, "history": history}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Synthetic Sentinel — Model Training")
    parser.add_argument("--baseline-only", action="store_true", help="Only train the baseline model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--max-len", type=int, default=256, help="Max token sequence length")
    parser.add_argument(
        "--freeze-roberta",
        action="store_true",
        help="Freeze the RoBERTa encoder and train only the classifier head",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optionally train on a reproducible subset of the training split",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help="Optionally validate on a reproducible subset of the validation split",
    )
    args = parser.parse_args()

    # Load data
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    print(f"Loaded train ({len(train_df)}) and val ({len(val_df)}) splits.\n")

    results = {}

    # 1. Baseline
    baseline_result = train_baseline(train_df, val_df)
    results["baseline"] = baseline_result

    # 2. Hybrid RoBERTa (unless --baseline-only)
    if not args.baseline_only:
        train_subset = train_df
        val_subset = val_df

        if args.max_train_samples is not None and args.max_train_samples < len(train_subset):
            train_subset = train_subset.sample(
                n=args.max_train_samples, random_state=RANDOM_SEED
            ).reset_index(drop=True)
            print(f"Using train subset: {len(train_subset)} samples")

        if args.max_val_samples is not None and args.max_val_samples < len(val_subset):
            val_subset = val_subset.sample(
                n=args.max_val_samples, random_state=RANDOM_SEED
            ).reset_index(drop=True)
            print(f"Using val subset: {len(val_subset)} samples")

        hybrid_result = train_hybrid(
            train_subset, val_subset,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            max_len=args.max_len,
            freeze_roberta=args.freeze_roberta,
        )
        results["hybrid"] = hybrid_result

    # Save results summary
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "training_results.json"), "w") as f:
        # Convert numpy types for JSON serialization
        clean_results = {}
        for k, v in results.items():
            clean_results[k] = {
                kk: (float(vv) if isinstance(vv, (np.floating, float)) else vv)
                for kk, vv in v.items()
                if kk != "history"
            }
        json.dump(clean_results, f, indent=2)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE — Summary")
    print("=" * 60)
    for model_name, result in results.items():
        f1_val = result.get("f1") or result.get("best_f1", "N/A")
        print(f"  {model_name}: F1 = {f1_val}")
    print(f"\nAll models saved to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
