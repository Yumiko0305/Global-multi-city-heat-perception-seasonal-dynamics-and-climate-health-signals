#!/usr/bin/env python3
"""
05_train_grouped_cv.py

Grouped 5-fold BERT training with class-weighted loss, plus optional monthly
holdout evaluation that reproduces the Accuracy / Recall / F1 curves in
Supplementary Figure S7 of the manuscript.

Expected input columns:
- sentence
- label
- city_id
Optional but useful:
- tweet_id, month, continent, original_split

TWO EVALUATION MODES
--------------------
1. Grouped 5-fold CV (default)
   City-grouped cross-validation used for model selection and hyperparameter
   tuning.  Ensures the model is evaluated on unseen cities, matching the
   study's generalisation goal.

2. Monthly holdout evaluation (--temporal_test_csv)
   A held-out test CSV is annotated with a 'month' column (YYYY-MM).
   After the final model is trained on --data_path, it is evaluated separately
   on each calendar month in the test file.  This produces the per-month
   Accuracy / Recall / F1 series reported in Supplementary Figure S7.

Outputs:
- per-fold metrics JSON
- combined fold predictions CSV
- final model trained on full labeled data
- monthly_test_metrics.csv (when --temporal_test_csv is provided)
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_class_weight
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments, set_seed


class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = list(labels)

    def __getitem__(self, idx):
        item = {k: self.encodings[k][idx] for k in self.encodings}
        item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
            token_type_ids=inputs.get("token_type_ids", None),
        )
        logits = outputs.get("logits")
        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_fct = nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True,
                        help="Gold-labelled training CSV (sentence, label, city_id, month, …)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="bert-base-uncased")
    parser.add_argument("--text_col", default="sentence")
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--group_col", default="city_id")
    parser.add_argument("--month_col", default="month",
                        help="Column containing YYYY-MM period labels (used for monthly eval)")
    parser.add_argument("--temporal_test_csv", default=None,
                        help=(
                            "Optional holdout test CSV with a 'month' column. "
                            "When provided, the final model is evaluated on each calendar month "
                            "separately, reproducing the monthly Accuracy/Recall/F1 curves in "
                            "Supplementary Figure S7."
                        ))
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_length", type=int, default=140)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def tokenize(tokenizer, texts: List[str], max_length: int):
    return tokenizer(texts, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")


def build_class_weights(y):
    classes = np.array(sorted(np.unique(y)))
    if len(classes) != 2:
        raise ValueError(f"Expected binary labels with both classes present, got {classes.tolist()}")
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return torch.tensor(weights, dtype=torch.float)


def make_args(output_dir, args, seed_offset: int = 0):
    return TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        seed=args.seed + seed_offset,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )


def train_one_fold(fold_id: int, train_df: pd.DataFrame, val_df: pd.DataFrame, tokenizer, args):
    train_enc = tokenize(tokenizer, train_df[args.text_col].astype(str).tolist(), args.max_length)
    val_enc = tokenize(tokenizer, val_df[args.text_col].astype(str).tolist(), args.max_length)

    train_ds = TweetDataset(train_enc, train_df[args.label_col].tolist())
    val_ds = TweetDataset(val_enc, val_df[args.label_col].tolist())
    class_weights = build_class_weights(train_df[args.label_col].values)

    model = BertForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=2)
    fold_out = os.path.join(args.output_dir, f"fold_{fold_id}")
    trainer = WeightedTrainer(
        model=model,
        args=make_args(fold_out, args, seed_offset=fold_id),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )
    trainer.train()
    metrics = trainer.evaluate()
    pred = trainer.predict(val_ds)
    preds = np.argmax(pred.predictions, axis=-1)
    report = classification_report(val_df[args.label_col].values, preds, digits=4, zero_division=0)

    pred_df = val_df.copy()
    pred_df["fold_id"] = fold_id
    pred_df["pred_label"] = preds
    pred_df["pred_prob_1"] = torch.softmax(torch.tensor(pred.predictions), dim=1).numpy()[:, 1]
    return trainer, metrics, report, pred_df


def summarize_metrics(fold_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    keys = ["eval_accuracy", "eval_precision", "eval_recall", "eval_f1"]
    out = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics if k in m]
        out[f"{k}_mean"] = float(np.mean(vals)) if vals else float("nan")
        out[f"{k}_std"] = float(np.std(vals)) if vals else float("nan")
    return out


def evaluate_monthly(
    model,
    tokenizer,
    test_df: pd.DataFrame,
    text_col: str,
    label_col: str,
    month_col: str,
    max_length: int,
    batch_size: int = 64,
) -> pd.DataFrame:
    """
    Evaluate the final model on each calendar month in test_df separately.

    This reproduces the monthly Accuracy / Recall / F1 series shown in
    Supplementary Figure S7.  The test split is the held-out set that was
    NOT seen during grouped CV training.

    Returns a DataFrame with columns:
        month, n, accuracy, precision, recall, f1
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    months = sorted(test_df[month_col].dropna().unique())
    rows = []

    for month in months:
        sub = test_df[test_df[month_col] == month].copy().reset_index(drop=True)
        if len(sub) == 0:
            continue
        if label_col in sub.columns and sub[label_col].nunique() < 2:
            # skip months with only one class present — metrics undefined
            continue

        enc = tokenizer(
            sub[text_col].astype(str).tolist(),
            truncation=True, padding="max_length",
            max_length=max_length, return_tensors="pt",
        )
        all_preds, all_probs = [], []
        for start in range(0, len(sub), batch_size):
            batch = {k: v[start: start + batch_size].to(device) for k, v in enc.items()}
            with torch.no_grad():
                logits = model(**batch).logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_preds.extend(np.argmax(probs, axis=1).tolist())
            all_probs.extend(probs[:, 1].tolist())

        row: Dict[str, object] = {"month": month, "n": int(len(sub))}
        if label_col in sub.columns:
            labels = sub[label_col].astype(int).values
            row["accuracy"] = float(accuracy_score(labels, all_preds))
            prec, rec, f1, _ = precision_recall_fscore_support(
                labels, all_preds, average="binary", zero_division=0
            )
            row["precision"] = float(prec)
            row["recall"] = float(rec)
            row["f1"] = float(f1)
        rows.append(row)

    cols = ["month", "n", "accuracy", "precision", "recall", "f1"]
    return pd.DataFrame(rows, columns=[c for c in cols if rows and c in rows[0]])


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data_path)
    required = [args.text_col, args.label_col, args.group_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df[args.text_col] = df[args.text_col].astype(str)
    df[args.label_col] = df[args.label_col].astype(int)
    df[args.group_col] = df[args.group_col].astype(str)
    df = df[df[args.text_col].str.strip() != ""].reset_index(drop=True)

    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    groups = df[args.group_col].values
    n_groups = df[args.group_col].nunique()
    n_folds = min(args.num_folds, n_groups)
    if n_folds < 2:
        raise ValueError("Need at least 2 unique groups for grouped CV")

    gkf = GroupKFold(n_splits=n_folds)
    fold_metrics = []
    fold_reports = {}
    pred_frames = []

    for fold_id, (tr_idx, va_idx) in enumerate(gkf.split(df, df[args.label_col], groups), start=1):
        train_df = df.iloc[tr_idx].reset_index(drop=True)
        val_df = df.iloc[va_idx].reset_index(drop=True)
        trainer, metrics, report, pred_df = train_one_fold(fold_id, train_df, val_df, tokenizer, args)
        fold_metrics.append(metrics)
        fold_reports[f"fold_{fold_id}"] = {"metrics": metrics, "classification_report": report}
        pred_frames.append(pred_df)

    cv_summary = summarize_metrics(fold_metrics)
    pred_all = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    pred_all.to_csv(os.path.join(args.output_dir, "cv_fold_predictions.csv"), index=False)

    with open(os.path.join(args.output_dir, "cv_results.json"), "w", encoding="utf-8") as f:
        json.dump({"fold_results": fold_reports, "cv_summary": cv_summary}, f, ensure_ascii=False, indent=2)

    # final model on full data
    full_enc = tokenize(tokenizer, df[args.text_col].tolist(), args.max_length)
    full_ds = TweetDataset(full_enc, df[args.label_col].tolist())
    class_weights = build_class_weights(df[args.label_col].values)
    final_model = BertForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=2)
    final_args = make_args(os.path.join(args.output_dir, "final_model"), args)
    final_args.evaluation_strategy = "no"
    final_args.save_strategy = "epoch"
    final_args.load_best_model_at_end = False

    final_trainer = WeightedTrainer(
        model=final_model,
        args=final_args,
        train_dataset=full_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )
    final_trainer.train()
    final_trainer.save_model(os.path.join(args.output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final_model"))

    # ── Monthly holdout evaluation (reproduces Supplementary Figure S7) ──────
    monthly_csv_path = None
    if args.temporal_test_csv:
        test_df = pd.read_csv(args.temporal_test_csv)
        test_df[args.text_col] = test_df[args.text_col].astype(str)
        if args.month_col not in test_df.columns:
            raise ValueError(
                f"--temporal_test_csv must contain a '{args.month_col}' column for monthly eval. "
                "If dates are available use 04_build_gold_labels.py to add month labels first."
            )
        monthly_df = evaluate_monthly(
            final_model, tokenizer,
            test_df,
            text_col=args.text_col,
            label_col=args.label_col,
            month_col=args.month_col,
            max_length=args.max_length,
            batch_size=args.eval_batch_size,
        )
        monthly_csv_path = os.path.join(args.output_dir, "monthly_test_metrics.csv")
        monthly_df.to_csv(monthly_csv_path, index=False)
        print("Monthly test-set metrics (Supplementary Figure S7):")
        print(monthly_df.to_string(index=False))
    # ─────────────────────────────────────────────────────────────────────────

    print(json.dumps({
        "n_rows": int(len(df)),
        "n_groups": int(n_groups),
        "n_folds": int(n_folds),
        "cv_summary": cv_summary,
        "artifacts": {
            "cv_results_json": os.path.join(args.output_dir, "cv_results.json"),
            "cv_predictions_csv": os.path.join(args.output_dir, "cv_fold_predictions.csv"),
            "final_model_dir": os.path.join(args.output_dir, "final_model"),
            "monthly_test_metrics_csv": monthly_csv_path,
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
