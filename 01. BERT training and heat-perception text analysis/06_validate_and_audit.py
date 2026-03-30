#!/usr/bin/env python3
"""
06_validate_and_audit.py

Evaluate a saved model on a holdout dataset and produce a monthly audit sheet.
Two modes:
1) --model_dir + --input_csv : load model, predict, compute metrics if labels exist
2) --predictions_csv only    : read an existing predictions file and create monthly audit samples

Monthly audit output mirrors the manuscript idea of manual verification per month.
If labels are present, monthly metrics are also exported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from transformers import BertForSequenceClassification, BertTokenizer


class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {k: self.encodings[k][idx] for k in self.encodings}

    def __len__(self):
        return len(self.encodings["input_ids"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--input_csv", default=None)
    parser.add_argument("--predictions_csv", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--text_col", default="sentence")
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--month_col", default="month")
    parser.add_argument("--continent_col", default="continent")
    parser.add_argument("--max_length", type=int, default=140)
    parser.add_argument("--audit_per_month", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def predict_dataset(model_dir: str, input_csv: str, text_col: str, max_length: int) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    if text_col not in df.columns:
        raise ValueError(f"Missing text column: {text_col}")

    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    enc = tokenizer(df[text_col].astype(str).tolist(), truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
    dataset = TweetDataset(enc)

    batch_size = 64
    probs = []
    preds = []
    for start in range(0, len(dataset), batch_size):
        batch = {k: v[start:start+batch_size] for k, v in enc.items()}
        if torch.cuda.is_available():
            batch = {k: v.cuda() for k, v in batch.items()}
        with torch.no_grad():
            logits = model(**batch).logits
            p = torch.softmax(logits, dim=1).cpu().numpy()
        probs.extend(p[:, 1].tolist())
        preds.extend(np.argmax(p, axis=1).tolist())

    out = df.copy()
    out["pred_label"] = preds
    out["pred_prob_1"] = probs
    return out


def overall_metrics(df: pd.DataFrame, label_col: str) -> Dict[str, float]:
    labels = df[label_col].astype(int).values
    preds = df["pred_label"].astype(int).values
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "classification_report": classification_report(labels, preds, digits=4, zero_division=0),
    }


def monthly_metrics(df: pd.DataFrame, label_col: str, month_col: str) -> pd.DataFrame:
    rows = []
    for month, sub in df.groupby(month_col):
        if sub[label_col].nunique() < 2:
            continue
        m = overall_metrics(sub, label_col)
        rows.append({
            "month": month,
            "n": int(len(sub)),
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
        })
    return pd.DataFrame(rows).sort_values("month") if rows else pd.DataFrame(columns=["month", "n", "accuracy", "precision", "recall", "f1"])


def make_audit_sample(df: pd.DataFrame, month_col: str, continent_col: str, audit_per_month: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    if month_col not in df.columns:
        df = df.copy()
        df[month_col] = "unknown"
    if continent_col not in df.columns:
        df = df.copy()
        df[continent_col] = "unknown"

    for month, sub_month in df.groupby(month_col):
        n_target = min(audit_per_month, len(sub_month))
        if n_target == 0:
            continue
        continent_counts = sub_month[continent_col].value_counts()
        alloc = (continent_counts / continent_counts.sum() * n_target).round().astype(int)
        # fix rounding drift
        diff = n_target - int(alloc.sum())
        if diff != 0:
            order = continent_counts.sort_values(ascending=False).index.tolist()
            for i in range(abs(diff)):
                alloc[order[i % len(order)]] += 1 if diff > 0 else -1
        for continent, n_take in alloc.items():
            if n_take <= 0:
                continue
            sub = sub_month[sub_month[continent_col] == continent]
            chosen_idx = rng.choice(sub.index.to_numpy(), size=min(n_take, len(sub)), replace=False)
            chosen = sub.loc[chosen_idx].copy()
            rows.append(chosen)

    audit_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=df.columns)
    if not audit_df.empty:
        audit_df["manual_label"] = ""
        audit_df["manual_review_notes"] = ""
        audit_df["audit_status"] = "pending"
    return audit_df


def main() -> None:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.predictions_csv:
        pred_df = pd.read_csv(args.predictions_csv)
    else:
        if not args.model_dir or not args.input_csv:
            raise ValueError("Provide either --predictions_csv or (--model_dir and --input_csv)")
        pred_df = predict_dataset(args.model_dir, args.input_csv, args.text_col, args.max_length)
        pred_df.to_csv(Path(args.output_dir) / "predictions.csv", index=False)

    summary = {"n_rows": int(len(pred_df))}

    if args.label_col in pred_df.columns:
        metrics = overall_metrics(pred_df, args.label_col)
        summary.update({k: v for k, v in metrics.items() if k != "classification_report"})
        with open(Path(args.output_dir) / "overall_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        month_df = monthly_metrics(pred_df, args.label_col, args.month_col)
        month_df.to_csv(Path(args.output_dir) / "monthly_metrics.csv", index=False)

    audit_df = make_audit_sample(pred_df, args.month_col, args.continent_col, args.audit_per_month, args.seed)
    audit_df.to_csv(Path(args.output_dir) / "monthly_audit_sample.csv", index=False)

    summary["audit_rows"] = int(len(audit_df))
    summary["artifacts"] = {
        "monthly_audit_sample": str(Path(args.output_dir) / "monthly_audit_sample.csv"),
        "monthly_metrics": str(Path(args.output_dir) / "monthly_metrics.csv") if args.label_col in pred_df.columns else None,
        "overall_metrics": str(Path(args.output_dir) / "overall_metrics.json") if args.label_col in pred_df.columns else None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
