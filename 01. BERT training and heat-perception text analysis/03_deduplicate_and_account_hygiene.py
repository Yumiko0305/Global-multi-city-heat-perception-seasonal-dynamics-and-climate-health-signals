#!/usr/bin/env python3
"""
03_deduplicate_and_account_hygiene.py

- removes exact retweets / exact quote tweets
- removes exact text duplicates
- collapses near duplicates with a light-weight shingle Jaccard heuristic
- excludes / flags templated promotional account behavior when account metadata exists

The manuscript mentions MinHash with 5-gram shingles and Jaccard threshold 0.85.
This implementation uses 5-gram shingles and a bucketed Jaccard approximation so it can run
without extra dependencies. If account columns are absent, account hygiene is skipped safely.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

TOKEN_RE = re.compile(r"[a-zA-Z0-9']+")
PROMO_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bbuy now\b", r"\blimited offer\b", r"\bpromo\b", r"\bdiscount\b", r"\bshop now\b", r"\blink in bio\b",
        r"\bsubscribe\b", r"\baffiliate\b", r"\bsponsored\b", r"\bsale\b",
    ]
]
RT_PATTERNS = [re.compile(r"^rt\s+@", re.IGNORECASE), re.compile(r"^qt\s+@", re.IGNORECASE)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--removal_log_csv", required=True)
    parser.add_argument("--text_col", default="sentence")
    parser.add_argument("--user_col", default=None)
    parser.add_argument("--month_col", default="month")
    parser.add_argument("--jaccard_threshold", type=float, default=0.85)
    parser.add_argument("--max_bucket_compare", type=int, default=200)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    tokens = TOKEN_RE.findall(str(text or "").lower())
    return " ".join(tokens)


def is_rt_or_qt(text: str) -> bool:
    text = str(text or "").strip()
    return any(p.search(text) for p in RT_PATTERNS)


def shingles(text: str, n: int = 5) -> Set[str]:
    toks = normalize_text(text).split()
    if len(toks) < n:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i:i+n]) for i in range(len(toks) - n + 1)}


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    denom = len(a | b)
    return len(a & b) / denom if denom else 0.0


def bucket_key(normalized: str) -> Tuple[str, int, int]:
    toks = normalized.split()
    prefix = " ".join(toks[:4]) if toks else ""
    return (prefix, len(toks) // 5, len(normalized) // 20)


def detect_near_duplicates(df: pd.DataFrame, text_col: str, threshold: float, max_bucket_compare: int) -> Dict[int, int]:
    bucket_map: Dict[Tuple[str, int, int], List[int]] = defaultdict(list)
    normalized_map: Dict[int, str] = {}
    shingles_map: Dict[int, Set[str]] = {}

    for idx, text in df[text_col].items():
        normalized = normalize_text(text)
        normalized_map[idx] = normalized
        shingles_map[idx] = shingles(text)
        bucket_map[bucket_key(normalized)].append(idx)

    duplicate_of: Dict[int, int] = {}
    for _, idxs in bucket_map.items():
        if len(idxs) <= 1:
            continue
        idxs = idxs[:max_bucket_compare]
        representatives: List[int] = []
        for idx in idxs:
            if idx in duplicate_of:
                continue
            found_rep = None
            for rep in representatives:
                if abs(len(normalized_map[idx]) - len(normalized_map[rep])) > 40:
                    continue
                if jaccard(shingles_map[idx], shingles_map[rep]) >= threshold:
                    found_rep = rep
                    break
            if found_rep is not None:
                duplicate_of[idx] = found_rep
            else:
                representatives.append(idx)
    return duplicate_of


def find_account_col(df: pd.DataFrame, preferred: Optional[str]) -> Optional[str]:
    if preferred and preferred in df.columns:
        return preferred
    for c in ["user_id", "account_id", "username", "screen_name", "user_name", "author_id"]:
        if c in df.columns:
            return c
    return None


def assign_month_if_missing(df: pd.DataFrame, month_col: str) -> pd.Series:
    if month_col in df.columns:
        return df[month_col].astype(str).fillna("unknown")
    if "created_at" in df.columns:
        return pd.to_datetime(df["created_at"], errors="coerce").dt.to_period("M").astype(str).fillna("unknown")
    return pd.Series(["unknown"] * len(df), index=df.index)


def promo_score(text: str) -> int:
    return sum(int(p.search(str(text or "")) is not None) for p in PROMO_PATTERNS)


def detect_account_hygiene(df: pd.DataFrame, text_col: str, user_col: Optional[str], month_col: str) -> Dict[int, str]:
    if user_col is None:
        return {}

    months = assign_month_if_missing(df, month_col)
    work = df.copy()
    work["__month"] = months
    work["__norm"] = work[text_col].map(normalize_text)
    work["__promo_score"] = work[text_col].map(promo_score)

    flags: Dict[int, str] = {}
    grouped = work.groupby([user_col, "__month"], dropna=False)
    for (_, _), sub in grouped:
        if len(sub) < 5:
            continue
        dup_ratio = sub["__norm"].duplicated(keep=False).mean()
        promo_ratio = (sub["__promo_score"] > 0).mean()
        if dup_ratio >= 0.80 or promo_ratio >= 0.60:
            reason = "templated_posting" if dup_ratio >= 0.80 else "bulk_promotional_behavior"
            for idx in sub.index:
                flags[idx] = reason
    return flags


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    if args.text_col not in df.columns:
        raise ValueError(f"Missing text column: {args.text_col}")

    work = df.copy()
    work["__normalized_text"] = work[args.text_col].map(normalize_text)

    removal_log = []
    keep_mask = pd.Series(True, index=work.index)

    # 1) remove explicit RT / QT
    for idx, text in work[args.text_col].items():
        if is_rt_or_qt(text):
            keep_mask.loc[idx] = False
            removal_log.append({"row_index": int(idx), "removal_reason": "rt_or_qt", "duplicate_of": "", "notes": "pattern match"})

    # 2) exact normalized duplicates
    dup_norm = work["__normalized_text"].duplicated(keep="first")
    for idx in work.index[dup_norm & keep_mask]:
        first_idx = int(work[work["__normalized_text"] == work.loc[idx, "__normalized_text"]].index[0])
        keep_mask.loc[idx] = False
        removal_log.append({"row_index": int(idx), "removal_reason": "exact_duplicate", "duplicate_of": first_idx, "notes": "normalized text duplicate"})

    remaining = work[keep_mask].copy()

    # 3) near duplicates (5-gram Jaccard)
    near_dup_map = detect_near_duplicates(remaining, args.text_col, args.jaccard_threshold, args.max_bucket_compare)
    for idx, rep in near_dup_map.items():
        if keep_mask.loc[idx]:
            keep_mask.loc[idx] = False
            removal_log.append({"row_index": int(idx), "removal_reason": "near_duplicate", "duplicate_of": int(rep), "notes": f"jaccard>={args.jaccard_threshold}"})

    # 4) account hygiene
    user_col = find_account_col(work, args.user_col)
    hygiene_flags = detect_account_hygiene(work[keep_mask].copy(), args.text_col, user_col, args.month_col)
    for idx, reason in hygiene_flags.items():
        if keep_mask.loc[idx]:
            keep_mask.loc[idx] = False
            removal_log.append({"row_index": int(idx), "removal_reason": reason, "duplicate_of": "", "notes": f"user_col={user_col}"})

    output_df = work[keep_mask].drop(columns=["__normalized_text"], errors="ignore").copy()
    removal_df = pd.DataFrame(removal_log)

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output_csv, index=False)
    removal_df.to_csv(args.removal_log_csv, index=False)

    summary = {
        "input_rows": int(len(df)),
        "kept_rows": int(len(output_df)),
        "removed_rows": int(len(removal_df)),
        "account_col_used": user_col,
        "removal_reason_counts": removal_df["removal_reason"].value_counts().to_dict() if not removal_df.empty else {},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
