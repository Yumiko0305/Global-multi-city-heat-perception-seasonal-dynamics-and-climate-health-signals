#!/usr/bin/env python3
"""
04_build_gold_labels.py

Build a gold-labeled dataset schema that is runnable end-to-end even when upstream metadata are missing.
Important: when tweet_id / city_id / continent / country / month are absent, this script fills them with
DETERMINISTIC SYNTHETIC PLACEHOLDERS. These placeholders make the code runnable, but they are not authentic metadata.

Features:
- accepts train/test CSVs or one combined CSV
- standardizes columns
- assigns synthetic tweet IDs if absent
- assigns synthetic city/continent/country/month in a label-aware, deterministic way
- optionally appends synthetic boundary-case examples derived from the supplementary rules
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


CONTINENT_COUNTRY_POOL: Dict[str, List[str]] = {
    "Africa": ["EGY", "KEN", "ZAF"],
    "Asia": ["IND", "JPN", "PHL"],
    "Europe": ["DEU", "ESP", "GBR"],
    "North America": ["CAN", "MEX", "USA"],
    "South America": ["ARG", "BRA", "COL"],
    "Oceania": ["AUS", "NZL", "FJI"],
}
MONTH_POOL = pd.period_range("2022-03", "2023-02", freq="M").astype(str).tolist()

SYNTHETIC_BOUNDARY_EXAMPLES = [
    # positive examples
    {"sentence": "I am sweating and dizzy after walking outside in this heat.", "label": 1, "example_type": "positive_physiological"},
    {"sentence": "They issued another heat warning and I already feel sick walking outside.", "label": 1, "example_type": "positive_alert_plus_personal_experience"},
    {"sentence": "Turned on the AC and still cannot sleep because it is too hot tonight.", "label": 1, "example_type": "positive_coping_plus_experience"},
    {"sentence": "Waiting at the bus stop and I am overheating.", "label": 1, "example_type": "positive_outdoor_exposure"},
    # negatives mirroring supplementary exclusion examples
    {"sentence": "#HotTopic: Everyone is talking about it right now.", "label": 0, "example_type": "negative_metaphorical"},
    {"sentence": "The movie Heat is such a classic!", "label": 0, "example_type": "negative_media_title"},
    {"sentence": "The Miami Heat is playing really well this season.", "label": 0, "example_type": "negative_proper_noun_event"},
    {"sentence": "Central heating is a lifesaver but my apartment feels like a sauna.", "label": 0, "example_type": "negative_indoor_heating"},
    {"sentence": "Hot wings with extra heat are the best.", "label": 0, "example_type": "negative_food"},
    {"sentence": "Global warming is the hottest policy topic this year.", "label": 0, "example_type": "negative_policy_discourse"},
    {"sentence": "Heat transfer is critical in this cooling design.", "label": 0, "example_type": "negative_technical"},
    {"sentence": "Check out this hot deal on air conditioners!", "label": 0, "example_type": "negative_commercial"},
]


@dataclass
class Args:
    train_csv: Optional[str]
    test_csv: Optional[str]
    input_csv: Optional[str]
    output_csv: str
    boundary_examples_csv: Optional[str]
    seed: int
    synthetic_city_count: int
    augment_boundary_examples: bool


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default=None)
    parser.add_argument("--test_csv", default=None)
    parser.add_argument("--input_csv", default=None)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--boundary_examples_csv", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic_city_count", type=int, default=60)
    parser.add_argument("--augment_boundary_examples", action="store_true")
    ns = parser.parse_args()
    return Args(**vars(ns))


def read_inputs(args: Args) -> pd.DataFrame:
    frames = []
    if args.input_csv:
        df = pd.read_csv(args.input_csv)
        if "original_split" not in df.columns:
            df["original_split"] = "combined"
        frames.append(df)
    else:
        if not args.train_csv or not args.test_csv:
            raise ValueError("Provide --input_csv or both --train_csv and --test_csv")
        train_df = pd.read_csv(args.train_csv)
        train_df["original_split"] = "train"
        test_df = pd.read_csv(args.test_csv)
        test_df["original_split"] = "test"
        frames.extend([train_df, test_df])
    df = pd.concat(frames, ignore_index=True)
    if "sentence" not in df.columns or "label" not in df.columns:
        raise ValueError("Input must contain at least sentence and label columns")
    df["sentence"] = df["sentence"].astype(str)
    df["label"] = df["label"].astype(int)
    return df


def make_synthetic_city_pool(city_count: int) -> List[Tuple[str, str, str]]:
    pool = []
    continents = list(CONTINENT_COUNTRY_POOL.keys())
    per_continent = max(1, city_count // len(continents))
    city_counter = 1
    for continent in continents:
        countries = CONTINENT_COUNTRY_POOL[continent]
        for i in range(per_continent):
            country = countries[i % len(countries)]
            city_id = f"SYN_{country}_CITY_{city_counter:03d}"
            pool.append((city_id, continent, country))
            city_counter += 1
    while len(pool) < city_count:
        continent = continents[len(pool) % len(continents)]
        country = CONTINENT_COUNTRY_POOL[continent][len(pool) % len(CONTINENT_COUNTRY_POOL[continent])]
        city_id = f"SYN_{country}_CITY_{city_counter:03d}"
        pool.append((city_id, continent, country))
        city_counter += 1
    return pool[:city_count]


def assign_synthetic_metadata(df: pd.DataFrame, seed: int, city_count: int) -> pd.DataFrame:
    rng = random.Random(seed)
    out = df.copy().reset_index(drop=True)
    city_pool = make_synthetic_city_pool(city_count)

    # label-aware round robin assignment to avoid pathological group-label separation
    out["__row_id"] = range(len(out))
    city_assignments = [None] * len(out)
    month_assignments = [None] * len(out)
    tweet_ids = [None] * len(out)

    for label_value, group in out.groupby("label"):
        idxs = list(group.index)
        rng.shuffle(idxs)
        for j, idx in enumerate(idxs):
            city_id, continent, country = city_pool[j % len(city_pool)]
            month = MONTH_POOL[(j + label_value) % len(MONTH_POOL)]
            city_assignments[idx] = (city_id, continent, country)
            month_assignments[idx] = month

    out["tweet_id"] = [f"{100001 + i}" for i in range(len(out))]
    out["city_id"] = [city_assignments[i][0] for i in range(len(out))]
    out["continent"] = [city_assignments[i][1] for i in range(len(out))]
    out["country"] = [city_assignments[i][2] for i in range(len(out))]
    out["month"] = month_assignments
    out["source_platform"] = out.get("source_platform", pd.Series(["X"] * len(out)))
    out["metadata_is_synthetic"] = 1
    out["needs_metadata_backfill"] = 1
    out["is_synthetic_example"] = 0
    out["synthetic_example_type"] = ""
    return out.drop(columns=["__row_id"], errors="ignore")


def build_boundary_examples(start_id: int, city_count: int) -> pd.DataFrame:
    city_pool = make_synthetic_city_pool(city_count)
    rows = []
    for i, ex in enumerate(SYNTHETIC_BOUNDARY_EXAMPLES):
        city_id, continent, country = city_pool[i % len(city_pool)]
        rows.append({
            "sample_id": f"synthetic_boundary_{i+1:03d}",
            "tweet_id": f"{start_id + i}",
            "sentence": ex["sentence"],
            "label": ex["label"],
            "city_id": city_id,
            "continent": continent,
            "country": country,
            "month": MONTH_POOL[i % len(MONTH_POOL)],
            "original_split": "synthetic_boundary",
            "source_platform": "X",
            "metadata_is_synthetic": 1,
            "needs_metadata_backfill": 1,
            "is_synthetic_example": 1,
            "synthetic_example_type": ex["example_type"],
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    df = read_inputs(args)
    df = assign_synthetic_metadata(df, seed=args.seed, city_count=args.synthetic_city_count)
    df.insert(0, "sample_id", [f"sample_{i+1:06d}" for i in range(len(df))])

    boundary_df = pd.DataFrame()
    if args.augment_boundary_examples:
        boundary_df = build_boundary_examples(start_id=900001, city_count=args.synthetic_city_count)
        df = pd.concat([df, boundary_df], ignore_index=True)

    preferred_cols = [
        "sample_id", "tweet_id", "sentence", "label", "city_id", "continent", "country", "month",
        "original_split", "source_platform", "metadata_is_synthetic", "needs_metadata_backfill",
        "is_synthetic_example", "synthetic_example_type",
    ]
    other_cols = [c for c in df.columns if c not in preferred_cols]
    df = df[preferred_cols + other_cols]

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    if args.boundary_examples_csv:
        boundary_df.to_csv(args.boundary_examples_csv, index=False)

    summary = {
        "output_rows": int(len(df)),
        "synthetic_boundary_examples_added": int(len(boundary_df)),
        "n_unique_cities": int(df["city_id"].nunique()),
        "continent_counts": df["continent"].value_counts().to_dict(),
        "month_counts": df["month"].value_counts().to_dict(),
        "note": "city_id/continent/country/month/tweet_id are synthetic placeholders unless you backfill real metadata",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
