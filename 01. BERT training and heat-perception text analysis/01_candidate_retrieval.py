#!/usr/bin/env python3
"""
01_candidate_retrieval.py

Keyword-based candidate retrieval with simple negation handling for heat-perception tweets.
Designed to approximate the manuscript/supplementary pipeline:
- lower-case
- tokenize / optional lemmatize
- curated positive lexicon across physiological / psychological / coping cues
- simple negation handling (e.g. "not hot", "no heat today")

Input: CSV with a text column (default: sentence)
Output: CSV with candidate flags and retrieval audit columns
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    _WORDNET_OK = True
except Exception:
    WordNetLemmatizer = None
    _WORDNET_OK = False


FIRST_PERSON_PATTERNS = [
    r"\bi\b", r"\bim\b", r"\bi'm\b", r"\bive\b", r"\bi've\b", r"\bme\b", r"\bmy\b", r"\bmyself\b",
    r"\bwe\b", r"\bwe're\b", r"\bours\b", r"\bus\b",
]

OUTDOOR_CONTEXT_PATTERNS = [
    r"\boutside\b", r"\boutdoors\b", r"\bwalking\b", r"\bwalked\b", r"\bbus stop\b", r"\bon the bus\b",
    r"\bon the train\b", r"\bon the street\b", r"\bin the sun\b", r"\bin traffic\b", r"\bcommute\b",
    r"\bpark\b", r"\bbeach\b", r"\bshade\b", r"\bqueue\b", r"\bline outside\b", r"\boutside at work\b",
]

NEGATION_PATTERNS = [
    r"\bnot\s+(?:too\s+)?hot\b",
    r"\bnot\s+feeling\s+hot\b",
    r"\bno\s+heat\s+today\b",
    r"\bno\s+hot\s+weather\b",
    r"\bwasn['’]?t\s+hot\b",
    r"\bisn['’]?t\s+hot\b",
    r"\baren['’]?t\s+hot\b",
    r"\bnever\s+gets\s+hot\b",
    r"\bwithout\s+heat\b",
]

# ---------------------------------------------------------------------------
# Keyword repository  (Supplementary Note S4.2)
#
# Five categories aligned with the manuscript:
#   1. ambient_heat        – heat environment / meteorological terms
#   2. physiological       – physical sensation and health symptoms
#   3. psychological       – cognitive / emotional reactions
#   4. coping_individual   – personal adaptive behaviours
#   5. coping_social       – community / infrastructure / policy coping terms
#
# Note on category 5 (coping_social):
#   These terms (cooling centres, heatwave shelters, etc.) are included in the
#   keyword repository per Supplementary Note S4.2.  Many resulting candidates
#   will subsequently be removed by Step 02's policy_advocacy_news exclusion
#   unless a personal-exposure override fires (first-person + physiological cue).
# ---------------------------------------------------------------------------
POSITIVE_LEXICON: Dict[str, List[str]] = {
    "physiological": [
        "sweat", "sweating", "sweaty", "drenched in sweat", "sticky", "clammy", "parched", "dehydrated",
        "thirsty", "dry throat", "heatstroke", "heat exhaustion", "dizzy", "lightheaded", "overheated",
        "overheating", "flushed", "red-faced", "heat stress", "heat fatigue", "heat rash", "nausea",
        "fainting", "muscle cramps", "rapid heartbeat", "breathing difficulty", "suffocating heat",
        "oppressive heat",
    ],
    "psychological": [
        "can't focus", "too hot to focus", "uncomfortable", "restless", "frustrated", "irritated", "drained",
        "sluggish", "fatigued", "mental exhaustion", "heat-induced anxiety", "overwhelmed", "short-tempered",
        "distracted", "too hot to sleep", "cannot sleep", "can't sleep", "miserable in this heat",
    ],
    "coping_individual": [
        "turned on the ac", "turned on the air conditioning", "air conditioner", "air conditioning", "ac on",
        "need ac", "seeking shade", "rest in the shade", "find cool spots", "cool down", "cold shower",
        "drink cold water", "drink iced tea", "go swimming", "jump in the pool", "portable fan", "ice pack",
        "stay hydrated", "wear light clothes", "avoid physical activity", "spray water mist", "cooling towel",
    ],
    # Social / community / infrastructure coping (Supplementary Note S4.2, Tier-4 extended)
    # Many of these will be filtered by Step 02 unless personal-exposure cues are present.
    "coping_social": [
        "cooling center", "cooling centre", "community cooling", "cooling station",
        "urban green space", "green space", "urban trees",
        "heatwave shelter", "heat shelter", "emergency shelter",
        "heat relief", "heat refuge",
        "urban tree planting", "tree canopy",
        "heat resilience", "heat action plan",
        "energy overload", "power grid",
        "heat emergency services", "heat helpline",
    ],
    "ambient_heat": [
        "hot", "heat", "heatwave", "heat wave", "sweltering", "boiling", "scorching", "blazing", "searing",
        "torrid", "sizzling", "temperature", "hottest day", "record heat", "heat index", "humidity",
        "heat dome", "extreme heat", "high temperature", "urban heat",
    ],
}


def build_term_patterns() -> Dict[str, List[re.Pattern]]:
    patterns: Dict[str, List[re.Pattern]] = {}
    for category, terms in POSITIVE_LEXICON.items():
        compiled = []
        for term in terms:
            escaped = re.escape(term).replace(r"\ ", r"\s+")
            compiled.append(re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE))
        patterns[category] = compiled
    return patterns


TERM_PATTERNS = build_term_patterns()
FIRST_PERSON_REGEXES = [re.compile(p, re.IGNORECASE) for p in FIRST_PERSON_PATTERNS]
OUTDOOR_REGEXES = [re.compile(p, re.IGNORECASE) for p in OUTDOOR_CONTEXT_PATTERNS]
NEGATION_REGEXES = [re.compile(p, re.IGNORECASE) for p in NEGATION_PATTERNS]
TOKEN_RE = re.compile(r"[a-zA-Z']+")


class SimpleLemmatizer:
    def __init__(self) -> None:
        self._lemmatizer = None
        if _WORDNET_OK:
            try:
                self._lemmatizer = WordNetLemmatizer()
            except Exception:
                self._lemmatizer = None

    def __call__(self, token: str) -> str:
        if self._lemmatizer is None:
            return token
        try:
            return self._lemmatizer.lemmatize(token)
        except Exception:
            return token


def normalize_text(text: str, lemmatizer: SimpleLemmatizer) -> Tuple[str, List[str]]:
    text = str(text or "").strip().lower()
    tokens = TOKEN_RE.findall(text)
    lemmas = [lemmatizer(tok) for tok in tokens]
    return " ".join(lemmas), lemmas


def has_any(regexes: List[re.Pattern], text: str) -> bool:
    return any(r.search(text) for r in regexes)


def collect_matches(text: str) -> Dict[str, List[str]]:
    matches: Dict[str, List[str]] = {k: [] for k in POSITIVE_LEXICON}
    for category, patterns in TERM_PATTERNS.items():
        for pattern, term in zip(patterns, POSITIVE_LEXICON[category]):
            if pattern.search(text):
                matches[category].append(term)
    return {k: v for k, v in matches.items() if v}


def candidate_decision(text: str, normalized_text: str) -> Tuple[bool, str, Dict[str, List[str]], bool, bool, bool]:
    matches = collect_matches(normalized_text)
    has_first_person = has_any(FIRST_PERSON_REGEXES, normalized_text)
    has_outdoor = has_any(OUTDOOR_REGEXES, normalized_text)
    negated = has_any(NEGATION_REGEXES, normalized_text)

    categories = set(matches.keys())
    # coping_individual is a strong individual-level signal; coping_social alone is not
    # (social coping tweets are often institutional news, filtered in Step 02)
    strong_signal = bool(categories.intersection({"physiological", "psychological", "coping_individual"}))
    ambient_only = categories == {"ambient_heat"} or ("ambient_heat" in categories and not strong_signal)

    if negated and not strong_signal:
        return False, "negated_heat_expression", matches, has_first_person, has_outdoor, negated

    if strong_signal:
        return True, "strong_positive_lexicon", matches, has_first_person, has_outdoor, negated

    if ambient_only and (has_first_person or has_outdoor):
        return True, "ambient_heat_plus_personal_or_outdoor_cue", matches, has_first_person, has_outdoor, negated

    if ambient_only:
        return False, "ambient_heat_without_personal_cue", matches, has_first_person, has_outdoor, negated

    return False, "no_positive_match", matches, has_first_person, has_outdoor, negated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--text_col", default="sentence")
    parser.add_argument("--keep_all_rows", action="store_true", help="Keep all rows with candidate flags; default writes candidates only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    if args.text_col not in df.columns:
        raise ValueError(f"Missing text column: {args.text_col}")

    lemmatizer = SimpleLemmatizer()
    out_rows = []
    for idx, row in df.iterrows():
        text = str(row.get(args.text_col, "") or "")
        normalized_text, _ = normalize_text(text, lemmatizer)
        is_candidate, reason, matches, has_first_person, has_outdoor, negated = candidate_decision(text, normalized_text)

        out = row.to_dict()
        out.update({
            "normalized_text": normalized_text,
            "retrieval_matches": json.dumps(matches, ensure_ascii=False),
            "retrieval_categories": "|".join(sorted(matches.keys())),
            "has_first_person_cue": int(has_first_person),
            "has_outdoor_cue": int(has_outdoor),
            "has_negation_pattern": int(negated),
            "candidate_flag": int(is_candidate),
            "candidate_reason": reason,
        })
        out_rows.append(out)

    out_df = pd.DataFrame(out_rows)
    if not args.keep_all_rows:
        out_df = out_df[out_df["candidate_flag"] == 1].copy()

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)

    summary = {
        "input_rows": int(len(df)),
        "output_rows": int(len(out_df)),
        "kept_candidates_only": int(not args.keep_all_rows),
        "nltk_wordnet_available": bool(_WORDNET_OK),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
