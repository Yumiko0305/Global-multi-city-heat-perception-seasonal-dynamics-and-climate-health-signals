#!/usr/bin/env python3
"""
02_rule_filtering.py

Apply exclusion rules after candidate retrieval.
Rule families follow the manuscript/supplementary materials:
1) policy / advocacy / news discourse
2) metaphorical / figurative uses
3) product or media titles / commercial or technical uses
4) indoor heating / food contexts
5) proper nouns and events

Includes an override for explicit personal-exposure cues:
- first person pronouns
- deictic time/place or outdoor context
- physiological / coping signals
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

TEXT_DEFAULT = "sentence"

FIRST_PERSON = [r"\bi\b", r"\bi'm\b", r"\bme\b", r"\bmy\b", r"\bwe\b", r"\bours\b", r"\bus\b"]
OUTDOOR = [
    r"\boutside\b", r"\boutdoors\b", r"\bwalking\b", r"\bwalk home\b", r"\bwalked\b", r"\bon the bus\b",
    r"\bbus stop\b", r"\bstreet\b", r"\bshade\b", r"\bin the sun\b", r"\bcommute\b", r"\boutside at work\b",
]
DEICTIC = [r"\btoday\b", r"\bright now\b", r"\bthis afternoon\b", r"\bthis morning\b", r"\bon my way\b"]
POSITIVE_PHYSIOLOGY = [
    r"\bsweat\w*\b", r"\bdehydrat\w*\b", r"\bthirsty\b", r"\bdizzy\b", r"\blightheaded\b", r"\boverheat\w*\b",
    r"\bnausea\b", r"\bheat exhaustion\b", r"\bheat stroke\b", r"\bcan't focus\b", r"\btoo hot to sleep\b",
    r"\bcold shower\b", r"\bshade\b", r"\bac\b", r"\bair conditioning\b",
]

RULES: Dict[str, Dict[str, List[str]]] = {
    "policy_advocacy_news": {
        "policy_or_climate_discourse": [
            r"\bglobal warming\b", r"\bclimate change\b", r"\bclimate action\b", r"\bcarbon emissions\b",
            r"\bgreenhouse effect\b", r"\burban heat island\b", r"\bheat resilience\b", r"\bheat mitigation\b",
            r"\bclimate emergency\b", r"\bpolicy\b", r"\badvocacy\b",
        ],
        "news_alert_forecast": [
            r"\bheat warning\b", r"\bheat advisory\b", r"\bwarning issued\b", r"\bforecast\b", r"\bheadline\b",
            r"\bnews\b", r"\breport\b", r"\bbreaking\b", r"\brecord heat\b", r"\bhottest day\b",
            r"\bweather report\b", r"\btemperature record\b",
        ],
    },
    "metaphorical_figurative": {
        "trendiness_or_popularity": [
            r"\bhot topic\b", r"\bhot take\b", r"\btrending\b", r"\bhot and trending\b", r"\bhot off the press\b",
            r"\bcatchy tune\b", r"\bpopular\b", r"\bheat up sales\b", r"\bheating up\b",
        ],
        "attractiveness_or_emotion": [
            r"\bshe looks hot\b", r"\bhe looks hot\b", r"\bso hot\s*!?$", r"\bheated argument\b", r"\bhot mess\b",
            r"\bhot streak\b", r"\bhot seat\b",
        ],
    },
    "product_media_technical": {
        "media_titles": [
            r"\bthe movie heat\b", r"\bmovie heat\b", r"\bhot stuff\b", r"\bheat soundtrack\b", r"\balbum\b",
            r"\bsong\b", r"\bepisode\b", r"\bnetflix\b", r"\bspotify\b",
        ],
        "commercial_or_business": [
            r"\bhot deal\b", r"\bhot deals\b", r"\bselling like hotcakes\b", r"\bhot market\b", r"\bhot stock\b",
            r"\bhot property\b", r"\bhot offers\b", r"\bhot leads\b",
        ],
        "scientific_or_technical": [
            r"\bheat transfer\b", r"\blatent heat\b", r"\bspecific heat\b", r"\bheat exchanger\b",
            r"\bheat capacity\b", r"\bgpu overheating\b", r"\bhot swapping\b", r"\bheat pump\b",
        ],
    },
    "indoor_heating_food": {
        "indoor_temperature_control": [
            r"\bheater\b", r"\bradiator\b", r"\bcentral heating\b", r"\bwarm indoors\b", r"\bturn down the radiator\b",
            r"\bhome into a sauna\b",
        ],
        "food_or_drink": [
            r"\bhot pot\b", r"\bhot sauce\b", r"\bhot wings\b", r"\bhot chocolate\b", r"\bhot & sour\b",
            r"\bspicy heat\b", r"\bhot cider\b", r"\bhot cross buns\b", r"\bhot tamales\b",
        ],
    },
    "proper_nouns_events": {
        "sports_charts_events": [
            r"\bmiami heat\b", r"\bbillboard hot 100\b", r"\bheat wave festival\b", r"\bhot springs\b",
            r"\bhot air balloon festival\b",
        ],
        "other_named_entities": [
            r"\bdeath valley\b", r"\bheat magazine\b",
        ],
    },
}


FIRST_PERSON_RE = [re.compile(p, re.IGNORECASE) for p in FIRST_PERSON]
OUTDOOR_RE = [re.compile(p, re.IGNORECASE) for p in OUTDOOR]
DEICTIC_RE = [re.compile(p, re.IGNORECASE) for p in DEICTIC]
POSITIVE_RE = [re.compile(p, re.IGNORECASE) for p in POSITIVE_PHYSIOLOGY]
COMPILED_RULES = {
    fam: {sub: [re.compile(p, re.IGNORECASE) for p in pats] for sub, pats in submap.items()}
    for fam, submap in RULES.items()
}


def has_any(regexes: List[re.Pattern], text: str) -> bool:
    return any(r.search(text) for r in regexes)


def explicit_personal_exposure(text: str) -> bool:
    strong_person = has_any(FIRST_PERSON_RE, text)
    context = has_any(OUTDOOR_RE, text) or has_any(DEICTIC_RE, text)
    heat_signal = has_any(POSITIVE_RE, text)
    return strong_person and (context or heat_signal)


def apply_rules(text: str) -> Tuple[bool, str, str, List[str]]:
    matches = []
    for family, subrules in COMPILED_RULES.items():
        for subcat, patterns in subrules.items():
            for pat in patterns:
                if pat.search(text):
                    matches.append((family, subcat, pat.pattern))
    if not matches:
        return True, "pass", "", []

    if explicit_personal_exposure(text):
        return True, "override_explicit_personal_exposure", "", [f"{a}:{b}" for a, b, _ in matches]

    first_family, first_subcat, _ = matches[0]
    return False, first_family, first_subcat, [f"{a}:{b}" for a, b, _ in matches]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--audit_csv", required=True, help="Write filtered-out rows / decisions here")
    parser.add_argument("--text_col", default=TEXT_DEFAULT)
    parser.add_argument("--keep_all_rows", action="store_true", help="Keep all rows in output with pass_flag; default keeps passed rows only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    if args.text_col not in df.columns:
        raise ValueError(f"Missing text column: {args.text_col}")

    all_rows = []
    for _, row in df.iterrows():
        text = str(row.get(args.text_col, "") or "").lower()
        passed, family, subcat, matched_rules = apply_rules(text)
        out = row.to_dict()
        out.update({
            "rule_filter_pass": int(passed),
            "rule_filter_decision": family if family else "pass",
            "rule_filter_subcategory": subcat,
            "rule_filter_matches": json.dumps(matched_rules, ensure_ascii=False),
            "explicit_personal_exposure": int(explicit_personal_exposure(text)),
        })
        all_rows.append(out)

    out_df = pd.DataFrame(all_rows)
    passed_df = out_df if args.keep_all_rows else out_df[out_df["rule_filter_pass"] == 1].copy()
    audit_df = out_df[out_df["rule_filter_pass"] == 0].copy()

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    passed_df.to_csv(args.output_csv, index=False)
    audit_df.to_csv(args.audit_csv, index=False)

    summary = {
        "input_rows": int(len(df)),
        "passed_rows": int((out_df["rule_filter_pass"] == 1).sum()),
        "filtered_rows": int((out_df["rule_filter_pass"] == 0).sum()),
        "top_decisions": out_df["rule_filter_decision"].value_counts().head(10).to_dict(),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
