#!/usr/bin/env python3
"""
00_data_collection_stub.py

DATA COLLECTION DOCUMENTATION STUB
====================================
This script documents the Twitter/X data collection parameters used in the manuscript:

    "Global multi-city heat perception: seasonal dynamics and climate–health signals"

WHY THIS IS A STUB
------------------
The raw tweet corpus was collected via the Twitter v2 Academic Research API.
Under the Twitter/X Developer Agreement and Policy (Section II.C), tweet text and
user metadata may not be redistributed to third parties.  The full corpus therefore
cannot be shared as-is.

What IS provided instead:
  - This documented stub (exact API query parameters, filters, and volumes)
  - sample_data/demo_tweets.csv  — 184 synthetic labelled examples that demonstrate
    the pipeline's structure and annotation schema WITHOUT containing real tweet text
  - train.csv / test.csv column schema (shared on request via institutional data-sharing
    agreement in compliance with Twitter Policy)

COLLECTION PARAMETERS (as used in the study)
---------------------------------------------
Platform          : Twitter/X Academic Research API v2
Endpoint          : GET /2/tweets/search/all  (full-archive search)
Collection period : 2022-03-01 to 2023-02-28 (12 months)
Language filter   : lang:en
Geo filter        : has:geo  (geotagged tweets only)
Location scope    : bounding boxes of 347 cities across 50 countries and 6 continents
                    (city list available in Supplementary Table S1)

Initial candidate keywords (before rule filtering):
  Tier-1 (high signal):  hot, heat, heatwave, heat wave, sweltering, scorching, boiling,
                          blazing, searing, torrid, sizzling, overheated, sweat, sweating,
                          heat exhaustion, heatstroke, dehydrated, suffocating heat
  Tier-2 (physiological): dizzy, lightheaded, flushed, clammy, parched, heat rash,
                           muscle cramps, nausea, heat stress
  Tier-3 (psychological): can't focus, too hot to sleep, irritable, restless, drained,
                           sluggish, heat-induced anxiety
  Tier-4 (coping):        turned on the ac, air conditioning, cold shower, seeking shade,
                           cooling center, stay hydrated, ice pack, portable fan, cold water
  Extended via TF-IDF iterative expansion (see Supplementary Note S4.4)

Exclusions applied at query level:
  -is:retweet  (retweets removed at source)
  -is:quote    (quote tweets removed at source — note: some downstream QT filtering
                in step 03 handles edge cases not caught here)

Volume thresholds applied AFTER collection:
  - Cities with fewer than 100,000 geotagged tweets (annual) were excluded
  - Final dataset: 347 cities meeting the minimum-volume criterion

APPROXIMATE COLLECTION VOLUMES (reported in manuscript)
---------------------------------------------------------
Metric                               Value
---------------------------------   -------
Cities initially queried            > 5,000
Cities meeting eligibility          347
Countries represented               50
Continents                          6
Median total tweets per city/yr     222,201  (IQR 139,786–391,354)
Median heat-perception tweets/yr    1,490    (IQR 1,184–1,840)

HOW TO REPRODUCE THE COLLECTION (requires Academic Research access)
-------------------------------------------------------------------
Step 1 – Obtain Twitter v2 Academic Research API credentials.
Step 2 – For each city in the Supplementary Table S1 city list, run:

    GET https://api.twitter.com/2/tweets/search/all
    query params:
        query      = (<keyword_list>) lang:en has:geo bounding_box:[lon_w lat_s lon_e lat_n] -is:retweet
        start_time = 2022-03-01T00:00:00Z
        end_time   = 2023-02-28T23:59:59Z
        tweet.fields = id,text,created_at,geo,lang,author_id
        expansions   = geo.place_id
        place.fields = full_name,country,country_code,geo,place_type
        max_results  = 100
        # paginate with next_token until exhausted

Step 3 – Save each page to JSONL; normalise to CSV with columns:
    tweet_id, sentence (= text), created_at, city_id, continent, country,
    author_id, place_id, lat, lon

Step 4 – Run steps 01–07 of this pipeline on the resulting CSV.

CONTACT
-------
For access to tweet IDs (for academic hydration), contact the corresponding author.
Tweet IDs can be re-hydrated via the Twitter API to reconstruct the corpus subject
to the requester's own API access and compliance with Twitter's terms.
"""

import json
import sys

COLLECTION_PARAMS = {
    "platform": "Twitter/X Academic Research API v2",
    "endpoint": "GET /2/tweets/search/all",
    "collection_period": {"start": "2022-03-01", "end": "2023-02-28"},
    "language_filter": "lang:en",
    "geo_filter": "has:geo",
    "rt_filter": "-is:retweet",
    "initial_cities_queried": ">5000",
    "eligible_cities_final": 347,
    "countries": 50,
    "continents": 6,
    "min_annual_tweets_threshold": 100000,
    "median_total_tweets_per_city_annual": 222201,
    "iqr_total_tweets": [139786, 391354],
    "median_heat_perception_tweets_per_city_annual": 1490,
    "iqr_heat_perception_tweets": [1184, 1840],
    "inter_annotator_kappa": ">0.9",
    "note": (
        "Raw tweet text cannot be redistributed under Twitter Developer Agreement. "
        "See sample_data/demo_tweets.csv for a synthetic demonstration dataset. "
        "Tweet IDs available on request for academic hydration."
    ),
}


def main() -> None:
    print("=" * 70)
    print("Heat Perception Study — Data Collection Parameters")
    print("=" * 70)
    print(json.dumps(COLLECTION_PARAMS, ensure_ascii=False, indent=2))
    print()
    print("This is a documentation stub. See module docstring for full details.")
    print("Run the pipeline starting from step 01 with your own collected data,")
    print("or use sample_data/demo_tweets.csv for a pipeline demonstration.")


if __name__ == "__main__":
    sys.exit(main())
