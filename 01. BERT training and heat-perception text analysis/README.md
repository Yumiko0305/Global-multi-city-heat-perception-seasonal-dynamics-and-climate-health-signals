# Heat Perception Tweet Classification Pipeline

**Manuscript:** *Global multi-city heat perception: seasonal dynamics and climate–health signals*
**Journal:** Nature Communications


---

## Overview

This repository contains the complete code pipeline for identifying heat-perception tweets and computing city-level heat-perception indices, as described in the manuscript.  The pipeline transforms a raw geotagged tweet corpus into the HPII, HPVI, and HPPI indices reported in the paper.

```
00_data_collection_stub.py          ← Documents Twitter API query parameters
01_candidate_retrieval.py           ← Keyword retrieval + negation handling
02_rule_filtering.py                ← Exclusion rule families (5 categories)
03_deduplicate_and_account_hygiene.py  ← RT/QT removal, deduplication
04_build_gold_labels.py             ← Gold-label schema builder
05_train_grouped_cv.py              ← BERT training: grouped CV + monthly eval
06_validate_and_audit.py            ← Holdout evaluation + monthly audit export
sample_data/
  generate_demo_sample.py           ← Generates the synthetic demo dataset
  demo_tweets.csv                   ← 620 synthetic labelled examples (see below)
  demo_metadata.json                ← Generation provenance record
```

---

## Data Availability and Twitter ToS Compliance

The original corpus was collected via the **Twitter/X Academic Research API v2** and contains 12 months of geotagged English tweets from 347 cities.

**Why raw tweet text is not provided here:**
Under the [Twitter/X Developer Agreement and Policy (Section II.C)](https://developer.twitter.com/en/developer-terms/agreement-and-policy), tweet text and user metadata may not be redistributed to third parties.

**What IS provided:**

| Item | Description |
|------|-------------|
| `sample_data/demo_tweets.csv` | 620 **synthetic** labelled examples covering all annotation categories. Suitable for running the full pipeline end-to-end. **NOT for scientific inference.** |
| `00_data_collection_stub.py` | Exact API query parameters, keyword tiers, city-eligibility thresholds, and reproduction instructions. |
| `train.csv` / `test.csv` schema | Available on request via institutional data-sharing agreement. Column schema: `label`, `sentence`. |
| Tweet IDs | Available on request for academic re-hydration via the Twitter API. |

---

## Quick Start — Demo Pipeline (Synthetic Data)

Run the entire pipeline on the provided synthetic demo dataset.  No Twitter API access required.

```bash
# 0. Generate the demo dataset (already committed; re-run to verify reproducibility)
python sample_data/generate_demo_sample.py --output_dir sample_data

# 1. Candidate retrieval
python 01_candidate_retrieval.py \
  --input_csv sample_data/demo_tweets.csv \
  --output_csv outputs/01_candidates.csv \
  --keep_all_rows

# 2. Rule filtering
python 02_rule_filtering.py \
  --input_csv outputs/01_candidates.csv \
  --output_csv outputs/02_filtered.csv \
  --audit_csv outputs/02_audit.csv \
  --keep_all_rows

# 3. Deduplication + account hygiene
python 03_deduplicate_and_account_hygiene.py \
  --input_csv outputs/02_filtered.csv \
  --output_csv outputs/03_deduped.csv \
  --removal_log_csv outputs/03_removal_log.csv

# 4. Build gold-label schema
python 04_build_gold_labels.py \
  --input_csv outputs/03_deduped.csv \
  --output_csv outputs/04_gold_labeled.csv \
  --augment_boundary_examples

# 5. Train BERT with grouped CV + monthly holdout evaluation
#    Split demo data into train/test by original_split column first
python 05_train_grouped_cv.py \
  --data_path outputs/04_gold_labeled.csv \
  --output_dir outputs/05_model \
  --model_name_or_path bert-base-uncased \
  --num_folds 3

# 5b. Monthly evaluation (reproduces Supplementary Figure S7)
#     Pass a test CSV with a 'month' column
python 05_train_grouped_cv.py \
  --data_path outputs/04_gold_labeled.csv \
  --output_dir outputs/05_model \
  --model_name_or_path bert-base-uncased \
  --num_folds 3 \
  --temporal_test_csv sample_data/demo_tweets.csv

# 6. Validate and audit
python 06_validate_and_audit.py \
  --model_dir outputs/05_model/final_model \
  --input_csv outputs/04_gold_labeled.csv \
  --output_dir outputs/06_audit

# 7. Aggregate city-day indices (HPII / HPVI / HPPI / typology)
python 07_aggregate_city_day_indices.py \
  --input_csv outputs/06_audit/predictions.csv \
  --output_dir outputs/07_indices
```

---

## Full-Data Workflow (With Real Corpus)

When running on an authenticated Twitter corpus, start from step 01 with the full raw CSV:

```bash
python 01_candidate_retrieval.py \
  --input_csv /path/to/raw_tweets.csv \
  --output_csv outputs/01_candidates.csv

# … then steps 02–07 as above
```

For the monthly evaluation that reproduces Figure S7, provide the held-out test split:

```bash
python 05_train_grouped_cv.py \
  --data_path train.csv \
  --output_dir outputs/model_full \
  --model_name_or_path bert-base-uncased \
  --temporal_test_csv test.csv
```

This writes `outputs/model_full/monthly_test_metrics.csv` with per-month
Accuracy / Recall / F1 values matching Supplementary Figure S7.

---

## Pipeline Steps in Detail

### Step 00 — Data Collection (documentation only)
`00_data_collection_stub.py` documents the exact Twitter API parameters, keyword tiers, and eligibility thresholds used during data collection.  See the module docstring for hydration instructions.

### Step 01 — Candidate Retrieval
Keyword-based filtering with five signal categories (Supplementary Note S4.2):

| Category | Example terms |
|----------|---------------|
| Physiological | sweat, heatstroke, heat exhaustion, dehydrated, dizzy, overheated |
| Psychological | can't focus, too hot to sleep, irritable, heat-induced anxiety |
| Coping (individual) | turned on the AC, cold shower, seeking shade, ice pack |
| Coping (social / infrastructure) | cooling centre, heatwave shelter, urban green space, heat resilience |
| Ambient heat + personal cue | hot, scorching, heatwave, heat dome (only when paired with first-person or outdoor context) |

Social/infrastructure coping terms are included in the keyword repository per Supplementary Note S4.2.  Most resulting candidates will be removed by Step 02's `policy_advocacy_news` exclusion unless a personal-exposure override fires.

Simple negation patterns (e.g., "not hot", "wasn't hot") are detected and suppress candidate inclusion unless strong physiological/individual-coping signals override.

### Step 02 — Rule Filtering
Five exclusion families (Supplementary Note S4.3):

| Family | Examples |
|--------|---------|
| Policy / advocacy / news | global warming, climate action, heat warning, weather forecast |
| Metaphorical / figurative | hot topic, hot take, heated argument, she looks hot |
| Product / media / commercial | movie Heat, hot deals, hot stock, GPU overheating |
| Indoor heating / food | central heating, radiator, hot pot, hot sauce |
| Proper nouns / events | Miami Heat, Billboard Hot 100, Hot Springs |

An explicit personal-exposure override reinstates tweets that contain first-person pronouns plus a physiological/coping signal, even if an exclusion rule fires.

### Step 03 — Deduplication & Account Hygiene
- Explicit retweet/quote-tweet removal (`^RT @`, `^QT @`)
- Exact normalized-text deduplication
- Near-duplicate collapse: 5-gram shingles, bucketed Jaccard approximation (threshold ≥ 0.85, approximating the MinHash protocol described in the manuscript)
- Templated/promotional account detection (duplicate-posting ratio ≥ 0.80 or promotional-keyword ratio ≥ 0.60)

> **Note on MinHash:** The manuscript describes full MinHash with 5-gram shingles and Jaccard ≥ 0.85. This implementation uses a bucket-key approximation for dependency-free execution. Results are equivalent for the deduplication rate reported; full MinHash (e.g., `datasketch`) can be substituted by replacing `detect_near_duplicates()`.

### Step 04 — Gold-Label Schema
Accepts `sentence` + `label` columns (the output of human annotation with Cohen's κ > 0.9; see Supplementary Note S4.4) and adds the metadata columns required by downstream steps.  When authentic `city_id / month / tweet_id` are absent, **deterministic synthetic placeholders** are assigned (`metadata_is_synthetic = 1`).  These must be replaced with real metadata before scientific interpretation.

### Step 05 — BERT Training
- Model: `bert-base-uncased` fine-tuned for binary sequence classification
- Training: grouped 5-fold cross-validation, groups = `city_id` (ensures each fold's validation cities are unseen during training)
- Loss: `CrossEntropyLoss` with class-balanced weights
- Hyperparameters: lr = 2e-5, batch = 32, epochs = 3, max_length = 140, weight_decay = 0.01
- **Monthly holdout evaluation** (`--temporal_test_csv`): after final-model training, evaluates on each calendar month separately, producing the Accuracy / Recall / F1 series in Supplementary Figure S7

### Step 06 — Validation & Audit
- Overall and monthly performance metrics on a holdout set
- Stratified monthly audit sample (500 per month, continent-proportional) for manual spot-check, matching the quality-assurance protocol described in the manuscript

---

## Requirements

```
python >= 3.9
pandas
numpy
scikit-learn
torch >= 2.0
transformers >= 4.35
nltk (optional, for WordNet lemmatization in step 01)
```

Install:
```bash
pip install pandas numpy scikit-learn torch transformers nltk
```

---

## Annotation Protocol Summary

- Annotators: three independent annotators with domain expertise
- Inter-annotator reliability: Cohen's κ > 0.9 (Supplementary Note S4.4)
- Disagreements resolved by majority vote with expert adjudication
- Multiple iterative rounds with TF-IDF-guided keyword expansion
- Training corpus: 52,945 tweets; Test corpus: 21,993 tweets (temporal split)
- Label distribution: ~38% positive (heat perception), ~62% negative

---

## Limitations and Data Availability

### Tweet corpus
The full labelled tweet corpus (74,938 tweets, 347 cities, 50 countries) was collected
via the **Twitter/X Academic Research API v2** and is subject to the
[Twitter/X Developer Agreement and Policy (Section II.C)](https://developer.twitter.com/en/developer-terms/agreement-and-policy),
which prohibits redistribution of tweet text or user metadata to third parties.

**What is shared in this repository:**

| Item | Status |
|------|--------|
| Complete pipeline code (Steps 00–07) | Included |
| Synthetic demo dataset (`sample_data/demo_tweets.csv`, 620 labelled examples) | Included |
| Reference city metadata (`sample_data/cities_reference.csv`, 348 cities) | Included |
| Reference monthly metrics (`sample_outputs/monthly_test_metrics_reference.csv`) | Included |
| Raw tweet text or user metadata | **Not included** (Twitter ToS) |
| `train.csv` / `test.csv` (real labelled corpus) | Available on request (institutional data-sharing agreement) |
| Tweet IDs for academic re-hydration | Available on request |

### Aggregate statistics only
City-level indices (HPII, HPVI, HPPI) derived from the real corpus are reported in
the manuscript tables and figures.  Only continent-level aggregate patterns are
described in the main text; fine-grained city-level outputs are not distributed here
to minimise any residual privacy risk.

### Synthetic demo dataset
`sample_data/demo_tweets.csv` contains **620 fully original synthetic sentences**
written specifically for this repository.  No sentence is derived from or identical
to any real tweet in the corpus.  It is flagged `is_synthetic = 1` throughout and
**must not be used for scientific inference**.  Its sole purpose is to allow
reviewers and readers to execute and inspect all pipeline steps end-to-end.

### Inter-annotator reliability
The Cohen's κ > 0.9 figure (Supplementary Note S4.4) was obtained during the manual
annotation phase of the study and is a property of the human labelling process, not
of the code.  It cannot be reproduced by running this pipeline; the annotation
protocol is documented in Supplementary Note S4.4 for transparency.

### Near-duplicate detection (Step 03)
The manuscript describes full MinHash with 5-gram shingles (Jaccard ≥ 0.85). This
implementation uses a bucket-key approximation that does not require the `datasketch`
library. Results are equivalent for the deduplication rate reported; `datasketch`-based
MinHash can be substituted by replacing `detect_near_duplicates()` in Step 03.

---

## Citation

If you use this code, please cite:

> You, M., Guan, C., Guo, Y., et al. (2025). Global multi-city heat perception: seasonal dynamics and climate–health signals. *Nature Communications*.
