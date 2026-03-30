# sample_outputs — Reference Output Files

This directory contains reference output files generated from the **real corpus**
(74,938 labelled tweets, 347 analytical cities, 50 countries, 6 continents).
Note: `sample_data/cities_reference.csv` contains 348 entries (the full reference
pool before final eligibility filtering reduced the analytical set to 347 cities).

---

## monthly_test_metrics_reference.csv

**Source:** Final BERT model trained on the full 52,945-tweet training set,
evaluated on the 21,993-tweet temporal holdout test set, stratified by
calendar month.

**What it shows:** The per-month Accuracy / Recall / F1 series plotted in
**Supplementary Figure S7**.

| Column | Description |
|--------|-------------|
| `month` | Calendar month (YYYY-MM) |
| `n_test` | Labelled test tweets in that month |
| `accuracy` | Proportion correctly classified |
| `precision` | Positive-class precision |
| `recall` | Positive-class recall |
| `f1` | Harmonic mean of precision and recall |

**Key patterns (matching Figure S7):**
- F1 peaks in northern-hemisphere summer (Jun–Aug): ~0.91–0.93
- F1 lowest in winter months (Dec–Feb): ~0.87–0.88
- Mean F1 across all 12 months: ~0.886

> **Note:** Running the demo pipeline on `sample_data/demo_tweets.csv` will produce
> a `monthly_test_metrics.csv` with lower values, because the 620-example synthetic
> training set is far smaller than the real 52,945-tweet corpus.
> The output file structure is identical; only the metric magnitudes differ.
