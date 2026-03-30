# 02. Heat-perception index construction

This module converts tweet-level heat-perception classification outputs into city-day, city-month, and city-season indices used in the manuscript. It implements HPII, HPVI, HPPI, seasonal typologies, counter-seasonal summaries, and tweet-volume strata.

## Files

- `07_aggregate_city_day_indices.py`: main aggregation and index-construction script.
- `requirements.txt`: minimal Python dependencies.
- `sample_data/synthetic_tweet_level_predictions_620.csv`: synthetic demo input with 620 tweet-level records.
- `sample_outputs/`: reference outputs generated from the synthetic demo input.

## Inputs

The script expects a tweet-level CSV that contains, at minimum:

- `city_id`: city identifier
- `pred_label` (or `label`): binary heat-perception indicator
- `date` or `month`: posting date or month

Recommended metadata columns:

- `continent`
- `country`
- `hemisphere`

Optional denominator sources:

- external city-day totals via `--total_posts_csv`
- per-row denominator column via `--total_posts_col`

If authentic dates or authentic city-day denominators are unavailable, the script can still run by constructing deterministic synthetic dates and/or using the observed input rows as a proxy denominator. These fallbacks are useful for code verification but should not be treated as research-grade substitutes for real city-day totals.

## Outputs

The script writes the following outputs:

- `city_day_indices.csv`
- `city_month_indices.csv`
- `city_season_indices.csv`
- `city_typology_summary.csv`
- `continent_season_hppi.csv`
- `tweet_volume_strata.csv`
- `eligibility_summary.csv`
- `aggregate_summary.json`

## Methods alignment

### HPII

HPII is computed as the proportion of heat-perception posts among total posts in a city-season. The script exports both the proportion scale and the percent scale. Seasonal HPII is classified into seven levels (I–VII) using Jenks natural breaks on the pooled city-season distribution.

### HPVI

HPVI is computed from the 7-day smoothed city-day HPII series as:

`HPVI = sd / (abs(mean) + epsilon)`

The script exports:

- `hpvi`: ratio scale
- `hpvi_percent`: percent scale (`hpvi * 100`)

HPVI categories are assigned on the ratio scale:

- `low`: `<= 0.10`
- `moderate`: `> 0.10` to `<= 0.30`
- `high`: `> 0.30`

These are equivalent to 10% and 30% thresholds on the percent scale.

### HPPI

HPPI is computed within each continent-season using a normalized Esteban-Ray-style polarization measure. The default `alpha` is `1.6`.

### Typologies

Each city-season is classified into one of four categories:

- `Perception Hotspots`
- `Stable High`
- `Stable Low`
- `Variability Low`

## Important code corrections in this version

This version includes two manuscript-critical fixes:

1. **Eligibility coverage bug fixed**: seasonal coverage is now computed against the expected calendar-day coverage within each city's actual observed date span, rather than dividing identical counts.
2. **HPVI unit clarification**: HPVI is now stored explicitly on both the ratio scale and the percent scale, avoiding ambiguity between formula notation and threshold notation.

## Example usage

Run the synthetic demo:

```bash
python 07_aggregate_city_day_indices.py \
  --input_csv sample_data/synthetic_tweet_level_predictions_620.csv \
  --output_dir sample_outputs
```

Example with manuscript-style options:

```bash
python 07_aggregate_city_day_indices.py \
  --input_csv /path/to/tweet_level_predictions.csv \
  --output_dir /path/to/output_dir \
  --city_col city_id \
  --continent_col continent \
  --country_col country \
  --hemisphere_col hemisphere \
  --date_col date \
  --indicator_col pred_label \
  --positive_label 1 \
  --hppi_alpha 1.6 \
  --high_hpii_min_level 5 \
  --apply_eligibility_filters \
  --min_heat_posts 1000 \
  --min_active_days 50 \
  --min_seasonal_coverage 0.15
```

## Notes for repository integration

If this module is placed inside a larger reproducibility repository, it should be positioned between:

1. text classification of heat-perception tweets, and
2. downstream health-outcome analyses.

In that workflow, the input to this module is the classified tweet-level output from the text-analysis pipeline, and the outputs here feed directly into the health-outcome modeling scripts.
