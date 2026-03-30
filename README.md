# Global Multi-City Heat Perception: Seasonal Dynamics and Climate–Health Signals

**Manuscript:** You, M., Guan, C., Guo, Y., et al. (2025). Global multi-city heat perception: seasonal dynamics and climate–health signals. *Nature Communications*.

This repository contains the complete analysis code supporting the manuscript. It is organised into two self-contained modules, each with its own README, requirements, and sample data.

---

## Repository Structure

```
01. BERT training and heat-perception text analysis/
    ├── README.md          ← Full pipeline documentation
    ├── requirements.txt
    ├── LICENSE
    ├── 00_data_collection_stub.py
    ├── 01_candidate_retrieval.py
    ├── 02_rule_filtering.py
    ├── 03_deduplicate_and_account_hygiene.py
    ├── 04_build_gold_labels.py
    ├── 05_train_grouped_cv.py
    ├── 06_validate_and_audit.py
    ├── 07_aggregate_city_day_indices.py
    ├── sample_data/          ← Synthetic demo dataset (620 tweets)
    └── sample_outputs/       ← Reference output for verification

02. Health-outcome analyses/
    ├── README.md          ← Full analysis documentation
    ├── 01_model_comparison_warmseason.R
    ├── 02_joint_exposure_warmseason.R
    ├── 03_model_comparison_allyear.R
    ├── 04_joint_exposure_allyear.R
    ├── 05_figures_warmseason.R
    └── 06_figures_allyear.R
```

---

## Module 1 — BERT Training and Heat-Perception Text Analysis

**Code owners:** Meizi You, Xiyuan Ren
**Data owners:** Meizi You, ChengHe Guan

Transforms a geotagged tweet corpus into city-level heat-perception indices (HPII, HPVI, HPPI) via keyword retrieval, rule filtering, deduplication, BERT fine-tuning, and index aggregation.

A synthetic demo dataset (no real tweets) is included so the full pipeline can be run without Twitter API access. See [`01. BERT training and heat-perception text analysis/README.md`](01.%20BERT%20training%20and%20heat-perception%20text%20analysis/README.md) for quick-start instructions.

**Requirements:** Python ≥ 3.9, pandas, numpy, scikit-learn, torch ≥ 2.0, transformers ≥ 4.35

---

## Module 2 — Health-Outcome Analyses

**Code owners:** Zhihu Xu, Meizi You
**Data owners:** Yuming Guo

Compares temperature-based and heat-perception-based DLNM models for health outcomes (emergency department visits, hospitalisation, mortality) across 7 cities in Australia, Brazil, and Mexico.

Input data (`final_dat.rds`) is available upon request via a data-sharing agreement with the respective data custodians. See [`02. Health-outcome analyses/README.md`](02.%20Health-outcome%20analyses/README.md) for details.

**Requirements:** R ≥ 4.3.3, dlnm 2.4.7, mixmeta 1.2.2, tidyverse packages

---

## Data Availability

Raw tweet text cannot be redistributed under the Twitter/X Developer Agreement. See Module 1 README for details on what is shared and how to request tweet IDs for re-hydration.

Health outcome data are available on request from the respective data custodians.

---

## License

MIT License — see [LICENSE](LICENSE).

---

## Contact

- Meizi You: my2683@nyu.edu | meizi.you@monash.edu
- Zhihu Xu: zhihu.xu@monash.edu
