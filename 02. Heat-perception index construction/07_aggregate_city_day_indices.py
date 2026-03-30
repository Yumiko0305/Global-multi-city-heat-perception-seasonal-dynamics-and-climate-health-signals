
#!/usr/bin/env python3
"""
07_aggregate_city_day_indices_nc_ready.py

Aggregate tweet-level labels/predictions into city-day indices and downstream seasonal metrics:
- HPII: city-day share of heat-perception tweets; seasonal HPII aggregated by city-season
- HPVI: coefficient of variation of the 7-day-smoothed daily HPII series within each city-season
- HPPI: continent-season polarization index, implemented as a normalized Esteban-Ray-style measure
  over city-level HPII or HPVI signals
- city typology outputs: peak season, total dominant season, both-coincide indicator,
  multi-season high-HPII indicator, counter-seasonal flags
- tweet-volume strata: low / medium / high (bottom 30% / middle 40% / top 30%)

Important caveat:
If authentic date or denominator data are absent, the script creates deterministic synthetic placeholders:
- dates are synthesized from month + within-city-month row order
- denominator defaults to the number of rows in the provided input per city-day
These placeholders make the pipeline runnable but are NOT research-grade substitutes for real city-day totals.

Nature Communications code-alignment notes:
- HPII is output both as a proportion and as percent. Seven Jenks levels (I-VII) are computed
  on the pooled city-season HPII distribution, matching the manuscript definition.
- HPVI is computed as a ratio with epsilon stabilization, and an explicit percent version is also
  exported. Tiering uses <=0.10, >0.10 to <=0.30, and >0.30 (equivalently 10% / 30%).
- The city eligibility seasonal coverage bug in the earlier draft has been fixed by comparing
  observed days against the expected calendar-day coverage within each city's actual date span.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


MONTH_NAME_TO_NUM = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

COUNTRY_HEMISPHERE = {
    "AUS": "south", "NZL": "south", "FJI": "south",
    "ARG": "south", "BRA": "south", "CHL": "south", "URY": "south", "PRY": "south",
    "ZAF": "south", "KEN": "north", "EGY": "north", "COL": "north",
    "USA": "north", "CAN": "north", "MEX": "north",
    "GBR": "north", "DEU": "north", "ESP": "north", "FRA": "north", "ITA": "north",
    "IND": "north", "JPN": "north", "PHL": "north", "CHN": "north",
}


@dataclass
class Config:
    input_csv: str
    output_dir: str
    total_posts_csv: Optional[str]
    city_col: str
    continent_col: str
    country_col: str
    hemisphere_col: str
    date_col: str
    month_col: str
    indicator_col: str
    positive_label: int
    total_posts_col: Optional[str]
    normalize_within_city_month: bool
    event_flag_col: Optional[str]
    winsor_lower: float
    winsor_upper: float
    hppi_alpha: float
    high_hpii_min_level: int
    synthetic_start_year: int
    synthetic_start_month: int
    apply_eligibility_filters: bool
    min_heat_posts: int
    min_active_days: int
    min_seasonal_coverage: float
    hpvi_epsilon: float


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True, help="Tweet-level labeled/predicted CSV")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--total_posts_csv",
        default=None,
        help="Optional city-day denominator CSV with city/date/total_posts columns",
    )

    parser.add_argument("--city_col", default="city_id")
    parser.add_argument("--continent_col", default="continent")
    parser.add_argument("--country_col", default="country")
    parser.add_argument("--hemisphere_col", default="hemisphere")
    parser.add_argument("--date_col", default="date")
    parser.add_argument("--month_col", default="month")
    parser.add_argument(
        "--indicator_col",
        default="pred_label",
        help="Positive/negative indicator column; falls back to label if absent",
    )
    parser.add_argument("--positive_label", type=int, default=1)
    parser.add_argument(
        "--total_posts_col",
        default=None,
        help="If input already contains a per-row city-day denominator column",
    )

    parser.add_argument(
        "--normalize_within_city_month",
        action="store_true",
        help="Apply within-city-month median de-centering and 1/99 winsorization",
    )
    parser.add_argument(
        "--event_flag_col",
        default=None,
        help="Optional event-week/public-holiday flag column (1=exclude when computing median)",
    )
    parser.add_argument("--winsor_lower", type=float, default=0.01)
    parser.add_argument("--winsor_upper", type=float, default=0.99)

    parser.add_argument(
        "--hppi_alpha",
        type=float,
        default=1.6,
        help="Moderate admissible convexity parameter for ER-style HPPI",
    )
    parser.add_argument(
        "--high_hpii_min_level",
        type=int,
        default=5,
        help="Treat HPII levels >= this threshold as high (default: Levels V-VII)",
    )
    parser.add_argument(
        "--hpvi_epsilon",
        type=float,
        default=1e-8,
        help="Stabilization constant in HPVI denominator: std / (abs(mean) + epsilon)",
    )

    parser.add_argument("--synthetic_start_year", type=int, default=2022)
    parser.add_argument("--synthetic_start_month", type=int, default=3)

    parser.add_argument(
        "--apply_eligibility_filters",
        action="store_true",
        help="Apply manuscript-style city eligibility filters before city-season summaries",
    )
    parser.add_argument("--min_heat_posts", type=int, default=1000)
    parser.add_argument("--min_active_days", type=int, default=50)
    parser.add_argument("--min_seasonal_coverage", type=float, default=0.15)

    ns = parser.parse_args()
    return Config(**vars(ns))


def fisher_jenks_breaks(values: Sequence[float], n_classes: int) -> List[float]:
    """Pure-python Fisher-Jenks breaks. Returns class boundaries including min and max."""
    arr = np.array(sorted(float(v) for v in values if pd.notna(v)), dtype=float)
    if len(arr) == 0:
        return [0.0, 1.0]
    uniq = np.unique(arr)
    if len(uniq) <= 1:
        v = float(uniq[0]) if len(uniq) else 0.0
        return [v] * (n_classes + 1)
    n_classes = max(2, min(n_classes, len(arr)))

    lower = np.zeros((len(arr) + 1, n_classes + 1), dtype=int)
    var = np.full((len(arr) + 1, n_classes + 1), np.inf, dtype=float)

    for i in range(1, n_classes + 1):
        lower[1, i] = 1
        var[1, i] = 0.0
        for j in range(2, len(arr) + 1):
            var[j, i] = np.inf

    for l in range(2, len(arr) + 1):
        s1 = 0.0
        s2 = 0.0
        w = 0.0
        variance = 0.0
        for m in range(1, l + 1):
            idx = l - m + 1
            val = arr[idx - 1]
            s2 += val * val
            s1 += val
            w += 1
            variance = s2 - (s1 * s1) / w
            if idx != 1:
                for j in range(2, n_classes + 1):
                    candidate = variance + var[idx - 1, j - 1]
                    if var[l, j] >= candidate:
                        lower[l, j] = idx
                        var[l, j] = candidate
        lower[l, 1] = 1
        var[l, 1] = variance

    k = len(arr)
    kclass = [0.0] * (n_classes + 1)
    kclass[n_classes] = arr[-1]
    count = n_classes
    while count > 1:
        idx = int(lower[k, count] - 2)
        kclass[count - 1] = arr[max(idx, 0)]
        k = int(lower[k, count] - 1)
        count -= 1
    kclass[0] = arr[0]
    return kclass


ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII"}


def assign_jenks_levels(series: pd.Series, n_classes: int = 7) -> Tuple[pd.Series, List[float]]:
    vals = series.dropna().astype(float)
    if vals.empty:
        return pd.Series([np.nan] * len(series), index=series.index), [0.0, 1.0]
    breaks = fisher_jenks_breaks(vals.tolist(), n_classes=n_classes)
    if len(set(breaks)) == 1:
        out = pd.Series([1] * len(series), index=series.index)
        return out, breaks

    labels = []
    for v in series.astype(float):
        if pd.isna(v):
            labels.append(np.nan)
            continue
        level = 1
        for i in range(1, len(breaks)):
            upper = breaks[i]
            if v <= upper or i == len(breaks) - 1:
                level = i
                break
        labels.append(level)
    return pd.Series(labels, index=series.index), breaks


def infer_hemisphere(country: Optional[str], continent: Optional[str]) -> str:
    if country is not None:
        code = str(country).strip().upper()
        if code in COUNTRY_HEMISPHERE:
            return COUNTRY_HEMISPHERE[code]
    continent_norm = str(continent).strip().lower() if continent is not None else ""
    if continent_norm in {"oceania", "south america"}:
        return "south"
    return "north"


def month_to_num(value: object) -> Optional[int]:
    if pd.isna(value):
        return None
    s = str(value).strip()
    if not s:
        return None
    if len(s) >= 7 and s[4] == "-":
        try:
            return int(s[5:7])
        except ValueError:
            return None
    s_lower = s.lower()
    if s_lower in MONTH_NAME_TO_NUM:
        return MONTH_NAME_TO_NUM[s_lower]
    try:
        x = int(float(s))
        if 1 <= x <= 12:
            return x
    except ValueError:
        pass
    return None


def season_from_month(month_num: int, hemisphere: str) -> str:
    hemisphere = hemisphere.lower()
    if hemisphere == "south":
        if month_num in (12, 1, 2):
            return "Summer"
        if month_num in (3, 4, 5):
            return "Fall"
        if month_num in (6, 7, 8):
            return "Winter"
        return "Spring"
    if month_num in (12, 1, 2):
        return "Winter"
    if month_num in (3, 4, 5):
        return "Spring"
    if month_num in (6, 7, 8):
        return "Summer"
    return "Fall"


def read_input(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.input_csv, low_memory=False)
    if cfg.indicator_col not in df.columns:
        if "label" in df.columns:
            df[cfg.indicator_col] = df["label"]
        else:
            raise ValueError(
                f"Missing indicator column {cfg.indicator_col} and no label fallback found"
            )
    if cfg.city_col not in df.columns:
        raise ValueError(f"Missing city column: {cfg.city_col}")
    df[cfg.city_col] = df[cfg.city_col].astype(str)
    df[cfg.indicator_col] = pd.to_numeric(
        df[cfg.indicator_col], errors="coerce"
    ).fillna(0).astype(int)

    if cfg.continent_col not in df.columns:
        df[cfg.continent_col] = "unknown"
    if cfg.country_col not in df.columns:
        df[cfg.country_col] = "UNK"

    if cfg.hemisphere_col not in df.columns:
        df[cfg.hemisphere_col] = [
            infer_hemisphere(c, k) for c, k in zip(df[cfg.country_col], df[cfg.continent_col])
        ]
        df["hemisphere_is_inferred"] = 1
    else:
        df[cfg.hemisphere_col] = df[cfg.hemisphere_col].fillna("")
        mask = df[cfg.hemisphere_col].astype(str).str.strip().eq("")
        df.loc[mask, cfg.hemisphere_col] = [
            infer_hemisphere(c, k)
            for c, k in zip(df.loc[mask, cfg.country_col], df.loc[mask, cfg.continent_col])
        ]
        df["hemisphere_is_inferred"] = mask.astype(int)

    return df


def synthesize_dates(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df.copy()
    out["date_is_synthetic"] = 0

    if cfg.date_col in out.columns:
        parsed = pd.to_datetime(out[cfg.date_col], errors="coerce")
        if parsed.notna().any():
            out["date"] = parsed
            return out

    if cfg.month_col in out.columns:
        months = out[cfg.month_col].apply(month_to_num)
        years = []
        for raw in out[cfg.month_col].astype(str):
            if len(raw) >= 7 and raw[4] == "-":
                try:
                    years.append(int(raw[:4]))
                    continue
                except ValueError:
                    pass
            years.append(cfg.synthetic_start_year)
        out["__synthetic_year"] = years
        out["__synthetic_month_num"] = months.fillna(cfg.synthetic_start_month).astype(int)
    else:
        order = np.arange(len(out))
        out["__synthetic_month_num"] = (
            ((cfg.synthetic_start_month - 1 + (order % 12)) % 12) + 1
        )
        out["__synthetic_year"] = cfg.synthetic_start_year + (
            (cfg.synthetic_start_month - 1 + (order % 12)) // 12
        )

    out["__within_city_month_order"] = out.groupby(
        [cfg.city_col, "__synthetic_year", "__synthetic_month_num"]
    ).cumcount()
    day = (out["__within_city_month_order"] % 28) + 1
    out["date"] = pd.to_datetime(
        {
            "year": out["__synthetic_year"],
            "month": out["__synthetic_month_num"],
            "day": day,
        },
        errors="coerce",
    )
    out["date_is_synthetic"] = 1
    return out.drop(columns=["__within_city_month_order"], errors="ignore")


def add_calendar_fields(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if out["date"].isna().any():
        raise ValueError("Date synthesis failed for some rows; please check month/date inputs")
    out["year"] = out["date"].dt.year.astype(int)
    out["month_num"] = out["date"].dt.month.astype(int)
    out["month"] = out["date"].dt.strftime("%Y-%m")
    out["season"] = [season_from_month(m, h) for m, h in zip(out["month_num"], out[cfg.hemisphere_col])]
    return out


def build_city_day_counts(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df.copy()
    out["is_heat_positive"] = (out[cfg.indicator_col] == cfg.positive_label).astype(int)

    group_cols = [cfg.city_col, "date"]
    meta_cols = [
        cfg.continent_col,
        cfg.country_col,
        cfg.hemisphere_col,
        "year",
        "month",
        "month_num",
        "season",
    ]

    agg = (
        out.groupby(group_cols, dropna=False)
        .agg(
            heat_posts=("is_heat_positive", "sum"),
            observed_posts=(cfg.indicator_col, "size"),
            **{c: (c, "first") for c in meta_cols if c in out.columns},
            date_is_synthetic=("date_is_synthetic", "max"),
            hemisphere_is_inferred=("hemisphere_is_inferred", "max"),
        )
        .reset_index()
    )

    if cfg.total_posts_csv:
        total_df = pd.read_csv(cfg.total_posts_csv, low_memory=False)
        if cfg.city_col not in total_df.columns:
            raise ValueError(f"Denominator file missing city column: {cfg.city_col}")
        date_key = cfg.date_col if cfg.date_col in total_df.columns else "date"
        if date_key not in total_df.columns:
            raise ValueError("Denominator file needs a date column")
        total_df["date"] = pd.to_datetime(total_df[date_key], errors="coerce")
        denom_col = (
            cfg.total_posts_col
            if cfg.total_posts_col and cfg.total_posts_col in total_df.columns
            else "total_posts"
        )
        if denom_col not in total_df.columns:
            raise ValueError(f"Denominator file missing total posts column: {denom_col}")
        total_df = total_df[[cfg.city_col, "date", denom_col]].copy()
        total_df = total_df.rename(columns={denom_col: "total_posts"})
        agg = agg.merge(total_df, on=[cfg.city_col, "date"], how="left")
        agg["denominator_source"] = np.where(
            agg["total_posts"].notna(),
            "external_city_day_totals",
            "observed_input_rows_proxy",
        )
        agg["total_posts"] = agg["total_posts"].fillna(agg["observed_posts"])
    elif cfg.total_posts_col and cfg.total_posts_col in out.columns:
        temp = (
            out.groupby(group_cols, dropna=False)[cfg.total_posts_col]
            .first()
            .reset_index()
            .rename(columns={cfg.total_posts_col: "total_posts"})
        )
        agg = agg.merge(temp, on=group_cols, how="left")
        agg["total_posts"] = agg["total_posts"].fillna(agg["observed_posts"])
        agg["denominator_source"] = np.where(
            agg["total_posts"].eq(agg["observed_posts"]),
            "observed_input_rows_proxy",
            "input_total_posts_col",
        )
    else:
        agg["total_posts"] = agg["observed_posts"]
        agg["denominator_source"] = "observed_input_rows_proxy"

    agg["denominator_is_proxy"] = agg["denominator_source"].eq(
        "observed_input_rows_proxy"
    ).astype(int)
    agg["daily_hpii_raw"] = np.where(
        agg["total_posts"] > 0, agg["heat_posts"] / agg["total_posts"], np.nan
    )
    agg["daily_hpii_percent_raw"] = agg["daily_hpii_raw"] * 100.0
    return agg.sort_values([cfg.city_col, "date"]).reset_index(drop=True)


def normalize_city_month_series(city_day: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = city_day.copy()
    if not cfg.normalize_within_city_month:
        out["daily_hpii_used"] = out["daily_hpii_raw"]
        out["daily_hpii_percent_used"] = out["daily_hpii_percent_raw"]
        out["daily_hpii_norm"] = out["daily_hpii_raw"]
        out["daily_hpii_percent_norm"] = out["daily_hpii_percent_raw"]
        return out

    out["__median_mask"] = 1
    if cfg.event_flag_col and cfg.event_flag_col in out.columns:
        out["__median_mask"] = (
            pd.to_numeric(out[cfg.event_flag_col], errors="coerce").fillna(0) == 0
        ).astype(int)

    def _normalize_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        base = g.loc[g["__median_mask"] == 1, "daily_hpii_raw"]
        median = g["daily_hpii_raw"].median() if base.empty else base.median()
        residual = g["daily_hpii_raw"] - median
        lo = residual.quantile(cfg.winsor_lower)
        hi = residual.quantile(cfg.winsor_upper)
        g["daily_hpii_norm"] = residual.clip(lower=lo, upper=hi)
        g["daily_hpii_percent_norm"] = g["daily_hpii_norm"] * 100.0
        return g

    out = (
        out.groupby([cfg.city_col, "month"], group_keys=False)
        .apply(_normalize_group)
        .reset_index(drop=True)
    )
    out["daily_hpii_used"] = out["daily_hpii_norm"]
    out["daily_hpii_percent_used"] = out["daily_hpii_percent_norm"]
    return out.drop(columns=["__median_mask"], errors="ignore")


def compute_expected_coverage(city_day: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Compute expected calendar-day coverage for each city-season across each city's actual
    min/max observed date span. This fixes the earlier bug where observed and expected days
    were identical, causing coverage fractions to collapse to 1.
    """
    rows: List[Dict[str, object]] = []
    for city, sub in city_day.groupby(cfg.city_col):
        hemisphere = str(sub[cfg.hemisphere_col].dropna().iloc[0]).lower()
        min_date = pd.to_datetime(sub["date"].min())
        max_date = pd.to_datetime(sub["date"].max())
        if pd.isna(min_date) or pd.isna(max_date):
            continue
        full_range = pd.DataFrame({"date": pd.date_range(min_date, max_date, freq="D")})
        full_range["month_num"] = full_range["date"].dt.month.astype(int)
        full_range["season"] = full_range["month_num"].apply(
            lambda m: season_from_month(int(m), hemisphere)
        )
        expected = (
            full_range.groupby("season")["date"]
            .nunique()
            .reset_index(name="expected_days")
        )
        observed = (
            sub.groupby("season")["date"]
            .nunique()
            .reset_index(name="observed_unique_days")
        )
        merged = expected.merge(observed, on="season", how="left")
        merged["observed_unique_days"] = merged["observed_unique_days"].fillna(0).astype(int)
        merged["coverage_frac"] = np.where(
            merged["expected_days"] > 0,
            merged["observed_unique_days"] / merged["expected_days"],
            np.nan,
        )
        merged[cfg.city_col] = city
        rows.extend(merged[[cfg.city_col, "season", "expected_days", "observed_unique_days", "coverage_frac"]].to_dict("records"))
    return pd.DataFrame(rows)


def apply_eligibility_filters(city_day: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    coverage = compute_expected_coverage(city_day, cfg)

    city_summary = (
        city_day.groupby(cfg.city_col)
        .agg(
            total_heat_posts=("heat_posts", "sum"),
            active_days=("date", "nunique"),
            season_n=("season", "nunique"),
        )
        .reset_index()
    )

    if coverage.empty:
        city_summary["coverage_min_frac"] = np.nan
        city_summary["coverage_mean_frac"] = np.nan
        city_summary["expected_days_total"] = np.nan
        city_summary["observed_days_total"] = np.nan
    else:
        coverage_summary = (
            coverage.groupby(cfg.city_col)
            .agg(
                coverage_min_frac=("coverage_frac", "min"),
                coverage_mean_frac=("coverage_frac", "mean"),
                expected_days_total=("expected_days", "sum"),
                observed_days_total=("observed_unique_days", "sum"),
            )
            .reset_index()
        )
        city_summary = city_summary.merge(coverage_summary, on=cfg.city_col, how="left")

    if not cfg.apply_eligibility_filters:
        city_summary["eligible_city"] = 1
        return city_day, city_summary

    city_summary["eligible_city"] = (
        (city_summary["total_heat_posts"] >= cfg.min_heat_posts)
        & (city_summary["active_days"] >= cfg.min_active_days)
        & (city_summary["coverage_min_frac"] >= cfg.min_seasonal_coverage)
    ).astype(int)
    keep = set(city_summary.loc[city_summary["eligible_city"] == 1, cfg.city_col])
    return city_day[city_day[cfg.city_col].isin(keep)].copy(), city_summary


def compute_monthly_city_indices(city_day: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    monthly = (
        city_day.groupby(
            [
                cfg.city_col,
                cfg.continent_col,
                cfg.country_col,
                cfg.hemisphere_col,
                "year",
                "month",
                "month_num",
                "season",
            ]
        )
        .agg(
            month_heat_posts=("heat_posts", "sum"),
            month_total_posts=("total_posts", "sum"),
            month_hpii_daily_sum=("daily_hpii_raw", "sum"),
            month_hpii_daily_mean=("daily_hpii_raw", "mean"),
            month_hpii_norm_sum=("daily_hpii_used", "sum"),
            month_active_days=("date", "nunique"),
        )
        .reset_index()
    )
    monthly["month_hpii"] = np.where(
        monthly["month_total_posts"] > 0,
        monthly["month_heat_posts"] / monthly["month_total_posts"],
        np.nan,
    )
    monthly["month_hpii_percent"] = monthly["month_hpii"] * 100.0
    return monthly.sort_values([cfg.city_col, "year", "month_num"]).reset_index(drop=True)


def compute_hpvi(city_day: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    base = city_day.sort_values([cfg.city_col, "date"]).copy()
    base["daily_hpii_7d"] = base.groupby(cfg.city_col)["daily_hpii_used"].transform(
        lambda s: s.rolling(window=7, min_periods=1).mean()
    )

    rows: List[Dict[str, object]] = []
    for keys, sub in base.groupby(
        [cfg.city_col, cfg.continent_col, cfg.country_col, cfg.hemisphere_col, "season"]
    ):
        arr = sub["daily_hpii_7d"].dropna().astype(float)
        if arr.empty:
            hpvi_ratio = np.nan
        else:
            mean_val = float(arr.mean())
            std_val = float(arr.std(ddof=0))
            if abs(mean_val) < 1e-15 and abs(std_val) < 1e-15:
                hpvi_ratio = 0.0
            else:
                hpvi_ratio = std_val / (abs(mean_val) + cfg.hpvi_epsilon)

        hpvi_percent = hpvi_ratio * 100.0 if pd.notna(hpvi_ratio) else np.nan
        rows.append(
            {
                cfg.city_col: keys[0],
                cfg.continent_col: keys[1],
                cfg.country_col: keys[2],
                cfg.hemisphere_col: keys[3],
                "season": keys[4],
                "hpvi": hpvi_ratio,
                "hpvi_percent": hpvi_percent,
                "hpvi_days_used": int(arr.shape[0]),
            }
        )

    hpvi_df = pd.DataFrame(rows)

    def hpvi_tier(v: float) -> str:
        if pd.isna(v):
            return "unknown"
        if v <= 0.10:
            return "low"
        if v <= 0.30:
            return "moderate"
        return "high"

    hpvi_df["hpvi_tier"] = hpvi_df["hpvi"].apply(hpvi_tier)
    hpvi_df["hpvi_high_binary"] = hpvi_df["hpvi"].apply(
        lambda x: 1 if pd.notna(x) and x > 0.30 else 0
    )
    return hpvi_df


def compute_city_season_indices(
    city_day: pd.DataFrame,
    monthly: pd.DataFrame,
    hpvi_df: pd.DataFrame,
    cfg: Config,
) -> Tuple[pd.DataFrame, List[float]]:
    seasonal = (
        city_day.groupby(
            [cfg.city_col, cfg.continent_col, cfg.country_col, cfg.hemisphere_col, "season"]
        )
        .agg(
            seasonal_heat_posts=("heat_posts", "sum"),
            seasonal_total_posts=("total_posts", "sum"),
            seasonal_hpii_daily_mean=("daily_hpii_raw", "mean"),
            seasonal_hpii_daily_sum=("daily_hpii_raw", "sum"),
            seasonal_hpii_norm_sum=("daily_hpii_used", "sum"),
            seasonal_active_days=("date", "nunique"),
            date_is_synthetic=("date_is_synthetic", "max"),
            denominator_is_proxy=("denominator_is_proxy", "max"),
        )
        .reset_index()
    )
    seasonal["seasonal_hpii"] = np.where(
        seasonal["seasonal_total_posts"] > 0,
        seasonal["seasonal_heat_posts"] / seasonal["seasonal_total_posts"],
        np.nan,
    )
    seasonal["seasonal_hpii_percent"] = seasonal["seasonal_hpii"] * 100.0

    peak_month = (
        monthly.sort_values([cfg.city_col, "season", "month_hpii"], ascending=[True, True, False])
        .groupby([cfg.city_col, "season"])
        .head(1)
    )
    peak_month = peak_month[
        [cfg.city_col, "season", "month", "month_hpii", "month_hpii_percent"]
    ].rename(
        columns={
            "month": "peak_month_in_season",
            "month_hpii": "peak_month_hpii",
            "month_hpii_percent": "peak_month_hpii_percent",
        }
    )
    seasonal = seasonal.merge(peak_month, on=[cfg.city_col, "season"], how="left")
    seasonal = seasonal.merge(
        hpvi_df,
        on=[cfg.city_col, cfg.continent_col, cfg.country_col, cfg.hemisphere_col, "season"],
        how="left",
    )

    levels, breaks = assign_jenks_levels(seasonal["seasonal_hpii_percent"], n_classes=7)
    seasonal["hpii_level"] = levels.astype("Int64")
    seasonal["hpii_level_roman"] = seasonal["hpii_level"].map(ROMAN)
    seasonal["high_hpii_binary"] = seasonal["hpii_level"].apply(
        lambda x: 1 if pd.notna(x) and int(x) >= cfg.high_hpii_min_level else 0
    )

    def typology(row: pd.Series) -> str:
        hi = int(row["high_hpii_binary"]) == 1
        hv = int(row["hpvi_high_binary"]) == 1 if pd.notna(row["hpvi_high_binary"]) else False
        if hi and hv:
            return "Perception Hotspots"
        if hi and not hv:
            return "Stable High"
        if (not hi) and (not hv):
            return "Stable Low"
        return "Variability Low"

    seasonal["intensity_variability_typology"] = seasonal.apply(typology, axis=1)
    return seasonal.sort_values([cfg.city_col, "season"]).reset_index(drop=True), breaks


def compute_city_typology_summary(
    seasonal: pd.DataFrame, monthly: pd.DataFrame, cfg: Config
) -> pd.DataFrame:
    if seasonal.empty:
        return pd.DataFrame(
            columns=[
                cfg.city_col,
                "peak_season",
                "total_dominant_season",
                "peak_total_coincide",
                "multi_season_high_hpii",
                "counter_seasonal_label",
            ]
        )

    peak_any = (
        monthly.sort_values([cfg.city_col, "month_hpii"], ascending=[True, False])
        .groupby(cfg.city_col)
        .head(1)
    )
    peak_any = peak_any[
        [
            cfg.city_col,
            cfg.continent_col,
            cfg.country_col,
            cfg.hemisphere_col,
            "season",
            "month",
            "month_hpii",
            "month_hpii_percent",
        ]
    ].rename(
        columns={
            "season": "peak_season",
            "month": "peak_month",
            "month_hpii": "peak_month_hpii_global",
            "month_hpii_percent": "peak_month_hpii_percent_global",
        }
    )

    total_dom = (
        seasonal.sort_values([cfg.city_col, "seasonal_hpii_daily_sum"], ascending=[True, False])
        .groupby(cfg.city_col)
        .head(1)
    )
    total_dom = total_dom[
        [cfg.city_col, "season", "seasonal_hpii_daily_sum", "seasonal_hpii", "seasonal_hpii_percent"]
    ].rename(
        columns={
            "season": "total_dominant_season",
            "seasonal_hpii_daily_sum": "total_dominant_hpii_sum",
            "seasonal_hpii": "total_dominant_hpii",
            "seasonal_hpii_percent": "total_dominant_hpii_percent",
        }
    )

    out = peak_any.merge(total_dom, on=cfg.city_col, how="outer")
    out["peak_total_coincide"] = (out["peak_season"] == out["total_dominant_season"]).astype(int)

    high_counts = (
        seasonal.groupby(cfg.city_col)["high_hpii_binary"]
        .sum()
        .reset_index(name="n_high_hpii_seasons")
    )
    high_counts["multi_season_high_hpii"] = (
        high_counts["n_high_hpii_seasons"] >= 2
    ).astype(int)
    out = out.merge(high_counts, on=cfg.city_col, how="left")

    season_lookup = seasonal[[cfg.city_col, "season", "seasonal_hpii_daily_sum"]].copy()

    summer_total = season_lookup[season_lookup["season"] == "Summer"][
        [cfg.city_col, "seasonal_hpii_daily_sum"]
    ].rename(columns={"seasonal_hpii_daily_sum": "summer_total_hpii_sum"})
    winter_total = season_lookup[season_lookup["season"] == "Winter"][
        [cfg.city_col, "seasonal_hpii_daily_sum"]
    ].rename(columns={"seasonal_hpii_daily_sum": "winter_total_hpii_sum"})
    summer_peak = seasonal[seasonal["season"] == "Summer"][
        [cfg.city_col, "peak_month_hpii", "peak_month_hpii_percent"]
    ].rename(
        columns={
            "peak_month_hpii": "summer_peak_month_hpii",
            "peak_month_hpii_percent": "summer_peak_month_hpii_percent",
        }
    )
    winter_peak = seasonal[seasonal["season"] == "Winter"][
        [cfg.city_col, "peak_month_hpii", "peak_month_hpii_percent"]
    ].rename(
        columns={
            "peak_month_hpii": "winter_peak_month_hpii",
            "peak_month_hpii_percent": "winter_peak_month_hpii_percent",
        }
    )

    out = out.merge(summer_total, on=cfg.city_col, how="left")
    out = out.merge(winter_total, on=cfg.city_col, how="left")
    out = out.merge(summer_peak, on=cfg.city_col, how="left")
    out = out.merge(winter_peak, on=cfg.city_col, how="left")

    peak_cs = out["winter_peak_month_hpii"].fillna(-np.inf) > out["summer_peak_month_hpii"].fillna(
        -np.inf
    )
    total_cs = out["winter_total_hpii_sum"].fillna(-np.inf) > out["summer_total_hpii_sum"].fillna(
        -np.inf
    )

    labels = []
    for a, b in zip(peak_cs, total_cs):
        if a and b:
            labels.append("Both")
        elif a:
            labels.append("Peak")
        elif b:
            labels.append("Total")
        else:
            labels.append("NoCounterSeasonal")
    out["counter_seasonal_label"] = labels
    out["counter_seasonal_binary"] = (
        out["counter_seasonal_label"] != "NoCounterSeasonal"
    ).astype(int)
    return out.sort_values(cfg.city_col).reset_index(drop=True)


def volume_strata(city_day: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    annual = (
        city_day.groupby(cfg.city_col)
        .agg(
            annual_total_posts=("total_posts", "sum"),
            annual_heat_posts=("heat_posts", "sum"),
            active_days=("date", "nunique"),
        )
        .reset_index()
    )
    q30 = annual["annual_total_posts"].quantile(0.30)
    q70 = annual["annual_total_posts"].quantile(0.70)

    def bucket(x: float) -> str:
        if x <= q30:
            return "Low"
        if x <= q70:
            return "Medium"
        return "High"

    annual["tweet_volume_stratum"] = annual["annual_total_posts"].apply(bucket)
    return annual


def _scaled_signal(series: pd.Series, kind: str) -> pd.DataFrame:
    vals = series.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if vals.max() > vals.min():
        scaled = (vals - vals.min()) / (vals.max() - vals.min())
    else:
        scaled = pd.Series([0.0] * len(vals), index=vals.index)

    if kind == "hpii":
        tiers = pd.cut(
            scaled,
            bins=[-np.inf, 1 / 3, 2 / 3, np.inf],
            labels=["low", "medium", "high"],
        )
    else:
        tiers = pd.cut(
            vals,
            bins=[-np.inf, 0.10, 0.30, np.inf],
            labels=["low", "medium", "high"],
        )
    tier_weight = (
        tiers.map({"low": 1.0, "medium": 2.0, "high": 3.0})
        .astype(float)
        .fillna(1.0)
    )
    adjusted_signal = (scaled + 1e-9) * tier_weight
    return pd.DataFrame(
        {
            "value": vals,
            "scaled_value": scaled,
            "tier": tiers.astype(str),
            "tier_weight": tier_weight,
            "adjusted_signal": adjusted_signal,
        }
    )


def normalized_er_polarization(values: pd.Series, alpha: float, kind: str) -> float:
    sig = _scaled_signal(values, kind=kind)
    adj = sig["adjusted_signal"].to_numpy(dtype=float)
    if adj.sum() <= 0 or len(adj) <= 1:
        return 0.0
    p = adj / adj.sum()
    dist = np.abs(adj[:, None] - adj[None, :])
    raw = np.sum((p[:, None] ** (1.0 + alpha)) * p[None, :] * dist)
    denom = np.sum(p ** (1.0 + alpha)) * max(float(dist.max()), 1e-12)
    score = raw / denom if denom > 0 else 0.0
    return float(np.clip(score, 0.0, 1.0))


def compute_hppi(seasonal: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (continent, season), sub in seasonal.groupby([cfg.continent_col, "season"]):
        hppi_hpii = normalized_er_polarization(
            sub["seasonal_hpii_percent"], alpha=cfg.hppi_alpha, kind="hpii"
        )
        hppi_hpvi = normalized_er_polarization(sub["hpvi"], alpha=cfg.hppi_alpha, kind="hpvi")
        rows.append(
            {
                cfg.continent_col: continent,
                "season": season,
                "n_cities": int(sub[cfg.city_col].nunique()),
                "hppi_hpii": hppi_hpii,
                "hppi_hpvi": hppi_hpvi,
            }
        )
    out = pd.DataFrame(rows)

    def tier(x: float) -> str:
        if x < 0.05:
            return "low"
        if x <= 0.15:
            return "moderate"
        return "high"

    if not out.empty:
        out["hppi_hpii_tier"] = out["hppi_hpii"].apply(tier)
        out["hppi_hpvi_tier"] = out["hppi_hpvi"].apply(tier)
    return out.sort_values([cfg.continent_col, "season"]).reset_index(drop=True)


def write_outputs(
    city_day: pd.DataFrame,
    monthly: pd.DataFrame,
    seasonal: pd.DataFrame,
    city_typology: pd.DataFrame,
    hppi: pd.DataFrame,
    volume: pd.DataFrame,
    eligibility: pd.DataFrame,
    jenks_breaks_percent: List[float],
    cfg: Config,
) -> Dict[str, str]:
    outdir = Path(cfg.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    paths = {
        "city_day_indices_csv": str(outdir / "city_day_indices.csv"),
        "city_month_indices_csv": str(outdir / "city_month_indices.csv"),
        "city_season_indices_csv": str(outdir / "city_season_indices.csv"),
        "city_typology_summary_csv": str(outdir / "city_typology_summary.csv"),
        "continent_season_hppi_csv": str(outdir / "continent_season_hppi.csv"),
        "tweet_volume_strata_csv": str(outdir / "tweet_volume_strata.csv"),
        "eligibility_summary_csv": str(outdir / "eligibility_summary.csv"),
        "aggregate_summary_json": str(outdir / "aggregate_summary.json"),
    }

    city_day.to_csv(paths["city_day_indices_csv"], index=False)
    monthly.to_csv(paths["city_month_indices_csv"], index=False)
    seasonal.to_csv(paths["city_season_indices_csv"], index=False)
    city_typology.to_csv(paths["city_typology_summary_csv"], index=False)
    hppi.to_csv(paths["continent_season_hppi_csv"], index=False)
    volume.to_csv(paths["tweet_volume_strata_csv"], index=False)
    eligibility.to_csv(paths["eligibility_summary_csv"], index=False)

    summary = {
        "n_city_days": int(len(city_day)),
        "n_city_months": int(len(monthly)),
        "n_city_seasons": int(len(seasonal)),
        "n_cities": int(city_day[cfg.city_col].nunique()) if not city_day.empty else 0,
        "n_continents": (
            int(city_day[cfg.continent_col].nunique())
            if cfg.continent_col in city_day.columns and not city_day.empty
            else 0
        ),
        "used_indicator_col": cfg.indicator_col,
        "denominator_proxy_fraction": (
            float(city_day["denominator_is_proxy"].mean()) if not city_day.empty else 0.0
        ),
        "synthetic_date_fraction": (
            float(city_day["date_is_synthetic"].mean()) if not city_day.empty else 0.0
        ),
        "hemisphere_inferred_fraction": (
            float(city_day["hemisphere_is_inferred"].mean()) if not city_day.empty else 0.0
        ),
        "normalization_applied": bool(cfg.normalize_within_city_month),
        "hpii_units": "proportion with parallel percent columns exported",
        "hpii_jenks_breaks_percent": jenks_breaks_percent,
        "high_hpii_min_level": int(cfg.high_hpii_min_level),
        "hpvi_units": "ratio with parallel percent columns exported",
        "hpvi_thresholds_ratio": {"low_max": 0.10, "moderate_max": 0.30},
        "hpvi_thresholds_percent": {"low_max": 10.0, "moderate_max": 30.0},
        "hpvi_epsilon": float(cfg.hpvi_epsilon),
        "hppi_alpha": float(cfg.hppi_alpha),
        "eligibility_filters_applied": bool(cfg.apply_eligibility_filters),
        "note": (
            "If date or denominator inputs were missing, synthetic dates and/or proxy denominators were used. "
            "Replace them with authentic city-day totals before research-grade reporting."
        ),
        "artifacts": paths,
    }
    with open(paths["aggregate_summary_json"], "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return paths


def main() -> None:
    cfg = parse_args()
    raw = read_input(cfg)
    raw = synthesize_dates(raw, cfg)
    raw = add_calendar_fields(raw, cfg)

    city_day = build_city_day_counts(raw, cfg)
    city_day = normalize_city_month_series(city_day, cfg)
    city_day, eligibility = apply_eligibility_filters(city_day, cfg)

    monthly = compute_monthly_city_indices(city_day, cfg)
    hpvi_df = compute_hpvi(city_day, cfg)
    seasonal, jenks_breaks_percent = compute_city_season_indices(city_day, monthly, hpvi_df, cfg)
    city_typology = compute_city_typology_summary(seasonal, monthly, cfg)
    hppi = compute_hppi(seasonal, cfg)
    volume = volume_strata(city_day, cfg)

    seasonal = seasonal.merge(
        volume[[cfg.city_col, "tweet_volume_stratum"]],
        on=cfg.city_col,
        how="left",
    )
    city_typology = city_typology.merge(
        volume[[cfg.city_col, "tweet_volume_stratum"]],
        on=cfg.city_col,
        how="left",
    )

    paths = write_outputs(
        city_day,
        monthly,
        seasonal,
        city_typology,
        hppi,
        volume,
        eligibility,
        jenks_breaks_percent,
        cfg,
    )

    print(
        json.dumps(
            {
                "status": "ok",
                "n_city_days": int(len(city_day)),
                "n_city_months": int(len(monthly)),
                "n_city_seasons": int(len(seasonal)),
                "denominator_proxy_fraction": (
                    float(city_day["denominator_is_proxy"].mean()) if not city_day.empty else 0.0
                ),
                "synthetic_date_fraction": (
                    float(city_day["date_is_synthetic"].mean()) if not city_day.empty else 0.0
                ),
                "artifacts": paths,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
