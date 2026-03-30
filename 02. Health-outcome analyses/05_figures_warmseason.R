## =============================================================================
## Script 05: Publication figures — warm-season analysis (primary)
##
## Produces all main and supplementary figures for the warm-season analysis:
##   Fig. 4  — Model comparison stacked bar chart
##   Fig. 5  — Joint exposure forest plot (panel a: all-cause; panel b: subgroups)
##   Supp.   — Barplots and heatmaps by outcome class for each outcome type
##
## Input data files (relative to working directory):
##   final_v2/data/02_pooled_RR_all_outcomes.csv
##   final_v2/data/03_model_comparison_winrates.csv
##   final_v2/data/04_joint_exposure_citylevel.csv
##   final_v2/data/05_subgroup_death_pooled.csv
##   which_better/Death.xlsx
##   which_better/Hospitalization.xlsx
##   which_better/ED.xlsx
##
## Outputs (PDF, Nature Communications dimensions):
##   output/figures_warmseason/Fig4_model_comparison.pdf        89 x 88 mm
##   output/figures_warmseason/Fig5_forest_plot.pdf            180 x 135 mm
##   output/figures_warmseason/Supp_barplot_Death.pdf          220 x 130 mm
##   output/figures_warmseason/Supp_barplot_Hosp.pdf           220 x 130 mm
##   output/figures_warmseason/Supp_barplot_ED.pdf             130 x 110 mm
##   output/figures_warmseason/Supp_heatmap_Death.pdf          160 x 200 mm
##   output/figures_warmseason/Supp_heatmap_Hosp.pdf           160 x 200 mm
##   output/figures_warmseason/Supp_heatmap_ED.pdf             130 x 110 mm
## =============================================================================

# Set working directory to the folder containing final_dat.rds before running this script.

suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(ggplot2)
  library(patchwork); library(readxl); library(forcats)
})

dir.create("output/figures_warmseason", recursive = TRUE, showWarnings = FALSE)

## =============================================================================
## Style constants (Nature Communications)
## =============================================================================

PT   <- 7; PTs <- 6
NC_s <- PTs / 2.845          # ggplot geom_text size (mm to pt)
XLIM <- c(0.28, 4.15)
XBK  <- c(0.5, 1, 1.5, 2, 3, 4)
XLB  <- c("0.5","1.0","1.5","2.0","3.0","\u22654.0")

## Category colors
CAT_COL  <- c(Cat2 = "#1B9E77", Cat3 = "#FD8D3C", Cat4 = "#D7191C")
CAT_SZ   <- c(Cat2 = 1.4, Cat3 = 1.4, Cat4 = 2.2)
CAT_LW   <- c(Cat2 = 0.38, Cat3 = 0.38, Cat4 = 0.85)
RED_CAT4 <- "#D7191C"

## 7-category bar colors (blue -> grey -> orange/red) — used in Fig 4 only
COL7 <- c(
  "P-P" = "#08306B", "P-S" = "#4292C6", "S-P" = "#9ECAE1",
  "S-S" = "#BDBDBD",
  "S-T" = "#FDAE6B", "T-S" = "#E6550D", "T-T" = "#A50F15"
)
CAT7_LEV <- c("P-P","P-S","S-P","S-S","S-T","T-S","T-T")

## 9-category colors — used in Supplementary (barplot + heatmap)
COL9 <- c(
  "P-P" = "#08306B", "P-S" = "#4292C6", "P-T" = "#C6DBEF",
  "S-P" = "#9ECAE1", "S-S" = "#BDBDBD", "S-T" = "#FDAE6B",
  "T-P" = "#FED976", "T-S" = "#E6550D", "T-T" = "#A50F15"
)
CAT9_LEV <- c("P-P","P-S","P-T","S-P","S-S","S-T","T-P","T-S","T-T")

th <- theme_bw(base_size = PT) + theme(
  panel.grid.minor   = element_blank(),
  panel.grid.major.y = element_blank(),
  panel.grid.major.x = element_line(color = "#F0F0F0", linewidth = 0.22),
  axis.text          = element_text(size = PTs),
  axis.title         = element_text(size = PT),
  legend.text        = element_text(size = PTs),
  legend.key.size    = unit(0.24, "cm"),
  plot.margin        = margin(1.5, 2, 1.5, 2, "mm")
)

## =============================================================================
## FIGURE 4 — Model comparison stacked bars (HPII >= 1 only)
## =============================================================================

mc_raw <- read.csv("final_v2/data/03_model_comparison_winrates.csv",
                   check.names = FALSE)

## Filter HPII>=1; relabel outcomes
mc <- mc_raw %>%
  filter(threshold == "HPII\u22651") %>%
  mutate(outcome = recode(outcome_type,
    "Death"                      = "Death",
    "Hospitlization"             = "Hospitalization",
    "Emergency department visit" = "Emergency\ndept. visits"),
    outcome = factor(outcome,
      levels = c("Death","Hospitalization","Emergency\ndept. visits")))

## Pivot to 7 long categories
mc_long <- mc %>%
  transmute(
    outcome, n_pairs,
    `P-P` = n_P_both, `P-S` = n_P_QAIC_only, `S-P` = n_P_R2_only,
    `S-S` = n_S_both,
    `S-T` = n_T_R2_only, `T-S` = n_T_QAIC_only, `T-T` = n_T_both
  ) %>%
  pivot_longer(`P-P`:`T-T`, names_to = "cat", values_to = "n") %>%
  mutate(cat = factor(cat, levels = CAT7_LEV),
         pct = n / n_pairs * 100)

## Summary annotation for each bar: "P x% | S x% | T x%"
mc_ann <- mc %>%
  mutate(ann = sprintf("P\u202f%.0f%%\u2003|\u2003S\u202f%.0f%%\u2003|\u2003T\u202f%.0f%%",
                       pct_P_either, pct_S_both, pct_T_either))

p4 <- ggplot(mc_long, aes(x = outcome, y = pct, fill = cat)) +
  geom_col(width = 0.58, color = NA) +
  geom_text(data = mc_ann,
            aes(x = outcome, y = 103, label = ann),
            inherit.aes = FALSE,
            size = NC_s * 0.92, vjust = 0, color = "grey25", fontface = "plain") +
  scale_fill_manual(values = COL7, name = NULL,
    guide = guide_legend(nrow = 1, override.aes = list(color = NA))) +
  scale_y_continuous(limits = c(0, 116), breaks = seq(0, 100, 25),
                     labels = paste0(seq(0, 100, 25), "%"), expand = c(0, 0)) +
  labs(x = NULL,
       y = "City\u2013outcome-class pairs (%)",
       caption = paste0(
         "P = perception-based model; T = temperature-based model; ",
         "S = similar fit (|dQAIC| <= 2 and |dR2| <= 0.02).\n",
         "First letter: QAIC criterion; second letter: pseudo-R2 criterion. ",
         "P-S, S-T etc. indicate split decisions across the two metrics.")) +
  th +
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(color = "#F0F0F0", linewidth = 0.22),
    legend.position    = "bottom",
    legend.margin      = margin(0, 0, 0, 0),
    plot.caption       = element_text(size = 5, color = "grey40", hjust = 0,
                                      lineheight = 1.2, margin = margin(t = 1.5, "mm"))
  )

ggsave("output/figures_warmseason/Fig4_model_comparison.pdf",
       p4, width = 89, height = 88, units = "mm")
cat("Fig4 saved (89x88mm)\n")

## =============================================================================
## FIGURE 5a — All-cause forest plot (city-level + pooled)
## =============================================================================

## Helper: map exposure_category to Cat2/3/4
exp_map <- c(
  "Normal & High" = "Cat2",
  "Heat & Low"    = "Cat3",
  "Heat & High"   = "Cat4"
)

## City-level rows
city_raw <- read.csv("final_v2/data/04_joint_exposure_citylevel.csv") %>%
  filter(threshold == "HPII\u22651",
         level == "city",
         outcome_class == "all-cause",
         exposure_category %in% names(exp_map),
         tolower(city) != "melbourne") %>%
  mutate(
    cat = factor(exp_map[exposure_category], levels = c("Cat2","Cat3","Cat4")),
    row_label = case_when(
      tolower(city) == "mexico city"      ~ "Mexico City",
      tolower(city) == "puerto vallarta"  ~ "Puerto Vallarta",
      tolower(city) == "playa del carmen" ~ "Playa del Carmen (b)",
      tolower(city) == "porto alegre"     ~ "Porto Alegre",
      tolower(city) == "rio de janeiro"   ~ "Rio de Janeiro",
      TRUE ~ city
    ),
    is_pooled    = FALSE, is_combined = FALSE,
    rr_hi_p      = pmin(RR_high, 4), trunc = RR_high > 4
  ) %>%
  select(row_label, cat, RR, RR_low, RR_high, is_pooled, is_combined, rr_hi_p, trunc)

## Pooled rows
pool_raw <- read.csv("final_v2/data/02_pooled_RR_all_outcomes.csv") %>%
  filter(case != "Category 1") %>%
  mutate(cat_code = sub("Category ","Cat", case),
         cat = factor(cat_code, levels = c("Cat2","Cat3","Cat4")))

pool_death <- pool_raw %>%
  filter(pool == "Death (n=3)") %>%
  mutate(row_label    = "Pooled \u2013 Death (n\u202f=\u202f3)",
         is_pooled    = TRUE, is_combined = FALSE,
         rr_hi_p      = pmin(RR_high, 4), trunc = RR_high > 4) %>%
  select(row_label, cat, RR, RR_low, RR_high, is_pooled, is_combined, rr_hi_p, trunc)

pool_hosp <- pool_raw %>%
  filter(pool == "Hospitalization (n=3)") %>%
  mutate(row_label    = "Pooled \u2013 Hosp. (n\u202f=\u202f3)",
         is_pooled    = TRUE, is_combined = FALSE,
         rr_hi_p      = pmin(RR_high, 4), trunc = RR_high > 4) %>%
  select(row_label, cat, RR, RR_low, RR_high, is_pooled, is_combined, rr_hi_p, trunc)

pool_comb <- pool_raw %>%
  filter(pool == "Combined (n=6)", cat_code == "Cat4") %>%
  mutate(row_label    = "Combined all sites (n\u202f=\u202f6)",
         is_pooled    = TRUE, is_combined = TRUE,
         rr_hi_p      = pmin(RR_high, 4), trunc = RR_high > 4) %>%
  select(row_label, cat, RR, RR_low, RR_high, is_pooled, is_combined, rr_hi_p, trunc)

## y-positions (larger y = higher in plot = top of figure)
ROW_Y <- c(
  "Mexico City"                           = 9.5,
  "Puerto Vallarta"                       = 8.5,
  "Playa del Carmen (b)"                  = 7.5,
  "Pooled \u2013 Death (n\u202f=\u202f3)" = 6.5,
  "Porto Alegre"                          = 5.0,
  "Rio de Janeiro"                        = 4.0,
  "S\u00e3o Paulo"                        = 3.0,
  "Pooled \u2013 Hosp. (n\u202f=\u202f3)" = 2.0,
  "Combined all sites (n\u202f=\u202f6)"  = 0.5
)

## Within-row: Cat2 = top (+0.20), Cat3 = middle (0), Cat4 = bottom (-0.20)
DODGE_Y <- c(Cat2 = +0.20, Cat3 = 0, Cat4 = -0.20)

## Handle unicode in city column for Sao Paulo
fix_sp <- function(df) {
  df %>% mutate(row_label = ifelse(
    grepl("Paulo|paulo", row_label, ignore.case = TRUE),
    "S\u00e3o Paulo", row_label))
}

pA_dat <- bind_rows(city_raw, pool_death, pool_hosp, pool_comb) %>%
  fix_sp() %>%
  mutate(
    y_center = ROW_Y[row_label],
    y_plot   = y_center + DODGE_Y[as.character(cat)]
  ) %>%
  filter(!is.na(y_center))

cat("Panel A rows covered:", sort(unique(pA_dat$row_label)), "\n")

## Block separator y-positions
sep_y1 <- mean(c(6.5, 5.0))   # between Death/Hosp blocks
sep_y2 <- mean(c(2.0, 0.5))   # between Hosp/Combined

## y-axis breaks and labels
y_breaks <- c(9.5, 8.5, 7.5, 6.5, 5.0, 4.0, 3.0, 2.0, 0.5)
y_labels <- c(
  "Mexico City", "Puerto Vallarta", "Playa del Carmen (b)",
  "Pooled \u2013 Death (n=3)",
  "Porto Alegre", "Rio de Janeiro", "S\u00e3o Paulo",
  "Pooled \u2013 Hosp. (n=3)",
  "Combined all sites (n=6)"
)

pA_city <- filter(pA_dat, !is_combined)
pA_comb <- filter(pA_dat,  is_combined)

pA <- ggplot() +
  ## Reference line
  geom_vline(xintercept = 1, linetype = 2, color = "grey50", linewidth = 0.32) +
  ## Block separators
  geom_hline(yintercept = c(sep_y1, sep_y2),
             linetype = "solid", color = "grey75", linewidth = 0.36) +
  ## Dotted line above pooled rows
  geom_hline(yintercept = c(mean(c(7.5, 6.5)), mean(c(3.0, 2.0))),
             linetype = "dotted", color = "grey78", linewidth = 0.28) +
  ## City + pooled CI bars
  geom_errorbarh(data = pA_city,
    aes(xmin = RR_low, xmax = rr_hi_p, y = y_plot, color = cat, linewidth = cat),
    height = 0.13, inherit.aes = FALSE) +
  ## City + pooled points
  geom_point(data = pA_city,
    aes(x = RR, y = y_plot, color = cat, size = cat,
        shape = ifelse(is_pooled, "diamond", "circle")),
    inherit.aes = FALSE) +
  ## Truncation markers
  geom_point(data = filter(pA_city, trunc),
    aes(x = 4.08, y = y_plot), shape = 9, size = 1.3, color = "grey30",
    inherit.aes = FALSE) +
  ## Combined row (Cat4 only, centered, larger)
  geom_errorbarh(data = pA_comb,
    aes(xmin = RR_low, xmax = rr_hi_p, y = y_plot),
    height = 0.28, linewidth = 0.95, color = RED_CAT4, inherit.aes = FALSE) +
  geom_point(data = pA_comb,
    aes(x = RR, y = y_plot),
    shape = 18, size = 4.2, color = RED_CAT4, inherit.aes = FALSE) +
  ## Block labels
  annotate("text", x = 0.30, y = 9.05, label = "Death",
           hjust = 0, size = NC_s * 0.90, fontface = "bold", color = "grey30") +
  annotate("text", x = 0.30, y = 4.55, label = "Hospitalization",
           hjust = 0, size = NC_s * 0.90, fontface = "bold", color = "grey30") +
  annotate("text", x = 0.30, y = 0.95, label = "Combined",
           hjust = 0, size = NC_s * 0.90, fontface = "bold", color = "grey30") +
  ## Scales
  scale_color_manual(values = CAT_COL, name = "Exposure category",
    labels = c(Cat2 = "Category 2", Cat3 = "Category 3", Cat4 = "Category 4"),
    guide = guide_legend(nrow = 1,
      override.aes = list(shape = 18, size = c(1.8, 1.8, 2.8),
                          linewidth = c(0.38, 0.38, 0.85)))) +
  scale_size_manual(values = CAT_SZ, guide = "none") +
  scale_linewidth_manual(values = CAT_LW, guide = "none") +
  scale_shape_identity(guide = "none") +
  scale_x_continuous(breaks = XBK, labels = XLB, limits = XLIM) +
  scale_y_continuous(breaks = y_breaks, labels = y_labels,
                     limits = c(-0.3, 10.3)) +
  labs(x = "Relative risk (95%\u202fCI)", y = NULL) +
  th +
  theme(
    axis.text.y     = element_text(size = PTs, hjust = 1),
    legend.position = "bottom",
    legend.margin   = margin(0, 0, 0, 0)
  )

## =============================================================================
## FIGURE 5b — Mortality subgroups, Category 4 only
## =============================================================================

KEEP_SUB <- c("age0_69","age70_","female","male","I","J")
SUB_LAB  <- c(
  "age0_69" = "Age 0\u201369",
  "age70_"  = "Age \u226570",
  "female"  = "Female",
  "male"    = "Male",
  "I"       = "ICD-I  Circulatory",
  "J"       = "ICD-J  Respiratory"
)
Y_SUB <- c(6, 5, 4, 3, 1.5, 0.5)
names(Y_SUB) <- SUB_LAB[KEEP_SUB]

pB_dat <- read.csv("final_v2/data/05_subgroup_death_pooled.csv") %>%
  filter(class %in% KEEP_SUB, case == "Heat & High") %>%
  mutate(
    sub_label = SUB_LAB[class],
    y_pos     = Y_SUB[sub_label],
    rr_hi_p   = pmin(rrhigh, 4),
    trunc     = rrhigh > 4
  ) %>%
  filter(!is.na(y_pos))

pB <- ggplot(pB_dat, aes(x = rr, y = y_pos)) +
  geom_vline(xintercept = 1, linetype = 2, color = "grey50", linewidth = 0.32) +
  ## Dashed separator: demographic vs ICD
  geom_hline(yintercept = 2.25, linetype = "dashed",
             color = "grey70", linewidth = 0.28) +
  ## CI bars
  geom_errorbarh(aes(xmin = rrlow, xmax = rr_hi_p),
                 height = 0.22, linewidth = 0.42, color = RED_CAT4) +
  ## Points (solid, same red, all same size)
  geom_point(size = 2.0, shape = 16, color = RED_CAT4) +
  ## Truncation
  geom_point(data = filter(pB_dat, trunc),
             aes(x = 4.08, y = y_pos), shape = 9, size = 1.3,
             color = "grey30", inherit.aes = FALSE) +
  ## Subgroup type labels
  annotate("text", x = 0.30, y = 6.45,
           label = "Age", hjust = 0,
           size = NC_s * 0.90, fontface = "bold", color = "grey30") +
  annotate("text", x = 0.30, y = 4.45,
           label = "Sex", hjust = 0,
           size = NC_s * 0.90, fontface = "bold", color = "grey30") +
  annotate("text", x = 0.30, y = 1.95,
           label = "Cause of death", hjust = 0,
           size = NC_s * 0.90, fontface = "bold", color = "grey30") +
  scale_x_continuous(breaks = XBK, labels = XLB, limits = XLIM) +
  scale_y_continuous(
    breaks = Y_SUB,
    labels = names(Y_SUB),
    limits = c(-0.2, 7.0)) +
  labs(x = "Relative risk (95%\u202fCI)", y = NULL,
       caption = paste0("Category 4: extreme-heat day + HPII\u202f\u22651\n",
                        "Mexican death cities only (n\u202f=\u202f3); ",
                        "95% CI not crossing 1.0 indicates statistical significance")) +
  th +
  theme(
    axis.text.y  = element_text(size = PTs, hjust = 1),
    legend.position = "none",
    plot.caption = element_text(size = 5, color = "grey40", hjust = 0,
                                lineheight = 1.2, margin = margin(t = 1.5, "mm"))
  )

## =============================================================================
## FIGURE 5 — Combine panels a and b
## =============================================================================

p5 <- pA + pB +
  plot_layout(widths = c(13, 8)) +
  plot_annotation(
    tag_levels  = "a",
    theme = theme(
      plot.tag          = element_text(size = PT + 1, face = "bold"),
      plot.tag.position = "topleft",
      plot.margin       = margin(0, 0, 0, 0, "mm")
    )
  )

ggsave("output/figures_warmseason/Fig5_forest_plot.pdf",
       p5, width = 180, height = 135, units = "mm")
cat("Fig5 saved (180x135mm)\n")

## =============================================================================
## SUPPLEMENTARY BARPLOTS — vertical 100% stacked bars, 9 categories
## x = outcome class (demographics + all ICD A-Z)
## fill = 9-category winner proportion
## =============================================================================

class_nice9 <- function(x) {
  dplyr::case_when(
    x == "all-cause" ~ "All-cause",
    x == "age0_69"   ~ "Age 0-69",
    x == "age70_"    ~ "Age 70+",
    x == "female"    ~ "Female",
    x == "male"      ~ "Male",
    TRUE ~ x            # keep ICD letters A-Z as-is
  )
}

make_supp_bar <- function(outcome_name, file_tag, w_mm = 220, h_mm = 130) {

  path <- paste0("which_better/", outcome_name, ".xlsx")
  raw  <- read_excel(path)

  dat <- raw %>%
    mutate(class = class_nice9(class)) %>%
    select(class, city_num,
           `P-P`,`P-S`,`P-T`,`S-P`,`S-S`,`S-T`,`T-P`,`T-S`,`T-T`) %>%
    filter(!is.na(city_num), city_num > 0) %>%
    pivot_longer(`P-P`:`T-T`, names_to = "cat", values_to = "n") %>%
    mutate(
      cat = factor(cat, levels = CAT9_LEV),
      n   = replace_na(n, 0),
      pct = n / city_num * 100
    ) %>%
    filter(is.finite(pct))

  demo        <- c("All-cause","Age 0-69","Age 70+","Female","Male")
  other       <- setdiff(unique(dat$class), demo)
  class_order <- c(demo[demo %in% unique(dat$class)], sort(other))
  dat <- dat %>% mutate(class = factor(class, levels = class_order))

  ggplot(dat, aes(x = class, y = pct, fill = cat)) +
    geom_col(width = 0.72, color = NA, position = "stack") +
    scale_fill_manual(values = COL9, name = "Model comparison",
      guide = guide_legend(nrow = 1, override.aes = list(color = NA))) +
    scale_y_continuous(limits = c(0, 101), breaks = seq(0, 100, 25),
                       labels = paste0(seq(0, 100, 25), "%"),
                       expand = c(0, 0)) +
    labs(x = NULL, y = "Proportion of cities",
         title = paste0(outcome_name,
                        ": proportion of winner categories across outcomes")) +
    theme_bw(base_size = PT) +
    theme(
      panel.grid.minor   = element_blank(),
      panel.grid.major.y = element_line(color = "#E8E8E8", linewidth = 0.25),
      panel.grid.major.x = element_blank(),
      axis.text.x        = element_text(size = PTs, angle = 45, hjust = 1),
      axis.text.y        = element_text(size = PTs),
      axis.title         = element_text(size = PT),
      legend.text        = element_text(size = PTs),
      legend.key.size    = unit(0.22, "cm"),
      legend.position    = "bottom",
      legend.margin      = margin(0, 0, 0, 0),
      plot.title         = element_text(size = PT, face = "bold"),
      plot.margin        = margin(4, 4, 4, 4, "mm")
    )
}

for (out in list(
    list(name = "Death",          tag = "Death", w = 220, h = 130),
    list(name = "Hospitalization", tag = "Hosp",  w = 220, h = 130),
    list(name = "ED",             tag = "ED",    w = 130, h = 110))) {
  p_s <- tryCatch(
    make_supp_bar(out$name, out$tag, out$w, out$h),
    error = function(e) { cat("  barplot ERROR:", out$name, conditionMessage(e), "\n"); NULL })
  if (!is.null(p_s)) {
    fname <- paste0("output/figures_warmseason/Supp_barplot_", out$tag, ".pdf")
    ggsave(fname, p_s, width = out$w, height = out$h, units = "mm")
    cat("Supp barplot saved:", fname, "\n")
  }
}

## =============================================================================
## SUPPLEMENTARY HEATMAPS — class (rows) x winner-cat (cols)
## fill = COUNT of cities with that winner, green gradient
## =============================================================================

make_supp_heat <- function(outcome_name, file_tag, w_mm = 220, h_mm = 180) {

  path <- paste0("which_better/", outcome_name, ".xlsx")
  raw  <- read_excel(path)

  dat <- raw %>%
    mutate(class = class_nice9(class)) %>%
    select(class, city_num,
           `P-P`,`P-S`,`P-T`,`S-P`,`S-S`,`S-T`,`T-P`,`T-S`,`T-T`) %>%
    filter(!is.na(city_num), city_num > 0) %>%
    pivot_longer(`P-P`:`T-T`, names_to = "cat", values_to = "n") %>%
    mutate(
      cat = factor(cat, levels = CAT9_LEV),
      n   = replace_na(n, 0L)
    )

  ## Row order: demographics first, then A-Z ICD
  demo        <- c("All-cause","Age 0-69","Age 70+","Female","Male")
  other       <- setdiff(unique(dat$class), demo)
  class_order <- c(demo[demo %in% unique(dat$class)], sort(other))
  dat <- dat %>%
    mutate(class = factor(class, levels = rev(class_order)))   # rev -> top-to-bottom

  n_max <- max(dat$n, na.rm = TRUE)

  ggplot(dat, aes(x = cat, y = class, fill = n)) +
    geom_tile(color = "white", linewidth = 0.4) +
    geom_text(aes(label = n), size = NC_s * 0.90, color = "black") +
    scale_fill_gradient(
      low    = "#FFFFFF", high = "#00441B",
      limits = c(0, n_max),
      name   = "Cities",
      breaks = seq(0, n_max, by = 1)
    ) +
    scale_x_discrete(position = "bottom") +
    labs(x = "Winner category", y = NULL,
         title = paste0(outcome_name,
                        ": heatmap of winner categories across outcomes")) +
    theme_bw(base_size = PT) +
    theme(
      axis.text.x    = element_text(size = PTs),
      axis.text.y    = element_text(size = PTs),
      legend.text    = element_text(size = PTs),
      legend.key.size = unit(0.30, "cm"),
      legend.title   = element_text(size = PTs),
      legend.position = "right",
      plot.title     = element_text(size = PT, face = "bold"),
      plot.margin    = margin(4, 4, 4, 4, "mm"),
      panel.grid     = element_blank()
    )
}

for (out in list(
    list(name = "Death",          tag = "Death", w = 160, h = 200),
    list(name = "Hospitalization", tag = "Hosp",  w = 160, h = 200),
    list(name = "ED",             tag = "ED",    w = 130, h = 110))) {
  p_h <- tryCatch(
    make_supp_heat(out$name, out$tag, out$w, out$h),
    error = function(e) { cat("  heatmap ERROR:", out$name, conditionMessage(e), "\n"); NULL })
  if (!is.null(p_h)) {
    fname <- paste0("output/figures_warmseason/Supp_heatmap_", out$tag, ".pdf")
    ggsave(fname, p_h, width = out$w, height = out$h, units = "mm")
    cat("Supp heatmap saved:", fname, "\n")
  }
}

cat("\n=== All figures complete ===\n")
cat("Main:  output/figures_warmseason/Fig4_model_comparison.pdf\n")
cat("Main:  output/figures_warmseason/Fig5_forest_plot.pdf\n")
cat("Supp:  output/figures_warmseason/Supp_barplot_*.pdf\n")
cat("Supp:  output/figures_warmseason/Supp_heatmap_*.pdf\n")

## =============================================================================
## Session information
## =============================================================================
sessionInfo()
