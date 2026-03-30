## =============================================================================
## Script 04: Joint temperature-perception exposure effects — full-year sensitivity
##
## Identical specification to Script 02 but uses all 365 days per city.
## Outputs are used for Supplementary Figure 5 (sensitivity).
##
## Input:  final_dat.rds
## Output: effect_size_allyear/hospitalization.csv
##         effect_size_allyear/death.csv
##         effect_size_allyear/city_level_rr.csv
## =============================================================================

# Set working directory to the folder containing final_dat.rds before running this script.

# NOTE: Full-year analysis — warm_season filter is intentionally omitted.

library(dplyr)
library(tidyr)
library(ggplot2)
library(lubridate)
library(splines)
library(dlnm)
library(patchwork)
library(scales)

rm(list = ls()); gc()

## =============================================================================
## Section 1: Load and prepare data (full year — no seasonal filter)
## =============================================================================

df <- read_rds('final_dat.rds')

# Exclude Benito Juárez (Cancún area) — geographically distinct from other Mexican sites
df <- df %>% filter(city != "Benito Juárez")

# Compute warm_season indicator (retained for reference only; not used to filter).
# Compute binary exposure indicators:
#   heatday  = 1 if daily temperature exceeds city-specific 95th percentile
#   percep2  = 1 if HPII > 0
df <- df %>%
  mutate(warm_season = case_when(
    country == 'Mexico' & month %in% 5:9             ~ 1,
    country == 'Mexico' & month %in% c(10:12, 1:4)  ~ 0,
    country != 'Mexico' & month %in% 5:9             ~ 0,
    country != 'Mexico' & month %in% c(10:12, 1:4)  ~ 1
  )) %>%
  group_by(city) %>%
  mutate(
    heatday = ifelse(temp > quantile(temp, 0.95), 1, 0),
    percep2 = ifelse(heatPerception > 0, 1, 0)
  ) %>%
  ungroup()
  # NOTE: Full-year analysis — warm_season filter is intentionally omitted.

## =============================================================================
## Section 2: Create 4-category joint exposure variable
## =============================================================================

# comb_exp categories:
#   1 = Normal temperature (heatday=0) & Low perception (percep2=0)  [REFERENCE]
#   2 = Normal temperature (heatday=0) & High perception (percep2=1)
#   3 = Extreme heat (heatday=1)       & Low perception (percep2=0)
#   4 = Extreme heat (heatday=1)       & High perception (percep2=1)

df %>%
  mutate(
    comb_exp = case_when(
      heatday == 0 & percep2 == 0 ~ 1,
      heatday == 0 & percep2 == 1 ~ 2,
      heatday == 1 & percep2 == 0 ~ 3,
      heatday == 1 & percep2 == 1 ~ 4
    ),
    comb_exp = factor(comb_exp, levels = c('1', '2', '3', '4'))
  ) -> df

## =============================================================================
## Section 3: Model fitting functions
## =============================================================================

# Create crossbasis with strata argvar (breaks at 2,3,4 for 4 categories) and strata arglag.
create_cb <- function(x, lag_days) {
  dlnm::crossbasis(x, lag = lag_days,
                   argvar = list(fun = 'strata', breaks = c(2, 3, 4)),
                   arglag = list(fun = "strata"))
}

# Fit quasi-Poisson DLNM with standard covariates.
fit_mod <- function(dat, exp, lag_days) {
  cb <- create_cb(exp, lag_days)
  fm <- as.formula("count ~ cb + factor(dow) + holiday + factor(month) + pm25")
  m  <- glm(fm, family = quasipoisson(link = "log"), data = dat)
  return(m)
}

compare_mod <- function(mod_temp, mod_perc) {
  bind_rows(
    calculate_metric(mod_temp) %>% mutate(exp = 'temp'),
    calculate_metric(mod_perc) %>% mutate(exp = 'perc')
  )
}

## =============================================================================
## Section 4: Loop 1 — city-level RR estimates (crosspred)
## =============================================================================

res   <- data.frame()
citys <- unique(df$city)

for (i in 1:length(citys)) {
  df_city <- df %>% dplyr::filter(city == citys[i])
  classes <- unique(df_city$class)
  for (j in 1:length(classes)) {
    cat(citys[i], ':', classes[j], '\n')
    data <- df_city %>% dplyr::filter(class == classes[j])
    tmp <- tryCatch({
      cb   <- create_cb(x = data$comb_exp, lag_days = 2)
      mod  <- fit_mod(dat = data, exp = data$comb_exp, lag_days = 2)
      pred <- crosspred(cb, mod, at = c(1, 2, 3, 4), cen = 1)
      data.frame(
        case   = c(1, 2, 3, 4),
        rr     = pred$allRRfit,
        rrlow  = pred$allRRlow,
        rrhigh = pred$allRRhigh,
        beta   = pred$allfit,
        se     = pred$allse
      ) %>%
        mutate(
          country = data$country[1],
          city    = data$city[1],
          type    = data$type[1],
          class   = classes[j]
        ) %>%
        relocate(case, rr, rrlow, rrhigh, beta, se, .after = class)
    }, error = function(e) {
      data.frame(
        country = data$country[1],
        city    = data$city[1],
        type    = data$type[1],
        class   = classes[j],
        case    = c(1, 2, 3, 4),
        rr      = NA_real_,
        rrlow   = NA_real_,
        rrhigh  = NA_real_,
        beta    = NA_real_,
        se      = NA_real_
      ) %>%
        relocate(case, rr, rrlow, rrhigh, beta, se, .after = class)
    })
    res <- rbind(res, tmp)
  }
}

res %>%
  as_tibble() %>%
  dplyr::filter(!is.na(rr)) %>%
  mutate(case = ifelse(case == 1, 'Normal & Low',
                       ifelse(case == 2, 'Normal & High',
                              ifelse(case == 3, 'Heat & Low', 'Heat & High')))) -> rr

## Save city-level RR estimates
dir.create('effect_size_allyear', showWarnings = FALSE)
write_csv(rr, 'effect_size_allyear/city_level_rr.csv')
cat("Saved: city_level_rr.csv (", nrow(rr), "rows )\n")

## Save category day counts per city x class
cat_counts <- df %>%
  filter(class == 'all-cause') %>%
  group_by(city, country, type) %>%
  summarise(
    n_days = n(),
    cat1_n = sum(comb_exp == 1, na.rm = TRUE),
    cat2_n = sum(comb_exp == 2, na.rm = TRUE),
    cat3_n = sum(comb_exp == 3, na.rm = TRUE),
    cat4_n = sum(comb_exp == 4, na.rm = TRUE),
    .groups = 'drop'
  )
write_csv(cat_counts, 'effect_size_allyear/category_day_counts.csv')
cat("Saved: category_day_counts.csv\n")

## =============================================================================
## Section 5: Loop 2 — crossreduce -> coef + vcov for meta-analysis
## =============================================================================

est   <- data.frame()
citys <- unique(df$city)

for (i in 1:length(citys)) {
  df_city <- df %>% dplyr::filter(city == citys[i])
  classes <- unique(df_city$class)
  for (j in 1:length(classes)) {
    cat(citys[i], ':', classes[j], '\n')
    data <- df_city %>% dplyr::filter(class == classes[j])
    tmp <- tryCatch({
      cb  <- create_cb(x = data$comb_exp, lag_days = 2)
      mod <- fit_mod(dat = data, exp = data$comb_exp, lag_days = 2)
      red <- crossreduce(cb, mod, type = 'overall')
      trimat <- row(vcov(red)) >= col(vcov(red))
      vvec   <- vcov(red)[trimat]
      data.frame(
        country = data$country[1],
        city    = data$city[1],
        type    = data$type[1],
        class   = classes[j],
        t(coef(red)),
        t(vvec)
      ) %>%
        purrr::set_names(c('country', 'city', 'type', 'class',
                           paste0('b', 1:3),
                           paste0('v', 1:3, '1'),
                           paste0('v', 2:3, '2'),
                           paste0('v', 3:3, '3')))
    }, error = function(e) {
      data.frame(
        country = data$country[1],
        city    = data$city[1],
        type    = data$type[1],
        class   = classes[j],
        t(rep(NA, 9))
      ) %>%
        purrr::set_names(c('country', 'city', 'type', 'class',
                           paste0('b', 1:3),
                           paste0('v', 1:3, '1'),
                           paste0('v', 2:3, '2'),
                           paste0('v', 3:3, '3')))
    })
    est <- rbind(est, tmp)
  }
}

est %>%
  as_tibble() %>%
  dplyr::filter(!is.na(b1)) -> est

## =============================================================================
## Section 6: Meta-analysis and figure — Hospitalization
## =============================================================================

est %>%
  as_tibble() %>%
  dplyr::filter(type == 'Hospitlization') -> tmp

classes <- unique(tmp$class)
library(mixmeta)
hosp <- data.frame()

for (i in 1:length(classes)) {
  tmp %>%
    dplyr::filter(class == classes[i]) -> metadf
  coef <- dplyr::select(metadf, matches("b[[:digit:]]")) |> data.matrix()
  vcov <- dplyr::select(metadf, matches("v[[:digit:]][[:digit:]]")) |>
    # Reconstruct symmetric variance-covariance matrix from lower triangle
    apply(1, function(x) {
      nred <- as.integer(substr(tail(names(x), 1), 2, 2))
      m <- matrix(NA, nred, nred)
      m[lower.tri(m, diag = T)] <- unlist(x)
      m[upper.tri(m)] <- t(m)[upper.tri(m)]
      m
    }, simplify = F)
  meta <- mixmeta(coef, vcov)
  cb <- crossbasis(
    x      = c(1, 2, 3, 4),
    lag    = 2,
    argvar = list(fun = 'strata', breaks = c(2, 3, 4)),
    arglag = list(fun = 'strata')
  )
  pred <- crosspred(cb, coef = coef(meta), vcov = vcov(meta), model.link = 'log', at = c(1, 2, 3, 4))
  data.frame(
    type   = tmp$type[1],
    class  = classes[i],
    case   = c('Normal & Low', 'Normal & High', 'Heat & Low', 'Heat & High'),
    rr     = pred$allRRfit,
    rrlow  = pred$allRRlow,
    rrhigh = pred$allRRhigh
  ) -> tmpres
  hosp <- rbind(hosp, tmpres)
}

upper_cap <- 3
plot_dat <- hosp %>%
  mutate(
    case       = factor(case, levels = c("Normal & Low", "Normal & High", "Heat & Low", "Heat & High")),
    class      = factor(class, levels = c("all-cause", "age0_69", "age70_", "female", "male", LETTERS)),
    ymin_plot  = rrlow,
    ymax_plot  = pmin(rrhigh, upper_cap),
    high_trunc = rrhigh > upper_cap
  )

case_cols <- c(
  "Normal & Low"  = "#4C78A8",
  "Normal & High" = "#72B7B2",
  "Heat & Low"    = "#F58518",
  "Heat & High"   = "#E45756"
)

ggplot(plot_dat, aes(x = case, y = rr, color = case)) +
  geom_hline(yintercept = 1, linetype = 2, linewidth = 0.4, color = "grey40") +
  geom_errorbar(aes(ymin = ymin_plot, ymax = ymax_plot), width = 0.12, linewidth = 0.5) +
  geom_point(size = 2) +
  geom_point(
    data = subset(plot_dat, high_trunc),
    aes(x = case, y = upper_cap),
    shape = 24, size = 2.6, fill = "black", color = "black",
    inherit.aes = FALSE
  ) +
  facet_wrap(~class, scales = "free_y", ncol = 4) +
  scale_color_manual(values = case_cols) +
  coord_cartesian(ylim = c(min(plot_dat$rrlow, na.rm = TRUE), upper_cap)) +
  labs(
    x     = NULL,
    y     = "Relative risk",
    color = NULL
  ) +
  theme_bw(base_size = 12) +
  theme(
    panel.grid.minor     = element_blank(),
    panel.grid.major.x   = element_blank(),
    strip.background     = element_rect(fill = "#F2F2F2", color = "#D9D9D9"),
    strip.text           = element_text(face = "bold"),
    axis.text.x          = element_text(angle = 35, hjust = 1),
    legend.position      = "bottom"
  ) +
  guides(color = guide_legend(nrow = 1, byrow = TRUE)) -> p

dir.create('effect_size_allyear', showWarnings = FALSE)
ggsave(p, filename = 'effect_size_allyear/hospitalization.pdf', width = 10, height = 12)
write_csv(plot_dat, file = 'effect_size_allyear/hospitalization.csv')

## =============================================================================
## Section 7: Meta-analysis and figure — Death
## =============================================================================

est %>%
  as_tibble() %>%
  dplyr::filter(type == 'Death') -> tmp

classes <- unique(tmp$class)
library(mixmeta)
death <- data.frame()

for (i in 1:length(classes)) {
  tmp %>%
    dplyr::filter(class == classes[i]) -> metadf
  coef <- dplyr::select(metadf, matches("b[[:digit:]]")) |> data.matrix()
  vcov <- dplyr::select(metadf, matches("v[[:digit:]][[:digit:]]")) |>
    # Reconstruct symmetric variance-covariance matrix from lower triangle
    apply(1, function(x) {
      nred <- as.integer(substr(tail(names(x), 1), 2, 2))
      m <- matrix(NA, nred, nred)
      m[lower.tri(m, diag = T)] <- unlist(x)
      m[upper.tri(m)] <- t(m)[upper.tri(m)]
      m
    }, simplify = F)
  meta <- mixmeta(coef, vcov)
  cb <- crossbasis(
    x      = c(1, 2, 3, 4),
    lag    = 2,
    argvar = list(fun = 'strata', breaks = c(2, 3, 4)),
    arglag = list(fun = 'strata')
  )
  pred <- crosspred(cb, coef = coef(meta), vcov = vcov(meta), model.link = 'log', at = c(1, 2, 3, 4))
  data.frame(
    type   = tmp$type[1],
    class  = classes[i],
    case   = c('Normal & Low', 'Normal & High', 'Heat & Low', 'Heat & High'),
    rr     = pred$allRRfit,
    rrlow  = pred$allRRlow,
    rrhigh = pred$allRRhigh
  ) -> tmpres
  death <- rbind(death, tmpres)
}

death %>% nrow()
upper_cap <- 3
plot_dat <- death %>%
  mutate(
    case       = factor(case, levels = c("Normal & Low", "Normal & High", "Heat & Low", "Heat & High")),
    class      = factor(class, levels = c("all-cause", "age0_69", "age70_", "female", "male", LETTERS)),
    ymin_plot  = rrlow,
    ymax_plot  = pmin(rrhigh, upper_cap),
    high_trunc = rrhigh > upper_cap
  )

case_cols <- c(
  "Normal & Low"  = "#4C78A8",
  "Normal & High" = "#72B7B2",
  "Heat & Low"    = "#F58518",
  "Heat & High"   = "#E45756"
)

ggplot(plot_dat, aes(x = case, y = rr, color = case)) +
  geom_hline(yintercept = 1, linetype = 2, linewidth = 0.4, color = "grey40") +
  geom_errorbar(aes(ymin = ymin_plot, ymax = ymax_plot), width = 0.12, linewidth = 0.5) +
  geom_point(size = 2) +
  geom_point(
    data = subset(plot_dat, high_trunc),
    aes(x = case, y = upper_cap),
    shape = 24, size = 2.6, fill = "black", color = "black",
    inherit.aes = FALSE
  ) +
  facet_wrap(~class, scales = "free_y", ncol = 4) +
  scale_color_manual(values = case_cols) +
  coord_cartesian(ylim = c(min(plot_dat$rrlow, na.rm = TRUE), upper_cap)) +
  labs(
    x     = NULL,
    y     = "Relative risk",
    color = NULL
  ) +
  theme_bw(base_size = 12) +
  theme(
    panel.grid.minor     = element_blank(),
    panel.grid.major.x   = element_blank(),
    strip.background     = element_rect(fill = "#F2F2F2", color = "#D9D9D9"),
    strip.text           = element_text(face = "bold"),
    axis.text.x          = element_text(angle = 35, hjust = 1),
    legend.position      = "bottom"
  ) +
  guides(color = guide_legend(nrow = 1, byrow = TRUE)) -> p
p
dir.create('effect_size_allyear', showWarnings = FALSE)
ggsave(p, filename = 'effect_size_allyear/death.pdf', width = 10, height = 12)
write_csv(plot_dat, file = 'effect_size_allyear/death.csv')

## =============================================================================
## Session information
## =============================================================================
sessionInfo()
