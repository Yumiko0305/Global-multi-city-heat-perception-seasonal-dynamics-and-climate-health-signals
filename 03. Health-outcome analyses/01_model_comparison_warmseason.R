## =============================================================================
## Script 01: Model comparison — warm-season analysis (primary)
##
## Compares the goodness-of-fit of two parallel DLNM models per city x outcome:
##   - Temperature-based model: exposure = binary heat-day (temp > 95th pctile)
##   - Perception-based model:  exposure = binary heat-perception (HPII >= 1)
##
## Criteria: QAIC (quasi-Akaike) and pseudo-R2
## Winner categories: P (perception), T (temperature), S (similar)
## Threshold: |dQAIC| <= 2 -> Similar; |dR2| <= 0.02 -> Similar
##
## Input:  final_dat.rds
## Output: which_better/Death.xlsx
##         which_better/Hospitalization.xlsx
##         which_better/ED.xlsx
## =============================================================================

# Set working directory to the folder containing final_dat.rds before running this script.

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
## Section 1: Load and prepare data
## =============================================================================

df <- read_rds('final_dat.rds')

# Exclude Benito Juárez (Cancún area) — geographically distinct from other Mexican sites
df <- df %>% filter(city != "Benito Juárez")

# Define warm season:
#   Mexico:              May–September (months 5–9)
#   Brazil and Australia: October–April (months 10–12, 1–4; Southern Hemisphere summer)
# Compute binary exposure indicators:
#   heatday  = 1 if daily temperature exceeds city-specific 95th percentile
#   percep2  = 1 if HPII >= 1
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
    percep2 = ifelse(heatPerception >= 1, 1, 0)
  ) %>%
  ungroup() %>%
  dplyr::filter(warm_season == 1)

## =============================================================================
## Section 2: Model fitting and metric functions
## =============================================================================

# Compute QAIC and pseudo-R2 for a fitted quasi-Poisson GLM.
# QAIC = -2 * log-likelihood + 2 * df * phi
#   phi     = estimated dispersion parameter
#   loglik  = sum of Poisson log-likelihood evaluated at fitted values
#   df      = total degrees of freedom used by the model
calculate_metric <- function(model) {
  phi    <- summary(model)$dispersion
  loglik <- sum(dpois(model$y, model$fitted.values, log = TRUE))
  qaic   <- -2 * loglik + 2 * summary(model)$df[3] * phi
  r2     <- 1 - (model$deviance / model$null.deviance)
  return(data.frame(QAIC = qaic, R2 = r2))
}

# Create a crossbasis with linear argvar and strata arglag (lag = lag_days).
create_cb <- function(x, lag_days) {
  dlnm::crossbasis(x, lag = lag_days,
                   argvar = list(fun = 'lin'),
                   arglag = list(fun = "strata"))
}

# Fit quasi-Poisson DLNM with standard covariates.
fit_mod <- function(dat, exp, lag_days) {
  cb <- create_cb(exp, lag_days)
  fm <- as.formula("count ~ cb + factor(dow) + holiday + factor(month) + pm25")
  m  <- glm(fm, family = quasipoisson(link = "log"), data = dat)
  return(m)
}

# Compute metrics for both models and return a combined data frame.
compare_mod <- function(mod_temp, mod_perc) {
  bind_rows(
    calculate_metric(mod_temp) %>% mutate(exp = 'temp'),
    calculate_metric(mod_perc) %>% mutate(exp = 'perc')
  )
}

## =============================================================================
## Section 3: Main loop — fit models for each city x outcome class
## =============================================================================

res   <- data.frame()
citys <- unique(df$city)

for (i in 1:length(citys)) {
  df_city <- df %>% dplyr::filter(city == citys[i])
  classes <- unique(df_city$class)
  for (j in 1:length(classes)) {
    cat(citys[i], ':', classes[j], '\n')
    data     <- df_city %>% dplyr::filter(class == classes[j])
    mod_temp <- fit_mod(dat = data, exp = data$heatday, lag_days = 2)
    mod_perc <- fit_mod(dat = data, exp = data$percep2, lag_days = 2)
    compare_mod(mod_temp, mod_perc) %>%
      mutate(
        country   = data$country[1],
        city      = citys[i],
        type      = data$type[1],
        class     = classes[j]
      ) %>%
      mutate(converage = c(mod_temp$converged, mod_perc$converged)) %>%
      relocate(country, city, type, class, exp, QAIC, R2) %>%
      pivot_wider(names_from = exp, values_from = c(QAIC, R2, converage)) -> tmp
    res <- rbind(res, tmp)
  }
}

## =============================================================================
## Section 4: Classify winners
## =============================================================================

# delta_qaic = QAIC_temp - QAIC_perc
#   > 2   -> P wins (perception model is better by QAIC)
#   < -2  -> T wins (temperature model is better by QAIC)
#   else  -> S (similar)
#
# delta_r2 = R2_temp - R2_perc
#   > 0.02  -> T wins (temperature explains more variance)
#   < -0.02 -> P wins (perception explains more variance)
#   else    -> S (similar)
#
# Combined winner = paste(qaic_winner, r2_winner, sep = "-")

res %>%
  dplyr::filter((converage_temp & converage_perc)) %>%
  dplyr::filter(!(is.infinite(R2_perc) | is.infinite(R2_temp))) %>%
  mutate(
    delta_qaic = QAIC_temp - QAIC_perc,
    delta_r2   = R2_temp   - R2_perc
  ) %>%
  mutate(
    qaic_winner = case_when(
      abs(delta_qaic) <= 2  ~ 'S',
      delta_qaic  >  2      ~ 'P',
      delta_qaic  < -2      ~ 'T'
    ),
    r2_winnter = case_when(
      abs(delta_r2) <= 0.02 ~ 'S',
      delta_r2  >  0.02     ~ 'T',
      delta_r2  < -0.02     ~ 'P'
    )
  ) %>%
  mutate(winner = paste0(qaic_winner, '-', r2_winnter)) -> df

## =============================================================================
## Section 5: Aggregate and export — Hospitalization
## =============================================================================

df %>%
  as_tibble() %>%
  dplyr::filter(type == 'Hospitlization') %>%
  group_by(class) %>%
  summarise(city_num = n()) -> city_num

df %>%
  as_tibble() %>%
  dplyr::filter(type == 'Hospitlization') %>%
  group_by(type, class, winner) %>%
  summarise(winner_num = n()) %>%
  ungroup() %>%
  mutate(class = factor(class, levels = c('all-cause', 'age0_69', 'age70_', 'female', 'male', LETTERS))) -> count

plot_dat <- expand.grid(
  type   = 'Hospitlization',
  class  = c('all-cause', 'age0_69', 'age70_', 'female', 'male', LETTERS),
  winner = c('P-P', 'P-S', 'P-T', 'S-P', 'S-S', 'S-T', 'T-P', 'T-S', 'T-T')
) %>%
  left_join(count, by = c('type', 'class', 'winner')) %>%
  as_tibble() %>%
  left_join(city_num, by = 'class') %>%
  mutate(winner_num = ifelse(is.na(winner_num), 0, winner_num)) %>%
  pivot_wider(names_from = winner, values_from = winner_num) %>%
  mutate(class = factor(class, levels = c('all-cause', 'age0_69', 'age70_', 'female', 'male', LETTERS))) %>%
  arrange(type, class) %>%
  relocate(`P-P`, `P-S`, `P-T`, `S-P`, `S-S`, `S-T`, `T-P`, `T-S`, `T-T`, .after = city_num)

dir.create('which_better', showWarnings = FALSE)
writexl::write_xlsx(plot_dat, path = 'which_better/Hospitalization.xlsx')

plot_long <- plot_dat %>%
  pivot_longer(
    cols      = `P-P`:`T-T`,
    names_to  = "winner",
    values_to = "n"
  ) %>%
  mutate(
    winner = factor(winner,
      levels = c("P-P", "P-S", "P-T", "S-P", "S-S", "S-T", "T-P", "T-S", "T-T")),
    class = factor(class,
      levels = rev(c("all-cause", "age0_69", "age70_", "female", "male", LETTERS)))
  )

# Color scheme: Perception = blue, Similar = grey, Temperature = red
winner_cols <- c(
  "P-P" = "#2166AC",  # deep blue
  "P-S" = "#67A9CF",  # blue
  "P-T" = "#B2ABD2",  # blue-purple
  "S-P" = "#92C5DE",  # light blue
  "S-S" = "#D9D9D9",  # neutral grey
  "S-T" = "#F4A582",  # light orange
  "T-P" = "#D8B365",  # tan
  "T-S" = "#EF8A62",  # orange-red
  "T-T" = "#B2182B"   # deep red
)

class_labs <- c(
  "all-cause" = "All-cause",
  "age0_69"   = "Age 0-69",
  "age70_"    = "Age 70+",
  "female"    = "Female",
  "male"      = "Male"
)

ggplot(plot_long %>%
         mutate(class = factor(class, levels = c('all-cause', 'age0_69', 'age70_', 'female', 'male', LETTERS))),
       aes(x = class, y = n, fill = winner)) +
  geom_col(position = "fill", width = 0.78, color = "white", linewidth = 0.25) +
  scale_fill_manual(values = winner_cols, drop = FALSE) +
  scale_y_continuous(
    labels = percent_format(accuracy = 1),
    expand = expansion(mult = c(0, 0.02))
  ) +
  guides(fill = guide_legend(nrow = 1, byrow = TRUE)) +
  scale_x_discrete(labels = function(x) ifelse(x %in% names(class_labs), class_labs[x], x)) +
  labs(
    x     = NULL,
    y     = "Proportion of cities",
    fill  = "Model comparison",
    title = "Hospitalization: proportion of winner categories across outcomes"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.minor   = element_blank(),
    panel.grid.major.y = element_line(color = "#EAEAEA", linewidth = 0.35),
    axis.text.x        = element_text(angle = 40, hjust = 1, vjust = 1),
    axis.title.y       = element_text(size = 12),
    plot.title         = element_text(face = "bold", size = 13),
    legend.position    = "bottom",
    legend.title       = element_text(size = 11),
    legend.text        = element_text(size = 10)
  ) -> p
ggsave(p, filename = 'which_better/Hospitalization_barplot.pdf', width = 10, height = 8)

ggplot(plot_long, aes(x = winner, y = class, fill = n)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = n), size = 3.3, fontface = "bold", color = "black") +
  scale_fill_gradientn(
    colours = c("#F7FCFD", "#CCECE6", "#66C2A4", "#238B45"),
    limits  = c(0, max(plot_long$n, na.rm = TRUE)),
    breaks  = 0:max(plot_long$n, na.rm = TRUE),
    name    = "Cities"
  ) +
  scale_x_discrete(drop = FALSE) +
  scale_y_discrete(labels = function(x) ifelse(x %in% names(class_labs), class_labs[x], x)) +
  labs(
    x     = "Winner category",
    y     = NULL,
    title = "Hospitalization: heatmap of winner categories across outcomes"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid  = element_blank(),
    axis.text.x = element_text(angle = 0, hjust = 1, vjust = 1),
    plot.title  = element_text(face = "bold", size = 13),
    legend.position = "right",
    legend.title    = element_text(size = 11),
    legend.text     = element_text(size = 10)
  ) -> p
ggsave(p, filename = 'which_better/Hospitalization_heatmap.pdf', width = 10, height = 8)

## =============================================================================
## Section 6: Aggregate and export — Death
## =============================================================================

df %>%
  as_tibble() %>%
  dplyr::filter(type == 'Death') %>%
  group_by(class) %>%
  summarise(city_num = n()) -> city_num

df %>%
  as_tibble() %>%
  dplyr::filter(type == 'Death') %>%
  group_by(type, class, winner) %>%
  summarise(winner_num = n()) %>%
  ungroup() %>%
  mutate(class = factor(class, levels = c('all-cause', 'age0_69', 'age70_', 'female', 'male', LETTERS))) -> count

plot_dat <- expand.grid(
  type   = 'Death',
  class  = c('all-cause', 'age0_69', 'age70_', 'female', 'male', LETTERS),
  winner = c('P-P', 'P-S', 'P-T', 'S-P', 'S-S', 'S-T', 'T-P', 'T-S', 'T-T')
) %>%
  left_join(count, by = c('type', 'class', 'winner')) %>%
  as_tibble() %>%
  left_join(city_num, by = 'class') %>%
  mutate(winner_num = ifelse(is.na(winner_num), 0, winner_num)) %>%
  pivot_wider(names_from = winner, values_from = winner_num) %>%
  mutate(class = factor(class, levels = c('all-cause', 'age0_69', 'age70_', 'female', 'male', LETTERS))) %>%
  arrange(type, class) %>%
  relocate(`P-P`, `P-S`, `P-T`, `S-P`, `S-S`, `S-T`, `T-P`, `T-S`, `T-T`, .after = city_num)

dir.create('which_better', showWarnings = FALSE)
writexl::write_xlsx(plot_dat, path = 'which_better/Death.xlsx')

plot_long <- plot_dat %>%
  pivot_longer(
    cols      = `P-P`:`T-T`,
    names_to  = "winner",
    values_to = "n"
  ) %>%
  mutate(
    winner = factor(winner,
      levels = c("P-P", "P-S", "P-T", "S-P", "S-S", "S-T", "T-P", "T-S", "T-T")),
    class = factor(class,
      levels = rev(c("all-cause", "age0_69", "age70_", "female", "male", LETTERS)))
  )

# Color scheme: Perception = blue, Similar = grey, Temperature = red
winner_cols <- c(
  "P-P" = "#2166AC",  # deep blue
  "P-S" = "#67A9CF",  # blue
  "P-T" = "#B2ABD2",  # blue-purple
  "S-P" = "#92C5DE",  # light blue
  "S-S" = "#D9D9D9",  # neutral grey
  "S-T" = "#F4A582",  # light orange
  "T-P" = "#D8B365",  # tan
  "T-S" = "#EF8A62",  # orange-red
  "T-T" = "#B2182B"   # deep red
)

class_labs <- c(
  "all-cause" = "All-cause",
  "age0_69"   = "Age 0-69",
  "age70_"    = "Age 70+",
  "female"    = "Female",
  "male"      = "Male"
)

ggplot(plot_long %>%
         mutate(class = factor(class, levels = c('all-cause', 'age0_69', 'age70_', 'female', 'male', LETTERS))),
       aes(x = class, y = n, fill = winner)) +
  geom_col(position = "fill", width = 0.78, color = "white", linewidth = 0.25) +
  scale_fill_manual(values = winner_cols, drop = FALSE) +
  scale_y_continuous(
    labels = percent_format(accuracy = 1),
    expand = expansion(mult = c(0, 0.02))
  ) +
  guides(fill = guide_legend(nrow = 1, byrow = TRUE)) +
  scale_x_discrete(labels = function(x) ifelse(x %in% names(class_labs), class_labs[x], x)) +
  labs(
    x     = NULL,
    y     = "Proportion of cities",
    fill  = "Model comparison",
    title = "Death: proportion of winner categories across outcomes"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.minor   = element_blank(),
    panel.grid.major.y = element_line(color = "#EAEAEA", linewidth = 0.35),
    axis.text.x        = element_text(angle = 40, hjust = 1, vjust = 1),
    axis.title.y       = element_text(size = 12),
    plot.title         = element_text(face = "bold", size = 13),
    legend.position    = "bottom",
    legend.title       = element_text(size = 11),
    legend.text        = element_text(size = 10)
  ) -> p
ggsave(p, filename = 'which_better/Death_barplot.pdf', width = 10, height = 8)

ggplot(plot_long, aes(x = winner, y = class, fill = n)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = n), size = 3.3, fontface = "bold", color = "black") +
  scale_fill_gradientn(
    colours = c("#F7FCFD", "#CCECE6", "#66C2A4", "#238B45"),
    limits  = c(0, max(plot_long$n, na.rm = TRUE)),
    breaks  = 0:max(plot_long$n, na.rm = TRUE),
    name    = "Cities"
  ) +
  scale_x_discrete(drop = FALSE) +
  scale_y_discrete(labels = function(x) ifelse(x %in% names(class_labs), class_labs[x], x)) +
  labs(
    x     = "Winner category",
    y     = NULL,
    title = "Death: heatmap of winner categories across outcomes"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid  = element_blank(),
    axis.text.x = element_text(angle = 0, hjust = 1, vjust = 1),
    plot.title  = element_text(face = "bold", size = 13),
    legend.position = "right",
    legend.title    = element_text(size = 11),
    legend.text     = element_text(size = 10)
  ) -> p
ggsave(p, filename = 'which_better/Death_heatmap.pdf', width = 10, height = 8)

## =============================================================================
## Section 7: Aggregate and export — Emergency Department Visits
## =============================================================================

df %>%
  as_tibble() %>%
  dplyr::filter(type == 'Emergency department visit') %>%
  group_by(class) %>%
  summarise(city_num = n()) -> city_num

df %>%
  as_tibble() %>%
  dplyr::filter(type == 'Emergency department visit') %>%
  group_by(type, class, winner) %>%
  summarise(winner_num = n()) %>%
  ungroup() %>%
  mutate(class = factor(class, levels = c('all-cause', 'age0_69', 'age70_', 'female', 'male', "I", "J"))) -> count

plot_dat <- expand.grid(
  type   = 'Emergency department visit',
  class  = c('all-cause', 'age0_69', 'age70_', 'female', 'male', "I", "J"),
  winner = c('P-P', 'P-S', 'P-T', 'S-P', 'S-S', 'S-T', 'T-P', 'T-S', 'T-T')
) %>%
  left_join(count, by = c('type', 'class', 'winner')) %>%
  as_tibble() %>%
  left_join(city_num, by = 'class') %>%
  mutate(winner_num = ifelse(is.na(winner_num), 0, winner_num)) %>%
  pivot_wider(names_from = winner, values_from = winner_num) %>%
  mutate(class = factor(class, levels = c('all-cause', 'age0_69', 'age70_', 'female', 'male', "I", "J"))) %>%
  arrange(type, class) %>%
  relocate(`P-P`, `P-S`, `P-T`, `S-P`, `S-S`, `S-T`, `T-P`, `T-S`, `T-T`, .after = city_num)

writexl::write_xlsx(plot_dat, path = 'which_better/ED.xlsx')

plot_long <- plot_dat %>%
  pivot_longer(
    cols      = `P-P`:`T-T`,
    names_to  = "winner",
    values_to = "n"
  ) %>%
  mutate(
    winner = factor(winner,
      levels = c("P-P", "P-S", "P-T", "S-P", "S-S", "S-T", "T-P", "T-S", "T-T")),
    class = factor(class,
      levels = rev(c("all-cause", "age0_69", "age70_", "female", "male", "I", "J")))
  )

# Color scheme: Perception = blue, Similar = grey, Temperature = red
winner_cols <- c(
  "P-P" = "#2166AC",  # deep blue
  "P-S" = "#67A9CF",  # blue
  "P-T" = "#B2ABD2",  # blue-purple
  "S-P" = "#92C5DE",  # light blue
  "S-S" = "#D9D9D9",  # neutral grey
  "S-T" = "#F4A582",  # light orange
  "T-P" = "#D8B365",  # tan
  "T-S" = "#EF8A62",  # orange-red
  "T-T" = "#B2182B"   # deep red
)

class_labs <- c(
  "all-cause" = "All-cause",
  "age0_69"   = "Age 0-69",
  "age70_"    = "Age 70+",
  "female"    = "Female",
  "male"      = "Male"
)

ggplot(plot_long %>%
         mutate(class = factor(class, levels = c('all-cause', 'age0_69', 'age70_', 'female', 'male', "I", "J"))),
       aes(x = class, y = n, fill = winner)) +
  geom_col(position = "fill", width = 0.78, color = "white", linewidth = 0.25) +
  scale_fill_manual(values = winner_cols, drop = FALSE) +
  scale_y_continuous(
    labels = percent_format(accuracy = 1),
    expand = expansion(mult = c(0, 0.02))
  ) +
  guides(fill = guide_legend(nrow = 1, byrow = TRUE)) +
  scale_x_discrete(labels = function(x) ifelse(x %in% names(class_labs), class_labs[x], x)) +
  labs(
    x     = NULL,
    y     = "Proportion of cities",
    fill  = "Model comparison",
    title = "Emergency department visit: proportion of winner categories across outcomes"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.minor   = element_blank(),
    panel.grid.major.y = element_line(color = "#EAEAEA", linewidth = 0.35),
    axis.text.x        = element_text(angle = 0, hjust = 1, vjust = 1),
    axis.title.y       = element_text(size = 12),
    plot.title         = element_text(face = "bold", size = 13),
    legend.position    = "bottom",
    legend.title       = element_text(size = 11),
    legend.text        = element_text(size = 10)
  ) -> p
ggsave(p, filename = 'which_better/ED_barplot.pdf', width = 10, height = 8)

ggplot(plot_long, aes(x = winner, y = class, fill = n)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = n), size = 3.3, fontface = "bold", color = "black") +
  scale_fill_gradientn(
    colours = c("#F7FCFD", "#CCECE6", "#66C2A4", "#238B45"),
    limits  = c(0, max(plot_long$n, na.rm = TRUE)),
    breaks  = 0:max(plot_long$n, na.rm = TRUE),
    name    = "Cities"
  ) +
  scale_x_discrete(drop = FALSE) +
  scale_y_discrete(labels = function(x) ifelse(x %in% names(class_labs), class_labs[x], x)) +
  labs(
    x     = "Winner category",
    y     = NULL,
    title = "Emergency department visit: heatmap of winner categories across outcomes"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid  = element_blank(),
    axis.text.x = element_text(angle = 0, hjust = 1, vjust = 1),
    plot.title  = element_text(face = "bold", size = 13),
    legend.position = "right",
    legend.title    = element_text(size = 11),
    legend.text     = element_text(size = 10)
  ) -> p
ggsave(p, filename = 'which_better/ED_heatmap.pdf', width = 10, height = 8)

## =============================================================================
## Session information
## =============================================================================
sessionInfo()
