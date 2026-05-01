################################################################################
# GUARDIAN BUSINESS JOURNALISM — LINGUISTIC CONVERGENCE ANALYSIS
# The Role of LLMs in Linguistic convergence of Guardian
# Business Journalism: A Multilevel Analysis, 2019-2025
################################################################################

library(dplyr)
library(ggplot2)
library(tidyr)
library(lme4)
library(lmerTest)
library(tidyverse)

# lmerTest must be loaded after lme4 to override lmer() with
# Satterthwaite p-values for fixed effects

################################################################################
# 1. LOAD DATA
################################################################################

df <- read.csv("post_processing_business_dm_rate_in.csv")

cat("Raw data:\n")
cat("  Articles:", nrow(df), "\n")
cat("  Authors:", n_distinct(df$author), "\n")

authors_both_periods <- df %>%
  group_by(author) %>%
  summarise(
    n_pre  = sum(post == 0),
    n_post = sum(post == 1),
    .groups = "drop"
  ) %>%
  filter(n_pre > 0 & n_post > 0) %>%
  pull(author)

df <- df %>% filter(author %in% authors_both_periods)

cat("Authors after filtering:", n_distinct(df$author), "\n")
cat("Articles after filtering:", nrow(df), "\n")

# Rename implicative to inferential
df$dm_inferential_rate <- df$dm_implicative_rate

# Raw outcomes vector — used for descriptive statistics only
outcomes_raw <- c(
  "burstiness",
  "avg_sentence_length",
  "dm_contrastive_rate",
  "dm_elaborative_rate",
  "dm_inferential_rate",
  "hedge_rate",
  "mattr_50"
)

################################################################################
# 2. DESCRIPTIVE STATISTICS
################################################################################

################################################################################
# 2.1 Article-Level Descriptive Statistics
################################################################################

desc_article <- data.frame(
  outcome  = outcomes_raw,
  n        = NA,
  mean     = NA,
  sd       = NA,
  median   = NA,
  iqr      = NA,
  min      = NA,
  max      = NA,
  skewness = NA
)

for (i in seq_along(outcomes_raw)) {
  v <- df[[outcomes_raw[i]]]
  desc_article$n[i]        <- sum(!is.na(v))
  desc_article$mean[i]     <- round(mean(v, na.rm = TRUE), 4)
  desc_article$sd[i]       <- round(sd(v, na.rm = TRUE), 4)
  desc_article$median[i]   <- round(median(v, na.rm = TRUE), 4)
  desc_article$iqr[i]      <- round(IQR(v, na.rm = TRUE), 4)
  desc_article$min[i]      <- round(min(v, na.rm = TRUE), 4)
  desc_article$max[i]      <- round(max(v, na.rm = TRUE), 4)
  desc_article$skewness[i] <- round(
    mean((v - mean(v, na.rm = TRUE))^3, na.rm = TRUE) / sd(v, na.rm = TRUE)^3,
    4)
}

print(desc_article)

################################################################################
# 2.2 Pre vs Post Descriptive Statistics
################################################################################

desc_prepost <- data.frame(
  outcome = rep(outcomes_raw, each = 2),
  period  = rep(c("Pre-ChatGPT", "Post-ChatGPT"), times = length(outcomes_raw)),
  n       = NA,
  mean    = NA,
  sd      = NA,
  median  = NA,
  iqr     = NA
)

for (i in seq_along(outcomes_raw)) {
  for (j in 0:1) {
    row <- (i - 1) * 2 + j + 1
    v   <- df[[outcomes_raw[i]]][df$post == j]
    desc_prepost$n[row]      <- length(v)
    desc_prepost$mean[row]   <- round(mean(v, na.rm = TRUE), 4)
    desc_prepost$sd[row]     <- round(sd(v, na.rm = TRUE), 4)
    desc_prepost$median[row] <- round(median(v, na.rm = TRUE), 4)
    desc_prepost$iqr[row]    <- round(IQR(v, na.rm = TRUE), 4)
  }
}

print(desc_prepost)

################################################################################
# 2.3 Author-Level Descriptive Statistics
################################################################################

articles_per_author <- df %>%
  group_by(author) %>%
  summarise(
    n_articles = n(),
    n_pre      = sum(post == 0),
    n_post     = sum(post == 1),
    .groups = "drop"
  ) %>%
  arrange(desc(n_articles))

cat("\nArticles per author (top 10):\n")
print(head(articles_per_author, 10))

cat("\nSummary of articles per author:\n")
cat("  Min:",    min(articles_per_author$n_articles), "\n")
cat("  Max:",    max(articles_per_author$n_articles), "\n")
cat("  Mean:",   round(mean(articles_per_author$n_articles), 1), "\n")
cat("  Median:", median(articles_per_author$n_articles), "\n")

################################################################################
# 2.4 Between-Author Variance Pre vs Post (Descriptive)
################################################################################

variance_desc <- data.frame(
  outcome         = outcomes_raw,
  sd_authors_pre  = NA,
  sd_authors_post = NA,
  pct_change      = NA
)

for (i in seq_along(outcomes_raw)) {
  pre_means <- df %>%
    filter(post == 0) %>%
    group_by(author) %>%
    summarise(m = mean(.data[[outcomes_raw[i]]], na.rm = TRUE), .groups = "drop") %>%
    pull(m)
  
  post_means <- df %>%
    filter(post == 1) %>%
    group_by(author) %>%
    summarise(m = mean(.data[[outcomes_raw[i]]], na.rm = TRUE), .groups = "drop") %>%
    pull(m)
  
  sd_pre  <- sd(pre_means)
  sd_post <- sd(post_means)
  
  variance_desc$sd_authors_pre[i]  <- round(sd_pre, 4)
  variance_desc$sd_authors_post[i] <- round(sd_post, 4)
  variance_desc$pct_change[i]      <- round((sd_post - sd_pre) / sd_pre * 100, 2)
}

print(variance_desc)
cat("\nNote: Negative pct_change indicates reduced between-author spread post-ChatGPT.\n")
cat("This is a descriptive signal only — formal testing follows in the MLM stages.\n")

################################################################################
# 2.5 Predictor Descriptive Statistics
################################################################################

cat("post (binary — 0 = Pre-ChatGPT, 1 = Post-ChatGPT):\n")
cat(sprintf("  Pre-ChatGPT  (post=0): %d articles (%.1f%%)\n",
            sum(df$post == 0), mean(df$post == 0) * 100))
cat(sprintf("  Post-ChatGPT (post=1): %d articles (%.1f%%)\n",
            sum(df$post == 1), mean(df$post == 1) * 100))

cat("\nlog_word_count (control variable, original scale):\n")
v <- df$log_word_count
cat(sprintf("  Mean: %.4f | SD: %.4f | Min: %.4f | Max: %.4f\n",
            mean(v), sd(v), min(v), max(v)))
cat("Note: log_word_count is standardised (z-scored) as log_wc_scaled before entering models.\n")

################################################################################
# 3. DATA TRANSFORMATION
################################################################################

################################################################################
# 3.1 Log Transformation
#
# Inferential DM rate is log-transformed prior to standardisation to address
# zero-inflation: approximately 50% of articles record zero inferential markers,
# violating the normality assumption. A log(x + 1) transformation is applied
# to retain zero-valued observations.
################################################################################

df$dm_inferential_rate_log <- log(df$dm_inferential_rate + 1)

################################################################################
# 3.2 Group-Mean Centring of log_word_count
#
# Purpose:
#   - Assess whether log_word_count varies substantially between authors
#   - ICC > 10% warrants group-mean centring to separate
#     within-author and between-author effects of article length
#   - Justifies decomposition into log_wc_within and log_wc_mean
################################################################################

icc_wc     <- lmer(log_word_count ~ 1 + (1 | author), data = df, REML = FALSE)
vc_wc      <- as.data.frame(VarCorr(icc_wc), comp = "Variance")
icc_wc_val <- vc_wc$vcov[1] / (vc_wc$vcov[1] + vc_wc$vcov[2])
cat("ICC for log_word_count:", round(icc_wc_val * 100, 1), "%\n")
cat("Substantial between-author variation detected — group-mean centring applied.\n")

df <- df %>%
  group_by(author) %>%
  mutate(
    log_wc_mean   = mean(log_word_count),
    log_wc_within = log_word_count - log_wc_mean
  ) %>%
  ungroup()

################################################################################
# 3.3 Standardisation
#
# All outcome variables are standardised (z-scored) prior to estimation
# to ensure comparability of coefficients across models.
# Raw units are retained in outcomes_raw for descriptive statistics only.
# The log-transformed inferential DM rate is standardised in place of the raw.
################################################################################

df$burstiness_z          <- scale(df$burstiness)
df$avg_sentence_length_z <- scale(df$avg_sentence_length)
df$dm_contrastive_rate_z <- scale(df$dm_contrastive_rate)
df$dm_elaborative_rate_z <- scale(df$dm_elaborative_rate)
df$dm_inferential_rate_z <- scale(df$dm_inferential_rate_log)
df$hedge_rate_z          <- scale(df$hedge_rate)
df$mattr_z               <- scale(df$mattr_50)

df$log_wc_within_scaled <- scale(df$log_wc_within)
df$log_wc_mean_scaled   <- scale(df$log_wc_mean)

# Outcomes vector (z-scored) — used in all model stages
outcomes <- c(
  "burstiness_z",
  "avg_sentence_length_z",
  "dm_contrastive_rate_z",
  "dm_elaborative_rate_z",
  "dm_inferential_rate_z",
  "hedge_rate_z",
  "mattr_z"
)

################################################################################
# 4. MODEL BUILDING
################################################################################

################################################################################
# 4.1 Stage 1 — Null Model
# Model: outcome ~ 1 + (1 | author)
# Purpose:
#   - Partition variance into between-author and within-author
#   - Calculate ICC to justify use of MLM
#   - Establish baseline variance components before predictors
################################################################################

null_results <- data.frame(
  outcome = outcomes,
  sig2u0  = NA,
  sig2e   = NA,
  icc     = NA,
  loglik  = NA
)

models_null <- list()

for (i in seq_along(outcomes)) {
  
  outcome <- outcomes[i]
  cat("\n------------------------------------------------------------\n")
  cat("Outcome:", outcome, "\n")
  cat("------------------------------------------------------------\n")
  
  formula_null <- as.formula(paste(outcome, "~ 1 + (1 | author)"))
  m0 <- lmer(formula_null, data = df, REML = FALSE)
  
  vc     <- as.data.frame(VarCorr(m0), comp = "Variance")
  sig2u0 <- vc$vcov[1]
  sig2e  <- vc$vcov[2]
  icc    <- sig2u0 / (sig2u0 + sig2e)
  
  cat("Between-author variance (sig2u0):", round(sig2u0, 4), "\n")
  cat("Within-author variance  (sig2e): ", round(sig2e,  4), "\n")
  cat("ICC:                             ", round(icc,    4), "\n")
  cat("Interpretation:", round(icc * 100, 1),
      "% of total variance is attributable to author-level differences\n")
  
  null_results$sig2u0[i] <- round(sig2u0, 4)
  null_results$sig2e[i]  <- round(sig2e,  4)
  null_results$icc[i]    <- round(icc,    4)
  null_results$loglik[i] <- round(as.numeric(logLik(m0)), 4)
  
  models_null[[outcome]] <- m0
}

print(null_results)
cat("\nICC > 0.05 suggests meaningful between-author clustering and justifies MLM.\n")

################################################################################
# 4.1b CATERPILLAR PLOTS — RANDOM INTERCEPTS (NULL MODEL)
#
# One plot per outcome variable showing:
#   - Each author's estimated random intercept (u0) from null model
#   - 95% confidence intervals around each estimate
#   - Authors ranked from lowest to highest intercept
#   - Horizontal line at zero (grand mean)
#   - Red = authors statistically distinguishable from average
#   - Directly visualises between-author variation justifying MLM
################################################################################

outcome_labels <- c(
  "burstiness_z"          = "Burstiness",
  "avg_sentence_length_z" = "Avg Sentence Length",
  "dm_contrastive_rate_z" = "Contrastive DM Rate",
  "dm_elaborative_rate_z" = "Elaborative DM Rate",
  "dm_inferential_rate_z" = "inferential DM Rate",
  "hedge_rate_z"          = "Hedge Rate",
  "mattr_z"               = "MATTR-50 (Lexical Diversity)"
)

for (outcome in outcomes) {
  
  cat("\nGenerating caterpillar plot for:", outcome, "\n")
  
  re      <- ranef(models_null[[outcome]], condVar = TRUE)$author
  condvar <- attr(ranef(models_null[[outcome]], condVar = TRUE)$author,
                  "postVar")[1, 1, ]
  
  plot_df <- data.frame(
    author    = rownames(re),
    intercept = re[, 1],
    se        = sqrt(condvar)
  ) %>%
    mutate(
      ci_lower = intercept - 1.96 * se,
      ci_upper = intercept + 1.96 * se,
      sig      = ifelse(ci_lower > 0 | ci_upper < 0, "Significant", "Not significant")
    ) %>%
    arrange(intercept) %>%
    mutate(rank = row_number())
  
  p <- ggplot(plot_df, aes(x = rank, y = intercept, colour = sig)) +
    geom_hline(yintercept = 0, linetype = "dashed", colour = "grey50", linewidth = 0.6) +
    geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper),
                  width = 0.3, linewidth = 0.5, alpha = 0.7) +
    geom_point(size = 2) +
    scale_colour_manual(
      values = c("Significant" = "#d7191c", "Not significant" = "#2c7bb6"),
      name   = "95% CI crosses zero"
    ) +
    scale_x_continuous(breaks = plot_df$rank, labels = plot_df$author) +
    labs(
      title    = paste("Random Intercepts —", outcome_labels[outcome]),
      subtitle = "Authors ranked by estimated intercept | Null model | Error bars = 95% CI",
      x        = "Author (ranked)",
      y        = "Estimated Random Intercept (SD units)"
    ) +
    theme_minimal(base_size = 10) +
    theme(
      plot.title       = element_text(face = "bold", size = 12),
      plot.subtitle    = element_text(size = 8, colour = "grey40"),
      axis.text.x      = element_text(angle = 90, hjust = 1, vjust = 0.5, size = 7),
      panel.grid.minor = element_blank(),
      legend.position  = "bottom"
    )
  
  print(p)
}

################################################################################
# 4.2 Stage 2 — Fixed Effects Model
# Model: outcome ~ post + log_wc_within_scaled + log_wc_mean_scaled + (1 | author)
# Purpose:
#   - Test whether post-ChatGPT period is associated with a
#     shift in the mean of each linguistic feature
#   - LRT vs null: do fixed effects improve fit?
#   - R1_squared: within-author variance explained
#   - R2_squared: between-author variance explained
################################################################################

stage2_results <- data.frame(
  outcome        = outcomes,
  beta_post      = NA,
  se_post        = NA,
  tval_post      = NA,
  pval_post      = NA,
  sig2u0         = NA,
  sig2e          = NA,
  R1_squared     = NA,
  R2_squared     = NA,
  lrt_null_chisq = NA,
  lrt_null_pval  = NA
)

models_stage2 <- list()

for (i in seq_along(outcomes)) {
  
  outcome <- outcomes[i]
  cat("\n------------------------------------------------------------\n")
  cat("Outcome:", outcome, "\n")
  cat("------------------------------------------------------------\n")
  
  m0 <- models_null[[outcome]]
  
  formula_s2 <- as.formula(paste(outcome, "~ post + log_wc_within_scaled + log_wc_mean_scaled + (1 | author)"))
  m1 <- lmerTest::lmer(formula_s2, data = df, REML = FALSE)
  
  coefs <- summary(m1)$coefficients
  cat("\nFixed effects:\n")
  print(round(coefs, 4))
  
  vc0 <- as.data.frame(VarCorr(m0), comp = "Variance")
  vc1 <- as.data.frame(VarCorr(m1), comp = "Variance")
  
  sig2u0_m0 <- vc0$vcov[1]
  sig2e_m0  <- vc0$vcov[2]
  sig2u0_m1 <- vc1$vcov[1]
  sig2e_m1  <- vc1$vcov[2]
  
  R1_2 <- (sig2e_m0  - sig2e_m1)  / sig2e_m0
  R2_2 <- (sig2u0_m0 - sig2u0_m1) / sig2u0_m0
  
  cat("\nVariance components:\n")
  cat("  sig2u0 (between-author):", round(sig2u0_m1, 4), "\n")
  cat("  sig2e  (within-author): ", round(sig2e_m1,  4), "\n")
  cat("  R1_squared:", round(R1_2, 4), "\n")
  cat("  R2_squared:", round(R2_2, 4), "\n")
  
  lrt_null <- anova(m1, m0)
  cat("\nLRT: Stage 2 vs Null (H0: beta1 = beta2w = beta2c = 0):\n")
  print(lrt_null)
  
  pval_col <- if ("Pr(>|t|)" %in% colnames(coefs)) "Pr(>|t|)" else NA
  
  stage2_results$beta_post[i]      <- round(coefs["post", "Estimate"], 4)
  stage2_results$se_post[i]        <- round(coefs["post", "Std. Error"], 4)
  stage2_results$tval_post[i]      <- round(coefs["post", "t value"], 4)
  stage2_results$pval_post[i]      <- if (!is.na(pval_col)) round(coefs["post", pval_col], 6) else NA
  stage2_results$sig2u0[i]         <- round(sig2u0_m1, 4)
  stage2_results$sig2e[i]          <- round(sig2e_m1,  4)
  stage2_results$R1_squared[i]     <- round(R1_2, 4)
  stage2_results$R2_squared[i]     <- round(R2_2, 4)
  stage2_results$lrt_null_chisq[i] <- round(lrt_null$Chisq[2], 4)
  stage2_results$lrt_null_pval[i]  <- round(lrt_null$`Pr(>Chisq)`[2], 6)
  
  models_stage2[[outcome]] <- m1
}

print(stage2_results)

cat("\nSignificant post coefficients (p < 0.05):\n")
sig <- stage2_results[!is.na(stage2_results$pval_post) & stage2_results$pval_post < 0.05,
                      c("outcome", "beta_post", "pval_post")]
if (nrow(sig) == 0) cat("  None.\n") else print(sig)

################################################################################
# 4.2b Stage 2b — Between-Author Variance Diagnostics
################################################################################

stage2b_results <- data.frame(
  outcome           = outcomes,
  sig2u0_null       = NA,
  sig2u0_stage2     = NA,
  sig2e_null        = NA,
  sig2e_stage2      = NA,
  icc_null          = NA,
  icc_stage2        = NA,
  R2_squared        = NA,
  R1_squared        = NA,
  direction         = NA
)

for (i in seq_along(outcomes)) {
  
  outcome <- outcomes[i]
  cat("\n------------------------------------------------------------\n")
  cat("Outcome:", outcome, "\n")
  cat("------------------------------------------------------------\n")
  
  vc_null <- as.data.frame(VarCorr(models_null[[outcome]]),   comp = "Variance")
  vc_s2   <- as.data.frame(VarCorr(models_stage2[[outcome]]), comp = "Variance")
  
  sig2u0_null <- vc_null$vcov[1]
  sig2e_null  <- vc_null$vcov[2]
  sig2u0_s2   <- vc_s2$vcov[1]
  sig2e_s2    <- vc_s2$vcov[2]
  
  icc_null <- sig2u0_null / (sig2u0_null + sig2e_null)
  icc_s2   <- sig2u0_s2  / (sig2u0_s2  + sig2e_s2)
  
  R2 <- (sig2u0_null - sig2u0_s2) / sig2u0_null
  R1 <- (sig2e_null  - sig2e_s2)  / sig2e_null
  
  direction <- ifelse(sig2u0_s2 < sig2u0_null, "CONVERGENCE", "DIVERGENCE")
  
  cat("Stage 1 Null:\n")
  cat("  sig2u0:", round(sig2u0_null, 4),
      "| sig2e:", round(sig2e_null, 4),
      "| ICC:",   round(icc_null,   4), "\n")
  
  cat("Stage 2 Fixed Effects:\n")
  cat("  sig2u0:", round(sig2u0_s2, 4),
      "| sig2e:", round(sig2e_s2,   4),
      "| ICC:",   round(icc_s2,     4), "\n")
  
  cat("  R1_squared (within-author variance explained): ", round(R1, 4), "\n")
  cat("  R2_squared (between-author variance explained):", round(R2, 4), "\n")
  cat("  Direction:", direction, "\n")
  
  stage2b_results$sig2u0_null[i]   <- round(sig2u0_null, 4)
  stage2b_results$sig2u0_stage2[i] <- round(sig2u0_s2,   4)
  stage2b_results$sig2e_null[i]    <- round(sig2e_null,   4)
  stage2b_results$sig2e_stage2[i]  <- round(sig2e_s2,     4)
  stage2b_results$icc_null[i]      <- round(icc_null,     4)
  stage2b_results$icc_stage2[i]    <- round(icc_s2,       4)
  stage2b_results$R2_squared[i]    <- round(R2,            4)
  stage2b_results$R1_squared[i]    <- round(R1,            4)
  stage2b_results$direction[i]     <- direction
}

print(stage2b_results)

cat("\nNote: Positive R2_squared = sig2u0 decreased from null to Stage 2.\n")
cat("This is the direct test of H1: between-author variance shrinks\n")
cat("when the post-ChatGPT period is accounted for.\n")
cat("Negative R2_squared = between-author variance increased (divergence).\n")

################################################################################
# 42c. NORMALITY DIAGNOSTICS 
################################################################################

for (outcome in outcomes) {
  m <- models_stage2[[outcome]]
  
  # QQ plot of residuals
  qqnorm(resid(m), main = paste("QQ Plot - Residuals:", outcome))
  qqline(resid(m), col = "red")
  
  # QQ plot of random effects
  qqnorm(ranef(m)$author[,1], main = paste("QQ Plot - Random Effects:", outcome))
  qqline(ranef(m)$author[,1], col = "red")
}

################################################################################
# 4.3 Stage 3 — Random Slope Model
# Model: outcome ~ post + log_wc_within_scaled + log_wc_mean_scaled + (1 + post | author)
# Purpose:
#   - Test whether authors differ in their response to post-ChatGPT
#   - sig2u1 = between-author variance in the post slope
#   - LRT vs Stage 2: is sig2u1 > 0?
#   - cor(u0, u1): negative = convergence pattern (high baseline
#     authors decreased more post-ChatGPT)
################################################################################

stage3_results <- data.frame(
  outcome   = outcomes,
  sig2u0    = NA,
  sig2u1    = NA,
  cor_u0u1  = NA,
  sig2e     = NA,
  pvr_lower = NA,
  pvr_upper = NA,
  lrt_chisq = NA,
  lrt_pval  = NA,
  converged = NA
)

models_stage3 <- list()

for (i in seq_along(outcomes)) {
  
  outcome <- outcomes[i]
  cat("\n------------------------------------------------------------\n")
  cat("Outcome:", outcome, "\n")
  cat("------------------------------------------------------------\n")
  
  m1 <- models_stage2[[outcome]]
  
  formula_s3 <- as.formula(paste(outcome,
                                 "~ post + log_wc_within_scaled + log_wc_mean_scaled + (1 + post | author)"))
  
  converged <- TRUE
  m2 <- tryCatch({
    lmerTest::lmer(formula_s3, data = df, REML = FALSE)
  }, warning = function(w) {
    cat("Convergence warning — switching to Nelder_Mead\n")
    converged <<- FALSE
    lmerTest::lmer(formula_s3, data = df, REML = FALSE,
                   control = lmerControl(optimizer = "Nelder_Mead"))
  })
  
  vc <- as.data.frame(VarCorr(m2), comp = "Variance")
  print(vc)
  
  sig2u0  <- vc$vcov[1]
  sig2u1  <- vc$vcov[2]
  sig2e   <- vc$vcov[4]
  cor_val <- attr(VarCorr(m2)$author, "correlation")[1, 2]
  
  cat("\nVariance components:\n")
  cat("  sig2u0 (between-author intercept variance):", round(sig2u0,  4), "\n")
  cat("  sig2u1 (between-author slope variance):    ", round(sig2u1,  4), "\n")
  cat("  cor(u0, u1):                               ", round(cor_val, 4), "\n")
  cat("  sig2e  (residual variance):                ", round(sig2e,   4), "\n")
  
  beta_post <- fixef(m1)["post"]
  
  pvr_lower <- beta_post - 1.96 * sqrt(sig2u1)
  pvr_upper <- beta_post + 1.96 * sqrt(sig2u1)
  
  cat("  PVR: [", round(pvr_lower, 3), ",", round(pvr_upper, 3), "]\n")
  
  lrt <- anova(m2, m1)
  cat("\nLRT: Stage 3 vs Stage 2 (H0: sig2u1 = 0):\n")
  print(lrt)
  
  stage3_results$sig2u0[i]    <- round(sig2u0,  4)
  stage3_results$sig2u1[i]    <- round(sig2u1,  4)
  stage3_results$cor_u0u1[i]  <- round(cor_val, 4)
  stage3_results$sig2e[i]     <- round(sig2e,   4)
  stage3_results$pvr_lower[i] <- round(pvr_lower, 3)
  stage3_results$pvr_upper[i] <- round(pvr_upper, 3)
  stage3_results$lrt_chisq[i] <- round(lrt$Chisq[2], 4)
  stage3_results$lrt_pval[i]  <- round(lrt$`Pr(>Chisq)`[2], 6)
  stage3_results$converged[i] <- converged
  
  models_stage3[[outcome]] <- m2
}

print(stage3_results)

cat("\nInterpretation guide:\n")
cat("  sig2u1 > 0 and p < 0.05: heterogeneous author responses — random slope justified\n")
cat("  cor(u0, u1) negative: authors with higher baselines decreased more post-ChatGPT\n")
cat("  (convergence pattern)\n")

################################################################################
# 43b. NORMALITY DIAGNOSTICS 
################################################################################
for (outcome in outcomes) {
  m <- models_stage3[[outcome]]
  
  qqnorm(resid(m), main = paste("QQ Plot - Residuals:", outcome, "Stage 3"))
  qqline(resid(m), col = "red")
  
  qqnorm(ranef(m)$author[,1], main = paste("QQ Plot - Random Effects (Intercept):", outcome))
  qqline(ranef(m)$author[,1], col = "red")
  
  qqnorm(ranef(m)$author[,2], main = paste("QQ Plot - Random Effects (Slope):", outcome))
  qqline(ranef(m)$author[,2], col = "red")
}

################################################################################
# CODE FOR DATA DISTRIBUTION PLOTS
################################################################################

# Packages
# install.packages(c("dplyr", "tidyr", "ggplot2", "knitr"))
library(dplyr)
library(tidyr)
library(ggplot2)
library(knitr)

# Outcome variables in raw scale
outcome_vars <- c(
  "avg_sentence_length",
  "burstiness",
  "mattr_50",
  "hedge_rate",
  "dm_contrastive_rate",
  "dm_elaborative_rate",
  "dm_inferential_rate"
)

outcome_labels <- c(
  avg_sentence_length = "Average sentence length",
  burstiness = "Burstiness",
  mattr_50 = "MATTR-50",
  hedge_rate = "Hedge rate",
  dm_contrastive_rate = "Contrastive DM rate",
  dm_elaborative_rate = "Elaborative DM rate",
  dm_inferential_rate = "Inferential DM rate"
)

# DESCRIPTIVE STATISTICS TABLE

desc_table <- df %>%
  summarise(
    across(
      all_of(outcome_vars),
      list(
        Mean = ~mean(.x, na.rm = TRUE),
        SD = ~sd(.x, na.rm = TRUE),
        Median = ~median(.x, na.rm = TRUE),
        Min = ~min(.x, na.rm = TRUE),
        Max = ~max(.x, na.rm = TRUE)
      ),
      .names = "{.col}_{.fn}"
    )
  ) %>%
  pivot_longer(
    everything(),
    names_to = c("Variable", ".value"),
    names_pattern = "(.+)_(Mean|SD|Median|Min|Max)"
  ) %>%
  mutate(
    Variable = recode(Variable, !!!outcome_labels),
    across(c(Mean, SD, Median, Min, Max), ~round(.x, 3))
  )

kable(
  desc_table,
  caption = "Table 3.0: Descriptive Statistics for Linguistic Outcome Variables"
)

# DISTRIBUTIONS OF RAW OUTCOME VARIABLES

df_long_raw <- df %>%
  select(all_of(outcome_vars)) %>%
  pivot_longer(
    cols = everything(),
    names_to = "Variable",
    values_to = "Value"
  ) %>%
  mutate(Variable = recode(Variable, !!!outcome_labels))

df_long_raw$Variable <- factor(
  df_long_raw$Variable,
  levels = c(
    "Average sentence length",
    "Burstiness",
    "MATTR-50",
    "Contrastive DM rate",
    "Elaborative DM rate",
    "Inferential DM rate",
    "Hedge rate"
  )
)

p_dist <- ggplot(df_long_raw, aes(x = Value)) +
  geom_histogram(bins = 40, colour = "white") +
  facet_wrap(~ Variable, scales = "free", ncol = 3) +
  labs(
    title = "Distribution of Linguistic Outcome Variables",
    x = NULL,
    y = "Number of articles"
  ) +
  theme_minimal()

print(p_dist)

# ARTICLE COUNTS BY YEAR AND PERIOD

df_counts_year <- df %>%
  mutate(
    period = ifelse(post == 1, "Post-ChatGPT", "Pre-ChatGPT")
  ) %>%
  count(year, period)

p_counts <- ggplot(df_counts_year, aes(x = year, y = n, fill = period)) +
  geom_col() +
  labs(
    title = "Article Counts by Year",
    x = "Year",
    y = "Number of articles",
    fill = NULL
  ) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "bottom")

print(p_counts)

# NUMBER OF ARTICLES PER AUTHOR AND PERIOD
df_author_period <- df %>%
  mutate(period = ifelse(post == 1, "Post", "Pre")) %>%
  count(author, period)

# Recompute ordering
author_order <- df_author_period %>%
  group_by(author) %>%
  summarise(total = sum(n)) %>%
  arrange(total)

df_author_period <- df_author_period %>%
  mutate(author = factor(author, levels = author_order$author))

# Plot
ggplot(df_author_period, aes(x = author, y = n, fill = period)) +
  geom_col() +
  coord_flip() +
  labs(
    title = "Article Distribution by Author and Period",
    x = "Author",
    y = "Number of articles",
    fill = "Period"
  ) +
  theme_minimal()

# ARTICLE LENGTH DISTRIBUTION (LOG SCALE)
ggplot(df, aes(x = word_count)) +
  geom_histogram(bins = 50) +
  scale_x_log10() +
  labs(
    title = "Distribution of Article Length (Log Scale)",
    x = "Word count (log scale)",
    y = "Frequency"
  ) +
  theme_minimal()