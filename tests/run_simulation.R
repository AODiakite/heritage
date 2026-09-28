#!/usr/bin/env Rscript

## ============================================================ 
## Simulation Script for Scenario 2: Varying Dimensions (p)
## Usage: Rscript run_simulation.R <array_task_id>
## Uses expand.grid for 500 combinations (100 seeds × 5 dimensions)
## ============================================================ 

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 1) {
  stop("Usage: Rscript run_simulation.R <array_task_id>")
}

array_task_id <- as.numeric(args[1])

# Set scenario name
scenario_name <- "varying_dimensions"

cat(sprintf("=== Starting Simulation ===\n"))
cat(sprintf("Scenario: %s\n", scenario_name))
cat(sprintf("Array Task ID: %d\n", array_task_id))
cat(sprintf("Working Directory: %s\n\n", getwd()))

# Load required libraries
library(glmnet)
library(MASS)
library(pedSimulate)
library(kinship2)

# Source HERITAGE only
source("HERITAGE.R")

# ============================================================ 
# Define model wrapper functions
# ============================================================ 

#' Model wrapper for NRF-FC
fit_nrf_fc <- function(data_train, data_tune) {
  t1 = Sys.time()
  gs_result <- enr_fc.gs(
    data_train$y, data_train$X, data_train$sX, data_train$sY,
    data_tune$y, data_tune$X, data_tune$sX, data_tune$sY,
    data_train$y, data_train$X, data_train$sX, data_train$sY,  # Dummy for test
    family = data_train$family,
    spillover_y = TRUE,
    n_lambda_beta = 30,
    n_lambda_delta = 30,
    verbose = TRUE,
    standardize = TRUE
  )
  t2 = Sys.time()
  # Computation tim in minutes
  
  compute_time <- as.numeric(difftime(t2, t1, units = "mins"))
  best_model = gs_result$best_model
  best_model$compute_time <- compute_time
  best_model
}

#' Model wrapper for NRF-FC without gamma
fit_nrf_fc_no_gamma <- function(data_train, data_tune) {
  t1 = Sys.time()
  gs_result <- enr_fc.gs(
    data_train$y, data_train$X, data_train$sX, data_train$sY,
    data_tune$y, data_tune$X, data_tune$sX, data_tune$sY,
    data_train$y, data_train$X, data_train$sX, data_train$sY,
    family = data_train$family,
    spillover_y = FALSE,
    n_lambda_beta = 30,
    n_lambda_delta = 30,
    verbose = TRUE,
    standardize = TRUE
  )
  t2 = Sys.time()
  # Computation time in minutes
  compute_time <- as.numeric(difftime(t2, t1, units = "mins"))
  best_model = gs_result$best_model
  best_model$compute_time <- compute_time
  best_model
}

#' Model wrapper for standard Lasso
fit_lasso <- function(data_train, data_tune) {
  t1 = Sys.time()
  gs_result <- lasso.gs(
    data_train$y, data_train$X,
    data_tune$y, data_tune$X,
    data_train$y, data_train$X,
    family = data_train$family,
    n_lambda = 30,
    verbose = FALSE
  )
  t2 = Sys.time()
  # Computation time in minutes
  compute_time <- as.numeric(difftime(t2, t1, units = "mins"))
  list(
    compute_time = compute_time,
    beta = gs_result$beta,
    delta = NULL,
    gamma = NULL,
    alpha = as.numeric(coef(gs_result$best_model)[1]),
    family = data_train$family
  )
}

#' Model wrapper for Lasso + Family FE
fit_lasso_fe <- function(data_train, data_tune) {
  t1 = Sys.time()
  gs_result <- lasso_fe.gs(
    data_train$y, data_train$X, data_train$family_id,
    data_tune$y, data_tune$X, data_tune$family_id,
    data_train$y, data_train$X, data_train$family_id,
    family = data_train$family,
    n_lambda = 30,
    verbose = FALSE
  )
  t2 = Sys.time()
  # Computation time in minutes
  compute_time <- as.numeric(difftime(t2, t1, units = "mins"))
  list(
    compute_time = compute_time,
    beta = gs_result$beta,
    delta = NULL,
    gamma = NULL,
    alpha = as.numeric(coef(gs_result$best_model)[1]),
    family_effects = gs_result$family_effects,
    family = data_train$family
  )
}

#' Model wrapper for glmmLasso (PGLMM)
fit_glmmlasso <- function(data_train, data_tune) {
  t1 = Sys.time()
  gs_result <- glmmLasso.gs(
    data_train$y, data_train$X, data_train$family_id,
    data_tune$y, data_tune$X, data_tune$family_id,
    data_train$y, data_train$X, data_train$family_id,
    family = data_train$family,
    n_lambda = 30,
    verbose = FALSE
  )
  t2 = Sys.time()
  # Computation time in minutes
  compute_time <- as.numeric(difftime(t2, t1, units = "mins"))
  
  list(
    compute_time = compute_time,
    beta = gs_result$beta,
    delta = NULL,
    gamma = NULL,
    alpha = gs_result$intercept,
    family = data_train$family
  )
}

#' Model wrapper for plmmr (Penalized LMM with Kinship)
fit_plmmr <- function(data_train, data_tune) {
  t1 = Sys.time()
  gs_result <- plmmr.gs(
    data_train$y, data_train$X, data_train$family_id,
    data_tune$y, data_tune$X, data_tune$family_id,
    data_train$y, data_train$X, data_train$family_id,
    K_train = data_train$K_train,
    penalty = "lasso",
    nlambda = 30,
    metric = "mse",
    standardize = FALSE,
    verbose = FALSE
  )
  t2 = Sys.time()
  # Computation time in minutes
  compute_time <- as.numeric(difftime(t2, t1, units = "mins"))
  list(
    compute_time = compute_time,
    beta = gs_result$beta,
    delta = NULL,
    gamma = NULL,
    alpha = gs_result$intercept,
    family = data_train$family
  )
}

#' Compute variable selection metrics
compute_selection_metrics <- function(beta_est, beta_true, causal_indices) {
  active_est <- which(beta_est != 0)
  TP <- sum(active_est %in% causal_indices)
  FP <- sum(!(active_est %in% causal_indices))
  FN <- sum(!(causal_indices %in% active_est))
  TN <- length(beta_true) - length(causal_indices) - FP
  sensitivity <- ifelse(TP + FN > 0, TP / (TP + FN), NA)
  specificity <- ifelse(TN + FP > 0, TN / (TN + FP), NA)
  precision <- ifelse(TP + FP > 0, TP / (TP + FP), NA)
  f1_score <- ifelse(!is.na(precision) && !is.na(sensitivity) && precision + sensitivity > 0,
                     2 * precision * sensitivity / (precision + sensitivity), 0)
  list(TP = TP, FP = FP, FN = FN, TN = TN, sensitivity = sensitivity,
       specificity = specificity, precision = precision, f1_score = f1_score,
       n_selected = length(active_est))
}

#' Compute estimation error metrics
compute_estimation_errors <- function(beta_est, beta_true) {
  error <- beta_est - beta_true
  list(
    l_inf = max(abs(error)),
    l2 = sqrt(sum(error^2)),
    l1 = sum(abs(error)),
    rmse = sqrt(mean(error^2)),
    mae = mean(abs(error)),
    relative_l_inf = max(abs(error)) / max(abs(beta_true))
  )
}

#' Helper to standardize data for prediction
standardize_for_prediction <- function(data_target, data_source) {
  data_scaled <- data_target
  
  # X
  X_center <- colMeans(data_source$X)
  X_scale <- apply(data_source$X, 2, sd)
  X_scale[X_scale < 1e-10] <- 1
  data_scaled$X <- sweep(data_target$X, 2, X_center, "-")
  data_scaled$X <- sweep(data_scaled$X, 2, X_scale, "/")
  
  # sX
  sX_center <- colMeans(data_source$sX)
  sX_scale <- apply(data_source$sX, 2, sd)
  sX_scale[sX_scale < 1e-10] <- 1
  data_scaled$sX <- sweep(data_target$sX, 2, sX_center, "-")
  data_scaled$sX <- sweep(data_scaled$sX, 2, sX_scale, "/")
  
  # sY
  sY_center <- mean(data_source$sY)
  sY_scale <- sd(data_source$sY)
  if (sY_scale < 1e-10) sY_scale <- 1
  data_scaled$sY <- (data_target$sY - sY_center) / sY_scale
  
  return(data_scaled)
}

#' Evaluate a single model
evaluate_model <- function(model_name, fit_func, data_train, data_tune, data_test,
                          scenario_name, family_type) {
  tryCatch({
    fit <- fit_func(data_train, data_tune)
    if (is.null(fit)) return(NULL)

    # Predictions on test set
    if ("enr_fc" %in% class(fit)) {
      y_pred <- predict(fit, newX = data_test$X, newsX = data_test$sX,
                       newsY = data_test$sY, type = "response")
    } else {
      eta <- fit$alpha + data_test$X %*% fit$beta
      y_pred <- switch(family_type,
        "gaussian" = eta,
        "binomial" = 1 / (1 + exp(-eta)),
        "poisson" = exp(eta)
      )
      y_pred <- as.vector(y_pred)
    }

    # Prediction metrics
    pred_metrics <- .compute_metrics(data_test$y, y_pred, family_type)

    # Selection metrics for beta
    if (!is.null(fit$beta)) {
      beta_metrics <- compute_selection_metrics(fit$beta, data_train$beta_true,
                                               data_train$causal_beta)
      beta_errors <- compute_estimation_errors(fit$beta, data_train$beta_true)
    } else {
      beta_metrics <- list(sensitivity = NA, specificity = NA,
                          precision = NA, f1_score = NA, n_selected = NA)
      beta_errors <- list(l_inf = NA, l2 = NA, l1 = NA, rmse = NA, mae = NA, relative_l_inf = NA)
    }

    # Selection metrics for delta
    delta_metrics <- list(sensitivity = NA, specificity = NA,
                         precision = NA, f1_score = NA, n_selected = NA)
    delta_errors <- list(l_inf = NA, l2 = NA, l1 = NA, rmse = NA, mae = NA, relative_l_inf = NA)
    if (!is.null(fit$delta)) {
      delta_metrics <- compute_selection_metrics(fit$delta, data_train$delta_true,
                                                 data_train$causal_delta)
      delta_errors <- compute_estimation_errors(fit$delta, data_train$delta_true)
    }

    # Gamma estimation error
    gamma_error <- NA
    if (!is.null(fit$gamma) && !is.null(data_train$gamma_true)) {
      gamma_error <- abs(fit$gamma - data_train$gamma_true)
    }

    # Get metric names
    metric_names <- names(pred_metrics)

    data.frame(
      model = model_name,
      scenario = scenario_name,
      family = family_type,
      compute_time = fit$compute_time,
      test_mse = ifelse("mse" %in% metric_names, pred_metrics$mse, NA),
      test_rmse = ifelse("rmse" %in% metric_names, pred_metrics$rmse, NA),
      test_mae = ifelse("mae" %in% metric_names, pred_metrics$mae, NA),
      test_r2 = ifelse("r2" %in% metric_names, pred_metrics$r2, NA),
      test_accuracy = ifelse("accuracy" %in% metric_names, pred_metrics$accuracy, NA),
      test_logloss = ifelse("logloss" %in% metric_names, pred_metrics$logloss, NA),
      test_precision = ifelse("precision" %in% metric_names, pred_metrics$precision, NA),
      test_recall = ifelse("recall" %in% metric_names, pred_metrics$recall, NA),
      test_f1 = ifelse("f1" %in% metric_names, pred_metrics$f1, NA),
      test_deviance = ifelse("deviance" %in% metric_names, pred_metrics$deviance, NA),
      test_mean_deviance = ifelse("mean_deviance" %in% metric_names, pred_metrics$mean_deviance, NA),
      beta_sensitivity = beta_metrics$sensitivity,
      beta_specificity = beta_metrics$specificity,
      beta_precision = beta_metrics$precision,
      beta_f1 = beta_metrics$f1_score,
      beta_n_selected = beta_metrics$n_selected,
      beta_l_inf = beta_errors$l_inf,
      beta_l2 = beta_errors$l2,
      beta_l1 = beta_errors$l1,
      beta_rmse = beta_errors$rmse,
      beta_mae = beta_errors$mae,
      beta_relative_l_inf = beta_errors$relative_l_inf,
      delta_sensitivity = delta_metrics$sensitivity,
      delta_specificity = delta_metrics$specificity,
      delta_precision = delta_metrics$precision,
      delta_f1 = delta_metrics$f1_score,
      delta_n_selected = delta_metrics$n_selected,
      delta_l_inf = delta_errors$l_inf,
      delta_l2 = delta_errors$l2,
      delta_l1 = delta_errors$l1,
      delta_rmse = delta_errors$rmse,
      delta_mae = delta_errors$mae,
      delta_relative_l_inf = delta_errors$relative_l_inf,
      gamma_error = gamma_error,
      stringsAsFactors = FALSE
    )
  }, error = function(e) {
    cat(sprintf("\n!!! ERROR in %s !!!\n", model_name))
    cat(sprintf("Error message: %s\n", e$message))
    cat("Traceback:\n")
    print(traceback())
    cat("========================\n\n")
    NULL
  })
}

# Create output directory
output_dir <- file.path("results", "raw")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# ============================================================
# Create parameter grid (500 combinations)
# ============================================================
seeds <- 1:100
p_values <- c(1000, 5000, 10000, 15000,20000)
n_familles <- c(500, 1000, 1500, 2000, 2500)
param_grid <- expand.grid(seed = seeds, p = p_values, nfam = n_familles)
# Check if array_task_id is valid
if (array_task_id < 1 || array_task_id > nrow(param_grid)) {
  stop(sprintf("Invalid array_task_id: %d (must be between 1 and %d)",
               array_task_id, nrow(param_grid)))
}

# Get parameters for this job
seed <- param_grid$seed[array_task_id]
p_val <- param_grid$p[array_task_id]
n_fam_val <- param_grid$nfam[array_task_id]
cat(sprintf("Parameters for this job:\n"))
cat(sprintf("  Seed: %d\n", seed))
cat(sprintf("  Number of SNPs (p): %d\n\n", p_val))
cat(sprintf("  Number of Families: %d\n\n", n_fam_val))

# Output file
output_file <- file.path(output_dir,
                         sprintf("seed_%04d_nfam_%04d_p_%04d.rds", seed,n_fam_val, p_val))

# Check if already completed
if (file.exists(output_file)) {
  cat(sprintf("Job already completed: %s\n", output_file))
  quit(status = 0)
}

# ============================================================ 
# Generate data
# ============================================================ 
tryCatch({

  # ============================================================ 
  # Step 1: Generate pedigree and genotypes (X matrix)
  # ============================================================ 
  cat("Step 1: Generating pedigree and genotypes...\n")

  # Fixed parameters
  n_founders <- n_fam_val
  n_generations <- 3
  n_offspring_per_mating <- 2
  d <- p_val  # Number of SNPs from parameter

  # Simulate pedigree
  cat("  - Simulating pedigree with simulatePed...\n")
  pedigree_df_raw <- simulatePed(
    F0size = n_founders,
    Va0 = 1,
    Ve = 1,
    littersize = n_offspring_per_mating,
    ngen = n_generations,
    fsel = "R",
    msel = "R"
  )

  # Extract relevant columns
  pedigree_df <- data.frame(
    id = pedigree_df_raw$ID,
    dam = pedigree_df_raw$DAM,
    sire = pedigree_df_raw$SIRE,
    gen = pedigree_df_raw$GEN,
    sex = ifelse(pedigree_df_raw$SEX == "m", 1, 2)
  )
  N <- nrow(pedigree_df)

  # Generate genotypes
  cat("  - Simulating genotypes with simulateGen...\n")
  AF <- runif(d, min = 0.01, max = 0.99)
  mut.rate <- rep(0, d)

  ped_for_sim <- data.frame(
    ID = pedigree_df$id,
    SIRE = pedigree_df$sire,
    DAM = pedigree_df$dam
  )

  gen_matrix <- simulateGen(ped_for_sim, AF, mut.rate)
  X <- as.matrix(gen_matrix)

  # Calculate kinship matrix K
  cat("  - Calculating kinship matrix K...\n")
  K <- kinship(
    id = pedigree_df$id,
    dadid = pedigree_df$sire,
    momid = pedigree_df$dam
  )
  K <- as.matrix(K)

  # Generate full ancestor list
  cat("  - Generating full ancestor list...\n")
  find_ancestors_recursive <- function(ind_id, ped_df) {
    if (ind_id == 0) return(integer(0))
    parent_row <- ped_df[ped_df$id == ind_id, ]
    if (nrow(parent_row) == 0) return(integer(0))
    dam_id <- parent_row$dam
    sire_id <- parent_row$sire
    direct_parents <- c(dam_id, sire_id)
    direct_parents <- direct_parents[direct_parents > 0]
    maternal_ancestors <- find_ancestors_recursive(dam_id, ped_df)
    paternal_ancestors <- find_ancestors_recursive(sire_id, ped_df)
    return(unique(c(direct_parents, maternal_ancestors, paternal_ancestors)))
  }

  ancestors <- vector("list", N)
  for (i in 1:N) {
    ancestors[[i]] <- find_ancestors_recursive(pedigree_df$id[i], pedigree_df)
  }

  # Generate family IDs
  family_id <- paste(pedigree_df$dam, pedigree_df$sire, sep = "-")

  # ============================================================ 
  # Step 2: Generate FIXED beta, delta, and alpha
  # ============================================================ 
  cat("\nStep 2: Generating beta, delta, and alpha coefficients...\n")

  # Use a FIXED seed for coefficient generation
  set.seed(1000)

  # Parameters
  n_causal_beta <- 15
  n_causal_delta <- 9

  # Adjust if p is small
  if (d < n_causal_beta) {
    n_causal_beta <- floor(d/2)
    n_causal_delta <- floor(n_causal_beta/2)
  }

  causal_beta <- sample(1:d, n_causal_beta)
  causal_delta <- sample(causal_beta, n_causal_delta)

  beta_true <- numeric(d)
  beta_true[causal_beta] <- rnorm(n_causal_beta, 0, 0.5)

  delta_true <- numeric(d)
  delta_true[causal_delta] <- rnorm(n_causal_delta, 0, 0.3)

  gamma_true <- 0  # Fixed gamma for this scenario
  alpha_true <- 1.0
  noise_sd <- 0.5

  cat(sprintf("  - Number of causal betas: %d\n", n_causal_beta))
  cat(sprintf("  - Number of causal deltas: %d\n", n_causal_delta))
  cat(sprintf("  - Gamma (spillover strength): %.2f\n", gamma_true))
  cat(sprintf("  - Alpha (intercept): %.2f\n", alpha_true))
  cat(sprintf("  - Noise SD: %.2f\n\n", noise_sd))

  # ============================================================ 
  # Step 3: Generate outcome y with spillover
  # Model: y = alpha + X*beta + sX*delta + sY*gamma + epsilon
  # ============================================================ 
  cat("Step 3: Generating outcome y with spillover...\n")

  set.seed(seed)  # Use job-specific seed for y generation

  # Step 3.1: Compute sX (does not depend on y)
  sX <- matrix(0, N, d)
  for (i in 1:N) {
    if (length(ancestors[[i]]) > 0) {
      # Calculate weighted sum
      for (j in ancestors[[i]]) {
        sX[i, ] <- sX[i, ] + K[i, j] * X[j, ]
      }
      # Normalize by sum of kinship coefficients
      sum_kinship <- sum(K[i, ancestors[[i]]])
      if (sum_kinship > 0) {
        sX[i, ] <- sX[i, ] / sum_kinship
      }
    }
  }

  # Step 3.2: Compute base linear predictor (without sY term)
  eta_base <- alpha_true + X %*% beta_true + sX %*% delta_true

  # Step 3.3: Generate error term
  epsilon <- rnorm(N, 0, noise_sd)

  # Step 3.4: Generate y in GENERATIONAL ORDER
  y <- numeric(N)
  gen_order <- order(pedigree_df$gen)

  for (idx in gen_order) {
    # Compute sY[idx] based on ancestors (already generated)
    sY_idx <- 0
    if (length(ancestors[[idx]]) > 0) {
      for (j in ancestors[[idx]]) {
        sY_idx <- sY_idx + K[idx, j] * y[j]
      }
      # Normalize by sum of kinship coefficients
      sum_kinship <- sum(K[idx, ancestors[[idx]]])
      if (sum_kinship > 0) {
        sY_idx <- sY_idx / sum_kinship
      }
    }

    # Generate y[idx] directly
    y[idx] <- eta_base[idx] + gamma_true * sY_idx + epsilon[idx]
  }

  # Step 3.5: Compute final spillovers for all individuals
  sY <- numeric(N)
  for (i in 1:N) {
    if (length(ancestors[[i]]) > 0) {
      for (j in ancestors[[i]]) {
        sY[i] <- sY[i] + K[i, j] * y[j]
      }
      # Normalize by sum of kinship coefficients
      sum_kinship <- sum(K[i, ancestors[[i]]])
      if (sum_kinship > 0) {
        sY[i] <- sY[i] / sum_kinship
      }
    }
  }

  cat(sprintf("  - Outcome y generated (N = %d)\n", N))
  cat(sprintf("  - Mean(y): %.4f, SD(y): %.4f\n\n", mean(y), sd(y)))

  # ============================================================ 
  # Step 4: Split data by family (60% train, 20% tune, 20% test)
  # ============================================================ 
  cat("Step 4: Splitting data into train/tune/test...\n")

  set.seed(seed)
  family_ids <- unique(family_id)
  n_families <- length(family_ids)

  shuffled_families <- sample(family_ids)

  n_train_families <- round(0.6 * n_families)
  n_tune_families <- round(0.2 * n_families)

  train_families <- shuffled_families[1:n_train_families]
  tune_families <- shuffled_families[(n_train_families + 1):(n_train_families + n_tune_families)]
  test_families <- shuffled_families[(n_train_families + n_tune_families + 1):n_families]

  idx_train <- which(family_id %in% train_families)
  idx_tune <- which(family_id %in% tune_families)
  idx_test <- which(family_id %in% test_families)

  cat(sprintf("  - Train: %d individuals from %d families\n",
              length(idx_train), n_train_families))
  cat(sprintf("  - Tune:  %d individuals from %d families\n",
              length(idx_tune), n_tune_families))
  cat(sprintf("  - Test:  %d individuals from %d families\n\n",
              length(idx_test), length(test_families)))

  # Prepare data lists
  data_train <- list(
    y = y[idx_train],
    X = X[idx_train, ],
    sX = sX[idx_train, ],
    sY = sY[idx_train],
    K_train = K[idx_train, idx_train],
    family_id = family_id[idx_train],
    beta_true = beta_true,
    delta_true = delta_true,
    gamma_true = gamma_true,
    causal_beta = causal_beta,
    causal_delta = causal_delta,
    family = "gaussian"
  )

  data_tune <- list(
    y = y[idx_tune],
    X = X[idx_tune, ],
    sX = sX[idx_tune, ],
    sY = sY[idx_tune],
    family_id = family_id[idx_tune]
  )

  data_test <- list(
    y = y[idx_test],
    X = X[idx_test, ],
    sX = sX[idx_test, ],
    sY = sY[idx_test],
    family_id = family_id[idx_test]
  )

  # ============================================================ 
  # Step 5: Fit models using HERITAGE
  # ============================================================ 
  cat("Step 5: Fitting models...\n")

  # Define models to compare
  models <- list(
    "Heritage" = fit_nrf_fc_no_gamma,
    # "Lasso" = fit_lasso,
    "Lasso + FE" = fit_lasso_fe,
    # "glmmLasso" = fit_glmmlasso,
    "plmmr" = fit_plmmr
  )

  # Evaluate each model
  results <- lapply(names(models), function(model_name) {
    cat(sprintf("  - Fitting %s\n", model_name))
    tryCatch({
      res_eval = evaluate_model(
        model_name,
        models[[model_name]],
        data_train, data_tune, data_test,
        scenario_name,
        "gaussian"
      )
      res_eval$nfam <- n_fam_val
      res_eval$SampleSize <- N
      res_eval$NSNPs <- d
      res_eval
    }, error = function(e) {
      cat(sprintf("    ERROR in %s: %s\n", model_name, e$message))
      NULL
    })
  })

  # Remove NULL results (failed models)
  results <- results[!sapply(results, is.null)]

  # Check if any models succeeded
  if (length(results) == 0) {
    cat("\n*** ERROR: All models failed! No results to save. ***\n")

    # Save partial results with metadata
    partial_results <- data.frame(
      seed = seed,
      p = p_val,
      nfam = n_fam_val,
      SampleSize = N,
      NSNPs = d,
      scenario = scenario_name,
      status = "all_models_failed"
    )
    saveRDS(partial_results, output_file)
    quit(status = 1)
  }

  results_df <- do.call(rbind, results)

  # Add metadata
  results_df$seed <- seed
  # results_df$param_value <- p_val
  results_df$scenario <- scenario_name

  # Save results
  saveRDS(results_df, output_file)
  cat(sprintf("\nResults saved to: %s\n", output_file))
  cat("=== Job Completed Successfully ===\n")

}, error = function(e) {
  cat(sprintf("\n*** FATAL ERROR: %s ***\n", e$message))
  cat("Traceback:\n")
  print(traceback())
  quit(status = 1)
})