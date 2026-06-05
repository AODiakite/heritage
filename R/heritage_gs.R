#' Grid Search for HERITAGE Hyperparameters
#'
#' Performs a two-dimensional grid search over (\code{lambda_beta},
#' \code{lambda_delta}) to find the penalty pair that optimises a
#' user-specified validation metric on a held-out tuning set. After selection,
#' the best model is evaluated on an independent test set.
#'
#' @details
#' Lambda sequences are generated automatically from the training data when
#' not supplied:
#' \itemize{
#'   \item \code{lambda_beta_seq} is taken from the glmnet regularisation path
#'     on \code{X_train}.
#'   \item \code{lambda_delta_seq} is built from the empirical correlation
#'     \eqn{\max_k |(\mathbf{s}_X^\top \mathbf{y})_k| / N} down to a fraction
#'     of that value on a log scale.
#' }
#'
#' @param y_train,y_tune,y_test  Numeric vectors of outcomes for the training,
#'   tuning, and test splits.
#' @param X_train,X_tune,X_test  Numeric matrices of individual covariates for
#'   each split.
#' @param sX_train,sX_tune,sX_test  Numeric matrices of spillover covariates
#'   for each split (see \code{\link{compute_spillover_X}}).
#' @param family  Character; \code{"gaussian"} (default), \code{"binomial"},
#'   or \code{"poisson"}.
#' @param n_lambda_beta   Integer; number of \code{lambda_beta} candidate
#'   values when \code{lambda_beta_seq} is not supplied. Default \code{20}.
#' @param n_lambda_delta  Integer; number of \code{lambda_delta} candidate
#'   values when \code{lambda_delta_seq} is not supplied. Default \code{20}.
#' @param lambda_beta_seq   Numeric vector; custom \code{lambda_beta} grid.
#'   If \code{NULL} (default), generated automatically.
#' @param lambda_delta_seq  Numeric vector; custom \code{lambda_delta} grid.
#'   If \code{NULL} (default), generated automatically.
#' @param metric  Character; metric to optimise on the tuning set:
#'   \itemize{
#'     \item Gaussian: \code{"mse"} (default), \code{"rmse"}, \code{"mae"},
#'       \code{"r2"}.
#'     \item Binomial: \code{"accuracy"} (default), \code{"auc"},
#'       \code{"logloss"}, \code{"f1"}.
#'     \item Poisson: \code{"mse"} (default), \code{"mae"},
#'       \code{"deviance"}, \code{"mean_deviance"}.
#'   }
#' @param max_iter  Integer; maximum HAMA iterations per fit. Default
#'   \code{100}.
#' @param tol  Numeric; convergence tolerance. Default \code{1e-6}.
#' @param standardize  Logical; standardise predictors inside glmnet calls.
#'   Default \code{FALSE}.
#' @param verbose  Logical; print progress. Default \code{TRUE}.
#'
#' @return An object of class \code{"heritage_gs"}, a list with:
#'   \describe{
#'     \item{\code{best_model}}{Fitted \code{"heritage"} object with the
#'       optimal penalty pair.}
#'     \item{\code{best_lambda_beta}}{Optimal \code{lambda_beta}.}
#'     \item{\code{best_lambda_delta}}{Optimal \code{lambda_delta}.}
#'     \item{\code{tune_metrics}}{Named list of tuning-set metrics for the
#'       best model.}
#'     \item{\code{test_metrics}}{Named list of test-set metrics for the best
#'       model.}
#'     \item{\code{grid_results}}{\code{data.frame} with all grid combinations
#'       and corresponding tuning-set metric values.}
#'     \item{\code{lambda_beta_seq}}{Lambda sequence used for beta.}
#'     \item{\code{lambda_delta_seq}}{Lambda sequence used for delta.}
#'     \item{\code{metric}}{Metric name that was optimised.}
#'     \item{\code{family}}{Family used.}
#'   }
#'
#' @seealso \code{\link{heritage}}, \code{\link{compute_spillover_X}},
#'   \code{\link{plot.heritage_gs}}
#'
#' @references
#' Abdoul Oudouss Diakite, Anne-Marie Madore, Celia M. T. Greenwood, Catherine Laprise (2026).
#' HERITAGE: Hierarchical effects regression with interactions for trait
#' analysis in genetics. Manuscript in preparation.
#'
#' @examples
#' \dontrun{
#' set.seed(1)
#' N <- 300; d <- 50
#' X  <- matrix(rnorm(N * d), N, d)
#' sX <- matrix(rnorm(N * d), N, d)
#' y  <- X[, 1] * 0.5 + sX[, 1] * 0.3 + rnorm(N, sd = 0.5)
#'
#' idx <- sample(N)
#' tr  <- idx[1:180]; tu <- idx[181:240]; te <- idx[241:300]
#'
#' gs <- heritage_gs(
#'   y_train  = y[tr],  X_train  = X[tr, ],  sX_train  = sX[tr, ],
#'   y_tune   = y[tu],  X_tune   = X[tu, ],  sX_tune   = sX[tu, ],
#'   y_test   = y[te],  X_test   = X[te, ],  sX_test   = sX[te, ],
#'   n_lambda_beta = 10, n_lambda_delta = 10
#' )
#' print(gs)
#' plot(gs)
#' }
#'
#' @importFrom glmnet glmnet
#' @importFrom stats predict
#' @export
heritage_gs <- function(y_train, X_train, sX_train,
                        y_tune,  X_tune,  sX_tune,
                        y_test,  X_test,  sX_test,
                        family           = c("gaussian", "binomial", "poisson"),
                        n_lambda_beta    = 20L,
                        n_lambda_delta   = 20L,
                        lambda_beta_seq  = NULL,
                        lambda_delta_seq = NULL,
                        metric           = NULL,
                        max_iter         = 100L,
                        tol              = 1e-6,
                        standardize      = FALSE,
                        verbose          = TRUE) {

  family <- match.arg(family)

  # --- Default metric ---------------------------------------------------------
  if (is.null(metric))
    metric <- switch(family,
                     gaussian = "mse", binomial = "accuracy", poisson = "mse")

  valid_metrics <- switch(
    family,
    gaussian = c("mse", "rmse", "mae", "r2"),
    binomial = c("accuracy", "auc", "logloss", "f1", "precision", "recall"),
    poisson  = c("mse", "mae", "deviance", "mean_deviance")
  )
  if (!(metric %in% valid_metrics))
    stop(sprintf("Invalid metric '%s' for family '%s'. Choose from: %s.",
                 metric, family, paste(valid_metrics, collapse = ", ")))

  minimize_metrics <- c("mse", "rmse", "mae", "logloss", "deviance",
                        "mean_deviance")
  maximize_metric  <- !(metric %in% minimize_metrics)

  if (verbose) {
    cat(sprintf("HERITAGE grid search -- family: %s\n", family))
    cat(sprintf("Optimising metric : %s (%s)\n", metric,
                if (maximize_metric) "maximise" else "minimise"))
  }

  # --- Build lambda sequences -------------------------------------------------
  if (is.null(lambda_beta_seq)) {
    if (verbose) cat("Generating lambda_beta sequence from glmnet path...\n")
    tmp             <- glmnet::glmnet(X_train, y_train, family = family,
                                      alpha = 1, nlambda = n_lambda_beta,
                                      standardize = standardize)
    lambda_beta_seq <- tmp$lambda
    if (verbose)
      cat(sprintf("  lambda_beta : [%.3e, %.3e] (%d values)\n",
                  max(lambda_beta_seq), min(lambda_beta_seq),
                  length(lambda_beta_seq)))
  }

  if (is.null(lambda_delta_seq)) {
    if (verbose) cat("Generating lambda_delta sequence...\n")
    n_tr         <- length(y_train)
    cors         <- abs(t(sX_train) %*% y_train) / n_tr
    lam_max_d    <- max(cors)
    if (lam_max_d < 1e-6) {
      lam_max_d <- max(lambda_beta_seq)
      if (verbose)
        cat("  Weak spillover signal -- using max(lambda_beta) as reference.\n")
    }
    ratio            <- if (nrow(sX_train) < ncol(sX_train)) 0.01 else 0.0001
    lambda_delta_seq <- exp(seq(log(lam_max_d),
                                log(lam_max_d * ratio),
                                length.out = n_lambda_delta))
    if (verbose)
      cat(sprintf("  lambda_delta: [%.3e, %.3e] (%d values)\n",
                  max(lambda_delta_seq), min(lambda_delta_seq),
                  length(lambda_delta_seq)))
  }

  # --- Grid search ------------------------------------------------------------
  n_comb <- length(lambda_beta_seq) * length(lambda_delta_seq)
  if (verbose)
    cat(sprintf("\nTesting %d combinations (%d x %d)...\n\n",
                n_comb, length(lambda_beta_seq), length(lambda_delta_seq)))

  grid_results <- data.frame(
    lambda_beta    = numeric(n_comb),
    lambda_delta   = numeric(n_comb),
    tune_metric    = numeric(n_comb),
    converged      = logical(n_comb),
    n_active_beta  = integer(n_comb),
    n_active_delta = integer(n_comb)
  )

  best_metric <- if (maximize_metric) -Inf else Inf
  best_model  <- NULL
  best_idx    <- NULL
  idx         <- 1L

  for (i in seq_along(lambda_beta_seq)) {
    for (j in seq_along(lambda_delta_seq)) {
      lb <- lambda_beta_seq[i]
      ld <- lambda_delta_seq[j]

      if (verbose && idx %% 10L == 0L)
        cat(sprintf("  Progress: %d / %d\n", idx, n_comb))

      fit <- tryCatch(
        heritage(y_train, X_train, sX_train,
                 family       = family,
                 lambda_beta  = lb,
                 lambda_delta = ld,
                 max_iter     = max_iter,
                 tol          = tol,
                 standardize  = standardize,
                 verbose      = FALSE),
        error = function(e) {
          if (verbose)
            message(sprintf("  Fit error (lb=%.2e, ld=%.2e): %s",
                            lb, ld, e$message))
          NULL
        }
      )

      if (!is.null(fit)) {
        y_pred_tune <- predict(fit,
                               newX  = X_tune,
                               newsX = sX_tune,
                               type  = "response")
        tune_met    <- .heritage_metrics(y_tune, y_pred_tune, family)
        cur         <- tune_met[[metric]]
        is_better   <- if (maximize_metric) cur > best_metric else
                                            cur < best_metric

        grid_results[idx, ] <- list(lb, ld, cur, fit$converged,
                                    length(fit$active_beta),
                                    length(fit$active_delta))

        if (is_better && !is.na(cur) && is.finite(cur)) {
          best_metric <- cur
          best_model  <- fit
          best_idx    <- idx
        }
      } else {
        grid_results[idx, ] <- list(lb, ld, NA, FALSE, NA, NA)
      }
      idx <- idx + 1L
    }
  }

  if (is.null(best_model))
    stop("Grid search failed: no model fit successfully.")

  best_lb <- grid_results$lambda_beta[best_idx]
  best_ld <- grid_results$lambda_delta[best_idx]

  if (verbose) {
    cat(sprintf("\n=== Grid search complete ===\n"))
    cat(sprintf("Best %-12s (tuning): %.6f\n", metric, best_metric))
    cat(sprintf("Best lambda_beta         : %.4e\n", best_lb))
    cat(sprintf("Best lambda_delta        : %.4e\n", best_ld))
    cat(sprintf("Active beta              : %d\n", length(best_model$active_beta)))
    cat(sprintf("Active delta             : %d\n", length(best_model$active_delta)))
  }

  # --- Final evaluation -------------------------------------------------------
  y_pred_tune  <- predict(best_model, newX = X_tune, newsX = sX_tune,
                          type = "response")
  y_pred_test  <- predict(best_model, newX = X_test, newsX = sX_test,
                          type = "response")
  tune_metrics <- .heritage_metrics(y_tune, y_pred_tune, family)
  test_metrics <- .heritage_metrics(y_test, y_pred_test, family)

  if (verbose) {
    cat("\n=== Test set performance ===\n")
    for (nm in names(test_metrics))
      cat(sprintf("  %-16s: %.6f\n", nm, test_metrics[[nm]]))
  }

  structure(
    list(
      best_model       = best_model,
      best_lambda_beta  = best_lb,
      best_lambda_delta = best_ld,
      tune_metrics     = tune_metrics,
      test_metrics     = test_metrics,
      grid_results     = grid_results,
      lambda_beta_seq  = lambda_beta_seq,
      lambda_delta_seq = lambda_delta_seq,
      metric           = metric,
      family           = family
    ),
    class = "heritage_gs"
  )
}


# =============================================================================
# S3 methods for heritage_gs
# =============================================================================

#' Print grid-search results for HERITAGE
#'
#' @param x  An object of class \code{"heritage_gs"}.
#' @param ... Currently unused.
#' @return \code{x}, invisibly.
#' @export
print.heritage_gs <- function(x, ...) {
  cat("HERITAGE -- Grid Search Results\n")
  cat("============================================================\n\n")
  cat(sprintf("Family           : %s\n",   x$family))
  cat(sprintf("Optimised metric : %s\n\n", x$metric))
  cat(sprintf("Best lambda_beta : %.4e\n", x$best_lambda_beta))
  cat(sprintf("Best lambda_delta: %.4e\n\n", x$best_lambda_delta))
  cat("Tuning set metrics:\n")
  for (nm in names(x$tune_metrics))
    cat(sprintf("  %-16s: %.6f\n", nm, x$tune_metrics[[nm]]))
  cat("\nTest set metrics:\n")
  for (nm in names(x$test_metrics))
    cat(sprintf("  %-16s: %.6f\n", nm, x$test_metrics[[nm]]))
  cat(sprintf("\nBest model: %d active beta, %d active delta\n",
              length(x$best_model$active_beta),
              length(x$best_model$active_delta)))
  invisible(x)
}

#' Summarise grid-search results for HERITAGE
#'
#' @param object  An object of class \code{"heritage_gs"}.
#' @param ... Currently unused.
#' @return \code{object}, invisibly.
#' @export
summary.heritage_gs <- function(object, ...) {
  print(object)
  cat("\n============================================================\n")
  cat("Grid search summary:\n\n")
  valid <- object$grid_results[!is.na(object$grid_results$tune_metric), ]
  conv  <- valid[valid$converged, ]
  cat(sprintf("Total combinations : %d\n", nrow(object$grid_results)))
  cat(sprintf("Valid fits         : %d\n", nrow(valid)))
  cat(sprintf("Converged fits     : %d\n", nrow(conv)))
  if (nrow(valid) > 0L) {
    cat(sprintf("\n%s on tuning set -- min: %.4f | mean: %.4f | max: %.4f\n",
                object$metric,
                min(valid$tune_metric,  na.rm = TRUE),
                mean(valid$tune_metric, na.rm = TRUE),
                max(valid$tune_metric,  na.rm = TRUE)))
  }
  invisible(object)
}

#' Plot grid-search results for HERITAGE
#'
#' Produces diagnostic plots for a HERITAGE penalty grid search.
#'
#' @param x     An object of class \code{"heritage_gs"}.
#' @param type  Character; type of plot:
#'   \describe{
#'     \item{\code{"heatmap"}}{(default) 2D colour map of the tuning metric
#'       over the log-penalty grid. The optimal pair is marked with a red
#'       star.}
#'     \item{\code{"paths"}}{Regularisation paths: number of active direct and
#'       spillover coefficients versus the respective log-penalty.}
#'     \item{\code{"convergence"}}{Convergence status across the grid.}
#'     \item{\code{"sparsity"}}{Scatter plots of active coefficient counts
#'       versus penalty values, with a smoothing trend.}
#'     \item{\code{"all"}}{All four panels in a 2x2 layout.}
#'   }
#' @param ... Currently unused.
#' @return \code{NULL}, invisibly (called for its side effects).
#' @importFrom graphics par image contour points legend abline lines grid plot axis
#'   plot.new text
#' @importFrom grDevices colorRampPalette rgb
#' @importFrom stats loess predict
#' @export
plot.heritage_gs <- function(x,
                              type = c("heatmap", "paths", "convergence",
                                       "sparsity", "all"),
                              ...) {
  type       <- match.arg(type)
  grid       <- x$grid_results
  valid_grid <- grid[!is.na(grid$tune_metric), ]
  maximize   <- !(x$metric %in% c("mse", "rmse", "mae", "logloss",
                                   "deviance", "mean_deviance"))

  switch(type,
    all = {
      old <- graphics::par(mfrow = c(2L, 2L), mar = c(4, 4, 2.5, 1.5))
      on.exit(graphics::par(old))
      .heritage_plot_heatmap(x, valid_grid, maximize)
      .heritage_plot_paths(x, valid_grid)
      .heritage_plot_convergence(x, grid)
      .heritage_plot_sparsity(x, valid_grid)
    },
    heatmap = {
      .heritage_plot_heatmap(x, valid_grid, maximize)
    },
    paths = {
      old <- graphics::par(mfrow = c(1L, 2L), mar = c(4.5, 4.5, 3, 2))
      on.exit(graphics::par(old))
      .heritage_plot_paths(x, valid_grid)
    },
    convergence = {
      .heritage_plot_convergence(x, grid)
    },
    sparsity = {
      old <- graphics::par(mfrow = c(1L, 2L), mar = c(4.5, 4.5, 3, 2))
      on.exit(graphics::par(old))
      .heritage_plot_sparsity(x, valid_grid)
    }
  )
  invisible(NULL)
}


# =============================================================================
# Internal plot helpers
# =============================================================================

#' @keywords internal
.heritage_plot_heatmap <- function(x, valid_grid, maximize) {
  lb_s <- sort(x$lambda_beta_seq)
  ld_s <- sort(x$lambda_delta_seq)
  mat  <- matrix(NA_real_, length(lb_s), length(ld_s))
  for (i in seq_len(nrow(valid_grid))) {
    ib <- which(lb_s == valid_grid$lambda_beta[i])
    id <- which(ld_s == valid_grid$lambda_delta[i])
    if (length(ib) && length(id)) mat[ib, id] <- valid_grid$tune_metric[i]
  }
  cols <- if (maximize)
    grDevices::colorRampPalette(
      c("#440154", "#31688e", "#35b779", "#fde724"))(100L)
  else
    grDevices::colorRampPalette(
      c("#fde724", "#35b779", "#31688e", "#440154"))(100L)

  graphics::image(log10(lb_s), log10(ld_s), mat, col = cols,
                  xlab = expression(log[10](lambda[beta])),
                  ylab = expression(log[10](lambda[delta])),
                  main = sprintf("Grid: %s (%s)", x$metric,
                                 if (maximize) "higher is better" else
                                               "lower is better"))
  if (sum(!is.na(mat)) >= 10L)
    suppressWarnings(
      graphics::contour(log10(lb_s), log10(ld_s), mat, add = TRUE,
                        col = "white", lwd = 0.8, labcex = 0.6,
                        method = "edge", nlevels = 8L)
    )
  graphics::points(log10(x$best_lambda_beta), log10(x$best_lambda_delta),
                   pch = 8L, cex = 2.5, col = "red", lwd = 3L)
  graphics::legend("topright", legend = "Best combination",
                   pch = 8L, col = "red", lwd = 3L, cex = 0.7, bg = "white")
}

#' @keywords internal
.heritage_plot_paths <- function(x, valid_grid) {
  # Integer y-axis ticks: active-coefficient counts are whole numbers, so force
  # integer tick marks instead of R's default (which can label 0.0, 0.5, ...).
  int_ticks <- function(counts) {
    rng <- range(counts)
    seq(floor(rng[1L]), ceiling(rng[2L]),
        by = max(1L, ceiling(diff(rng) / 6)))
  }

  # Slice the 2D grid at the selected penalty of the other dimension so the
  # paths show integer active-coefficient counts for the actual fitted models,
  # rather than averages across the whole grid.
  bd <- valid_grid[valid_grid$lambda_delta == x$best_lambda_delta, ]
  bd <- bd[order(bd$lambda_beta, decreasing = TRUE), ]
  graphics::plot(log10(bd$lambda_beta), bd$n_active_beta,
                 type = "b", pch = 19L, col = "#2c7fb8", yaxt = "n",
                 xlab = expression(log[10](lambda[beta])),
                 ylab = "Active beta",
                 main = "Regularisation path: direct effects")
  graphics::axis(2L, at = int_ticks(bd$n_active_beta))
  graphics::grid()
  graphics::abline(v = log10(x$best_lambda_beta), col = "red", lty = 2L,
                   lwd = 2L)
  graphics::legend("topright",
                   legend = sprintf("Best: %.2e", x$best_lambda_beta),
                   lty = 2L, col = "red", cex = 0.8, bg = "white")

  dd <- valid_grid[valid_grid$lambda_beta == x$best_lambda_beta, ]
  dd <- dd[order(dd$lambda_delta, decreasing = TRUE), ]
  graphics::plot(log10(dd$lambda_delta), dd$n_active_delta,
                 type = "b", pch = 19L, col = "#41ab5d", yaxt = "n",
                 xlab = expression(log[10](lambda[delta])),
                 ylab = "Active delta",
                 main = "Regularisation path: spillover effects")
  graphics::axis(2L, at = int_ticks(dd$n_active_delta))
  graphics::grid()
  graphics::abline(v = log10(x$best_lambda_delta), col = "red", lty = 2L,
                   lwd = 2L)
  graphics::legend("topright",
                   legend = sprintf("Best: %.2e", x$best_lambda_delta),
                   lty = 2L, col = "red", cex = 0.8, bg = "white")
}

#' @keywords internal
.heritage_plot_convergence <- function(x, grid) {
  lb_s <- sort(x$lambda_beta_seq)
  ld_s <- sort(x$lambda_delta_seq)
  mat  <- matrix(NA_real_, length(lb_s), length(ld_s))
  for (i in seq_len(nrow(grid))) {
    ib <- which(lb_s == grid$lambda_beta[i])
    id <- which(ld_s == grid$lambda_delta[i])
    if (length(ib) && length(id)) mat[ib, id] <- as.numeric(grid$converged[i])
  }
  if (all(is.na(mat))) {
    graphics::plot.new()
    graphics::text(0.5, 0.5, "No convergence data", cex = 1.5)
    return(invisible(NULL))
  }
  uv   <- unique(as.vector(mat[!is.na(mat)]))
  cols <- if (length(uv) == 1L) {
    if (uv == 1L) "#4575b4" else "#d73027"
  } else c("#d73027", "#4575b4")

  graphics::image(log10(lb_s), log10(ld_s), mat, col = cols,
                  xlab = expression(log[10](lambda[beta])),
                  ylab = expression(log[10](lambda[delta])),
                  main = "Convergence status")
  graphics::points(log10(x$best_lambda_beta), log10(x$best_lambda_delta),
                   pch = 8L, cex = 2L, col = "yellow", lwd = 2L)
  graphics::legend("topright",
                   legend = c("Converged", "Not converged", "Best"),
                   fill   = c("#4575b4", "#d73027", NA),
                   pch    = c(NA, NA, 8L),
                   col    = c(NA, NA, "yellow"),
                   cex = 0.7, bg = "white")
}

#' @keywords internal
.heritage_plot_sparsity <- function(x, valid_grid) {
  graphics::plot(log10(valid_grid$lambda_beta), valid_grid$n_active_beta,
                 pch = 19L, col = grDevices::rgb(0.2, 0.5, 0.7, 0.5),
                 cex = 0.8,
                 xlab = expression(log[10](lambda[beta])),
                 ylab = "Active beta",
                 main = "Sparsity: direct effects")
  graphics::grid()
  if (nrow(valid_grid) > 5L) {
    sm <- stats::loess(n_active_beta ~ log10(lambda_beta),
                       data = valid_grid, span = 0.3)
    sx <- sort(log10(valid_grid$lambda_beta))
    graphics::lines(sx, stats::predict(sm, data.frame(lambda_beta = 10^sx)),
                    col = "#2c7fb8", lwd = 2L)
  }
  graphics::points(log10(x$best_lambda_beta),
                   length(x$best_model$active_beta),
                   pch = 8L, cex = 2L, col = "red", lwd = 2L)

  graphics::plot(log10(valid_grid$lambda_delta), valid_grid$n_active_delta,
                 pch = 19L, col = grDevices::rgb(0.2, 0.7, 0.4, 0.5),
                 cex = 0.8,
                 xlab = expression(log[10](lambda[delta])),
                 ylab = "Active delta",
                 main = "Sparsity: spillover effects")
  graphics::grid()
  if (nrow(valid_grid) > 5L) {
    sm <- stats::loess(n_active_delta ~ log10(lambda_delta),
                       data = valid_grid, span = 0.3)
    sx <- sort(log10(valid_grid$lambda_delta))
    graphics::lines(sx, stats::predict(sm, data.frame(lambda_delta = 10^sx)),
                    col = "#41ab5d", lwd = 2L)
  }
  graphics::points(log10(x$best_lambda_delta),
                   length(x$best_model$active_delta),
                   pch = 8L, cex = 2L, col = "red", lwd = 2L)
}
