#' Fit the HERITAGE Model
#'
#' Fits a hierarchical sparse regression model for continuous traits in familial
#' cohorts. The model decomposes phenotypic variation into (1) a direct effect
#' of individual covariates \eqn{\boldsymbol{\beta}^\top x_i} and (2) a
#' kinship-weighted ancestral spillover \eqn{\boldsymbol{\delta}^\top s_X^{(i)}}
#' reflecting genetic nurture. Estimation proceeds via the Hierarchical
#' Alternating Minimization Algorithm (HAMA), which alternates between updating
#' direct effects (\eqn{\boldsymbol{\beta}}) and spillover effects
#' (\eqn{\boldsymbol{\delta}}) while enforcing the hierarchical sparsity
#' constraint \eqn{\mathrm{supp}(\boldsymbol{\delta}) \subseteq
#' \mathrm{supp}(\boldsymbol{\beta})}.
#'
#' @details
#' The conditional mean of outcome \eqn{y_i} is
#' \deqn{
#'   \mathbb{E}[y_i \mid \mathbf{X}, \mathbf{y}_{-i}, \mathbf{K}] =
#'   \eta_i = \alpha_y + \boldsymbol{\beta}^\top x_i +
#'   \boldsymbol{\delta}^\top s_X^{(i)},
#' }
#' where the spillover covariate is
#' \deqn{
#'   s_X^{(i)} = \frac{\displaystyle\sum_{j \in \mathcal{N}_i}
#'               \phi_{ij}\, x_j}
#'               {\displaystyle\sum_{j \in \mathcal{N}_i} \phi_{ij}},
#' }
#' \eqn{\mathcal{N}_i := \mathrm{Anc}(i)} is the set of all ancestors of
#' \eqn{i} in the pedigree, and \eqn{\phi_{ij} = K_{ij}} is the kinship
#' coefficient. Founders have \eqn{s_X^{(i)} = 0}.
#'
#' The joint estimator minimises the penalised objective
#' \deqn{
#'   (\hat{\boldsymbol{\theta}}, \hat{\boldsymbol{\delta}}) =
#'   \arg\min_{\boldsymbol{\theta},\,\boldsymbol{\delta}}
#'   \left\{
#'     \frac{1}{2}\sum_{i=1}^{N}(y_i - \eta_i)^2
#'     + \lambda_\beta \|\boldsymbol{\beta}\|_1
#'     + \lambda_\delta \|\boldsymbol{\delta}_{(S)}\|_1
#'   \right\},
#'   \quad \text{s.t.}\;\mathrm{supp}(\boldsymbol{\delta}) \subseteq
#'   \mathrm{supp}(\boldsymbol{\beta}),
#' }
#' where \eqn{S = \mathrm{supp}(\boldsymbol{\beta})}.
#'
#' The penalty parameters \code{lambda_beta} and \code{lambda_delta} must be
#' supplied explicitly. Use \code{\link{heritage_gs}} to select them
#' automatically via grid search on a held-out tuning set.
#'
#' @param y  Numeric vector of outcomes (\eqn{N \times 1}).
#' @param X  Numeric matrix of individual covariates (\eqn{N \times d}),
#'   e.g. standardised genotype dosages.
#' @param sX Numeric matrix of pre-computed spillover covariates
#'   (\eqn{N \times d}). Each row \eqn{i} contains the kinship-weighted
#'   average of the covariates of \eqn{i}'s ancestors. Use
#'   \code{\link{compute_spillover_X}} to build this matrix.
#' @param family  Character; outcome distribution: \code{"gaussian"} (default),
#'   \code{"binomial"}, or \code{"poisson"}.
#' @param lambda_beta  Positive numeric; \eqn{\ell_1} penalty on the direct
#'   effects \eqn{\boldsymbol{\beta}}. Use \code{\link{heritage_gs}} for
#'   automatic selection.
#' @param lambda_delta  Positive numeric; \eqn{\ell_1} penalty on the spillover
#'   effects \eqn{\boldsymbol{\delta}} (restricted to the active set
#'   \eqn{S = \mathrm{supp}(\boldsymbol{\beta})}). Use
#'   \code{\link{heritage_gs}} for automatic selection.
#' @param max_iter  Positive integer; maximum HAMA iterations. Default
#'   \code{100}.
#' @param tol  Positive numeric; convergence tolerance on the
#'   \eqn{\ell_\infty} norm of successive parameter changes. Default
#'   \code{1e-6}.
#' @param standardize  Logical; whether to standardise columns internally
#'   (passed to \code{\link[glmnet]{glmnet}}). Default \code{FALSE}.
#' @param verbose  Logical; print progress messages. Default \code{TRUE}.
#'
#' @return An object of class \code{"heritage"}, a named list with:
#'   \describe{
#'     \item{\code{alpha}}{Fitted intercept.}
#'     \item{\code{beta}}{Numeric vector (\eqn{d \times 1}) of direct effects.}
#'     \item{\code{delta}}{Numeric vector (\eqn{d \times 1}) of spillover
#'       effects.}
#'     \item{\code{active_beta}}{Integer indices of non-zero beta.}
#'     \item{\code{active_delta}}{Integer indices of non-zero delta.}
#'     \item{\code{fitted}}{Fitted values on the response scale.}
#'     \item{\code{residuals}}{Residuals \eqn{y - \hat{y}}.}
#'     \item{\code{deviance}}{Final model deviance.}
#'     \item{\code{iterations}}{Number of HAMA iterations performed.}
#'     \item{\code{converged}}{Logical; \code{TRUE} if the convergence
#'       criterion was met.}
#'     \item{\code{lambda_beta}}{Value of \code{lambda_beta} used.}
#'     \item{\code{lambda_delta}}{Value of \code{lambda_delta} used
#'       (\code{NULL} when the active set is empty).}
#'     \item{\code{family}}{Character; outcome family used.}
#'   }
#'
#' @seealso
#'   \code{\link{heritage_gs}} for automatic hyperparameter selection,
#'   \code{\link{compute_spillover_X}} to build \code{sX}.
#'
#' @references
#' Abdoul Oudouss Diakite, Anne-Marie Madore, Catherine Laprise (2026).
#' HERITAGE: Hierarchical effects regression with interactions for trait
#' analysis in genetics. Manuscript in preparation.
#'
#' @examples
#' set.seed(42)
#' N <- 200; d <- 50
#' X  <- matrix(rnorm(N * d), N, d)
#' sX <- matrix(rnorm(N * d), N, d)
#' beta_true  <- c(rep(0.5, 5), rep(0, d - 5))
#' delta_true <- c(rep(0.3, 5), rep(0, d - 5))
#' y <- X %*% beta_true + sX %*% delta_true + rnorm(N, sd = 0.5)
#'
#' fit <- heritage(y, X, sX, lambda_beta = 0.05, lambda_delta = 0.05)
#' print(fit)
#' summary(fit)
#'
#' @importFrom glmnet glmnet
#' @importFrom stats coef cov var glm
#' @export
heritage <- function(y, X, sX,
                     family       = c("gaussian", "binomial", "poisson"),
                     lambda_beta  = NULL,
                     lambda_delta = NULL,
                     max_iter     = 100L,
                     tol          = 1e-6,
                     standardize  = FALSE,
                     verbose      = TRUE) {

  family <- match.arg(family)
  N <- length(y)
  d <- ncol(X)

  if (nrow(X)  != N) stop("X must have the same number of rows as length(y).")
  if (nrow(sX) != N) stop("sX must have the same number of rows as length(y).")
  if (ncol(sX) != d) stop("sX must have the same number of columns as X.")
  if (is.null(lambda_beta))
    stop("lambda_beta must be specified. Use heritage_gs() for automatic selection.")

  # --- Initialisation ---------------------------------------------------------
  alpha <- switch(family,
                  gaussian = mean(y),
                  binomial = log(mean(y) / (1 - mean(y))),
                  poisson  = log(mean(y)))
  beta      <- numeric(d)
  delta     <- numeric(d)
  converged <- FALSE

  for (iter in seq_len(max_iter)) {

    beta_old  <- beta
    delta_old <- delta
    alpha_old <- alpha

    # --- Step 1: update beta (direct effects) via Lasso ----------------------
    fit_beta  <- glmnet::glmnet(X, y,
                                family      = family,
                                alpha       = 1,
                                lambda      = lambda_beta,
                                offset      = as.vector(sX %*% delta),
                                standardize = standardize,
                                intercept   = TRUE)
    coef_beta <- as.vector(stats::coef(fit_beta))
    alpha     <- coef_beta[1]
    beta      <- coef_beta[-1]

    # --- Step 2: identify active set S ----------------------------------------
    S        <- which(beta != 0)
    n_active <- length(S)

    if (verbose && (iter == 1L || iter %% 10L == 0L))
      message(sprintf("Iter %d: %d active predictors in beta", iter, n_active))

    # --- Step 3: update delta (spillover effects) on S via Lasso -------------
    if (n_active > 0L) {

      Z            <- sX[, S, drop = FALSE]
      offset_delta <- as.vector(alpha + X %*% beta)

      if (ncol(Z) == 1L) {
        # Edge case: single active predictor -- soft-thresholding
        lam_d      <- if (is.null(lambda_delta)) 0.01 else lambda_delta
        z_vec      <- as.vector(Z)
        resid      <- y - offset_delta
        z_var      <- stats::var(z_vec)
        coef_spill <- if (!is.na(z_var) && z_var > 1e-10) {
          rho <- stats::cov(z_vec, resid) / z_var
          if (!is.na(rho)) sign(rho) * max(0, abs(rho) - lam_d) else 0
        } else 0
        delta_new    <- numeric(d)
        delta_new[S] <- coef_spill

      } else {
        if (is.null(lambda_delta))
          stop("lambda_delta must be specified. Use heritage_gs() for automatic selection.")

        fit_spill  <- glmnet::glmnet(Z, y,
                                     family         = family,
                                     alpha          = 1,
                                     lambda         = lambda_delta,
                                     offset         = offset_delta,
                                     standardize    = standardize,
                                     intercept      = FALSE,
                                     penalty.factor = rep(1, n_active))
        coef_spill <- as.vector(stats::coef(fit_spill))[-1]

        delta_new    <- numeric(d)
        delta_new[S] <- coef_spill[seq_len(n_active)]
      }
      delta <- delta_new

    } else {
      fit0  <- stats::glm(y ~ 1, family = family)
      alpha <- stats::coef(fit0)[1]
      delta <- numeric(d)
    }

    # --- Step 4: convergence check --------------------------------------------
    max_change <- max(max(abs(beta  - beta_old)),
                      max(abs(delta - delta_old)),
                      abs(alpha - alpha_old))

    if (verbose && iter %% 10L == 0L)
      message(sprintf("  Max parameter change: %.2e", max_change))

    if (max_change < tol) {
      converged <- TRUE
      if (verbose)
        message(sprintf("Converged at iteration %d (max change: %.2e)",
                        iter, max_change))
      break
    }
  }

  if (!converged && verbose)
    warning(sprintf("HERITAGE did not converge after %d iterations.", max_iter))

  # --- Final predictions ------------------------------------------------------
  eta_final <- alpha + X %*% beta + sX %*% delta

  fitted_final <- switch(family,
                         gaussian = eta_final,
                         binomial = 1 / (1 + exp(-eta_final)),
                         poisson  = exp(eta_final))

  residuals_final <- y - fitted_final

  deviance_final <- switch(
    family,
    gaussian = sum(residuals_final^2),
    binomial = -2 * sum(y * log(fitted_final + 1e-10) +
                          (1 - y) * log(1 - fitted_final + 1e-10)),
    poisson  = 2 * sum(y * log((y + 1e-10) / (fitted_final + 1e-10)) -
                         (y - fitted_final))
  )

  structure(
    list(
      alpha        = alpha,
      beta         = beta,
      delta        = delta,
      active_beta  = which(beta  != 0),
      active_delta = which(delta != 0),
      fitted       = as.vector(fitted_final),
      residuals    = as.vector(residuals_final),
      deviance     = deviance_final,
      iterations   = iter,
      converged    = converged,
      lambda_beta  = lambda_beta,
      lambda_delta = if (n_active > 0L) lambda_delta else NULL,
      family       = family
    ),
    class = "heritage"
  )
}


# =============================================================================
# S3 methods
# =============================================================================

#' Print a fitted HERITAGE model
#'
#' @param x  An object of class \code{"heritage"}.
#' @param ... Currently unused.
#' @return \code{x}, invisibly.
#' @importFrom utils head
#' @export
print.heritage <- function(x, ...) {
  cat("HERITAGE -- Hierarchical Effects Regression with Interactions\n")
  cat("============================================================\n\n")
  cat(sprintf("Family    : %s\n",   x$family))
  cat(sprintf("Converged : %s (%d iterations)\n", x$converged, x$iterations))
  cat(sprintf("Deviance  : %.4f\n\n", x$deviance))
  cat(sprintf("Intercept (alpha)         : %.4f\n", x$alpha))
  cat(sprintf("Direct effects  (beta)    : %d / %d non-zero\n",
              length(x$active_beta), length(x$beta)))
  if (length(x$active_beta) > 0L) {
    top <- utils::head(x$active_beta, 10L)
    cat(sprintf("  Active indices: %s%s\n",
                paste(top, collapse = ", "),
                if (length(x$active_beta) > 10L) ", ..." else ""))
  }
  cat(sprintf("Spillover effects (delta) : %d / %d non-zero\n",
              length(x$active_delta), length(x$delta)))
  if (length(x$active_delta) > 0L) {
    top <- utils::head(x$active_delta, 10L)
    cat(sprintf("  Active indices: %s%s\n",
                paste(top, collapse = ", "),
                if (length(x$active_delta) > 10L) ", ..." else ""))
  }
  cat(sprintf("\nlambda_beta  : %.4e\n", x$lambda_beta))
  if (!is.null(x$lambda_delta))
    cat(sprintf("lambda_delta : %.4e\n", x$lambda_delta))
  invisible(x)
}

#' Summarise a fitted HERITAGE model
#'
#' Prints a detailed summary including all non-zero coefficients.
#'
#' @param object  An object of class \code{"heritage"}.
#' @param ... Currently unused.
#' @return \code{object}, invisibly.
#' @export
summary.heritage <- function(object, ...) {
  print(object)
  cat("\n============================================================\n")
  cat("Non-zero coefficients:\n\n")
  if (length(object$active_beta) > 0L) {
    cat("Direct effects (beta):\n")
    print(data.frame(Index       = object$active_beta,
                     Coefficient = object$beta[object$active_beta]),
          row.names = FALSE)
  }
  if (length(object$active_delta) > 0L) {
    cat("\nSpillover effects (delta):\n")
    print(data.frame(Index       = object$active_delta,
                     Coefficient = object$delta[object$active_delta]),
          row.names = FALSE)
  }
  invisible(object)
}

#' Predict from a fitted HERITAGE model
#'
#' Computes predictions for new observations using fitted direct and
#' spillover effects.
#'
#' @param object  An object of class \code{"heritage"}.
#' @param newX    Numeric matrix of new individual covariates
#'   (\eqn{N_{\mathrm{new}} \times d}). Required.
#' @param newsX   Numeric matrix of new spillover covariates
#'   (\eqn{N_{\mathrm{new}} \times d}). Required.
#' @param type    Character; \code{"link"} (default) returns the linear
#'   predictor; \code{"response"} applies the inverse link function.
#' @param ...     Currently unused.
#'
#' @return Numeric vector of length \eqn{N_{\mathrm{new}}}.
#' @export
predict.heritage <- function(object,
                             newX  = NULL,
                             newsX = NULL,
                             type  = c("link", "response"),
                             ...) {
  type <- match.arg(type)
  if (is.null(newX))  stop("newX must be provided.")
  if (is.null(newsX)) stop("newsX must be provided.")

  eta <- object$alpha + newX %*% object$beta + newsX %*% object$delta

  if (type == "link") return(as.vector(eta))

  fitted <- switch(object$family,
                   gaussian = eta,
                   binomial = 1 / (1 + exp(-eta)),
                   poisson  = exp(eta))
  as.vector(fitted)
}
