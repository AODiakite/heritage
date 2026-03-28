# Internal utility functions (not exported)
# Used by heritage_gs().

#' Compute performance metrics
#'
#' @param y      Observed outcomes.
#' @param y_pred Predicted values (response scale).
#' @param family Character; outcome family.
#' @return Named list of numeric scalars.
#' @keywords internal
.heritage_metrics <- function(y, y_pred, family) {
  out <- list()

  if (family == "gaussian") {
    out$mse  <- mean((y - y_pred)^2)
    out$rmse <- sqrt(out$mse)
    out$mae  <- mean(abs(y - y_pred))
    ss_res   <- sum((y - y_pred)^2)
    ss_tot   <- sum((y - mean(y))^2)
    out$r2   <- 1 - ss_res / ss_tot

  } else if (family == "binomial") {
    y_class      <- ifelse(y_pred > 0.5, 1, 0)
    out$accuracy <- mean(y_class == y)

    out$auc <- tryCatch({
      if (requireNamespace("pROC", quietly = TRUE)) {
        as.numeric(pROC::auc(pROC::roc(y, y_pred, quiet = TRUE)))
      } else NA_real_
    }, error = function(e) NA_real_)

    eps          <- 1e-15
    yp           <- pmax(pmin(y_pred, 1 - eps), eps)
    out$logloss  <- -mean(y * log(yp) + (1 - y) * log(1 - yp))
    tp           <- sum(y == 1L & y_class == 1L)
    fp           <- sum(y == 0L & y_class == 1L)
    fn           <- sum(y == 1L & y_class == 0L)
    out$precision <- if (tp + fp > 0L) tp / (tp + fp) else 0
    out$recall    <- if (tp + fn > 0L) tp / (tp + fn) else 0
    out$f1        <- if (out$precision + out$recall > 0)
      2 * out$precision * out$recall / (out$precision + out$recall) else 0

  } else if (family == "poisson") {
    out$mse           <- mean((y - y_pred)^2)
    out$mae           <- mean(abs(y - y_pred))
    eps               <- 1e-10
    out$deviance      <- 2 * sum(y * log((y + eps) / (y_pred + eps)) -
                                 (y - y_pred))
    out$mean_deviance <- out$deviance / length(y)
  }

  out
}
