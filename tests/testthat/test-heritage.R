## Tests for heritage()

set.seed(123)
N <- 100; d <- 20
X  <- matrix(stats::rnorm(N * d), N, d)
sX <- matrix(stats::rnorm(N * d), N, d)
beta_true  <- c(rep(0.5, 3), rep(0, d - 3))
delta_true <- c(rep(0.3, 3), rep(0, d - 3))
y <- X %*% beta_true + sX %*% delta_true + stats::rnorm(N, sd = 0.5)

fit <- heritage(y, X, sX,
                lambda_beta  = 0.05,
                lambda_delta = 0.05,
                verbose      = FALSE)

test_that("heritage() returns an object of class 'heritage'", {
  expect_s3_class(fit, "heritage")
})

test_that("heritage() output contains all required components", {
  nms <- c("alpha", "beta", "delta", "active_beta", "active_delta",
           "fitted", "residuals", "deviance", "iterations", "converged",
           "lambda_beta", "lambda_delta", "family")
  expect_true(all(nms %in% names(fit)))
})

test_that("beta and delta have length d", {
  expect_length(fit$beta,  d)
  expect_length(fit$delta, d)
})

test_that("fitted values and residuals have length N", {
  expect_length(fit$fitted,    N)
  expect_length(fit$residuals, N)
})

test_that("hierarchical sparsity constraint is satisfied", {
  # supp(delta) must be a subset of supp(beta)
  expect_true(all(fit$active_delta %in% fit$active_beta))
})

test_that("deviance is non-negative for Gaussian family", {
  expect_gte(fit$deviance, 0)
})

test_that("predict.heritage() returns a vector of length N", {
  pred <- predict(fit, newX = X, newsX = sX, type = "response")
  expect_length(pred, N)
})

test_that("predict.heritage() link and response are identical for Gaussian", {
  p_link     <- predict(fit, newX = X, newsX = sX, type = "link")
  p_response <- predict(fit, newX = X, newsX = sX, type = "response")
  expect_equal(p_link, p_response)
})

test_that("predict.heritage() errors without newX or newsX", {
  expect_error(predict(fit, newsX = sX), "newX")
  expect_error(predict(fit, newX  = X),  "newsX")
})

## Tests for compute_spillover_X()

test_that("compute_spillover_X() returns matrix of correct dimensions", {
  K   <- diag(N)
  anc <- c(list(integer(0), integer(0)),
           lapply(3:N, function(i) c(1L, 2L)))
  sX2 <- compute_spillover_X(X, K, anc)
  expect_equal(dim(sX2), c(N, d))
})

test_that("founders receive zero spillover row", {
  K   <- diag(N)
  anc <- c(list(integer(0), integer(0)),
           lapply(3:N, function(i) c(1L, 2L)))
  sX2 <- compute_spillover_X(X, K, anc)
  expect_true(all(sX2[1L, ] == 0))
  expect_true(all(sX2[2L, ] == 0))
})

test_that("non-founders receive non-zero spillover when X is non-zero", {
  X_pos <- matrix(1, N, d)  # all ones => spillover should be 1 everywhere
  K     <- diag(N)
  anc   <- c(list(integer(0), integer(0)),
             lapply(3:N, function(i) c(1L, 2L)))
  sX2   <- compute_spillover_X(X_pos, K, anc)
  # For K = I, row i gets K[i, anc_i] = K[i, 1] and K[i, 2] = 0 (off-diagonal)
  # All spillovers for non-founders will be 0 because K = I is off-diagonal zero
  # Use a denser K to test properly
  K2 <- matrix(0.25, N, N); diag(K2) <- 1
  sX3 <- compute_spillover_X(X_pos, K2, anc)
  expect_true(all(sX3[3L, ] == 1))  # weighted avg of all-ones = 1
})
