#' Find All Ancestors of an Individual in a Pedigree
#'
#' Recursively traverses a pedigree data frame to collect the full set of
#' ancestors (parents, grandparents, ...) of a given individual.
#' Founders (individuals whose dam and sire are both 0 or absent) return an
#' empty integer vector.
#'
#' @details
#' The function walks up the pedigree recursively: it retrieves the dam and
#' sire of \code{ind_id}, then recurses on each parent, accumulating all
#' ancestors. The result is de-duplicated with \code{unique()}.
#'
#' For large pedigrees this function can be slow due to R's recursion overhead.
#' Consider calling \code{\link{build_ancestor_list}} once and caching the
#' result rather than calling \code{find_ancestors} repeatedly.
#'
#' @param ind_id  Integer or numeric scalar; the ID of the individual whose
#'   ancestors are sought. A value of \code{0} is interpreted as "no parent"
#'   and returns \code{integer(0)}.
#' @param pedigree  Data frame with at least three columns:
#'   \describe{
#'     \item{\code{id}}{Individual ID.}
#'     \item{\code{dam}}{Dam (mother) ID; \code{0} if unknown.}
#'     \item{\code{sire}}{Sire (father) ID; \code{0} if unknown.}
#'   }
#'
#' @return Integer vector of row positions (indices into \code{pedigree}) of
#'   all ancestors of \code{ind_id}, or \code{integer(0)} for founders.
#'
#' @seealso \code{\link{build_ancestor_list}}, \code{\link{compute_spillover_X}}
#'
#' @examples
#' ped <- data.frame(
#'   id   = 1:5,
#'   dam  = c(0, 0, 1, 1, 3),
#'   sire = c(0, 0, 2, 2, 4)
#' )
#' find_ancestors(5L, ped)   # returns c(3, 4, 1, 2)
#' find_ancestors(1L, ped)   # returns integer(0)  -- founder
#'
#' @export
find_ancestors <- function(ind_id, pedigree) {
  if (is.null(ind_id) || length(ind_id) == 0L || ind_id == 0L)
    return(integer(0))

  row <- pedigree[pedigree$id == ind_id, ]
  if (nrow(row) == 0L) return(integer(0))

  dam_id  <- row$dam[1L]
  sire_id <- row$sire[1L]

  parents <- c(dam_id, sire_id)
  parents <- parents[!is.na(parents) & parents != 0L]

  unique(c(
    parents,
    find_ancestors(dam_id,  pedigree),
    find_ancestors(sire_id, pedigree)
  ))
}


#' Build the Full Ancestor List for a Pedigree
#'
#' Applies \code{\link{find_ancestors}} to every individual in a pedigree data
#' frame and returns a list suitable for passing directly to
#' \code{\link{compute_spillover_X}}.
#'
#' @details
#' The returned list uses \strong{row indices} (1 to \eqn{N}) as ancestor
#' references, not pedigree IDs. This is the format expected by
#' \code{\link{compute_spillover_X}} and \code{\link{heritage}}.
#'
#' Processing is done in pedigree order (row 1 to row \eqn{N}). A progress
#' message is printed every 200 individuals when \code{verbose = TRUE}.
#'
#' @param pedigree  Data frame with columns \code{id}, \code{dam}, and
#'   \code{sire} (as returned by \pkg{pedSimulate} after formatting, or any
#'   standard pedigree data frame). A value of \code{0} in \code{dam} or
#'   \code{sire} indicates an unknown parent.
#' @param verbose  Logical; print progress every 200 individuals.
#'   Default \code{TRUE}.
#'
#' @return A list of length \eqn{N} (one element per row of \code{pedigree}).
#'   Element \code{i} is an integer vector of \strong{row indices} of all
#'   ancestors of individual \code{i} in \code{pedigree}, or \code{integer(0)}
#'   for founders.
#'
#' @seealso \code{\link{find_ancestors}}, \code{\link{compute_spillover_X}}
#'
#' @examples
#' ped <- data.frame(
#'   id   = 1:5,
#'   dam  = c(0, 0, 1, 1, 3),
#'   sire = c(0, 0, 2, 2, 4)
#' )
#' anc <- build_ancestor_list(ped, verbose = FALSE)
#' anc[[5]]   # ancestors of individual 5
#' anc[[1]]   # integer(0) -- founder
#'
#' @export
build_ancestor_list <- function(pedigree, verbose = TRUE) {
  N         <- nrow(pedigree)
  ancestors <- vector("list", N)

  for (i in seq_len(N)) {
    if (verbose && i %% 200L == 0L)
      message(sprintf("  Building ancestor list: %d / %d", i, N))
    ancestors[[i]] <- find_ancestors(pedigree$id[i], pedigree)
  }

  if (verbose) message(sprintf("  Done: %d individuals processed.", N))
  ancestors
}


#' Compute the Kinship-Weighted Ancestral Spillover Covariates
#'
#' Builds the spillover matrix \eqn{\mathbf{s}_X} required by
#' \code{\link{heritage}}. For each individual \eqn{i},
#' \deqn{
#'   s_X^{(i)} = \frac{\displaystyle\sum_{j \in \mathcal{N}_i}
#'               \phi_{ij}\, x_j}
#'               {\displaystyle\sum_{j \in \mathcal{N}_i} \phi_{ij}},
#' }
#' where \eqn{\mathcal{N}_i := \mathrm{Anc}(i)} is the set of known ancestors
#' of \eqn{i} in the pedigree and \eqn{\phi_{ij} = K_{ij}} is the kinship
#' coefficient. Individuals with no recorded ancestors (founders) receive a
#' zero spillover row.
#'
#' @details
#' The kinship matrix \eqn{\mathbf{K}} is treated as fixed and exogenous.  It
#' can be computed from pedigree records using packages such as \pkg{kinship2},
#' or from genomic data using \pkg{GENESIS} (genomic relationship matrix).
#'
#' The \code{ancestors} list must respect the directed acyclic structure of the
#' pedigree: ancestors can influence descendants, but not the reverse.
#'
#' @param X         Numeric matrix of covariates (\eqn{N \times d}).
#' @param K         Numeric matrix; \eqn{N \times N} kinship matrix.
#'   \eqn{K_{ij}} is the kinship coefficient between individuals \eqn{i} and
#'   \eqn{j} (e.g. 0.25 for a parent-offspring pair).
#' @param ancestors A list of length \eqn{N}. Element \code{ancestors[[i]]}
#'   is an integer vector of row indices (in \code{X} and \code{K}) of all
#'   known ancestors of individual \eqn{i}. An empty vector indicates a
#'   founder.
#'
#' @return Numeric matrix (\eqn{N \times d}); kinship-weighted ancestral
#'   covariate averages. Row \eqn{i} is zero for founders.
#'
#' @seealso \code{\link{heritage}}, \code{\link{heritage_gs}}
#'
#' @references
#' Abdoul Oudouss Diakite, Anne-Marie Madore, Celia M. T. Greenwood, Catherine Laprise (2026).
#' HERITAGE: Hierarchical effects regression with interactions for trait
#' analysis in genetics. Manuscript in preparation.
#'
#' @examples
#' # 4 individuals, 3 covariates
#' # Individuals 3 and 4 have individuals 1 and 2 as parents
#' X <- matrix(c(0,1,2, 1,0,1, 1,1,0, 2,0,1), nrow = 4, byrow = TRUE)
#' K <- diag(4)  # simplified identity kinship
#' ancestors <- list(integer(0), integer(0), c(1L, 2L), c(1L, 2L))
#'
#' sX <- compute_spillover_X(X, K, ancestors)
#' sX
#'
#' @export
compute_spillover_X <- function(X, K, ancestors) {
  N  <- nrow(X)
  d  <- ncol(X)
  sX <- matrix(0, N, d)

  for (i in seq_len(N)) {
    anc_i <- ancestors[[i]]
    if (length(anc_i) > 0L) {
      w     <- K[i, anc_i]
      w_sum <- sum(w)
      if (w_sum > 0)
        sX[i, ] <- colSums(w * X[anc_i, , drop = FALSE]) / w_sum
    }
  }
  sX
}
