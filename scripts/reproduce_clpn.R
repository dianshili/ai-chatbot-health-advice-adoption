# Reproduce the directed network from scored inputs. No measurement code.
suppressPackageStartupMessages({
  library(glmnet)
  library(jsonlite)
})
args <- commandArgs(trailingOnly = TRUE)
script_arg <- commandArgs()[grepl("^--file=", commandArgs())][1]
root <- dirname(dirname(normalizePath(sub("^--file=", "", script_arg))))
out <- file.path(root, "results", "clpn")
n_boot <- 0L
if ("--output" %in% args) out <- args[match("--output", args) + 1L]
if ("--bootstraps" %in% args) n_boot <- as.integer(args[match("--bootstraps", args) + 1L])
dir.create(out, recursive = TRUE, showWarnings = FALSE)
cfg <- jsonlite::fromJSON(file.path(root, "config", "analysis.json"))
# The original run used this generator, not R's default Mersenne-Twister.
# A seed alone does not reproduce the original cross-validation folds.
RNGkind("L'Ecuyer-CMRG")
dat <- read.csv(file.path(root, "data", "analysis_scores.csv"), check.names = FALSE)
nodes <- read.csv(file.path(root, "data", "network_nodes.csv"), check.names = FALSE)
stopifnot(nrow(dat) == 2149L, !anyDuplicated(dat$release_id))

# Model preprocessing starts here, after questionnaire scoring.
X_node <- scale(as.matrix(dat[, nodes$wave1_column, drop = FALSE]))
Y_node <- scale(as.matrix(dat[, nodes$wave2_column, drop = FALSE]))
stopifnot(all(is.finite(X_node)), all(is.finite(Y_node)))
num <- dat[, cfg$network_controls$numeric, drop = FALSE]
for (v in names(num)) {
  missing <- is.na(num[[v]])
  if (any(missing)) {
    num[[paste0(v, "_missing")]] <- as.numeric(missing)
    num[[v]][missing] <- median(num[[v]], na.rm = TRUE)
  }
}
binary <- dat[, cfg$network_controls$binary, drop = FALSE]
for (v in names(binary)) {
  missing <- is.na(binary[[v]])
  if (any(missing)) {
    binary[[paste0(v, "_missing")]] <- as.numeric(missing)
    binary[[v]][missing] <- 0
  }
}
fac <- dat[, cfg$network_controls$categorical, drop = FALSE]
fac[] <- lapply(fac, function(x) {
  x <- as.character(x)
  x[is.na(x) | x == ""] <- "Missing"
  factor(x)
})
fac_mm <- model.matrix(~ ., data = fac)
fac_mm <- fac_mm[, colnames(fac_mm) != "(Intercept)", drop = FALSE]
controls <- cbind(as.matrix(num), as.matrix(binary), fac_mm)
controls <- apply(controls, 2, as.numeric)
sds <- apply(controls, 2, sd)
controls <- scale(controls[, is.finite(sds) & sds > 0, drop = FALSE])
controls[!is.finite(controls)] <- 0
p <- ncol(X_node)
q <- ncol(controls)
stopifnot(p == 30L, q == 61L)
network_input <- cbind(X_node, controls, Y_node)
write.csv(network_input, file.path(out, "standardized_model_input.csv"), row.names = FALSE)

estimate_clpn <- function(data, foldid = NULL, ...) {
  data <- as.matrix(data)
  X <- data[, seq_len(p), drop = FALSE]
  C <- data[, p + seq_len(q), drop = FALSE]
  Y <- data[, p + q + seq_len(p), drop = FALSE]
  design <- cbind(X, C)
  penalty <- c(rep(1, p), rep(0, q))
  if (is.null(foldid)) foldid <- sample(rep(seq_len(10), length.out = nrow(design)))
  coef <- matrix(0, p, p, dimnames = list(nodes$node, nodes$node))
  lambda <- numeric(p)
  for (j in seq_len(p)) {
    fit <- glmnet::cv.glmnet(x = design, y = Y[, j], family = "gaussian",
      alpha = 1, nfolds = 10, foldid = foldid, penalty.factor = penalty,
      standardize = FALSE, intercept = TRUE, type.measure = "deviance")
    lambda[j] <- fit$lambda.min
    coef[, j] <- as.numeric(stats::coef(fit, s = fit$lambda.min))[1L + seq_len(p)]
  }
  list(graph = coef, lambda = lambda)
}

cat("Estimating main cross-lagged network\n")
set.seed(20260627L)
foldid <- sample(rep(seq_len(10), length.out = nrow(network_input)))
main <- estimate_clpn(network_input, foldid = foldid)
write.csv(main$graph, file.path(out, "network_coefficients_with_ar.csv"), row.names = TRUE)
write.csv(data.frame(node = nodes$node, lambda = main$lambda), file.path(out, "selected_lambda.csv"), row.names = FALSE)
G <- main$graph
diag(G) <- 0
write.csv(G, file.path(out, "network_coefficients_without_ar.csv"), row.names = TRUE)
bridge_out <- vapply(seq_len(p), function(i) sum(G[i, nodes$domain != nodes$domain[i]]), numeric(1))
centrality <- data.frame(node = nodes$node, label = nodes$label, domain = nodes$domain,
  out_expected_influence = rowSums(G), in_expected_influence = colSums(G),
  bridge_expected_influence_1step = bridge_out)
write.csv(centrality, file.path(out, "network_centrality.csv"), row.names = FALSE)
benchmark <- read.csv(file.path(root, "benchmarks", "network_coefficients.csv"), row.names = 1, check.names = FALSE)
benchmark <- as.matrix(benchmark[nodes$node, nodes$node])
error <- max(abs(main$graph - benchmark))
checks <- list(n = nrow(network_input), nodes_per_wave = p, encoded_covariates = q,
  max_abs_coefficient_difference = error, coefficients_match_1e_8 = error <= 1e-8,
  nonzero_cross_lagged_edges = sum(G != 0), bootstrap_replicates_each = n_boot)
old_cent <- read.csv(file.path(root, "benchmarks", "network_centrality.csv"), check.names = FALSE)
old_cent <- old_cent[match(nodes$node, old_cent$Node), ]
checks$max_abs_out_ei_difference <- max(abs(centrality$out_expected_influence - old_cent$Out_Expected_Influence))
checks$max_abs_in_ei_difference <- max(abs(centrality$in_expected_influence - old_cent$In_Expected_Influence))
checks$max_abs_bridge_ei_difference <- max(abs(centrality$bridge_expected_influence_1step - old_cent$Bridge_Expected_Influence_1step))
cat("Main network maximum absolute coefficient difference", format(error, digits = 10), "\n")
jsonlite::write_json(checks, file.path(out, "verification.json"), pretty = TRUE, auto_unbox = TRUE, digits = NA)

if (n_boot > 0L) {
  if (!requireNamespace("bootnet", quietly = TRUE)) stop("Install the documented bootnet package before requesting bootstrap execution.")
  community <- match(nodes$domain, c("Outcome", "User", "Agent", "Interaction", "Context"))
  set.seed(20260627L)
  net <- bootnet::estimateNetwork(data = as.data.frame(network_input), default = "none",
    fun = function(data, ...) estimate_clpn(data)$graph, labels = nodes$node, directed = TRUE)
  statistics <- c("edge", "outExpectedInfluence", "inExpectedInfluence", "bridgeExpectedInfluence")
  set.seed(202606271L)
  np <- bootnet::bootnet(net, type = "nonparametric", nBoots = n_boot,
    statistics = statistics, communities = community, directed = TRUE,
    includeDiagonal = FALSE, memorysaver = TRUE, nCores = 1,
    caseMin = 0.10, caseMax = 0.75)
  set.seed(202606272L)
  cases <- bootnet::bootnet(net, type = "case", nBoots = n_boot,
    statistics = statistics, communities = community, directed = TRUE,
    includeDiagonal = FALSE, memorysaver = TRUE, nCores = 1,
    caseMin = 0.10, caseMax = 0.75)
  write.csv(summary(np, statistics = "edge"), file.path(out, "bootstrap_edge_summary.csv"), row.names = FALSE)
  checks$bootstrap_execution_completed <- TRUE
  checks$bootstrap_scope <- if (n_boot < 1000L) "Execution smoke test only; not manuscript inferential results." else "Requested original replicate count; compare regenerated bootstrap summaries separately."
  if (n_boot >= 1000L) {
    cs <- bootnet::corStability(cases, statistics = statistics)
    write.csv(data.frame(statistic = names(cs), cs_coefficient = as.numeric(cs)), file.path(out, "bootstrap_case_stability.csv"), row.names = FALSE)
  }
  jsonlite::write_json(checks, file.path(out, "verification.json"), pretty = TRUE, auto_unbox = TRUE, digits = NA)
}
capture.output(sessionInfo(), file = file.path(out, "session_info.txt"))
print(checks)
