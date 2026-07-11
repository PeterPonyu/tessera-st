# _bayesspace.R --- BayesSpace (Bayesian t-error + Potts/MRF spatial clustering, i.e. the HMRF
# method family) worker, called by bayesspace_bench.py via `Rscript` + a CSV round-trip. This is the
# SAME invocation pattern as experiments/_mclust.R (rpy2 does not build on this Python 3.13 env, so we
# shell out and exchange data through temp CSVs). NOTE: mclust is only an R *backend* in this repo;
# BayesSpace is the first R-based *method*, so it produces cluster labels directly (like SpatialLeiden),
# not an embedding to be re-clustered.
#
# STAGED --- NOT yet executed. Requires:  BiocManager::install(c("BayesSpace","SingleCellExperiment"))
#
# args (all positional):  emb.csv  coords.csv  k  out.txt  seed  platform  nrep
#   emb.csv     n x d PCA embedding, header row  (the panel's shared latent; caller caps d<=15,
#               BayesSpace's default working dimensionality)
#   coords.csv  n x 2 spatial coordinates (x,y), header row
#   k           number of spatial domains (BayesSpace `q`)
#   out.txt     worker writes n integer labels (1..k), one per line (read back by the Python driver)
#   seed        RNG seed
#   platform    "Visium" | "ST"  --- selects BayesSpace's MRF neighbour lattice (hex vs square). The
#               caller passes the real platform for genuine Visium data and "ST" (square lattice) for
#               everything else; see the neighbour-structure CAVEAT in bayesspace_bench.py.
#   nrep        MCMC iterations (BayesSpace default 50000; caller may lower for a first sweep)
#
# Returns non-zero / writes nothing on failure, so the Python driver drops BayesSpace for that platform
# exactly like any other method that fails on a platform (fail-soft, logged).

suppressMessages({
  library(BayesSpace)
  library(SingleCellExperiment)
})

a        <- commandArgs(trailingOnly = TRUE)
emb      <- as.matrix(read.csv(a[1]))
coords   <- as.matrix(read.csv(a[2]))
k        <- as.integer(a[3])
outfile  <- a[4]
seed     <- as.integer(a[5])
platform <- a[6]
nrep     <- as.integer(a[7])
set.seed(seed)

# BayesSpace clusters over `reducedDim(sce,"PCA")` using an MRF neighbour graph it derives from integer
# (row,col) lattice indices in colData. Our harness carries no native array indices (generic h5ad ->
# float x,y), so we SNAP coordinates onto an integer grid to give BayesSpace a lattice to work on. This
# is an APPROXIMATION of BayesSpace's intended Visium/ST lattice and is the #1 documented caveat in
# bayesspace_bench.py: a genuine Visium run would use the real array row/col instead. Grid resolution
# ~sqrt(2n) per axis keeps most spots in distinct cells while preserving spatial adjacency.
n   <- nrow(coords)
gr  <- as.integer(ceiling(sqrt(2 * n)))
col <- as.integer(cut(coords[, 1], breaks = gr, labels = FALSE, include.lowest = TRUE))
row <- as.integer(cut(coords[, 2], breaks = gr, labels = FALSE, include.lowest = TRUE))

sce <- SingleCellExperiment(
  assays  = list(logcounts = t(emb)),          # placeholder assay; clustering uses the PCA below
  colData = DataFrame(row = row, col = col))
reducedDim(sce, "PCA") <- emb

# model="t" + the Potts MRF prior IS the Bayesian/HMRF family the panel currently lacks. init via mclust
# (BayesSpace's own default) so the MCMC starts from a sensible partition. gamma is the MRF smoothing
# strength (BayesSpace defaults: 3 for Visium hex, 2 for ST square).
res <- tryCatch(
  spatialCluster(sce, q = k, d = ncol(emb), platform = platform,
                 init.method = "mclust", model = "t",
                 gamma = ifelse(platform == "Visium", 3, 2),
                 nrep = nrep, burn.in = as.integer(nrep * 0.2)),
  error = function(e) { message("BayesSpace fail: ", conditionMessage(e)); NULL })

if (is.null(res)) quit(status = 1)
write.table(as.integer(res$spatial.cluster), outfile, row.names = FALSE, col.names = FALSE)
