# install_bayesspace.R --- reproducible install of the BayesSpace HMRF baseline used by
# experiments/bayesspace_bench.py (+ its worker experiments/_bayesspace.R).
#
# WHY THIS EXISTS (a plain `BiocManager::install("BayesSpace")` is NOT enough here)
#   On this machine's toolchain (system R 4.3.3 -> Bioconductor 3.18 -> BayesSpace 1.12.0) the naive
#   install FAILS at the C++ compile step:
#       error: *** C++14 compiler required; enable C++14 mode ... or use an earlier Armadillo
#   Cause: BiocManager pulls the current CRAN RcppArmadillo (>=15.x, Armadillo >=15 requires C++14+),
#   but BayesSpace 1.12.0's Makevars pins CXX_STD = CXX11, so it compiles with -std=gnu++11 and trips
#   Armadillo's compiler check. This is a BayesSpace/Armadillo version-skew issue, not a repo-code bug.
#
# FIX (contained, no persistent global change): make R compile the package's "C++11" request under
#   C++17 by overriding CXX11STD in a throwaway user-Makevars, wired in only for this session's installs
#   via R_MAKEVARS_USER. This satisfies Armadillo's C++14+ requirement and leaves ~/.R/Makevars untouched.
#
# USAGE:  Rscript experiments/install_bayesspace.R
#   Installs into the same R library the repo's other R baseline (mclust, via _mclust.R) already uses
#   -- the default user library on PATH `Rscript`. Idempotent: re-running is a no-op once both load.

options(repos = c(CRAN = "https://cloud.r-project.org"), timeout = 3600, Ncpus = max(1L, parallel::detectCores() %/% 2L))

need <- c("BayesSpace", "SingleCellExperiment")
have <- vapply(need, requireNamespace, logical(1), quietly = TRUE)
if (all(have)) {
  message("Already installed: ",
          paste(sprintf("%s v%s", need, vapply(need, function(p) as.character(packageVersion(p)), "")),
                collapse = ", "))
  quit(status = 0)
}

if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
message("R ", getRversion(), " | Bioconductor ", as.character(BiocManager::version()),
        " | RcppArmadillo ",
        if (requireNamespace("RcppArmadillo", quietly = TRUE)) as.character(packageVersion("RcppArmadillo")) else "not-yet")

# --- the C++17 override, scoped to this process only ---
mk <- tempfile("Makevars_cxx17_")
writeLines("CXX11STD = -std=gnu++17", mk)
Sys.setenv(R_MAKEVARS_USER = mk)   # inherited by the R CMD INSTALL child that BiocManager spawns
message("Forcing BayesSpace's C++11 build to compile as -std=gnu++17 via R_MAKEVARS_USER=", mk)

BiocManager::install(need[!have], update = FALSE, ask = FALSE)

ok <- vapply(need, requireNamespace, logical(1), quietly = TRUE)
for (i in seq_along(need))
  message(sprintf("LOAD %-22s %s%s", need[i], if (ok[i]) "OK v" else "FAIL",
                  if (ok[i]) as.character(packageVersion(need[i])) else ""))
if (!all(ok)) quit(status = 1)
message("BayesSpace install OK -- experiments/bayesspace_bench.py can now run.")
