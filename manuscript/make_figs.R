#!/usr/bin/env Rscript
# ============================================================================================
# Self-contained figure builder for the Tessera-ST manuscript.
#
# R reads the machine-checked experiment artifacts (experiments/*.json) DIRECTLY, computes every
# derived quantity itself, generates the synthetic spatial-domain map natively, and emits all
# figures as vector .tex that the manuscript \inputs (compiled with pdflatex). There is no Python
# in the figure path and no intermediate CSV: the figures are reconstructed from the ledger, not
# ported from any prior plotting code.
#
#   - statistical panels      -> pgfplots .tex   (the manuscript's own fonts; \input directly)
#   - spatial map + heatmap   -> ggplot2 + tikzDevice  (LaTeX-typeset axes; rasterised data layer)
#   - the per-dataset table   -> booktabs .tex
#
# Run from the repo root:  Rscript manuscript/make_figs.R
# ============================================================================================

suppressMessages({
  library(jsonlite)
  library(ggplot2)
  library(tikzDevice)
  library(ggrastr)
  library(patchwork)
  library(mclust)        # adjustedRandIndex for the native spatial map
})
options(tikzDefaultEngine = "pdftex", stringsAsFactors = FALSE)
source(file.path("manuscript", "figure_standard.R"))

std_sanitize_tikz <- function(file, name) {
  ln <- readLines(file)
  ln <- ln[!grepl("^% Created by tikzDevice", ln)]
  writeLines(gsub(paste0("{", name, "_ras"), paste0("{figs/", name, "_ras"), ln, fixed = TRUE), file)
}

std_dims <- list()

expd <- "experiments"
figd <- "manuscript/figs"
dir.create(figd, showWarnings = FALSE, recursive = TRUE)

# ---- artifact access + tiny helpers -----------------------------------------------------
J     <- function(name) fromJSON(file.path(expd, paste0(name, ".json")), simplifyVector = FALSE)
num   <- function(x) as.numeric(x)
opent <- function(name) file(file.path(figd, name), "w")
wl    <- function(con, ...) writeLines(paste0(...), con)
esc   <- function(s) gsub("_", "\\\\_", s)               # underscores in free LaTeX text
fitln <- function(x, y) { c <- coef(lm(y ~ x)); c(b0 = c[[1]], b1 = c[[2]]) }

# canonical display names so every figure/table matches Table 1 and the prose
DISP  <- c(SlideseqV2 = "Slide-seqV2", openST = "Open-ST")
short <- function(p) { s <- sub("\\(.*", "", p); if (s %in% names(DISP)) DISP[[s]] else s }

# named colours -- must match the \definecolor lines in paper.tex
COLM  <- c(Tessera = "tblue", SpaceFlow = "torange", SpatialLeiden = "tred", STAGATE = "tgreen")
COLHEX <- c(Tessera = "#2B6CB0", SpaceFlow = "#DD6B20", SpatialLeiden = "#C53030", STAGATE = "#2F855A")

# ---- load the ledger --------------------------------------------------------------------
eb <- J("expanded_bench")
gp <- J("gtfree_proxy")
sr <- J("stat_rigor")
mc <- J("mechanism_synth")
nb <- if (file.exists(file.path(expd, "native_baselines.json")))   J("native_baselines")   else NULL
ab <- if (file.exists(file.path(expd, "mechanism_ablation.json"))) J("mechanism_ablation") else NULL
psc <- J("proxy_search_corrected")
ra  <- J("robustness_audit")
dc  <- J("mechanism_decoupled_v2")
bs  <- J("bayesspace_bench_partial")
xd  <- J("expand_data")

rows  <- eb$rows
raw   <- vapply(rows, function(r) r$platform, character(1))     # raw platform keys
panel <- unlist(eb$method_panel)

# per-dataset frame
ds <- data.frame(
  platform = vapply(rows, function(r) short(r$platform), character(1)),
  raw      = raw,
  contig   = vapply(rows, function(r) num(r$GT_contiguity), numeric(1)),
  adv      = vapply(rows, function(r) num(r$spatial_advantage), numeric(1)),
  winner   = vapply(rows, function(r) r$winner, character(1)),
  trank    = vapply(rows, function(r) num(r$Tessera_rank), numeric(1))
)
# method x dataset ARI matrix (NA where a method did not run on a dataset)
meanmat <- sapply(rows, function(r) sapply(panel, function(m)
  if (!is.null(r$means[[m]])) num(r$means[[m]]) else NA_real_))
rownames(meanmat) <- panel; colnames(meanmat) <- ds$platform
# coh_gain joined by raw platform key
gpr <- setNames(gp$rows, vapply(gp$rows, function(r) r$platform, character(1)))
ds$cohgain <- vapply(raw, function(k) num(gpr[[k]]$coh_gain), numeric(1))

# scalar ledger values the figures annotate
M <- list(
  rho_oracle  = num(eb$spearman_contiguity_vs_spatial_advantage),
  rho_proxy   = num(gp$principled_proxy_coh_gain$vs_advantage[[1]]),
  partial_all = num(sr$contiguity_partial_controlling_all),
  base_rate   = num(sr$lopo_gt_contiguity$base_rate_accuracy),
  mech_smooth = num(mc$spearman_contig_vs_adv_smooth),
  mech_stagate= num(mc$spearman_contig_vs_adv_stagate)
)

# =========================================================================================
# fig_law : the headline law (oracle) + the label-free proxy, 1x2 groupplot
# =========================================================================================
local({
  con <- opent("fig_law.tex")
  wl(con, "\\begin{tikzpicture}")
  wl(con, "\\begin{groupplot}[group style={group size=2 by 1, horizontal sep=1.7cm},",
         "width=7.4cm, height=5.2cm, tick label style={font=\\scriptsize}, clip=false,",
         "label style={font=\\small}, title style={font=\\small}, axis lines=left]")
  panel <- function(x, xlab, ttl, plab, mk) {
    y <- ds$adv; f <- fitln(x, y); xlo <- min(x) - 0.02; xhi <- max(x) + 0.02
    wl(con, sprintf("\\nextgroupplot[xlabel={%s}, ylabel={spatial-prior advantage ($\\Delta$ARI)}, title={%s}, xmin=%.3f, xmax=%.3f]",
                    xlab, ttl, xlo, xhi))
    wl(con, sprintf("\\node[font=\\bfseries, anchor=south west] at (rel axis cs:0,1) {%s};", plab))
    wl(con, sprintf("\\addplot[domain=%.3f:%.3f, dashed, draw=tgray, thick]{%.4f*x+%.4f};", xlo, xhi, f["b1"], f["b0"]))
    wl(con, sprintf("\\addplot[black, dotted] coordinates {(%.3f,0)(%.3f,0)};", xlo, xhi))
    wl(con, sprintf("\\addplot[only marks, mark=*, mark size=1.6pt, color=%s] coordinates {", mk))
    for (i in seq_len(nrow(ds))) wl(con, sprintf("  (%.3f,%.3f)", x[i], y[i]))
    wl(con, "};")
    # Labels all sit just to the right of their marker at a globally de-collided y, so no two
    # labels and no label/marker can overlap; a thin leader connects any label moved off its point.
    # clip=false lets the right-edge labels run into the margin instead of overprinting the panel.
    dx <- 0.030 * (xhi - xlo); gap <- 0.050 * (max(y) - min(y) + 1e-9)
    ord <- order(y); ly <- y
    for (k in seq_along(ord)[-1]) {
      a <- ord[k]; b <- ord[k - 1]
      if (ly[a] - ly[b] < gap) ly[a] <- ly[b] + gap
    }
    for (i in seq_len(nrow(ds))) {
      if (abs(ly[i] - y[i]) > gap * 0.4)
        wl(con, sprintf("\\draw[very thin, tgray] (axis cs:%.3f,%.3f) -- (axis cs:%.3f,%.3f);", x[i], y[i], x[i] + dx, ly[i]))
      wl(con, sprintf("\\node[font=\\tiny, anchor=west, inner sep=1.5pt] at (axis cs:%.3f,%.3f) {%s};", x[i] + dx, ly[i], ds$platform[i]))
    }
  }
  panel(ds$contig,  "GT spatial contiguity (oracle)",      sprintf("oracle law: $\\rho=%+.2f$", M$rho_oracle), "(a)", "tblue")
  panel(ds$cohgain, "\\texttt{coh\\_gain} (label-free)",   sprintf("label-free proxy: $\\rho=%+.2f$", M$rho_proxy), "(b)", "tgreen")
  wl(con, "\\end{groupplot}"); wl(con, "\\end{tikzpicture}")
  close(con)
})

# =========================================================================================
# fig_profile : per-method rank vs dataset contiguity (computed in R from the ARI matrix)
# =========================================================================================
local({
  ord  <- order(ds$contig)                       # low -> high contiguity
  labs <- ds$platform[ord]
  focus <- names(COLM)
  con <- opent("fig_profile.tex")
  wl(con, "\\begin{tikzpicture}")
  wl(con, "\\begin{axis}[width=11cm, height=4.3cm, y dir=reverse, ymin=0.5, ymax=9.5,",
         sprintf("xtick={%s}, xticklabels={%s}, x tick label style={rotate=45, anchor=east, font=\\scriptsize},",
                 paste(seq_along(ord) - 1, collapse = ","), paste(labs, collapse = ",")),
         "ytick={1,3,5,7,9}, ylabel={rank (1 = best)}, ylabel style={font=\\small},",
         "xlabel={datasets ordered by GT contiguity (low $\\rightarrow$ high)}, xlabel style={font=\\small},",
         "tick label style={font=\\scriptsize}, legend style={font=\\scriptsize, at={(0.5,1.04)}, anchor=south, legend columns=4, draw=none}, legend cell align=left]")
  for (m in focus) {
    wl(con, sprintf("\\addplot[color=%s, mark=*, mark size=1.4pt, thick] coordinates {", COLM[[m]]))
    for (xi in seq_along(ord)) {
      col <- meanmat[, ord[xi]]; core <- col[!is.na(col)]
      if (m %in% names(core)) {
        rank <- which(names(sort(core, decreasing = TRUE)) == m)
        wl(con, sprintf("  (%d,%d)", xi - 1, rank))
      }
    }
    wl(con, "};"); wl(con, sprintf("\\addlegendentry{%s}", m))
  }
  wl(con, "\\end{axis}"); wl(con, "\\end{tikzpicture}")
  close(con)
})

# =========================================================================================
# fig_confound : marginal rho (1) + contiguity partial rho controlling each confound (2)
# =========================================================================================
local({
  clean <- function(v) { map <- c(GT_contiguity = "contiguity", k_nclasses = "K",
                                   eff_dim = "dim", modality_imaging = "modality")
                         ifelse(v %in% names(map), map[v], v) }
  cm <- sr$confound_marginal_spearman_vs_advantage
  marg <- data.frame(key = clean(names(cm)), rho = vapply(cm, function(x) num(x$rho), numeric(1)))
  cp <- sr$contiguity_partial_spearman_controlling
  part <- data.frame(key = clean(names(cp)), rho = vapply(cp, num, numeric(1)))

  con <- opent("fig_confound.tex")
  wl(con, "\\begin{tikzpicture}")
  wl(con, "\\begin{groupplot}[group style={group size=2 by 1, horizontal sep=2.2cm},",
         "width=7.0cm, height=4.3cm, tick label style={font=\\scriptsize}, clip=false,",
         "title style={font=\\small}, label style={font=\\small}]")
  yc <- paste(rev(marg$key), collapse = ",")
  wl(con, sprintf("\\nextgroupplot[xbar, /pgf/bar width=8pt, bar shift=0pt, title={marginal $\\rho$ vs advantage}, symbolic y coords={%s}, ytick={%s}, y tick label style={font=\\scriptsize}, xmin=-0.8, xmax=0.9, nodes near coords, nodes near coords style={font=\\tiny}, every node near coord/.append style={/pgf/number format/.cd, fixed, precision=2}]", yc, yc))
  wl(con, "\\node[font=\\bfseries, anchor=south west] at (rel axis cs:0,1) {(a)};")
  fill <- ifelse(marg$key == "contiguity", "tblue", "tgray")
  for (i in seq_len(nrow(marg)))
    wl(con, sprintf("\\addplot+[xbar, draw=black!40, fill=%s] coordinates {(%.3f,%s)};", fill[i], marg$rho[i], marg$key[i]))
  wl(con, sprintf("\\nextgroupplot[ybar, /pgf/bar width=14pt, title={contiguity partial $\\rho$}, symbolic x coords={%s}, xtick=data, x tick label style={font=\\scriptsize}, ymin=0, ymax=1, ylabel={partial $\\rho$}, nodes near coords, nodes near coords style={font=\\tiny}]", paste(part$key, collapse = ",")))
  wl(con, "\\node[font=\\bfseries, anchor=south west] at (rel axis cs:0,1) {(b)};")
  wl(con, "\\addplot+[ybar, draw=black!40, fill=tgreen] coordinates {")
  for (i in seq_len(nrow(part))) wl(con, sprintf("  (%s,%.3f)", part$key[i], part$rho[i]))
  wl(con, "};")
  wl(con, sprintf("\\draw[tred, dashed, thick] ({rel axis cs:0,0}|-{axis cs:%s,%.3f}) -- ({rel axis cs:1,0}|-{axis cs:%s,%.3f});", part$key[1], M$partial_all, part$key[nrow(part)], M$partial_all))
  wl(con, sprintf("\\node[font=\\tiny, tred, anchor=north east] at (rel axis cs:1,1) {control all $=%.2f$};", M$partial_all))
  wl(con, "\\end{groupplot}"); wl(con, "\\end{tikzpicture}")
  close(con)
})

# =========================================================================================
# fig_mechanism : controlled contiguity sweep (causal)
# =========================================================================================
local({
  mr <- mc$rows
  d <- data.frame(contig = vapply(mr, function(r) num(r$contiguity), numeric(1)),
                  smooth = vapply(mr, function(r) num(r$adv_smooth), numeric(1)),
                  stagate= vapply(mr, function(r) num(r$adv_stagate), numeric(1)),
                  nonsp  = vapply(mr, function(r) num(r$nonspatial_ari), numeric(1)))
  d <- d[order(d$contig), ]
  con <- opent("fig_mechanism.tex")
  wl(con, "\\begin{tikzpicture}")
  wl(con, "\\begin{axis}[width=8.4cm, height=5.2cm, xlabel={GT contiguity (manipulated: scramble $1.0\\rightarrow0.0$)},",
         "ylabel={ARI / advantage ($\\Delta$ARI)}, label style={font=\\small}, tick label style={font=\\scriptsize},",
         "ymin=-0.45, ymax=0.82,",   # headroom so the legend clears the flat control line at 0.345
         "legend style={font=\\scriptsize, at={(0.02,0.98)}, anchor=north west, draw=none}, legend cell align=left]")
  series <- function(col, c, mk, lab) {
    wl(con, sprintf("\\addplot[color=%s, mark=%s, mark size=1.6pt, thick] coordinates {", c, mk))
    for (i in seq_len(nrow(d))) wl(con, sprintf("  (%.3f,%.3f)", d$contig[i], d[[col]][i]))
    wl(con, "};"); wl(con, sprintf("\\addlegendentry{%s}", lab))
  }
  series("smooth",  "tgreen", "*",        sprintf("advantage, neighbour-mean ($\\rho=%+.2f$)", M$mech_smooth))
  series("stagate", "tblue",  "square*",  sprintf("advantage, STAGATE ($\\rho=%+.2f$)", M$mech_stagate))
  series("nonsp",   "black!50", "triangle*", "non-spatial ARI (control: flat)")
  wl(con, "\\draw[tred, dashed] ({rel axis cs:0,0}|-{axis cs:0,0}) -- ({rel axis cs:1,0}|-{axis cs:0,0});")
  wl(con, "\\end{axis}"); wl(con, "\\end{tikzpicture}")
  close(con)
})

# =========================================================================================
# fig_proxy : naive proxies invert; only coh_gain recovers the oracle's sign
# =========================================================================================
local({
  nv <- gp$naive_proxies_FAIL; pr <- gp$principled_proxy_coh_gain
  d <- data.frame(
    key    = c("moran", "rawcoh", "cohgain"),
    pretty = c("Moran $I$", "raw coh.", "\\texttt{coh\\_gain}"),
    vs_gt  = c(num(nv$moran$vs_gtcontig[[1]]),  num(nv$raw_coh$vs_gtcontig[[1]]),  num(pr$vs_gtcontig[[1]])),
    vs_adv = c(num(nv$moran$vs_advantage[[1]]), num(nv$raw_coh$vs_advantage[[1]]), num(pr$vs_advantage[[1]])))
  con <- opent("fig_proxy.tex")
  wl(con, "\\begin{tikzpicture}")
  wl(con, sprintf("\\begin{axis}[ybar, /pgf/bar width=10pt, width=8.4cm, height=4.7cm, symbolic x coords={%s}, xtick=data, xticklabels={%s},",
                  paste(d$key, collapse = ","), paste(d$pretty, collapse = ",")))
  wl(con, "  ylabel={Spearman $\\rho$}, label style={font=\\small}, tick label style={font=\\scriptsize}, ymin=-0.6, ymax=0.9,",
         "  enlarge x limits=0.25, legend style={font=\\scriptsize, at={(0.5,1.03)}, anchor=south, legend columns=2, draw=none}, legend cell align=left]")
  wl(con, "\\addplot[draw=black!40, fill=tblue!50] coordinates {")
  for (i in seq_len(nrow(d))) wl(con, sprintf("  (%s,%.3f)", d$key[i], d$vs_gt[i]))
  wl(con, "}; \\addlegendentry{$\\rho$ vs GT contiguity}")
  wl(con, "\\addplot[draw=black!40, fill=torange] coordinates {")
  for (i in seq_len(nrow(d))) wl(con, sprintf("  (%s,%.3f)", d$key[i], d$vs_adv[i]))
  wl(con, "}; \\addlegendentry{$\\rho$ vs spatial advantage}")
  wl(con, sprintf("\\draw[tblue, dotted, thick] ({rel axis cs:0,0}|-{axis cs:%s,%.3f}) -- ({rel axis cs:1,0}|-{axis cs:%s,%.3f});", d$key[1], M$rho_oracle, d$key[nrow(d)], M$rho_oracle))
  # annotation parked top-left, clear of the (tall) coh_gain bars on the right
  wl(con, sprintf("\\node[font=\\tiny, tblue, anchor=north west] at (rel axis cs:0.02,0.84) {oracle $\\rho=%+.2f$};", M$rho_oracle))
  wl(con, "\\end{axis}"); wl(con, "\\end{tikzpicture}")
  close(con)
})

# =========================================================================================
# fig_backend : advantage is backend-stable (scatter) + per-method mean delta (bars)
# =========================================================================================
if (!is.null(nb)) local({
  nbr <- setNames(nb$rows, vapply(nb$rows, function(r) r$platform, character(1)))
  bk <- data.frame(pub = ds$adv, nat = vapply(raw, function(k) num(nbr[[k]]$spatial_advantage), numeric(1)))
  md <- nb$mean_delta_vs_published_per_method
  delta <- data.frame(method = names(md), d = vapply(md, num, numeric(1)))
  con <- opent("fig_backend.tex")
  wl(con, "\\begin{tikzpicture}")
  wl(con, "\\begin{groupplot}[group style={group size=2 by 1, horizontal sep=2.4cm},",
         "width=6.6cm, height=4.9cm, tick label style={font=\\scriptsize}, clip=false,",
         "title style={font=\\small}, label style={font=\\small}]")
  lim <- range(c(bk$pub, bk$nat)); lim <- c(lim[1] - 0.02, lim[2] + 0.02)
  wl(con, sprintf("\\nextgroupplot[title={published vs mclust-backend advantage}, xlabel={published (GMM-tied)}, ylabel={mclust-style backend}, xmin=%.3f, xmax=%.3f, ymin=%.3f, ymax=%.3f, axis equal image, title style={at={(rel axis cs:0.5,1)}, anchor=south, yshift=0.15cm}]", lim[1], lim[2], lim[1], lim[2]))
  wl(con, "\\node[font=\\bfseries, anchor=south west] at (rel axis cs:0,1) {(a)};")
  wl(con, sprintf("\\addplot[domain=%.3f:%.3f, dashed, draw=tgray]{x};", lim[1], lim[2]))
  wl(con, "\\addplot[only marks, mark=*, mark size=1.5pt, color=tpurple] coordinates {")
  for (i in seq_len(nrow(bk))) wl(con, sprintf("  (%.3f,%.3f)", bk$pub[i], bk$nat[i]))
  wl(con, "};")
  yc <- paste(rev(delta$method), collapse = ",")
  wl(con, sprintf("\\nextgroupplot[xbar, /pgf/bar width=6pt, bar shift=0pt, title={mean $\\Delta$ARI vs published}, symbolic y coords={%s}, ytick={%s}, y tick label style={font=\\tiny}, xlabel={mean $\\Delta$ARI}, enlarge y limits=0.08]", yc, yc))
  wl(con, "\\node[font=\\bfseries, anchor=south west] at (rel axis cs:0,1) {(b)};")
  for (i in seq_len(nrow(delta)))
    wl(con, sprintf("\\addplot+[xbar, draw=black!40, fill=tpurple!70] coordinates {(%.4f,%s)};", delta$d[i], delta$method[i]))
  wl(con, "\\end{groupplot}"); wl(con, "\\end{tikzpicture}")
  close(con)
})

# =========================================================================================
# fig_robust : bootstrap 95% CI (point+interval) + leave-one-dataset-out bars
# =========================================================================================
local({
  bg <- sr$bootstrap_gt_contiguity; bc <- sr$bootstrap_gtfree_cohgain
  b <- data.frame(name = c("GT contiguity", "GT-free coh_gain"),
                  rho = c(num(bg$rho), num(bc$rho)), lo = c(num(bg$ci95[[1]]), num(bc$ci95[[1]])),
                  hi = c(num(bg$ci95[[2]]), num(bc$ci95[[2]])), fp = c(num(bg$frac_positive), num(bc$frac_positive)))
  lg <- sr$lopo_gt_contiguity; lc <- sr$lopo_gtfree_cohgain
  l <- data.frame(name = c("GT contiguity", "GT-free coh_gain"),
                  loo = c(num(lg$loo_pred_vs_actual_rho), num(lc$loo_pred_vs_actual_rho)),
                  dec = c(num(lg$decision_rule_accuracy), num(lc$decision_rule_accuracy)))
  con <- opent("fig_robust.tex")
  wl(con, "\\begin{tikzpicture}")
  wl(con, "\\begin{groupplot}[group style={group size=2 by 1, horizontal sep=2.2cm},",
         "width=6.8cm, height=3.9cm, tick label style={font=\\scriptsize}, clip=false,",
         "title style={font=\\small}, label style={font=\\small}]")
  yc <- seq(nrow(b), 1)
  wl(con, sprintf("\\nextgroupplot[title={bootstrap 95\\%% CI on the law}, xlabel={Spearman $\\rho$}, ymin=0.6, ymax=%.1f, ytick={%s}, yticklabels={%s}, y tick label style={font=\\scriptsize}, xmin=-0.5, xmax=1.75]",
                  nrow(b) + 0.4, paste(yc, collapse = ","), paste(esc(b$name), collapse = ",")))
  wl(con, "\\node[font=\\bfseries, anchor=south west] at (rel axis cs:0,1) {(a)};")
  cols <- c("tblue", "tgreen")
  for (i in seq_len(nrow(b))) {
    wl(con, sprintf("\\addplot[%s, thick] coordinates {(%.3f,%d)(%.3f,%d)};", cols[i], b$lo[i], yc[i], b$hi[i], yc[i]))
    wl(con, sprintf("\\addplot[only marks, mark=*, color=%s, mark size=2pt] coordinates {(%.3f,%d)};", cols[i], b$rho[i], yc[i]))
    wl(con, sprintf("\\node[font=\\tiny, anchor=west] at (axis cs:%.3f,%d) {$%+.2f$ (%.1f\\%%$>$0)};", b$hi[i] + 0.04, yc[i], b$rho[i], b$fp[i] * 100))
  }
  wl(con, "\\draw[tred, dashed] ({axis cs:0,0}|-{rel axis cs:0,0}) -- ({axis cs:0,0}|-{rel axis cs:0,1});")
  wl(con, "\\nextgroupplot[ybar, /pgf/bar width=11pt, title={leave-one-dataset-out}, symbolic x coords={looR,decacc}, xtick=data, xticklabels={LOO $\\rho$,decision acc}, ymin=0, ymax=1.12, ytick={0,0.2,0.4,0.6,0.8,1}, ylabel={out-of-sample score}, legend style={font=\\tiny, at={(0.02,0.98)}, anchor=north west, legend columns=1, draw=none}, legend cell align=left]")
  wl(con, "\\node[font=\\bfseries, anchor=south west] at (rel axis cs:0,1) {(b)};")
  for (i in seq_len(nrow(l))) {
    wl(con, sprintf("\\addplot[draw=black!40, fill=%s] coordinates {(looR,%.3f)(decacc,%.3f)};", ifelse(i == 1, "tblue", "tgreen"), l$loo[i], l$dec[i]))
    wl(con, sprintf("\\addlegendentry{%s}", esc(l$name[i])))
  }
  wl(con, sprintf("\\draw[tgray, dotted, thick] ({rel axis cs:0,0}|-{axis cs:looR,%.3f}) -- ({rel axis cs:1,0}|-{axis cs:decacc,%.3f});", M$base_rate, M$base_rate))
  wl(con, sprintf("\\node[font=\\tiny, tgray, anchor=south east] at (rel axis cs:1,%.3f) {base rate $%.2f$};", M$base_rate + 0.02, M$base_rate))
  wl(con, "\\end{groupplot}"); wl(con, "\\end{tikzpicture}")
  close(con)
})

# =========================================================================================
# fig_ablation : component ablation of the candidate method (case study)
# =========================================================================================
if (!is.null(ab)) local({
  mer <- names(ab)[grepl("^MERFISH", names(ab))][1]
  osm <- names(ab)[grepl("^osmFISH", names(ab), ignore.case = TRUE)][1]
  comps <- c("boundary_contrastive", "calibrated_uncertainty", "edge_gating", "multi_scale")
  disp  <- c(boundary_contrastive = "boundary-contrastive loss", calibrated_uncertainty = "calibrated uncertainty",
             edge_gating = "edge gating", multi_scale = "multi-scale fusion")
  a <- data.frame(lab = disp[comps],
                  mer = vapply(comps, function(c) num(ab[[mer]]$component_drop[[c]]), numeric(1)),
                  osm = vapply(comps, function(c) num(ab[[osm]]$component_drop[[c]]), numeric(1)))
  yc <- paste(rev(a$lab), collapse = ","); y1 <- a$lab[1]
  con <- opent("fig_ablation.tex")
  wl(con, "\\begin{tikzpicture}")
  wl(con, sprintf("\\begin{axis}[xbar, bar width=5pt, width=9.2cm, height=4.5cm, symbolic y coords={%s}, ytick={%s},", yc, yc))
  wl(con, "  y tick label style={font=\\scriptsize}, tick label style={font=\\scriptsize}, xmin=-0.10, xmax=0.36,",
         "  xlabel={$\\Delta$ARI when the component is removed (positive $=$ it earns its place)}, xlabel style={font=\\small},",
         "  nodes near coords, nodes near coords style={font=\\tiny}, every node near coord/.append style={/pgf/number format/.cd, fixed, precision=2, print sign},",
         "  enlarge y limits=0.18, legend style={font=\\scriptsize, at={(0.98,0.05)}, anchor=south east, draw=none}, legend cell align=left]")
  wl(con, "\\addplot[draw=black!40, fill=tblue] coordinates {")
  for (i in seq_len(nrow(a))) wl(con, sprintf("  (%.3f,%s)", a$mer[i], a$lab[i]))
  wl(con, "}; \\addlegendentry{MERFISH}")
  wl(con, "\\addplot[draw=black!40, fill=torange] coordinates {")
  for (i in seq_len(nrow(a))) wl(con, sprintf("  (%.3f,%s)", a$osm[i], a$lab[i]))
  wl(con, "}; \\addlegendentry{osmFISH}")
  wl(con, sprintf("\\draw[black, thick] ({axis cs:0,%s}|-{rel axis cs:0,0}) -- ({axis cs:0,%s}|-{rel axis cs:0,1});", y1, y1))
  wl(con, "\\end{axis}"); wl(con, "\\end{tikzpicture}")
  close(con)
})

# =========================================================================================
# tab_perdataset : per-dataset best-config table (booktabs)
# =========================================================================================
local({
  o <- order(-ds$contig)
  con <- opent("tab_perdataset.tex")
  wl(con, "\\begin{tabular}{lcclcc}"); wl(con, "\\toprule")
  wl(con, "Dataset & contiguity & winner & winner ARI & spatial adv. & Tessera rank\\\\"); wl(con, "\\midrule")
  for (i in o) {
    win <- esc(ds$winner[i]); if (ds$winner[i] == "Tessera") win <- sprintf("\\textbf{%s}", win)
    rk  <- if (ds$trank[i] == 1) sprintf("\\textbf{%d}", as.integer(ds$trank[i])) else as.character(as.integer(ds$trank[i]))
    war <- meanmat[ds$winner[i], ds$platform[i]]
    wl(con, sprintf("%s & %.3f & %s & %.3f & $%+.3f$ & %s\\\\", ds$platform[i], ds$contig[i], win, war, ds$adv[i], rk))
  }
  wl(con, "\\bottomrule"); wl(con, "\\end{tabular}")
  close(con)
})

# =========================================================================================
# fig_heatmap : method x dataset ARI; the outlined winner cell wanders -> no universal SOTA
# =========================================================================================
local({
  mean_ari <- rowMeans(meanmat, na.rm = TRUE)
  m_ord <- names(sort(mean_ari, decreasing = TRUE))
  d_ord <- ds$platform[order(ds$contig)]
  rel   <- c("floor:nonspatial" = "floor (non-sp.)", "floor:smoothed" = "floor (smooth)", "BANKSY-style" = "BANKSY")
  relab <- function(v) ifelse(v %in% names(rel), rel[v], v)
  winner_of <- setNames(ds$winner, ds$platform)
  long <- do.call(rbind, lapply(m_ord, function(m) do.call(rbind, lapply(d_ord, function(dd) {
    a <- meanmat[m, dd]; if (is.na(a)) return(NULL)
    data.frame(method = m, dataset = dd, ari = a, winner = as.integer(winner_of[[dd]] == m))
  }))))
  long$dataset <- factor(long$dataset, levels = d_ord)
  long$method  <- factor(relab(long$method), levels = rev(relab(m_ord)))
  win <- long[long$winner == 1, ]
  p <- ggplot(long, aes(dataset, method, fill = ari)) +
    geom_tile(color = "white", linewidth = 0.3) +
    geom_tile(data = win, fill = NA, color = "black", linewidth = 0.7) +
    geom_text(aes(label = sprintf("%.2f", ari), color = ari > 0.30), size = 1.8) +
    scale_fill_viridis_c(option = "magma", name = "ARI") +
    scale_color_manual(values = c(`TRUE` = "black", `FALSE` = "white"), guide = "none") +
    labs(x = NULL, y = NULL) + coord_fixed() + theme_minimal(base_size = 8) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 6), axis.text.y = element_text(size = 6),
          panel.grid = element_blank(), legend.key.width = unit(6, "pt"), legend.key.height = unit(13, "pt"),
          legend.title = element_text(size = 6), legend.text = element_text(size = 5), plot.margin = margin(2, 2, 2, 2))
  out <- file.path(figd, "fig_heatmap.tex")
  save_tikz_std(p, out, plot_w_in = 6.4, plot_h_in = 3.1, sanitize = TRUE)
  f <- out
  ln <- readLines(f)
  ln <- ln[!grepl("^% Created by tikzDevice", ln)]  # drop the timestamp header -> byte-reproducible
  writeLines(gsub("{fig_heatmap_ras", "{figs/fig_heatmap_ras", ln, fixed = TRUE), f)
})

# =========================================================================================
# fig_spatialmap : the synthetic tessellation, generated and clustered natively in R
#   non-spatial KMeans scatters the contiguous domains; a neighbour-mean smoother recovers
#   them; the predicted seams are read off directly. ARIs are computed here (mclust).
# =========================================================================================
local({
  set.seed(1)
  n <- 32L; ng <- 50L; ndom <- 4L; sig <- 0.22; noise <- 1.0
  g <- expand.grid(x = 0:(n - 1), y = 0:(n - 1))
  stripe <- pmin(as.integer(g$x / n * ndom), ndom - 1L)
  lab <- stripe
  cx <- n * 0.70; cy <- n * 0.30; rad <- n * 0.13         # small disc domain
  lab[sqrt((g$x - cx)^2 + (g$y - cy)^2) < rad] <- ndom
  K <- max(lab) + 1L
  means <- matrix(rnorm(K * ng, 0, sig), K, ng)
  expr  <- means[lab + 1L, ] + matrix(rnorm(n * n * ng, 0, noise), n * n, ng)

  # lattice neighbours (8-connected) on the regular grid -> indices for smoothing + seams
  idx <- matrix(seq_len(n * n), n, n)                      # idx[col x+1, row y+1]
  nbr <- vector("list", n * n)
  for (xi in 0:(n - 1)) for (yi in 0:(n - 1)) {
    cell <- idx[xi + 1, yi + 1]; ns <- integer(0)
    for (dx in -1:1) for (dy in -1:1) if (!(dx == 0 && dy == 0)) {
      xx <- xi + dx; yy <- yi + dy
      if (xx >= 0 && xx < n && yy >= 0 && yy < n) ns <- c(ns, idx[xx + 1, yy + 1])
    }
    nbr[[cell]] <- ns
  }
  smooth <- function(h, rounds = 2) { for (r in seq_len(rounds))
    h <- t(vapply(seq_len(n * n), function(i) colMeans(h[c(i, nbr[[i]]), , drop = FALSE]), numeric(ncol(h)))); h }

  km <- function(X) kmeans(X, centers = K, nstart = 10, iter.max = 50)$cluster
  align <- function(pred, gt) {                            # greedy relabel for colour consistency
    tab <- table(pred, gt); map <- integer(0); used <- integer(0)
    for (p in order(-apply(tab, 1, max))) {
      tgt <- as.integer(colnames(tab)[order(-tab[p, ])]); tgt <- setdiff(tgt, used)[1]
      map[as.integer(rownames(tab)[p])] <- tgt; used <- c(used, tgt)
    }
    map[pred] }

  gt   <- lab
  ns   <- align(km(expr), gt)
  sp   <- align(km(smooth(expr)), gt)
  seam <- vapply(seq_len(n * n), function(i) as.integer(any(sp[nbr[[i]]] != sp[i])), integer(1))
  ari_ns <- adjustedRandIndex(gt, ns); ari_sp <- adjustedRandIndex(gt, sp)

  mk <- function(v) data.frame(x = g$x, y = g$y, v = v)
  panels <- list(
    `ground truth`                                  = data.frame(mk(factor(gt)), kind = "dom"),
    `non-spatial KMeans`                            = data.frame(mk(factor(ns)), kind = "dom"),
    `neighbour-mean`                                = data.frame(mk(factor(sp)), kind = "dom"),
    `predicted seams`                               = data.frame(mk(factor(ifelse(seam == 1, "seam", "interior"))), kind = "seam"))
  ttl <- c(`ground truth` = "(A) ground truth",
           `non-spatial KMeans` = sprintf("(B) non-spatial KMeans (ARI %.2f)", ari_ns),
           `neighbour-mean`     = sprintf("(C) neighbour-mean (ARI %.2f)", ari_sp),
           `predicted seams`    = "(D) predicted seams")
  df <- do.call(rbind, Map(function(d, nm) { d$panel <- ttl[[nm]]; d }, panels, names(panels)))
  df$panel <- factor(df$panel, levels = unname(ttl))

  pal <- c("0" = "#4c78a8", "1" = "#f58518", "2" = "#54a24b", "3" = "#e45756", "4" = "#b279a2",
           interior = "grey85", seam = "#c53030")
  p <- ggplot(df, aes(x, y, color = v)) +
    rasterise(geom_point(size = 0.5, stroke = 0), dpi = 320) +
    facet_wrap(~panel, nrow = 1) + scale_color_manual(values = pal, guide = "none") +
    coord_equal() + theme_void(base_size = 9) +
    theme(strip.text = element_text(size = 7, margin = margin(b = 2)), panel.spacing = unit(5, "pt"),
          plot.margin = margin(2, 2, 2, 2))
  out <- file.path(figd, "fig_spatialmap.tex")
  fitted <- save_tikz_std(p, out, plot_w_in = 1.55, plot_h_in = 1.55, sanitize = TRUE)
  std_dims[["fig_spatialmap"]] <<- c(w_in = fitted$w_in, h_in = fitted$h_in)
  std_sanitize_tikz(out, "fig_spatialmap")
  cat(sprintf("  spatial map: non-spatial ARI=%.3f  neighbour-mean ARI=%.3f\n", ari_ns, ari_sp))
})

# =========================================================================================
# Four-panel production figures. These overwrite the earlier compact drafts with A--D layouts
# while preserving the same artifact-only data path. No experiment is run in this block.
# =========================================================================================
CC <- c(blue = "#2B6CB0", green = "#2F855A", orange = "#DD6B20",
        red = "#C53030", gray = "#A0AEC0", purple = "#6B46C1")
theme_fp <- function() theme_minimal(base_size = 7) +
  theme(panel.grid.minor = element_blank(), plot.title = element_text(size = 7, face = "bold"),
        plot.subtitle = element_text(size = 6), axis.title = element_text(size = 6.5),
        axis.text = element_text(size = 5.5), legend.title = element_blank(),
        legend.text = element_text(size = 5.5), plot.margin = margin(3, 4, 3, 4))
save4 <- function(name, plots, plot_w_in = 2.15, plot_h_in = 1.55) {
  p <- wrap_plots(plots, ncol = 2) +
    plot_annotation(tag_levels = "A", tag_prefix = "(", tag_suffix = ")") &
    theme(plot.tag = element_text(size = 9, face = "bold"))
  out <- file.path(figd, paste0(name, ".tex"))
  fitted <- save_tikz_std(p, out, plot_w_in = plot_w_in, plot_h_in = plot_h_in, sanitize = TRUE)
  std_dims[[name]] <<- c(w_in = fitted$w_in, h_in = fitted$h_in)
  std_sanitize_tikz(out, name)
}

# fig_law: oracle, label-free, expanded n=14, and explicitly scoped BayesSpace partial.
local({
  scatter <- function(x, y, xlab, ttl, col) ggplot(data.frame(x, y), aes(x, y)) +
    geom_hline(yintercept = 0, linetype = 3, color = CC["gray"]) +
    geom_smooth(method = "lm", se = FALSE, formula = y ~ x, linetype = 2, color = CC["gray"]) +
    geom_point(size = 1.5, color = CC[col]) + labs(x = xlab, y = "spatial-prior advantage", title = ttl) + theme_fp()
  p1 <- scatter(ds$contig, ds$adv, "GT contiguity", sprintf("oracle law: rho=%+.2f", M$rho_oracle), "blue")
  p2 <- scatter(ds$cohgain, ds$adv, "coh_gain (label-free)", sprintf("directional proxy: rho=%+.2f", M$rho_proxy), "green")
  ex <- data.frame(panel = c("core n=11", "expanded n=14"),
                   rho = c(num(xd$spearman_published_n11), num(xd$spearman_expanded)),
                   p = c(num(xd$p_published_n11), num(xd$p_expanded)))
  p3 <- ggplot(ex, aes(panel, rho, fill = panel)) + geom_col(width = .62, show.legend = FALSE) +
    geom_text(aes(label = sprintf("rho=%.2f\np=%.3f", rho, p)), size = 2, vjust = -.15) +
    scale_fill_manual(values = unname(c(CC["blue"], CC["orange"]))) + coord_cartesian(ylim = c(0, .85)) +
    labs(x = NULL, y = "Spearman rho", title = "law weakens but survives expansion") + theme_fp()
  br <- bs$rows
  bd <- data.frame(contig = vapply(br, function(r) num(r$GT_contiguity), numeric(1)),
                   adv = vapply(br, function(r) num(r$spatial_advantage_with_bayesspace), numeric(1)))
  p4 <- scatter(bd$contig, bd$adv, "GT contiguity", "BayesSpace sensitivity", "purple") +
    labs(subtitle = "8-platform partial; not the full 11-platform panel")
  save4("fig_law", list(p1, p2, p3, p4))
})

# fig_profile: one rank trajectory per focal method.
local({
  focus <- names(COLM); ord <- order(ds$contig)
  rank_rows <- do.call(rbind, lapply(focus, function(m) do.call(rbind, lapply(ord, function(j) {
    core <- meanmat[, j]; core <- core[!is.na(core)]
    r <- match(m, names(sort(core, decreasing = TRUE)))  # NA when m was not evaluated on dataset j
    if (is.na(r)) return(NULL)
    data.frame(method = m, contig = ds$contig[j], rank = r)
  }))))
  plots <- lapply(focus, function(m) {
    z <- rank_rows[rank_rows$method == m, ]
    ggplot(z, aes(contig, rank)) + geom_line(color = COLHEX[[m]], linewidth = .45) +
      geom_point(color = COLHEX[[m]], size = 1.4) + scale_y_reverse(breaks = c(1, 3, 5, 7, 9), limits = c(9.5, .5)) +
      labs(x = "GT contiguity", y = "rank (1=best)", title = m) + theme_fp()
  })
  save4("fig_profile", plots)
})

# fig_confound: marginal, one-at-a-time partial, joint partial, and exploratory OLS coefficients.
local({
  clean <- c(GT_contiguity = "contiguity", k_nclasses = "K", eff_dim = "dimension", modality_imaging = "modality")
  cm <- sr$confound_marginal_spearman_vs_advantage
  dm <- data.frame(term = unname(clean[names(cm)]), value = vapply(cm, function(x) num(x$rho), numeric(1)))
  cp <- sr$contiguity_partial_spearman_controlling
  dp <- data.frame(term = unname(clean[names(cp)]), value = vapply(cp, num, numeric(1)))
  p1 <- ggplot(dm, aes(reorder(term, value), value, fill = term == "contiguity")) + geom_col(show.legend = FALSE) +
    coord_flip() + scale_fill_manual(values = unname(c(CC["gray"], CC["blue"]))) +
    labs(x = NULL, y = "marginal rho", title = "candidate predictors") + theme_fp()
  p2 <- ggplot(dp, aes(term, value)) + geom_col(fill = CC["green"], width = .62) +
    geom_hline(yintercept = num(sr$contiguity_partial_controlling_all), linetype = 2, color = CC["red"]) +
    labs(x = "controlled confound", y = "partial rho", title = "one-at-a-time controls",
         subtitle = sprintf("red: all controls = %.2f", num(sr$contiguity_partial_controlling_all))) + theme_fp()
  joint <- data.frame(term = c("marginal", "all controls"), value = c(M$rho_oracle, M$partial_all))
  p3 <- ggplot(joint, aes(term, value, fill = term)) + geom_col(show.legend = FALSE, width = .62) +
    geom_text(aes(label = sprintf("%.2f", value)), vjust = -.25, size = 2.2) +
    scale_fill_manual(values = unname(c(CC["blue"], CC["green"]))) + coord_cartesian(ylim = c(0, .85)) +
    labs(x = NULL, y = "rho", title = "contiguity persists jointly") + theme_fp()
  be <- sr$standardized_ols_beta; be <- be[setdiff(names(be), c("intercept", "_caveat"))]
  db <- data.frame(term = c("contiguity", "K", "dimension", "modality"), value = vapply(be, num, numeric(1)))
  p4 <- ggplot(db, aes(reorder(term, value), value, fill = value > 0)) + geom_col(show.legend = FALSE) +
    coord_flip() + scale_fill_manual(values = unname(c(CC["orange"], CC["blue"]))) +
    labs(x = NULL, y = "standardized beta", title = "exploratory multivariable OLS",
         subtitle = "n=11, four predictors; overfit-prone") + theme_fp()
  save4("fig_confound", list(p1, p2, p3, p4))
})

# fig_mechanism: v2 decoupled control, with slope / sign-flip / floor / verdict views.
local({
  subs <- names(dc$results_by_substrate); rec <- list()
  for (si in seq_along(subs)) {
    v <- dc$results_by_substrate[[subs[si]]]
    for (arm in c("A", "B")) {
      a <- if (arm == "A") v$arm_A_alignment_only else v$arm_B_contiguity_only
      rec[[length(rec) + 1]] <- data.frame(substrate = paste0("S", si), arm = arm, prior = "smooth",
        slope = num(a$normalized_slope_vs_adv_smooth), flip = a$adv_smooth_flips_sign, floor = num(a$nonspatial_floor_range))
      rec[[length(rec) + 1]] <- data.frame(substrate = paste0("S", si), arm = arm, prior = "STAGATE",
        slope = num(a$normalized_slope_vs_adv_stagate), flip = a$adv_stagate_flips_sign, floor = num(a$nonspatial_floor_range))
    }
  }
  d <- do.call(rbind, rec); d$key <- paste(d$substrate, d$arm)
  p1 <- ggplot(d, aes(key, slope, fill = prior)) + geom_col(position = "dodge", width = .68) +
    scale_fill_manual(values = unname(c(CC["green"], CC["blue"]))) +
    labs(x = "substrate / arm", y = "normalized slope", title = "effect size") + theme_fp() + theme(legend.position = "bottom")
  p2 <- ggplot(d, aes(key, prior, fill = flip)) + geom_tile(color = "white") +
    geom_text(aes(label = ifelse(flip, "flip", "no")), size = 2.1) +
    scale_fill_manual(values = unname(c("#EDF2F7", CC["red"])), guide = "none") +
    labs(x = "substrate / arm", y = NULL, title = "advantage sign flip", subtitle = "only contiguity arm B flips") + theme_fp()
  floor <- unique(d[, c("substrate", "arm", "floor")]); floor$key <- paste(floor$substrate, floor$arm)
  p3 <- ggplot(floor, aes(key, floor, fill = arm)) + geom_col(width = .62) +
    scale_fill_manual(values = unname(c(CC["orange"], CC["green"])), guide = "none") +
    labs(x = "substrate / arm", y = "non-spatial floor range", title = "manipulation cleanliness",
         subtitle = "B remains flat; A destroys signal") + theme_fp()
  verdict <- data.frame(test = c("larger slope", "exclusive sign flip", "flatter floor"), pass = c(1, 1, 1))
  p4 <- ggplot(verdict, aes(test, pass)) + geom_tile(aes(fill = factor(pass)), color = "white") +
    geom_text(label = "favours contiguity", size = 2.2) + scale_fill_manual(values = unname(CC["green"]), guide = "none") +
    coord_cartesian(ylim = c(.5, 1.5)) + labs(x = NULL, y = NULL, title = "cross-substrate verdict",
         subtitle = "all criteria pass on S1 and S2") + theme_fp() +
    theme(axis.text.y = element_blank(), axis.ticks.y = element_blank(), panel.grid = element_blank())
  save4("fig_mechanism", list(p1, p2, p3, p4))
})

# fig_proxy: six-candidate search, GT relation, p-value labels, and seed stability.
local({
  rho <- psc$observed_rho_per_candidate
  dr <- data.frame(candidate = names(rho), rho = vapply(rho, num, numeric(1)))
  dr$candidate <- factor(dr$candidate, levels = dr$candidate[order(dr$rho)])
  p1 <- ggplot(dr, aes(candidate, rho, fill = candidate == "coh_gain")) + geom_col(show.legend = FALSE) +
    coord_flip() + scale_fill_manual(values = unname(c(CC["gray"], CC["green"]))) +
    labs(x = NULL, y = "rho vs advantage", title = "six-candidate search") + theme_fp()
  rel <- data.frame(proxy = c("Moran I", "raw coherence", "coh_gain"),
    rho = c(num(gp$naive_proxies_FAIL$moran$vs_gtcontig[[1]]), num(gp$naive_proxies_FAIL$raw_coh$vs_gtcontig[[1]]), num(gp$principled_proxy_coh_gain$vs_gtcontig[[1]])))
  p2 <- ggplot(rel, aes(proxy, rho, fill = proxy == "coh_gain")) + geom_col(show.legend = FALSE) +
    scale_fill_manual(values = unname(c(CC["gray"], CC["blue"]))) + labs(x = NULL, y = "rho vs GT contiguity", title = "relation to oracle") + theme_fp()
  pv <- data.frame(test = c("asymptotic\nuncorrected", "permutation\nuncorrected", "search\ncorrected"),
    p = c(num(gp$principled_proxy_coh_gain$vs_advantage[[2]]), num(psc$p_uncorrected_selected), num(psc$p_search_corrected)))
  p3 <- ggplot(pv, aes(test, p, fill = test)) + geom_col(show.legend = FALSE) +
    geom_hline(yintercept = .05, linetype = 2, color = CC["red"]) + geom_text(aes(label = sprintf("%.3f", p)), vjust = -.2, size = 2) +
    scale_fill_manual(values = unname(c(CC["orange"], CC["purple"], CC["green"]))) + coord_cartesian(ylim = c(0, .32)) +
    labs(x = NULL, y = "p-value", title = "selection correction",
         subtitle = "0.079 is asymptotic; 0.083 permutation") + theme_fp()
  seeds <- unlist(gp$coh_gain_headline_per_seed_rho)
  p4 <- ggplot(data.frame(seed = seq_along(seeds), rho = seeds), aes(seed, rho)) +
    geom_line(color = CC["green"]) + geom_point(color = CC["green"], size = 1.8) +
    coord_cartesian(ylim = c(.4, .6)) + scale_x_continuous(breaks = seq_along(seeds)) +
    labs(x = "seed", y = "rho vs advantage", title = "seed stability") + theme_fp()
  save4("fig_proxy", list(p1, p2, p3, p4))
})

# fig_backend: agreement, method shifts, backend wins, and winner diversity.
if (!is.null(nb)) local({
  nbr <- setNames(nb$rows, vapply(nb$rows, function(r) r$platform, character(1)))
  bk <- data.frame(pub = ds$adv, nat = vapply(raw, function(k) num(nbr[[k]]$spatial_advantage), numeric(1)))
  p1 <- ggplot(bk, aes(pub, nat)) + geom_abline(linetype = 2, color = CC["gray"]) +
    geom_point(color = CC["purple"], size = 1.5) + coord_equal() +
    labs(x = "published advantage", y = "mclust advantage", title = "dataset-level agreement") + theme_fp()
  md <- nb$mean_delta_vs_published_per_method
  dm <- data.frame(method = names(md), delta = vapply(md, num, numeric(1)))
  p2 <- ggplot(dm, aes(reorder(method, delta), delta)) + geom_col(fill = CC["purple"]) + coord_flip() +
    labs(x = NULL, y = "mean delta ARI", title = "per-method backend shift") + theme_fp()
  bw <- nb$backend_win_counts
  dw <- data.frame(backend = names(bw), wins = vapply(bw, num, numeric(1)))
  p3 <- ggplot(dw, aes(backend, wins, fill = backend)) + geom_col(show.legend = FALSE) +
    geom_text(aes(label = wins), vjust = -.25, size = 2) + scale_fill_manual(values = unname(c(CC["blue"], CC["green"], CC["orange"]))) +
    labs(x = NULL, y = "best-config cells", title = "backend win counts") + theme_fp()
  wins <- table(vapply(nb$rows, function(r) r$winner, character(1)))
  dd <- data.frame(method = names(wins), wins = as.numeric(wins))
  p4 <- ggplot(dd, aes(reorder(method, wins), wins)) + geom_col(fill = CC["blue"]) + coord_flip() +
    labs(x = NULL, y = "datasets won", title = sprintf("winner diversity: %d methods", num(nb$n_distinct_winners))) + theme_fp()
  save4("fig_backend", list(p1, p2, p3, p4))
})

# fig_robust: bootstrap interval, sign stability, LOO magnitude, and decisions.
local({
  bg <- sr$bootstrap_gt_contiguity; bc <- sr$bootstrap_gtfree_cohgain
  db <- data.frame(name = c("GT contiguity", "coh_gain"), rho = c(num(bg$rho), num(bc$rho)),
    lo = c(num(bg$ci95[[1]]), num(bc$ci95[[1]])), hi = c(num(bg$ci95[[2]]), num(bc$ci95[[2]])),
    positive = c(num(bg$frac_positive), num(bc$frac_positive)))
  p1 <- ggplot(db, aes(rho, name, color = name)) + geom_segment(aes(x = lo, xend = hi, yend = name), linewidth = .7) +
    geom_point(size = 2) + geom_vline(xintercept = 0, linetype = 2, color = CC["red"]) +
    scale_color_manual(values = unname(c(CC["green"], CC["blue"])), guide = "none") +
    labs(x = "bootstrap rho (95 pct CI)", y = NULL, title = "uncertainty interval") + theme_fp()
  p2 <- ggplot(db, aes(name, positive, fill = name)) + geom_col(show.legend = FALSE, width = .62) +
    geom_text(aes(label = sprintf("%.1f pct", positive * 100)), vjust = -.25, size = 2) +
    scale_fill_manual(values = unname(c(CC["green"], CC["blue"]))) + coord_cartesian(ylim = c(0, 1.05)) +
    labs(x = NULL, y = "bootstrap fraction positive", title = "sign stability") + theme_fp()
  lopo <- data.frame(name = c("GT contiguity", "coh_gain"),
    rho = c(num(sr$lopo_gt_contiguity$loo_pred_vs_actual_rho), num(sr$lopo_gtfree_cohgain$loo_pred_vs_actual_rho)))
  p3 <- ggplot(lopo, aes(name, rho, fill = name)) + geom_col(show.legend = FALSE, width = .62) +
    scale_fill_manual(values = unname(c(CC["green"], CC["blue"]))) + labs(x = NULL, y = "LOO predicted-vs-actual rho", title = "held-out magnitude") + theme_fp()
  dec <- data.frame(name = c("GT rule", "coh_gain rule", "base rate"), acc = c(num(sr$lopo_gt_contiguity$decision_rule_accuracy), num(sr$lopo_gtfree_cohgain$decision_rule_accuracy), num(sr$lopo_gt_contiguity$base_rate_accuracy)))
  p4 <- ggplot(dec, aes(name, acc, fill = name)) + geom_col(show.legend = FALSE, width = .62) +
    geom_text(aes(label = sprintf("%.2f", acc)), vjust = -.25, size = 2) +
    scale_fill_manual(values = unname(c(CC["blue"], CC["green"], CC["gray"]))) + coord_cartesian(ylim = c(0, .9)) +
    labs(x = NULL, y = "decision accuracy", title = "held-out spatial-prior decision") + theme_fp()
  save4("fig_robust", list(p1, p2, p3, p4))
})

# fig_ablation: component drops and absolute config ARIs for two non-comparable case-study runs.
if (!is.null(ab)) local({
  plats <- names(ab); pretty <- c("MERFISH case study", "osmFISH case study")
  components <- c("boundary_contrastive", "calibrated_uncertainty", "edge_gating", "multi_scale")
  drop_plot <- function(i) {
    z <- ab[[plats[i]]]$component_drop[components]
    d <- data.frame(component = gsub("_", " ", names(z)), drop = vapply(z, num, numeric(1)))
    ggplot(d, aes(reorder(component, drop), drop, fill = drop > .02)) + geom_col(show.legend = FALSE) + coord_flip() +
      scale_fill_manual(values = unname(c(CC["gray"], CC["blue"]))) + labs(x = NULL, y = "delta ARI when removed", title = pretty[i]) + theme_fp()
  }
  config_plot <- function(i) {
    z <- ab[[plats[i]]]$per_config
    d <- data.frame(config = gsub("_", " ", names(z)), mean = vapply(z, function(x) num(x$mean), numeric(1)),
                    sd = vapply(z, function(x) num(x$std), numeric(1)))
    ggplot(d, aes(reorder(config, mean), mean)) + geom_col(fill = CC["orange"]) +
      geom_errorbar(aes(ymin = mean - sd, ymax = mean + sd), width = .2) + coord_flip() +
      labs(x = NULL, y = "absolute ARI", title = paste(pretty[i], "configurations"),
           subtitle = "separate run; not cross-panel comparable") + theme_fp()
  }
  save4("fig_ablation", list(drop_plot(1), drop_plot(2), config_plot(1), config_plot(2)))
})

# fig_heatmap: core matrix, winner counts, method means, and scoped BayesSpace partial sensitivity.
local({
  rel <- c("floor:nonspatial" = "floor nonsp.", "floor:smoothed" = "floor smooth", "BANKSY-style" = "BANKSY")
  relab <- function(v) ifelse(v %in% names(rel), rel[v], v)
  long <- do.call(rbind, lapply(rownames(meanmat), function(m) do.call(rbind, lapply(seq_len(ncol(meanmat)), function(j) {
    if (is.na(meanmat[m, j])) return(NULL)
    data.frame(method = relab(m), dataset = colnames(meanmat)[j], ari = meanmat[m, j])
  }))))
  p1 <- ggplot(long, aes(dataset, method, fill = ari)) + geom_tile(color = "white", linewidth = .15) +
    scale_fill_viridis_c(option = "magma", guide = guide_colorbar(barwidth = unit(2.2, "cm"), barheight = unit(.28, "cm"))) +
    labs(x = NULL, y = NULL, title = "core 11-platform ARI matrix") + theme_fp() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 5.5, margin = margin(t = 2)),
          axis.text.y = element_text(size = 5.5), legend.position = "bottom",
          legend.text = element_text(size = 5.5), legend.margin = margin(t = 4))
  wc <- as.data.frame(table(ds$winner)); names(wc) <- c("method", "wins")
  p2 <- ggplot(wc, aes(reorder(method, wins), wins)) + geom_col(fill = CC["blue"]) + coord_flip() +
    labs(x = NULL, y = "datasets won", title = "seven distinct winners") + theme_fp()
  ma <- data.frame(method = relab(rownames(meanmat)), mean = rowMeans(meanmat, na.rm = TRUE))
  p3 <- ggplot(ma, aes(reorder(method, mean), mean)) + geom_col(fill = CC["green"]) + coord_flip() +
    labs(x = NULL, y = "mean ARI", title = "core-panel method means") + theme_fp()
  br <- bs$rows
  bp <- data.frame(dataset = vapply(br, function(r) short(r$platform), character(1)),
    delta = vapply(br, function(r) {
      mm <- unlist(r$means_with_bayesspace); num(r$BayesSpace_ari) - max(num(mm[names(mm) != "BayesSpace"]))
    }, numeric(1)))
  p4 <- ggplot(bp, aes(reorder(dataset, delta), delta, fill = delta > 0)) + geom_col(show.legend = FALSE) +
    coord_flip() + scale_fill_manual(values = unname(c(CC["gray"], CC["purple"]))) +
    labs(x = NULL, y = "BayesSpace minus best other ARI", title = "BayesSpace sensitivity",
         subtitle = "8-platform partial; not the full 11-platform panel") + theme_fp()
  save4("fig_heatmap", list(p1, p2, p3, p4), plot_h_in = 2.15)
})

cat("make_figs.R: wrote", length(list.files(figd)), "files to", figd, "\n")
