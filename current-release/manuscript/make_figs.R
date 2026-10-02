#!/usr/bin/env Rscript
# Legacy vector-source and saved-result table renderer. Reads existing JSON
# ledgers without training tissue models or generating synthetic observations.
# Authoritative printed statistical PDFs: scripts/polish_stat_panels.py.
# Tissue-object PDFs: scripts/real_spatial_maps.py. The architecture schematic
# alone is compiled by scripts/compile_figures.py. See scripts/rebuild_current.sh.
# Run from the package root: Rscript manuscript/make_figs.R

suppressMessages({
  library(jsonlite)
  library(ggplot2)
  library(tikzDevice)
  library(ggrastr)
  library(patchwork)
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
    wl(con, sprintf("\\addplot[only marks, mark=*, mark size=2.4pt, color=%s] coordinates {", mk))
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
      wl(con, sprintf("\\node[font=\\scriptsize, anchor=west, inner sep=1.5pt] at (axis cs:%.3f,%.3f) {%s};", x[i] + dx, ly[i], ds$platform[i]))
    }
  }
  panel(ds$contig,  "GT spatial contiguity (oracle)",      sprintf("observational: $\\rho=%+.2f$", M$rho_oracle), "A", "tblue")
  panel(ds$cohgain, "\\texttt{coh\\_gain} (label-free)",   sprintf("label-free proxy: $\\rho=%+.2f$", M$rho_proxy), "B", "tgreen")
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
    wl(con, sprintf("\\addplot[color=%s, mark=*, mark size=2.2pt, thick] coordinates {", COLM[[m]]))
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
  wl(con, sprintf("\\nextgroupplot[xbar, /pgf/bar width=8pt, bar shift=0pt, title={marginal $\\rho$ vs advantage}, symbolic y coords={%s}, ytick={%s}, y tick label style={font=\\scriptsize}, xmin=-0.8, xmax=0.9, nodes near coords, nodes near coords style={font=\\scriptsize}, every node near coord/.append style={/pgf/number format/.cd, fixed, precision=2}]", yc, yc))
  wl(con, "\\node[font=\\bfseries, anchor=south west] at (rel axis cs:0,1) {A};")
  fill <- ifelse(marg$key == "contiguity", "tblue", "tgray")
  for (i in seq_len(nrow(marg)))
    wl(con, sprintf("\\addplot+[xbar, draw=black!40, fill=%s] coordinates {(%.3f,%s)};", fill[i], marg$rho[i], marg$key[i]))
  wl(con, sprintf("\\nextgroupplot[ybar, /pgf/bar width=14pt, title={contiguity partial $\\rho$}, symbolic x coords={%s}, xtick=data, x tick label style={font=\\scriptsize}, ymin=0, ymax=1, ylabel={partial $\\rho$}, nodes near coords, nodes near coords style={font=\\scriptsize}]", paste(part$key, collapse = ",")))
  wl(con, "\\node[font=\\bfseries, anchor=south west] at (rel axis cs:0,1) {B};")
  wl(con, "\\addplot+[ybar, draw=black!40, fill=tgreen] coordinates {")
  for (i in seq_len(nrow(part))) wl(con, sprintf("  (%s,%.3f)", part$key[i], part$rho[i]))
  wl(con, "};")
  wl(con, sprintf("\\draw[tred, dashed, thick] ({rel axis cs:0,0}|-{axis cs:%s,%.3f}) -- ({rel axis cs:1,0}|-{axis cs:%s,%.3f});", part$key[1], M$partial_all, part$key[nrow(part)], M$partial_all))
  wl(con, sprintf("\\node[font=\\scriptsize, tred, anchor=north east] at (rel axis cs:1,1) {control all $=%.2f$};", M$partial_all))
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
    wl(con, sprintf("\\addplot[color=%s, mark=%s, mark size=2.3pt, thick] coordinates {", c, mk))
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
  wl(con, sprintf("\\node[font=\\scriptsize, tblue, anchor=north west] at (rel axis cs:0.02,0.84) {oracle $\\rho=%+.2f$};", M$rho_oracle))
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
  wl(con, "\\node[font=\\bfseries, anchor=south west] at (rel axis cs:0,1) {A};")
  wl(con, sprintf("\\addplot[domain=%.3f:%.3f, dashed, draw=tgray]{x};", lim[1], lim[2]))
  wl(con, "\\addplot[only marks, mark=*, mark size=2.3pt, color=tpurple] coordinates {")
  for (i in seq_len(nrow(bk))) wl(con, sprintf("  (%.3f,%.3f)", bk$pub[i], bk$nat[i]))
  wl(con, "};")
  yc <- paste(rev(delta$method), collapse = ",")
  wl(con, sprintf("\\nextgroupplot[xbar, /pgf/bar width=6pt, bar shift=0pt, title={mean $\\Delta$ARI vs published}, symbolic y coords={%s}, ytick={%s}, y tick label style={font=\\scriptsize}, xlabel={mean $\\Delta$ARI}, enlarge y limits=0.08]", yc, yc))
  wl(con, "\\node[font=\\bfseries, anchor=south west] at (rel axis cs:0,1) {B};")
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
  wl(con, sprintf("\\nextgroupplot[title={bootstrap 95\\%% CI on the association}, xlabel={Spearman $\\rho$}, ymin=0.6, ymax=%.1f, ytick={%s}, yticklabels={%s}, y tick label style={font=\\scriptsize}, xmin=-0.5, xmax=1.75]",
                  nrow(b) + 0.4, paste(yc, collapse = ","), paste(esc(b$name), collapse = ",")))
  wl(con, "\\node[font=\\bfseries, anchor=south west] at (rel axis cs:0,1) {A};")
  cols <- c("tblue", "tgreen")
  for (i in seq_len(nrow(b))) {
    wl(con, sprintf("\\addplot[%s, thick] coordinates {(%.3f,%d)(%.3f,%d)};", cols[i], b$lo[i], yc[i], b$hi[i], yc[i]))
    wl(con, sprintf("\\addplot[only marks, mark=*, color=%s, mark size=2pt] coordinates {(%.3f,%d)};", cols[i], b$rho[i], yc[i]))
    wl(con, sprintf("\\node[font=\\scriptsize, anchor=west] at (axis cs:%.3f,%d) {$%+.2f$ (%.1f\\%%$>$0)};", b$hi[i] + 0.04, yc[i], b$rho[i], b$fp[i] * 100))
  }
  wl(con, "\\draw[tred, dashed] ({axis cs:0,0}|-{rel axis cs:0,0}) -- ({axis cs:0,0}|-{rel axis cs:0,1});")
  wl(con, "\\nextgroupplot[ybar, /pgf/bar width=11pt, title={leave-one-dataset-out}, symbolic x coords={looR,decacc}, xtick=data, xticklabels={LOO $\\rho$,decision acc}, ymin=0, ymax=1.12, ytick={0,0.2,0.4,0.6,0.8,1}, ylabel={out-of-sample score}, legend style={font=\\scriptsize, at={(0.02,0.98)}, anchor=north west, legend columns=1, draw=none}, legend cell align=left]")
  wl(con, "\\node[font=\\bfseries, anchor=south west] at (rel axis cs:0,1) {B};")
  for (i in seq_len(nrow(l))) {
    wl(con, sprintf("\\addplot[draw=black!40, fill=%s] coordinates {(looR,%.3f)(decacc,%.3f)};", ifelse(i == 1, "tblue", "tgreen"), l$loo[i], l$dec[i]))
    wl(con, sprintf("\\addlegendentry{%s}", esc(l$name[i])))
  }
  wl(con, sprintf("\\draw[tgray, dotted, thick] ({rel axis cs:0,0}|-{axis cs:looR,%.3f}) -- ({rel axis cs:1,0}|-{axis cs:decacc,%.3f});", M$base_rate, M$base_rate))
  wl(con, sprintf("\\node[font=\\scriptsize, tgray, anchor=south east] at (rel axis cs:1,%.3f) {base rate $%.2f$};", M$base_rate + 0.02, M$base_rate))
  wl(con, "\\end{groupplot}"); wl(con, "\\end{tikzpicture}")
  close(con)
})

# =========================================================================================
# fig_ablation : component ablation of the candidate method (case study)
# =========================================================================================
if (!is.null(ab)) local({
  mer <- names(ab)[grepl("^MERFISH", names(ab))][1]
  osm <- names(ab)[grepl("^osmFISH", names(ab), ignore.case = TRUE)][1]
  comps <- c("boundary_contrastive", "edge_gating", "multi_scale")
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
         "  nodes near coords, nodes near coords style={font=\\scriptsize}, every node near coord/.append style={/pgf/number format/.cd, fixed, precision=2, print sign},",
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
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 6), axis.text.y = element_text(size = 7),
          panel.grid = element_blank(), legend.key.width = unit(6, "pt"), legend.key.height = unit(13, "pt"),
          legend.title = element_text(size = 7), legend.text = element_text(size = 5), plot.margin = margin(2, 2, 2, 2))
  out <- file.path(figd, "fig_heatmap.tex")
  save_tikz_std(p, out, plot_w_in = 6.4, plot_h_in = 3.1, sanitize = TRUE)
  f <- out
  ln <- readLines(f)
  ln <- ln[!grepl("^% Created by tikzDevice", ln)]  # drop the timestamp header -> byte-reproducible
  writeLines(gsub("{fig_heatmap_ras", "{figs/fig_heatmap_ras", ln, fixed = TRUE), f)
})

# Tissue maps are rendered only from saved observations by real_spatial_maps.py.
# The former unused synthetic map / kmeans rerun is removed from this renderer.

# =========================================================================================
# Compact journal figures above are the authority. Figure 1 is replaced by
# scripts/real_spatial_maps.py after this statistical rendering step.
cat("Compact journal figures regenerated from bundled experiment ledgers.\n")
