#!/usr/bin/env python3
"""Render new manuscript objects from verified field-identified seed records.

This only aggregates recorded predictions; it does not fit or select models.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

MAN = Path(__file__).resolve().parents[1]
ROOT = MAN.parent
RUN = ROOT/"experiments/generated/field-identified-fixed-backend-20260923-v1"
LEDGER = MAN/"ledgers/revision_20260923"
COLORS = ["#0072B2", "#009E73", "#D55E00", "#7851B8", "#555555", "#B04778"]
font_manager.findfont(font_manager.FontProperties(family="Arial"), fallback_to_default=False)
# IEEEtran scales a 7.16-inch master slightly below its 516-TeX-pt text width.
# A 7.6-pt source floor remains above 7.5 pt after that embedding transform.
plt.rcParams.update({"font.family":"Arial", "font.size":8.3, "pdf.fonttype":42,
    "axes.spines.top":False,"axes.spines.right":False,"axes.labelsize":8.3,
    "xtick.labelsize":7.6,"ytick.labelsize":7.6,"legend.fontsize":7.6})


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(fig,name):
    fig.savefig(MAN/"figs"/f"{name}.pdf",metadata={"CreationDate":None,"ModDate":None})
    fig.savefig(MAN/"figs"/f"{name}.png",dpi=240)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--from-run',action='store_true',help='Import already-verified run ledgers; never fit models')
    parser.add_argument('--figures-only',action='store_true',help='Render only the three figures; do not write result TeX or ledgers')
    args=parser.parse_args()
    if args.figures_only and args.from_run:
        parser.error('--figures-only uses the committed ledgers, not --from-run')
    source_root=RUN if args.from_run else LEDGER
    summary=json.loads((source_root/"summary.json").read_text())
    qa=json.loads((source_root/"QA.json").read_text())
    if qa["failed_runs"] or qa["completed_method_seeds"]!=147:
        raise RuntimeError("This manuscript table requires the complete retained seven-method panel")
    if not args.figures_only:
        LEDGER.mkdir(parents=True,exist_ok=True)
    if args.from_run:
        for filename in ["summary.json","QA.json","protocol.json"]:
            shutil.copy2(RUN/filename,LEDGER/filename)
    rows=summary["rows"]
    methods=["expression-only","neighbor-mean","BANKSY-style","SpatialLeiden","Tessera","STAGATE","SEDR"]
    names=[r["dataset"] for r in rows]
    byname={r["dataset"]:r for r in rows}
    # Leiden's fixed search does not always attain the reference class count.
    # Keep the actual outputs visible beside its ARI rather than describing
    # every method as exactly K-conditioned in the printed comparison.
    codex_leiden = []
    for seed in (1, 2, 3):
        record=json.loads((RUN/"CODEX"/f"SpatialLeiden-seed{seed}.json").read_text())
        if record["status"] != "complete" or record["n_classes"] != byname["CODEX"]["n_classes"]:
            raise ValueError(f"CODEX Leiden record is not comparable at seed {seed}")
        codex_leiden.append(record["fitted_n_clusters"])
    if codex_leiden != [51, 51, 54]:
        raise ValueError(f"Unexpected CODEX Leiden output cluster counts: {codex_leiden}")
    source={"protocol_id":summary["protocol_id"],"renderer_sha256":sha(Path(__file__)),
            "input_sha256":{f:sha(LEDGER/f) for f in ["summary.json","QA.json","protocol.json","estimand-audit.json"]},
            "figures":["fig_field_comparison","fig_field_controls","fig_rank_audit"],
            "historical_panels_not_replaced":["n11","n12","n14","native backend","proxy search","ablation"]}
    # Fixed method comparisons: SD over seed-paired deltas, never a tissue CI.
    fig=plt.figure(figsize=(7.16,3.2))
    ax=fig.add_axes([.10,.23,.88,.56])
    positions=np.arange(len(rows))
    for i,m in enumerate(methods[1:]):
        delta=np.array([np.array(r["methods"][m]["raw_per_seed"])-np.array(r["methods"]["expression-only"]["raw_per_seed"]) for r in rows])
        ax.errorbar(positions+(i-2.5)*.115,delta.mean(1),yerr=delta.std(1,ddof=1),
            fmt="o",ms=3,color=COLORS[i],capsize=1.5,label=m)
    ax.axhline(0,color="#777777",lw=.8)
    ax.set_xticks(positions,names,rotation=0)
    ax.set_ylabel("Raw ARI minus expression-only ARI")
    ax.set_xlim(-.6,6.6)
    fig.text(.54,.97,"Field-identified named-method contrasts",ha="center",va="top",fontsize=9)
    fig.text(.54,.91,"Fixed KMeans for embedding methods; SpatialLeiden communities",ha="center",va="top",fontsize=7.8)
    fig.text(.54,.855,"No backend or refinement selection",ha="center",va="top",fontsize=7.6)
    fig.legend(*ax.get_legend_handles_labels(),loc="lower center",bbox_to_anchor=(.53,.005),
               ncol=3,frameon=False,columnspacing=1.8,labelspacing=.35)
    save(fig,"fig_field_comparison")
    # Two direct protocol consequences, not a comparison of all old/new models.
    fig=plt.figure(figsize=(7.16,2.9))
    left=fig.add_axes([.10,.28,.35,.56]);right=fig.add_axes([.60,.28,.37,.56])
    for i,n in enumerate(["MIBI","CODEX"]):
        r=byname[n]; m=r["methods"]["neighbor-mean"]
        left.plot([0,1],[m["diagnostic_only_pooled_coordinate_ari_mean"],m["raw_mean"]],"o-",c=COLORS[i],label=n,ms=4)
    left.set_xticks([0,1],["Pooled frames","Within-field"])
    left.set_xlim(-.25,1.25);left.set_ylabel("Neighbour-mean raw ARI")
    # Keep the legend out of the data and independent of the A/B header row.
    fig.legend(*left.get_legend_handles_labels(),frameon=False,loc="lower center",
               bbox_to_anchor=(.275,.065),ncol=2,columnspacing=1.2,handlelength=1.5)
    control=np.array([r["methods"]["expression-only"]["diagnostic_only_refined_control_ari_mean"]-r["methods"]["expression-only"]["raw_mean"] for r in rows])
    yy=np.arange(len(rows))[::-1]
    right.barh(yy,control,color=[COLORS[0] if v>=0 else COLORS[2] for v in control],height=.65)
    right.set_yticks(yy,names);right.axvline(0,color="#777",lw=.7)
    right.set_xlabel("Refined control minus\ntrue expression-only ARI")
    right.set_xlim(-.055,.17)
    fig.text(.018,.97,"A",weight="bold",fontsize=11,va="top");fig.text(.28,.97,"Coordinate-frame consequence",ha="center",fontsize=9,va="top")
    fig.text(.51,.97,"B",weight="bold",fontsize=11,va="top");fig.text(.785,.97,"Control contamination",ha="center",fontsize=9,va="top")
    fig.text(.52,.015,"Diagnostic counterfactuals only; neither enters the expression-only comparator",ha="center",fontsize=7.6)
    save(fig,"fig_field_controls")
    audit=json.loads((LEDGER/"estimand-audit.json").read_text())
    fig=plt.figure(figsize=(7.16,2.05));ax=fig.add_axes([.22,.31,.73,.47])
    for y,key,c in [(1,"legacy_joint_partial_range",COLORS[2]),(0,"midrank_joint_partial_range",COLORS[0])]:
        lo,hi=audit[key];ax.plot([lo,hi],[y,y],color=c,lw=3);ax.scatter([lo,hi],[y,y],color=c,s=20)
        ax.text((lo+hi)/2,y+.20,f"{lo:.3f}" if abs(hi-lo)<1e-12 else f"{lo:.3f} to {hi:.3f}",ha="center",fontsize=8.3)
    ax.set_yticks([1,0],["Ordinal tied ranks","Average tied ranks"])
    ax.set_xlim(.5,1);ax.set_ylim(-.45,1.6);ax.set_xlabel("Joint partial correlation across row permutations")
    fig.text(.55,.965,"Historical-panel rank diagnostic (200 row orders)",ha="center",va="top",fontsize=9)
    save(fig,"fig_rank_audit")

    if args.figures_only:
        print('Rendered three figures from committed inputs; result TeX and ledgers not written')
        return

    # Data-driven TeX keeps values bound to the exact seed-level results.
    lines=[r"\subsection{Field-identified tissue comparisons}\label{sec:field}",
      "All seven retained objects completed seven procedures at each of three seeds (147 method--seed runs). Independent raw-HDF5 checks reproduce selected cells, fields and reference labels; rescoring saved predictions reproduces every ARI. All stored spatial graph edges are within field. Table~\\ref{tab:field-inputs} gives the physical and annotation counts; Table~\\ref{tab:field-ari} gives the raw named-method readouts. This is a completed reanalysis of the explicitly retained scope, not the earlier pilot and not a replacement for every historical method or dataset.",
      r"\begin{table*}[!htbp]\centering\small",
      r"\caption{New field-identified panel. Cell and field counts refer to the actual model inputs; label counts exclude missing/unknown reference labels. $C$ averages within-field labelled-neighbour agreement over cells with neighbours. $C_0$ is the field-preserving permutation expectation; adjusted $C$ equals $(C-C_0)/(1-C_0)$. CODEX niche annotations are cluster-derived proxies, not independent biological labels.}",
      r"\label{tab:field-inputs}",r"\begin{tabular}{lrrrrrrr}\toprule",
      r"Object & Cells & Labelled & Fields & Classes & $C$ & $C_0$ & adjusted $C$\\\midrule"]
    for r in rows:
        vals=[r['dataset'],f"{r['n_cells']:,}",f"{r['n_labelled']:,}",str(r['n_fields']),str(r['n_classes'])]
        vals += [f"{r[k]:.3f}" for k in ['contiguity_cell_weighted','within_field_permutation_expectation','chance_adjusted']]
        lines.append(' & '.join(vals)+r'\\')
    lines += [r"\bottomrule\end{tabular}\end{table*}",
      r"\begin{table*}[!htbp]\centering\footnotesize",
      r"\caption{Raw ARI under the new fixed protocol, mean $\pm$ sample SD over seeds 1, 2 and 3. All embedding methods use KMeans with ten initialisations; SpatialLeiden returns its community assignments. The CODEX reference has 100 classes, but SpatialLeiden yields 51, 51 and 54 clusters; its ARI is computable but this is not an equal-$K$ comparison. No row is chosen by backend or refinement performance. Expression-only is PCA--KMeans without any coordinate input or refinement. SD describes optimisation variation, not uncertainty across tissues. SpaceFlow, GraphST and SpaGCN have no replacement scores and are not included in the numerical method set.}",
      r"\label{tab:field-ari}",r"\setlength{\tabcolsep}{3.2pt}\begin{tabular}{lrrrrrrr}\toprule",
      r"Object & Expression & Mean smooth & BANKSY-style & Leiden & Tessera & STAGATE & SEDR\\\midrule"]
    for r in rows:
        lines.append(r['dataset']+' & '+' & '.join(f"${r['methods'][m]['raw_mean']:.3f}\\pm{r['methods'][m]['raw_sd']:.3f}$" for m in methods)+r'\\')
    lines += [r"\bottomrule\end{tabular}\end{table*}",
      f"For CODEX, the fixed SpatialLeiden resolution search produced {codex_leiden[0]}/{codex_leiden[1]}/{codex_leiden[2]} clusters in seeds 1/2/3 against 100 reference niche classes. Its reported ARI remains a protocol-specific score, but it is not an equal-class-count comparison with fixed-$K$ embedding methods. No revised count-matched sensitivity is supplied here; method ranking for this object remains qualified.",
      r"Figure~\ref{fig:field-comparison} displays each named method minus the unrefined expression-only control, rather than an oracle maximum. The paired seed variation describes optimisation on these same objects, not tissue-population uncertainty.",
      r"\begin{figure*}[!htbp]\centering\includegraphics[width=\linewidth]{figs/fig_field_comparison.pdf}",
      r"\caption{Each named spatial method minus the genuinely expression-only comparator on the same cells and original seed identities. Dots are three-seed means; whiskers are the sample SD of the paired seed differences, not tissue-level confidence intervals. KMeans and raw assignments are fixed, so no true-label winner or refinement selection defines the contrast. Missing external methods are not imputed or included in an oracle maximum.}",
      r"\label{fig:field-comparison}\end{figure*}"]
    mib,code=byname['MIBI'],byname['CODEX']
    lines += [f"Within-field contiguity is {mib['contiguity_cell_weighted']:.3f} for MIBI-TOF and {code['contiguity_cell_weighted']:.3f} for CODEX in this labelled-neighbour definition. CODEX contains {code['n_labelled']:,} labelled cells in the 16,000-cell model input; one labelled singleton contributes no contiguity term. These definitions explain why this value need not equal an earlier all-cell diagnostic. No coordinates are translated to manufacture field separation.",
      f"The matched neighbour-mean diagnostic isolates the graph consequence while holding features, PCA, KMeans and seed identities fixed. Pooling local frames gives mean ARI {mib['methods']['neighbor-mean']['diagnostic_only_pooled_coordinate_ari_mean']:.3f} on MIBI-TOF versus {mib['methods']['neighbor-mean']['raw_mean']:.3f} within field, and {code['methods']['neighbor-mean']['diagnostic_only_pooled_coordinate_ari_mean']:.4f} versus {code['methods']['neighbor-mean']['raw_mean']:.3f} on CODEX. These are counterfactual audit arms, not inputs to the repaired comparator. They show that the coordinate error changes a model outcome as well as the contiguity axis.",
      f"Spatially refining the expression-only assignments changes its mean ARI by {control.min():+.3f} to {control.max():+.3f} across the retained objects. The sign is not fixed, and even a visually smoother control is no longer expression-only. The refinement counterfactual is stored separately and excluded from every named-method contrast (Fig.~\\ref{{fig:field-controls}}).",
      r"\begin{figure*}[!htbp]\centering\includegraphics[width=\linewidth]{figs/fig_field_controls.pdf}",
      r"\caption{Direct consequences of two protocol errors, evaluated on the same inputs and seeds as the repaired analysis. (A) Three-seed mean neighbour-mean PCA--KMeans ARI with pooled local coordinates versus within-field coordinates for MIBI and CODEX. All other fitting settings are fixed. (B) Change in the nominal control's ARI if its expression-only labels are spatially refined. These diagnostic arms are not eligible controls, and no maximum across them is used.}",
      r"\label{fig:field-controls}\end{figure*}"]
    values=summary['correlations']
    lines.append(f"Descriptive cross-object Spearman correlations between contiguity and the named contrasts range from {min(x['rho'] for x in values.values()):.3f} to {max(x['rho'] for x in values.values()):.3f}; for Tessera it is {values['Tessera']['rho']:.3f} and for STAGATE {values['STAGATE']['rho']:.3f}. Field-chance adjustment leaves their ranks unchanged in this small set. These are method-specific summaries of seven selected objects, not confirmation of the old oracle correlation or of a deployment rule. Average-rank partial associations are row-order invariant but do not exclude confounding; no significance direction was an acceptance condition.")
    (LEDGER/"field_results.tex").write_text('\n\n'.join(lines)+'\n')
    source['generated_sha256']={str(p.relative_to(MAN)):sha(p) for p in [LEDGER/'field_results.tex']+
        [MAN/'figs'/f'{n}.{ext}' for n in source['figures'] for ext in ['pdf','png']]}
    (LEDGER/"figure-source.json").write_text(json.dumps(source,indent=2)+'\n')
    print('Generated original-manuscript field section, three figures and bound result ledgers')


if __name__=="__main__":main()
