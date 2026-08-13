/** Per-portal science-gateway configs — claim-forward H1, no venue/Cite/BibTeX. */

export const SITE_CONFIGS = {
  b1: {
    slug: 'b1-site',
    shortName: 'Cofolding',
    title:
      "Protein–ligand cofolding confidence on Runs N' Poses degrades under ligand-similarity shift",
    kicker: 'ZF Lab · cofolding reliability',
    lead: 'The object is the deposited pose, the pocket, and the task — ligand RMSD versus interface DockQ versus affinity — not an ECE fold-change.',
    physicalObject:
      'Deposited protein–ligand complexes: pose, pocket occupancy, and task type on Runs N\' Poses.',
    primaryClaim:
      'Boltz-2 confidence stays high while ligand RMSD fails when training analogs are absent — pose error is not captured by the reported score alone.',
    archiveDoi: '10.5281/zenodo.21870002',
    siteUrl: 'https://peterponyu.github.io/b1-site/',
    mediaSource: 'capsules/b1-cofolding-confidence/docs/media',
    stats: [
      { value: '3', label: 'audited complexes · 8e5i / 8og8 / 8q0u' },
      { value: '8', label: "cofolding models on Runs N' Poses" },
      { value: '182', label: 'post-cutoff systems for pocket occupancy (Fig 8 only)' },
    ],
    resultsFigures: [
      { src: '/media/F_case_study.png', alt: 'Three ligand case cards at similar confidence with opposite pose outcomes', caption: 'Case cards: 8e5i holds at 0.15 Å; 8og8 and 8q0u fail at 6.3 Å and 15.6 Å while scores stay high.' },
      { src: '/media/F1_four_panel.png', alt: 'Four-panel pose comparison across audited complexes', caption: 'Fig 1 — opposite poses at similar Boltz-2 confidence.' },
      { src: '/media/F5_double_dissociation.png', alt: 'Known-pocket Vina versus learned cofolding', caption: 'Fig 5 — Vina redock succeeds where cofolding fails on hard examples.' },
      { src: '/media/F8_postcutoff_pocket.png', alt: 'Post-cutoff pocket occupancy analysis', caption: 'Fig 8 — pocket novelty after the training cutoff (scoped n=182).' },
    ],
    evidenceTiles: [
      { value: '0.15 Å', label: '8e5i ligand RMSD · training-similar hold' },
      { value: '6.3 / 15.6 Å', label: '8og8 / 8q0u RMSD · high score, failed pose' },
      { value: '≤2 Å', label: 'Vina redock on known-pocket hard cases' },
    ],
    methodsSummary:
      'Eight cofolding models scored on Runs N\' Poses deposited complexes. Ligand similarity via Tanimoto to training analogs; interface quality via DockQ; affinity tasks held separate from pose metrics. Pocket occupancy audited post-cutoff only where n is scoped on-panel.',
    claims: [
      { claim: 'High cofolding confidence does not imply correct pose when ligand analogs are absent.', scope: 'Runs N\' Poses · three audited complexes + post-cutoff pocket panel.', refutation: 'Show matched-confidence cases where RMSD stays ≤2 Å without training analogs.' },
      { claim: 'Known-pocket Vina can recover poses cofolding misses on the same structures.', scope: '8og8 and 8q0u hard examples only.', refutation: 'Demonstrate cofolding beats Vina on those pairs under the same pocket definition.' },
    ],
  },
  a3: {
    slug: 'a3-site',
    shortName: 'PLM boundary',
    title:
      'Where protein language model zero-shot mutation-effect scores hold and where they break',
    kicker: 'ZF Lab · PLM reliability',
    lead: 'Zero-shot log-likelihood ratios score substitution consequences — DMS fitness, ClinVar labels, thermodynamic ΔΔG, antibody liabilities. The likelihood is a fold-stability proxy: it does not read the binding partner.',
    physicalObject:
      'Single-residue substitutions and their biological consequences across six decision axes.',
    primaryClaim:
      'ESM-2 LLR tracks binding ΔΔG only as a weak fold-destabilization proxy — correlation collapses for antibody–antigen interfaces and non-interface positions.',
    archiveDoi: '10.5281/zenodo.21869995',
    siteUrl: 'https://peterponyu.github.io/a3-site/',
    mediaSource: 'capsules/a3-plm-boundary-atlas/docs/figures',
    stats: [
      { value: '0.725', label: 'protein-mean AUROC · n=69 plateau' },
      { value: '0.702', label: 'full ProteinGym · 217 assays' },
      { value: '0.20', label: 'binding ΔΔG Spearman overall' },
    ],
    resultsFigures: [
      { src: '/media/F7_binding_boundary.png', alt: 'Binding boundary figure showing interface collapse', caption: 'Fig 7 — binding partner is the strongest physical break (hero).' },
      { src: '/media/F5_stability_ddg.png', alt: 'Stability ΔΔG calibration', caption: 'Fig 5 — weak fold-stability proxy across homology hold-out.' },
      { src: '/media/F8_developability_map.png', alt: 'Antibody developability liabilities', caption: 'Fig 8 — no BH-significant polyreactivity or aggregation hits.' },
      { src: '/media/F11_boundary_atlas.png', alt: 'Synthesis boundary atlas', caption: 'Fig 11 — synthesis map, not new measurements.' },
    ],
    evidenceTiles: [
      { value: '0.08', label: 'antibody–antigen binding Spearman' },
      { value: '≈0', label: 'non-interface binding positions' },
      { value: '0/14', label: 'BH-significant polyreactivity sets' },
    ],
    methodsSummary:
      'Uniform raw-score audit of ESM-2 LLR across six axes with bootstrap intervals. Scores never clamped or per-assay rescaled. 69-protein plateau reported separately from 217-assay ProteinGym corroboration.',
    claims: [
      { claim: '650M parameters plateau for DMS discrimination; 3B adds no gain.', scope: '69-protein subset; 0.725 at 650M and 3B within bootstrap CI.', refutation: 'Show significant AUROC gain at 3B on the same panel with the same protocol.' },
      { claim: 'LLR does not read the binding partner — fold proxy only.', scope: 'SKEMPI binding ΔΔG; Fig 7 locked table.', refutation: 'Demonstrate interface-aware correlation without fold confound on the same table.' },
    ],
  },
  b28: {
    slug: 'b28-site',
    shortName: 'Proteomic transfer',
    title:
      'A proteomic tumor classifier is overconfident where it transfers worst, and random cross-validation hides it',
    kicker: 'ZF Lab · proteomic transfer',
    lead: 'The object is a held-out tumor proteome versus its paired normal. Six umich cohorts; the thing that moves is which cancer type is left out, and whether the classifier uses protein-specific abundance or global intensity and missingness.',
    physicalObject:
      'Paired tumor and normal proteomes under leave-one-cohort-out holdout.',
    primaryClaim:
      'Random cross-validation inflates AUROC and hides miscalibration where LOGO transfer is worst — abundance beats intensity only where signal is trivial.',
    archiveDoi: '10.5281/zenodo.21870024',
    siteUrl: 'https://peterponyu.github.io/b28-site/',
    mediaSource: 'capsules/b28-proteomics-transfer/docs/figures',
    stats: [
      { value: '4', label: 'figures (complete gallery)' },
      { value: '6', label: 'cohorts under LOGO' },
      { value: '1,191', label: 'samples · 695 tumor / 496 normal' },
    ],
    resultsFigures: [
      { src: '/media/b28_F1_transfer.png', alt: 'Transfer performance across held-out cohorts', caption: 'Fig 1 — AUROC and ECE move with held-out cancer type.' },
      { src: '/media/b28_F2_signal_ladder.png', alt: 'Abundance versus intensity signal ladder', caption: 'Fig 2 — protein abundance versus global intensity/missingness.' },
      { src: '/media/b28_F3_calibration.png', alt: 'LOGO versus random CV calibration', caption: 'Fig 3 — random CV hides overconfidence under LOGO.' },
      { src: '/media/b28_F4_riskcoverage.png', alt: 'Risk-coverage and cohort composition', caption: 'Fig 4 — retained-set composition bias (LUAD/LSCC).' },
    ],
    evidenceTiles: [
      { value: '0.946', label: 'pooled LOGO AUROC (abundance)' },
      { value: '0.110', label: 'expected calibration error' },
      { value: 'r=−0.963', label: 'transfer worst ↔ overconfidence' },
    ],
    methodsSummary:
      'Six umich proteomic cohorts (CCRCC, LUAD, UCEC, HNSCC, LSCC, PDAC) with leave-one-cohort-out tumor-versus-normal classification. Compare protein-specific abundance to global intensity and missingness features. No new mass-spec measurements — evaluation protocol only.',
    claims: [
      { claim: 'Classifier is most overconfident where LOGO transfer is worst.', scope: 'Six cohorts · abundance model · ECE versus held-out type.', refutation: 'Show flat ECE across holdouts under the same LOGO splits.' },
      { claim: 'Random k-fold CV reports higher AUROC and masks miscalibration.', scope: 'Same 1,191 samples · paired comparison in Fig 3.', refutation: 'Match LOGO calibration under random splits with batch-aware blocking.' },
    ],
  },
  spgd: {
    slug: 'SPGD-site',
    shortName: 'SPGD',
    title: 'Spot composition across platform, compartment, and donor',
    kicker: 'ZF Lab · spatial deconvolution',
    lead: 'SPGD estimates the cell-type mix inside each spatial spot. The object that moves is that mix — imaging platform, tissue compartment, mixing regime, donor — not a rank table on the same mixtures.',
    physicalObject:
      'Spot-level cell-type composition across openST, MERFISH, and STARmap substrates.',
    primaryClaim:
      'Under a zero-tuning budget SPGD is competitive with default SOTA on spatial maps — no method dominates across platform, compartment, and donor axes.',
    archiveDoi: '10.5281/zenodo.21869991',
    siteUrl: 'https://peterponyu.github.io/SPGD-site/',
    mediaSource: 'capsules/spgd-deconv/docs/figures',
    stats: [
      { value: '3', label: 'platforms: openST, MERFISH, STARmap' },
      { value: '8', label: 'substrates · same/cross-platform · simulation · donor' },
      { value: '0.203', label: 'DestVI STARmap RMSE lock' },
    ],
    resultsFigures: [
      { src: '/media/fig_spatial.png', alt: 'Spatial maps of tumor stroma macrophage proportions', caption: 'Spatial maps — tumor · stroma · macrophage on real tissue.' },
      { src: '/media/fig_rare.png', alt: 'Rare cell type recovery', caption: 'Rare-type mixing regimes on simulated and real ground truth.' },
      { src: '/media/fig_crossdonor.png', alt: 'Cross-donor replication', caption: 'Donor held out — spots from one patient, reference from another.' },
    ],
    evidenceTiles: [
      { value: '0.91 / 0.84 / 0.55', label: 'tumor / stroma / macrophage PCC' },
      { value: '3/40', label: 'controlled paired losses vs SOTA panel' },
      { value: '0', label: 'dataset-specific tuning knobs' },
    ],
    methodsSummary:
      'Specificity-weighted Poisson self-gating estimator with platform correction. Thirteen methods on eight substrates with bootstrap CIs. Algorithmic constants fixed a priori — training-free, GPU-free, zero-tuning budget versus default SOTA configs.',
    claims: [
      { claim: 'SPGD wins most paired comparisons and loses a bounded set on MERFISH and simulation.', scope: '42-cell rank pool · six metrics · bootstrap CIs.', refutation: 'Show a single method dominates all substrates under the same zero-tuning budget.' },
      { claim: 'Spatial maps — not leaderboard rank — carry the biological read.', scope: 'Tumor/stroma/macrophage on real tissue; cross-donor replication.', refutation: 'Demonstrate rank-only gains without map-level concordance on the locked references.' },
    ],
  },
};
