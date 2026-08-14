# Tessera-ST claim ledger

Created: 2026-06-25

## Allowed now

- This is a research scaffold. The deep model trains end-to-end and the curated ablation grid
  runs on both the synthetic tessellation and the real-DLPFC adapter path.
- The synthetic smoke demonstrates that the machinery runs and that the loss decreases. It is
  a software-correctness artifact, **not** evidence of biological or benchmark superiority.
- Baseline methods are referenced for parity planning only; no upstream source is copied.

## Claim status

Claim status has two evidence-governance states:

- `LOCKED-pending`: the proposed scientific claim has not passed its listed gates and must not be added to
  the public study summary as an established result. The component-superiority claims C1--C5 below remain in this
  state.
- `ADOPTED-verified`: the result artifact and its bounded wording have been checked by
  `experiments/verify_manuscript.py` and may appear in the public study summary with the recorded caveats. R2, R5,
  and R6 are in this state. Adoption verifies evidence provenance; it does not convert Tessera into a SOTA
  method or graduate any broader superiority claim.

Thus `LOCKED` applies to pending claim rows, not retroactively to the adopted, verifier-pinned results that
already support the causal-law study.

| # | Claim (locked) | Missing evidence |
|---|----------------|------------------|
| C1 | Edge gating reduces over-smoothing → higher boundary-F1 than a uniform-smoothing encoder at matched ARI on DLPFC | real-DLPFC ablation (gating on/off) on ≥3 sections + seed variance |
| C2 | Multi-scale fusion improves ARI on layered cortex vs. single-scale | real-DLPFC ablation (multi_scale on/off) across the 12 sections |
| C3 | Boundary-contrastive loss improves NMI/embedding geometry | ablation (loss on/off) + negative control |
| C4 | Calibrated head lowers ECE without hurting ARI | held-out split + conformal calibration on real data |
| C5 | Full Tessera is competitive with named graph encoders on DLPFC ARI | same-data parity runs of STAGATE/GraphST/SEDR/BANKSY behind the adapter |

## Graduation gates (all required before any row → paper-ready)

1. **Real-data run.** ≥3 DLPFC sections processed through `tessera ablate-real`, metrics emitted.
2. **Full ablation.** The curated grid reproduced on real data with seed variance (≥3 seeds).
3. **Same-data baseline parity.** Named baselines run on the identical inputs; ARI/NMI compared.
4. **Boundary + calibration evidence.** boundary-F1 and ECE reported, not just ARI/NMI.
5. **Clean leakage scan.** `scripts/check_independence.sh` green; no baseline brand in `src/`.
6. **Failure-mode analysis + human sign-off.** Where Tessera loses is documented honestly.

Until those gates pass, no C1--C5 superiority prose lands and no such row leaves `LOCKED-pending`.
This restriction does not bar the bounded R2/R5/R6 results already in `ADOPTED-verified` state.

## Component-retention policy

The four components are hypotheses. `full` is not assumed final. Each benchmark run reports a
**data-selected** config (best Tessera variant on that data) alongside reference baselines
(`ref:nonspatial-kmeans`, `ref:smoothed-kmeans`) and the SOTA-class `backbone`. A component that
does not earn its place on the real-DLPFC gate is **cut from the shipped model**, not retained
for narrative symmetry. On the current synthetic fixture the selected config is *not* `full` —
that is recorded honestly, and the real verdict waits on DLPFC.

## Recorded result — synthetic benchmark (LOCKED, honest-negative)

Run: `tessera smoke-synth --epochs 120` on the SNR-hard tessellation (domain_signal 0.4,
iid_noise 1.0). With reference baselines in the same table:

- **Tessera does NOT beat a naive `smoothed-kmeans` baseline** (best Tessera variant ARI 0.87 vs
  0.99). Recorded as a negative result. Likely cause: the linear-Gaussian synthetic's optimum is
  literally neighbourhood averaging, so a naive smoother is near-optimal and a GNN has no room to
  win. Synthetic data of this form cannot fairly demonstrate GNN > averaging.
- Component signal is real: removing `boundary_contrastive` collapses ARI 0.87 -> 0.46 (load-
  bearing); `multi_scale` is dropped by data selection (`no_multi_scale` wins); `edge_gating`
  marginal; `calibrated_uncertainty` affects only ECE.

Implication: no superiority claim is even plausible from synthetic. The DLPFC gate is the only
place "Tessera vs baselines/SOTA" can be settled. We do not tune the fixture to manufacture a win.

## Recorded result — real DLPFC 151673, rough test (PROMISING, not yet a claim)

Run: `tessera ablate-real .../dlpfc_maynard_2021_151673.h5ad --label-key ground_truth
--epochs 150 --device cuda` (local data, zero network cost; 3639 spots, 7 layers).

- **Tessera clears the baseline bar on real tissue**: data-selected `no_multi_scale` reaches
  ARI 0.470 / NMI 0.587 vs naive `smoothed-kmeans` 0.265 / 0.455 and non-spatial 0.191. ARI ~0.47
  on 151673 is within reach of the named-SOTA band (~0.5-0.6) at 150 epochs, no tuning.
- Component verdicts confirmed on real data: `boundary_contrastive` is load-bearing (removing it
  drops 0.470 -> 0.273, back to baseline); `multi_scale` is a liability (dropping it is best) and
  should be cut or redesigned; `edge_gating`/`calibrated_uncertainty` marginal.
- Honest shortfalls: small-domain recovery is poor for everyone (small_IoU 0.09-0.17); ECE is high
  (0.54-0.67) so the "boundary calibration" angle is not yet earned.

Status: this justifies investing further cost. Still NOT a paper claim — needs multi-section runs
(12 sections available locally), seed variance, named-SOTA parity on identical inputs, and the
multi_scale cut. No row graduates yet.

## Recorded result — DLPFC 151673, 3 seeds (corrects the single-seed optimism)

Run: `tessera ablate-real ... --epochs 120 --seeds 1,2,3`. Mean±std over seeds:

- Tessera ARI **0.42 ± 0.04** (best config), vs baselines 0.27 ± 0.001 / 0.19 ± 0.003. Clears the
  baseline bar robustly (lower bound 0.38 > 0.27), but the **seed variance is ~15x the baselines'**
  — Tessera is unstable, and the single-seed 0.47 was partly luck.
- The single-seed component ranking is NOT reliable: selected config drifted from `no_multi_scale`
  (1 seed) to `no_edge_gating` (3 seeds); edge_gating/multi_scale/calibrated differences (0.38-0.42)
  sit inside ±0.04. The earlier "cut multi_scale" call is retracted as noise.
- Robust across seeds: `boundary_contrastive` is load-bearing (drop -> 0.25, low variance).
- vs SOTA: literature STAGATE/GraphST on 151673 ~0.5-0.6, so Tessera at 0.42 is most likely BELOW
  SOTA. No real SOTA numbers run yet — `backbone` is a stand-in, not SOTA.
- Internal metrics (ASW/DBI) are NOT counted as evidence: computed in each method's own embedding,
  structurally favouring deep models; judged by external ARI/NMI instead.

Open before any claim: reduce seed variance, run real STAGATE/SEDR parity on identical inputs,
multi-section runs.

## Recorded result — epochs sweep (refutes under-fitting; finds training degradation)

`experiments/diag_epochs.py`, full config, 151673, 3 seeds per budget:

- epochs 120 -> ARI 0.417±0.042 ; 400 -> 0.339±0.003 ; 800 -> 0.328±0.011.
- **Training longer makes ARI WORSE and variance shrink** — the model converges to a stable but
  inferior solution. The 120-epoch 0.42 is an early-stopping fluke, not a reliable score.
- Diagnosis: likely over-smoothing / representation collapse in the graph autoencoder (a method
  branded "anti-over-smoothing" is itself collapsing during training). This is a core-mechanism
  defect, not a tuning issue. "Add epochs" is the wrong fix.

Next: instrument embedding collapse (mean pairwise distance / effective rank vs epoch) to confirm;
if confirmed, the encoder/loss needs an anti-collapse mechanism (e.g. variance/covariance
regularisation) before ARI/stability/SOTA comparisons are meaningful.

## Recorded result — collapse probe (refutes collapse; finds objective misalignment)

`experiments/diag_collapse.py`, full config, 151673, seed 1, ARI + collapse signals per 50 epochs:

- Embedding does NOT collapse: effective rank holds ~20, embedding variance and mean pairwise
  distance RISE (0.3->7.2, 4.3->20.9). Loss falls monotonically (8.1->1.5).
- ARI peaks ~0.44 at epoch ~175 then FALLS to 0.34 while loss keeps falling. The self-supervised
  proxy (reconstruction + boundary-contrastive) is anti-correlated with layer ARI past the peak —
  contrastive over-spreads the embedding, so KMeans cuts expression gradients, not layers.
- Diagnosis (3 probes): not under-fitting (worse with epochs), not collapse (embedding spreads),
  but OBJECTIVE MISALIGNMENT. The 0.44 peak is only reachable by GT-peeking early stop (leakage).
  This is a loss-design defect, not tunable away.

Decision point: either redesign the loss to align with domain structure (e.g. drop/repurpose the
contrastive term, add a spatial-consistency objective) and re-probe for monotonic ARI, or accept
that a contrastive-embedding domain method is the wrong bet here. No claim is viable until the
training objective tracks the task. Cost discipline note: 3 local probes (zero network) exposed a
core-mechanism defect BEFORE any broader benchmark investment.

## Recorded result — comprehensive marker-anchored table (DLPFC 151673, 3 seeds)

`experiments/dlpfc_151673_full.json`. Full panel incl. the EXTERNAL (non-circular) `marker_purity`
(predefined literature layer markers). Triangulating the three independent kinds of evidence —
ARI/NMI (weakly circular), marker_purity (external), component ablation:

- All three agree on direction: configs with high ARI (no_edge_gating 0.42, no_multi_scale,
  full ~0.42) also have the highest marker_purity (~0.572-0.574); the collapsed configs
  (no_boundary_contrastive 0.246, backbone 0.259) have the lowest (~0.545). So `boundary_contrastive`
  captures REAL layer structure — confirmed by an anchor that is NOT derived from this data's
  clustering. ARI 0.42 > baseline 0.27 is therefore a real signal, not a circular artefact.
- BUT scale matters and is honestly noted: on the external anchor Tessera only edges out
  smoothed-kmeans (0.574 vs 0.554) while on ARI the gap looks large (0.42 vs 0.27). The external
  evidence says Tessera's advantage over the naive baseline is REAL but MILDER than ARI alone implies.
- Unchanged caveats: seed variance still high (±0.04), the train-objective/ARI misalignment
  (diag_collapse) still stands. No single metric decides; this is the triangulated read.

Net: the method has genuine, externally-corroborated signal (boundary-contrastive term), is not a
circular illusion, but its edge over a naive smoother is modest on non-circular evidence and it is
unstable. Worth fixing the objective/variance before SOTA parity; not yet a claim.

## Recorded result — real SOTA (STAGATE) + the clustering-backend confound (decisive)

`experiments/run_sota.py`, STAGATE on 151673, 3 seeds, SAME eval panel as everything else:

- STAGATE+KMeans ARI 0.246; STAGATE+GMM(tied) ARI **0.577**. Swapping ONLY the clustering backend
  doubles the score. The backend is a confound bigger than between-method differences. Our earlier
  "everyone uses KMeans = fair" protocol systematically under-rated methods designed for mclust/GMM.
- STAGATE+GMM beats Tessera-best(+KMeans) on nearly every axis, including the less-circular
  marker_purity (0.617 vs 0.572): ARI 0.577>0.42, NMI 0.71>0.56, CHAOS 1.002<1.057, PAS 0.012<0.097,
  small_IoU 0.203>0.15. Real SOTA is clearly ahead of Tessera as currently built.
- Symmetry caveat (do NOT one-stroke-kill Tessera): KMeans may also under-rate Tessera. A fair
  verdict needs Tessera evaluated under GMM too — pending. Backend is now a reported axis, not a
  fixed assumption; enthrone no single backend either.

## Recorded result — symmetric backend fairness (final SOTA positioning)

`experiments/run_tessera_gmm.py`. GMM lifts Tessera too (full 0.416->0.527; no_edge_gating
0.420->0.484), confirming the backend confound is bidirectional — KMeans under-rated Tessera as
well, so the earlier "Tessera 0.42 vs STAGATE 0.58" gap was a protocol artefact.

Under symmetric, fair GMM backend on 151673:
- ARI: STAGATE 0.577 > Tessera.full 0.527 > Tessera.no_edge_gating 0.484.
- marker_purity (less-circular): STAGATE 0.617 > Tessera ~0.575-0.580. NMI: 0.712 > 0.638.
- Real SOTA leads Tessera comprehensively but MODESTLY (~0.05 ARI), not the ~0.16 KMeans implied.

Verdict (multi-angle, symmetric, no single-metric/backend enthroned): Tessera as currently built is
below real SOTA on every axis, but the margin is small and the KMeans protocol had exaggerated it.
Backend is now a reported axis. To matter, Tessera still needs the train-objective/variance fixes;
it is not yet competitive, but the honest gap is narrow.

## Recorded result — exhaustive SOTA-mechanism improvement sweep (honest negative)

After studying SOTA winning mechanisms (STAGATE: GAT+pure-recon; SEDR: DEC self-training; GraphST:
DGI contrastive; BANKSY: neighbour mean+AGF), each borrowed idea was probed on 151673 under the
fair GMM backend (`experiments/diag_{dec,anneal,recon_gmm,attention}.py`):

| Attempt (mechanism) | GMM ARI | vs STAGATE 0.577 |
|---|---|---|
| full (KMeans, old protocol) | 0.42 | below (artefact) |
| full (GMM, symmetric) | 0.527 | below |
| + DEC self-training (SEDR) | KMeans up / GMM hurt | no |
| + contrastive anneal | 0.44 (worse) | no |
| pure reconstruction (STAGATE-style) | 0.42 | far below |
| + GAT attention x gate (architecture) | ~0.55 peak, unstable | no |

Every path plateaus at 0.52-0.56 and none passes STAGATE's 0.577; degradation + high seed variance
persist. Root cause is twofold: (1) DLPFC domain detection is a SATURATED task — mature SOTA is near
the ceiling for graph-autoencoder methods; (2) Tessera has no structural novelty that breaks past
it, and its contrastive term causes intrinsic long-training decay. Honest conclusion: borrowing
known SOTA mechanisms gets Tessera to ~parity-minus, not past SOTA. Beating SOTA here would need
genuine research novelty (open-ended, high-uncertainty), not mechanism transfer.

## Recorded result — comprehensive multi-SOTA panel (retracts the ARI-only "stop-loss")

`experiments/run_sota_panel.py` + `run_final_panel.py`. Runnable SOTA: STAGATE (GAT), BANKSY-style
(non-DL neighbour augmentation, clean-room). SEDR + GraphST both numba/numpy-blocked in this env
(recorded, not hidden). Full 11-metric panel incl. internal ASW/DBI/CAL, fair GMM backend, 3 seeds.

Per-metric winners (no method dominates — SOTA leads only on SOME axes):
- STAGATE (3): ARI 0.581, NMI 0.702, marker_purity 0.603.
- BANKSY-style (4): CHAOS, PAS, CAL, ECE.
- Tessera.full (4): ASW 0.209, DBI 1.555 (cleanest geometric separation), boundary_F1 0.454,
  small_IoU 0.192 — i.e. exactly its thesis axes (boundary + geometry).

**Retraction:** the earlier "exhaustive sweep -> can't beat SOTA -> stop-loss" entry was
single-metric tunnel vision on ARI (the very error in [[comprehensive-evaluation-no-metric-dismissal]]).
Comprehensively, Tessera ties for most metrics won and leads on its design-intended axes. It is a
peer method with a distinct profile, NOT a weak STAGATE clone.

Honest caveats kept: only 2 SOTA runnable; Tessera ARI 0.527 still trails STAGATE 0.581 (the
mainstream label-agreement axis); single section 151673; all metrics semi-circular/conditioned;
variance non-trivial. Direction: hold the boundary/geometry lead, narrow the ARI gap.

## Recorded result — cross-section (3 donors) corrects the single-section picture

`experiments/run_multisection_panel.py`. Real 10x h5 per section, one per donor (151507/Br5292,
151669/Br5595, 151673/Br8100), Tessera.full / STAGATE / BANKSY-style, GMM backend, 3 seeds. Means
across the 3 sections:

- The 151673 "Tessera wins 4 axes" was single-section luck. Across donors Tessera robustly wins only
  ASW (internal, semi-circular) and ECE (calibration). small_IoU flips to WORST (0.272 vs STAGATE
  0.440); boundary_F1 advantage evaporates (0.357 ~ BANKSY 0.360); ARI/NMI trail (Tessera lowest).
- Notable: BANKSY-style (simplest, non-DL neighbour augmentation) has the HIGHEST cross-section ARI
  (0.512 > STAGATE 0.471 > Tessera 0.396). Simple beats complex on DLPFC here.
- Tessera's only durable + meaningful edge is calibration (ECE), via its native cluster head — hard-
  clustering SOTA can't give it. So the honest differentiation is the calibrated-uncertainty niche,
  NOT all-round domain detection (it loses ARI cross-section and its boundary/small-domain wins do
  not generalise).

Caveats: 3 of 6-available sections; still semi-circular metrics; ECE itself is GT-derived. But the
direction is now data-backed across donors, not from one slide.

## Recorded result — FULL panel, 4 real SOTA (151673, GMM, 3 seeds)

`experiments/run_full_panel_v2.py` (numba-stub `experiments/_numba_stub.py` lets SEDR/GraphST run
without changing the env). Tessera + STAGATE + SEDR + GraphST + BANKSY-style + 2 floors, 11 metrics.

ARI order: STAGATE 0.580 > SEDR 0.570 > floor:smoothed 0.532 > Tessera 0.527 > BANKSY 0.492 >
floor:nonspatial 0.421 > GraphST 0.290* (*GraphST under-fit by my adapter — no mclust/refine; not
its true level). Per metric, Tessera beats STAGATE 4/11 and SEDR 4/11.

Tessera's ONLY full-field, non-semi-circular win: boundary_F1 = 0.454 (beats all 6 opponents) — its
thesis axis (sharpest domain boundaries). Internal ASW/DBI also best but semi-circular. Otherwise:
ARI mid-pack (below even floor:smoothed); CHAOS/PAS among the worst (contrastive hurts spatial
smoothness); the earlier "calibration edge" is GONE once SEDR is included (SEDR ECE 0.189 < Tessera
0.247) — adding one more real SOTA erased a pseudo-advantage, exactly why SOTA coverage must be >1.

Honest standing: Tessera is a peer-ish method whose single defensible, non-circular advantage is
boundary sharpness, not label-agreement. Caveats remain: single section/dataset, unified GMM backend
(each SOTA's own pipeline may score higher), GraphST adapter not optimised, all metrics semi-circular.

## ★★★★★★★★★★ R6 — decoupling robustness check strengthens the causal leg (2026-07-13)

Triggered by the referee-style objection that the primary synthetic sweep (`mechanism_synth.py`, R3)
scrambles positions and so erodes BOTH the labels' spatial contiguity AND the generic expression↔position
alignment any spatial method exploits — so maybe the law tracks alignment, not contiguity specifically.
Ran a two-arm decoupling control at the primary experiment's full fidelity
(`experiments/mechanism_decoupled_v2.{py,json}`: 400-epoch STAGATE, 5 seeds, 11-level 0→1 sweep, 2 independent
substrates S1/S2, GPU, deterministic, runtime 2371 s). Arm A (alignment-only): fix coords+labels
(contiguity constant), swap a growing fraction of expression vectors. Arm B (contiguity-only): hold alignment
and signal fixed (expression regenerated from the current label), overwrite a growing fraction of labels with
random domain ids. **Three discriminators, all favouring contiguity on both substrates and both priors
(neighbour-mean / STAGATE):** (1) effect size — normalized_slope(B) − normalized_slope(A) = +0.28 … +0.58;
(2) sign-flip — ONLY Arm B reproduces the study's headline positive→negative crossing (S1 near contiguity
0.58, S2 near 0.76); Arm A's advantage only decays toward 0, never flips; (3) clean-vs-confounded — Arm B holds
the non-spatial floor flat (range 0.03–0.05), while Arm A's expression-swap collapses the floor (by 0.36 on
S1, 0.68 on S2), i.e. its shrinking advantage is substantially generic signal loss — the confound a referee
flags, landing on the arm NOT used to support the law. Cross-substrate rollup:
`norm_slope_both_favor_B` / `only_B_flips_sign` / `B_floor_flatter_than_A` all TRUE on both substrates.

**Corrects a stale auto-verdict.** The earlier low-fidelity v1 (`experiments/mechanism_decoupled.{py,json}`,
120-epoch, 1 substrate, 6-level) carried an auto-generated verdict claiming this experiment "does NOT support
contiguity-specificity." That verdict is WRONG — an artifact of using Spearman ρ as the tie-break, but both
arms are monotone BY CONSTRUCTION so ρ saturates near 1 for both and carries no effect-size information. v2's
S1 (same substrate) reproduces v1's per-level numbers almost exactly (contrast norm_slope B−A: v1 +0.305/+0.286
vs v2 +0.302/+0.283), so restoring fidelity did NOT change the underlying result — it added the denser sweep
(resolves the sign-flip crossing), the 2nd substrate, and the sign-flip/floor-flatness discriminators that ρ
misses. The v1 json has been annotated with a `superseded_by` pointer; it is not cited anywhere in the
study report or this ledger.

**Evidence state.** v2 is ADOPTED as the R6 robustness artifact and checked directly from the local
`mechanism_decoupled_v2.json` by `experiments/verify_manuscript.py` (effect-size direction, exclusive sign
flip, floor flatness, and all four hand-typed `tab:decouple` rows). The JSON remains a git-ignored generated
artifact under repository policy, but the archival tooling requires it in the release archive. The
bounded R6 robustness result is therefore `ADOPTED-verified`; adoption closes evidence provenance and
permits the recorded causal-control wording, but it does not graduate a new Tessera-superiority claim.

**Study report.** New paragraph "Decoupling contiguity from alignment" + `Table tab:decouple` added to
§`sec:causal` (§VII). Framed honestly as strengthening the CONTROLLED-MANIPULATION leg only (the observational
11-platform leg is independent), and NOT as "alignment is irrelevant" (Arm A's advantage does track alignment
— it is just confounded and never flips sign); the correct claim is "contiguity is the lever movable in
isolation, and only moving it reproduces the full sign-flip." Status is `ADOPTED-verified` under the
honest-claims guardrail: this is a robustness check on the existing causal claim, not a new superiority claim.

## ★★★★★★★★★ R5 — search-corrected proxy + causal-first reframe (study retitle, 2026-07-01)

**Status: `ADOPTED-verified`.** The search-corrected proxy wording and causal-first reframe are pinned by
`experiments/verify_manuscript.py`; this status does not imply a method-superiority claim.

Triggered by a falsification-first audit of the label-free `coh_gain` proxy: its reported p=0.079 was an
UNCORRECTED per-candidate value, but `coh_gain` was SELECTED as the best of six label-free candidates
(raw_coh, smooth_coh, coh_gain, partition_shift, sil_gain, smoothing_score), so its significance must be
corrected for that search. `experiments/proxy_search_corrected.py` (+ `proxy_search_corrected.json`):
Westfall–Young max-statistic permutation null (max |Spearman ρ| over the 6 candidates vs permuted
spatial_advantage, B=10,000, seed 1234). Observed max-|ρ| = **0.551** (coh_gain, the argmax) sits at ≈72.5th
percentile of the null → **search-corrected p = 0.275 (~0.28)**; the uncorrected per-candidate permutation p
is 0.083 (consistent with the asymptotic 0.079). So the proxy is **directional but NOT significant** once
selection is priced in — reframed everywhere as an "honestly marginal label-free prior," NOT an established
predictor.

**Causal-first reframe.** The study was retitled from "No universal SOTA in
spatial-domain detection" to **"A causal law for spatial-domain detection: the value of a spatial prior is
set by the ground truth's spatial contiguity."** Abstract + intro now LEAD with the CAUSAL synthetic-sweep
law (ρ=+0.98 to +1.00, p<0.001) as the primary result; the winner-count / no-universal-SOTA finding is
demoted to supporting evidence (its § heading renamed to "No universal winner across methods"). The proxy p
is replaced everywhere (Table `tab:law` row, §`sec:causal`, Limitations) with the search-corrected p=0.28,
with the uncorrected 0.079 kept alongside for honesty; the causal sweep is stated as carrying the weight of
the label-free claim.

**Related work + differentiation.** Added a Related Work paragraph and `Table tab:related-benchmarks`
positioning tessera-st against the 5 directly-overlapping 2024–2026 benchmarks (Yuan et al. Nat. Methods
2024; Chen et al. iMeta 2025; Kang et al. NAR 2025; Descoeudres/Canzar et al. bioRxiv 2026; Sun et al.
bioRxiv 2025 Smoothness Entropy). All 5 added to comparison references with identifiers verified live on 2026-07-01
(each annotated VERIFIED; bioRxiv Canzar uses the newer 10.64898 prefix, not 10.1101 — confirmed, not
guessed). None runs a causal contiguity perturbation or validates a label-free proxy against a held-out
spatial-advantage target — that combination is the wedge.

**Verifier.** `experiments/verify_manuscript.py` now loads `proxy_search_corrected.json` and asserts (i)
coh_gain is the argmax over the full 6-candidate family, (ii) observed ρ=0.551, (iii) search-corrected
p=0.275 and that it is >0.05 AND > the uncorrected p (meaningful, not deleted); it also ties the "0.28"
search-corrected string and "search-corrected"/"p<0.001" into the study text. Exit 0 preserved. README status line
updated from "research scaffold / no claim graduated" to the current 11-platform causal-law state.

## ★★★★★★★★ R4 — verification and native-baseline round (2026-06-29): ultragoal 4/4, code-review APPROVE

**G001 Metadata:** the study text now has author "Zeyu Fu" (from git) with CLEARLY-MARKED [to be completed]
placeholders for affiliation/ORCID/email/funding/acknowledgements (NOT fabricated — no verified
institutional details in env); Keywords; Data availability (every platform cited to source + Squidpy);
Code availability (clean-room note); Author contributions (Z.F.); Competing interests
(none). **G002 Reproducibility tooling:** the deposit script (token ONLY from an environment variable
via Authorization: Bearer header — never URL/logs; allowlist archive) was reviewed but not executed. **G003
NativeBaselinesFull:** GraphST already got real R mclust on its 6 ≤8k platforms; SpaGCN re-run with its
GENUINE native louvain pipeline in a dedicated env (tessera-spagcn: py3.10/numpy1.26/real numba 0.65.1) —
on 7/11 platforms (fails ≤36-dim protein panels, like SpaceFlow/GraphST) it CONFIRMS the conclusion: wins
only openST (1/7), inverted profile ρ=+0.67. So the kmeans-init handicap did NOT drive no-universal-SOTA.
**G004 FinalGate:** verify_manuscript exit 0 (now also checks the compiled study text); pytest
23; clean-room intact; the compiled study is 13pp / 0 overfull / 0 undefined (current 2026-07-14 build). Independent code-review APPROVE after one
REQUEST-CHANGES round (fixed: a deposit-token leak into an error URL via query param → Bearer header; the study text
brought into the verifier; 3 LOW). Two human-only inputs remain by design: real affiliation/ORCID + a deposit
token. Bugs caught+fixed this round: missing `import os` (crash) and a self-matching pgrep watcher (4.6h spin).

## ★★★★★★★ R3 — adversarial expansion (data / SOTA / mechanism, 2026-06-28): user distrust of "complete" → deeper stress tests

Triggered by the user's refusal to accept R2 as "complete" — "扩展数据？扩展SOTA？扩展机制？". A filesystem
re-audit found "platforms exhausted" was OVERSTATED. Three honest expansions, all machine-checked
(`verify_manuscript.py` exit 0); a code-review caught a data-provenance issue (fixed by genuine re-run).

**Mechanism — the law is CAUSAL, not just correlational (`experiments/mechanism_synth.py`, §9).** Controlled
synthetic sweep: manipulate ONLY spatial contiguity (spatially scramble a fraction of cells, carrying each
cell's expression+label), expression signal held fixed (non-spatial ARI = 0.345 at all 8 levels, range
0.000). Spatial-prior advantage rises monotonically and FLIPS SIGN (−0.34→+0.48 neighbour-mean; −0.33→+0.54
STAGATE): ρ=+0.98 / +1.00 (p<0.001). Upgrades the cross-platform correlation to a manipulated cause. Honest
weighting: the neighbour-mean arm is near-definitional; the STAGATE (learned GNN) arm is load-bearing.

**Data — "platforms exhausted" was WRONG; law WEAKENS but survives (`experiments/expand_data.py`, §8d).**
Found 3 independent cancer ST platforms missed by §7 (st_COAD/LIHC/OV, cell-type GT, contiguity 0.22–0.69;
CESC/NSCLC/PRAD correctly excluded as technical `segmentation_method`-only; Xenium/COAD-Visium have no
domain GT). At n=14 the lead correlation **drops from +0.72 (p=0.012) to +0.56 (p=0.038)** — still positive
+ significant, but honestly attenuated (st_OV is a counter-example: contig 0.685, advantage −0.102). n=11
kept as headline (diverse, de-duplicated); n=14 reported as the robustness check that tempers it. No
universal SOTA unmoved (7 winners). *(Code-review HIGH: first run hand-seeded COAD/LIHC from a prior log
and skipped them — provenance disguised; FIXED by deleting the cache and re-running all 3 genuinely.)*

**SOTA — SpaGCN (7th, most-cited) does NOT break no-universal-SOTA (`experiments/spagcn_panel.py`, §8e).**
Installed SpaGCN; numpy-2.5 blocks its native louvain/pynndescent path (needs real-numba JIT classes the
stub can't emulate), so run with **kmeans-init + stubbed numba + no histology** — honestly disclosed as
UNDERSTATING it (GraphST-style caveat). Across 11 platforms it wins only openST (1/11, 0.005 margin =
within noise) → **8 distinct winners**. Its inverted rank profile (ρ(contig,rank)=+0.56, worse on
contiguous) is NOT claimed as confirming the law — it is a handicapped config; the only claim is "doesn't
become universal". Also fixed a stub bug (added `numba.experimental`/`structref`) + a vacuous-pass guard
(earlier SpaGCN failed all 11 yet printed PASS — caught and hardened).

**Resilience fix:** scripts made per-platform resumable (`*_partial.json`) after the workspace repeatedly
interrupted heavy jobs — root cause diagnosed as machine oversubscription (load 90+/24 cores, swap full,
multiple concurrent Claude sessions + parent-project R/python deconvolution pipelines), not a code bug.

## ★★★★★★ R2 — label-free rule and statistical-rigor round (G001–G004, 2026-06-28): backend fairness

**Status: `ADOPTED-verified`.** The prospective-rule, statistical-rigor, and backend-fairness results are
pinned by `experiments/verify_manuscript.py` with their recorded limitations.

Second ultragoal round (plan archived at `.omc/ultragoal/archive/round1-G001-G010/`) closing the three gaps
a referee would block §7 on. All numbers machine-checked (`verify_manuscript.py` exit 0, incl. new R2 asserts).

**G001 — the rule is now PROSPECTIVE (label-free).** §6/§7 selected a method from *GT* contiguity — circular,
since the GT is what's unknown at deployment. `experiments/gtfree_proxy.py` (+ `gtfree_proxy_search.py`):
the OBVIOUS label-free proxies **invert** — Moran's I of expression PCs ρ_vs_adv = **−0.34**, unsupervised
kNN same-cluster fraction **−0.42** (vs GT oracle +0.72) — because raw-expression smoothness reflects
platform/cell-type structure, *opposite* to where spatial priors help (imaging domains are single-cell-noisy
yet contiguous). A PRINCIPLED proxy **`coh_gain`** (extra spatial coherence unsupervised clustering gains
from spatial smoothing) recovers the sign: vs GT contiguity **+0.56 (p=0.07)**, vs spatial advantage
**+0.55 (p=0.08)**, seed-robust [0.55,0.48,0.52]. Honest cost: recovers direction, not the oracle's
significance — the price of having no labels. Selected from a 6-candidate search, LOPO-validated in G003.

**G003 — the law survives its own stress tests.** `experiments/stat_rigor.py`: (1) 10k-bootstrap — GT
ρ=0.72, **95% CI [0.24,0.92], 98.7% positive**; coh_gain ρ=0.55, 94.8% positive (CI crosses 0 — honestly
marginal). (2) Leave-one-platform-out — GT predicts advantage *magnitude* held-out (LOO ρ=0.51) and the
"is a spatial method worth it" decision (adv>0.05) is right **0.82 vs 0.55 base rate**; the **label-free**
coh_gain rule still beats base rate out-of-sample (**0.73 vs 0.55**). (3) Confounds — contiguity is the
strongest marginal predictor and stays predictive controlling n_classes (0.55), dimensionality (0.86),
modality (0.80), and **all three at once (partial ρ=0.67)**: not a proxy for cluster count/dims/modality.

**G002 — backend fairness (DONE, real R mclust).** `experiments/native_baselines.py`: replaced the earlier
in-Python BIC-GMM analog with the **actual recommended clusterer — R's `mclust`** (BIC-based GMM model
selection), called via an `Rscript` subprocess (rpy2 fails to build on Python 3.13; CSV round-trip instead).
Insight: the numba stub only no-ops JIT and does NOT touch torch training, so the methods' embeddings were
already native-quality — only the clustering backend was non-native, which this fixes. Added to every
method's best-config, all 11 platforms (1-seed sensitivity). RESULT: real mclust genuinely used (wins 20/81
best-config cells, vs gmm-tied 44 / KMeans 17), shifts each method's mean ARI ≤0.016. The law **strengthens**
with the genuine clusterer: ρ(contiguity, advantage) = **+0.79 (p=0.004)**, slightly above the published
GMM-tied +0.72 (p=0.012); 6 distinct winners. verify_manuscript pins +0.79 and asserts mclust-R won >0.
(A bug — `os` used without import — crashed the first attempt and a self-matching pgrep watcher spun for
4.6h; both caught and fixed. The earlier Python-analog figures +0.73/25-of-81/≤0.022 are superseded.)

**G004 — science report.** Added a confound-controlled DLPFC benchmark section and five figures
produced by `experiments/make_figures.py` from the JSON artifacts.
Clean-room intact; verify_manuscript exit 0.

## ★★★★★ EXPANDED RESULT — 11 platforms, 6 methods, 3 seeds: the law is about spatial priors, not Tessera (§7, supersedes ★★★★/★★★)

**HARDENED round 2 (G009–G010, 2026-06-27):** at the user's request, added a 6th method of a NEW family —
**SpatialLeiden** (spatially-blended features + Leiden community detection tuned to k; clean, leidenalg/igraph,
brand-free) — bumped **seeds 2→3**, and added **ρ seed-robustness**. NSF skipped honestly (needs TF2.5/numpy1.19).
Local independent platforms are EXHAUSTED (verified: aether_serial_merfish = 3D stack of the MERFISH slice
already used; all openst_hnscc_* = sections of the one openST sample; niche_merfish_slice = byte-identical to the
MERFISH slice; Xenium = manifest-only) — declined to pad n with same-biology data. **Hardened numbers
(`expanded_bench.json`, supersede the round-1 figures below):** ρ(contig, spatial-prior advantage) = **+0.72,
p=0.012, SEED-ROBUST per-seed ρ ∈ [0.69, 0.77]** (LEAD, significant); ρ(contig, Tessera rank) = **−0.36, p=0.28
(ns)**; ρ(contig, SpaceFlow rank) = **−0.59, n=9, p=0.10 (marginal)**. **7 distinct winners across 11 platforms**
(SpatialLeiden newly wins CODEX). NEW finding — **SpatialLeiden has an INVERTED profile**: #1–#2 on the four
lowest-contiguity platforms (CODEX/MIBI/SlideseqV2/openST), #9 on the most-contiguous MERFISH — a graph-community
method excels exactly where the autoencoder/contrastive methods fail, the sharpest illustration that method–task
fit is real and method-specific. MERFISH winner across THREE independent runs = SpaceFlow→Tessera→SpaceFlow (it
flips — confirms the seed-level tie). All numbers machine-checked (verify_manuscript.py exit 0 incl seed-robust
asserts, SpatialLeiden-wins-CODEX, independent recompute, dup-guard); 23 pytest; clean-room intact; code-review
re-APPROVE. The round-1 detail (n=11 dedup, scipy fix, 5-method numbers +0.74/−0.26/−0.43) is retained below.



`experiments/expanded_bench.py` + `expanded_bench.json`. Two stress tests on the §5–§6 story: (i) add a
5th real deep SOTA — **SpaceFlow** (DeepGraphInfomax + spatial regularisation; runs under the numba stub,
brand confined to experiments/), a *different* spatial encoder from Tessera; (ii) add local platforms
(Slide-seqV2 hippo / CODEX spleen). ENTIRE sweep **re-run uniformly** (not appended) because adding a
method changes winners + ranks. 2 seeds, best-config, ranks within a consistent core panel, Spearman via
**scipy.stats.spearmanr** (mid-rank ties + p-values). **Code-review (G008 final gate) caught a CRITICAL
bug + 3 honest issues, all fixed before publishing:** (1) squidpy `mibitof` is a **byte-duplicate** of the
"colorectal" MIBI-TOF h5ad → dropped, **n=11 not 12**, duplicate-guard added; (2) hand-rolled Spearman used
ordinal not mid-rank ties → switched to scipy + p-values; (3) the MERFISH/osmFISH "wins" are within 2-seed
noise → reframed as ties; (4) SpaceFlow-rank ρ is n=9 → disclosed.

Three Spearman correlations (contiguity vs …, corrected):
- **spatial-prior advantage = +0.74, p=0.010** (n=6:+0.77, n=9:+0.68) — ROBUST + SIGNIFICANT, denominator-free.
  THE headline: spatial priors help iff GT contiguous. Independent scipy recompute in verify confirms it.
- **Tessera rank = −0.26, p=0.45 (ns)** (n=6:−0.71, n=9:−0.60) — **DISSOLVES & not significant**. Tessera is a
  backend-robust **generalist**, #2–#4 nearly everywhere (even #2 on LEAST contiguous CODEX, 0.022).
- **SpaceFlow rank = −0.43, n=9, p=0.25 (ns)** — a 2nd spatial encoder trends the same but underpowered
  (failed on the 3 protein/sparse panels).

**HONEST DISRUPTION / de-enthroning:** §5's "Tessera *uniquely* best on MERFISH *and* osmFISH" OVERCLAIMS —
the top of the contiguous-task leaderboard is a **seed-level TIE among spatial methods**: this run Tessera
nominally leads MERFISH (0.456±0.009 vs SpaceFlow 0.418±0.021) and osmFISH (0.447±0.013 vs STAGATE 0.425),
**but an independent seed pair had SpaceFlow win MERFISH (0.413 vs 0.405)** — the winner flips across seeds.
Defensible claim: on contiguous spatial-domain tasks a spatial method wins; Tessera is one of 2–3
competitive spatial methods (leads osmFISH more stably); the value of *spatial priors* (not any one method)
is what significantly tracks GT geometry. **6 distinct winners across 11 platforms** (SEDR/STAGATE/Tessera/
BANKSY-style/floor:nonspatial/floor:smoothed; SpaceFlow wins 0 this run).

Honest failures (logged, not hidden): SpaceFlow native HVG + GraphST fail on 32–36-dim protein panels;
large platforms subsampled to 16k. All §7 numbers machine-checked (verify_manuscript.py exit 0: n=11, 6
winners, the three ρ + p, independent scipy recompute, duplicate-guard, MERFISH-top-2-tie, osmFISH-Tessera).
23 pytest pass; clean-room intact. The review-driven correction itself embodies the comprehensive-evaluation
/ no-enthroning discipline: a duplicate + a tie-handling artifact + single-seed noise were inflating a
method-centric story; removing them left a cleaner, significant, mechanism-centric one.

## ★★★★ PREDICTIVE LAW — GT spatial contiguity predicts method choice (§6, superseded by §7 above)

`experiments/task_fit_law.py` + `task_fit_law.json`. Define GT spatial contiguity = mean fraction of
each cell's spatial kNN sharing its GT label. Across the 6 platforms it predicts:
- Tessera rank: Spearman(contiguity, rank) = **−0.71** (higher contiguity ⇒ Tessera ranks better).
- Spatial-prior value: Spearman(contiguity, spatial advantage) = **+0.77** (higher ⇒ spatial prior
  helps; at lowest contiguity seqFISH 0.624 it HURTS, −0.041).
Contiguity range: MERFISH 0.931 (Tessera #1) → MIBI-TOF 0.271 (cell-type, scattered). RULE: measure
your GT's contiguity; HIGH ⇒ spatial/boundary-aware method (Tessera best on highest-contiguity domain
tasks); LOW ⇒ non-spatial clusterer is floor+ceiling, smoothing hurts. This upgrades "no universal
SOTA" from a warning into actionable method selection. Caveats: ρ≈0.71–0.77 is a strong trend not a
deterministic law; STARmap is an exception (contig 0.92 but Tessera 4th); n=6 (more platforms sharpen
significance). `verify_manuscript.py` checks the rho values and public-study numbers.

## ★★★ SIX-PLATFORM RESULT — no universal SOTA; Tessera wins true spatial-domain tasks

`experiments/multi_platform_bench.py` + `multi_platform_bench.json`. 6 platforms (Visium DLPFC / seqFISH
embryo / MERFISH hypothalamus / STARmap cortex / osmFISH cortex / MIBI-TOF), 6 methods, GMM, 1 seed.

Best method per platform (ARI): DLPFC→SEDR 0.572 · seqFISH→floor:nonspatial 0.451 · **MERFISH→Tessera
0.267** · STARmap→BANKSY 0.586 · **osmFISH→Tessera 0.426** · MIBI-TOF→floor:nonspatial 0.25.

Two results: **(a) NO UNIVERSAL SOTA** — 4 different winners across 6 platforms; spatial prior helps on
4, hurts on 2; backend-confound direction ranges +0.155 to −0.004 (sign flips). **(b) METHOD–TASK FIT:
Tessera is the BEST method on MERFISH + osmFISH** — both true spatial-domain tasks (imaging, anatomical
domain/region GT, contiguous domains) — and worst on seqFISH/MIBI-TOF (cell-type GT, scattered). Rank
tracks GT geometry. **This OVERTURNS the earlier "Tessera is an honest negative"**: it is the best method
for the task it was designed for; DLPFC (layers, mid-pack 3/6) hid that. The user's insistence on
disrupting the written-in-stone DLPFC result + going cross-platform produced the real positive finding.

HARDENED (`experiments/hardened_bench.py`, publication-grade): 2 seeds (mean±std), each method at ITS
OWN best config (best of {KMeans,GMM}×{refine off,on}), GraphST added. Headline SURVIVES: 5 distinct
winners now (more diverse than single-seed's 4); Tessera still best on MERFISH (0.406±0.043) and osmFISH
(0.449±0.015); best-config even lifts Tessera to 2/7 on DLPFC (0.571, just under SEDR 0.577). GraphST
runs 5/6 platforms (fails on MIBI-TOF's 36-protein panel under the stub — recorded).

Remaining caveats: 2 seeds (not 10); absolute ARI modest on the hard MERFISH/osmFISH tasks; named-SOTA
not given their full native (mclust+refine) pipelines. The pattern (Tessera↔contiguous-domain GT) is
consistent across platforms and mechanistically expected.

## ★★ CROSS-PLATFORM RESULT — DLPFC conclusions FLIP on seqFISH (first cross-platform evidence)

`experiments/seqfish_bench.py` + `seqfish_bench.json`. seqFISH mouse embryo (19416 cells, 351 genes,
22 cell types) — different platform/tissue/GT-type/gene-count from DLPFC. Same benchmark, GMM:

- **"Spatial methods beat non-spatial floor" REVERSES.** floor:nonspatial (plain KMeans/GMM) ARI 0.451
  ranks #1 (it was LAST on DLPFC, 0.344); every spatial-smoothing method drops to 0.32-0.37. Cause:
  cell types are spatially scattered, so spatial smoothing erases signal.
- **Backend-confound direction REVERSES.** DLPFC: GMM helps SOTA (STAGATE Δ+0.20). seqFISH: STAGATE's
  GMM is WORSE (Δ−0.026) while non-spatial floor's GMM jumps (+0.115). Direction is dataset-dependent.
- **The DLPFC "SOTA" collapses.** STAGATE drops from DLPFC rank 1 (0.494) to seqFISH last (0.324).
- Tessera: ARI still mid (5/6), BUT ASW/DBI (internal geometry) again best of all methods on seqFISH
  and backend fairly robust (Δ0.021) — its two cross-platform-stable traits hold.

**Implication (upgrades the whole result):** method ranking, the value of spatial priors, and even the
backend-confound direction are all HIGHLY PLATFORM-DEPENDENT. A single dataset — including multi-section
DLPFC — cannot establish them; DLPFC's "SOTA" loses to plain KMeans on seqFISH. This refutes the
generality of every DLPFC-only conclusion above. (1 seed, 1 cross-platform dataset — directional, needs
more platforms to fully nail, but the flip is unambiguous.)

## Recorded result — refinement fix (the standard step the benchmark was missing)

Added spatial label refinement (majority vote over spatial kNN — the SOTA-standard post-step used by
STAGATE/GraphST/SpaGCN; `src/tessera_st/eval/refine.py`, optional `refine=` in `fit_predict`, +3 tests).
Cross-section validation (`experiments/refine_xsec.json`, 5 methods × 3 donors × GMM, refine on/off):

- **Tessera gains the MOST** (ΔARI +0.021, largest of any method; STAGATE/SEDR/floor/BANKSY +0.001–0.005).
  Same source as its backend robustness — a consistent embedding refines cleanly. Corroborates the
  backend-robust finding.
- **But the picture is unchanged**: Tessera+refine ARI 0.413 still LAST of 5 (vs SEDR 0.488 / STAGATE
  0.478 / floor 0.474). The single-section 151673 result (Tessera+refine 0.569 > STAGATE) was another
  single-section fluke,证伪 cross-section.

Net: a real, cross-section-robust improvement (Tessera is refinement's biggest beneficiary) and a more
reliable benchmark (all methods now use the standard post-step), but it does not make Tessera SOTA.
16 torch-free tests pass incl. new `test_refine.py`.

## ★ SCIENTIFIC RESULT — ultragoal complete (4/4 stories, quality-gated)

Durable output: a confound-controlled DLPFC spatial-domain benchmark (7 methods x 3 donors x 2 backends x 11 metrics x seeds). Three findings:

1. **Clustering backend is a dominant, usually-uncontrolled confound.** Max per-method ΔARI(GMM−KMeans)
   = 0.20 = the entire between-method ARI spread; STAGATE most sensitive (0.20), Tessera most robust
   (0.036). Method ranking is not identifiable without fixing the backend.
2. **A parameter-free neighbour-mean floor is competitive** (cross-section ARI 0.473, rank 3/7, within
   0.02 of best DL). New methods must beat floor:smoothed, not just floor:nonspatial.
3. **Structural ARI↔boundary_F1 Pareto frontier.** 4 mechanism-transfer experiments (BANKSY-aug,
   SEDR-VAE, GAT-attn, decoupled subspaces) all slid ALONG, none pushed OUTWARD → intrinsic trade-off.

**Tessera verdict (honest negative):** NOT a SOTA-beating domain method (cross-section ARI 5/7;
single-section boundary_F1 lead does not generalise → 3/7). Its only cross-section-robust, non-circular
distinctive property is **backend-robust embeddings** (ΔARI 0.036).

Quality gate: `verify_manuscript.py` exit 0 (18 headline numbers match JSON artifacts); code-reviewer
APPROVE (0 critical/high/medium, no correctness bug, honest-negative intact); ai-slop clean (1
overstatement fixed). Ledger: `.omc/ultragoal/`. Evidence: `experiments/{rigorous_bench,
pareto_frontier,tessera_verdict,full_panel_v2,sota_stagate_151673}.json`. Scope: single dataset
(DLPFC), 3/6 sections — cross-platform is future work.

## Evaluation caveat — the ARI-based verdicts above are semi-circular, not final

Every "Tessera vs SOTA / vs baseline" and "ARI peaks then falls" statement above leans on ARI/NMI
against the DLPFC manual layers. That ground truth is itself an expression+histology-derived
clustering — weakly circular, not physical truth. So:

- "0.42 is below SOTA 0.5-0.6" means "less agreement with *that* annotation", NOT "wrong". Do not
  one-stroke-kill the method on it.
- "ARI falls while ASW/embedding-spread rises" = two semi-circular metrics diverging; it tells us
  the late-training embedding organises by something other than the manual layers, which may be
  noise OR a real non-layer structure. ARI alone cannot adjudicate.
- The standing fix is to triangulate across a **full multi-metric panel** — including a less-
  circular (NOT non-circular) marker-prior view (`marker_purity`, itself conditioned on a marker
  list that drifts across platform/sample) — rather than ranking, or steering training, by any
  single metric. Earlier ledger lines calling marker_purity "external / non-circular" overstate it:
  it lowers circularity, it does not remove it.

No verdict (positive or negative) graduates from a single metric. Comprehensive panel + external
anchor + cross-sample reproducibility are required.

## Evidence-artifact tiers (current 40-JSON inventory)

The public evidence surface is deliberately narrower than the development history. Filenames below
refer to the 40 current `experiments/*.json` artifacts.

**Tier 1 — verifier-pinned mainline evidence (19).** These artifacts may support public claims only with
the bounded wording and caveats asserted by `experiments/verify_manuscript.py`:

- `expand_data.json`, `expanded_bench.json`, `gtfree_proxy.json`, `hardened_bench.json`,
  `mechanism_ablation.json`, `mechanism_decoupled_v2.json`, `mechanism_synth.json`,
  `multi_platform_bench.json`, `native_baselines.json`, `pareto_frontier.json`,
  `proxy_search_corrected.json`, `rigorous_bench.json`, `robustness_audit.json`, `seqfish_bench.json`,
  `spagcn_native.json`, `spagcn_panel.json`, `stat_rigor.json`, `task_fit_law.json`, and
  `tessera_verdict.json`.

**Tier 2 — development-stage only (21; not public evidence).** These are partial/checkpoint,
superseded v1, or early single-section artifacts retained for auditability. They must not be cited as
finished-study evidence or used to introduce a headline number:

- `bayesspace_bench.json`, `bayesspace_bench_partial.json`, `dlpfc_151673.json`,
  `dlpfc_151673_3seed.json`, `dlpfc_151673_full.json`, `expand_data_partial.json`,
  `final_panel_151673.json`, `full_panel_v2.json`, `gtfree_proxy_search.json`,
  `mechanism_decoupled.json`, `mechanism_decoupled_v2_ckpt.json`, `multisection_panel.json`,
  `native_baselines_partial.json`, `refine_xsec.json`, `sota_panel_151673.json`,
  `sota_stagate_151673.json`, `spagcn_native_partial.json`, `spagcn_panel_partial.json`,
  `synth_ablation.json`, `task_fit_law_extended.json`, and `tessera_gmm_151673.json`.

Naming note: the public study uses **leave-one-dataset-out (LODO)**; historical JSON keys remain
`lopo_*` because the scripts originally called the same split leave-one-platform-out. The study defines
this mapping at first use; the keys are not a second validation protocol.

## Honest-negative policy

If the ablation shows a component does not help (e.g. multi-scale fails to beat single-scale on
DLPFC), that is recorded here as a useful negative result and the component is reported as such —
never quietly dropped or reframed.
