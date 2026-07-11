# SUBMISSION-KIT — tessera-st flagship manuscript

Generated: 2026-07-02 (mechanical-fixes-only polish pass; no new compute).

## Manifest
- Title: *A causal law for spatial-domain detection: the value of a spatial prior is set by
  the ground truth's spatial contiguity*
- Author: Zeyu Fu (ORCID 0009-0001-8329-0108), Army Medical University, Chongqing, China.
- Framing: HONEST-NEGATIVE / causal-law (CLAIM_LEDGER §R5). Tessera is a worked example, NOT
  the hero; it is NOT SOTA-beating (cross-section ARI rank 5/7, boundary-F1 rank 3,
  backend-robustness rank 1). ARI is explicitly semi-circular. Do not re-inflate.
- Main source: `manuscript/paper.tex`
- Bibliography: `manuscript/refs.bib` — 30 entries (was 31; orphan `mcinnes2018umap` removed).
- Compiled PDF: `manuscript/paper.pdf`
- Pages: 19
- Backups: `manuscript/paper.tex.prepolish.bak`, `manuscript/refs.bib.prepolish.bak`

## Target venue
- UNDECIDED — pitched at the Nature Methods / Nature Communications / Nature Genetics tier.
- Keep prose, framing, and formatting VENUE-AGNOSTIC. Do not hard-commit to one journal's
  template, word limit, or house style until a venue is chosen.

## Quality gates PASSED this pass
- LaTeX compile: rc=0 across `lualatex -> bibtex -> lualatex -> lualatex` (Makefile `pdf` seq).
- `experiments/verify_manuscript.py`: ALL VERIFICATION CHECKS PASS (headline numbers incl.
  ari_rank/boundary_f1_rank/backend_robustness_rank and causal-law rhos intact).
- Undefined references: 0. Multiply-defined labels: 0.
- BibTeX: no warnings; corrected `sun2025smoothness` / `yuan2024benchmark` render in
  `paper.bbl`; removed `mcinnes2018umap` absent from `paper.bbl`.

## Known cosmetic items (non-blocking)
- 2 Overfull \hbox, both < 20pt: title (12.78pt, line 47) and tab:related-benchmarks
  (5.04pt). Not fixed (clearing them is layout judgment: title `\\`-rebreak or `\sloppy`,
  table column rebalancing). Safe to submit as-is; optionally clear before camera-ready.

## Remaining PRE-SUBMISSION actions (deferred, need user input or a ledger pass)
1. BRCA provenance (BLOCKER for the datasets count): supply the real accession for the 11th
   dataset (BRCA breast-tumour Visium), add a `refs.bib` entry + `\citep`, add it to the
   Methods §Data enumeration (`paper.tex:192-197`) and Data-availability (`:595-599`), and to
   `BASELINE_REFERENCES.md`. Currently the prose lists only 10 of the stated 11. Do NOT
   fabricate a citation. (LESSONS D2, D3)
2. coh_gain uncorrected p (0.079 vs 0.083): decide relabel-vs-repoint and route through the
   CLAIM_LEDGER gate + `verify_manuscript.py` line 275. Left at 0.079 to keep the machine gate
   green; the asymptotic/permutation-null mismatch is real but non-inflating (both >0.05).
   (LESSONS D1)
3. fig:ablation caption: add the half-sentence noting its absolute ARIs are a separate
   case-study run, not comparable to the cross-dataset panels (prevents a false "Tessera beats
   the winner" read). Highest-value honest-framing item. (LESSONS D4)
4. CLAIM_LEDGER.md:315: refresh stale "12pp / 0 overfull" to "19pp / 2 overfull (title
   12.8pt; tab:related-benchmarks 5.0pt) / 0 undefined". (LESSONS D5)
5. Author-count / affiliation / competing-interests / funding statements: confirm present and
   venue-appropriate once the venue is chosen.

## Zero-APC / venue note
- The corresponding author (China-based) is NOT eligible for income-based APC waivers; assume
  $0 free open-access routes for this paper. Factor APC into the venue decision: prefer a
  no-fee or read-and-publish-covered route, or a subscription/transformative journal where the
  author's institution has an agreement. Do NOT assume a fee waiver will be granted.
- Venue remains UNDECIDED; keep the manuscript venue-agnostic until the fee/route is confirmed
  alongside the scientific-fit decision.

## Venue reassessment (2026-07-02, web-verified)

**DOWNGRADE the Nature Methods/Commun/Genet tier — structurally wrong for an honest-negative paper.**
Those venues (plus Genome Biology, Cell Reports Methods) desk-reject-triage on perceived broad
significance and positive/novel findings, publish low volumes, and charge $4.3k–$5.7k APC. A
"no-universal-winner" causal-law/benchmark paper is exactly the genre they under-publish.

- **Best EDITORIAL fit (all require an APC — no $0 route, no China waiver):**
  - *BMC Bioinformatics* (~$2,890) — high-volume home for multi-method "no clear winner" benchmarks.
  - *GigaScience* (~$2,500–2,638) — reproducibility/FAIR ethos matches the 11-dataset + claim-ledger
    discipline (verify current volume; sources conflict 61–178/yr).
  - *NAR Genomics & Bioinformatics* (~$2,633) — has a named "Methods and Benchmark Surveys" article type.
  - *F1000Research* — the only venue whose policy explicitly welcomes negative/null results (open
    post-pub review removes gatekeeping risk); lower prestige.
  - **PeerJ** (Bioinformatics & Genomics section) — **cheapest genuine option: ~$599–799 one-time
    lifetime membership, reusable across future papers**; culture friendly to rigorous non-glamour work;
    lower ST-community visibility.
- **If holding hard $0 (subscription-hybrid only):** IEEE/ACM TCBB (verify fee page) or an Elsevier
  hybrid (*Computational Biology and Chemistry* / *Computers in Biology and Medicine*). Tradeoff: weaker
  editorial fit for a negative-result paper than the venues above.
- **Unlock:** an Army Medical Univ. / CALIS Read-&-Publish deal would make BMC/GigaScience/NARGAB $0 —
  check with the library before venue-lock.
- **SCI GATE (overrides cost/fit — binding for Army Medical University):** only **SCIE / Web-of-Science
  Core** venues count for evaluation. Per the portfolio venue strategy, **NARGAB and Bioinformatics
  Advances are ESCI-only → they do NOT count**; PeerJ and F1000Research SCIE status must be verified
  against AMU's list before use. BMC Bioinformatics and GigaScience are SCIE (confirm), but carry an APC.
- **RECOMMENDATION (revised, SCI-first):** **IEEE/ACM TCBB** (SCIE Q1, $0 subscription route) is the
  primary — it satisfies BOTH the $0 constraint and the SCI requirement. Consider PeerJ (one-time
  ~$599–799) or BMC Bioinformatics only if (a) their SCIE status is confirmed AND (b) a cost is
  authorized. **RECONFIRM TCBB's $0 subscription route survived ACM's 2026-01-01 mandatory-OA shift
  before submitting** (IEEE/ACM co-publish). Authoritative plan: `.future-directions/VENUE_STRATEGY_SCI.md`.
  **DECISION PENDING (user).**
