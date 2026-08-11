# VENUE-LOCK — tessera-st

**Status:** `LOCKED`  
**Locked date:** 2026-08-10  
**Decision:** Option A (SCI-first + $0 traditional route)  
**Body formatting:** remains venue-agnostic until IEEE template migration at upload.

## Primary

| Field | Value |
|---|---|
| **Venue** | IEEE Transactions on Computational Biology and Bioinformatics (TCBB; formerly IEEE/ACM TCBB; IEEE acronym **TCBBIO**) |
| **WoS** | SCIE (portfolio strategy + aggregator corroboration; Clarivate MJL ISSN probe for 1557-9964 returned empty UI this session — library login reconfirm recommended before quoting IF in cover letter) |
| **Route** | Traditional / subscription (hybrid) — **no APC** if Gold-OA not elected |
| **Optional Gold-OA APC** | \$2,800 (IEEE 2026 APC list, updated 2025-12-30) |
| **Page budget** | 12 pages free allowance; **\$220/page** overlength beyond 12 (effective date noted 2018-02-18 on IEEE list) |
| **Current MS length** | **Upload surface `manuscript/submission_flat/paper.pdf` = 13 pp** (compile 2026-08-10). Main-tree `manuscript/paper.pdf` currently expands to 21 pp (diverged content/float packing) — **do not upload main-tree PDF**; keep flat pack as upload artifact or re-sync carefully. |
| **Page contingency** | **Accept voluntary ~\$220 for 1 page overage** (13 vs 12 free) rather than pre-submit trim; optional post-acceptance trim if IEEE layout expands further. Do **not** block submission on a ≤12-page rewrite. If a future sync reintroduces the 21 pp main-tree expansion, stop and trim before upload (9×\$220 ≈ \$1,980). |
| **ACM OA risk** | Journal now IEEE-titled (Wikipedia / publisher: ACM co-publish ended ~2025). 2026 IEEE APC PDF still lists **TCBBIO = Hybrid Open Access**. Human should email `tcbb-eic@computer.org` (or current EIC contact) once before upload to reconfirm traditional route still offered. |

### Evidence pointers (dated 2026-08-10)

1. **IEEE 2026 APC PDF** (Hybrid OA list, updated 2025-12-30): row `127 TCBBIO … Hybrid Open Access $2,800 … $220.00 … 12pgs` — confirms hybrid + optional OA + overlength schedule. Source mirror used this session: [NCUE CONCERT IEEE APC PDF](https://jumper.ncue.edu.tw/file/downloadFile/2026%20IEEE%20Publication's%20APCs_Fully&Hybrid%20OA-2025.12.pdf).
2. **Portfolio ladder:** `active/.future-directions/VENUE_STRATEGY_SCI.md` §tessera-causal-law (TCBB Tier-A SCIE+$0).
3. **NARGAB:** aggregator (wos-journal.info) shows **ESCI only** (Genetics & Heredity / Mathematical & Computational Biology — ESCI). **Not primary.** Remains paid Tier-B / non-SCIE for AMU unless library Clarivate screenshot proves SCIE promotion.

## Ordered fallbacks

| Rank | Venue | WoS (2026-08-10 live check) | Net cost | When to use |
|---|---|---|---|---|
| 1 (primary) | IEEE TCBB / TCBBIO | SCIE (reconfirm via AMU Clarivate login) | \$0 traditional; optional OA \$2,800; ~\$220 if 13th page billed | Default |
| 2 | *Computational Biology and Chemistry* (Elsevier) | SCIE (Biology; CS Interdisciplinary Apps) | \$0 subscription/green; optional Gold-OA ~\$3,150 | A→B if TCBB \$0 route dies **or** desk out-of-scope |
| 3a | *BMC Bioinformatics* | SCIE (Peeref / Springer index list) | ~\$2,890 APC; no China auto-waiver | Genre-fit paid Option B; needs APC auth or R&P |
| 3b | *GigaScience* | SCIE (Multidisciplinary Sciences) | ~\$2,500–2,900 APC band (reconfirm OUP sticker) | Same Option B lane; FAIR/repro ethos |
| 4 | *PeerJ* (Bioinformatics & Genomics section) | SCIE (Multidisciplinary Sciences, IF ~2.9 aggregators; PeerJ site claims SCIE) | Membership ~\$599–799 lifetime preferred over per-paper APC | Option C only if user prefers membership economics **and** AMU list accepts PeerJ SCIE credit |
| — | *NAR Genomics & Bioinformatics* | **ESCI** (live aggregator; in-repo ESCI warnings dominate) | ~\$2,633 APC | **Not primary.** Do not submit as SCI-credit primary without same-day Clarivate SCIE proof |

## A→B pivot triggers (any one → leave TCBB for CBC, then paid BMC/GigaScience)

1. Live check / editorial reply shows TCBB no longer offers traditional/\$0 hybrid (mandatory OA / APC unavoidable).
2. TCBB editorial desk indicates out-of-scope for causal-law / benchmark meta-analysis.
3. User explicitly authorizes BMC/GigaScience APC (or library confirms R&P = \$0).

**Invalidation (do not pursue as primary):** Nature Methods / Communications / Genetics (honest-negative desk-reject + high APC). NARGAB as primary without Clarivate SCIE confirmation.

## Trigger status (as of lock)

| Trigger | Status |
|---|---|
| TCBB hybrid/\$0 intact on IEEE APC materials | **Clear** (2026 list) |
| Editorial email reconfirm | **Human remaining** (non-blocking for LOCKED; do before upload) |
| AMU Clarivate screenshot for IF/quartile quote | **Human remaining** (non-blocking; do not invent IF in cover letter) |
| Library R&P for BMC/GigaScience | Unknown — only matters if pivoting to Option B |

## Pitch constraints (binding)

- Cover letter: **causal-law + controlled contiguity**, honest non-SOTA (ARI rank 5/7), Spotscape/SpatialESD differentiation, why TCBB.
- Do **not** re-inflate Tessera ARI SOTA / universal-winner claims.
- No Boundary-CA / new experiments as submission gate.
