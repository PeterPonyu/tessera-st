# READY_TO_UPLOAD / READY_TO_FORMAT — tessera-st

**Filed:** 2026-08-10  
**Stop line:** human submission steps (ScholarOne / IEEE Author Center). No agent upload.

## Machine gates (this pass)

| Gate | Status | Notes |
|---|---|---|
| Venue LOCKED | **Done** | `VENUE-LOCK.md` — TCBB primary |
| Differentiation audit 1–4 | **Done** | `docs/DIFFERENTIATION_AUDIT_2026-08-10.md` |
| Cover letter (4 bullets) | **Done** | `COVER-LETTER.md` — fill funding before paste |
| `SUBMISSION-KIT.md` synced | **Done** | Target venue ≠ UNDECIDED |
| `submission_flat/` text synced | **Done** | Related Work + Discussion harden |
| PDF recompile (upload flat) | **PASS** rc=0 | `submission_flat/paper.pdf` **13 pp** |
| PDF recompile (main tree) | PASS rc=0 but **21 pp** | Not upload surface; diverge noted |
| `experiments/verify_manuscript.py` | **PASS** exit 0 | ALL CHECKS PASS |
| Undefined refs / overfull | **PASS** (flat) | 0 undefined / 0 multiply-defined; cosmetic under/overfull only |
| scivcd figure audit | Optional / non-blocking | polish only unless critical |

## Human remaining (submission)

1. Email TCBB editorial office: reconfirm traditional/\$0 hybrid still offered post-rename.
2. Optional: AMU library Clarivate/JCR screenshot for SCIE + IF before quoting metrics externally.
3. Fill **Funding** line in `COVER-LETTER.md`.
4. IEEE template migration / ScholarOne upload when ready (body still IEEEtran-compatible; confirm class).
5. Accept ~\$220 voluntary overlength if final layout stays at 13 pp (or trim after acceptance).
6. Do **not** start Boundary-CA / new GPU experiments as a submit gate.
7. If any A→B trigger fires → follow `VENUE-LOCK.md` fallbacks (CBC, then BMC/GigaScience with APC auth).

## Blockers

| Item | Blocking submit? |
|---|---|
| Editorial email reconfirm | Soft — LOCKED on IEEE APC PDF; email before upload |
| Funding statement blank | Yes for form completeness |
| Clarivate MJL UI empty for old ISSN | Soft — use library login; do not invent IF |
| NARGAB SCIE contradiction | Resolved for this paper: **not primary** |
