# Follow-up evidence and source status — 11 October 2026

**The historical-runner gap is resolved for all five follow-up modes.** Four
modes had referenced two earlier versions of the retained runner; both versions
were recovered byte for byte and match the SHA256 values in their original
receipts. The whole-section mode already matched the retained runner.

| Modes | Recorded producer SHA256 | Preserved source |
|---|---|---|
| scale, re-scoring, nulls | `9261ceb03afebf4735769bd28bd5672f4093e278e724505a9c2bcf48e41a2915` | `source/provenance/followup-runners/9261ceb03afebf4735769bd28bd5672f4093e278e724505a9c2bcf48e41a2915/run_followup.py` |
| transfer | `d1d90c72da262b51283ad95f64fbaab1676d452063e68b5e1b8d29acf0d53a6b` | `source/provenance/followup-runners/d1d90c72da262b51283ad95f64fbaab1676d452063e68b5e1b8d29acf0d53a6b/run_followup.py` |
| whole sections | `c96d933da9b71ec039df0c7dbfe1eaf5c7f5ec1c9d07918858300590b19aa538` | `source/scripts/run_followup.py` |

Recovery used the retained runner as evidence. Removing only its later
whole-section extension yields the transfer version; removing the transfer
extension as well yields the earlier scale/re-scoring/null version. Candidates
were accepted only after their complete bytes matched the recorded SHA256.
Every analysis/helper function in the recovered runners is identical to its
retained counterpart; only later extensions and entry-point choices differ.
The recipe and bindings are preserved in
`source/scripts/recover_followup_runners.py` and
`source/provenance/followup-runners/recovery.json`.

Verify the recovery read-only from the revision root:

```bash
PYTHONDONTWRITEBYTECODE=1 python source/scripts/recover_followup_runners.py --verify
```

The old gate-diagnostic producer is also preserved at its recorded hash before
extending the auditor. The current auditor checks the saved diagnostics against
all 39 original prediction files without rewriting their receipt or CSV.
All original run receipts and numerical outputs remain unchanged. There was no
new model fit, separately executed replay, or new biological measurement.

The retrospective audit verifies 1,170 scale rows, 234 re-scoring rows, 39 null
rows, 108 whole-section rows, 20 transfer rows and 39 saved-gate profiles. It
checks nested patient exclusion, summary arithmetic, output hashes and source
availability. Exact source recovery establishes which source bytes the receipts
name; table consistency checking is distinct from executing those analyses again.
Because all requested sources were recovered exactly, the requested fallback
reruns were unnecessary.

The preceding missing-source state is preserved in
`backups/before-runner-recovery-20261011-90pfenpx/`. The prior reply's technical
blocker was source availability, not a failed numerical audit or missing inputs.
That blocker is now closed. The independent-validation limitations below remain.

Interpretation boundaries:

- Bounded lung/BCC scale controls: each patient is excluded from its own
  configuration selection, but selection folds overlap. Small-cohort sign-flip
  enumeration is descriptive and does not remove this dependence.
- Whole sections: the same patients are expanded without the 6,000-cell cap;
  controls use fixed cohort-modal settings informed by bounded analyses. This
  is a sampling sensitivity comparison, not independent nested validation.
- Transfer: transformations and category centroids are source-only. The fixed
  `k=200` control was informed by earlier lung analyses and is descriptive;
  there is no claim of independent per-fold scale tuning.
- Common-class K: classes of at least 1% prevalence determine K, but all valid
  labels remain in ARI. BCC outcomes are mixed at this resolution.
- Display: rare classes are grayed out only for maps and cluster-color matching;
  original predictions, scores and numerical primary ledgers are unchanged.

The additional 11 October gate-weight analysis is CPU post-processing of 39
hash-verified original predictions, with its own script/output receipt. It
measures relative weights over directed multiset edges, not new fitted models
or new patients. It demonstrates why small gate magnitudes cannot be interpreted
as uniform neighbor weights. The no-gate ablation remains the relevant test of
whether learned gating improved agreement.

No new external release, upload or submission was made. All author-side checks
remain distinct from independent scientific review.

The later external comparison is separate from this recovery. Eight published
HER2ST sections with pathologist spot labels were scored under
`validation-external/PROTOCOL.md`. That run fits Tessera only on those sections
and does not change the receipts or numerical outputs covered above. Its result
and the limits that still block submission are in
`validation-external/VALIDATION_RESULT.md` and `validation-extension/RESULT_EXTENSION.md`.
