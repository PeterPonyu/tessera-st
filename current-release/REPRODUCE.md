# Reproduction levels and exact portable commands

The archive's primary supported check is saved-result verification, with no new
model or clustering fits. From a fresh extraction with the recorded supported
Python environment, run the four commands in README.md. Files checked by the
numerical verifier remain unchanged. `MANIFEST.sha256` binds the final package;
it should be regenerated once the final document/citation overlay is complete.

For an independent experimental rerun, use a fresh output root outside the
archive. The package-only wrapper imports the original byte-preserved producers
and changes only input/output locations and the acquisition adapter. A new run
receives its own portable wrapper receipt and its producer-generated scientific
receipts. Packaging tested import/input preflight only; no command below was
executed as a new fit during this closure. New fitting can differ numerically
from the historical CUDA results and is not represented as a replay.

Primary fits and patient-held-out transfer can use the included frozen derived
inputs, without obtaining the full raw assays:

```bash
python portable/run_analysis.py primary --device cpu --output-root /path/to/new-primary --check
python portable/run_analysis.py transfer --device cpu --output-root /path/to/new-transfer --check
```

After reviewing preflight output, omit `--check` to fit. The unchanged producers
keep their locked estimator, graph, preprocessing inputs, seeds and targets. The
fresh output root must not exist. CUDA is optional via `--device cuda`; the
historical platform did not guarantee bitwise deterministic CUDA reductions.

HER2ST/extension fitting requires the checksum-bound acquired files:

```bash
python portable/acquire.py her2st --source-dir /path/to/manual/her2st --destination /path/to/verified-her2st
python portable/run_analysis.py her2st --raw-dir /path/to/verified-her2st --output-root /path/to/new-her2st --check
python portable/run_analysis.py enhanced --raw-dir /path/to/verified-her2st --output-root /path/to/new-enhanced --check
python portable/run_analysis.py hardening --raw-dir /path/to/verified-her2st --output-root /path/to/new-hardening --check
python portable/run_analysis.py dev --output-root /path/to/new-dev --check
```

`portable/acquire.py` accepts `--download` and an optional `--proxy URL`; direct
requests are the default. It rejects size/SHA256 differences. The HER2ST historical
J1 missing-count exclusion remains enforced. The CRC acquisition command also
checks publisher MD5, and has no author-local proxy requirement:

```bash
python portable/acquire.py crc --source-dir /path/to/manual/crc --destination /path/to/verified-crc
python portable/run_analysis.py crc --output-root /path/to/new-crc --check
```

The CRC wrapper refits from the three compact, frozen PCA inputs supplied here;
this does not repeat raw-assay preprocessing. To rebuild those inputs, use the
original `validation-independent-crc/run_pilot.py` preparation routines with the
exact four verified raw ZIP files and original protocol/fetch receipts in a
fresh copy of that validation tree. The original protocol and amendment locks
must remain intact. The original fetch script and historical workspace gate are
preserved for provenance; use `portable/acquire.py` for portable acquisition.

For breadth controls or original primary assay preprocessing, arrange the exact
checksum-bound files under a new directory using the `lab_relative_path` layout
from `provenance/upstream-inputs.json`:

```bash
python portable/run_analysis.py breadth --lab-root /path/to/assay-sources --output-root /path/to/new-breadth --check
python portable/run_analysis.py prepare --lab-root /path/to/assay-sources --output-root /path/to/new-inputs --check
```

The primary raw source manifest includes exported datasets as well as source
assays. Some exports and annotations were prepared earlier; the source hashes
and adapters define those inputs more narrowly than a publication name or DOI.
The complete original raw-to-export processing and vendor assets are not
reimplemented by this archive. The saved derived inputs are sufficient for the
listed saved-result checks and derived-input refits.

Large raw assays, large whole-section expression-derived input matrices, raw
archive contents checks and author-local historical backup integration checks
are outside the default verification. Missing historical fitted embeddings and
partitions are disclosed in `provenance/verification-scope.json`. Their saved
score/selection arithmetic can be checked, but a checksum-bound receipt alone
does not reconstruct missing scientific predictions.
