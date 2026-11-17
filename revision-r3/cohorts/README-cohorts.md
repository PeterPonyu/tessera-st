# Round-3 cohorts: fetch, inspect, prepare, test

This folder fetches and prepares the three cohorts of PROTOCOL-R3, section 3.2: mouse spinal-cord MERFISH, the four colorectal Visium patients outside the pilot, and Wu et al. breast Visium. Each unit becomes one bounded input, `data-r3/<cohort>/<dataset>/input.npz` with `input.json`, and `data-r3/index.json` lists every dataset.

## Preconditions

- **Protocol.** PROTOCOL-R3 was frozen at commit `0d2326e` (17 Nov 2026, 01:00 PST) and deposited as doi:10.5281/zenodo.24303149. `fetch_r3.py --execute` refuses unless that commit exists and the protocol has no uncommitted change. Every receipt records the freeze commit.
- **Approval.** Each download needs the author's yes, naming the file, its source and its size (the tables below).
- **Connection.** Downloads use a direct connection and ignore `HTTP(S)_PROXY`.
- **Interpreter flags.** Run Python with `-I -B`. The downloads are untrusted input, so `-I` isolates the interpreter, and `-B` writes no bytecode into the repository.

```bash
PY="/home/zeyufu/miniconda3/envs/dl/bin/python -I -B"
C=/home/zeyufu/Desktop/singlecell-genomics-research/projects/tessera-st/repository/revision-r3/cohorts
D=/home/zeyufu/Desktop/singlecell-genomics-research/projects/tessera-st/data-r3
```

## 1. Test and dry run (no data)

```bash
$PY -m pytest $C/test_cohorts_r3.py -q -p no:cacheprovider   # 19 tests on synthetic stand-ins, about 5 s
$PY $C/fetch_r3.py                                           # Zenodo API metadata only, about 10 s
```

The dry run prints every planned file with its size and md5, and checks each against the frozen manifest in `common_cohorts.COHORTS`. For the colorectal cohort it re-derives the smallest-archive choice from the live listing.

## 2. Fetch (one approval per cohort)

| Command | Files | Bytes |
|---|---|---|
| `$PY $C/fetch_r3.py --execute spinal` | `MERFISH_spinal_cord_resolved_0718.h5ad`, `SNI_merged_0917.h5ad` | 214,871,813 (205 MiB) |
| `$PY $C/fetch_r3.py --execute colorectal` | `SN048_A121573_Rep1.zip`, `SN048_A416371_Rep2.zip`, `SN123_A798015_Rep1.zip`, `SN123_A938797_Rep1_X.zip` | 473,346,239 (451 MiB) |
| `$PY $C/fetch_r3.py --execute wu_breast` | `filtered_count_matrices.tar.gz`, `spatial.tar.gz`, `metadata.tar.gz` | 210,203,822 (200 MiB) |

**Where files land.** Files go to `$D/raw/<cohort>/`, each with a `<file>.receipt.json` holding the URL, bytes, md5, sha256, UTC start and end, and the freeze commit. The colorectal selection record is `$D/raw/colorectal/selection.json`.

**Rerunning.** A rerun resumes `.part` files, and skips a file whose receipt matches.

**Time.** Download time has not been measured, because no data were fetched. At 2–10 MB/s, expect about 0.5–2 min each for spinal and Wu, and 1–4 min for colorectal.

**What is not downloaded.** The colorectal pathology archive is not downloaded again. It is read from `revisions/tessera-20261008-expanded/validation-independent-crc/raw/Pathology_SpotAnnotations.zip`, and its md5 and its existing sha256 receipt are verified first.

## 3. Inspect (schema only; seconds each)

```bash
$PY $C/inspect_r3.py $D/raw/spinal/MERFISH_spinal_cord_resolved_0718.h5ad
$PY $C/inspect_r3.py $D/raw/spinal/SNI_merged_0917.h5ad
$PY $C/inspect_r3.py $D/raw/wu_breast/metadata.tar.gz --member '*_metadata.csv'
$PY $C/inspect_r3.py $D/raw/wu_breast/filtered_count_matrices.tar.gz     # member list only
$PY $C/inspect_r3.py $D/raw/wu_breast/spatial.tar.gz                     # member list only
```

Each command writes `<file>.inspect.json`. The receipt gives column names, dtypes, and the numbers of unique and missing values. For text, categorical and boolean columns with at most 60 values, it also lists those values with their counts. Matrices are described by shape and dtype only.

Read the spinal receipt and fix these choices before preparation:

- **Units and frames.** The obs columns of the mouse (unit) and the section (frame).
- **Coordinates.** Either `obsm:<key>` or `obs:<x>,<y>`.
- **Cell types.** The cell-type column of every deposited level, coarsest first; each level becomes a target.
- **Matrix.** `--matrix` is `X`, `raw` or `layer:<name>`. Prefer raw counts: `input.json` records `count_normalization` as true only when the matrix held integer counts.
- **Branch** (protocol 3.2):

| Branch | Condition | Action |
|---|---|---|
| A | A per-cell lamina column exists | Pass `--lamina` |
| B | A per-cell relative-depth column exists, and the paper states numeric lamina boundaries | Copy the boundaries into `LAMINA_BINS` in `prep_spinal.py`, cite the page and figure in the commit, then pass `--depth`. The script refuses while the table is empty |
| C | Neither exists, or the paper states no numeric boundaries | No lamina target. Pass `--confirm-branch-c` if an obs column name mentions laminae or depth; those columns are recorded as `lamina_like_columns_not_used` |

## 4. Prepare

```bash
$PY $C/prep_spinal.py --file healthy --mouse <col> --section <col> --coords obsm:<key> \
    --celltype <level-1 col> --celltype <level-2 col> [--lamina <col> | --depth <col> | --confirm-branch-c] \
    [--matrix layer:<counts>]                       # 10 datasets spinal/healthy-<mouse>, about 1-2 min
$PY $C/prep_spinal.py --file sni --group <condition col> ...   # same columns; descriptive transfer only
$PY $C/prep_colorectal.py                           # 4 datasets colorectal/<patient>, about 3-4 min, about 7 GB RAM
$PY $C/prep_wu.py                                   # 6 datasets wu_breast/<patient>, about 5-6 min, about 7 GB RAM
```

The time estimates use the frozen preprocessing on synthetic matrices of the same size: a 4,992 × 36,601 Visium section took 38 s at a 6.7 GB peak, and a 6,000 × 500 MERFISH unit took 4.5 s.

**Contents of each dataset.**
- **`input.npz`:** `x` (float32, the frozen `run_her2st_validation.preprocess` PCA, up to 50 components), `coords` (float64), `field` (frame identifiers), `y_<target>` (int64, −1 missing) and `classes_<target>`.
- **`input.json`:** source DOI and file sha256, unit, frames, selection record, class counts and provenance per target, the spinal protocol branch, and coordinate units with micrometres per unit.
  - Visium: 100 µm divided by the median nearest-neighbour distance.
  - MERFISH: 1 only when the file states µm, in a `uns` unit entry or a coordinate-name suffix; null otherwise.

**Targets.**

| Cohort | Targets |
|---|---|
| Spinal cord | `celltype_l1`, `celltype_l2`, … in every branch, plus `lamina` in branch A or B |
| Colorectal | `all`: pathologist labels, joined by the pilot's own `run_pilot.load_raw`, imported unchanged |
| Wu breast | `all`: `Classification` as released; `tumor`: a class is tumor if its name contains "cancer", "DCIS" or "Invasive", in any case. The mapping is in `input.json` |

Missing labels are empty values, NaN, the frozen encoder's tokens (`nan`, `none`, `<na>`, `unknown`, `na`) and `unlabelled`.

**Exclusions.** A colorectal section that fails a pilot eligibility check is written to `colorectal/exclusions.json` and is not replaced; the script then exits with code 1.

**Bounding** (protocol 3.3, reads no labels). For each unit:
- whole frames are visited in sha256 order of the frame identifier, and a frame is admitted if the running total stays at or below 6,000;
- a first frame larger than 6,000 is subsampled with RandomState(0).

Frame identifiers:
- spinal cord: `<mouse>|<section>`, from the deposited values;
- colorectal: the section name;
- Wu breast: the sample identifier.

This follows the deposited wording. It differs from the round-2 `prepare_data.selection`, which hashes `'selection-v1|' + identifier` and subsamples only when no whole frame fits.

## 5. Check and clean up

```bash
$PY -m pytest $C/test_cohorts_r3.py -q -p no:cacheprovider
$PY - <<'EOF'
import json, sys
import numpy as np
sys.path.insert(0, '/home/zeyufu/Desktop/singlecell-genomics-research/projects/tessera-st/repository/revision-r3/cohorts')
import common_cohorts as cc
from train_core import build_graph          # frozen six-neighbour rule; raises on a cross-frame edge
for row in json.loads((cc.DATA_R3 / 'index.json').read_text())['datasets']:
    d = np.load(cc.DATA_R3 / row['input'], allow_pickle=False)
    info = json.loads((cc.DATA_R3 / row['input']).with_name('input.json').read_text())
    edge, _ = build_graph(d['coords'], d['field'], 'Tessera', 1)
    assert (d['field'][edge[0]] == d['field'][edge[1]]).all() and len(d['x']) <= 6000
    cover = {t: round(1 - v['n_missing'] / info['n_obs'], 3) for t, v in info['targets'].items()}
    print(row['dataset'], info['n_obs'], 'obs', len(info['frames']), 'frames', 'label coverage', cover,
          'um/unit', info.get('um_per_coordinate_unit'), 'nn6/nn1', round(info['median_6nn_distance'] / info['median_nn_distance'], 2))
EOF
```

For Visium, `nn6/nn1` should be close to 1, as on a hexagonal spot lattice. For MERFISH, check that `median_nn_distance` is plausible for cells in the stated unit.

Once the inputs are verified, the raw archives may be deleted ("record, not raw"). Keep:
- the receipts, the inspection receipts and `selection.json`;
- `fetch_r3.py`, which re-fetches exactly the same files.

## Notes for the record

- **Spinal record version.** Zenodo 18039571, the version the protocol names, is not the latest version of its concept record. Version 20533289 (3 Jun 2026) holds re-exported files with different md5s: `MERFISH_spinal_cord_0531.h5ad` (98,970,246 B) and `SNI_merged_0531.h5ad`. The manifest pins 18039571.
- **Spinal licence.** The spinal record states no licence; the other two are CC BY 4.0.
- **Colorectal patient A938797.** The smallest archive for this patient is `SN123_A938797_Rep1_X.zip`. The `_X` suffix is not explained on the record, whose text says only that this replicate's FASTQ files are missing. Its pathology file, `Pathologist_Annotations_SN123_A938797_Rep1_X.csv`, exists.
