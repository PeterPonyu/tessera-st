# Rights and provenance

The preserved MIT LICENSE applies to author-owned Tessera software. Its scope
excludes third-party datasets, images, upstream method code and model weights
from a blanket grant. Author-written article text, tables and result aggregates
are identified as CC BY 4.0 in that preserved license text.

This archive has mixed rights. Derived input measurements, source annotations,
calibration and independently supplied images retain their source-specific terms.
Neither preprocessing nor drawing a coordinate map converts third-party rights
to MIT or makes measurements statistically independent.

No 10x breast H&E image/background is redistributed in this new package. Breast
Figure 7 consists of author-generated maps of measured spot locations, the
supplied dominant Xenium-derived reference and retained PCA50/Tessera assignments.
The old H&E figure, its derived raster assets and obsolete compiled documents
containing it are excluded. Its source archive, historical image hash and the
user-authorized replacement are recorded in
`provenance/breast-image-source-and-rights.json`; source annotations, measurements
and calibration retain their original terms. No additional image permission is
asserted.

The distinct CRC tissue images and registration derivatives are retained under
the CC BY 4.0 license recorded for the Dataset record
<https://zenodo.org/records/7760264>, DOI `10.5281/zenodo.7760264`. The saved evidence
is `validation-independent-crc/source-record.json`: `metadata.resource_type.type`
is `dataset`, `metadata.license.id` is `cc-by-4.0`, and the record description
identifies the sample ZIP files as Space Ranger outputs including processed count
matrices and histological images. The retained images come from
`SN84_A120838_Rep1.zip`, `SN124_A595688_Rep2.zip` and
`SN124_A551763_Rep2.zip`; publisher checksums match the preserved acquisition
receipt. This dataset-level license evidence is separate from the associated
article license. Exact archive/image hashes and attribution are recorded in
`provenance/crc-images-source-and-rights.json`.

`provenance/upstream-inputs.json` records source identifiers and acquisition
checksums. `provenance/original-files.json` records historical source/evidence
hashes; `provenance/final-document-overlay.json` identifies authorized presentation
replacements in the final candidate. The original acquired research files,
validation protocols, locks and primary saved numerical evidence remain intact.

The legacy acquisition helper is preserved byte-for-byte in
`source/provenance/acquisition/fetch_gate.py` for receipt verification. Its local
proxy restrictions are historical provenance; portable acquisition uses
`portable/acquire.py` and an optional per-request proxy.
