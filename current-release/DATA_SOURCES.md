# Input sources and reproduction identities

No new biological measurement was collected. The corrected convenience panel
uses seven previously published assay objects. Exact processed filenames,
SHA-256 source hashes, selected cell indices, annotation keys, frame keys and
package versions are in each `input.json`, the protocol and saved `rows.npz`.
Filename and byte identity are both required: matching a paper citation alone
does not establish an identical processed H5AD object.

- DLPFC: Maynard et al. 2021, registered Visium section **151673**.
  https://github.com/LieberInstitute/spatialLIBD ; processed file
  `raw/dlpfc_maynard_2021_visium/dlpfc_maynard_2021_151673.h5ad`, `Region`,
  one registered `uns:spatial` library.
- MERFISH: Moffitt et al. 2018 mouse hypothalamus; retained
  `baselines/serial3d_ref/merfish_mouse_hypothalamus/merfish_0.h5ad`,
  `domain` labels and `slice_id` frames. The source publication is cited in
  `manuscript/refs.bib`; the processed object must match the stored hash.
- STARmap: Wang et al. 2018 mouse visual cortex;
  `processed/starmap_mouse_vcortex_wang2018/anndata.h5ad`, `layer_label`,
  `section_id`. https://www.starmapresources.org/ .
- MIBI-TOF: Hartmann et al. 2021 scMEP; Squidpy `mibitof` source object,
  `raw/squidpy/mibitof.h5ad`, `Cluster`, `library_id`.
  https://squidpy.readthedocs.io/en/stable/api/squidpy.datasets.mibitof.html .
- Open-ST: Schott et al. 2024 HNSCC;
  `processed/openst_hnscc_sub15k/anndata.h5ad`, `ground_truth`, `section_id`.
  https://github.com/rajewsky-lab/openst .
- CODEX: Goltsev et al. 2018 spleen;
  `processed/codex_spleen_goltsev2018/anndata.h5ad`, `niche`, `section_id`.
  https://squidpy.readthedocs.io/en/stable/api/squidpy.datasets.codex.html .
  Niches are **cluster-derived proxies**, not independent histological truth.
- Zhuang: retained atlas-anatomy brain section object
  `processed/niche_abc_zhuang1_brain_slice/anndata.h5ad`, `domain`,
  `section_id`; Allen Brain Cell Atlas data access:
  https://github.com/AllenInstitute/abc_atlas_access .

The ledger does not provide a separately verified direct download/accession
for every author-processed H5AD. That acquisition gap is explicit; this release
supports complete saved-prediction rescore and traceability, not a claim of
one-command reconstruction of all author-preprocessed assays from the web.

Frame-unestablished historical objects (seqFISH, osmFISH, BRCA, IMC,
Slide-seqV2, st_COAD, st_LIHC, st_OV) are outside the corrected inference.
Their citations remain for history; they are not replacement panel inputs.
Source papers, source-image provenance and dataset conditions remain applicable.
