"""Tests of the round-3 cohort code (run: python -m pytest revision-r3/cohorts/test_cohorts_r3.py -q).

Every h5ad, archive, listing and server below is synthetic and lives in a temporary folder; no real
data file is read and nothing is written under data-r3.
"""
from __future__ import annotations

import gzip
import hashlib
import http.server
import inspect
import io
import json
import sys
import tarfile
import threading
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.io import mmwrite

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_cohorts as cc  # noqa: E402
import fetch_r3  # noqa: E402
import inspect_r3  # noqa: E402
import prep_colorectal  # noqa: E402
import prep_spinal  # noqa: E402
import prep_wu  # noqa: E402
from train_core import build_graph  # noqa: E402  (revision-r1: the frozen six-neighbour graph rule)

SPINAL = dict(mouse='animal', section='slice', coords='obsm:spatial', celltypes=['class_l1', 'class_l2'])


# ---------------------------------------------------------------- synthetic stand-ins
def receipt(path, where=None):
    """A receipt as fetch_r3 writes it; the preparation scripts read back only its SHA-256."""
    cc.dump(Path(where) if where else cc.receipt_path(path), dict(file=Path(path).name, sha256=cc.sha256(path)))


def count_matrix(n, genes, seed):
    rng = np.random.default_rng(seed)
    m = rng.poisson(1.5, (n, genes))
    m[np.arange(n), rng.integers(0, genes, n)] += 1          # no all-zero observation
    return m


def hex_lattice(rows, cols, pitch):
    r, c = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    return np.column_stack([(c * pitch + (r % 2) * pitch / 2).ravel(), (r * pitch * np.sqrt(3) / 2).ravel()])


def spinal_h5ad(folder, sizes=None, label_seed=0, name='toy.h5ad'):
    """Toy MERFISH release: sections of each mouse in their own overlapping coordinate frames."""
    import anndata as ad
    sizes = sizes or {('M1', 's1'): 40, ('M1', 's2'): 30, ('M2', 's1'): 35, ('M2', 's3'): 45, ('M3', 's1'): 50}
    n = sum(sizes.values())
    rng = np.random.default_rng(0)
    obs = pd.DataFrame({'animal': np.repeat([m for m, _ in sizes], list(sizes.values())),
                        'slice': np.repeat([s for _, s in sizes], list(sizes.values())),
                        'rel_depth': rng.uniform(0, 1, n), 'qc_score': rng.integers(10, 15, n)},
                       index=[f'cell{i}' for i in range(n)])
    labels = np.random.default_rng(label_seed)
    obs['class_l1'] = pd.Categorical(labels.choice(['Neuron', 'Astrocyte', 'Oligodendrocyte'], n))
    obs['class_l2'] = pd.Categorical(labels.choice(['Exc-1', 'Exc-2', 'Inh-1', 'Astro', 'Oligo'], n))
    obs['lamina'] = pd.Categorical(np.where(rng.random(n) < 0.2, None, rng.choice(['I', 'II', 'III'], n)))
    genes = [f'G{j}' for j in range(12)] + ['Blank-0', 'Blank-1']
    a = ad.AnnData(sparse.csr_matrix(count_matrix(n, len(genes), 1).astype(np.float32)), obs=obs,
                   var=pd.DataFrame(index=genes))
    a.obsm['spatial'] = rng.uniform(0, 300, (n, 2))
    a.uns['spatial_unit'] = 'um'
    path = Path(folder) / name
    a.write_h5ad(path)
    receipt(path)
    return path


def visium_zip(path, section, coords, counts, in_tissue):
    """Toy Space Ranger archive laid out like the colorectal release."""
    barcodes = [f'BC{i:04d}-1' for i in range(len(coords))]
    m = sparse.csc_matrix(counts.T)                         # features x barcodes, as Space Ranger stores it
    h5 = io.BytesIO()
    with h5py.File(h5, 'w') as f:
        g = f.create_group('matrix')
        g['data'], g['indices'], g['indptr'], g['shape'] = m.data, m.indices, m.indptr, np.array(m.shape)
        g['barcodes'] = np.array(barcodes, 'S')
        g.create_group('features')['name'] = np.array([f'G{j}' for j in range(counts.shape[1])], 'S')
        g['features']['feature_type'] = np.array(['Gene Expression'] * counts.shape[1], 'S')
    positions = pd.DataFrame({'barcode': barcodes, 'in_tissue': in_tissue.astype(int), 'array_row': 0, 'array_col': 0,
                              'pxl_row_in_fullres': coords[:, 1], 'pxl_col_in_fullres': coords[:, 0]})
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr(f'{section}/filtered_feature_bc_matrix.h5', h5.getvalue())
        z.writestr(f'{section}/spatial/tissue_positions_list.csv', positions.to_csv(header=False, index=False))
        z.writestr(f'{section}/spatial/scalefactors_json.json',
                   json.dumps({'tissue_hires_scalef': 0.1, 'spot_diameter_fullres': 90.0}))
        z.writestr(f'{section}/spatial/tissue_hires_image.png', b'placeholder')
    receipt(path)
    return np.array(barcodes)


def tar_add(tar, name, data: bytes):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def wu_archives(raw, patients):
    """Toy versions of the three Wu et al. archives (filtered counts, spatial, metadata)."""
    raw.mkdir(parents=True)
    tars = {name: tarfile.open(raw / name, 'w:gz') for name in cc.COHORTS['wu_breast']['files']}
    classes = np.array(['Invasive cancer + stroma', 'DCIS', 'Stroma', 'Lymphocytes', ''])
    for k, patient in enumerate(patients):
        coords = hex_lattice(11, 11, 140.0)
        n, genes = len(coords), 25
        barcodes = [f'AAAC{i:05d}-1' for i in range(n)]
        mtx = io.BytesIO()
        mmwrite(mtx, sparse.coo_matrix(count_matrix(n, genes, k).T))
        base = f'filtered_count_matrices/{patient}_filtered_count_matrix/'
        tar_add(tars['filtered_count_matrices.tar.gz'], base + 'matrix.mtx.gz', gzip.compress(mtx.getvalue()))
        features = ''.join(f'ENSG{j}\tG{j}\tGene Expression\n' for j in range(genes))
        tar_add(tars['filtered_count_matrices.tar.gz'], base + 'features.tsv.gz', gzip.compress(features.encode()))
        tar_add(tars['filtered_count_matrices.tar.gz'], base + 'barcodes.tsv.gz',
                gzip.compress(''.join(b + '\n' for b in barcodes).encode()))
        positions = pd.DataFrame({'b': barcodes, 't': 1, 'r': 0, 'c': 0, 'y': coords[:, 1], 'x': coords[:, 0]})
        tar_add(tars['spatial.tar.gz'], f'spatial/{patient}_spatial/tissue_positions_list.csv',
                positions.to_csv(header=False, index=False).encode())
        tar_add(tars['spatial.tar.gz'], f'spatial/{patient}_spatial/tissue_hires_image.png', b'placeholder')
        meta = pd.DataFrame({'patientid': patient, 'Classification': classes[np.arange(n) % len(classes)]},
                            index=barcodes).iloc[3:]                    # three spots without metadata
        tar_add(tars['metadata.tar.gz'], f'metadata/{patient}_metadata.csv', meta.to_csv().encode())
        tar_add(tars['metadata.tar.gz'], f'metadata/._{patient}_metadata.csv', b'\x00\x05\x16\x07 AppleDouble')
    for name, tar in tars.items():
        tar.close()
        receipt(raw / name)


# ---------------------------------------------------------------- bounding rule
def hash_order(frames):
    """Independent restatement of the protocol's visiting order (SHA-256 of the frame identifier)."""
    return sorted(set(frames), key=lambda f: hashlib.sha256(f.encode()).hexdigest())


def test_bound_is_label_blind_by_signature():
    assert list(inspect.signature(cc.bound).parameters) == ['frames', 'eligible']


def test_bound_packs_whole_frames_in_hash_order_within_budget():
    sizes = {f'M1|s{i}': n for i, n in enumerate([2500, 1800, 3000, 900, 2200, 700, 1300])}
    frames = np.repeat(list(sizes), list(sizes.values()))
    keep, record = cc.bound(frames, np.ones(len(frames), bool))
    assert sizes[hash_order(frames)[0]] <= 6000                 # no subsampling case
    expected, total = [], 0
    for f in hash_order(frames):
        if total + sizes[f] <= 6000:
            expected.append(f)
            total += sizes[f]
    assert sorted(set(frames[keep])) == sorted(expected) and len(keep) == total <= 6000
    assert all((frames[keep] == f).sum() == sizes[f] for f in expected)          # whole frames only
    assert np.array_equal(keep, cc.bound(frames, np.ones(len(frames), bool))[0])  # deterministic
    assert [v['frame'] for v in record['visits']] == hash_order(frames) and record['n_selected'] == total


def test_bound_subsamples_an_oversized_first_frame_with_randomstate0():
    first, second = hash_order(['M1|a', 'M1|b'])
    frames = np.array([first] * 7000 + [second] * 3000)
    eligible = np.ones(len(frames), bool)
    eligible[:10] = False                                       # ineligible rows are never drawn
    keep, record = cc.bound(frames, eligible)
    candidates = np.flatnonzero(eligible & (frames == first))
    assert np.array_equal(keep, np.sort(np.random.RandomState(0).choice(candidates, 6000, replace=False)))
    assert record['visits'] == [dict(frame=first, n_eligible=6990, n_selected=6000),
                                dict(frame=second, n_eligible=3000, n_selected=0)]


def test_bound_skips_an_oversized_later_frame():
    first, second, third = hash_order(['M1|a', 'M1|b', 'M1|c'])
    frames = np.array([first] * 2000 + [second] * 6500 + [third] * 3000)
    keep, _ = cc.bound(frames, np.ones(len(frames), bool))
    assert set(frames[keep]) == {first, third} and len(keep) == 5000


def test_prepared_selection_is_label_blind(tmp_path):
    sizes = {('M1', f's{i}'): n for i, n in enumerate([2500, 2000, 1800, 1500])}       # 7,800 cells > cap
    outs = []
    for seed in (0, 1):                                           # same cells, different labels
        folder = tmp_path / f'labels{seed}'
        folder.mkdir()
        h5ad = spinal_h5ad(folder, sizes=sizes, label_seed=seed)
        prep_spinal.prepare(h5ad, 'toy', root=folder / 'data-r3', confirm_branch_c=True, **SPINAL)
        outs.append(np.load(folder / 'data-r3' / 'spinal' / 'toy-M1' / 'input.npz', allow_pickle=False))
    a, b = outs
    assert len(a['x']) <= 6000 and len(set(a['field'])) < 4
    assert np.array_equal(a['field'], b['field']) and np.array_equal(a['coords'], b['coords'])
    assert np.array_equal(a['x'], b['x']) and not np.array_equal(a['y_celltype_l1'], b['y_celltype_l1'])


# ---------------------------------------------------------------- frames and graphs
def test_frozen_six_neighbour_graph_has_no_cross_frame_edges():
    xy = np.random.RandomState(0).rand(150, 2) * 50
    coords = np.vstack([xy, xy + 0.01, xy[::-1]])               # three frames on top of one another
    frames = np.repeat(['M1|s1', 'M1|s2', 'M2|s1'], 150)
    edge, _ = build_graph(coords, frames, 'Tessera', 1)
    assert edge.shape[1] == 2 * 6 * len(coords) and (frames[edge[0]] == frames[edge[1]]).all()


# ---------------------------------------------------------------- spinal cord branches
def test_prep_spinal_branch_a_end_to_end(tmp_path):
    root = tmp_path / 'data-r3'
    infos = prep_spinal.prepare(spinal_h5ad(tmp_path), 'toy', lamina='lamina', root=root, **SPINAL)
    assert [i['dataset'] for i in infos] == ['toy-M1', 'toy-M2', 'toy-M3']
    for info in infos:
        d = np.load(root / 'spinal' / info['dataset'] / 'input.npz', allow_pickle=False)
        assert d['x'].dtype == np.float32 and d['x'].shape[1] <= 50 and d['coords'].dtype == np.float64
        assert sorted(k for k in d.files if k.startswith('y_')) == ['y_celltype_l1', 'y_celltype_l2', 'y_lamina']
        assert all(d[k].dtype == np.int64 for k in d.files if k.startswith('y_'))
        assert all(f.startswith(info['unit'] + '|') for f in d['field'])        # frames nested in the mouse
        edge, _ = build_graph(d['coords'], d['field'], 'Tessera', 1)
        assert (d['field'][edge[0]] == d['field'][edge[1]]).all()
        assert info['protocol_branch'] == 'A' and info['um_per_coordinate_unit'] == 1.0
        assert info['coordinate_unit_source'] == "uns['spatial_unit']"
        assert info['control_features_removed'] == 2 and info['count_normalization'] is True
        assert info['targets']['lamina']['n_missing'] == (d['y_lamina'] < 0).sum() > 0
        assert info['targets']['celltype_l1']['provenance'] == 'expression-assigned'
    index = json.loads((root / 'index.json').read_text())
    assert [row['dataset'] for row in index['datasets']] == ['toy-M1', 'toy-M2', 'toy-M3']


def test_prep_spinal_branch_c_needs_confirmation_when_a_lamina_column_exists(tmp_path):
    h5ad = spinal_h5ad(tmp_path)
    with pytest.raises(ValueError, match='may hold laminae'):
        prep_spinal.prepare(h5ad, 'toy', root=tmp_path / 'data-r3', **SPINAL)
    infos = prep_spinal.prepare(h5ad, 'toy', root=tmp_path / 'data-r3', confirm_branch_c=True, **SPINAL)
    assert {i['protocol_branch'] for i in infos} == {'C'} and all('lamina' not in i['targets'] for i in infos)
    assert infos[0]['lamina_like_columns_not_used'] == ['rel_depth', 'lamina']


def test_prep_spinal_takes_every_deposited_cell_type_level(tmp_path):
    columns = dict(SPINAL, celltypes=['class_l1', 'class_l2', 'class_l1'])
    infos = prep_spinal.prepare(spinal_h5ad(tmp_path), 'toy', lamina='lamina', root=tmp_path / 'r', **columns)
    assert sorted(infos[0]['targets']) == ['celltype_l1', 'celltype_l2', 'celltype_l3', 'lamina']
    assert infos[0]['columns']['celltype_l3'] == 'class_l1'


def test_branch_b_refuses_while_the_boundary_table_is_empty(tmp_path, monkeypatch):
    assert prep_spinal.LAMINA_BINS == []
    with pytest.raises(ValueError, match='LAMINA_BINS'):
        prep_spinal.lamina_from_depth(np.array([0.1, 0.5]))
    h5ad = spinal_h5ad(tmp_path)
    with pytest.raises(ValueError, match='LAMINA_BINS'):
        prep_spinal.prepare(h5ad, 'toy', depth='rel_depth', root=tmp_path / 'data-r3', **SPINAL)
    assert not (tmp_path / 'data-r3').exists()
    monkeypatch.setattr(prep_spinal, 'LAMINA_BINS', [('I', 0.0, 0.1), ('II', 0.1, 0.3), ('III-IV', 0.3, 0.6)])
    y, names = prep_spinal.lamina_from_depth(np.array([0.05, 0.1, 0.45, 0.7, np.nan]))
    assert names == ['I', 'II', 'III-IV'] and y.tolist() == [0, 1, 2, -1, -1]
    infos = prep_spinal.prepare(h5ad, 'toy', depth='rel_depth', root=tmp_path / 'data-r3', **SPINAL)
    assert {i['protocol_branch'] for i in infos} == {'B'} and infos[0]['lamina_bins'][0] == ['I', 0.0, 0.1]
    with pytest.raises(ValueError, match='non-overlapping'):
        prep_spinal.check_bins([('I', 0.0, 0.2), ('II', 0.1, 0.3)])


# ---------------------------------------------------------------- colorectal
def test_colorectal_smallest_archive_selection():
    sizes = {'SN1_A100_Rep1.zip': 50, 'SN2_A100_Rep2.zip': 80,
             'SN1_A200_Rep1.zip': 60, 'SN2_A200_Rep2.zip': 55,
             'SN1_A300_Rep1.zip': 58, 'SN2_A300_Rep2.zip': 90,
             'SN1_A400_Rep1_X.zip': 100, 'SN2_A400_Rep2.zip': 140,       # a suffixed archive can be the smallest
             'SN1_A500_Rep1.zip': 150, 'SN2_A500_Rep2.zip': 149,
             'SN1_A600_Rep1.zip': 70, 'SN2_A600_Rep2.zip': 70,           # tie: the name decides
             'Pathology_SpotAnnotations.zip': 1, 'Visium Frozen_SN V1.ndpi': 10, 'DeconvolutionResults_X.zip': 5}
    pilot = ['SN1_A100_Rep1.zip', 'SN2_A200_Rep2.zip', 'SN1_A300_Rep1.zip']
    assert fetch_r3.colorectal_selection(sizes, pilot) == {
        'A400': ('SN1_A400_Rep1_X.zip', 100), 'A500': ('SN2_A500_Rep2.zip', 149), 'A600': ('SN1_A600_Rep1.zip', 70)}
    with pytest.raises(ValueError, match='pilot'):
        fetch_r3.colorectal_selection(sizes, ['SN2_A100_Rep2.zip', 'SN2_A200_Rep2.zip', 'SN1_A300_Rep1.zip'])


def test_prep_colorectal_end_to_end_with_pilot_join(tmp_path):
    root, raw = tmp_path / 'data-r3', tmp_path / 'data-r3' / 'raw' / 'colorectal'
    raw.mkdir(parents=True)
    coords = hex_lattice(12, 12, 150.0)
    in_tissue = np.ones(len(coords), bool)
    in_tissue[:4] = False
    good, bad = 'SN9_A900001_Rep1', 'SN9_A900002_Rep2'
    labels = {}
    for k, section in enumerate([good, bad]):
        barcodes = visium_zip(raw / f'{section}.zip', section, coords, count_matrix(len(coords), 30, k), in_tissue)
        names = np.where(coords[:, 0] < coords[:, 0].mean(), 'Tumor', 'Stroma') if section == good else 'Tumor'
        table = pd.DataFrame({'Barcode': barcodes, 'Pathologist Annotations': names}).iloc[4:]
        table.iloc[:10, 1] = ''                                     # ten unlabelled tissue spots
        labels[section] = table
    annotations = tmp_path / 'Pathology_SpotAnnotations.zip'
    with zipfile.ZipFile(annotations, 'w') as z:
        for section, table in labels.items():
            z.writestr(f'Pathology_SpotAnnotations/Pathologist_Annotations_{section}.csv', table.to_csv(index=False))
    receipt(annotations, annotations.with_name(annotations.name + '.sha256.json'))
    excluded = prep_colorectal.prepare_all([good, bad], root, annotations)
    assert [e['section'] for e in excluded] == [bad]                   # one class only: recorded, not replaced
    info = json.loads((root / 'colorectal' / 'A900001' / 'input.json').read_text())
    d = np.load(root / 'colorectal' / 'A900001' / 'input.npz', allow_pickle=False)
    assert len(d['x']) == len(coords) - 4 and set(d['field']) == {good}
    assert d['classes_all'].tolist() == ['Stroma', 'Tumor'] and (d['y_all'] == -1).sum() == 10
    assert info['um_per_coordinate_unit'] == pytest.approx(100 / 150) and info['unit'] == 'A900001'
    assert info['join']['match_rate'] == 1.0 and info['targets']['all']['provenance'] == 'pathologist'


# ---------------------------------------------------------------- Wu breast
def test_wu_tumor_mapping():
    classes = ['Invasive cancer + stroma + lymphocytes', 'DCIS', 'Cancer trapped in lymphocyte aggregation',
               'Normal glands + lymphocytes', 'Stroma', 'Artefact', 'Uncertain', 'Lymphocytes', 'invasive CANCER']
    tumor = {c for c, v in prep_wu.tumor_mapping(classes).items() if v == 'tumor'}
    assert tumor == {classes[0], 'DCIS', classes[2], 'invasive CANCER'}


def test_prep_wu_end_to_end(tmp_path):
    root = tmp_path / 'data-r3'
    patients = ('CID4465', 'CID44971')
    wu_archives(root / 'raw' / 'wu_breast', patients)
    infos = prep_wu.prepare(root, patients=patients)
    assert [i['dataset'] for i in infos] == list(patients)
    for info in infos:
        d = np.load(root / 'wu_breast' / info['dataset'] / 'input.npz', allow_pickle=False)
        classes = d['classes_all'].tolist()
        assert classes == ['DCIS', 'Invasive cancer + stroma', 'Lymphocytes', 'Stroma']
        tumor = np.array([info['tumor_mapping'][c] == 'tumor' for c in classes])
        labelled = d['y_all'] >= 0
        assert np.array_equal(d['y_tumor'][labelled], tumor[d['y_all'][labelled]].astype(int))
        assert (d['y_tumor'][~labelled] == -1).all() and d['classes_tumor'].tolist() == ['non-tumor', 'tumor']
        assert (~labelled).sum() == 3 + 24          # three spots absent from the metadata, 24 empty values
        assert info['um_per_coordinate_unit'] == pytest.approx(100 / 140) and info['join']['rule'] == 'exact'


# ---------------------------------------------------------------- fetch and inspect
class RangeHandler(http.server.BaseHTTPRequestHandler):
    payload = b''

    def do_GET(self):
        start = int(self.headers['Range'].split('=')[1].split('-')[0]) if self.headers.get('Range') else 0
        self.send_response(206 if start else 200)
        if start:
            self.send_header('Content-Range', f'bytes {start}-{len(self.payload) - 1}/{len(self.payload)}')
        self.send_header('Content-Length', str(len(self.payload) - start))
        self.end_headers()
        self.wfile.write(self.payload[start:])

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    payload = np.random.default_rng(0).bytes(300_000)
    httpd = http.server.ThreadingHTTPServer(('127.0.0.1', 0), type('Handler', (RangeHandler,), dict(payload=payload)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f'http://127.0.0.1:{httpd.server_address[1]}/file.bin', payload
    httpd.shutdown()


def test_download_resumes_verifies_md5_and_writes_receipt(tmp_path, server):
    url, payload = server
    dest = tmp_path / 'file.bin'
    dest.with_name('file.bin.part').write_bytes(payload[:120_000])        # an interrupted earlier attempt
    rec = fetch_r3.download(url, dest, len(payload), hashlib.md5(payload).hexdigest())
    assert dest.read_bytes() == payload and not dest.with_name('file.bin.part').exists()
    assert rec['resumed_from_bytes'] == 120_000 and rec['sha256'] == hashlib.sha256(payload).hexdigest()
    assert {'url', 'bytes', 'md5', 'sha256', 'utc_start', 'utc_end'} <= set(json.loads(cc.receipt_path(dest).read_text()))


def test_download_refuses_a_wrong_md5(tmp_path, server):
    url, payload = server
    with pytest.raises(ValueError, match='md5'):
        fetch_r3.download(url, tmp_path / 'file.bin', len(payload), '0' * 32)
    assert not (tmp_path / 'file.bin').exists() and not cc.receipt_path(tmp_path / 'file.bin').exists()


def test_execute_checks_the_freeze_before_any_request(tmp_path, monkeypatch):
    monkeypatch.setattr(cc, 'freeze_record', lambda: None)
    monkeypatch.setattr(fetch_r3, 'record_metadata', lambda *a, **k: pytest.fail('network used before the freeze check'))
    with pytest.raises(SystemExit, match='not frozen'):
        fetch_r3.main(['--execute', 'spinal', '--root', str(tmp_path)])


def test_plan_flags_listing_that_differs_from_the_manifest():
    files = [dict(key=name, size=size, checksum=f'md5:{md5}', links=dict(self=f'https://example.org/{name}'))
             for name, (size, md5) in cc.COHORTS['wu_breast']['files'].items()]
    jobs, _ = fetch_r3.plan('wu_breast', dict(files=files))
    assert all(job['matches_manifest'] for job in jobs)
    files[0]['checksum'] = 'md5:' + '0' * 32
    jobs, _ = fetch_r3.plan('wu_breast', dict(files=files))
    assert [job['matches_manifest'] for job in jobs] == [False, True, True]


def test_inspect_reports_schema_only(tmp_path):
    import anndata as ad
    h5ad = spinal_h5ad(tmp_path)
    report = inspect_r3.inspect(h5ad)
    columns = report['obs']['columns']
    assert set(columns['class_l1']['categories']) == {'Neuron', 'Astrocyte', 'Oligodendrocyte'}
    assert 'categories' not in columns['rel_depth'] and columns['lamina']['n_missing'] > 0
    assert columns['qc_score']['n_unique'] == 5 and 'categories' not in columns['qc_score']   # numeric: never listed
    assert report['obsm']['spatial']['shape'] == [200, 2] and report['uns']['spatial_unit'] == 'um'
    assert set(report['X']) == {'type', 'shape', 'dtype', 'format'}
    text = json.dumps(report)
    a = ad.read_h5ad(h5ad)
    assert not any(repr(float(v))[:8] in text for v in a.obsm['spatial'][:20].ravel())   # no coordinates
    assert not any(repr(float(v))[:8] in text for v in a.obs['rel_depth'][:20])        # no numeric values
    wu_archives(tmp_path / 'raw', ('CID4465',))
    archive = inspect_r3.inspect(tmp_path / 'raw' / 'metadata.tar.gz', ['*.csv'])
    assert list(archive['tables']) == ['metadata/CID4465_metadata.csv'] and len(archive['members']) == 2
    assert 'DCIS' in archive['tables']['metadata/CID4465_metadata.csv']['columns']['Classification']['categories']
