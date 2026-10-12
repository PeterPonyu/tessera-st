#!/usr/bin/env python3
"""Prepare bounded original-assay inputs; selection never sees outcomes."""
from __future__ import annotations
import argparse
import hashlib
import io
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
from scipy.io import mmread
from sklearn.decomposition import PCA
from common import ROOT, DATA, OBJECTS, dump, sha, encode

LAB = Path('/home/zeyufu/Desktop/labs/active')
CAP = 6000
MARKERS = ['EPCAM','KRT8','KRT18','KRT19','KRT5','KRT14','PTPRC','CD3D',
           'CD3E','CD68','CD79A','MS4A1','COL1A1','COL1A2','DCN','ACTA2',
           'PECAM1','VWF','MKI67','S100A8','S100A9','LGR5','MMP7']

def selection(fields, eligible, strata=None):
    """Whole-field packing in fixed hash order; no reference labels accepted."""
    fields = np.asarray(fields, str)
    candidates = np.flatnonzero(eligible)
    if len(candidates) <= CAP:
        return candidates
    groups = {f: candidates[fields[candidates] == f] for f in np.unique(fields[candidates])}
    rank = lambda f: hashlib.sha256(('selection-v1|' + f).encode()).hexdigest()
    if len(groups) == 1:
        return np.sort(np.random.RandomState(0).choice(candidates, CAP, replace=False))
    if strata is None:
        ordered = sorted(groups, key=rank)
    else:
        # A coordinate frame cannot straddle conditions in the export.
        by = {}
        for f, ix in groups.items():
            s = np.unique(np.asarray(strata, str)[ix])
            if len(s) != 1:
                raise ValueError('condition-straddling frame')
            by.setdefault(str(s[0]), []).append(f)
        pools = [sorted(by[s], key=rank) for s in sorted(by)]
        ordered = [p[i] for i in range(max(map(len, pools))) for p in pools if i < len(p)]
    keep = []
    left = CAP
    for f in ordered:
        if len(groups[f]) <= left:
            keep.extend(groups[f]); left -= len(groups[f])
    if not keep:
        f = ordered[0]
        keep = np.random.RandomState(0).choice(groups[f], CAP, replace=False)
    return np.sort(keep)

def finish(name, X, xy, fields, labels, ids, genes, keep, meta, extras=None):
    X = np.asarray(X.toarray() if hasattr(X, 'toarray') else X, np.float32)
    xy = np.asarray(xy, np.float64)
    if len(X) != len(keep) or len(set(ids)) != len(ids):
        raise ValueError('row identity/shape failure')
    signed = bool(meta.get('signed_source_exception',False))
    if not np.isfinite(X).all() or ((X < 0).any() and not signed) or not np.isfinite(xy).all():
        raise ValueError('invalid source values')
    counts = bool((X>=0).all() and np.allclose(X, np.round(X), atol=1e-5))
    # Numerical QC occurs before feature selection, without labels.
    nonzero = np.abs(X).sum(1) > 0
    X, xy, fields, labels, ids, keep = [np.asarray(a)[nonzero] for a in
                                        [X, xy, fields, labels, ids, keep]]
    extra = {k: np.asarray(v)[nonzero] for k, v in (extras or {}).items()}
    extra = {k: v.astype(str) if v.dtype.kind=='O' else v for k,v in extra.items()}
    genes = np.asarray(genes, str)
    control = np.array([any(t in g.lower() for t in ['negative','negprb','falsecode','systemcontrol'])
                        or g.lower().startswith('blank') for g in genes])
    removed = genes[control].tolist()
    X, genes = X[:, ~control], genes[~control]
    if counts:
        X = np.log1p(X / np.maximum(X.sum(1, keepdims=True), 1) * 1e4)
    marker_names = [g for g in MARKERS if g in genes]
    marker = X[:, [int(np.flatnonzero(genes == g)[0]) for g in marker_names]]
    if len(genes) > 3000:
        ix = np.argsort(-X.var(0), kind='stable')[:3000]
        X, genes = X[:, ix], genes[ix]
    sd = X.std(0); sd[sd == 0] = 1
    z = np.clip((X - X.mean(0))/sd, -10, 10)
    p = PCA(n_components=min(50, z.shape[1], len(z)-1), svd_solver='randomized', random_state=0)
    pc = p.fit_transform(z).astype(np.float32)
    y, classes = encode(labels)
    order = sorted(set(fields))
    field_counts = {str(f): int(np.sum(fields == f)) for f in order}
    # Map selection is also independent of labels/scores: densest retained field.
    display = sorted(order, key=lambda f: (-field_counts[f], str(f)))[0]
    out = DATA / name
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/'input.npz', x=pc, gene_log=X, gene=genes, coords=xy,
        field=np.asarray(fields, str), ref=y, ref_name=np.asarray(labels, str),
        obs_id=np.asarray(ids, str), original_index=keep, marker_log=marker,
        marker_gene=np.asarray(marker_names, str), **extra)
    meta.update(object=name, input_adapter_sha256=sha(Path(__file__)),
        n_selected=len(pc), n_labelled=int((y>=0).sum()),
        n_classes=len(classes), classes=classes, n_features=len(genes), n_pc=pc.shape[1],
        n_fields=len(order), field_counts=field_counts, display_field=str(display),
        count_normalization=counts, control_features_removed=removed,
        pca_explained_variance=float(p.explained_variance_ratio_.sum()),
        zero_count_selected_excluded=int((~nonzero).sum()),
        selection='protocol SHA256 whole-field packing; cap6000; no label/result selection',
        selected_id_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
        matrix_sha256=sha(out/'input.npz'))
    dump(out/'input.json', meta)
    pd.DataFrame({'source_row':keep,'obs_id':ids,'field':fields,'reference':labels}).to_csv(out/'selection.csv.gz',index=False)
    print(name, 'rows', len(pc), 'fields', field_counts, 'K', len(classes), flush=True)

def lung():
    path = LAB/'spgd-tme/data/external/he_nsclc/cosmx_lung.zip'
    digest = sha(path)
    with zipfile.ZipFile(path) as z:
        samples = pd.read_csv(z.open('samples.tsv'),sep='\t')
        for name in ['Lung5_Rep2','Lung6','Lung9_Rep1','Lung12','Lung13']:
            base = name+'/qc/'
            o = pd.read_csv(z.open(base+'observations.tsv'),sep='\t',index_col=0)
            l = pd.read_csv(z.open(name+'/labels.tsv'),sep='\t',index_col=0).reindex(o.index)
            xy = pd.read_csv(z.open(base+'coordinates.tsv'),sep='\t',index_col=0).reindex(o.index).iloc[:,:2].to_numpy()
            f = pd.read_csv(z.open(base+'features.tsv'),sep='\t',index_col=0)
            ids = o.index.astype(str).to_numpy()
            if not all(s.count('_') == 1 and s.split('_')[0].isdigit() and s.split('_')[1].isdigit() for s in ids):
                raise ValueError('unrecognized FOV-cell convention')
            fields = np.array([name+'|FOV'+s.split('_')[0] for s in ids])
            keep = selection(fields, o.selected.to_numpy(bool) & np.isfinite(xy).all(1))
            X = mmread(io.BytesIO(z.read(base+'counts.mtx'))).tocsr()
            if X.shape != (len(o),len(f)):
                raise ValueError('NSCLC matrix orientation/alignment')
            meta = dict(cohort='NSCLC', patient=str(samples.loc[samples['sample']==name,'patient'].iloc[0]),
                source_path=str(path), source_sha256=digest, source_section=name, n_source=len(o),
                n_eligible=int(o.selected.sum()), label_key='label', secondary_key='cell_type',
                target='computational spatial-region annotation: Mclust of 200-neighbor cell-type frequencies',
                frame_key='FOV prefix of archived observation ID',
                units='archived coordinate units; no physical scale asserted',
                source_doi='10.1038/s41587-022-01483-z', processed_doi='10.5281/zenodo.15487520')
            finish('NSCLC_'+name, X[keep], xy[keep], fields[keep],l.label.astype(str).to_numpy()[keep],
                ids[keep], f.index.astype(str), keep, meta,
                {'secondary_name':l.cell_type.astype(str).to_numpy()[keep]})

def bcc():
    base = LAB/'deconv-lab/data/realgt3_benchmark/downloads/bcc_export'
    paths = [base/n for n in ['counts_genes_x_cells.mtx.gz','cells.txt','features.txt','metadata.csv']]
    m = pd.read_csv(paths[3],index_col=0)
    ids = pd.read_csv(paths[1],header=None).iloc[:,0].astype(str).to_numpy()
    genes = pd.read_csv(paths[2],header=None).iloc[:,0].astype(str).to_numpy()
    if not np.array_equal(ids, m.index.astype(str)):
        raise ValueError('BCC source cell order mismatch')
    X = mmread(paths[0]).tocsr()
    if X.shape != (len(genes),len(ids)):
        raise ValueError('BCC genes-by-cells orientation')
    digests = {str(p):sha(p) for p in paths}
    xy = m[['x','y']].to_numpy(float)
    fields = (m.Run_Tissue_name.astype(str)+'|'+m.Patient_ID.astype(str)+'|FOV'+m.fov.astype(str)).to_numpy()
    for patient in ['PatientA','PatientB','PatientC','PatientD']:
        eligible = (m.Patient_ID==patient).to_numpy() & (~m.low_quality.to_numpy(bool)) & np.isfinite(xy).all(1)
        keep = selection(fields,eligible,m.Condition.to_numpy())
        meta = dict(cohort='BCC',patient=patient,n_source=int((m.Patient_ID==patient).sum()),
            n_eligible=int(eligible.sum()),source_files_sha256=digests,
            label_key='celltype',target='source InSituType molecular cell-type annotation',
            frame_key='Run_Tissue_name|Patient_ID|fov',units='source export coordinate units',
            source_doi='10.5281/zenodo.14330691',
            selected_conditions=m.iloc[keep].Condition.value_counts().to_dict())
        finish('BCC_'+patient, X[:,keep].T,xy[keep],fields[keep],m.celltype.astype(str).to_numpy()[keep],
               ids[keep],genes,keep,meta,{'condition':m.Condition.astype(str).to_numpy()[keep]})

def breast():
    base = LAB/'deconv-lab/data/realgt_benchmark'
    path = base/'benchmark_spots_counts.h5ad'
    a = ad.read_h5ad(path)
    t = pd.read_csv(base/'truth_proportions.csv',index_col=0).reindex(a.obs_names)
    if t.isna().any().any() or not np.allclose(t.sum(1),1):
        raise ValueError('breast truth/barcode alignment')
    image_base = base/'downloads/visium_spatial/spatial'
    pos = pd.read_csv(image_base/'tissue_positions.csv',index_col=0).reindex(a.obs_names)
    xy = a.obs[['pxl_col_in_fullres','pxl_row_in_fullres']].to_numpy(float)
    delta = np.max(np.abs(xy-pos[['pxl_col_in_fullres','pxl_row_in_fullres']].to_numpy(float)))
    if delta != 0 or not np.array_equal(xy,np.asarray(a.obsm['spatial'])):
        raise ValueError('image/coordinate registration mismatch')
    import h5py
    from scipy.sparse import csc_matrix, csr_matrix
    with h5py.File(base/'downloads/visium_filtered.h5','r') as h:
        g = h['matrix']; raw = csc_matrix((g['data'][:],g['indices'][:],g['indptr'][:]),shape=g['shape'][:])
        gn = np.asarray(g['features/name'][:],str); bc=np.asarray(g['barcodes'][:],str)
    symbols,inverse=np.unique(gn,return_inverse=True)
    collapse=csr_matrix((np.ones(len(gn)),(inverse,np.arange(len(gn)))),shape=(len(symbols),len(gn)))
    collapsed=(collapse@raw).tocsr()
    gi=pd.Index(symbols).get_indexer(a.var_names); bi=pd.Index(bc).get_indexer(a.obs_names)
    delta_counts=collapsed[gi][:,bi].T.tocsr()-a.X
    delta_counts.eliminate_zeros()
    if (gi<0).any() or (bi<0).any() or delta_counts.nnz:
        raise ValueError('Visium extracted counts differ from measured raw counts')
    scales = __import__('json').loads((image_base/'scalefactors_json.json').read_text())
    manifest = __import__('json').loads((base/'benchmark_manifest.json').read_text())
    img = image_base/'tissue_hires_image.png'
    from PIL import Image
    w,h=Image.open(img).size
    overlay = xy*scales['tissue_hires_scalef']
    if not ((overlay[:,0]>=0)&(overlay[:,0]<w)&(overlay[:,1]>=0)&(overlay[:,1]<h)).all():
        raise ValueError('spots lie outside genuine image')
    meta=dict(cohort='Breast',patient='one tissue block',n_source=a.n_obs,n_eligible=a.n_obs,
        source_path=str(path),source_sha256=sha(path),raw_count_sha256=sha(base/'downloads/visium_filtered.h5'),
        target='dominant co-registered Xenium category; mixed Visium spots',label_key='truth.argmax',
        frame_key='single registered Visium library',units='full-resolution image pixels',
        source_doi='10.1038/s41467-023-43458-x',image_path=str(img),image_sha256=sha(img),
        registration_max_abs_pixel_error=float(delta),raw_count_exact_equality=True,
        scalefactors=scales,um_per_fullres_pixel=manifest['um_per_fullres_px'],
        truth_columns=t.columns.tolist(),truth_sha256=sha(base/'truth_proportions.csv'))
    finish('Breast_Visium',a.X,xy,np.repeat('Breast_Visium',a.n_obs),t.idxmax(1).to_numpy(),
        np.asarray(a.obs_names,str),np.asarray(a.var_names,str),np.arange(a.n_obs),meta,
        {'composition':t.to_numpy(np.float32),'purity':t.max(1).to_numpy(np.float32)})

def h5_object(name,path,key,field_key,cohort,units,source_doi,qc=None,crc=False):
    a=ad.read_h5ad(path,backed='r')
    try:
        xy=(a.obs[['CenterX_local_mm','CenterY_local_mm']].to_numpy(float)*1000
            if crc else np.asarray(a.obsm['spatial'],float)[:,:2])
        fields=np.array([name+'|'+str(f) for f in a.obs[field_key]])
        eligible=np.isfinite(xy).all(1)
        if qc:
            eligible &= a.obs[qc].to_numpy(bool)
        keep=selection(fields,eligible)
        matrix_slot='X'
        X=a.X[keep]
        extra={}
        if crc:
            protein=[c for c in a.obs if c.startswith('Mean.')]
            extra=dict(protein=a.obs.iloc[keep][protein].to_numpy(float),protein_name=np.asarray(protein,str),
                secondary_name=a.obs.lv1.astype(str).to_numpy()[keep],histology_name=a.obs.typ.astype(str).to_numpy()[keep])
            # protein_name is feature metadata, not a per-observation vector.
            extra.pop('protein_name')
        meta=dict(cohort=cohort,patient='one source object; donor scope not multiplied by fields',
            n_source=a.n_obs,n_eligible=int(eligible.sum()),source_path=str(path),source_sha256=sha(path),
            label_key=key,target='source molecular annotation',frame_key=field_key,units=units,
            source_doi=source_doi,source_matrix_slot=matrix_slot)
        if crc:
            meta.update(source_tissue=sorted(set(a.obs.tissue.astype(str))),protein_columns=protein,
                        source_X_name=str(a.uns.get('X_name')),
                        caveat='TVA section, not assumed invasive colorectal carcinoma')
        if name=='MIBI':
            meta.update(signed_source_exception=True,
                protocol_amendment='Signed preprocessed proteomic X retained as noncount values; amendment before MIBI fitting',
                caveat='X and raw.X both contain signed centered measurements; no raw-count claim')
        finish(name,X,xy[keep],fields[keep],a.obs[key].astype(str).to_numpy()[keep],
            np.asarray(a.obs_names,str)[keep],np.asarray(a.var_names,str),keep,meta,extra)
    finally:
        a.file.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--cohorts',nargs='+',default=['lung','bcc','breast','crc','old']);args=p.parse_args()
    if 'lung' in args.cohorts: lung()
    if 'bcc' in args.cohorts: bcc()
    if 'breast' in args.cohorts: breast()
    if 'crc' in args.cohorts:
        h5_object('Colorectal_TVA232',LAB/'spgd-tme/data/external/crc_cosmx/232.h5ad',
            'lv2','fov','Colorectal TVA','micrometers; source local millimeters x1000',
            '10.5281/zenodo.15574384',qc='fil',crc=True)
    if 'old' in args.cohorts:
        base=LAB/'spatial-omics-reform/data'
        h5_object('MIBI',base/'raw/squidpy/mibitof.h5ad','Cluster','library_id','MIBI',
            'stored local coordinate units','10.1038/s41587-020-0651-8')
        h5_object('OpenST_HNSCC26',base/'processed/openst_hnscc_sub15k/anndata.h5ad',
            'ground_truth','section_id','Open-ST HNSCC','stored coordinate units',
            'GSE251926; Schott et al. 2024')
    done=[n for n in OBJECTS if (DATA/n/'input.json').exists()]
    dump(DATA/'index.json',{'protocol':'tessera-tumor-expansion-20261008-v1',
        'objects':done,'inputs':{n:sha(DATA/n/'input.npz') for n in done}})

if __name__=='__main__': main()
