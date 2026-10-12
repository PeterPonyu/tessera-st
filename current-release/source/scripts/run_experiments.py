#!/usr/bin/env python3
"""Fixed-input retraining and component controls; no best-run selection."""
from __future__ import annotations
import argparse
import importlib.metadata
import platform
import socket
import sys
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, roc_auc_score
from common import ROOT, DATA, RESULTS, OBJECTS, ARMS, SEEDS, sha, dump, read, code_hashes, encode

sys.path.insert(0,str(ROOT/'source/src'))
from tessera_st.config import AblationConfig, TrainConfig, ModelConfig
from tessera_st.model.tessera import TesseraNet
from tessera_st.model.graph import build_knn_edges
from tessera_st.losses import total_loss

def configuration(arm):
    cfg=dict(edge_gating=True,multi_scale=True,boundary_contrastive=True,calibrated_uncertainty=False)
    if arm=='NoGate': cfg['edge_gating']=False
    if arm=='NoScale': cfg['multi_scale']=False
    if arm=='NoContrast': cfg['boundary_contrastive']=False
    if arm=='SelfOnly': cfg.update(edge_gating=False,boundary_contrastive=False)
    return AblationConfig(**cfg)

def graph(xy,fields,arm,seed):
    if arm=='SelfOnly':
        row=np.arange(len(xy),dtype=np.int64)
        return np.stack([row,row]),np.zeros(len(row),np.float32)
    xy=np.asarray(xy,float).copy()
    if arm=='PermutedXY':
        rng=np.random.RandomState(1000+seed)
        for f in np.unique(fields):
            ix=np.flatnonzero(fields==f);xy[ix]=xy[rng.permutation(ix)]
    ei,dist=build_knn_edges(xy,k=6,field_ids=fields)
    if (fields[ei[0]]!=fields[ei[1]]).any() or (ei[0]==ei[1]).any():
        raise ValueError('cross-field or self graph edge')
    return ei,((dist-dist.mean())/(dist.std()+1e-8)).astype(np.float32)

def train_net(x,xy,fields,k,arm,seed,device='cuda'):
    torch.manual_seed(seed)
    torch.cuda.empty_cache() if device=='cuda' else None
    torch.cuda.reset_peak_memory_stats() if device=='cuda' else None
    gen=torch.Generator(device=device).manual_seed(seed)
    tc=TrainConfig(epochs=120,seed=seed,device=device)
    ab=configuration(arm)
    model=TesseraNet(ModelConfig(in_dim=x.shape[1]),k,ab).to(device)
    edge,dist=graph(xy,fields,arm,seed)
    tx=torch.as_tensor(x,dtype=torch.float32,device=device)
    te=torch.as_tensor(edge,dtype=torch.long,device=device)
    td=torch.as_tensor(dist,dtype=torch.float32,device=device)
    opt=torch.optim.Adam(model.parameters(),lr=tc.lr,weight_decay=tc.weight_decay)
    history=[]
    started=time.perf_counter()
    model.train()
    for epoch in range(tc.epochs):
        opt.zero_grad();out=model(tx,te,td)
        loss,parts=total_loss(out,tx,te,tc,ab,generator=gen)
        if not torch.isfinite(loss): raise ValueError('nonfinite training loss')
        loss.backward();opt.step()
        if epoch%12==0 or epoch==119:
            history.append(dict(epoch=epoch,loss=float(loss.detach()),**parts))
    model.eval()
    with torch.no_grad():
        out=model(tx,te,td)
        embedding=out.z.cpu().numpy();boundary=out.boundary.cpu().numpy()
        gate=out.gates[-1].cpu().numpy()
    elapsed=time.perf_counter()-started
    memory=int(torch.cuda.max_memory_allocated()) if device=='cuda' else None
    params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model,embedding,boundary,gate,edge,dist,history,elapsed,memory,params

def neighbor_embedding(x,edge):
    out=np.zeros_like(x);np.add.at(out,edge[1],x[edge[0]])
    n=np.bincount(edge[1],minlength=len(x))
    return out/np.maximum(n[:,None],1)

def scores(y,pred,edge,boundary=None,gate=None,secondary=None):
    ok=y>=0
    src,dst=edge
    valid=(y[src]>=0)&(y[dst]>=0)
    same=(y[src[valid]]==y[dst[valid]])
    expected=[]
    result=dict(ari=float(adjusted_rand_score(y[ok],pred[ok])),
        nmi=float(normalized_mutual_info_score(y[ok],pred[ok])),
        reference_edge_homogeneity=float(same.mean()),
        predicted_edge_homogeneity=float((pred[src]==pred[dst]).mean()))
    if gate is not None and len(np.unique(same))==2:
        result['gate_seam_auroc']=float(roc_auc_score(~same,1-gate[valid]))
    else: result['gate_seam_auroc']=None
    if secondary is not None:
        sy,classes=encode(secondary);sv=sy>=0
        result.update(secondary_ari=float(adjusted_rand_score(sy[sv],pred[sv])),
            secondary_nmi=float(normalized_mutual_info_score(sy[sv],pred[sv])),secondary_n_classes=len(classes))
    return result

def environment():
    return dict(hostname=socket.gethostname(),python=sys.version,platform=platform.platform(),
        packages={n:importlib.metadata.version(n) for n in ['numpy','scipy','scikit-learn','torch']},
        cuda=torch.version.cuda,gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        deterministic_algorithms=False,
        seed_scope='torch manual_seed, negative generator, KMeans; CUDA scatter not guaranteed bitwise deterministic')

def lock_run():
    record=dict(protocol='tessera-tumor-expansion-20261008-v1',seeds=SEEDS,arms=ARMS,
        training=dict(epochs=120,hidden=64,embed=32,n_layers=3,dropout=.1,lr=.001,
            weight_decay=.0001,contrastive_weight=.5,temp=.2,negatives=10),
        code_sha256=code_hashes(),environment=environment())
    # Rendering and transfer scripts may be added later; lock the exact estimator code.
    record['code_sha256']={k:v for k,v in record['code_sha256'].items() if
        '/src/' in k or k in ['PROTOCOL.md','source/scripts/common.py','source/scripts/run_experiments.py']}
    path=RESULTS/'run-lock.json'
    if path.exists():
        previous=read(path)
        if previous['code_sha256']!=record['code_sha256'] or previous['training']!=record['training']:
            raise ValueError('fitting code changed; new run identity required')
    else: dump(path,record)

def main():
    p=argparse.ArgumentParser();p.add_argument('--objects',nargs='+',default=OBJECTS,choices=OBJECTS)
    p.add_argument('--arms',nargs='+',default=ARMS,choices=ARMS);p.add_argument('--seeds',type=int,nargs='+',default=SEEDS)
    p.add_argument('--device',default='cuda');args=p.parse_args()
    torch.set_num_threads(4)
    lock_run()
    for name in args.objects:
        d=np.load(DATA/name/'input.npz',allow_pickle=False);meta=read(DATA/name/'input.json')
        if sha(DATA/name/'input.npz')!=meta['matrix_sha256']: raise ValueError('input binding')
        x,xy,fields,y=d['x'],d['coords'],d['field'],d['ref'];k=meta['n_classes']
        canonical,_=graph(xy,fields,'Tessera',1)
        outdir=RESULTS/name;outdir.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(outdir/'graph.npz',edge_index=canonical)
        for seed in args.seeds:
            for arm in args.arms:
                stem=outdir/f'{arm}-seed{seed}'
                if stem.with_suffix('.json').exists():
                    rec=read(stem.with_suffix('.json'))
                    if rec['input_sha256']!=meta['matrix_sha256'] or sha(stem.with_suffix('.npz'))!=rec['prediction_sha256']:
                        raise ValueError('cached output binding')
                    continue
                start=time.perf_counter();model=None;gate=None;boundary=None;history=[];memory=None;params=0
                if arm in ['PCA50','PCA30','NeighborMean']:
                    emb=x if arm=='PCA50' else x[:,:min(30,x.shape[1])]
                    if arm=='NeighborMean': emb=neighbor_embedding(x,canonical).astype(np.float32)
                    edge=canonical
                    elapsed=time.perf_counter()-start
                else:
                    model,emb,boundary,gate,edge,dist,history,elapsed,memory,params=train_net(x,xy,fields,k,arm,seed,args.device)
                if not np.isfinite(emb).all(): raise ValueError('nonfinite embedding')
                km=KMeans(k,n_init=10,random_state=seed).fit(emb);pred=km.labels_
                pack=dict(labels=pred,embedding=emb,centers=km.cluster_centers_,edge_index=edge)
                if boundary is not None: pack.update(boundary=boundary,last_gate=gate)
                np.savez_compressed(stem.with_suffix('.npz'),**pack)
                sc=scores(y,pred,edge,boundary,gate,d['secondary_name'] if 'secondary_name' in d else None)
                if 'purity' in d:
                    sc['purity_strata']={str(th):dict(n=int((d['purity']>=th).sum()),
                        ari=float(adjusted_rand_score(y[d['purity']>=th],pred[d['purity']>=th])))
                        for th in [0,.5,.75] if (d['purity']>=th).sum()>1}
                rec=dict(object=name,cohort=meta['cohort'],seed=seed,arm=arm,n_cells=len(x),K=k,
                    input_sha256=meta['matrix_sha256'],prediction_sha256=sha(stem.with_suffix('.npz')),
                    fit_seconds=elapsed,total_seconds=time.perf_counter()-start,
                    gpu_peak_allocated_bytes=memory,trainable_parameters=params,history=history,
                    ablation=configuration(arm).as_dict() if model is not None else None,
                    n_edges=edge.shape[1],cross_field_edges=int((fields[edge[0]]!=fields[edge[1]]).sum()),
                    self_edges=int((edge[0]==edge[1]).sum()),scores=sc)
                dump(stem.with_suffix('.json'),rec)
                print(name,arm,seed,'ARI',round(sc['ari'],4),'fit_s',round(elapsed,2),flush=True)
                del model;torch.cuda.empty_cache() if args.device=='cuda' else None
    print('Requested fits complete.',flush=True)

if __name__=='__main__': main()
