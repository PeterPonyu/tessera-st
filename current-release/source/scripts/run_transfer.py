#!/usr/bin/env python3
"""Source-only transformations and frozen-encoder NSCLC patient transfer."""
from __future__ import annotations
import argparse
import time
import sys
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score,f1_score,recall_score,confusion_matrix
from sklearn.metrics.pairwise import euclidean_distances
from common import ROOT,DATA,RESULTS,OBJECTS,SEEDS,dump,sha,read,valid,code_hashes
from run_experiments import train_net,graph,environment

NAMES=OBJECTS[:5]
ARMS=['PCA50','SelfOnly','Tessera']

def transform_source(raw,target):
    mean=raw.mean(0);sd=raw.std(0);sd[sd==0]=1
    z=np.clip((raw-mean)/sd,-10,10).astype(np.float32)
    t=np.clip((target-mean)/sd,-10,10).astype(np.float32)
    p=PCA(n_components=min(50,z.shape[1],len(z)-1),svd_solver='randomized',random_state=0)
    return p.fit_transform(z).astype(np.float32),p.transform(t).astype(np.float32),mean,sd,p

def centroid_predict(source,target,labels):
    labels=np.asarray(labels,str);mask=valid(labels)
    classes=np.asarray(sorted(set(labels[mask])),str)
    cent=np.stack([source[labels==c].mean(0) for c in classes])
    distance=euclidean_distances(target,cent,squared=True)
    pred=classes[np.argmin(distance,1)]
    ordered=np.sort(distance,axis=1)
    margin=(ordered[:,1]-ordered[:,0])/np.maximum(ordered[:,1],1e-8) if len(classes)>1 else np.zeros(len(target))
    return pred,cent,classes,margin

def transfer_score(truth,pred,classes):
    truth=np.asarray(truth,str);mask=valid(truth)
    truth,pred=truth[mask],pred[mask]
    target_classes=sorted(set(truth));supported=[c for c in target_classes if c in classes]
    all_classes=sorted(set(target_classes)|set(pred))
    return dict(accuracy=float(accuracy_score(truth,pred)),
        balanced_accuracy=float(recall_score(truth,pred,labels=target_classes,average='macro',zero_division=0)),
        macro_f1_all_target=float(f1_score(truth,pred,labels=target_classes,average='macro',zero_division=0)),
        macro_f1_supported=float(f1_score(truth,pred,labels=supported,average='macro',zero_division=0)) if supported else None,
        n_target_labelled=len(truth),n_supported_target_classes=len(supported),
        absent_source_target_classes=[c for c in target_classes if c not in classes],
        unsupported_target_cell_fraction=float(np.mean(~np.isin(truth,classes))),
        confusion_labels=all_classes,confusion=confusion_matrix(truth,pred,labels=all_classes).tolist())

def main():
    p=argparse.ArgumentParser();p.add_argument('--targets',nargs='+',default=NAMES,choices=NAMES)
    p.add_argument('--seeds',type=int,nargs='+',default=SEEDS);p.add_argument('--device',default='cuda');args=p.parse_args()
    torch.set_num_threads(4)
    outdir=RESULTS/'transfer';outdir.mkdir(parents=True,exist_ok=True)
    codes=code_hashes();codes={k:v for k,v in codes.items() if '/src/' in k or k in
        ['PROTOCOL.md','source/scripts/common.py','source/scripts/run_experiments.py','source/scripts/run_transfer.py']}
    lock=dict(protocol='tessera-tumor-expansion-20261008-v1-source-transfer',code_sha256=codes,
        arms=ARMS,seeds=SEEDS,environment=environment(),transformation='source-only mean/std/PCA50; target never fit')
    if (outdir/'run-lock.json').exists():
        if read(outdir/'run-lock.json')['code_sha256']!=codes: raise ValueError('transfer code identity changed')
    else: dump(outdir/'run-lock.json',lock)
    inputs={n:np.load(DATA/n/'input.npz',allow_pickle=False) for n in NAMES}
    for target in args.targets:
        target_data=inputs[target]
        sources=[n for n in NAMES if n!=target]
        if len({read(DATA/n/'input.json')['patient'] for n in NAMES})!=5:
            raise ValueError('held-out patient identity')
        genes=sorted(set.intersection(*[set(inputs[n]['gene']) for n in NAMES]))
        # All five inputs must carry the same full source panel: intersection
        # is schema alignment, not target-informed feature selection.
        if not all(set(inputs[n]['gene'])==set(genes) for n in NAMES):
            raise ValueError('source/target gene schemas differ; explicit source-only selector needed')
        raw={n:inputs[n]['gene_log'][:,np.array([np.flatnonzero(inputs[n]['gene']==g)[0] for g in genes])] for n in NAMES}
        src_raw=np.concatenate([raw[n] for n in sources]);tar_raw=raw[target]
        sx,tx,mean,sd,pc=transform_source(src_raw,tar_raw)
        xy=np.concatenate([inputs[n]['coords'] for n in sources]);fields=np.concatenate([inputs[n]['field'] for n in sources])
        if set(fields)&set(target_data['field']): raise ValueError('source/target graph frame overlap')
        labels={key:np.concatenate([inputs[n][key] for n in sources]) for key in ['ref_name','secondary_name']}
        k=len(set(labels['ref_name'][valid(labels['ref_name'])]))
        source_ids=np.concatenate([np.array([n+'|'+s for s in inputs[n]['obs_id']]) for n in sources])
        target_ids=np.array([target+'|'+s for s in target_data['obs_id']])
        if set(source_ids)&set(target_ids): raise ValueError('source-target observation overlap')
        np.savez_compressed(outdir/(target+'-transform.npz'),mean=mean,sd=sd,pca_mean=pc.mean_,
            pca_components=pc.components_,gene=np.asarray(genes,str),source_obs_id=source_ids,target_obs_id=target_ids)
        for seed in args.seeds:
            for arm in ARMS:
                stem=outdir/f'{target}-{arm}-seed{seed}'
                if stem.with_suffix('.json').exists():
                    r=read(stem.with_suffix('.json'))
                    if sha(stem.with_suffix('.npz'))!=r['prediction_sha256']: raise ValueError('cached transfer output')
                    continue
                start=time.perf_counter();model=None;history=[];memory=None
                if arm=='PCA50': se,te=sx,tx
                else:
                    model,se,_,_,edge,dist,history,_,memory,_=train_net(sx,xy,fields,k,arm,seed,args.device)
                    target_edge,target_dist=graph(target_data['coords'],target_data['field'],arm,seed)
                    # The distance standardizer is also source-only, unlike
                    # a per-target zscore in the descriptive object fits.
                    if arm!='SelfOnly':
                        from tessera_st.model.graph import build_knn_edges
                        _,source_distance=build_knn_edges(xy,k=6,field_ids=fields)
                        _,target_distance=build_knn_edges(target_data['coords'],k=6,field_ids=target_data['field'])
                        target_dist=((target_distance-source_distance.mean())/(source_distance.std()+1e-8)).astype(np.float32)
                    with torch.no_grad():
                        t=model(torch.as_tensor(tx,dtype=torch.float32,device=args.device),
                            torch.as_tensor(target_edge,dtype=torch.long,device=args.device),
                            torch.as_tensor(target_dist,dtype=torch.float32,device=args.device))
                        te=t.z.cpu().numpy()
                    torch.save(model.state_dict(),stem.with_suffix('.pt'))
                pack=dict(source_embedding=se,target_embedding=te)
                results={}
                for key,title in [('ref_name','region'),('secondary_name','cell_type')]:
                    pred,cent,classes,margin=centroid_predict(se,te,labels[key])
                    pack.update({title+'_prediction':pred,title+'_centroid':cent,
                        title+'_classes':classes,title+'_margin':margin})
                    results[title]=transfer_score(target_data[key],pred,classes)
                np.savez_compressed(stem.with_suffix('.npz'),**pack)
                dump(stem.with_suffix('.json'),dict(target=target,source_patients=sources,
                    arm=arm,seed=seed,n_source=len(sx),n_target=len(tx),
                    input_sha256={n:sha(DATA/n/'input.npz') for n in NAMES},
                    transform_sha256=sha(outdir/(target+'-transform.npz')),
                    prediction_sha256=sha(stem.with_suffix('.npz')),
                    weight_sha256=sha(stem.with_suffix('.pt')) if model is not None else None,
                    total_seconds=time.perf_counter()-start,gpu_peak_allocated_bytes=memory,
                    history=history,scores=results,patient_disjoint=True,target_transform_fitted=False))
                print('TRANSFER',target,arm,seed,'region acc',round(results['region']['accuracy'],4),
                    'celltype acc',round(results['cell_type']['accuracy'],4),'seconds',round(time.perf_counter()-start,1),flush=True)
                del model;torch.cuda.empty_cache() if args.device=='cuda' else None
    print('Requested transfer fits complete.',flush=True)

if __name__=='__main__': main()
