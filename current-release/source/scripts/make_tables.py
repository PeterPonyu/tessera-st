#!/usr/bin/env python3
"""Bind manuscript numbers and complete summary tables to fitted records."""
import numpy as np
import pandas as pd
from common import ROOT,DATA,RESULTS,OBJECTS,ARMS,SEEDS,read,dump,sha

MAN=ROOT/'source/manuscript';LED=MAN/'ledgers'
SHORT={n:n.replace('NSCLC_','').replace('BCC_Patient','BCC ').replace('Breast_Visium','Breast Visium').replace('Colorectal_TVA232','TVA 232').replace('OpenST_HNSCC26','Open-ST HNSCC') for n in OBJECTS}

def tex(s):return str(s).replace('&',r'\&').replace('_',r'\_').replace('%',r'\%')
def fmt(v):return f'{v:.3f}'

def main():
    g=pd.read_csv(LED/'primary_summary.csv');by={(r.object,r.arm):r for r in g.itertuples(index=False)}
    boot=read(LED/'patient_contrasts.json');m=[read(DATA/n/'input.json') for n in OBJECTS]
    total=sum(x['n_selected'] for x in m);labelled=sum(x['n_labelled'] for x in m);frames=sum(x['n_fields'] for x in m)
    numbers={'TotalObservations':f'{total:,}','TotalLabelled':f'{labelled:,}','TotalFrames':str(frames),
        'PrimaryRuns':str(len(OBJECTS)*len(ARMS)*len(SEEDS)),'TransferRuns':'45'}
    for cohort in ['NSCLC','BCC']:
        b=boot[cohort];prefix=cohort.title() if cohort=='NSCLC' else 'Bcc'
        numbers[prefix+'Delta']=f'{b["delta_ari_mean"]:+.3f}'
        numbers[prefix+'Lower']=f'{b["percentile_ci95"][0]:+.3f}'
        numbers[prefix+'Upper']=f'{b["percentile_ci95"][1]:+.3f}'
        names=OBJECTS[:5] if cohort=='NSCLC' else OBJECTS[5:9]
        for a,suffix in [('PCA50','Pca'),('Tessera','Full'),('NeighborMean','Mean')]:
            numbers[prefix+suffix]=fmt(np.mean([by[n,a].ari_mean for n in names]))
        delta=np.array([by[n,'Tessera'].ari_mean-by[n,'NeighborMean'].ari_mean for n in names])
        rng=np.random.RandomState(20261009);samples=delta[rng.randint(0,len(delta),(10000,len(delta)))].mean(1)
        numbers[prefix+'VsMean']=f'{delta.mean():+.3f}';ci=np.quantile(samples,[.025,.975])
        numbers[prefix+'VsMeanLower']=f'{ci[0]:+.3f}';numbers[prefix+'VsMeanUpper']=f'{ci[1]:+.3f}'
    for n,prefix in [('Breast_Visium','Breast'),('Colorectal_TVA232','Tva'),('OpenST_HNSCC26','Hnscc'),('MIBI','Mibi')]:
        for a,suffix in [('PCA50','Pca'),('Tessera','Full'),('NeighborMean','Mean'),('NoContrast','NoContrast')]:
            numbers[prefix+suffix]=fmt(by[n,a].ari_mean)
        numbers[prefix+'Delta']=f'{by[n,"Tessera"].ari_mean-by[n,"PCA50"].ari_mean:+.3f}'
    full_minus_pca=[by[n,'Tessera'].ari_mean-by[n,'PCA50'].ari_mean for n in OBJECTS]
    numbers['PositiveObjects']=str(sum(d>0 for d in full_minus_pca))
    numbers['NegativeObjects']=str(sum(d<0 for d in full_minus_pca))
    no_con=[by[n,'Tessera'].ari_mean-by[n,'NoContrast'].ari_mean for n in OBJECTS]
    numbers['ContrastHelpsObjects']=str(sum(d>0 for d in no_con))
    numbers['ContrastHurtsObjects']=str(sum(d<0 for d in no_con))
    transfer=pd.read_csv(LED/'all_transfer_scores.csv')
    for target,prefix in [('region','RegionTransfer'),('cell_type','CellTransfer')]:
        for arm,suffix in [('PCA50','Pca'),('SelfOnly','Self'),('Tessera','Full')]:
            v=transfer[(transfer.target==target)&(transfer.arm==arm)]
            numbers[prefix+suffix+'Acc']=fmt(v.accuracy.mean())
            numbers[prefix+suffix+'Balanced']=fmt(v.balanced_accuracy.mean())
    scores=pd.read_csv(LED/'all_primary_scores.csv')
    neural=scores[scores.arm=='Tessera']
    numbers['FullPeakMemory']=f'{neural.gpu_GiB.max():.2f}'
    numbers['FullTimeMin']=f'{neural.fit_seconds.min():.2f}'
    numbers['FullTimeMax']=f'{neural.fit_seconds.max():.2f}'
    macro='\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in numbers.items())+'\n'
    (LED/'numbers.tex').write_text(macro)
    rows=[]
    for n,mi in zip(OBJECTS,m):
        rows.append(' & '.join([tex(SHORT[n]),f'{mi["n_selected"]:,}',f'{mi["n_labelled"]:,}',str(mi['n_fields']),str(mi['n_classes']),str(mi['n_features'])])+r'\\')
    (LED/'input_rows.tex').write_text('\n'.join(rows)+'\n')
    arms=['PCA50','NeighborMean','Tessera','SelfOnly'];rows=[]
    for n in OBJECTS:
        row=[tex(SHORT[n])]+[fmt(by[n,a].ari_mean) for a in arms]+[f'{by[n,"Tessera"].ari_mean-by[n,"PCA50"].ari_mean:+.3f}']
        rows.append(' & '.join(row)+r'\\')
    (LED/'result_rows.tex').write_text('\n'.join(rows)+'\n')
    for metric in ['ari','nmi']:
        for page,arms in enumerate([ARMS[:5],ARMS[5:]]):
            rows=[]
            for n in OBJECTS:
                row=[tex(SHORT[n])]
                for a in arms:
                    r=by[n,a];row.append(f'{getattr(r,metric+"_mean"):.3f} $\\pm$ {getattr(r,metric+"_sd"):.3f}')
                rows.append(' & '.join(row)+r'\\')
            (LED/f'{metric}_rows_{page+1}.tex').write_text('\n'.join(rows)+'\n')
    rows=[]
    for target in ['region','cell_type']:
        for n in OBJECTS[:5]:
            row=[tex(SHORT[n]),'Region' if target=='region' else 'Cell type']
            for arm in ['PCA50','SelfOnly','Tessera']:
                vals=transfer[(transfer.patient==n)&(transfer.arm==arm)&(transfer.target==target)]
                row.append(f'{vals.accuracy.mean():.3f} / {vals.balanced_accuracy.mean():.3f}')
            rows.append(' & '.join(row)+r'\\')
    (LED/'transfer_rows.tex').write_text('\n'.join(rows)+'\n')
    # Full annotation dictionary, including classes absent in the selected fields.
    dictionary={n:dict(classes=mi['classes'],target=mi['target'],frame_counts=mi['field_counts'],
        n_source=mi['n_source'],n_eligible=mi['n_eligible'],n_selected=mi['n_selected'],
        n_labelled=mi['n_labelled'],matrix_sha256=mi['matrix_sha256']) for n,mi in zip(OBJECTS,m)}
    dump(LED/'source_dictionary.json',dictionary)
    dump(LED/'manuscript_numbers.json',numbers)
    print(numbers)

if __name__=='__main__':main()
