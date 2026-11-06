"""Frontiers presentation derivatives from saved Tessera measurements only."""
from pathlib import Path
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.lines import Line2D
import fitz

from inventory_paths import OUT as _OUT, PROJECT
OUT = _OUT / 'frontiers-redraw'  # inventory copy: outputs below OUT
LEGACY = PROJECT / 'manuscripts/scientific-reports'
FOLLOW = PROJECT / 'results/followup-20261010'
LED = LEGACY / 'source/manuscript/ledgers'
DEST = OUT / 'source'
QA = OUT / 'qa'
OBJECTS = ['NSCLC_Lung5_Rep2','NSCLC_Lung6','NSCLC_Lung9_Rep1','NSCLC_Lung12','NSCLC_Lung13',
           'BCC_PatientA','BCC_PatientB','BCC_PatientC','BCC_PatientD','Breast_Visium','Colorectal_TVA232','MIBI','OpenST_HNSCC26']
SHORT = {n:n.replace('NSCLC_','').replace('BCC_Patient','BCC ').replace('Breast_Visium','Breast Visium').replace('Colorectal_TVA232','Colorectal TVA').replace('OpenST_HNSCC26','Open-ST HNSCC') for n in OBJECTS}
COLOR = {'PCA50':'#176b93','NeighborMean':'#657780','Tuned':'#5d3f8f','Tessera':'#bd5b30','SelfOnly':'#527762'}
plt.rcParams.update({'font.family':'Arial','font.size':9,'axes.labelsize':9,'xtick.labelsize':8.5,
                     'ytick.labelsize':8.5,'legend.fontsize':8.5,'pdf.fonttype':42,'axes.spines.top':False,
                     'axes.spines.right':False,'axes.linewidth':.6,'axes.unicode_minus':False})
INPUTS = {}
PLOT_DATA = {}


def track(path):
    path = Path(path)
    INPUTS[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def csv(path): return pd.read_csv(track(path))
def read(path): return json.loads(track(path).read_text())


def panel(fig, rect, letter, title):
    ax = fig.add_axes(rect)
    fig.text(rect[0]-.008, rect[1]+rect[3]+.025, letter, weight='bold', size=12)
    fig.text(rect[0]+.026, rect[1]+rect[3]+.025, title, size=8.5)
    return ax


def direction(ax, left='Control', right='Tessera'):
    ax.set_xlabel(ax.get_xlabel()+'\n< '+left+'   '+right+' >',size=8.5,linespacing=1.4)


def save(fig, name):
    # Exact manually positioned canvas; no automatic tight-layout or crop.
    fig.savefig(DEST/(name+'.pdf'), dpi=300, metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(QA/(name+'.png'),dpi=145)
    plt.close(fig)


summary = csv(LED/'primary_summary.csv').set_index(['object','arm'])
scale_summary = read(FOLLOW/'scale_controls_summary.json')
nested = {r['patient']:r for group in ['NSCLC','BCC'] for r in scale_summary[group]['per_patient']}


def comparison():
    fig=plt.figure(figsize=(6.9,6.75))
    y=np.arange(13)
    axes=[panel(fig,[x,.46,w,.46],letter,title) for x,w,letter,title in
          [(.155,.245,'a','Partition agreement'),(.445,.245,'b','Subsampled fields'),(.745,.235,'c','Whole sections')]]
    for ax in axes:ax.set_ylim(12.6,-.6);ax.set_yticks(y);ax.tick_params(axis='y',length=0)
    axes[0].set_yticklabels([SHORT[n] for n in OBJECTS])
    for ax in axes[1:]:ax.set_yticklabels([])
    for arm,marker in [('PCA50','o'),('NeighborMean','s'),('Tessera','o'),('Tuned','D')]:
        values=[nested[n]['nested_control_ari'] if arm=='Tuned' and n in nested else
                np.nan if arm=='Tuned' else summary.loc[(n,arm),'ari_mean'] for n in OBJECTS]
        axes[0].scatter(values,y,s=16,marker=marker,color=COLOR[arm],label=arm)
    axes[0].set_xlim(0,.56);axes[0].set_xticks([0,.2,.4]);axes[0].set_xlabel('ARI')
    vs_pca=[summary.loc[(n,'Tessera'),'ari_mean']-summary.loc[(n,'PCA50'),'ari_mean'] for n in OBJECTS[:9]]
    vs_tuned=[summary.loc[(n,'Tessera'),'ari_mean']-nested[n]['nested_control_ari'] for n in OBJECTS[:9]]
    whole=read(FOLLOW/'whole_sections_summary.json')
    wp=whole['NSCLC']['tessera_minus_pca50']+whole['BCC']['tessera_minus_pca50']
    wt=whole['NSCLC']['tessera_minus_selected_control']+whole['BCC']['tessera_minus_selected_control']
    for ax,p,t in [(axes[1],vs_pca,vs_tuned),(axes[2],wp,wt)]:
        ax.axvline(0,color='black',lw=.6)
        ax.scatter(p,np.arange(9),s=17,facecolors='white',edgecolors=COLOR['PCA50'],label='vs PCA50')
        ax.scatter(t,np.arange(9),s=17,marker='D',facecolors='white',edgecolors=COLOR['Tuned'],label='vs tuned')
        ax.set_xlim(-.30,.22);ax.set_xticks([-.2,0,.2]);ax.set_xlabel('Tessera - comparator (ARI)');direction(ax)
    rescore=csv(FOLLOW/'rescore.csv')
    r=rescore[rescore.object.isin(OBJECTS[:5])].groupby(['object','arm']).ari_celltype_at_celltype_K.mean()
    d=panel(fig,[.155,.105,.245,.205],'d','Cell-type cluster count')
    for arm,marker in [('PCA50','o'),('NeighborMean','s'),('Tessera','o')]:
        d.scatter([r[n,arm] for n in OBJECTS[:5]],np.arange(5),s=17,marker=marker,color=COLOR[arm])
    d.set_yticks(np.arange(5),[SHORT[n] for n in OBJECTS[:5]]);d.set_ylim(4.6,-.6);d.set_xlim(0,.4);d.set_xlabel('Cell-type ARI')
    scales=csv(FOLLOW/'scale_controls.csv')
    ks=[6,10,15,30,50,100,200];shades=['#26759c','#b55738','#688b45','#9c6295','#448c81']
    for rect,letter,title,names,kind in [([.465,.105,.235,.205],'e','Lung neighborhood scale',OBJECTS[:5],'NeighborMean'),
                                        ([.76,.105,.22,.205],'f','BCC neighborhood scale',OBJECTS[5:9],'BANKSY-style')]:
        ax=panel(fig,rect,letter,title)
        for n,col in zip(names,shades):
            sub=scales[(scales.object==n)&(scales.kind==kind)]
            if kind=='BANKSY-style':sub=sub[np.isclose(sub.lam,.2)]
            values=sub.groupby('k').ari.mean().reindex(ks)
            ax.plot(ks,values,'o-',color=col,lw=.8,ms=2,label=SHORT[n].replace('Lung','L').replace('BCC ','B'))
        ax.set_xscale('log');ax.set_xticks([6,30,200],['6','30','200']);ax.set_ylim(bottom=0);ax.set_xlabel('Neighbors k');ax.set_ylabel('ARI')
        ax.legend(frameon=False,fontsize=8.5,ncol=2,loc='upper left',handlelength=.9,columnspacing=.5,handletextpad=.2)
    handles=[Line2D([0],[0],marker=m,ls='',color=COLOR[a],label=l) for a,m,l in
             [('PCA50','o','PCA50'),('NeighborMean','s','Same-graph mean'),('Tuned','D','Tuned control'),('Tessera','o','Tessera')]]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.53,.008),ncol=4,frameon=False)
    PLOT_DATA['figure3']={'bounded_vs_pca':vs_pca,'bounded_vs_tuned':vs_tuned,'whole_vs_pca':wp,'whole_vs_tuned':wt,
                          'note':'Whole-section contrasts are available only for the nine NSCLC/BCC objects; scale curves use saved three-seed means.'}
    save(fig,'figure-03')


def ablations():
    nulls=csv(FOLLOW/'geometry_nulls.csv').groupby('object').ari.mean()
    arms=['NoGate','NoScale','NoContrast','SelfOnly','PermutedXY','PermutedNoContrast']
    matrix=np.array([[summary.loc[(n,'Tessera'),'ari_mean']-(nulls[n] if a=='PermutedNoContrast' else summary.loc[(n,a),'ari_mean']) for a in arms] for n in OBJECTS])
    delta=np.array([summary.loc[(n,'Tessera'),'ari_mean']-summary.loc[(n,'SelfOnly'),'ari_mean'] for n in OBJECTS])
    fig=plt.figure(figsize=(6.9,6.35));a=panel(fig,[.16,.495,.47,.415],'a','Full Tessera minus component control')
    rgba=plt.get_cmap('RdBu_r')(np.linspace(0,1,256));rgba[:,:3]=.55+.45*rgba[:,:3]
    cmap=LinearSegmentedColormap.from_list('light-diverging',rgba)
    lim=max(.05,np.abs(matrix).max())
    im=a.imshow(matrix,aspect='auto',cmap=cmap,norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim))
    a.set_yticks(np.arange(13),[SHORT[n] for n in OBJECTS]);a.set_xticks(np.arange(6),['No gate','No fusion','No contrast','Self-only','Permuted XY','Permuted\nno contrast'],rotation=70,ha='right')
    a.tick_params(length=0)
    for i in range(13):
        for j in range(6):a.text(j,i,f'{matrix[i,j]:+.2f}' if abs(matrix[i,j])>=.005 else '0.00',ha='center',va='center',size=8.5)
    b=panel(fig,[.755,.495,.225,.415],'b','Full Tessera minus self-only')
    b.barh(np.arange(13),delta,color=[COLOR['Tessera'] if v>0 else COLOR['SelfOnly'] for v in delta],height=.65)
    b.set_ylim(12.5,-.5);b.set_yticks(np.arange(13),[]);b.axvline(0,color='black',lw=.6);b.set_xlabel('ARI difference');direction(b,'Self-only')
    cax=fig.add_axes([.65,.57,.017,.33]);fig.colorbar(im,cax=cax).set_label('Tessera - control (ARI)',size=8.5)
    g=csv(FOLLOW/'gate_baseline.csv').groupby('object').mean(numeric_only=True)
    c=panel(fig,[.16,.10,.34,.235],'c','Seam discrimination')
    c.scatter(g.expression_difference_auroc,g.gate_auroc,s=17,color='black');c.plot([.35,.8],[.35,.8],'--',color='#aaa',lw=.7)
    c.axhline(.5,color='#ccc',lw=.5);c.axvline(.5,color='#ccc',lw=.5);c.set_xlim(.35,.8);c.set_ylim(.35,.8)
    c.set_xlabel('PCA-feature difference AUROC');c.set_ylabel('Learned gate AUROC')
    d=panel(fig,[.65,.10,.33,.235],'d','Gate density: fixed lung anchor')
    n=OBJECTS[0]
    with np.load(track(PROJECT/'data'/n/'input.npz'),allow_pickle=False) as data:ref=data['ref']
    with np.load(track(PROJECT/'results'/n/'Tessera-seed1.npz'),allow_pickle=False) as pred:edge=pred['edge_index'];gates=pred['last_gate']
    valid=(ref[edge[0]]>=0)&(ref[edge[1]]>=0);same=ref[edge[0]]==ref[edge[1]];bins=np.linspace(0,1,31)
    densities={}
    for mask,col,label in [(same&valid,COLOR['PCA50'],'Same region'),(~same&valid,COLOR['Tessera'],'Different region')]:
        density,_=np.histogram(gates[mask],bins=bins,density=True);d.stairs(np.log10(1+density),bins,color=col,lw=1.1,label=label);densities[label]=density.tolist()
    d.set_xlabel('Gate value');d.set_ylabel('log10(1 + edge density)');d.legend(frameon=False,fontsize=8.5,loc='upper right')
    fig.text(.16,.018,'Positive ARI differences favor full Tessera; red heatmap cells are positive.',size=8.5)
    PLOT_DATA['figure4']={'full_minus_control':matrix.tolist(),'full_minus_self_only':delta.tolist(),'anchor':n,'seed':1,'histogram_densities':densities,'bins':bins.tolist()}
    save(fig,'figure-04')


def transfer():
    primary=csv(LED/'all_transfer_scores.csv').groupby(['patient','arm','target'])[['accuracy','balanced_accuracy']].mean()
    means=csv(FOLLOW/'transfer_neighbor_mean.csv').set_index(['patient','arm','target'])
    arms=['PCA50','SelfOnly','NeighborMean6','NeighborMean200','Tessera']
    # Use the exact saved names rather than inventing a spelling.
    mean_names=list(means.index.get_level_values('arm').unique())
    arms[2:4]=sorted(mean_names,key=lambda a:float(''.join(c for c in a if c.isdigit())))
    labels=['PCA50','Self-only','Mean 6','Mean 200','Tessera'];fig=plt.figure(figsize=(6.9,5.05))
    for i,(target,metric) in enumerate([('region','balanced_accuracy'),('cell_type','balanced_accuracy'),('region','accuracy'),('cell_type','accuracy')]):
        rect=[.09 if i%2==0 else .60,.58 if i<2 else .13,.35,.30]
        ax=panel(fig,rect,'abcd'[i],('Regions' if target=='region' else 'Cell types')+(': balanced accuracy' if metric=='balanced_accuracy' else ': accuracy'))
        matrix=np.array([[float((means if a in mean_names else primary).loc[(n,a,target),metric]) for n in OBJECTS[:5]] for a in arms])
        for values in matrix.T:ax.plot(np.arange(5),values,color='#ccd0d4',lw=.75)
        for j,a in enumerate(arms):
            col=COLOR['NeighborMean'] if a in mean_names else COLOR[a]
            ax.scatter(np.full(5,j),matrix[j],s=14,color=col);ax.plot([j-.22,j+.22],[matrix[j].mean()]*2,color='black',lw=1.5)
        ax.set_xticks(np.arange(5),labels,rotation=25,ha='right');ax.set_ylim(0,1);ax.set_yticks([0,.25,.5,.75,1]);ax.set_ylabel('Held-out patient score')
        PLOT_DATA[f'figure9_{i}']=matrix.tolist()
    fig.text(.5,.018,'Lines join one held-out patient; black bars are equal-patient means over five folds.',ha='center',size=8.5)
    save(fig,'figure-09')


def landscape():
    table=csv(LED/'landscape.csv');N=len(table);fig=plt.figure(figsize=(6.9,6.9))
    titles=['Same-graph mean','Test-selected','PCA50','Seam AUROC']
    xs=[.235,.432,.629,.826];width=.155
    fields=['tessera_minus_same_graph','tessera_minus_oracle','tessera_minus_pca50']
    for i,(x,title) in enumerate(zip(xs,titles)):
        ax=panel(fig,[x,.34,width,.57],'abcd'[i],title)
        ax.set_ylim(N-.5,-.5);ax.set_yticks(np.arange(N),table.short_label if i==0 else [])
        ax.tick_params(axis='y',length=0)
        if i<3:
            ax.scatter(table[fields[i]],np.arange(N),s=14,marker=['s','D','o'][i],facecolors='white',edgecolors=[COLOR['NeighborMean'],COLOR['Tuned'],COLOR['PCA50']][i])
            ax.axvline(0,color='black',lw=.5);ax.set_xlim(-.4,.5);ax.set_xticks([-.3,0,.3]);ax.set_xlabel('Tessera - control');direction(ax)
        else:
            ax.scatter(table.gate_auroc,np.arange(N),s=14,color=COLOR['Tessera'],label='Gate')
            ax.scatter(table.raw_difference_auroc,np.arange(N),s=14,marker='s',facecolors='white',edgecolors=COLOR['PCA50'],label='Feature difference')
            ax.axvline(.5,color='#aaa',lw=.7,ls='--');ax.set_xlim(.2,.9);ax.set_xticks([.3,.5,.8]);ax.set_xlabel('AUROC')
    groups=table.group.drop_duplicates().tolist()
    e=panel(fig,[.235,.105,.746,.08],'e','Best uniform scale, selected using each object’s labels')
    for i,group in enumerate(groups):
        vals=table.loc[table.group==group,'best_uniform_k'].to_numpy();e.scatter(vals,np.full(len(vals),i),s=15,color='#657780')
    e.set_yticks(np.arange(len(groups)),groups);e.set_ylim(len(groups)-.5,-.5);e.set_xscale('log');e.set_xticks([6,15,50,200],['6','15','50','200']);e.set_xlabel('Neighbors k')
    fig.text(.56,.012,'Seams: orange circles = learned gate; blue squares = PCA-feature difference.',ha='center',size=8.5)
    PLOT_DATA['figure12']=table.to_dict('records');save(fig,'figure-12')


def architecture():
    path=track(LEGACY/'source/manuscript/figures/figure13-architecture.svg')
    root=ET.fromstring(path.read_bytes())
    # The original viewBox ended at y=937 although its objective baseline is y=967.
    root.set('viewBox','20 83 1356 919');root.set('height','919')
    regular=Path(subprocess.check_output(['fc-match','-f','%{file}','Arial'],text=True))
    bold=Path(subprocess.check_output(['fc-match','-f','%{file}','Arial:style=Bold'],text=True))
    arial={False:fitz.Font(fontfile=str(regular)),True:fitz.Font(fontfile=str(bold))}
    termes={False:fitz.Font(fontfile='/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-regular.otf'),
            True:fitz.Font(fontfile='/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-bold.otf')}
    for element in root.iter():
        if element.tag.endswith('text'):
            value=''.join(element.itertext());is_bold=element.get('font-weight')=='bold';x=float(element.get('x','0'))
            scale=termes[is_bold].text_length(value)/arial[is_bold].text_length(value)
            element.set('font-family','Arial, sans-serif')
            element.set('transform',f'translate({x:g},0) scale({scale:.9f},1) translate({-x:g},0)')
    svg=DEST/'figure-02.svg';ET.ElementTree(root).write(svg,encoding='utf-8',xml_declaration=True)
    subprocess.run(['inkscape',str(svg),'--export-type=pdf','--export-filename='+str(DEST/'figure-02.pdf')],check=True,capture_output=True)
    with fitz.open(DEST/'figure-02.pdf') as doc:doc[0].get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(QA/'figure-02.png')
    PLOT_DATA['figure2']={'original_viewBox':path.read_text().split('viewBox=')[1].split('>')[0][:35],'new_viewBox':root.attrib['viewBox'],'objective_baseline_y':967}


def finite_metadata(value):
    if isinstance(value,dict):return {k:finite_metadata(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [finite_metadata(v) for v in value]
    if isinstance(value,(float,np.floating)) and not np.isfinite(value):return None
    return value


if __name__ == '__main__':
    import runpy
    runpy.run_path(str(Path(__file__).with_name('frontiers_redraw_current_figures.py')), run_name='__main__')
