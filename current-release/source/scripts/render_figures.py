#!/usr/bin/env python3
"""Result-bound publication figures; PGF shares the manuscript font contract.

Geometry is explicit. Every statistical figure is a 496.8-pt (text-width)
canvas and each axes, marker, title, colorbar and legend is placed in points
by the functions below (see figure_layout.py). Figures are saved without
tight-bbox cropping, so LaTeX places them at 100% and type keeps its designed
size. Overlapping or off-canvas text is rejected at save time.
"""
from __future__ import annotations
import argparse
import itertools
import string
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('pgf')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize,TwoSlopeNorm,LinearSegmentedColormap,to_rgba
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator,FuncFormatter
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score
from common import ROOT,DATA,RESULTS,OBJECTS,ARMS,SEEDS,read,dump,sha
from figure_layout import Canvas,check_text_layout,TEXT_WIDTH,MARKER_SIZE,CELL_TITLE_SIZE,HEADER

MAN=ROOT/'source/manuscript';FIG=MAN/'figures';QA=ROOT/'qa';LED=MAN/'ledgers'
FOLLOW=RESULTS/'followup-20261010'
INK='#000000';BLUE='#176b93';ORANGE='#bd5b30';GREEN='#527762';PURPLE='#82618e';GRAY='#8c8c8c'
SAND='#a37b3c';SLATE='#657780';LIGHT='#c9cfd4'
COLORS=['#26759c','#b55738','#688b45','#9c6295','#448c81','#c79e39',
        '#727fb2','#a07147','#c16c86','#78a8ba','#929353','#7f6c83',
        '#aa4c53','#5d878c','#b398bc','#9e9e9e','#be8969','#597057',
        '#cfac79','#839ec2','#a66764','#589774','#9b8640','#7e76a2',
        '#b3839c','#65847a','#b7af69','#988064','#a2b7c3','#4e6878',
        '#b18f9e','#72a075','#ba764e','#8d8baf','#a6a8a8']
SHORT={n:n.replace('NSCLC_','').replace('BCC_Patient','BCC ').replace('Breast_Visium','Breast Visium').replace('Colorectal_TVA232','Colorectal TVA').replace('OpenST_HNSCC26','Open-ST HNSCC') for n in OBJECTS}
ARM_SHORT={'PCA50':'PCA50','PCA30':'PCA30','NeighborMean':'Neighbor mean','Tessera':'Tessera',
           'NoGate':'No gate','NoScale':'No fusion','NoContrast':'No contrast','SelfOnly':'Self-only','PermutedXY':'Permuted XY'}
# Method colours are fixed across every figure; cohorts are never coloured with them.
METHOD_COLORS={'PCA50':BLUE,'Tessera':ORANGE,'SelfOnly':GREEN,'NeighborMean':SLATE,'Tuned':'#5d3f8f',
               'NoContrast':PURPLE,'NoGate':'#4f8fae','NoScale':SAND,'PermutedXY':'#8a8f93'}
COHORT_MARKER={'NSCLC':'o','BCC':'s','Other':'^'}
LUNG,BCC=OBJECTS[:5],OBJECTS[5:9]
STYLE_AUDIT = {}
PANEL_LABEL_SIZE = MARKER_SIZE
W = TEXT_WIDTH
TICK = 8; LABEL = 9
X_AXIS = 30          # tick labels plus a one-line axis title below an axes (points)
Y_AXIS = 40          # numeric tick labels plus a rotated axis title (points)
LANE = 22            # left lane for row markers in dense grids
plt.rcParams.update({'font.family':'serif','font.size':9,'font.weight':'normal',
    'axes.titleweight':'normal','axes.labelweight':'normal','axes.labelsize':LABEL,
    'axes.titlesize':9,'xtick.labelsize':TICK,'ytick.labelsize':TICK,'legend.fontsize':8,
    'axes.spines.right':False,'axes.spines.top':False,'axes.unicode_minus':False,
    'axes.linewidth':.6,'xtick.major.width':.6,'ytick.major.width':.6,
    'xtick.major.size':3,'ytick.major.size':3,'xtick.major.pad':2.5,'ytick.major.pad':2.5,
    'axes.labelpad':3,'legend.handlelength':1.4,
    'pgf.rcfonts':False,'pgf.texsystem':'pdflatex','pgf.preamble':(MAN/'paper_fonts.tex').read_text()})


def cohort(name):
    return 'NSCLC' if name in LUNG else 'BCC' if name in BCC else 'Other'


def save(cv,name):
    fig=cv.fig
    FIG.mkdir(parents=True,exist_ok=True);QA.mkdir(parents=True,exist_ok=True)
    labels=[];titles=[];label_sizes=[];regular_sizes=[]
    for artist in fig.findobj(Text):
        if not artist.get_text():continue
        weight=artist.get_fontweight();gid=artist.get_gid()
        assert weight in ('normal',400) or gid=='panel-label', (name,artist.get_text(),weight)
        assert artist.get_fontfamily()==['serif'], (name,artist.get_text(),artist.get_fontfamily())
        assert to_rgba(artist.get_color())==(0.,0.,0.,1.), (name,artist.get_text(),artist.get_color())
        if gid=='panel-label':
            assert weight=='bold' and len(artist.get_text())==1 and artist.get_text().islower()
            labels.append(artist.get_text())
            label_sizes.append(artist.get_fontsize())
        elif gid=='panel-title':titles.append(artist.get_text())
        if gid!='panel-label':regular_sizes.append(artist.get_fontsize())
    assert len(labels)==len(titles) and len(set(labels))==len(labels),name
    assert sorted(labels)==[chr(97+i) for i in range(len(labels))],(name,labels)
    assert min(label_sizes)>max(regular_sizes), (name,label_sizes,max(regular_sizes))
    assert min(regular_sizes)>=7,(name,min(regular_sizes))
    layout_check=check_text_layout(fig,name)
    # No tight bbox: the PDF is exactly the designed canvas, placed at 100%.
    fig.savefig(FIG/(name+'.pdf'),backend='pgf',dpi=330,metadata={'CreationDate':None,'ModDate':None})
    layout=getattr(fig,'_tessera_layout',{});layout['text_layout_check']=layout_check
    order=sorted(range(len(labels)),key=lambda i:labels[i])
    STYLE_AUDIT[name]={'labels':[labels[i] for i in order],'titles':[titles[i] for i in order],
        'separate_text_artists':True,'only_panel_labels_bold':True,
        'font_family':'serif / newtxtext (Times-compatible)','all_text_black':True,
        'panel_label_size_points':min(label_sizes),
        'maximum_regular_text_size_points':max(regular_sizes),
        'minimum_regular_text_size_points':min(regular_sizes),
        'title_alignment':'marker at the block edge, regular title 16 pt to its right on a shared baseline',
        'outer_padding_inches':0,'canvas_width_points':W,'placement':'100% of manuscript text width',
        'method_color_contract':METHOD_COLORS,'layout':layout,
        'pdf_sha256':sha(FIG/(name+'.pdf'))}
    dump(QA/'statistical_figure_style_audit.json',STYLE_AUDIT)
    subprocess.run(['pdftoppm','-singlefile','-r','100','-jpeg',str(FIG/(name+'.pdf')),str(QA/name)],check=True,capture_output=True)
    plt.close(fig)


def light_heatmap(name,white_fraction=.52):
    """Keep numeric text black by lightening the color scale, not the values."""
    rgba=plt.get_cmap(name)(np.linspace(0,1,256))
    rgba[:,:3]=white_fraction+(1-white_fraction)*rgba[:,:3]
    return LinearSegmentedColormap.from_list(name+'-black-text',rgba)


def colorbar(cv,mappable,x,y,w,h,label=None,title=None,ticks=None,size=7.5):
    cax=cv.axes(x,y,w,h);bar=cv.fig.colorbar(mappable,cax=cax)
    bar.ax.tick_params(labelsize=size,length=2.5,width=.5,pad=2);bar.outline.set_linewidth(.5)
    if ticks is not None:bar.set_ticks(ticks)
    if label:bar.set_label(label,fontsize=size,labelpad=3)
    if title:cv.text(x+w/2,y-4,title,size=size,ha='center')
    return bar


def input_data(name):
    return np.load(DATA/name/'input.npz',allow_pickle=False),read(DATA/name/'input.json')

def prediction(name,arm,seed=1):
    return np.load(RESULTS/name/f'{arm}-seed{seed}.npz',allow_pickle=False)

def fit_record(name,arm,seed=1): return read(RESULTS/name/f'{arm}-seed{seed}.json')

def display_reference(y, minimum_fraction=.01):
    """Gray <1% classes by object-level prevalence, without changing score inputs."""
    y=np.asarray(y);ok=y>=0;cs,counts=np.unique(y[ok],return_counts=True)
    common=cs[counts>=minimum_fraction*ok.sum()]
    return np.where(np.isin(y,common),y,-1)

def visible_classes(data,meta):
    shown=display_reference(data['ref'])
    return set(np.asarray(meta['classes'])[np.unique(shown[shown>=0])])

def display_match(y,pred):
    shown=display_reference(y);cs=np.unique(shown[shown>=0]);ps=np.unique(pred)
    if not len(cs):return np.full(len(pred),-1,dtype=int),{}
    tab=np.array([[np.sum((y==c)&(pred==p)) for p in ps] for c in cs])
    r,c=linear_sum_assignment(-tab);mapping={int(ps[j]):int(cs[i]) for i,j in zip(r,c)}
    return np.array([mapping.get(int(p),-1) for p in pred]),mapping

def labels_colors(y,classes,union=None):
    union=classes if union is None else union
    return ['#c8ccd0' if i<0 else COLORS[union.index(classes[int(i)])%len(COLORS)] for i in y]

def limits(xy,pad=.015):
    lo=xy.min(0);hi=xy.max(0);span=hi-lo
    return (lo[0]-pad*span[0],hi[0]+pad*span[0]),(lo[1]-pad*span[1],hi[1]+pad*span[1])

def scatter_map(ax,xy,c,size=2.2):
    ax.scatter(xy[:,0],xy[:,1],s=size,c=c,linewidths=0,rasterized=True)

DISPLAY_NAMES={'Bcell':'B cell','Tcell':'T cell','PlasmaCell':'Plasma cell','Cancer.cells':'Cancer cells',
    'Normal.Kerat':'Normal keratinocyte','MoMacDC':'Mono/Mac/DC','Tcell_CD4':'T cell CD4','Tcell_CD8':'T cell CD8',
    'Imm_other':'Other immune','epi':'Epithelial','imm':'Immune','str':'Stromal'}

def display_name(c):
    return DISPLAY_NAMES.get(c,c.replace('_',' '))

def class_legend(cv,classes,y,max_width=W-8,visible=None):
    handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor=COLORS[i%len(COLORS)],
        markeredgewidth=0,markersize=4.5,label=display_name(c)) for i,c in enumerate(classes) if visible is None or c in visible]
    if visible is not None:
        handles.append(Line2D([0],[0],marker='o',color='none',markerfacecolor='#c8ccd0',
            markeredgewidth=0,markersize=4.5,label='Rare / unknown / unmatched'))
    legend=cv.fitted_legend(handles,W/2,y,max_width,size=7.5,columnspacing=1.1)
    cv.fig._tessera_layout['category_legend']={'columns':legend._ncols,'rows':int(np.ceil(len(handles)/legend._ncols)),
        'top_points':y}
    return legend

def legend_height(cv,legend):
    cv.fig.canvas.draw()
    return legend.get_window_extent(cv.fig.canvas.get_renderer()).height*72/cv.fig.dpi

def ari_badge(ax,text):
    ax.text(.985,.015,text,transform=ax.transAxes,fontsize=7,ha='right',va='bottom',
        bbox=dict(facecolor='white',alpha=.85,edgecolor='none',pad=1.2))

def method_handles(keys,labels=None,marker='o'):
    labels=labels or [ARM_SHORT.get(k,k) for k in keys]
    return [Line2D([0],[0],marker=marker,ls='',color=METHOD_COLORS[k],markeredgewidth=0,markersize=5,label=l)
            for k,l in zip(keys,labels)]

def resize(cv,height):
    """Shrink a provisional canvas to its content, keeping every point position."""
    old=cv.H;fig=cv.fig
    # Read original (pre-aspect) boxes before the canvas changes size; reading
    # afterwards would re-apply equal aspect to stale fractions and shrink maps.
    boxes=[(ax,ax.get_position(original=True)) for ax in fig.axes]
    fig.set_size_inches(cv.W/72,height/72,forward=False)
    for ax,b in boxes:
        top=(1-b.y1)*old;h=b.height*old
        ax.set_position([b.x0,1-(top+h)/height,b.width,h/height])
    for t in fig.texts:
        x,y=t.get_position();t.set_position((x,1-(1-y)*old/height))
    for lg in fig.legends:
        bb=lg.get_bbox_to_anchor();x0,y0=fig.transFigure.inverted().transform((bb.x0,bb.y0))
        lg.set_bbox_to_anchor((x0,1-(1-y0)*old/height),transform=fig.transFigure)
    cv.H=height;fig._tessera_layout['canvas_points']=[cv.W,height]


def compact_map_grid(fig,axes,gap_inches=.085,colorbars=()):
    """Retained utility: pack equal-aspect maps without stretching coordinates."""
    axes=np.asarray(axes)
    for ax in axes.ravel():ax.apply_aspect()
    widths=[max(ax.get_position().width for ax in axes[:,j]) for j in range(axes.shape[1])]
    gap=gap_inches/fig.get_figwidth();start=(1-sum(widths)-gap*(len(widths)-1))/2
    for j,width in enumerate(widths):
        for ax in axes[:,j]:
            box=ax.get_position();ax.set_position([start,box.y0,box.width,box.height])
        start+=width+gap


def compose_seam_and_map(fig,seam,map_ax,colorbar_ax,slot):
    """Retained utility: explicit side-by-side scatter, equal-aspect map and bar."""
    width_points=fig.get_figwidth()*72;height_points=fig.get_figheight()*72
    ratio=abs(np.diff(map_ax.get_xlim())[0]/np.diff(map_ax.get_ylim())[0])
    map_width=slot.height*height_points*ratio/width_points
    bar_width=5/width_points;bar_gap=5/width_points;panel_gap=14/width_points
    bar_left=slot.x1-26/width_points-bar_width
    map_left=bar_left-bar_gap-map_width
    seam.set_position([slot.x0,slot.y0,map_left-panel_gap-slot.x0,slot.height])
    map_ax.set_position([map_left,slot.y0,map_width,slot.height]);map_ax.set_anchor('W')
    colorbar_ax.set_position([bar_left,slot.y0,bar_width,slot.height])


def heat_numbers(ax,matrix,fmt,size):
    for i,j in itertools.product(range(matrix.shape[0]),range(matrix.shape[1])):
        ax.text(j,i,fmt(matrix[i,j]),ha='center',va='center',fontsize=size,color=INK)

def rotated_band(cv,labels,size,angle):
    """Height needed below an axes for rotated category labels."""
    return cv.text_width(labels,size)*np.sin(np.radians(angle))+size*np.cos(np.radians(angle))+8

def identity_scatter(ax,floor=.4):
    lim=max(ax.get_xlim()[1],ax.get_ylim()[1],floor);ax.plot([0,lim],[0,lim],color=GRAY,lw=.8,ls='--',zorder=0)
    ax.set_xlim(0,lim);ax.set_ylim(0,lim*1.06)

def object_rows(ax,names,gap_rows=True):
    """Object rows with a half-row gap between cohorts; returns y positions."""
    y=[];pos=0;previous=None
    for n in names:
        if gap_rows and previous is not None and cohort(n)!=cohort(previous):pos+=.6
        y.append(pos);pos+=1;previous=n
    ax.set_yticks(y,[SHORT[n] for n in names]);ax.set_ylim(pos-.4,-.6);ax.tick_params(axis='y',length=0)
    return np.array(y)


# ---------------------------------------------------------------- summaries
def summarize():
    rows=[]
    for n in OBJECTS:
        for a in ARMS:
            for seed in SEEDS:
                r=fit_record(n,a,seed)
                rows.append(dict(object=n,cohort=r['cohort'],arm=a,seed=seed,
                    ari=r['scores']['ari'],nmi=r['scores']['nmi'],
                    secondary_ari=r['scores'].get('secondary_ari',np.nan),
                    gate_seam_auroc=r['scores']['gate_seam_auroc'],
                    predicted_edge_homogeneity=r['scores']['predicted_edge_homogeneity'],
                    reference_edge_homogeneity=r['scores']['reference_edge_homogeneity'],
                    fit_seconds=r['fit_seconds'],total_seconds=r['total_seconds'],
                    gpu_GiB=r['gpu_peak_allocated_bytes']/2**30 if r['gpu_peak_allocated_bytes'] else 0,
                    parameters=r['trainable_parameters']))
    df=pd.DataFrame(rows);LED.mkdir(parents=True,exist_ok=True);df.to_csv(LED/'all_primary_scores.csv',index=False)
    group=df.groupby(['object','arm']).agg(ari_mean=('ari','mean'),ari_sd=('ari','std'),
        nmi_mean=('nmi','mean'),nmi_sd=('nmi','std'),secondary_ari_mean=('secondary_ari','mean'),
        fit_seconds_mean=('fit_seconds','mean'),gpu_GiB_max=('gpu_GiB','max')).reset_index()
    group.to_csv(LED/'primary_summary.csv',index=False)
    by={(r['object'],r['arm']):r for r in group.to_dict('records')}
    bootstrap={};rng=np.random.RandomState(20261008)
    for cohort_name,names in [('NSCLC',OBJECTS[:5]),('BCC',OBJECTS[5:9])]:
        delta=np.array([by[(n,'Tessera')]['ari_mean']-by[(n,'PCA50')]['ari_mean'] for n in names])
        draws=delta[rng.randint(0,len(delta),(10000,len(delta)))].mean(1)
        bootstrap[cohort_name]=dict(n_patients=len(delta),delta_ari_mean=float(delta.mean()),
            delta_patient_values=delta.tolist(),percentile_ci95=np.quantile(draws,[.025,.975]).tolist(),
            unit='patient, fixed retained fields; small convenience cohort',B=10000)
    dump(LED/'patient_contrasts.json',bootstrap)
    return df,group,by,bootstrap


def followup():
    scale=pd.read_csv(FOLLOW/'scale_controls.csv')
    scale['config']=scale.kind+'|'+scale.k.astype(str)+'|'+scale.lam.astype(str)
    return dict(scale=scale,scale_summary=read(FOLLOW/'scale_controls_summary.json'),
        rescore=pd.read_csv(FOLLOW/'rescore.csv'),gate=pd.read_csv(FOLLOW/'gate_baseline.csv'),
        nulls=pd.read_csv(FOLLOW/'geometry_nulls.csv'),whole=read(FOLLOW/'whole_sections_summary.json'),
        transfer=pd.read_csv(FOLLOW/'transfer_neighbor_mean.csv'))


# ------------------------------------------------------- Supplementary Fig. 1
def inputs_figure():
    meta=[read(DATA/n/'input.json') for n in OBJECTS]
    cv=Canvas(plt,300)
    names=[SHORT[n] for n in OBJECTS]
    left=cv.text_width(names,TICK)+8
    block=244
    yy=np.arange(len(meta));source=np.array([m['n_source'] for m in meta]);sel=np.array([m['n_selected'] for m in meta])
    a=cv.panel(0,0,206,block,'a','Analysis scope',left=left,bottom=X_AXIS)
    a.barh(yy,source,color='#d9dfe3',height=.7,label='Source observations')
    a.barh(yy,sel,color=BLUE,height=.43,label='Modeled observations')
    a.set_xscale('log');a.set_yticks(yy,names);a.invert_yaxis();a.set_xlabel('Observations (log scale)')
    a.tick_params(axis='y',length=0)
    specs=[('b','Frame IDs',GREEN,[m['n_fields'] for m in meta],'Retained frames',[1,3,5],None),
           ('c','Classes',SAND,[m['n_classes'] for m in meta],'Observed classes',None,None),
           ('d','Shared input',PURPLE,[m['pca_explained_variance'] for m in meta],'Retained PCA variance',[0,.5,1],(0,1))]
    x=214
    for (letter,title,color,values,xlabel,ticks,xlim),w in zip(specs,[90,90,96.8]):
        ax=cv.panel(x,0,w,block,letter,title,left=6,bottom=X_AXIS,right=10)
        ax.barh(yy,values,color=color,height=.6);ax.set_yticks(yy,[]);ax.tick_params(axis='y',length=0)
        ax.invert_yaxis();ax.set_xlabel(xlabel)
        if ticks:ax.set_xticks(ticks)
        if xlim:ax.set_xlim(*xlim)
        x+=w
    handles,labels=a.get_legend_handles_labels()
    box=a.get_position();centre=(box.x0+box.x1)/2*W
    cv.legend(handles,centre,block+6,ncol=2,size=7.5,columnspacing=.9)
    cv.text(W/2,296,'Five NSCLC and four BCC patients; other objects do not add patient-level replication',size=7.5,ha='center')
    save(cv,'figure02-cohort')


# ------------------------------------------------------------------ Figure 3
def contrast_rows(ax,names,vs_pca,vs_control):
    y=object_rows(ax,names)
    ax.axvline(0,color=INK,lw=.6)
    for yy,a,b in zip(y,vs_pca,vs_control):
        ax.plot([a,b],[yy,yy],color=LIGHT,lw=1.4,zorder=1,solid_capstyle='round')
    ax.scatter(vs_pca,y,s=22,facecolors='white',edgecolors=METHOD_COLORS['PCA50'],linewidths=1.1,zorder=3)
    ax.scatter(vs_control,y,s=24,marker='D',facecolors='white',edgecolors=METHOD_COLORS['Tuned'],linewidths=1.1,zorder=3)
    return y

def comparison(by,fu):
    cv=Canvas(plt,470)
    summary=fu['scale_summary'];nested={r['patient']:r for c in ['NSCLC','BCC'] for r in summary[c]['per_patient']}
    names=list(OBJECTS);left=cv.text_width([SHORT[n] for n in names],TICK)+8
    # a: absolute agreement per object for the four representative representations.
    a=cv.panel(0,0,262,250,'a','Tuned control leads',left=left,bottom=X_AXIS,right=10)
    y=object_rows(a,names)
    series={'PCA50':[by[n,'PCA50']['ari_mean'] for n in names],
            'NeighborMean':[by[n,'NeighborMean']['ari_mean'] for n in names],
            'Tuned':[nested[n]['nested_control_ari'] if n in nested else np.nan for n in names],
            'Tessera':[by[n,'Tessera']['ari_mean'] for n in names]}
    stacked=np.array(list(series.values()))
    for yy,lo,hi in zip(y,np.nanmin(stacked,0),np.nanmax(stacked,0)):
        a.plot([lo,hi],[yy,yy],color=LIGHT,lw=1.2,zorder=1)
    for key,marker,size in [('PCA50','o',20),('NeighborMean','s',16),('Tuned','D',20),('Tessera','o',24)]:
        a.scatter(series[key],y,s=size,marker=marker,color=METHOD_COLORS[key],zorder=3,linewidths=0)
    a.set_xlabel('Raw ARI (three-seed mean)');a.set_xlim(0,.56)
    # b, c: patient-level contrasts on the bounded fields and on whole sections.
    patients=LUNG+BCC;left_b=cv.text_width([SHORT[n] for n in patients],TICK)+8
    b=cv.panel(270,0,226.8,250,'b','Tessera trails the tuned control',left=left_b,bottom=X_AXIS,right=8)
    contrast_rows(b,patients,[by[n,'Tessera']['ari_mean']-by[n,'PCA50']['ari_mean'] for n in patients],
        [by[n,'Tessera']['ari_mean']-nested[n]['nested_control_ari'] for n in patients])
    b.set_xlabel('Tessera ARI difference');b.set_xlim(-.29,.21)
    whole=fu['whole']
    c=cv.panel(0,262,262,180,'c','Whole sections keep the order',left=left_b,bottom=X_AXIS,right=10)
    wp=whole['NSCLC']['tessera_minus_pca50']+whole['BCC']['tessera_minus_pca50']
    wc=whole['NSCLC']['tessera_minus_selected_control']+whole['BCC']['tessera_minus_selected_control']
    contrast_rows(c,patients,wp,wc);c.set_xlabel('Tessera ARI difference');c.set_xlim(-.29,.21)
    # d: the same lung predictions' embeddings against supplied cell types at the cell-type K.
    r=fu['rescore'];r=r[r.object.isin(LUNG)].groupby(['object','arm']).ari_celltype_at_celltype_K.mean()
    d=cv.panel(270,262,226.8,180,'d','Cell types reverse the gain',left=cv.text_width(['Lung9_Rep1'],TICK)+8,bottom=X_AXIS,right=8)
    yl=object_rows(d,LUNG)
    for yy,n in zip(yl,LUNG):
        vals=[r[n,k] for k in ['PCA50','NeighborMean','Tessera']]
        d.plot([min(vals),max(vals)],[yy,yy],color=LIGHT,lw=1.2,zorder=1)
    for key,marker,size in [('PCA50','o',20),('NeighborMean','s',16),('Tessera','o',24)]:
        d.scatter([r[n,key] for n in LUNG],yl,s=size,marker=marker,color=METHOD_COLORS[key],zorder=3,linewidths=0)
    d.set_xlabel('Raw ARI at cell-type $K$');d.set_xlim(0,.4)
    handles=method_handles(['PCA50','NeighborMean','Tuned','Tessera'],['PCA50','Neighbor mean ($k=6$)','Tuned spatial control','Tessera'])
    handles[1].set_marker('s');handles[2].set_marker('D')
    handles+=[Line2D([0],[0],marker='o',ls='',mfc='white',mec=METHOD_COLORS['PCA50'],markeredgewidth=1.1,markersize=5,label='Tessera minus PCA50'),
              Line2D([0],[0],marker='D',ls='',mfc='white',mec=METHOD_COLORS['Tuned'],markeredgewidth=1.1,markersize=4.5,label='Tessera minus tuned control')]
    cv.fitted_legend(handles,W/2,448,W-10,size=7.5)
    save(cv,'figure03-comparison')


# ------------------------------------------------------------------ Figure 4
def ablations(df,by,fu):
    cv=Canvas(plt,452)
    nulls=fu['nulls'].groupby('object').ari.mean()
    arms=['NoGate','NoScale','NoContrast','SelfOnly','PermutedXY','PermutedNoContrast']
    labels=['No gate','No fusion','No contrast','Self-only','Permuted XY','Permuted, no contrast']
    matrix=np.array([[ (nulls[n] if a=='PermutedNoContrast' else by[n,a]['ari_mean'])-by[n,'Tessera']['ari_mean']
                       for a in arms] for n in OBJECTS])
    lim=max(.05,np.max(np.abs(matrix)))
    names=[SHORT[n] for n in OBJECTS]
    left=cv.text_width(names,TICK)+8;bottom=cv.text_width(labels,TICK)+8
    a=cv.panel(0,0,272,250,'a','Removing the gate does not hurt',left=left,bottom=bottom,right=0)
    im=a.imshow(matrix,cmap=light_heatmap('RdBu_r'),norm=TwoSlopeNorm(vcenter=0,vmin=-lim,vmax=lim),aspect='auto')
    a.set_yticks(range(13),names);a.set_xticks(range(len(arms)),labels,rotation=90,ha='center',va='top')
    a.tick_params(length=0)
    for s in a.spines.values():s.set_visible(False)
    for yline in [4.5,8.5]:a.axhline(yline,color='white',lw=1.6)
    heat_numbers(a,matrix,lambda v:'0.00' if abs(v)<.005 else f'{v:+.2f}',7)
    colorbar(cv,im,276,20,6,250-20-bottom,title=r'$\Delta$ARI')
    # b: does the learned gate know more about reference seams than a trivial expression difference?
    gate=fu['gate'].groupby('object').mean(numeric_only=True)
    b=cv.panel(318,0,178.8,250,'b','Difference beats the gate',left=Y_AXIS,bottom=X_AXIS+10,right=8)
    for n in OBJECTS:
        b.scatter(gate.loc[n,'expression_difference_auroc'],gate.loc[n,'gate_auroc'],s=22,marker=COHORT_MARKER[cohort(n)],
            color=INK if cohort(n)=='NSCLC' else '#6e6e6e' if cohort(n)=='BCC' else '#a9a9a9',linewidths=0)
    b.plot([.35,.8],[.35,.8],color=GRAY,lw=.8,ls='--',zorder=0);b.axhline(.5,color=LIGHT,lw=.6,zorder=0);b.axvline(.5,color=LIGHT,lw=.6,zorder=0)
    b.set_xlim(.35,.8);b.set_ylim(.35,.8);b.set_xticks([.4,.5,.6,.7,.8]);b.set_yticks([.4,.5,.6,.7,.8])
    b.set_xlabel('Seam AUROC of\nPCA-feature difference');b.set_ylabel('Seam AUROC of learned gate')
    b.legend(handles=[Line2D([0],[0],marker=COHORT_MARKER[k],ls='',color=c,markeredgewidth=0,markersize=5,label=k)
        for k,c in [('NSCLC',INK),('BCC','#6e6e6e'),('Other','#a9a9a9')]],frameon=False,fontsize=7,loc='upper left',
        borderaxespad=0,handletextpad=.2)
    # c: last-layer gates in the predetermined lung anchor.
    n=OBJECTS[0];d,m=input_data(n);p=prediction(n,'Tessera');edge=p['edge_index'];valid=(d['ref'][edge[0]]>=0)&(d['ref'][edge[1]]>=0)
    same=d['ref'][edge[0]]==d['ref'][edge[1]]
    c=cv.panel(0,262,248,180,'c','Gate weights by seam',left=Y_AXIS,bottom=X_AXIS,right=10)
    for mask,col,label in [(same&valid,METHOD_COLORS['PCA50'],'Same source region'),(~same&valid,ORANGE,'Different source region')]:
        c.hist(p['last_gate'][mask],bins=np.linspace(0,1,31),density=True,histtype='step',lw=1.1,color=col,label=label)
    c.set_xlabel('Gate value');c.set_ylabel('Edge density');c.legend(frameon=False,fontsize=7,borderaxespad=0,loc='upper right')
    # d: the geometry contrast against the correct reference (self-only), not a collapsed permutation.
    dd=cv.panel(252,262,244.8,180,'d','Contrast carries the gain',left=left,bottom=X_AXIS,right=8)
    y=object_rows(dd,OBJECTS)
    delta=[by[n,'Tessera']['ari_mean']-by[n,'SelfOnly']['ari_mean'] for n in OBJECTS]
    dd.barh(y,delta,color=[ORANGE if v>0 else METHOD_COLORS['SelfOnly'] for v in delta],height=.65)
    dd.axvline(0,color=INK,lw=.6);dd.set_xlabel('Raw ARI difference')
    save(cv,'figure04-ablations')


# ----------------------------------------------- Figures 5, 6 and S5 (maps)
ARMS_MAP=['Reference','PCA50','NeighborMean','Tessera']
MAP_TITLES={'Reference':'Reference','PCA50':'PCA50','NeighborMean':'Neighbor mean','Tessera':'Tessera'}

def map_row(cv,y,data,name,classes,mw,gap,x0,size=2.2):
    """One Reference/PCA50/Neighbor-mean/Tessera row; returns (bottom, map box)."""
    d,m=data
    ix=d['field']==m['display_field'];pts=d['coords'][ix];xlim,ylim=limits(pts)
    for j,arm in enumerate(ARMS_MAP):
        yv=display_reference(d['ref']) if arm=='Reference' else display_match(d['ref'],prediction(name,arm)['labels'])[0]
        x=x0+j*(mw+gap)
        cv.text(x,y+9,MAP_TITLES[arm],size=CELL_TITLE_SIZE)
        ax,box=cv.map_axes(x,y+12,mw,1e6,xlim,ylim)
        scatter_map(ax,pts,labels_colors(yv[ix],m['classes'],classes),size)
        if arm!='Reference':
            mean=np.mean([fit_record(name,arm,s)['scores']['ari'] for s in SEEDS])
            ari_badge(ax,f'ARI {mean:.2f}')
    bottom=box[1]+box[3]
    cv.text(x0,bottom+8,f"{m['display_field'].split('|')[-1]}; {int(ix.sum()):,} of {m['n_selected']:,} cells",size=7)
    return bottom+10,box

def cohort_maps(names,figure_name,height_budget=545):
    metas=[input_data(n)[1] for n in names];classes=sorted(set(c for m in metas for c in m['classes']))
    rows=len(names);gap=5
    aspect=np.mean([np.ptp(input_data(n)[0]['coords'][input_data(n)[0]['field']==m['display_field']][:,1])/
                    np.ptp(input_data(n)[0]['coords'][input_data(n)[0]['field']==m['display_field']][:,0]) for n,m in zip(names,metas)])
    width_limit=(W-LANE-3*gap)/4
    height_limit=(height_budget-44-rows*30)/rows/aspect
    mw=min(width_limit,height_limit)
    x0=LANE+(W-LANE-4*mw-3*gap)/2
    cv=Canvas(plt,1000);y=0
    for i,n in enumerate(names):
        top=y
        bottom,box=map_row(cv,y,input_data(n),n,classes,mw,gap,x0)
        title=SHORT[n] if rows>1 else 'Open-ST HNSCC'
        cv.row_marker(x0-LANE,top+11,string.ascii_lowercase[i],title,(top+12+box[1]+box[3])/2)
        y=bottom+6
    visible=set().union(*(visible_classes(input_data(n)[0],m) for n,m in zip(names,metas)))
    legend=class_legend(cv,classes,y+1,visible=visible)
    resize(cv,y+1+legend_height(cv,legend)+3)
    save(cv,figure_name)


# ------------------------------------------------------------------ Figure 7
def histology():
    d,m=input_data('Breast_Visium')
    scale=m['scalefactors']['tissue_hires_scalef'];xy=d['coords']*scale
    center=np.median(xy,axis=0);half=np.ptp(xy,axis=0)*.22
    roi=(center[0]-half[0],center[0]+half[0],center[1]-half[1],center[1]+half[1])
    selected=(xy[:,0]>=roi[0])&(xy[:,0]<=roi[1])&(xy[:,1]>=roi[2])&(xy[:,1]<=roi[3])
    dump(LED/'histology_roi.json',dict(policy='median observed-coordinate center, +/-22% per-axis extent; no score/label selection',
        hires_pixel_bounds=list(roi),n_spots=int(selected.sum()),image_sha256=m['image_sha256'],
        registration_pixel_error=m['registration_max_abs_pixel_error'],scale=scale))
    labels={'Spot locations':None,'Reference':display_reference(d['ref']),
            'PCA50':display_match(d['ref'],prediction('Breast_Visium','PCA50')['labels'])[0],
            'Tessera':display_match(d['ref'],prediction('Breast_Visium','Tessera')['labels'])[0]}
    cv=Canvas(plt,1000);gap=5;mw=(W-LANE-3*gap)/4;y=0
    for zoom,letter,row_title in [(False,'a','Full tissue'),(True,'b','Fixed ROI')]:
        xlim=(roi[0],roi[1]) if zoom else (xy[:,0].min()-20,xy[:,0].max()+20);ylim=(roi[2],roi[3]) if zoom else (xy[:,1].min()-20,xy[:,1].max()+20)
        top=y
        for j,(title,values) in enumerate(labels.items()):
            x=LANE+j*(mw+gap)
            cv.text(x,y+9,title,size=CELL_TITLE_SIZE)
            ax,box=cv.map_axes(x,y+12,mw,1e6,xlim,ylim)
            ax.set_xlim(*xlim);ax.set_ylim(ylim[1],ylim[0])
            if values is None:
                ax.scatter(xy[:,0],xy[:,1],s=9 if zoom else 2.6,c='#aaaaaa',edgecolors='white',linewidths=.08,rasterized=True)
            else:
                ax.scatter(xy[:,0],xy[:,1],s=9 if zoom else 2.6,c=labels_colors(values,m['classes']),alpha=.78,
                    edgecolors='white',linewidths=.08,rasterized=True)
            if not zoom and j==0:
                ax.add_patch(Rectangle((roi[0],roi[2]),roi[1]-roi[0],roi[3]-roi[2],fill=False,edgecolor=ORANGE,lw=.8))
            if j==0:
                length_um=500 if zoom else 2000;px=length_um/m['um_per_fullres_pixel']*scale
                xx=xlim[0]+.06*(xlim[1]-xlim[0]);yy=ylim[1]-.06*(ylim[1]-ylim[0])
                ax.plot([xx,xx+px],[yy,yy],color='black',lw=1.8,solid_capstyle='butt')
                ax.text(xx,yy-.03*(ylim[1]-ylim[0]),f'{length_um/1000:g} mm',fontsize=7,color='black',va='bottom',
                    bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=.8))
            if title in ('PCA50','Tessera'):
                ari_badge(ax,f"ARI {np.mean([fit_record('Breast_Visium',title,s)['scores']['ari'] for s in SEEDS]):.2f}")
        bottom=box[1]+box[3]
        cv.row_marker(0,top+11,letter,row_title,(top+12+bottom)/2)
        y=bottom+8
    legend=class_legend(cv,m['classes'],y,visible=visible_classes(d,m))
    resize(cv,y+legend_height(cv,legend)+3)
    save(cv,'figure07-breast-histology')


# ------------------------------------------------------------------ Figure 8
def breast_mix():
    d,m=input_data('Breast_Visium');comp=d['composition'];pur=d['purity']
    cols=m['truth_columns'];tumor=[i for i,c in enumerate(cols) if c in ['Invasive_Tumor','Prolif_Invasive_Tumor','DCIS_1','DCIS_2']]
    fraction=comp[:,tumor].sum(1);entropy=-(comp*np.log(comp+1e-12)).sum(1)/np.log(comp.shape[1])
    full=prediction('Breast_Visium','Tessera')
    cv=Canvas(plt,372)
    counts=[int((pur>=v).sum()) for v in [0,.5,.75]]
    a=cv.panel(0,0,248,170,'a','Agreement by spot purity',left=Y_AXIS,bottom=X_AXIS+10,right=10)
    for arm in ['PCA50','NeighborMean','Tessera']:
        vals=[];sds=[]
        for th in [0,.5,.75]:
            sc=[fit_record('Breast_Visium',arm,s)['scores']['purity_strata'][str(th)]['ari'] for s in SEEDS]
            vals.append(np.mean(sc));sds.append(np.std(sc,ddof=1))
        a.errorbar([0,1,2],vals,yerr=sds,fmt='o-',color=METHOD_COLORS[arm],ms=3.5,capsize=2,lw=1,label=ARM_SHORT[arm])
    a.set_xticks([0,1,2],[f'All\n{counts[0]:,}',f'$\\geq$0.50\n{counts[1]:,}',f'$\\geq$0.75\n{counts[2]:,}'])
    a.set_xlim(-.3,2.3);a.set_ylabel('Raw ARI');a.set_xlabel('Dominant-category proportion; spots')
    a.legend(frameon=False,fontsize=7,loc='upper left',borderaxespad=0,handletextpad=.3)
    b=cv.panel(252,0,244.8,170,'b','Mixed-spot composition',left=Y_AXIS,bottom=X_AXIS+10,right=8)
    b.hist(pur,bins=np.linspace(0,1,21),color=GREEN,edgecolor='white',linewidth=.5)
    b.set_xlabel('Dominant-category proportion');b.set_ylabel('Measured spots')
    row=180;h=184
    xlim,ylim=limits(d['coords']);ratio=(ylim[1]-ylim[0])/(xlim[1]-xlim[0])
    mh=h-HEADER-6;mw=mh/ratio;bar_x=W-34;map_x=bar_x-6-mw
    cv.header(map_x,row,'d','Tumor and DCIS fraction')
    md,box=cv.map_axes(map_x,row+HEADER,mw,mh,xlim,ylim)
    scatter_map(md,d['coords'],plt.cm.magma(fraction),2.5)
    colorbar(cv,plt.cm.ScalarMappable(norm=Normalize(0,1),cmap='magma'),bar_x,row+HEADER,5,mh,label='Fraction',ticks=[0,.5,1])
    c=cv.panel(0,row,map_x-16,h,'c','Seam score and spot mixing',left=Y_AXIS,bottom=X_AXIS,right=4)
    c.scatter(entropy,full['boundary'],s=2.5,color=METHOD_COLORS['PCA50'],alpha=.35,linewidths=0,rasterized=True)
    rho=spearmanr(entropy,full['boundary']).statistic
    c.set_xlabel('Source composition entropy');c.set_ylabel('Gate-derived seam score')
    c.text(.02,.97,'Spearman $\\rho = %.2f$'%rho,transform=c.transAxes,va='top',fontsize=7.5)
    cv.fig._tessera_layout['seam_and_source_map']={'c_d_gap_points':16,'map_colorbar_gap_points':6,
        'colorbar_width_points':5,'measured_coordinate_aspect_preserved':True,'automatic_colorbar_shrink':False}
    save(cv,'figure08-breast-mixtures')


# --------------------------------------------- Supplementary Figs 3 and 7 (maps)
def scale_bar(ax,xlim,ylim,length,label):
    xx=xlim[0]+.04*(xlim[1]-xlim[0]);yy=ylim[1]-.05*(ylim[1]-ylim[0])
    ax.plot([xx,xx+length],[yy,yy],color='black',lw=1.5,solid_capstyle='butt')
    ax.text(xx,yy-.025*(ylim[1]-ylim[0]),label,fontsize=7,va='bottom',
        bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=.8))

def map_matrix_with_bars(cv,specs,bar_label,bar_space=38,gap=8,y0=0):
    """Rows of three equal maps; a colorbar row reserves a bar slot beside every map."""
    mw=(W-LANE-3*bar_space-2*gap)/3;y=y0
    for letter,row_title,cells,with_bars in specs:
        top=y;bottom=y
        for j,(title,xy,draw) in enumerate(cells):
            x=LANE+j*(mw+bar_space+gap)
            cv.text(x,y+9,title,size=CELL_TITLE_SIZE)
            xlim,ylim=limits(xy)
            ax,box=cv.map_axes(x,y+12,mw,1e6,xlim,ylim)
            mappable=draw(ax,xlim,ylim)
            if with_bars and mappable is not None:
                colorbar(cv,mappable,box[0]+box[2]+4,box[1],5,box[3],label=bar_label,size=7)
            bottom=max(bottom,box[1]+box[3])
        cv.row_marker(0,top+11,letter,row_title,(top+12+bottom)/2)
        y=bottom+8
    return y

def colorectal():
    n='Colorectal_TVA232';d,m=input_data(n);ix=d['field']==m['display_field'];xy=d['coords'][ix]
    coarse=d['secondary_name'];coarse_classes=sorted(set(coarse[~np.isin(coarse,['nan','unknown'])]));coarse_y=np.array([coarse_classes.index(c) if c in coarse_classes else -1 for c in coarse])
    lv2map={i:pd.Series(coarse[d['ref']==i]).mode().iloc[0] for i in range(m['n_classes'])}
    def assignment(arm):
        if arm=='Reference':y=coarse_y
        else:
            matched=display_match(d['ref'],prediction(n,arm)['labels'])[0]
            y=np.array([coarse_classes.index(lv2map[v]) if v>=0 and lv2map[v] in coarse_classes else -1 for v in matched])
        def draw(ax,xlim,ylim):
            scatter_map(ax,xy,labels_colors(y[ix],coarse_classes),2.5);scale_bar(ax,xlim,ylim,100,'100\\,\\textmu m')
        return draw
    def channel(col):
        values=np.log1p(np.maximum(d['protein'][ix,m['protein_columns'].index(col)],0));norm=Normalize(*np.quantile(values,[.02,.98]))
        def draw(ax,xlim,ylim):
            scatter_map(ax,xy,plt.cm.magma(norm(values)),2.5);scale_bar(ax,xlim,ylim,100,'100\\,\\textmu m')
            return plt.cm.ScalarMappable(norm=norm,cmap='magma')
        return draw
    names={'Mean.PanCK':'PanCK','Mean.CD45':'CD45','Mean.DAPI':'DAPI (nuclear stain)'}
    cv=Canvas(plt,1000)
    y=map_matrix_with_bars(cv,[('a','Assignments',[(arm+' compartments',xy,assignment(arm)) for arm in ['Reference','PCA50','Tessera']],False),
        ('b','Measured channels',[(names[col],xy,channel(col)) for col in names],True)],'log1p intensity')
    legend=class_legend(cv,coarse_classes,y,visible=set(coarse_classes))
    resize(cv,y+legend_height(cv,legend)+3)
    save(cv,'figure09-colorectal-proteins')


# ------------------------------------------------------------------ Figure 9
TRANSFER_ARMS=['PCA50','SelfOnly','NeighborMean-k6','NeighborMean-k200','Tessera']
TRANSFER_LABELS=['PCA50','Self-only','Mean\n$k=6$','Mean\n$k=200$','Tessera']

def transfers(fu):
    rows=[]
    for n in OBJECTS[:5]:
        for a in ['PCA50','SelfOnly','Tessera']:
            for seed in SEEDS:
                r=read(RESULTS/'transfer'/f'{n}-{a}-seed{seed}.json')
                for target in ['region','cell_type']:
                    rows.append(dict(patient=n,arm=a,seed=seed,target=target,**{k:v for k,v in r['scores'][target].items() if np.isscalar(v) or v is None}))
    df=pd.DataFrame(rows);df.to_csv(LED/'all_transfer_scores.csv',index=False)
    primary=df.groupby(['patient','arm','target'])[['accuracy','balanced_accuracy']].mean()
    nm=fu['transfer'].set_index(['patient','arm','target'])
    def values(arm,target,metric):
        source=nm if arm.startswith('NeighborMean') else primary
        return np.array([source.loc[(n,arm,target),metric] for n in LUNG])
    cv=Canvas(plt,378)
    specs=[('a','Regions, balanced accuracy','region','balanced_accuracy'),('b','Cell types, balanced accuracy','cell_type','balanced_accuracy'),
           ('c','Regions, accuracy','region','accuracy'),('d','Cell types, accuracy','cell_type','accuracy')]
    for i,(letter,title,target,metric) in enumerate(specs):
        x=0 if i%2==0 else 252;y=0 if i<2 else 186
        ax=cv.panel(x,y,248 if i%2==0 else 244.8,180,letter,title,left=Y_AXIS,bottom=X_AXIS+6,right=8)
        matrix=np.array([values(a,target,metric) for a in TRANSFER_ARMS])
        xs=np.arange(len(TRANSFER_ARMS))
        for p in range(len(LUNG)):
            ax.plot(xs,matrix[:,p],color=LIGHT,lw=.8,zorder=1)
        for j,arm in enumerate(TRANSFER_ARMS):
            key='NeighborMean' if arm.startswith('NeighborMean') else arm
            ax.scatter(np.full(len(LUNG),j),matrix[j],s=12,color=METHOD_COLORS[key],zorder=3,linewidths=0)
            ax.plot([j-.24,j+.24],[matrix[j].mean()]*2,color=INK,lw=1.4,zorder=4)
        ax.set_xticks(xs,TRANSFER_LABELS);ax.tick_params(axis='x',labelsize=7.5)
        ax.set_ylim(0,1);ax.set_yticks([0,.25,.5,.75,1]);ax.set_ylabel('Held-out patient score');ax.set_xlim(-.5,len(xs)-.5)
    cv.text(W/2,374,'Lines join the same held-out patient; black bars are equal-patient means over five folds',size=7.5,ha='center')
    save(cv,'figure10-patient-transfer')
    return df


# ----------------------------------------------------------------- Figure 10
def stability(df,by):
    stable=[]
    for n in OBJECTS:
        for arm in ['PCA50','Tessera','SelfOnly','PermutedXY']:
            preds=[prediction(n,arm,s)['labels'] for s in SEEDS]
            stable.append(dict(object=n,arm=arm,seed_pair_ari_mean=float(np.mean([adjusted_rand_score(a,b) for a,b in itertools.combinations(preds,2)]))))
    st=pd.DataFrame(stable);st.to_csv(LED/'seed_stability.csv',index=False)
    cv=Canvas(plt,276)
    names=[SHORT[n] for n in OBJECTS];left=cv.text_width(names,TICK)+8
    a=cv.panel(0,0,248,270,'a','Partition stability across seeds',left=left,bottom=X_AXIS,right=10)
    y=object_rows(a,OBJECTS)
    for j,arm in enumerate(['PCA50','SelfOnly','Tessera']):
        v=st[st.arm==arm].set_index('object').loc[OBJECTS,'seed_pair_ari_mean']
        a.scatter(v,y+(j-1)*.2,s=12,color=METHOD_COLORS[arm],label=ARM_SHORT[arm],linewidths=0)
    a.set_xlim(0,1.02);a.set_xlabel('Mean ARI between seed partitions')
    a.legend(frameon=False,fontsize=7,loc='lower left',borderaxespad=0,handletextpad=.2)
    hom=df[df.arm=='Tessera'].groupby('object').reference_edge_homogeneity.mean().reindex(OBJECTS)
    b=cv.panel(252,0,244.8,270,'b','Target smoothness and contrast',left=Y_AXIS+4,bottom=X_AXIS,right=10)
    for n in OBJECTS:
        b.scatter(hom[n],by[n,'Tessera']['ari_mean']-by[n,'PCA50']['ari_mean'],s=22,marker=COHORT_MARKER[cohort(n)],
            color=INK if cohort(n)=='NSCLC' else '#6e6e6e' if cohort(n)=='BCC' else '#a9a9a9',linewidths=0)
    l6=OBJECTS[1];b.annotate('Lung6',(hom[l6],by[l6,'Tessera']['ari_mean']-by[l6,'PCA50']['ari_mean']),
        xytext=(-26,-10),textcoords='offset points',fontsize=7,arrowprops=dict(arrowstyle='-',lw=.5,color=INK))
    b.axhline(0,color=INK,lw=.6);b.set_xlabel('Reference edge homogeneity');b.set_ylabel('Tessera minus PCA50 ARI')
    b.legend(handles=[Line2D([0],[0],marker=COHORT_MARKER[k],ls='',color=c,markeredgewidth=0,markersize=5,label=k)
        for k,c in [('NSCLC',INK),('BCC','#6e6e6e'),('Other','#a9a9a9')]],frameon=False,fontsize=7,loc='upper left',
        borderaxespad=0,handletextpad=.2)
    save(cv,'figure11-stability')


# ------------------------------------------------------ Supplementary Fig. 4
def efficiency(df,by):
    cv=Canvas(plt,300)
    n_obs=np.array([read(DATA/o/'input.json')['n_selected'] for o in OBJECTS])
    width=(W-LANE-14)/2;axes=[]
    for r,(letter,row_title,titles) in enumerate([('a','Cost + memory',['Recorded run time','Peak GPU allocation']),
            ('b','Training',['Reconstruction loss','Gate-weighted contrastive loss'])]):
        y=r*148;row=[]
        for j,title in enumerate(titles):
            x=LANE+j*(width+14);cv.text(x,y+10,title,size=CELL_TITLE_SIZE)
            row.append(cv.axes(x+Y_AXIS+4,y+16,width-Y_AXIS-10,148-16-X_AXIS))
        cv.row_marker(0,y+11,letter,row_title,y+16+(148-16-X_AXIS)/2);axes.append(row)
    for arm in ['PCA50','SelfOnly','Tessera']:
        g=df[df.arm==arm].groupby('object').agg(t=('total_seconds','mean'),mem=('gpu_GiB','max')).reindex(OBJECTS)
        axes[0][0].scatter(n_obs,g.t,s=12,label=ARM_SHORT[arm],color=METHOD_COLORS[arm],linewidths=0)
        if arm!='PCA50':axes[0][1].scatter(n_obs,g.mem,s=12,label=ARM_SHORT[arm],color=METHOD_COLORS[arm],linewidths=0)
    ax=axes[0][0];ax.set_yscale('log');ax.set_xlabel('Modeled observations');ax.set_ylabel('Seconds')
    lo,hi=ax.get_ylim();ax.set_ylim(lo,hi*4)
    ax.legend(frameon=False,fontsize=7,loc='upper left',ncol=3,columnspacing=.8,handletextpad=.15,borderaxespad=0)
    ax=axes[0][1];ax.set_xlabel('Modeled observations');ax.set_ylabel('GiB')
    lo,hi=ax.get_ylim();ax.set_ylim(lo,hi*1.25)
    ax.legend(frameon=False,fontsize=7,loc='upper left',ncol=2,borderaxespad=0,handletextpad=.15)
    for arm in ['Tessera','NoContrast','SelfOnly']:
        h=fit_record(OBJECTS[0],arm,1)['history'];axes[1][0].plot([r['epoch'] for r in h],[r['recon'] for r in h],color=METHOD_COLORS[arm],label=ARM_SHORT[arm],lw=1.1)
    axes[1][0].set_xlabel('Training epoch');axes[1][0].set_ylabel('MSE');axes[1][0].legend(frameon=False,fontsize=7,borderaxespad=0)
    for arm in ['Tessera','NoGate','NoScale']:
        h=fit_record(OBJECTS[0],arm,1)['history'];axes[1][1].plot([r['epoch'] for r in h],[r['contrastive'] for r in h],color=METHOD_COLORS[arm],label=ARM_SHORT[arm],lw=1.1)
    axes[1][1].set_xlabel('Training epoch');axes[1][1].set_ylabel('Loss');axes[1][1].legend(frameon=False,fontsize=7,borderaxespad=0)
    save(cv,'figure12-efficiency-mechanism')


# ------------------------------------------------------------- Supplements
def mibi():
    n='MIBI';d,m=input_data(n);fields=sorted(set(d['field']))
    cv=Canvas(plt,1000);gap=5;mw=112;x0=LANE+(W-LANE-4*mw-3*gap)/2;y=0
    preds={a:(display_reference(d['ref']) if a=='Reference' else display_match(d['ref'],prediction(n,a)['labels'])[0]) for a in ARMS_MAP}
    for i,f in enumerate(fields):
        ix=d['field']==f;pts=d['coords'][ix];xlim,ylim=limits(pts);top=y
        for j,a in enumerate(ARMS_MAP):
            x=x0+j*(mw+gap);cv.text(x,y+9,MAP_TITLES[a],size=CELL_TITLE_SIZE)
            ax,box=cv.map_axes(x,y+12,mw,mw,xlim,ylim)
            scatter_map(ax,pts,labels_colors(preds[a][ix],m['classes']),3)
        bottom=box[1]+box[3]
        cv.row_marker(x0-LANE,top+11,string.ascii_lowercase[i],f.split('|')[-1].replace('point','P'),(top+12+bottom)/2)
        y=bottom+8
    legend=class_legend(cv,m['classes'],y,visible=visible_classes(d,m));resize(cv,y+legend_height(cv,legend)+3);save(cv,'supplement02-mibi')

def rna_markers():
    def marker(n,gene):
        d,m=input_data(n);ix=d['field']==m['display_field'];values=d['marker_log'][ix,list(d['marker_gene']).index(gene)]
        norm=Normalize(*np.quantile(values,[0,.98]));xy=d['coords'][ix]
        def draw(ax,xlim,ylim):
            # Low values first, so high-expressing cells are not hidden by zeros.
            order=np.argsort(values);ax.scatter(xy[order,0],xy[order,1],s=2.5,c=plt.cm.viridis(norm(values[order])),
                linewidths=0,rasterized=True)
            return plt.cm.ScalarMappable(norm=norm,cmap='viridis')
        return gene,xy,draw
    cv=Canvas(plt,1000)
    y=map_matrix_with_bars(cv,[('a','Lung',[marker(OBJECTS[0],g) for g in ['EPCAM','COL1A1','PTPRC']],True),
        ('b','BCC',[marker('BCC_PatientA',g) for g in ['KRT14','COL1A1','PTPRC']],True)],'log expression',bar_space=36)
    resize(cv,y+2);save(cv,'supplement03-rna-markers')

def scale_curves(by,fu):
    s=fu['scale'];means=s.groupby(['object','config']).ari.mean()
    cv=Canvas(plt,214);ks=[6,10,15,30,50,100,200]
    specs=[('a','Lung regions: neighbour mean',LUNG,'NeighborMean','nan'),('b','BCC cell types: BANKSY-style, $\\lambda=0.2$',BCC,'BANKSY-style','0.2')]
    shades=[COLORS[i] for i in (0,1,2,3,6)]  # patient identity colours, independent of method colours
    for i,(letter,title,names,kind,lam) in enumerate(specs):
        ax=cv.panel(0 if i==0 else 252,0,248 if i==0 else 244.8,186,letter,title,left=Y_AXIS,bottom=X_AXIS,right=8)
        for n,col in zip(names,shades):
            vals=[means[n,f'{kind}|{float(k)}|{lam}'] for k in ks]
            ax.plot(ks,vals,'o-',ms=2.6,lw=.9,color=col,label=SHORT[n].replace('BCC ','Patient '))
            ax.plot([260],[by[n,'Tessera']['ari_mean']],marker='D',ms=3.5,color=col,ls='',clip_on=False)
            ax.plot([4.5],[by[n,'PCA50']['ari_mean']],marker='o',mfc='white',mec=col,ms=3.5,ls='',clip_on=False)
        ax.set_xscale('log');ax.set_xticks(ks,[str(k) for k in ks]);ax.set_xlim(4,290);ax.minorticks_off()
        ax.set_xlabel('Spatial neighbours $k$');ax.set_ylabel('Raw ARI')
        ax.legend(frameon=False,fontsize=7,loc='upper left',borderaxespad=0,ncol=2,columnspacing=.8,handletextpad=.3)
        lo,hi=ax.get_ylim();ax.set_ylim(0,hi*1.25)
    cv.text(W/2,210,'Open circles at the left: PCA50; diamonds at the right: full Tessera, same patient',size=7.5,ha='center')
    save(cv,'supplement05-scale-controls')


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs-only',action='store_true');args=p.parse_args()
    LED.mkdir(parents=True,exist_ok=True)
    inputs_figure()
    if args.inputs_only:return
    fu=followup()
    df,group,by,boot=summarize();comparison(by,fu);ablations(df,by,fu)
    cohort_maps(OBJECTS[:5],'figure05-nsclc-maps')
    cohort_maps(OBJECTS[5:9],'figure06-bcc-maps')
    histology();breast_mix();colorectal();transfers(fu);stability(df,by);efficiency(df,by)
    cohort_maps(['OpenST_HNSCC26'],'supplement01-hnscc');mibi();rna_markers();scale_curves(by,fu)
    stale=FIG/'supplement04-secondary-targets.pdf'
    if stale.exists():stale.unlink()
    dump(LED/'figure_manifest.json',{'figures':{p.name:sha(p) for p in sorted(FIG.glob('*.pdf'))},
        'primary_score_table_sha256':sha(LED/'all_primary_scores.csv'),
        'transfer_score_table_sha256':sha(LED/'all_transfer_scores.csv'),
        'font_contract_sha256':sha(MAN/'paper_fonts.tex')})

if __name__=='__main__':main()
