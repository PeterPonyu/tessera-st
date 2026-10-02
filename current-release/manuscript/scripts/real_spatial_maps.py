#!/usr/bin/env python3
"""Draw existing tissue images and locked maps, with MIBI fields kept separate.

This renderer never computes or refits a prediction. The superseded pooled-field
MIBI candidate is explicitly excluded from the current input set.
"""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.lines import Line2D

HERE=Path(__file__).resolve().parent
OBJECTS=HERE.parent/'figs/objects'
FIGS=HERE.parent/'figs'
COLORS=['#5B4DA3','#5B8DC4','#67B6C4','#70AC85','#C5AD53','#D78C51','#8C665C','#CC79A7']

def axes(fig,x,y,w,h,tag,title,subtitle):
    a=fig.add_axes([x,y,w,h]);a.set_axis_off()
    fig.text(x-.008,y+h+.075,tag,weight='bold',fontsize=10,va='top')
    fig.text(x+w/2,y+h+.075,title,ha='center',va='top',fontsize=8)
    fig.text(x+w/2,y+h+.038,subtitle,ha='center',va='top',fontsize=7.6)
    return a

def scale_bar(a,x,y,span,label):
    a.plot([x,x+span],[y,y],color='#222222',lw=1.8)
    a.text(x+span/2,y-80,label,fontsize=7.6,ha='center',va='bottom',color='#222222')

def render():
    p=json.loads((OBJECTS/'provenance.json').read_text());dl,mi=p['cases']
    font_manager.findfont(font_manager.FontProperties(family='Arial'),fallback_to_default=False)
    plt.rcParams.update({'font.family':'Arial','font.size':8,'pdf.fonttype':42})
    fig=plt.figure(figsize=(7.16,4.58),facecolor='white')
    xs=[.035,.278,.521,.764];w=.220;h=.330
    df=pd.read_csv(OBJECTS/'DLPFC.csv');im=Image.open(OBJECTS/dl['image']);scale=dl['image_scale']
    bounds=(2500,11600,12100,1900)
    titles=['Source histology','Reference layers','Expression only','Neighbour mean']
    subs=['DLPFC section 151673',f"{dl['n_labelled']:,} labelled spots",'Saved seed-1 map','Saved seed-1 map']
    for j in range(4):
        a=axes(fig,xs[j],.580,w,h,chr(65+j),titles[j],subs[j])
        if j==0:a.imshow(im,extent=[0,im.width/scale,im.height/scale,0])
        else:
            key=['reference','nonspatial_display','smoothed_display'][j-1]
            a.scatter(df.x,df.y,c=[COLORS[int(k)] if k>=0 else '#dddddd' for k in df[key]],s=1.75,lw=0,rasterized=True)
        a.set_xlim(bounds[:2]);a.set_ylim(bounds[2:]);a.set_aspect('equal')
        if j==0:scale_bar(a,4000,11500,2000,'2,000 px')
    fig.legend(handles=[Line2D([],[],marker='o',ls='',color=COLORS[i],markersize=4,label=k.replace('Layer','L')) for i,k in enumerate(dl['classes'])],loc='center',bbox_to_anchor=(.52,.553),ncol=7,frameon=False,fontsize=7.6,handletextpad=.2,columnspacing=.8)
    # Source RGB and exact supplied segmentation, never concatenated coordinates.
    fields=[mi['fields'][0]]+mi['fields']
    for j,f in enumerate(fields):
        table=pd.read_csv(OBJECTS/f['table']);rgb=np.asarray(Image.open(OBJECTS/f['image']))
        title='Source RGB composite' if j==0 else 'Reference cell classes'
        a=axes(fig,xs[j],.105,w,h,chr(69+j),title,f"{f['field']} · {f['n_cells']:,} cells")
        if j==0:a.imshow(rgb)
        else:
            seg=np.asarray(Image.open(OBJECTS/f"MIBI-{f['field']}-segmentation.png"));lut=np.ones((int(seg.max())+1,4));lut[:,:3]=.965
            from matplotlib.colors import to_rgba
            for r in table.itertuples():lut[int(r.segmentation_id)]=to_rgba(COLORS[int(r.reference)])
            a.imshow(lut[seg],interpolation='nearest')
        a.set_xlim(0,1024);a.set_ylim(1024,0);a.set_aspect('equal')
        if j==0:
            a.plot([120,320],[960,960],color='white',lw=2);a.text(220,910,'200 px',color='white',ha='center',fontsize=7.6)
    names={'Endothelial':'Endothelial','Epithelial':'Epithelial','Fibroblast':'Fibroblast','Imm_other':'Other immune','Myeloid_CD11c':'CD11c myeloid','Myeloid_CD68':'CD68 myeloid','Tcell_CD4':'CD4 T cell','Tcell_CD8':'CD8 T cell'}
    fig.legend(handles=[Line2D([],[],marker='s',ls='',color=COLORS[i],markersize=4,label=names[k]) for i,k in enumerate(mi['classes'])],loc='center',bbox_to_anchor=(.515,.067),ncol=4,frameon=False,fontsize=7.6,handletextpad=.2,columnspacing=.9)
    fig.text(.51,.012,'MIBI-TOF: three separate fields · pixel scales are source coordinates, not calibrated physical lengths',ha='center',fontsize=7.6,color='#444444')
    fig.savefig(FIGS/'fig_spatialmap.pdf',dpi=450,metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(FIGS/'fig_spatialmap.png',dpi=300);plt.close(fig)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--render-only',action='store_true');parser.parse_args();render()
