#!/usr/bin/env python3
"""Render existing ledgers at their printed size; no fitting or inference.

Panel dimensions, title centres and annotation offsets are explicit. Missing
method runs remain missing and are visibly marked NR, never imputed as zero.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
P=Path(__file__).resolve().parents[1];E=P.parent/'experiments'
def J(n):return json.loads((E/(n+'.json')).read_text())
B,G,O,PURPLE='#0072B2','#009E73','#D55E00','#7851B8'
plt.rcParams.update({'font.family':'Arial','font.size':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.labelsize':8,'axes.titlesize':9,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'legend.fontsize':7.5,'axes.linewidth':.7,'lines.linewidth':1.1})
font_manager.findfont(font_manager.FontProperties(family='Arial'),fallback_to_default=False)
def save(fig,n):
 fig.savefig(P/'figs'/f'{n}.pdf',metadata={'CreationDate':None,'ModDate':None})
 fig.savefig(P/'figs'/f'{n}.png',dpi=300);plt.close(fig)
def hdr(fig,ax,title,letter=None,xletter=None,y=.965):
 b=ax.get_position();fig.text((b.x0+b.x1)/2,y,title,ha='center',va='top',fontsize=9)
 if letter:fig.text(b.x0-.05 if xletter is None else xletter,y,letter,weight='bold',fontsize=11,va='top')
def short(s):return s.split('(')[0].replace('SlideseqV2','Slide-seqV2').replace('openST','Open-ST')

def render_mechanism(mc):
 # The 0.95-column IEEE embedding scales this master by ~0.9743.
 # Keep even tick and legend text above the actual 7.5-pt print floor.
 fig=plt.figure(figsize=(3.4,2.75));a=fig.add_axes([.20,.20,.76,.52])
 rr=sorted(mc['rows'],key=lambda r:r['contiguity']);x=[r['contiguity'] for r in rr]
 for k,c,m,l in [('adv_smooth',G,'o','Neighbour-mean advantage'),('adv_stagate',B,'s','STAGATE advantage'),('nonspatial_ari','#777','^','Non-spatial ARI')]:
  a.plot(x,[r[k] for r in rr],marker=m,ms=3.5,c=c,label=l)
 a.axhline(0,c='#999',ls='--',lw=.7);a.set_ylim(-.4,.63)
 a.set_xlabel('Manipulated GT contiguity',fontsize=8.3);a.set_ylabel('ARI / advantage (ΔARI)',fontsize=8.3)
 a.tick_params(labelsize=7.8)
 fig.legend(*a.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.57,.925),ncol=1,frameon=False,fontsize=7.8,labelspacing=.15)
 hdr(fig,a,'Synthetic position intervention');save(fig,'fig_mechanism')

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--only',choices=['fig_mechanism'],help='Render only this current figure, leaving historical assets untouched')
 args=parser.parse_args()
 if args.only:
  render_mechanism(J('mechanism_synth'))
  print('fig_mechanism rendered from saved ledger; no synthetic generation or model fitting.')
  raise SystemExit(0)

eb,gp,sr,nb,mc,ab=[J(n) for n in ['expanded_bench','gtfree_proxy','stat_rigor','native_baselines','mechanism_synth','mechanism_ablation']]
rows=gp['rows']
# Saved observations only; the association titles use saved ledger values.
fig=plt.figure(figsize=(7.16,3.2));axes=[fig.add_axes([.085,.20,.39,.64]),fig.add_axes([.59,.20,.39,.64])]
left={'CODEX':(4,7),'MIBI-TOF':(-26,16),'Slide-seqV2':(7,-13),'Open-ST':(-23,-15),'seqFISH':(9,4),'IMC':(3,5),'BRCA':(6,-4),'DLPFC':(5,-10),'STARmap':(3,4),'osmFISH':(-24,7),'MERFISH':(-23,7)}
right={'BRCA':(-25,8),'CODEX':(7,4),'Open-ST':(-20,-14),'seqFISH':(7,-13),'Slide-seqV2':(8,5),'MIBI-TOF':(-30,16),'IMC':(4,5),'DLPFC':(4,4),'osmFISH':(5,4),'STARmap':(-35,7),'MERFISH':(-28,7)}
for a,key,title,c,l,off in zip(axes,['GT_contiguity','coh_gain'],['Observed association: ρ = +0.72','Label-free proxy: ρ = +0.55'],[B,G],'AB',[left,right]):
 x=[r[key] for r in rows];y=[r['spatial_advantage'] for r in rows]
 a.scatter(x,y,s=20,c=c,zorder=3);a.axhline(0,c='#aaa',ls=':',lw=.7)
 a.set_ylim(-.075,.38);a.set_xlim(min(x)-.065,max(x)+.075)
 for r,xx,yy in zip(rows,x,y):
  n=short(r['platform']);a.annotate(n,(xx,yy),xytext=off[n],textcoords='offset points',fontsize=7.2,arrowprops={'arrowstyle':'-','lw':.4,'color':'#999'},zorder=4)
 a.set_xlabel('GT contiguity (reference labels)' if key=='GT_contiguity' else 'coh_gain (label-free)')
 hdr(fig,a,title,l,.018 if l=='A' else .523)
axes[0].set_ylabel('Spatial-prior advantage (ΔARI)');save(fig,'fig_law')
# Bootstrap and LODO estimates, read exactly from the existing ledger.
fig=plt.figure(figsize=(7.16,2.75));aa=fig.add_axes([.16,.26,.30,.55]);bb=fig.add_axes([.62,.26,.36,.55])
for i,key in enumerate(['bootstrap_gt_contiguity','bootstrap_gtfree_cohgain']):
 r=sr[key];rho=r['rho'];lo,hi=r['ci95'];y=1-i
 aa.errorbar(rho,y,xerr=[[rho-lo],[hi-rho]],fmt='o',color=[B,G][i],capsize=2.5,ms=4)
 aa.text(.4,y+.24,f'{rho:+.2f} [{lo:.2f}, {hi:.2f}]',ha='center',fontsize=7.5)
aa.set_yticks([1,0],['GT contiguity','Label-free proxy']);aa.set_ylim(-.42,1.6);aa.set_xlim(-.25,1.05);aa.axvline(0,c='#aaa',ls=':',lw=.7);aa.set_xlabel('Spearman correlation')
x=np.array([0,1]);w=.3
for i,k in enumerate(['lopo_gt_contiguity','lopo_gtfree_cohgain']):
 r=sr[k];bb.bar(x+(i-.5)*w,[r['loo_pred_vs_actual_rho'],r['decision_rule_accuracy']],w,color=[B,G][i],label=['GT contiguity','Label-free proxy'][i])
bb.axhline(sr['lopo_gt_contiguity']['base_rate_accuracy'],c='#888',ls=':',lw=.8);bb.text(1.30,.57,'base rate',ha='right',fontsize=7)
bb.set_xticks(x,['LODO ρ','Decision accuracy']);bb.set_ylim(0,1.03)
fig.legend(*bb.get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.75,.015),ncol=2,frameon=False,handlelength=1,columnspacing=.8)
hdr(fig,aa,'Saved 95% bootstrap intervals','A',.018);hdr(fig,bb,'Leave-one-dataset-out','B',.53);save(fig,'fig_robust')
# Native-unit confound comparisons, with the joint-control reference outside bars.
fig=plt.figure(figsize=(7.16,2.8));aa=fig.add_axes([.115,.22,.35,.59]);bb=fig.add_axes([.615,.22,.36,.59])
cm=sr['confound_marginal_spearman_vs_advantage'];keys=list(cm);ys=np.arange(len(keys))[::-1];vals=[cm[k]['rho'] for k in keys]
labels={'GT_contiguity':'Contiguity','k_nclasses':'Class count','eff_dim':'Dimension','modality_imaging':'Modality'}
aa.barh(ys,vals,color=[B]+['#9AA8B4']*3,height=.54);aa.axvline(0,c='#aaa',lw=.7)
for y,v in zip(ys,vals):aa.text(v+(0.025 if v>=0 else -.025),y,f'{v:+.2f}',ha='left' if v>=0 else 'right',va='center',fontsize=7.5)
aa.set_yticks(ys,[labels[k] for k in keys]);aa.set_xlim(-.85,.97);aa.set_xlabel('Spearman correlation')
cp=sr['contiguity_partial_spearman_controlling'];vals=list(cp.values());bb.bar(range(3),vals,color=G,width=.56)
for i,v in enumerate(vals):bb.text(i,v+.025,f'{v:.2f}',ha='center',fontsize=7.5)
bb.axhline(sr['contiguity_partial_controlling_all'],ls='--',color=O,lw=.9);bb.set_ylim(0,1.08);bb.set_xticks(range(3),['Class count','Dimension','Modality']);bb.set_ylabel('Partial Spearman correlation')
fig.text(.785,.865,'All controls: ρ = 0.667',color=O,ha='center',fontsize=7.5)
hdr(fig,aa,'Marginal association','A',.018);hdr(fig,bb,'Contiguity after adjustment','B',.53);save(fig,'fig_confound')
# Square identity comparison and a dedicated nine-method horizontal-bar panel.
fig=plt.figure(figsize=(7.16,3.45));aa=fig.add_axes([.10,.20,.34,.64]);bb=fig.add_axes([.685,.20,.29,.64])
native={r['platform']:r for r in nb['rows']};x=[r['spatial_advantage'] for r in eb['rows']];y=[native[r['platform']]['spatial_advantage'] for r in eb['rows']]
lim=(min(x+y)-.025,max(x+y)+.025);aa.plot(lim,lim,c='#aaa',ls='--',lw=.7);aa.scatter(x,y,c=PURPLE,s=21);aa.set_xlim(lim);aa.set_ylim(lim);aa.set_aspect('equal',adjustable='box');aa.set_xlabel('Historical advantage (ΔARI)');aa.set_ylabel('mclust-backend advantage (ΔARI)')
d=nb['mean_delta_vs_published_per_method'];names=list(d);yy=np.arange(len(names))[::-1];bb.barh(yy,list(d.values()),color=PURPLE,alpha=.8,height=.65);bb.set_yticks(yy,[n.replace('floor:nonspatial','Non-spatial floor').replace('floor:smoothed','Smoothed floor') for n in names]);bb.axvline(0,c='#aaa',lw=.7);bb.set_xlabel('Mean ARI difference');bb.set_xticks([0,.01]);bb.set_xlim(-.006,.018)
hdr(fig,aa,'Backend agreement','A',.018);hdr(fig,bb,'Method-wise backend differences','B',.53);save(fig,'fig_backend')
# Same rank ordering as the original R renderer; no new outcome estimate.
ordered=sorted(eb['rows'],key=lambda r:r['GT_contiguity']);focus=['Tessera','SpaceFlow','SpatialLeiden','STAGATE']
fig=plt.figure(figsize=(6.444,3.2));a=fig.add_axes([.075,.31,.91,.46])
for name,c in zip(focus,[B,O,'#C13237',G]):
 ys=[]
 for r in ordered:
  eligible={k:v for k,v in r['means'].items() if k in eb['method_panel']}
  ys.append(sorted(eligible,key=eligible.get,reverse=True).index(name)+1 if name in eligible else np.nan)
 a.plot(range(len(ordered)),ys,'o-',ms=3.4,c=c,label=name)
a.set_xticks(range(len(ordered)),[short(r['platform']) for r in ordered],rotation=35,ha='right');a.set_yticks([1,3,5,7,9]);a.set_ylim(9.4,.6);a.set_ylabel('Rank (1 = best)');a.set_xlabel('Datasets ordered by increasing reference-label contiguity')
fig.legend(*a.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.54,.92),ncol=4,frameon=False,handlelength=1.5,columnspacing=1)
hdr(fig,a,'Method rankings across the stored tissue panel');save(fig,'fig_profile')
# Explicit NR cells replace ambiguous white gaps in the method table.
methods=eb['method_panel'];matrix=np.array([[r['means'].get(m,np.nan) for r in ordered] for m in methods])
order=np.argsort(-np.nanmean(matrix,axis=1));matrix=matrix[order];methods=[methods[i] for i in order]
fig=plt.figure(figsize=(7.16,4.0));a=fig.add_axes([.17,.23,.745,.63]);cax=fig.add_axes([.935,.38,.015,.28])
cmap=plt.get_cmap('magma').copy();cmap.set_bad('#eeeeee');im=a.imshow(np.ma.masked_invalid(matrix),aspect='auto',cmap=cmap,vmin=np.nanmin(matrix),vmax=np.nanmax(matrix))
for i,m in enumerate(methods):
 for j,r in enumerate(ordered):
  v=matrix[i,j]
  if np.isfinite(v):a.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=7,color='black' if im.norm(v)>.62 else 'white')
  else:a.text(j,i,'NR',ha='center',va='center',fontsize=7,color='#555')
  if r['winner']==m:a.add_patch(Rectangle((j-.5,i-.5),1,1,fill=False,lw=1.0,edgecolor='black'))
a.set_xticks(range(len(ordered)),[short(r['platform']) for r in ordered],rotation=40,ha='right');a.set_yticks(range(len(methods)),[m.replace('floor:nonspatial','Non-spatial floor').replace('floor:smoothed','Smoothed floor').replace('BANKSY-style','BANKSY') for m in methods]);a.tick_params(length=0);a.set_xticks(np.arange(-.5,len(ordered),1),minor=True);a.set_yticks(np.arange(-.5,len(methods),1),minor=True);a.grid(which='minor',c='white',lw=.4);a.tick_params(which='minor',length=0);a.spines[['left','bottom']].set_visible(False)
cb=fig.colorbar(im,cax=cax);cb.ax.set_title('ARI',fontsize=8,pad=5);cb.ax.tick_params(labelsize=7)
hdr(fig,a,'Stored method-by-dataset agreement');fig.text(.53,.027,'Outlined cells: recorded winners.  NR: no recorded run; values are not imputed.',ha='center',fontsize=7.5);save(fig,'fig_heatmap')
# One-column synthetic figure: native aspect, external legend, no legend covers data.
render_mechanism(mc)
# Saved proxy correlations on their natural common scale.
fig=plt.figure(figsize=(4.44,2.9));a=fig.add_axes([.16,.22,.80,.53]);nv=gp['naive_proxies_FAIL'];pr=gp['principled_proxy_coh_gain'];xx=np.arange(3)
y1=[nv['moran']['vs_gtcontig'][0],nv['raw_coh']['vs_gtcontig'][0],pr['vs_gtcontig'][0]];y2=[nv['moran']['vs_advantage'][0],nv['raw_coh']['vs_advantage'][0],pr['vs_advantage'][0]]
a.bar(xx-.17,y1,.32,color='#7BACCE',label='GT contiguity');a.bar(xx+.17,y2,.32,color=O,label='Spatial advantage');a.axhline(eb['spearman_contiguity_vs_spatial_advantage'],c=B,ls=':',lw=.8);a.set_ylim(-.5,.88);a.set_xticks(xx,['Moran I','Raw coherence','coh_gain']);a.set_ylabel('Spearman correlation');a.text(-.4,.75,'Reference association: ρ = +0.72',fontsize=7.3,color=B)
fig.legend(*a.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.56,.87),ncol=2,frameon=False);hdr(fig,a,'Saved proxy associations');save(fig,'fig_proxy')
# Direct saved component drops: sign is stated algebraically, not ambiguously.
fig=plt.figure(figsize=(7.16,2.8));a=fig.add_axes([.28,.25,.69,.52]);comps=['boundary_contrastive','edge_gating','multi_scale'];yy=np.arange(3)[::-1]
for i,(key,c) in enumerate(zip(ab,[B,O])):
 vals=[ab[key]['component_drop'][k] for k in comps];ys=yy+(i-.5)*.24;a.barh(ys,vals,height=.22,color=c,label=short(key))
 for y,v in zip(ys,vals):a.text(v+(.008 if v>=0 else -.008),y,f'{v:+.2f}',va='center',ha='left' if v>=0 else 'right',fontsize=8)
a.set_yticks(yy,['Boundary-contrastive loss','Edge gating','Multi-scale fusion']);a.set_xlim(-.085,.34);a.axvline(0,c='#555',lw=.8);a.set_xlabel('ARI(full model) − ARI(with component removed)');fig.legend(*a.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.61,.9),ncol=2,frameon=False);hdr(fig,a,'Stored component-ablation contrasts');save(fig,'fig_ablation')
print('Nine statistical figures rendered from existing ledgers; no model or calibration fitting.')
