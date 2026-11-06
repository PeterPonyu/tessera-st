"""Use automatic subplot alignment for denser derivatives of saved measurements."""
from pathlib import Path
import importlib.util
import json
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm



def load(name):
    # inventory copy: sibling module; outputs below inventory OUT
    spec=importlib.util.spec_from_file_location('figure_module',Path(__file__).with_name('frontiers_redraw_figures.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);root=m.OUT
    def save(fig,label):
        pad = .08 if label in ['figure-03', 'figure-12'] else .04
        fig.savefig(m.DEST/(label+'.pdf'),bbox_inches='tight',pad_inches=pad,dpi=300,metadata={'CreationDate':None,'ModDate':None})
        fig.savefig(m.QA/(label+'.png'),bbox_inches='tight',pad_inches=pad,dpi=150);m.plt.close(fig)
    m.save=save;return m,root


def title(ax,letter,text):ax.set_title(letter+'  '+text,loc='left',fontsize=9.5,pad=5)


def tessera():
    m,root=load('tessera-st');m.architecture();m.ablations();m.transfer()
    fig=m.plt.figure(figsize=(6.9,5.8),layout='constrained');gs=fig.add_gridspec(2,3,height_ratios=[1.75,1])
    axes=[fig.add_subplot(gs[0,i]) for i in range(3)];y=np.arange(13)
    for ax,letter,text in zip(axes,'abc',['Partition agreement','Subsampled fields','Whole sections']):
        title(ax,letter,text);ax.set_ylim(12.6,-.6);ax.set_yticks(y);ax.tick_params(axis='y',length=0)
    axes[0].set_yticklabels([m.SHORT[n] for n in m.OBJECTS])
    for ax in axes[1:]:ax.set_yticklabels([])
    for arm,marker in [('PCA50','o'),('NeighborMean','s'),('Tessera','o'),('Tuned','D')]:
        values=[m.nested[n]['nested_control_ari'] if arm=='Tuned' and n in m.nested else np.nan if arm=='Tuned' else m.summary.loc[(n,arm),'ari_mean'] for n in m.OBJECTS]
        axes[0].scatter(values,y,s=16,marker=marker,color=m.COLOR[arm])
    axes[0].set_xlim(0,.56);axes[0].set_xticks([0,.2,.4]);axes[0].set_xlabel('ARI')
    vp=[m.summary.loc[(n,'Tessera'),'ari_mean']-m.summary.loc[(n,'PCA50'),'ari_mean'] for n in m.OBJECTS[:9]]
    vt=[m.summary.loc[(n,'Tessera'),'ari_mean']-m.nested[n]['nested_control_ari'] for n in m.OBJECTS[:9]]
    whole=m.read(m.FOLLOW/'whole_sections_summary.json');wp=whole['NSCLC']['tessera_minus_pca50']+whole['BCC']['tessera_minus_pca50'];wt=whole['NSCLC']['tessera_minus_selected_control']+whole['BCC']['tessera_minus_selected_control']
    for ax,p,t in [(axes[1],vp,vt),(axes[2],wp,wt)]:
        ax.axvline(0,color='black',lw=.7);ax.scatter(p,np.arange(9),s=17,facecolors='white',edgecolors=m.COLOR['PCA50']);ax.scatter(t,np.arange(9),s=17,marker='D',facecolors='white',edgecolors=m.COLOR['Tuned'])
        ax.set_xlim(-.30,.22);ax.set_xticks([-.2,0,.2]);ax.set_xlabel('Tessera - control\n< Control   Tessera >',fontsize=8.5)
    d=fig.add_subplot(gs[1,0]);title(d,'d','Cell-type cluster count')
    rescore=m.csv(m.FOLLOW/'rescore.csv');r=rescore[rescore.object.isin(m.OBJECTS[:5])].groupby(['object','arm']).ari_celltype_at_celltype_K.mean()
    for arm,marker in [('PCA50','o'),('NeighborMean','s'),('Tessera','o')]:d.scatter([r[n,arm] for n in m.OBJECTS[:5]],np.arange(5),s=17,marker=marker,color=m.COLOR[arm])
    d.set_yticks(np.arange(5),[m.SHORT[n] for n in m.OBJECTS[:5]]);d.set_ylim(4.6,-.6);d.set_xlim(0,.4);d.set_xlabel('Cell-type ARI')
    scales=m.csv(m.FOLLOW/'scale_controls.csv');ks=[6,10,15,30,50,100,200];shades=['#26759c','#b55738','#688b45','#9c6295','#448c81']
    for i,letter,text,names,kind in [(1,'e','Lung neighborhood scale',m.OBJECTS[:5],'NeighborMean'),(2,'f','BCC neighborhood scale',m.OBJECTS[5:9],'BANKSY-style')]:
        ax=fig.add_subplot(gs[1,i]);title(ax,letter,text)
        for n,col in zip(names,shades):
            sub=scales[(scales.object==n)&(scales.kind==kind)]
            if kind=='BANKSY-style':sub=sub[np.isclose(sub.lam,.2)]
            values=sub.groupby('k').ari.mean().reindex(ks);ax.plot(ks,values,'o-',color=col,lw=1.1,ms=2,label=m.SHORT[n].replace('Lung','L').replace('BCC ','B'))
        ax.set_xscale('log');ax.set_xticks([6,30,200],['6','30','200']);ax.set_ylim(bottom=0);ax.set_xlabel('Neighbors k');ax.set_ylabel('ARI')
        # Constrained layout reserves space for a legend outside the data box.
        ax.legend(loc='upper center',bbox_to_anchor=(.5,-.36),frameon=False,ncol=2,fontsize=8.5,handlelength=.9,columnspacing=.6,handletextpad=.25)
    handles=[Line2D([0],[0],marker=marker,ls='',color=m.COLOR[arm],label=label) for arm,marker,label in [('PCA50','o','PCA50'),('NeighborMean','s','Same-graph mean'),('Tuned','D','Tuned control'),('Tessera','o','Tessera')]]
    fig.legend(handles=handles,loc='outside upper center',ncol=4,frameon=False,fontsize=8.5)
    m.PLOT_DATA['figure3']={'bounded_vs_pca':vp,'bounded_vs_tuned':vt,'whole_vs_pca':wp,'whole_vs_tuned':wt,'note':'Saved comparisons; legends are outside the scale-curve data boxes.'};m.save(fig,'figure-03')
    table=m.csv(m.LED/'landscape.csv');fig=m.plt.figure(figsize=(6.9,6.25),layout='constrained');gs=fig.add_gridspec(2,4,height_ratios=[4,1]);N=len(table)
    fields=['tessera_minus_same_graph','tessera_minus_oracle','tessera_minus_pca50']
    for i,text in enumerate(['Same-graph mean','Test-selected','PCA50','Seam AUROC']):
        ax=fig.add_subplot(gs[0,i]);title(ax,'abcd'[i],text);ax.set_ylim(N-.5,-.5);ax.set_yticks(np.arange(N),table.short_label if i==0 else []);ax.tick_params(axis='y',length=0)
        if i<3:
            ax.scatter(table[fields[i]],np.arange(N),s=16,marker=['s','D','o'][i],facecolors='white',edgecolors=[m.COLOR['NeighborMean'],m.COLOR['Tuned'],m.COLOR['PCA50']][i]);ax.axvline(0,color='black',lw=.6);ax.set_xlim(-.4,.5);ax.set_xticks([-.3,0,.3]);ax.set_xlabel('Tessera - control\n< Control   Tessera >',fontsize=8.5)
        else:
            ax.scatter(table.gate_auroc,np.arange(N),s=16,color=m.COLOR['Tessera']);ax.scatter(table.raw_difference_auroc,np.arange(N),s=16,marker='s',facecolors='white',edgecolors=m.COLOR['PCA50']);ax.axvline(.5,color='#aaa',ls='--',lw=.8);ax.set_xlim(.2,.9);ax.set_xticks([.3,.5,.8]);ax.set_xlabel('AUROC')
    seam_keys=[Line2D([0],[0],marker='o',ls='',color=m.COLOR['Tessera'],label='Learned gate'),Line2D([0],[0],marker='s',ls='',markerfacecolor='white',color=m.COLOR['PCA50'],label='PCA-feature difference')]
    fig.legend(handles=seam_keys,loc='outside upper center',ncol=2,frameon=False,fontsize=8.5)
    ax=fig.add_subplot(gs[1,:]);title(ax,'e','Best uniform scale, selected using each object’s labels');groups=table.group.drop_duplicates().tolist();counts=[]
    for row,group in enumerate(groups):
        frequency=table.loc[table.group==group,'best_uniform_k'].value_counts().sort_index()
        for k,n in frequency.items():
            ax.scatter(k,row,s=42+24*n,color='#657780',edgecolors='white',lw=.6);ax.text(k,row,str(n),color='white',fontsize=8.5,ha='center',va='center',weight='bold');counts.append({'group':group,'k':int(k),'objects':int(n)})
    assert sum(item['objects'] for item in counts)==24
    ax.set_yticks(np.arange(len(groups)),groups);ax.set_ylim(len(groups)-.6,-.6);ax.set_xscale('log');ax.set_xticks([6,10,15,30,50,100,200],['6','10','15','30','50','100','200']);ax.set_xlabel('Neighbors k; bubble numbers count objects')
    m.PLOT_DATA['figure12']=table.to_dict('records');m.PLOT_DATA['figure12_scale_multiplicities']=counts;m.save(fig,'figure-12')
    (root/'qa/plot-data.json').write_text(json.dumps(m.finite_metadata(m.PLOT_DATA),indent=2,default=float,allow_nan=False)+'\n');(root/'qa/figure-input-hashes.json').write_text(json.dumps(m.INPUTS,indent=2)+'\n')


def raag():
    m,root=load('reference-anchored-geometry');m.paired_tables();m.chance()
    original=m.read(m.LEGACY/'evidence/scale-correspondence/results.json');df=m.pd.DataFrame(original['scale']);base=df[(df.k==15)&(df['rank']==50)].set_index('id')
    metrics=['reachability_survival','edge_survival','separation_survival','allocation_coherence']
    for key in metrics:df['delta_'+key]=[row[key]-base.loc[row.id,key] for _,row in df.iterrows()]
    rows=[(bg,native) for bg in m.BG for native in [True,False]];settings=[(k,d) for k in [5,15,30] for d in [10,25,50]];matrices=[]
    fig,axs=m.plt.subplots(2,2,figsize=(6.9,5.35),layout='constrained')
    for i,(key,text,ax) in enumerate(zip(metrics,['Reachability R','Edge survival change','Separation change','Native coherence change'],axs.flat)):
        title(ax,'abcd'[i],text);matrix=np.full((8,9),np.nan);field=key if i==0 else 'delta_'+key
        for ri,(bg,native) in enumerate(rows):
            for ci,(k,d) in enumerate(settings):
                selected=df[(df.dataset==bg)&(df.native==native)&(df.k==k)&(df['rank']==d)]
                if i<3 or native:matrix[ri,ci]=selected[field].mean()
        cmap=m.plt.get_cmap('viridis' if i==0 else 'RdBu_r').copy();cmap.set_bad('#e5e5e5')
        if i==0:im=ax.imshow(matrix,cmap=cmap,vmin=0,vmax=1,aspect='auto')
        else:
            limit=max(.01,np.nanmax(np.abs(matrix)));im=ax.imshow(matrix,cmap=cmap,norm=TwoSlopeNorm(vcenter=0,vmin=-limit,vmax=limit),aspect='auto')
        for ri in range(8):
            for ci in range(9):
                v=matrix[ri,ci];printed='—' if not np.isfinite(v) else '0.00' if abs(v)<.005 else f'{v:.2f}'
                color='white' if np.isfinite(v) and ((i==0 and v<.5) or (i>0 and abs(v)>.65*limit)) else 'black';ax.text(ci,ri,printed,ha='center',va='center',fontsize=8.5,color=color)
        ax.set_xticks(np.arange(9),[f'{k}\n{d}' for k,d in settings]);ax.tick_params(length=0,pad=2);ax.set_yticks(np.arange(8),[m.BG[bg]+('/P' if native else '/Z') for bg,native in rows] if i%2==0 else []);ax.set_xlabel('k above / d below',fontsize=8.5)
        fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.07,pad=.04,aspect=25);matrices.append(matrix.tolist())
    m.PLOTDATA['scale']={'matrices':matrices,'row_order':rows,'settings':settings,'original_record_count':len(df),'numeric_annotations':'Two decimals; magnitude below .005 prints zero; gray is inapplicable.'};m.save(fig,'scale_reference_sensitivity')
    corr=m.pd.DataFrame(original['correspondence']);assert np.all(corr.reachability_survival==1)
    fig,axs=m.plt.subplots(1,3,figsize=(6.9,2.25),layout='constrained',sharey=True)
    for i,(key,text,ax) in enumerate(zip(['edge_survival','separation_survival','allocation_coherence'],['Edge survival E','Separation S','Native coherence C'],axs)):
        title(ax,'abc'[i],text);ax.axhline(1,color='black',ls='--',lw=.8)
        for (identity,background),g in corr.groupby(['id','dataset']):
            curve=g.groupby('requested_fraction')[key].mean().sort_index();seed=int(identity.rsplit('s',1)[-1]);ax.plot(curve.index*100,curve.values,color=m.COLORS[list(m.BG).index(background)],lw=1.1,ls=['-','--',':'][seed])
        ax.set_ylim(-.015,1.04);ax.set_xticks([0,50,100]);ax.set_xlabel('Reassigned rows (%)',fontsize=8.5);ax.set_yticks([0,.5,1])
    axs[0].set_ylabel('Score')
    handles=[Line2D([0],[0],color=col,label=m.BG[k]) for k,col in zip(m.BG,m.COLORS)]
    handles += [Line2D([0],[0],color='#444',ls=style,label=f'Seed {i}') for i,style in enumerate(['-','--',':'])]
    handles += [Line2D([0],[0],color='black',ls='--',label='R=1 ceiling')]
    fig.legend(handles=handles,loc='outside upper center',ncol=4,frameon=False,fontsize=8.5)
    m.PLOTDATA['correspondence']=corr.to_dict('records');m.save(fig,'row_correspondence_sensitivity')
    (root/'qa/plot-data.json').write_text(json.dumps(m.finite_metadata(m.PLOTDATA),indent=2,default=float,allow_nan=False)+'\n');(root/'qa/figure-input-hashes.json').write_text(json.dumps(m.HASHES,indent=2)+'\n')


if __name__=='__main__':tessera();print('Denser aligned plots drawn exclusively from saved results.')
