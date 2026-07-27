from pathlib import Path
import os
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Circle

ROOT=Path(os.environ.get("PNEUMONIA_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
OUT=ROOT/"reports"/"figures"/"nature_redesign"; OUT.mkdir(parents=True,exist_ok=True)
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Microsoft YaHei","DejaVu Sans"],
 "svg.fonttype":"none","pdf.fonttype":42,"font.size":7,"axes.linewidth":.7,"axes.spines.top":False,
 "axes.spines.right":False,"legend.frameon":False,"figure.facecolor":"white","axes.facecolor":"white"})
INK="#24313D"; MUTED="#6B7785"; BLUE="#3B73A6"; BLUE2="#AFCBE0"; PALE="#EEF4F7"; CORAL="#C86D67"; CORAL2="#F2D4D0"; GOLD="#D8A94B"; GREEN="#77A989"; LINE="#8B99A6"

def save(fig,name):
    base=OUT/name
    fig.savefig(base.with_suffix('.png'),dpi=600,bbox_inches='tight',facecolor='white')
    fig.savefig(base.with_suffix('.svg'),bbox_inches='tight',facecolor='white')
    fig.savefig(base.with_suffix('.pdf'),bbox_inches='tight',facecolor='white')
    fig.savefig(base.with_suffix('.tiff'),dpi=600,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def box(ax,x,y,w,h,text,fc,ec=LINE,lw=.8,fs=7,bold=False,r=.04):
    p=FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=.018,rounding_size={r}",fc=fc,ec=ec,lw=lw)
    ax.add_patch(p); ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs,color=INK,weight='bold' if bold else 'normal',linespacing=1.25)
    return p
def arrow(ax,a,b,color=LINE,lw=1.1,rad=0):
    ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='-|>',color=color,lw=lw,mutation_scale=8,connectionstyle=f'arc3,rad={rad}'))

def study_design():
    fig,ax=plt.subplots(figsize=(7.2,3.35)); ax.set_xlim(0,12); ax.set_ylim(0,5.4); ax.axis('off')
    ax.text(.15,5.12,'a',fontsize=9,weight='bold'); ax.text(.48,5.12,'Cohort construction and leakage-aware evaluation',fontsize=10.5,weight='bold',color=INK)
    box(ax,.25,2.05,1.55,1.15,'Kaggle v2\n5,856 CXRs',PALE,bold=True,fs=7.5)
    ax.text(2.05,4.45,'Patient-key derivation',fontsize=7.0,weight='bold',color=INK)
    box(ax,2.08,3.25,1.75,.75,'Filename patient ID',BLUE2)
    box(ax,2.08,2.05,1.75,.75,'Subtype namespace',CORAL2)
    box(ax,2.08,.85,1.75,.75,'Leakage audit',"#F3E7C9")
    arrow(ax,(1.8,2.63),(2.08,3.62)); arrow(ax,(1.8,2.63),(2.08,2.42)); arrow(ax,(1.8,2.63),(2.08,1.22))
    ax.plot([4.28,4.28],[.55,4.6],color="#D8DEE3",lw=.8)
    ax.text(4.55,4.45,'Grouped stratified split',fontsize=7.0,weight='bold',color=INK)
    split=[('Train','4,699','80%',BLUE2,3.55),('Validation','582','10%',"#DDE8D6",2.25),('Test','575','10%',CORAL2,.95)]
    for lab,n,pct,c,y in split:
        box(ax,4.6,y,1.75,.82,f'{lab}  {pct}\n{n} images',c,bold=(lab=='Test'))
        arrow(ax,(3.83,2.42),(4.6,y+.41),rad=(y-2.25)*-.04)
    ax.plot([6.8,6.8],[.55,4.6],color="#D8DEE3",lw=.8)
    ax.text(7.05,4.45,'Locked test evaluation',fontsize=7.0,weight='bold',color=INK)
    box(ax,7.05,2.95,2.0,1.05,'Stage 1\nNormal vs pneumonia\nn = 575',"#D9E7F1",ec=BLUE,bold=True)
    box(ax,9.62,2.95,2.0,1.05,'Stage 2\nBacterial vs viral\nn = 417',"#F0D5D2",ec=CORAL,bold=True)
    box(ax,7.05,1.15,2.0,.88,'Bootstrap CI\nPaired comparison',PALE)
    box(ax,9.62,1.15,2.0,.88,'Source-domain audit\nPatient aggregation',"#F3E7C9")
    arrow(ax,(6.35,1.36),(7.05,3.48)); arrow(ax,(9.05,3.48),(9.62,3.48),color=CORAL)
    arrow(ax,(8.05,2.95),(8.05,2.03)); arrow(ax,(10.62,2.95),(10.62,2.03))
    ax.text(.25,.2,'Patient grouping precedes all model fitting; validation selects checkpoints, thresholds and ensemble weights; test data are used only for final evaluation.',fontsize=6.6,color=MUTED)
    save(fig,'figure1_study_design')

def architecture():
    fig,ax=plt.subplots(figsize=(7.2,4.55)); ax.set_xlim(0,14); ax.set_ylim(0,8.8); ax.axis('off')
    ax.text(.1,8.5,'a',fontsize=9,weight='bold'); ax.text(.43,8.5,'Two-stage prediction pathway',fontsize=10.5,weight='bold',color=INK)
    # image stack
    for k in range(3):
        ax.add_patch(FancyBboxPatch((.35+k*.08,5.35-k*.08),1.25,1.55,boxstyle='round,pad=.02',fc=['#F7F8F9','#E8EDF1','#DCE5EA'][k],ec=LINE,lw=.7))
    ax.text(1.05,6.05,'Chest X-ray\n224 × 224',ha='center',va='center',fontsize=7,color=INK)
    box(ax,2.25,5.28,2.0,1.45,'EfficientNet-B0\nmultiscale features',"#D9E7F1",ec=BLUE,bold=True)
    # feature maps
    for k,h in enumerate([1.15,.9,.68]):
        x=4.72+k*.26; y=5.45+(1.15-h)/2
        ax.add_patch(Polygon([[x,y],[x+.8,y+.15],[x+.8,y+h+.15],[x,y+h]],closed=True,fc=[BLUE2,"#C4D9E7","#D9E7F1"][k],ec=BLUE,lw=.6))
    ax.text(5.38,5.12,'Feature pyramid',ha='center',fontsize=6.5,color=MUTED)
    box(ax,6.18,5.28,1.65,1.45,'GeM pooling\n+ dropout',"#F3E7C9",ec=GOLD,bold=True)
    box(ax,8.42,5.28,1.65,1.45,'Stage 1 head\nP(pneumonia)',"#D9E7F1",ec=BLUE,bold=True)
    arrow(ax,(1.68,6.03),(2.25,6.03)); arrow(ax,(4.25,6.03),(4.7,6.03)); arrow(ax,(5.83,6.03),(6.18,6.03)); arrow(ax,(7.83,6.03),(8.42,6.03))
    box(ax,10.68,6.15,2.25,.9,'Normal',PALE,ec=LINE,bold=True)
    box(ax,10.68,4.83,2.25,.9,'Pneumonia',CORAL2,ec=CORAL,bold=True)
    arrow(ax,(10.07,6.03),(10.68,6.6),color=BLUE,rad=-.1); arrow(ax,(10.07,6.03),(10.68,5.28),color=CORAL,rad=.1)
    # Stage 2
    ax.text(.1,3.85,'b',fontsize=9,weight='bold'); ax.text(.43,3.85,'Etiologic classification and validation-only fusion',fontsize=10.5,weight='bold',color=INK)
    box(ax,.35,1.88,1.65,1.12,'Pneumonia\nsubset',CORAL2,ec=CORAL,bold=True)
    variants=[('EfficientNet-B0','baseline'),('PneuNet-v2','ConvNeXt + GeM'),('PneuNet-noAtt','attention ablation'),('Embedding LR','1280-d fusion')]
    xs=[2.45,4.5,6.55,8.6]
    for (name,sub),x,c in zip(variants,xs,["#D9E7F1","#D9E7F1","#D9E7F1","#F3E7C9"]):
        box(ax,x,1.78,1.72,1.35,f'{name}\n{sub}',c,ec=BLUE if c!="#F3E7C9" else GOLD,bold=True,fs=6.6)
    # parallel fan-out and merge buses keep arrows outside label regions
    arrow(ax,(2.0,2.44),(2.22,3.42)); ax.plot([2.22,10.32],[3.42,3.42],color=LINE,lw=1)
    for x in xs: arrow(ax,(x+.86,3.42),(x+.86,3.13))
    box(ax,11.15,1.65,2.15,1.62,'Validation-only\nlogistic stacking\nP(viral)',"#DCE9DF",ec=GREEN,bold=True,fs=7.2)
    ax.plot([3.31,10.32],[1.42,1.42],color=GREEN,lw=1)
    for x in xs: arrow(ax,(x+.86,1.78),(x+.86,1.42),color=GREEN)
    arrow(ax,(10.32,1.42),(11.15,2.15),color=GREEN)
    box(ax,11.22,.38,.88,.62,'Bacterial',"#D9E7F1",ec=BLUE,fs=6.5)
    box(ax,12.38,.38,.88,.62,'Viral',CORAL2,ec=CORAL,fs=6.5)
    arrow(ax,(12.22,1.65),(11.67,1.0),color=BLUE); arrow(ax,(12.22,1.65),(12.82,1.0),color=CORAL)
    ax.text(.35,.38,'All checkpoints, thresholds and stacking weights are selected on validation data before the independent test is opened.',fontsize=6.5,color=MUTED)
    save(fig,'figure2_model_architecture')

def results():
    met=pd.read_csv(ROOT/'reports/final_metrics.csv'); grouped=pd.read_csv(ROOT/'reports/grouped_bootstrap_auc.csv'); src=pd.read_csv(ROOT/'reports/source_directory_grouped_ci.csv')
    fig=plt.figure(figsize=(7.2,5.6)); gs=fig.add_gridspec(2,5,height_ratios=[1.18,1],wspace=1.0,hspace=.82)
    ax=fig.add_subplot(gs[0,:3]); q=met[(met.task=='stage2')&met.model.isin(['hog_linear_svm','resnet18','efficientnet_b0','pneunet_no_attention','pneunet_v2','pneunet_embedding_svm','validation_stacked_ensemble','cernet_resnet50','cernet_resnet50_tuned'])].sort_values('roc_auc')
    labels={'hog_linear_svm':'HOG + SVM','resnet18':'ResNet-18','efficientnet_b0':'EfficientNet-B0','pneunet_no_attention':'PneuNet-noAtt','pneunet_v2':'PneuNet-v2','pneunet_embedding_svm':'Embedding LR','validation_stacked_ensemble':'Validation-fitted stack','cernet_resnet50':'CER-Net baseline','cernet_resnet50_tuned':'CER-Net tuned'}
    gmap=grouped.set_index('model'); ci=np.array([[gmap.loc['stage2_'+m,'ci95_low'],gmap.loc['stage2_'+m,'ci95_high']] if 'stage2_'+m in gmap.index else json.loads(c) for m,c in zip(q.model,q.roc_auc_ci95)]); y=np.arange(len(q)); vals=q.roc_auc.to_numpy()
    cols=[GREEN if m=='validation_stacked_ensemble' else CORAL if 'cernet' in m else BLUE for m in q.model]
    ax.errorbar(vals,y,xerr=np.vstack([vals-ci[:,0],ci[:,1]-vals]),fmt='none',ecolor='#A2ADB6',lw=.8,capsize=2)
    ax.scatter(vals,y,c=cols,s=27,edgecolor='white',linewidth=.5,zorder=3); ax.set_yticks(y,[labels[x] for x in q.model]); ax.set_xlim(.68,.91); ax.set_xlabel('ROC-AUC (95% bootstrap CI)'); ax.axvline(.9,ls='--',lw=.7,color='#BBC2C8')
    ax.set_title('a  Independent stage-2 discrimination',loc='left',weight='bold',fontsize=9)
    # stage gap hero
    ax2=fig.add_subplot(gs[0,3:]); s1=met[(met.task=='stage1')&(met.model=='pneunet')].iloc[0]; s2=met[(met.task=='stage2')&(met.model=='validation_stacked_ensemble')].iloc[0]
    ax2.bar([0,1],[s1.roc_auc,s2.roc_auc],color=[BLUE,GREEN],width=.55); ax2.set_ylim(.7,1.015); ax2.set_xticks([0,1],['Stage 1\nPneuNet','Stage 2\nvalidation-fitted stack']); ax2.set_ylabel('ROC-AUC')
    for i,v in enumerate([s1.roc_auc,s2.roc_auc]): ax2.text(i,v+.012,f'{v:.3f}',ha='center',weight='bold',fontsize=8)
    ax2.set_title('b  Task difficulty gap',loc='left',weight='bold',fontsize=9)
    # cernet
    ax3=fig.add_subplot(gs[1,:2]); cs=met[(met.task=='stage2')&met.model.isin(['cernet_resnet50','cernet_resnet50_tuned'])].set_index('model')
    vals=[cs.loc['cernet_resnet50','roc_auc'],cs.loc['cernet_resnet50_tuned','roc_auc']]
    ax3.barh([0,1],vals,color=[CORAL2,CORAL],height=.55); ax3.set_yticks([0,1],['CER-Net baseline','Tuned CER-Net']); ax3.set_xlim(.70,.86); ax3.set_xlabel('ROC-AUC'); ax3.invert_yaxis()
    for i,v in enumerate(vals): ax3.text(v+.005,i,f'{v:.3f}',va='center',fontsize=7,weight='bold')
    ax3.annotate('+0.043',xy=(vals[1],1),xytext=(.738,.52),arrowprops=dict(arrowstyle='->',color=INK,lw=.7),fontsize=7,color=INK)
    ax3.set_title('c  CER-Net optimization',loc='left',weight='bold',fontsize=9)
    # source shift
    ax4=fig.add_subplot(gs[1,2:]); names=['pneunet_v2','validation_stacked_ensemble','cernet_resnet50_tuned']; labs=['PneuNet-v2','Validation stack','Tuned CER-Net']
    ss=src[src.model.isin(names)]; piv=ss.pivot(index='model',columns='source',values='roc_auc').reindex(names); lo=ss.pivot(index='model',columns='source',values='ci95_low').reindex(names); hi=ss.pivot(index='model',columns='source',values='ci95_high').reindex(names)
    x=np.arange(3); w=.32; ax4.bar(x-w/2,piv['train'],w,color=BLUE,yerr=np.vstack([piv['train']-lo['train'],hi['train']-piv['train']]),capsize=2); ax4.bar(x+w/2,piv['test'],w,color=CORAL,yerr=np.vstack([piv['test']-lo['test'],hi['test']-piv['test']]),capsize=2)
    ax4.set_xticks(x,labs,rotation=12,ha='right'); ax4.set_ylim(.7,1.015); ax4.set_ylabel('ROC-AUC')
    ax4.text(.02,.965,'■  Original train source',transform=ax4.transAxes,color=BLUE,fontsize=6.3,va='top')
    ax4.text(.40,.965,'■  Original test source',transform=ax4.transAxes,color=CORAL,fontsize=6.3,va='top')
    ax4.set_title('d  Source-directory stress test',loc='left',weight='bold',fontsize=9)
    fig.text(.01,.005,'Stage 2, n = 417. Error bars use 2,000 filename-group bootstrap resamples. Source subgroups: original train n = 364, original test n = 53.',fontsize=6.2,color=MUTED)
    save(fig,'figure3_main_results')

if __name__=='__main__': study_design(); architecture(); results(); print(OUT)
