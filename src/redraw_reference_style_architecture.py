from pathlib import Path
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Arc, Polygon
import numpy as np

ROOT=Path(os.environ.get("PNEUMONIA_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
OUT=ROOT/"reports"/"figures"/"nature_redesign"
OUT.mkdir(parents=True,exist_ok=True)
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Microsoft YaHei","DejaVu Sans"],
 "font.size":6,"svg.fonttype":"none","pdf.fonttype":42,"figure.facecolor":"white"})

INK="#24313D"; MUTED="#657482"; LINE="#8E9BA6"
BLUE="#3975AD"; BLUE_BG="#EAF2F8"; GREEN="#5D9B72"; GREEN_BG="#EAF4EC"
PURPLE="#7B65B2"; PURPLE_BG="#F0ECF8"; ORANGE="#E58B50"; ORANGE_BG="#FCEFE6"
GRAY_BG="#F5F7F8"; GOLD="#D6A644"; RED="#C96A64"

def rr(ax,x,y,w,h,fc='white',ec=LINE,lw=.65,r=.055,z=1):
    p=FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=.012,rounding_size={r}",fc=fc,ec=ec,lw=lw,zorder=z)
    ax.add_patch(p); return p
def txt(ax,x,y,s,fs=6,weight='normal',color=INK,ha='center',va='center',**kw):
    ax.text(x,y,s,fontsize=fs,weight=weight,color=color,ha=ha,va=va,linespacing=1.12,zorder=5,**kw)
def arrow(ax,a,b,color=INK,lw=.9):
    ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='-|>',lw=lw,color=color,mutation_scale=7),zorder=8)
def header(ax,x,w,title,fc,ec):
    rr(ax,x,.28,w,7.36,'white',ec,.85,.07)
    ax.add_patch(FancyBboxPatch((x,.28+6.91),w,.45,boxstyle='round,pad=.012,rounding_size=.07',fc=fc,ec=ec,lw=.85,zorder=2))
    ax.add_patch(Rectangle((x,.28+6.91),w,.18,fc=fc,ec='none',zorder=3))
    txt(ax,x+w/2,7.41,title,7.0,'bold',ec)
def card(ax,x,y,w,h,title,subtitle='',fc='white',ec=LINE,icon=None):
    rr(ax,x,y,w,h,fc,ec,.55,.045)
    left=x+.13 if icon else x+w/2
    if icon=='image':
        for k in range(3): ax.add_patch(Rectangle((x+.10+k*.035,y+h/2-.19-k*.025),.31,.38,fc=['#D6E0E7','#E5EBEF','#F3F5F7'][k],ec=ec,lw=.4,zorder=3))
        left=x+.57
    elif icon=='cnn':
        for k in range(3): ax.add_patch(Polygon([[x+.09+k*.08,y+h/2-.20],[x+.31+k*.08,y+h/2-.15],[x+.31+k*.08,y+h/2+.17],[x+.09+k*.08,y+h/2+.12]],closed=True,fc=[BLUE_BG,"#C8DCEC","#A8C8DF"][k],ec=BLUE,lw=.35,zorder=3))
        left=x+.66
    elif icon=='grid':
        for i in range(4):
            for j in range(3): ax.add_patch(Rectangle((x+.09+i*.075,y+h/2-.17+j*.11),.06,.085,fc=BLUE_BG if (i+j)%2 else '#A9C7DD',ec='white',lw=.2,zorder=3))
        left=x+.62
    elif icon=='gear':
        ax.add_patch(Circle((x+.25,y+h/2),.15,fc=GRAY_BG,ec=INK,lw=.55,zorder=3)); ax.add_patch(Circle((x+.25,y+h/2),.045,fc='white',ec=INK,lw=.4,zorder=4)); left=x+.55
    txt(ax,left,y+h*.62,title,6.1,'bold',ec if ec!=LINE else INK,ha='left' if icon else 'center')
    if subtitle: txt(ax,left,y+h*.31,subtitle,5.2,'normal',MUTED,ha='left' if icon else 'center')

def draw():
    fig,ax=plt.subplots(figsize=(7.2,5.45)); ax.set_xlim(0,11.5); ax.set_ylim(0,8.25); ax.axis('off')
    gap=.13; x1=.08; w1=2.45; x2=x1+w1+gap; w2=2.30; x3=x2+w2+gap; w3=3.53; x4=x3+w3+gap; w4=2.18
    header(ax,x1,w1,'1. Data and filename-group split',BLUE_BG,BLUE)
    header(ax,x2,w2,'2. Stage 1 screening',GREEN_BG,GREEN)
    header(ax,x3,w3,'3. Group-OOF fixed stacking',PURPLE_BG,PURPLE)
    header(ax,x4,w4,'4. Prediction and audit',ORANGE_BG,ORANGE)

    # panel 1
    txt(ax,x1+.16,6.82,'A. Input cohort',5.8,'bold',BLUE,ha='left')
    card(ax,x1+.13,5.90,w1-.26,.72,'Pediatric chest X-rays','Kaggle v2 · 5,856 images',BLUE_BG,BLUE,'image')
    card(ax,x1+.13,4.95,w1-.26,.72,'Three labels','Normal · bacterial · viral','white',BLUE,'grid')
    ax.plot([x1+.14,x1+w1-.14],[4.72,4.72],ls=(0,(2,2)),color='#B9C5CE',lw=.55)
    txt(ax,x1+.16,4.48,'B. Leakage-aware partition',5.8,'bold',BLUE,ha='left')
    card(ax,x1+.13,3.55,w1-.26,.72,'Filename-group derivation','Subtype-namespaced IDs','white',BLUE,'gear')
    # split matrix
    rr(ax,x1+.13,2.22,w1-.26,1.05,GRAY_BG,BLUE,.55,.04)
    rows=[('Train','4,699','80%'),('Validation','582','10%'),('Test','575','10%')]
    for i,(a,b,c) in enumerate(rows):
        y=2.96-i*.29; ax.add_patch(Rectangle((x1+.25,y-.09,.20,.16),.20,.16,fc=[BLUE_BG,GREEN_BG,ORANGE_BG][i],ec=[BLUE,GREEN,ORANGE][i],lw=.35))
        txt(ax,x1+.54,y,a,5.2,'bold',INK,ha='left'); txt(ax,x1+1.60,y,b,5.2,color=INK); txt(ax,x1+2.10,y,c,5.0,color=MUTED)
    rr(ax,x1+.34,.78,w1-.68,.88,GREEN_BG,GREEN,.55,.05); txt(ax,x1+w1/2,1.34,'Filename-group-disjoint',5.8,'bold',GREEN); txt(ax,x1+w1/2,1.02,'Test set remains locked',5.0,color=MUTED)

    # panel 2
    txt(ax,x2+.16,6.82,'Feature extraction',5.8,'bold',GREEN,ha='left')
    card(ax,x2+.13,5.80,w2-.26,.82,'Image preprocessing','RGB · resize · normalization',GREEN_BG,GREEN,'image')
    card(ax,x2+.13,4.64,w2-.26,.88,'PneuNet','EfficientNet-B0 backbone',GREEN_BG,GREEN,'cnn')
    arrow(ax,(x2+w2/2,5.80),(x2+w2/2,5.52),GREEN)
    # feature matrix
    rr(ax,x2+.20,3.62,w2-.40,.70,'white',GREEN,.55,.04)
    for i in range(10):
        for j in range(3): ax.add_patch(Rectangle((x2+.35+i*.15,3.82+j*.10),.13,.08,fc=plt.cm.Blues(.2+.06*((i+j)%8)),ec='white',lw=.15))
    txt(ax,x2+w2/2,3.68,'Final feature tensor (1,280 channels)',5.1,color=MUTED)
    arrow(ax,(x2+w2/2,4.64),(x2+w2/2,4.32),GREEN)
    card(ax,x2+.13,2.52,w2-.26,.78,'GeM pooling + dropout','Compact discriminative embedding','white',GREEN)
    arrow(ax,(x2+w2/2,3.62),(x2+w2/2,3.30),GREEN)
    card(ax,x2+.13,1.48,w2-.26,.72,'Stage 1 head','< 0.5: Normal  |  >= 0.5: Stage 2','white',GREEN)
    arrow(ax,(x2+w2/2,2.52),(x2+w2/2,2.20),GREEN)
    rr(ax,x2+.31,.63,w2-.62,.52,GREEN_BG,GREEN,.55,.05); txt(ax,x2+w2/2,.89,'ROC-AUC 0.995 · n = 575',5.5,'bold',GREEN)

    # panel 3
    txt(ax,x3+.16,6.82,'Frozen embedding learners',5.8,'bold',PURPLE,ha='left')
    cards=[('ResNet-18','group-fold logistic'),('EfficientNet-B0','group-fold logistic'),('DenseNet-121','group-fold logistic'),('5-fold OOF','no group crosses folds')]
    for i,(a,b) in enumerate(cards):
        yy=6.12-i*.67; card(ax,x3+.16,yy,w3*.52,.52,a,b,'white',PURPLE,'cnn' if i<3 else 'grid')
    # probability matrix
    rr(ax,x3+2.20,4.08,1.15,2.56,PURPLE_BG,PURPLE,.55,.04)
    txt(ax,x3+2.775,6.36,'Development OOF',5.4,'bold',PURPLE); txt(ax,x3+2.775,6.12,'probabilities',5.1,color=MUTED)
    # Decorative probability-matrix motif only; values are fixed and are not experimental measurements.
    mat=np.array([[.18,.32,.71,.46],[.25,.62,.43,.30],[.74,.29,.55,.81],[.48,.39,.68,.22],
                  [.79,.57,.36,.65],[.33,.76,.41,.58],[.83,.44,.69,.27],[.21,.67,.31,.52]])
    for i in range(8):
        for j in range(4): ax.add_patch(Rectangle((x3+2.37+j*.20,4.58+(7-i)*.16),.18,.14,fc=plt.cm.Purples(.15+.65*mat[i,j]),ec='white',lw=.15))
    txt(ax,x3+2.775,4.32,'N cases x 3 learners',4.8,color=MUTED)
    for yy in [6.38,5.71,5.04,4.37]: arrow(ax,(x3+2.00,yy),(x3+2.18,yy),PURPLE,.65)
    ax.plot([x3+.15,x3+w3-.15],[3.78,3.78],ls=(0,(2,2)),color='#C5BADF',lw=.55)
    txt(ax,x3+.16,3.53,'Fixed-transform meta-learning',5.8,'bold',PURPLE,ha='left')
    card(ax,x3+.16,2.52,1.48,.72,'Meta-learner','L2 logistic regression','white',PURPLE,'gear')
    card(ax,x3+1.84,2.52,1.51,.72,'Fixed transform','Dev. empirical CDF','white',PURPLE,'grid')
    arrow(ax,(x3+2.78,4.08),(x3+2.52,3.24),PURPLE)
    rr(ax,x3+.16,1.40,w3-.32,.80,PURPLE_BG,PURPLE,.55,.05)
    txt(ax,x3+w3/2,1.91,'Independent test probability',5.8,'bold',PURPLE); txt(ax,x3+w3/2,1.62,'No test-cohort ranking',5.1,color=MUTED)
    rr(ax,x3+.40,.60,w3-.80,.50,GREEN_BG,GREEN,.55,.05); txt(ax,x3+w3/2,.85,'Test cases transformed independently',5.25,'bold',GREEN)

    # panel 4
    txt(ax,x4+.16,6.82,'Locked-test output',5.8,'bold',ORANGE,ha='left')
    rr(ax,x4+.20,5.68,w4-.40,.95,'white',ORANGE,.55,.05)
    # gauge
    cx=x4+w4/2; cy=5.98
    ax.add_patch(Arc((cx,cy),.90,.62,theta1=0,theta2=180,color=ORANGE,lw=1.2));
    for a in np.linspace(0,np.pi,5): ax.plot([cx+.38*np.cos(a),cx+.46*np.cos(a)],[cy+.27*np.sin(a),cy+.32*np.sin(a)],color=LINE,lw=.45)
    ax.plot([cx,cx+.31*np.cos(.72*np.pi)],[cy,cy+.24*np.sin(.72*np.pi)],color=RED,lw=1.1); ax.add_patch(Circle((cx,cy),.035,fc=RED,ec='none'))
    txt(ax,cx,6.45,'Viral probability',5.6,'bold',ORANGE); txt(ax,cx,5.79,'P(viral)',4.8,color=MUTED)
    arrow(ax,(cx,5.68),(cx,5.39),ORANGE)
    card(ax,x4+.20,4.48,w4-.40,.72,'OOF-selected threshold','Frozen before locked test','white',ORANGE)
    arrow(ax,(cx,4.48),(cx,4.19),ORANGE)
    rr(ax,x4+.20,3.25,w4-.40,.74,'white',ORANGE,.55,.05); txt(ax,cx,3.76,'Thresholded class',5.6,'bold',ORANGE); txt(ax,cx,3.46,'Bacterial  |  Viral',5.2,color=INK)
    arrow(ax,(cx,3.25),(cx,2.96),ORANGE)
    rr(ax,x4+.20,2.02,w4-.40,.75,ORANGE_BG,ORANGE,.55,.05); txt(ax,cx,2.53,'CER-Net-only audit',5.4,'bold',ORANGE); txt(ax,cx,2.24,'Reject if max probability < 0.57',4.7,color=MUTED)
    arrow(ax,(cx,2.02),(cx,1.73),ORANGE)
    rr(ax,x4+.20,.72,w4-.40,.82,'white',ORANGE,.55,.05)
    ax.add_patch(Circle((x4+.48,1.14),.14,fc=BLUE_BG,ec=BLUE,lw=.55)); txt(ax,x4+.48,1.14,'B',5.3,'bold',BLUE)
    ax.add_patch(Circle((x4+.88,1.14),.14,fc=ORANGE_BG,ec=ORANGE,lw=.55)); txt(ax,x4+.88,1.14,'V',5.3,'bold',ORANGE)
    txt(ax,x4+1.48,1.27,'Predicted class',4.7,'bold',INK); txt(ax,x4+1.48,1.03,'stack: no rejection',4.4,color=MUTED)

    # main arrows between stages
    arrow(ax,(x1+w1,3.80),(x2,3.80),INK,1.0); arrow(ax,(x2+w2,3.80),(x3,3.80),INK,1.0); arrow(ax,(x3+w3,3.80),(x4,3.80),INK,1.0)

    # legend
    rr(ax,.08,-.38,11.32,.52,'white',LINE,.55,.045)
    txt(ax,.22,-.12,'Legend',5.5,'bold',INK,ha='left')
    items=[(BLUE_BG,BLUE,'Input / split'),(GREEN_BG,GREEN,'Stage 1'),(PURPLE_BG,PURPLE,'Stage 2 / stacking'),(ORANGE_BG,ORANGE,'Prediction'),(GRAY_BG,LINE,'Audit / metadata')]
    xx=1.10
    for fc,ec,label in items:
        rr(ax,xx,-.25,.28,.26,fc,ec,.45,.025); txt(ax,xx+.38,-.12,label,4.9,color=INK,ha='left'); xx+=2.02
    fig.subplots_adjust(left=.008,right=.992,top=.99,bottom=.075)
    base=OUT/'figure2_model_architecture_reference_style'
    fig.savefig(base.with_suffix('.png'),dpi=600,bbox_inches='tight',facecolor='white')
    fig.savefig(base.with_suffix('.svg'),bbox_inches='tight',facecolor='white')
    fig.savefig(base.with_suffix('.pdf'),bbox_inches='tight',facecolor='white')
    fig.savefig(base.with_suffix('.tiff'),dpi=600,bbox_inches='tight',facecolor='white')
    plt.close(fig); print(base)

if __name__=='__main__': draw()
