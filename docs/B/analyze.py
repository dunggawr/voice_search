import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
W="/home/claude/work/"
m=pd.read_csv("/root/.claude/uploads/2f4e7de6-8a24-5741-ac2f-b9ea86cbb9e1/7513dfb1-metadata_master.csv")
f=pd.read_csv(W+"features_prosodic.csv"); d=m.merge(f,on="file_id")
F=[c for c in f.columns if c!="file_id"]
BG,TX,SEC,GR="#fcfcfb","#0b0b0b","#52514e","#e3e2de"
COL={"Nam":"#2a78d6","UK":"#eb6834","US":"#1baf7a"}
plt.rcParams.update({"figure.facecolor":BG,"axes.facecolor":BG,"axes.edgecolor":GR,"axes.labelcolor":SEC,"xtick.color":SEC,"ytick.color":SEC,"text.color":TX,"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.color":GR,"grid.linewidth":.6,"font.size":10})

# 1. thong ke mo ta
desc=d[F].describe().T[["mean","std","min","25%","50%","75%","max"]].round(3)
desc.to_csv(W+"table_descriptive.csv")

# 2. F ratio (inter/intra) - speaker co >=2 file
def fratio(df):
    g=df.groupby("speaker_id").filter(lambda x:len(x)>=2); rows=[]
    for c in F:
        grp=[x[c].dropna().values for _,x in g.groupby("speaker_id")]
        Fv,p=stats.f_oneway(*grp)
        mu=g.groupby("speaker_id")[c].mean(); inter=mu.var(ddof=1)
        intra=g.groupby("speaker_id")[c].var(ddof=1).mean()
        ss_b=sum(len(x)*(x.mean()-g[c].mean())**2 for x in grp); ss_t=((g[c]-g[c].mean())**2).sum()
        rows.append((c,inter,intra,inter/intra,Fv,p,ss_b/ss_t))
    return pd.DataFrame(rows,columns=["feature","inter_var","intra_var","inter_intra","F_anova","p_value","eta2"]).sort_values("F_anova",ascending=False)
R=fratio(d); R.round(4).to_csv(W+"table_fratio.csv",index=False)
R2=fratio(d[d.duration_sec>=2]); R2.round(4).to_csv(W+"table_fratio_dur_ge2s.csv",index=False)
print(R.round(3).to_string()); print("\n>=2s subset, n files",(d.duration_sec>=2).sum()); print(R2.round(3).to_string())

# 3. theo nhom ngon ngu/vung
grp=d.groupby("region_accent")[F].mean().round(2); print("\n",grp.to_string()); grp.to_csv(W+"table_by_region.csv")
# F0 outliers
print("\nf0_max>250:",(d.f0_max>250).sum()," f0_mean>180:",(d.f0_mean>180).sum()," f0_mean<85:",(d.f0_mean<85).sum())

# --- bieu do
reg=d.drop_duplicates("speaker_id").set_index("speaker_id").region_accent
cnt=d.speaker_id.value_counts(); sel=[]
for k in ["Nam","UK","US"]:
    ids=[s for s in cnt.index if cnt[s]>=8 and reg[s]==k]
    med=d[d.speaker_id.isin(ids)].groupby("speaker_id").f0_mean.median().sort_values()
    sel+=list(med.index[np.linspace(0,len(med)-1,6).astype(int)])
print("boxplot speakers",len(sel),{k:sum(reg[s]==k for s in sel) for k in COL})
fig,ax=plt.subplots(figsize=(9,4.2))
bp=ax.boxplot([d[d.speaker_id==s].f0_mean for s in sel],patch_artist=True,widths=.55,showfliers=False,medianprops={"color":TX})
for b,s in zip(bp["boxes"],sel): b.set_facecolor(COL[reg[s]]); b.set_edgecolor(BG); b.set_alpha(.9)
ax.set_xticks(range(1,len(sel)+1)); ax.set_xticklabels(sel,rotation=60,ha="right",fontsize=7); ax.set_ylabel("f0_mean (Hz)")
ax.set_title("F0 trung bình theo speaker (18 speaker: 6 mỗi nhóm, số file ≥ 8)",loc="left",fontsize=11)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=c,label=k) for k,c in COL.items()],frameon=False,ncol=3,loc="upper left"); ax.grid(axis="x",visible=False)
fig.tight_layout(); fig.savefig(W+"figures/boxplot_f0_speaker.png",dpi=160); plt.close()

fig,ax=plt.subplots(figsize=(8,3.8)); bins=np.linspace(d.f0_mean.min(),d.f0_mean.max(),35)
for k,c in COL.items(): ax.hist(d[d.region_accent==k].f0_mean,bins=bins,color=c,alpha=.75,label=f"{k} (n={(d.region_accent==k).sum()})",edgecolor=BG,linewidth=.8)
ax.set_xlabel("f0_mean (Hz)"); ax.set_ylabel("Số file"); ax.set_title("Phân bố F0 trung bình theo nhóm Nam / UK / US",loc="left",fontsize=11); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(W+"figures/hist_f0_region.png",dpi=160); plt.close()

fig,ax=plt.subplots(figsize=(7.5,4)); r=R.sort_values("eta2")
ax.barh(r.feature,r.eta2,color="#2a78d6",height=.6); ax.set_xlabel("η² (tỉ lệ phương sai do khác speaker)"); ax.grid(axis="y",visible=False)
for i,(v,fv) in enumerate(zip(r.eta2,r.F_anova)): ax.text(v+.01,i,f"{v:.2f}  (F={fv:.1f})",va="center",fontsize=8,color=SEC)
ax.set_xlim(0,max(r.eta2)*1.3); ax.set_title("Khả năng phân biệt speaker của từng đặc trưng",loc="left",fontsize=11)
fig.tight_layout(); fig.savefig(W+"figures/bar_fratio.png",dpi=160); plt.close()

fig,ax=plt.subplots(figsize=(7,5.8)); cm=d[F].corr()
im=ax.imshow(cm,cmap="RdBu_r",vmin=-1,vmax=1); ax.grid(False)
ax.set_xticks(range(len(F))); ax.set_xticklabels(F,rotation=60,ha="right",fontsize=8); ax.set_yticks(range(len(F))); ax.set_yticklabels(F,fontsize=8)
for i in range(len(F)):
    for j in range(len(F)): ax.text(j,i,f"{cm.iloc[i,j]:.1f}",ha="center",va="center",fontsize=6.5,color="#fff" if abs(cm.iloc[i,j])>.6 else TX)
fig.colorbar(im,shrink=.8); ax.set_title("Tương quan giữa các đặc trưng",loc="left",fontsize=11)
fig.tight_layout(); fig.savefig(W+"figures/heatmap_corr.png",dpi=160); plt.close()
cm.round(2).to_csv(W+"table_corr.csv")
