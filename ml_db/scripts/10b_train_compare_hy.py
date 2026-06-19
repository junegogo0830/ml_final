# -*- coding: utf-8 -*-
"""
10b_train_compare_hy.py
반기 시퀀스(sequences_hy.npz)로 LSTM/CNN+LSTM/TFT 비교 + z_score baseline.
손실 focal, 평가 ROC/PR/Gini/KS + Bootstrap CI. test 현실분포 유지.
"""
import numpy as np
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

NPZ=Path(__file__).resolve().parent.parent / "db" / "sequences_hy.npz"
SEED=42; EPOCHS=60; BATCH=64; LR=1e-3; GAMMA=2.0; PATIENCE=10
torch.manual_seed(SEED); np.random.seed(SEED)

class FocalLoss(nn.Module):
    def __init__(s,a,g=2.0): super().__init__(); s.a,s.g=a,g
    def forward(s,l,t):
        p=torch.sigmoid(l); ce=nn.functional.binary_cross_entropy_with_logits(l,t,reduction="none")
        pt=p*t+(1-p)*(1-t); at=s.a*t+(1-s.a)*(1-t); return (at*(1-pt)**s.g*ce).mean()

class LSTMNet(nn.Module):
    def __init__(s,d,h=64): super().__init__(); s.lstm=nn.LSTM(d,h,batch_first=True); s.head=nn.Sequential(nn.Linear(h,32),nn.ReLU(),nn.Dropout(0.3),nn.Linear(32,1))
    def forward(s,x,m): o,_=s.lstm(x); i=(m.sum(1)-1).long().clamp(min=0); return s.head(o[torch.arange(o.size(0)),i]).squeeze(-1)
class CNNLSTM(nn.Module):
    def __init__(s,d,h=64): super().__init__(); s.conv=nn.Conv1d(d,32,2,padding=1); s.relu=nn.ReLU(); s.lstm=nn.LSTM(32,h,batch_first=True); s.head=nn.Sequential(nn.Linear(h,32),nn.ReLU(),nn.Dropout(0.3),nn.Linear(32,1))
    def forward(s,x,m): c=s.relu(s.conv(x.transpose(1,2))).transpose(1,2)[:,:x.size(1),:]; o,_=s.lstm(c); i=(m.sum(1)-1).long().clamp(min=0); return s.head(o[torch.arange(o.size(0)),i]).squeeze(-1)
class TFTLite(nn.Module):
    def __init__(s,d,h=64,heads=4): super().__init__(); s.proj=nn.Linear(d,h); s.grn=nn.Sequential(nn.Linear(h,h),nn.ELU(),nn.Linear(h,h),nn.Dropout(0.2)); s.attn=nn.MultiheadAttention(h,heads,batch_first=True); s.norm=nn.LayerNorm(h); s.head=nn.Sequential(nn.Linear(h,32),nn.ReLU(),nn.Dropout(0.3),nn.Linear(32,1))
    def forward(s,x,m): h=s.proj(x); h=h+s.grn(h); a,_=s.attn(h,h,h,key_padding_mask=(m==0)); h=s.norm(h+a); i=(m.sum(1)-1).long().clamp(min=0); return s.head(h[torch.arange(h.size(0)),i]).squeeze(-1)

def ks(y,sc): fpr,tpr,_=roc_curve(y,sc); return float(np.max(tpr-fpr))
def boot(y,sc,fn,n=1000):
    rng=np.random.default_rng(SEED); idx=np.arange(len(y)); v=[]
    for _ in range(n):
        b=rng.choice(idx,len(idx),replace=True)
        if y[b].sum() in (0,len(b)): continue
        v.append(fn(y[b],sc[b]))
    return np.percentile(v,2.5),np.percentile(v,97.5)

def train_eval(Net,Xtr,Mtr,ytr,Xte,Mte,yte,alpha):
    tr=DataLoader(TensorDataset(torch.tensor(Xtr),torch.tensor(Mtr),torch.tensor(ytr)),batch_size=BATCH,shuffle=True)
    vX=torch.tensor(Xte); vM=torch.tensor(Mte)
    torch.manual_seed(SEED); m=Net(Xtr.shape[2]); opt=torch.optim.Adam(m.parameters(),lr=LR,weight_decay=1e-4); crit=FocalLoss(alpha,GAMMA)
    bp,bs,w=0,None,0
    for ep in range(EPOCHS):
        m.train()
        for xb,mb,yb in tr:
            opt.zero_grad(); loss=crit(m(xb,mb),yb); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        m.eval()
        with torch.no_grad(): s=torch.sigmoid(m(vX,vM)).numpy()
        pr=average_precision_score(yte,s)
        if pr>bp: bp,bs,w=pr,{k:v.clone() for k,v in m.state_dict().items()},0
        else:
            w+=1
            if w>=PATIENCE: break
    if bs: m.load_state_dict(bs)
    m.eval()
    with torch.no_grad(): s=torch.sigmoid(m(vX,vM)).numpy()
    roc=roc_auc_score(yte,s); pr=average_precision_score(yte,s)
    return dict(roc=roc,pr=pr,gini=2*roc-1,ks=ks(yte,s),pr_ci=boot(yte,s,average_precision_score))

def main():
    d=np.load(NPZ,allow_pickle=True)
    Xtr,Mtr,ytr=d["X_train"],d["M_train"],d["y_train"]
    Xte,Mte,yte=d["X_test"],d["M_test"],d["y_test"]; feat=list(d["feature_names"])
    print(f"train {Xtr.shape} / test {Xte.shape} 양성 train {ytr.mean()*100:.1f}% test {yte.mean()*100:.1f}%")
    alpha=float(1-ytr.mean())
    # baseline z_score
    zi=feat.index("z_score"); li=(Mte.sum(1)-1).astype(int)
    zl=Xte[np.arange(len(Xte)),li,zi]
    roc=roc_auc_score(yte,-zl); pr=average_precision_score(yte,-zl)
    print(f"\n[z_score baseline] ROC {roc:.4f} PR {pr:.4f} Gini {2*roc-1:.4f} KS {ks(yte,-zl):.4f}")
    base_pr=pr
    res=[("z_score",dict(roc=roc,pr=pr,gini=2*roc-1,ks=ks(yte,-zl)))]
    for name,Net in [("LSTM",LSTMNet),("CNN+LSTM",CNNLSTM),("TFT-lite",TFTLite)]:
        r=train_eval(Net,Xtr,Mtr,ytr,Xte,Mte,yte,alpha)
        ci=r["pr_ci"]
        print(f"[{name}] ROC {r['roc']:.4f} PR {r['pr']:.4f} Gini {r['gini']:.4f} KS {r['ks']:.4f} PR_CI[{ci[0]:.3f},{ci[1]:.3f}]")
        res.append((name,r))
    print("\n"+"="*60+"\n최종 (반기 TTM, test 현실분포)")
    print(f"{'model':14s}{'ROC':>9s}{'PR':>9s}{'Gini':>9s}{'KS':>9s}")
    for nm,r in res: print(f"{nm:14s}{r['roc']:>9.4f}{r['pr']:>9.4f}{r['gini']:>9.4f}{r['ks']:>9.4f}")
    best=max(res[1:],key=lambda x:x[1]['pr'])
    print(f"\n>> 베스트(PR): {best[0]} PR {best[1]['pr']:.4f} (baseline {base_pr:.4f})")

if __name__=="__main__":
    main()
