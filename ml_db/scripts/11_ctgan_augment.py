# -*- coding: utf-8 -*-
"""
11_ctgan_augment.py
CTGAN 으로 train 양성 시퀀스 증강 -> LSTM/CNN+LSTM/TFT 재학습 비교.

핵심:
  - train 양성만 합성. test 는 절대 안 건드림 (현실 분포 유지).
  - 시퀀스(T x D) -> flatten(T*D) -> CTGAN -> reshape 복원.
  - 증강 전/후 PR-AUC, ROC-AUC, Gini, KS 비교표.
  - 표준화는 09 에서 이미 train 기준 -> 합성도 표준화 공간에서.

전제: 10_train_compare.py 의 모델/평가 코드 재사용.
"""

import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

NPZ = Path(__file__).resolve().parent.parent / "db" / "sequences.npz"
SEED = 42
EPOCHS = 60
BATCH = 64
LR = 1e-3
GAMMA = 2.0
PATIENCE = 10
DEVICE = "cpu"

# CTGAN 증강 배수 (양성 159 -> 159 * MULT 만큼 합성 추가)
AUG_MULT = 3
CTGAN_EPOCHS = 300

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============== Focal / 모델 (10 과 동일) ==============
class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0):
        super().__init__(); self.alpha, self.gamma = alpha, gamma
    def forward(self, logits, target):
        p = torch.sigmoid(logits)
        ce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p_t = p*target + (1-p)*(1-target)
        a_t = self.alpha*target + (1-self.alpha)*(1-target)
        return (a_t * (1-p_t)**self.gamma * ce).mean()


class LSTMNet(nn.Module):
    def __init__(self, d, h=64):
        super().__init__()
        self.lstm = nn.LSTM(d, h, batch_first=True)
        self.head = nn.Sequential(nn.Linear(h,32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32,1))
    def forward(self, x, m):
        o,_ = self.lstm(x); i=(m.sum(1)-1).long().clamp(min=0)
        return self.head(o[torch.arange(o.size(0)), i]).squeeze(-1)


class CNNLSTM(nn.Module):
    def __init__(self, d, h=64):
        super().__init__()
        self.conv=nn.Conv1d(d,32,2,padding=1); self.relu=nn.ReLU()
        self.lstm=nn.LSTM(32,h,batch_first=True)
        self.head=nn.Sequential(nn.Linear(h,32),nn.ReLU(),nn.Dropout(0.3),nn.Linear(32,1))
    def forward(self,x,m):
        c=self.relu(self.conv(x.transpose(1,2))).transpose(1,2)[:,:x.size(1),:]
        o,_=self.lstm(c); i=(m.sum(1)-1).long().clamp(min=0)
        return self.head(o[torch.arange(o.size(0)),i]).squeeze(-1)


class TFTLite(nn.Module):
    def __init__(self,d,h=64,heads=4):
        super().__init__()
        self.proj=nn.Linear(d,h)
        self.grn=nn.Sequential(nn.Linear(h,h),nn.ELU(),nn.Linear(h,h),nn.Dropout(0.2))
        self.attn=nn.MultiheadAttention(h,heads,batch_first=True)
        self.norm=nn.LayerNorm(h)
        self.head=nn.Sequential(nn.Linear(h,32),nn.ReLU(),nn.Dropout(0.3),nn.Linear(32,1))
    def forward(self,x,m):
        h=self.proj(x); h=h+self.grn(h)
        a,_=self.attn(h,h,h,key_padding_mask=(m==0))
        h=self.norm(h+a); i=(m.sum(1)-1).long().clamp(min=0)
        return self.head(h[torch.arange(h.size(0)),i]).squeeze(-1)


def ks_stat(y,s):
    fpr,tpr,_=roc_curve(y,s); return np.max(tpr-fpr)

def bootstrap_ci(y,s,fn,n=1000,seed=SEED):
    rng=np.random.default_rng(seed); idx=np.arange(len(y)); vals=[]
    for _ in range(n):
        bi=rng.choice(idx,len(idx),replace=True)
        if y[bi].sum() in (0,len(bi)): continue
        vals.append(fn(y[bi],s[bi]))
    return np.percentile(vals,2.5), np.percentile(vals,97.5)

def train_eval(Net, Xtr, Mtr, ytr, Xte, Mte, yte, alpha):
    d_in=Xtr.shape[2]
    tr=DataLoader(TensorDataset(torch.tensor(Xtr,dtype=torch.float32),
                                torch.tensor(Mtr,dtype=torch.float32),
                                torch.tensor(ytr,dtype=torch.float32)),
                  batch_size=BATCH, shuffle=True)
    vX=torch.tensor(Xte,dtype=torch.float32); vM=torch.tensor(Mte,dtype=torch.float32)
    torch.manual_seed(SEED); model=Net(d_in).to(DEVICE)
    opt=torch.optim.Adam(model.parameters(),lr=LR,weight_decay=1e-4)
    crit=FocalLoss(alpha,GAMMA); best_pr,best_state,wait=0,None,0
    for ep in range(EPOCHS):
        model.train()
        for xb,mb,yb in tr:
            opt.zero_grad(); loss=crit(model(xb,mb),yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        model.eval()
        with torch.no_grad(): s=torch.sigmoid(model(vX,vM)).numpy()
        pr=average_precision_score(yte,s)
        if pr>best_pr: best_pr,best_state,wait=pr,{k:v.clone() for k,v in model.state_dict().items()},0
        else:
            wait+=1
            if wait>=PATIENCE: break
    if best_state: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad(): s=torch.sigmoid(model(vX,vM)).numpy()
    roc=roc_auc_score(yte,s); pr=average_precision_score(yte,s)
    pr_ci=bootstrap_ci(yte,s,average_precision_score)
    return {"roc":roc,"pr":pr,"gini":2*roc-1,"ks":ks_stat(yte,s),"pr_ci":pr_ci}


def ctgan_augment(Xtr, Mtr, ytr):
    """train 양성 시퀀스를 CTGAN 으로 합성. flatten->생성->reshape."""
    from sdv.single_table import CTGANSynthesizer
    from sdv.metadata import SingleTableMetadata
    import pandas as pd

    pos_mask = ytr == 1
    Xp = Xtr[pos_mask]                 # (Npos, T, D)
    Mp = Mtr[pos_mask]
    Npos, T, D = Xp.shape
    flat = Xp.reshape(Npos, T*D)       # (Npos, T*D)
    cols = [f"f{i}" for i in range(T*D)]
    df = pd.DataFrame(flat, columns=cols)

    meta = SingleTableMetadata()
    for c in cols:
        meta.add_column(c, sdtype="numerical")

    print(f"  CTGAN 학습 시작 (양성 {Npos}개, {T*D}차원, {CTGAN_EPOCHS}ep)...")
    syn = CTGANSynthesizer(meta, epochs=CTGAN_EPOCHS, verbose=False)
    syn.fit(df)
    n_new = Npos * AUG_MULT
    gen = syn.sample(n_new).values.astype(np.float32)   # (n_new, T*D)
    Xnew = gen.reshape(n_new, T, D)

    # 마스크: 합성은 전부 풀시퀀스(유효)로 가정 -> 양성 평균 마스크 패턴 복사
    # 안전하게: 가장 흔한 마스크(전체 유효) 사용
    Mnew = np.ones((n_new, T), dtype=np.float32)
    # flag 컬럼(is_capital_impaired, 마지막 컬럼)은 0/1 로 반올림
    Xnew[:, :, -1] = (Xnew[:, :, -1] > 0.5).astype(np.float32)

    ynew = np.ones(n_new, dtype=np.float32)
    print(f"  합성 양성 {n_new}개 생성 완료")

    Xaug = np.concatenate([Xtr, Xnew], axis=0)
    Maug = np.concatenate([Mtr, Mnew], axis=0)
    yaug = np.concatenate([ytr, ynew], axis=0)
    # 셔플
    perm = np.random.default_rng(SEED).permutation(len(yaug))
    return Xaug[perm], Maug[perm], yaug[perm]


def main():
    d=np.load(NPZ, allow_pickle=True)
    Xtr,Mtr,ytr=d["X_train"],d["M_train"],d["y_train"]
    Xte,Mte,yte=d["X_test"],d["M_test"],d["y_test"]
    print(f"원본 train {Xtr.shape} 양성 {int(ytr.sum())} / test 양성 {int(yte.sum())}")

    nets=[("LSTM",LSTMNet),("CNN+LSTM",CNNLSTM),("TFT-lite",TFTLite)]

    # 1) 증강 전 (baseline)
    print("\n[증강 전]")
    before={}
    alpha=float(1-ytr.mean())
    for name,Net in nets:
        r=train_eval(Net,Xtr,Mtr,ytr,Xte,Mte,yte,alpha)
        before[name]=r
        print(f"  {name:10s} ROC {r['roc']:.4f} PR {r['pr']:.4f} "
              f"Gini {r['gini']:.4f} KS {r['ks']:.4f}")

    # 2) CTGAN 증강
    print("\n[CTGAN 증강 중]")
    Xa,Ma,ya=ctgan_augment(Xtr,Mtr,ytr)
    print(f"  증강 후 train {Xa.shape} 양성 {int(ya.sum())} "
          f"(비율 {ya.mean()*100:.1f}%)")
    alpha_a=float(1-ya.mean())

    # 3) 증강 후
    print("\n[증강 후]")
    after={}
    for name,Net in nets:
        r=train_eval(Net,Xa,Ma,ya,Xte,Mte,yte,alpha_a)
        after[name]=r
        print(f"  {name:10s} ROC {r['roc']:.4f} PR {r['pr']:.4f} "
              f"Gini {r['gini']:.4f} KS {r['ks']:.4f}")

    # 4) 비교표
    print("\n"+"="*64)
    print("CTGAN 증강 전/후 비교 (test, PR-AUC 중심)")
    print(f"{'model':10s} {'PR_전':>8s} {'PR_후':>8s} {'Δ':>7s} "
          f"{'ROC_후':>8s} {'KS_후':>7s}  PR_후 95%CI")
    for name,_ in nets:
        b,a=before[name],after[name]
        dlt=a['pr']-b['pr']
        ci=a['pr_ci']
        flag=" ↑" if dlt>0 else " ↓"
        print(f"{name:10s} {b['pr']:>8.4f} {a['pr']:>8.4f} {dlt:>+7.4f}{flag} "
              f"{a['roc']:>8.4f} {a['ks']:>7.4f}  [{ci[0]:.3f},{ci[1]:.3f}]")

    best=max(after.items(), key=lambda kv: kv[1]['pr'])
    print(f"\n>> 증강 후 베스트: {best[0]} (PR {best[1]['pr']:.4f})")
    print("\n완료. 증강이 PR 올렸으면 채택, 아니면 GraphSAGE 로.")


if __name__=="__main__":
    main()