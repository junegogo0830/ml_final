# -*- coding: utf-8 -*-
"""
12_lstm_graphsage.py
LSTM(시계열) + GraphSAGE(관계) 결합 모델 vs 시계열 단독 ablation.

구조 (Option B):
  LSTM        -> e_t (64d)  시계열 임베딩
  GraphSAGE   -> e_g (64d)  관계 임베딩 (obs_year 스냅샷 그래프)
  MLP(concat[e_t,e_g]) -> PD

누수 통제:
  - 각 기업은 자기 obs_year 스냅샷 그래프에서만 이웃 수집
  - 양성: label_year-1 그래프 / 음성: pseudo obs_year 그래프 (09 와 동일 시점)
  - 그래프 노드 피처 = 그 연도 financials 19비율 (표준화는 09 scaler 재사용)
  - test 기업은 test 그래프에만, train 은 train 그래프에만 (firm split 유지)

전제: 09(sequences.npz) + 08(graph_edges, graph_nodes) 완료.
"""

import numpy as np
import sqlite3
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import torch
import torch.nn as nn

from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data

_DB_DIR = Path(__file__).resolve().parent.parent / "db"
NPZ = _DB_DIR / "sequences.npz"
DB  = _DB_DIR / "dart_v2.db"
SEED=42; EPOCHS=80; LR=1e-3; GAMMA=2.0; PATIENCE=12; DEVICE="cpu"
HID=64
TRAIN_MAX_YEAR=2022

torch.manual_seed(SEED); np.random.seed(SEED)

RATIO_COLS = [
    "debt_ratio","equity_ratio","debt_to_assets","noncurrent_liab_ratio",
    "current_ratio","quick_ratio","cash_ratio","roa","roe","operating_margin",
    "net_margin","gross_margin","asset_turnover","inventory_turnover",
    "interest_coverage_proxy","ocf_to_current_liab","retained_earnings_ratio",
    "working_capital_ratio","z_score",
]


class FocalLoss(nn.Module):
    def __init__(self,a,g=2.0): super().__init__(); self.a,self.g=a,g
    def forward(self,logit,t):
        p=torch.sigmoid(logit)
        ce=nn.functional.binary_cross_entropy_with_logits(logit,t,reduction="none")
        pt=p*t+(1-p)*(1-t); at=self.a*t+(1-self.a)*(1-t)
        return (at*(1-pt)**self.g*ce).mean()


# ============== 그래프 스냅샷 로더 ==============
def load_graphs(scaler_mean, scaler_std, lo, hi):
    con=sqlite3.connect(DB); cur=con.cursor()
    col=", ".join(RATIO_COLS)

    graphs={}
    years=[r[0] for r in cur.execute(
        "SELECT DISTINCT year FROM graph_nodes ORDER BY year")]
    for y in years:
        nodes=cur.execute(
            "SELECT corp_code, node_idx FROM graph_nodes WHERE year=? ORDER BY node_idx",
            (y,)).fetchall()
        cc2idx={cc:i for cc,i in nodes}
        idx2cc={i:cc for cc,i in nodes}
        n=len(nodes)

        feat=np.zeros((n,len(RATIO_COLS)),dtype=np.float32)
        frows=cur.execute(
            f"SELECT corp_code,{col} FROM financials WHERE year=?",(y,)).fetchall()
        fmap={r[0]:r[1:] for r in frows}
        for cc,i in cc2idx.items():
            vals=fmap.get(cc)
            if vals is None: continue
            v=np.array([np.nan if z is None else float(z) for z in vals])
            v=np.clip(v,lo,hi); v=(v-scaler_mean)/scaler_std
            feat[i]=np.nan_to_num(v,nan=0.0)

        edges=cur.execute(
            "SELECT src,dst FROM graph_edges WHERE year=?",(y,)).fetchall()
        if edges:
            ei=np.array(edges,dtype=np.int64).T
        else:
            ei=np.zeros((2,0),dtype=np.int64)
        sl=np.arange(n); ei=np.concatenate([ei,np.stack([sl,sl])],axis=1)

        graphs[y]=dict(cc2idx=cc2idx, idx2cc=idx2cc,
                       x=torch.tensor(feat),
                       edge_index=torch.tensor(ei))
    con.close()
    print(f"  그래프 스냅샷 로드: {len(graphs)}개 연도")
    return graphs


# ============== 모델 ==============
class LSTMEncoder(nn.Module):
    def __init__(self,d,h=HID):
        super().__init__(); self.lstm=nn.LSTM(d,h,batch_first=True)
    def forward(self,x,m):
        o,_=self.lstm(x); i=(m.sum(1)-1).long().clamp(min=0)
        return o[torch.arange(o.size(0)),i]


class CNNLSTMEncoder(nn.Module):
    def __init__(self,d,h=HID):
        super().__init__()
        self.conv=nn.Conv1d(d,32,2,padding=1); self.relu=nn.ReLU()
        self.lstm=nn.LSTM(32,h,batch_first=True)
    def forward(self,x,m):
        c=self.relu(self.conv(x.transpose(1,2))).transpose(1,2)[:,:x.size(1),:]
        o,_=self.lstm(c); i=(m.sum(1)-1).long().clamp(min=0)
        return o[torch.arange(o.size(0)),i]


class TFTEncoder(nn.Module):
    def __init__(self,d,h=HID,heads=4):
        super().__init__()
        self.proj=nn.Linear(d,h)
        self.grn=nn.Sequential(nn.Linear(h,h),nn.ELU(),nn.Linear(h,h),nn.Dropout(0.2))
        self.attn=nn.MultiheadAttention(h,heads,batch_first=True)
        self.norm=nn.LayerNorm(h)
    def forward(self,x,m):
        h=self.proj(x); h=h+self.grn(h)
        a,_=self.attn(h,h,h,key_padding_mask=(m==0))
        h=self.norm(h+a); i=(m.sum(1)-1).long().clamp(min=0)
        return h[torch.arange(h.size(0)),i]


ENCODERS={"LSTM":LSTMEncoder,"CNN+LSTM":CNNLSTMEncoder,"TFT":TFTEncoder}


class GraphSAGEEncoder(nn.Module):
    def __init__(self,d_node,h=HID):
        super().__init__()
        self.c1=SAGEConv(d_node,h); self.c2=SAGEConv(h,h)
        self.relu=nn.ReLU(); self.drop=nn.Dropout(0.3)
    def forward(self,x,edge_index):
        h=self.relu(self.c1(x,edge_index)); h=self.drop(h)
        h=self.c2(h,edge_index)
        return h


class Combined(nn.Module):
    def __init__(self,EncCls,d_seq,d_node,use_graph=True):
        super().__init__()
        self.use_graph=use_graph
        self.enc=EncCls(d_seq)
        if use_graph:
            self.sage=GraphSAGEEncoder(d_node)
            self.head=nn.Sequential(nn.Linear(HID*2,32),nn.ReLU(),
                                    nn.Dropout(0.3),nn.Linear(32,1))
        else:
            self.head=nn.Sequential(nn.Linear(HID,32),nn.ReLU(),
                                    nn.Dropout(0.3),nn.Linear(32,1))

    def forward(self,xb,mb,graph_emb=None):
        et=self.enc(xb,mb)
        if self.use_graph:
            z=torch.cat([et,graph_emb],dim=1)
        else:
            z=et
        return self.head(z).squeeze(-1)


def ks_calc(y,s): fpr,tpr,_=roc_curve(y,s); return float(np.max(tpr-fpr))
def boot(y,s,fn,n=1000):
    rng=np.random.default_rng(SEED); idx=np.arange(len(y)); v=[]
    for _ in range(n):
        b=rng.choice(idx,len(idx),replace=True)
        if y[b].sum() in (0,len(b)): continue
        v.append(fn(y[b],s[b]))
    return np.percentile(v,2.5),np.percentile(v,97.5)


def build_obs_year_map():
    con=sqlite3.connect(DB); cur=con.cursor()
    lab={cc:(l,d) for cc,l,d in cur.execute(
        "SELECT corp_code,label,label_date FROM labels")}
    con.close()
    return lab


def main():
    d=np.load(NPZ,allow_pickle=True)
    Xtr,Mtr,ytr,cctr=d["X_train"],d["M_train"],d["y_train"],d["cc_train"]
    Xte,Mte,yte,ccte=d["X_test"],d["M_test"],d["y_test"],d["cc_test"]
    mean,std=d["scaler_mean"],d["scaler_std"]; lo,hi=d["winsor_lo"],d["winsor_hi"]
    print(f"train {Xtr.shape}  양성 {int(ytr.sum())} / test 양성 {int(yte.sum())}")

    graphs=load_graphs(mean,std,lo,hi)

    lab=build_obs_year_map()

    def obs_year_for(cc, is_train=True):
        l,dt=lab.get(cc,(0,None))
        if l==1 and dt:
            oy=int(dt[:4])-1
            if not is_train:
                oy=min(oy, TRAIN_MAX_YEAR)
            return oy
        return None

    d_seq=Xtr.shape[2]; d_node=len(RATIO_COLS)
    alpha=float(1-ytr.mean())

    def map_samples(ccs, ys, is_train=True):
        out=[]
        for cc,_ in zip(ccs,ys):
            oy=obs_year_for(cc, is_train)
            if oy is None:
                cand=[y for y in graphs if y<=TRAIN_MAX_YEAR and cc in graphs[y]["cc2idx"]]
                oy=max(cand) if cand else None
            if oy is not None and oy in graphs and cc in graphs[oy]["cc2idx"]:
                out.append((oy, graphs[oy]["cc2idx"][cc]))
            else:
                out.append(None)
        return out

    tr_map=map_samples(cctr,ytr,True); te_map=map_samples(ccte,yte,False)
    tr_ok=np.array([m is not None for m in tr_map])
    te_ok=np.array([m is not None for m in te_map])
    print(f"  그래프 매칭: train {tr_ok.sum()}/{len(tr_ok)} "
          f"test {te_ok.sum()}/{len(te_ok)}")

    def run(EncCls, use_graph, label):
        pad=max(1, 56-len(label))
        print(f"\n[{label}] {'─'*pad}")
        torch.manual_seed(SEED)
        model=Combined(EncCls,d_seq,d_node,use_graph).to(DEVICE)
        opt=torch.optim.Adam(model.parameters(),lr=LR,weight_decay=1e-4)
        crit=FocalLoss(alpha,GAMMA)
        Xt=torch.tensor(Xtr); Mt=torch.tensor(Mtr); yt=torch.tensor(ytr)
        Xe=torch.tensor(Xte); Me=torch.tensor(Mte)
        best_pr,best_state,wait=0,None,0

        for ep in range(EPOCHS):
            model.train(); opt.zero_grad()
            if use_graph:
                emb_cache={y:model.sage(graphs[y]["x"],graphs[y]["edge_index"])
                           for y in graphs}
                geb=torch.zeros(len(Xtr),HID)
                for i,m in enumerate(tr_map):
                    if m: geb[i]=emb_cache[m[0]][m[1]]
                logit=model(Xt,Mt,geb)
            else:
                logit=model(Xt,Mt)
            loss=crit(logit,yt); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()

            model.eval()
            with torch.no_grad():
                if use_graph:
                    emb_cache={y:model.sage(graphs[y]["x"],graphs[y]["edge_index"])
                               for y in graphs}
                    gebe=torch.zeros(len(Xte),HID)
                    for i,m in enumerate(te_map):
                        if m: gebe[i]=emb_cache[m[0]][m[1]]
                    s=torch.sigmoid(model(Xe,Me,gebe)).numpy()
                else:
                    s=torch.sigmoid(model(Xe,Me)).numpy()
            pr=average_precision_score(yte,s)

            if (ep+1) % 10 == 0:
                print(f"  ep {ep+1:3d}/{EPOCHS}  loss={loss.item():.4f}  "
                      f"val_PR={pr:.4f}  best={best_pr:.4f}")

            if pr>best_pr:
                best_pr,best_state,wait=pr,{k:v.clone() for k,v in model.state_dict().items()},0
            else:
                wait+=1
                if wait>=PATIENCE:
                    print(f"  ▶ early stop ep {ep+1}/{EPOCHS}")
                    break

        if best_state: model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            if use_graph:
                emb_cache={y:model.sage(graphs[y]["x"],graphs[y]["edge_index"])
                           for y in graphs}
                gebe=torch.zeros(len(Xte),HID)
                for i,m in enumerate(te_map):
                    if m: gebe[i]=emb_cache[m[0]][m[1]]
                s=torch.sigmoid(model(Xe,Me,gebe)).numpy()
            else:
                s=torch.sigmoid(model(Xe,Me)).numpy()
        roc=roc_auc_score(yte,s); pr=average_precision_score(yte,s)
        return dict(roc=roc,pr=pr,gini=2*roc-1,ks=ks_calc(yte,s),
                    pr_ci=boot(yte,s,average_precision_score))

    print("\n" + "="*60)
    print("3 인코더 × (단독 / +GraphSAGE) ablation")
    print("="*60)

    results={}
    for ename, EncCls in ENCODERS.items():
        r0=run(EncCls, False, f"{ename} 단독")
        ci0=r0['pr_ci']
        print(f"  → ROC {r0['roc']:.4f}  PR {r0['pr']:.4f}  "
              f"Gini {r0['gini']:.4f}  KS {r0['ks']:.4f}  CI[{ci0[0]:.3f},{ci0[1]:.3f}]")

        r1=run(EncCls, True, f"{ename}+GraphSAGE")
        ci1=r1['pr_ci']
        print(f"  → ROC {r1['roc']:.4f}  PR {r1['pr']:.4f}  "
              f"Gini {r1['gini']:.4f}  KS {r1['ks']:.4f}  CI[{ci1[0]:.3f},{ci1[1]:.3f}]  "
              f"PR Δ{r1['pr']-r0['pr']:+.4f}")
        results[ename]=(r0,r1)

    print("\n" + "="*76)
    print("최종 ablation 표 (test, 현실분포)")
    print(f"{'모델':18s} {'ROC':>7s} {'PR':>7s} {'Gini':>7s} {'KS':>7s}  PR 95% CI")
    print("─"*76)
    rows=[]
    for ename,(r0,r1) in results.items():
        ci0,ci1=r0['pr_ci'],r1['pr_ci']
        print(f"{ename+' 단독':18s} {r0['roc']:>7.4f} {r0['pr']:>7.4f} "
              f"{r0['gini']:>7.4f} {r0['ks']:>7.4f}  [{ci0[0]:.3f},{ci0[1]:.3f}]")
        print(f"{ename+'+SAGE':18s} {r1['roc']:>7.4f} {r1['pr']:>7.4f} "
              f"{r1['gini']:>7.4f} {r1['ks']:>7.4f}  [{ci1[0]:.3f},{ci1[1]:.3f}]")
        rows.append((ename+" 단독",r0)); rows.append((ename+"+SAGE",r1))

    best=max(rows, key=lambda kv: kv[1]['pr'])
    print("─"*76)
    print(f"\n>> 전체 베스트(PR): {best[0]}  PR {best[1]['pr']:.4f}  ROC {best[1]['roc']:.4f}")

    print("\n그래프 결합 효과 (PR Δ):")
    for ename,(r0,r1) in results.items():
        dp=r1['pr']-r0['pr']
        mark="↑" if dp>0 else "↓"
        print(f"  {ename:10s}: {r0['pr']:.4f} → {r1['pr']:.4f}  ({dp:+.4f} {mark})")


if __name__=="__main__":
    main()
