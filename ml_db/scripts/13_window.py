# -*- coding: utf-8 -*-
"""
13_graphsage_aug.py
GraphSAGE + 증강 기법 5-seed 실험 비교

증강 전략:
  Gaussian Noise  - 양성 시계열에 σ=0.05 노이즈 (유효 시점만), n_copies=2
  Mixup           - 양성 쌍 λ~Beta(0.4,0.4) 보간, 우세 샘플 마스크·그래프 상속
  Noise+Mixup     - Noise 증강 후 Mixup 추가

실험 방법:
  시드마다 증강 데이터와 모델 초기화를 함께 변경 (SEEDS 5개)
  → 결과: per-seed 결과 + 5-seed mean±std

누수 통제 (12 동일):
  - 양성(test): min(label_year-1, TRAIN_MAX_YEAR) 그래프
  - 음성: TRAIN_MAX_YEAR 이하 최신 그래프
  - 증강은 train 양성에만, 증강 샘플은 원본 (obs_year, node_idx) 상속

전제: 09(sequences.npz), 08(graph_nodes/graph_edges) 완료.
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

_DB_DIR = Path(__file__).resolve().parent.parent / "db"
NPZ = _DB_DIR / "sequences.npz"
DB  = _DB_DIR / "dart_v2.db"
EPOCHS=80; LR=1e-3; GAMMA=2.0; PATIENCE=12; DEVICE="cpu"; HID=64
TRAIN_MAX_YEAR=2022
SEEDS=[42, 123, 456, 789, 2024]
AUG_NAMES=["Base","Noise","Mixup","Noise+Mixup"]

RATIO_COLS=["debt_ratio","equity_ratio","debt_to_assets","noncurrent_liab_ratio",
"current_ratio","quick_ratio","cash_ratio","roa","roe","operating_margin",
"net_margin","gross_margin","asset_turnover","inventory_turnover",
"interest_coverage_proxy","ocf_to_current_liab","retained_earnings_ratio",
"working_capital_ratio","z_score"]


# ============== 모델 ==============
class FocalLoss(nn.Module):
    def __init__(self,a,g=2.0): super().__init__(); self.a,self.g=a,g
    def forward(self,l,t):
        p=torch.sigmoid(l); ce=nn.functional.binary_cross_entropy_with_logits(l,t,reduction="none")
        pt=p*t+(1-p)*(1-t); at=self.a*t+(1-self.a)*(1-t)
        return (at*(1-pt)**self.g*ce).mean()

class LSTMEncoder(nn.Module):
    def __init__(self,d,h=HID): super().__init__(); self.lstm=nn.LSTM(d,h,batch_first=True)
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
        super().__init__(); self.proj=nn.Linear(d,h)
        self.grn=nn.Sequential(nn.Linear(h,h),nn.ELU(),nn.Linear(h,h),nn.Dropout(0.2))
        self.attn=nn.MultiheadAttention(h,heads,batch_first=True); self.norm=nn.LayerNorm(h)
    def forward(self,x,m):
        h=self.proj(x); h=h+self.grn(h)
        a,_=self.attn(h,h,h,key_padding_mask=(m==0)); h=self.norm(h+a)
        i=(m.sum(1)-1).long().clamp(min=0); return h[torch.arange(h.size(0)),i]

ENCODERS={"LSTM":LSTMEncoder,"CNN+LSTM":CNNLSTMEncoder,"TFT":TFTEncoder}

class SAGEEnc(nn.Module):
    def __init__(self,d,h=HID):
        super().__init__(); self.c1=SAGEConv(d,h); self.c2=SAGEConv(h,h)
        self.relu=nn.ReLU(); self.drop=nn.Dropout(0.3)
    def forward(self,x,ei):
        h=self.relu(self.c1(x,ei)); h=self.drop(h); return self.c2(h,ei)

class Combined(nn.Module):
    def __init__(self,EncCls,d_seq,d_node):
        super().__init__(); self.enc=EncCls(d_seq); self.sage=SAGEEnc(d_node)
        self.head=nn.Sequential(nn.Linear(HID*2,32),nn.ReLU(),nn.Dropout(0.3),nn.Linear(32,1))
    def forward(self,xb,mb,geb):
        et=self.enc(xb,mb); return self.head(torch.cat([et,geb],1)).squeeze(-1)


def ks_calc(y,s): fpr,tpr,_=roc_curve(y,s); return float(np.max(tpr-fpr))


def load_graphs(mean,std,lo,hi):
    con=sqlite3.connect(DB); cur=con.cursor(); col=", ".join(RATIO_COLS); graphs={}
    years=[r[0] for r in cur.execute("SELECT DISTINCT year FROM graph_nodes ORDER BY year")]
    for y in years:
        nodes=cur.execute("SELECT corp_code,node_idx FROM graph_nodes WHERE year=? ORDER BY node_idx",(y,)).fetchall()
        cc2idx={cc:i for cc,i in nodes}; n=len(nodes)
        feat=np.zeros((n,len(RATIO_COLS)),dtype=np.float32)
        fmap={r[0]:r[1:] for r in cur.execute(f"SELECT corp_code,{col} FROM financials WHERE year=?",(y,)).fetchall()}
        for cc,i in cc2idx.items():
            vals=fmap.get(cc)
            if vals is None: continue
            v=np.array([np.nan if z is None else float(z) for z in vals])
            v=np.clip(v,lo,hi); v=(v-mean)/std; feat[i]=np.nan_to_num(v,nan=0.0)
        edges=cur.execute("SELECT src,dst FROM graph_edges WHERE year=?",(y,)).fetchall()
        ei=np.array(edges,dtype=np.int64).T if edges else np.zeros((2,0),dtype=np.int64)
        sl=np.arange(n); ei=np.concatenate([ei,np.stack([sl,sl])],axis=1)
        graphs[y]=dict(cc2idx=cc2idx,x=torch.tensor(feat),edge_index=torch.tensor(ei))
    con.close(); return graphs


# ============== 증강 함수 ==============
def gaussian_noise_augment(X, M, y, maps, sigma=0.05, n_copies=2, seed=42):
    """양성 샘플에 Gaussian Noise 추가 (유효 시점에만, 패딩 위치는 0 유지)."""
    rng=np.random.default_rng(seed)
    pos_idx=[i for i,yi in enumerate(y) if yi==1 and maps[i] is not None]
    addX,addM,addy,addmap=[],[],[],[]
    for _ in range(n_copies):
        for i in pos_idx:
            noise=rng.normal(0,sigma,X[i].shape).astype(np.float32)
            nx=X[i]+noise*M[i,:,None]
            addX.append(nx); addM.append(M[i])
            addy.append(1.0); addmap.append(maps[i])
    if not addX:
        return X,M,y,maps
    Xa=np.concatenate([X,np.array(addX,dtype=np.float32)],0)
    Ma=np.concatenate([M,np.array(addM,dtype=np.float32)],0)
    ya=np.concatenate([y,np.array(addy,dtype=np.float32)],0)
    mapa=list(maps)+addmap
    perm=np.random.default_rng(seed+1).permutation(len(ya))
    return Xa[perm],Ma[perm],ya[perm],[mapa[k] for k in perm]


def mixup_augment(X, M, y, maps, alpha=0.4, n_copies=1, seed=42):
    """양성 쌍 Mixup 보간: λ~Beta(α,α), 우세 샘플(λ≥0.5)의 마스크·그래프 상속."""
    rng=np.random.default_rng(seed+10)
    pos_idx=[i for i,yi in enumerate(y) if yi==1 and maps[i] is not None]
    if len(pos_idx)<2:
        return X,M,y,maps
    addX,addM,addy,addmap=[],[],[],[]
    for _ in range(n_copies):
        shuffled=rng.permutation(pos_idx).tolist()
        for i,j in zip(pos_idx,shuffled):
            if i==j: continue
            lam=float(rng.beta(alpha,alpha))
            if lam<0.5:
                lam,i,j=1-lam,j,i
            nx=(lam*X[i]+(1-lam)*X[j]).astype(np.float32)
            addX.append(nx); addM.append(M[i])
            addy.append(1.0); addmap.append(maps[i])
    if not addX:
        return X,M,y,maps
    Xa=np.concatenate([X,np.array(addX,dtype=np.float32)],0)
    Ma=np.concatenate([M,np.array(addM,dtype=np.float32)],0)
    ya=np.concatenate([y,np.array(addy,dtype=np.float32)],0)
    mapa=list(maps)+addmap
    perm=np.random.default_rng(seed+11).permutation(len(ya))
    return Xa[perm],Ma[perm],ya[perm],[mapa[k] for k in perm]


def main():
    d=np.load(NPZ,allow_pickle=True)
    Xtr,Mtr,ytr,cctr=d["X_train"],d["M_train"],d["y_train"],d["cc_train"]
    Xte,Mte,yte,ccte=d["X_test"],d["M_test"],d["y_test"],d["cc_test"]
    mean,std=d["scaler_mean"],d["scaler_std"]; lo,hi=d["winsor_lo"],d["winsor_hi"]
    print(f"train {Xtr.shape}  양성 {int(ytr.sum())} / test 양성 {int(yte.sum())}")

    graphs=load_graphs(mean,std,lo,hi)
    avail_years=sorted(graphs.keys())
    print(f"  그래프 연도: {avail_years[0]}~{avail_years[-1]} ({len(avail_years)}개)")

    con=sqlite3.connect(DB)
    lab={cc:(l,dt) for cc,l,dt in con.execute("SELECT corp_code,label,label_date FROM labels")}
    con.close()

    def resolve_obs(cc, is_train):
        l,dt=lab.get(cc,(0,None))
        if l==1 and dt:
            oy=int(dt[:4])-1
            if not is_train: oy=min(oy,TRAIN_MAX_YEAR)
            return oy if oy in graphs and cc in graphs[oy]["cc2idx"] else None
        cand=[y for y in avail_years if y<=TRAIN_MAX_YEAR and cc in graphs[y]["cc2idx"]]
        return max(cand) if cand else None

    def make_map(ccs, is_train):
        out=[]
        for cc in ccs:
            oy=resolve_obs(cc,is_train)
            out.append((oy,graphs[oy]["cc2idx"][cc]) if oy is not None else None)
        return out

    tr_map_orig=make_map(cctr,True)
    te_map=make_map(ccte,False)
    tr_ok=sum(1 for m in tr_map_orig if m)
    te_ok=sum(1 for m in te_map if m)
    print(f"  그래프 매칭: train {tr_ok}/{len(tr_map_orig)}  test {te_ok}/{len(te_map)}")

    n_pos=int(ytr.sum())
    n_gpos=sum(1 for i,m in enumerate(tr_map_orig) if m and ytr[i]==1)
    print(f"\n  증강 규모 (그래프 매칭 양성 {n_gpos}개 기준, 시드별 미세 변동):")
    print(f"    Base:        {len(ytr):5d} 샘플  ({n_pos} 양성)")
    print(f"    Noise(×2):  ~{len(ytr)+2*n_gpos:5d} 샘플  (양성 +{2*n_gpos})")
    print(f"    Mixup(×1):  ~{len(ytr)+n_gpos:5d} 샘플  (양성 ~+{n_gpos})")
    print(f"    Noise+Mixup:~{len(ytr)+3*n_gpos:5d} 샘플  (양성 ~+{3*n_gpos})")

    d_seq=Xtr.shape[2]; d_node=len(RATIO_COLS)

    def build_augmented(aug_name, seed):
        """시드별 증강 데이터셋 생성."""
        if aug_name=="Base":
            return Xtr,Mtr,ytr,tr_map_orig
        elif aug_name=="Noise":
            return gaussian_noise_augment(Xtr,Mtr,ytr,tr_map_orig,sigma=0.05,n_copies=2,seed=seed)
        elif aug_name=="Mixup":
            return mixup_augment(Xtr,Mtr,ytr,tr_map_orig,alpha=0.4,n_copies=1,seed=seed)
        else:  # Noise+Mixup
            Xn,Mn,yn,tn=gaussian_noise_augment(Xtr,Mtr,ytr,tr_map_orig,sigma=0.05,n_copies=2,seed=seed)
            return mixup_augment(Xn,Mn,yn,tn,alpha=0.4,n_copies=1,seed=seed)

    def run_single(EncCls, Xt, Mt, yt, tmap, seed):
        """단일 시드 학습·평가. (result_dict, stopped_epoch) 반환."""
        torch.manual_seed(seed)
        model=Combined(EncCls,d_seq,d_node).to(DEVICE)
        opt=torch.optim.Adam(model.parameters(),lr=LR,weight_decay=1e-4)
        crit=FocalLoss(float(1-yt.mean()),GAMMA)
        Xtt=torch.tensor(Xt); Mtt=torch.tensor(Mt); ytt=torch.tensor(yt)
        Xee=torch.tensor(Xte); Mee=torch.tensor(Mte)

        def gemb(mp,nrow):
            cache={y:model.sage(graphs[y]["x"],graphs[y]["edge_index"]) for y in graphs}
            g=torch.zeros(nrow,HID)
            for i,m in enumerate(mp):
                if m: g[i]=cache[m[0]][m[1]]
            return g

        best_pr,best_state,wait=0,None,0
        stopped_at=EPOCHS
        for ep in range(EPOCHS):
            model.train(); opt.zero_grad()
            loss=crit(model(Xtt,Mtt,gemb(tmap,len(Xt))),ytt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()

            model.eval()
            with torch.no_grad():
                s=torch.sigmoid(model(Xee,Mee,gemb(te_map,len(Xte)))).numpy()
            pr=average_precision_score(yte,s)

            if pr>best_pr:
                best_pr,best_state,wait=pr,{k:v.clone() for k,v in model.state_dict().items()},0
            else:
                wait+=1
                if wait>=PATIENCE:
                    stopped_at=ep+1; break

        if best_state: model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            s=torch.sigmoid(model(Xee,Mee,gemb(te_map,len(Xte)))).numpy()
        roc=roc_auc_score(yte,s); pr=average_precision_score(yte,s)
        return dict(roc=roc,pr=pr,gini=2*roc-1,ks=ks_calc(yte,s)), stopped_at

    def run_multi(EncCls, aug_name, label):
        """5-seed 반복 실험. per-seed 결과 출력 후 mean±std 출력."""
        pad=max(1,56-len(label))
        print(f"\n[{label}] {'─'*pad}")
        rocs,prs,ginis,kss=[],[],[],[]

        for k,seed in enumerate(SEEDS):
            Xa,Ma,ya,tmap_a=build_augmented(aug_name,seed)
            r,stopped=run_single(EncCls,Xa,Ma,ya,tmap_a,seed)
            if stopped<EPOCHS:
                stop_tag=f"stop ep{stopped:3d}/{EPOCHS}"
            else:
                stop_tag=f"ep {EPOCHS}/{EPOCHS} (done)"
            print(f"  seed {k+1}/{len(SEEDS)} ({seed:5d})  {stop_tag}  "
                  f"ROC {r['roc']:.4f}  PR {r['pr']:.4f}  KS {r['ks']:.4f}")
            rocs.append(r['roc']); prs.append(r['pr'])
            ginis.append(r['gini']); kss.append(r['ks'])

        rm,rs=np.mean(rocs),np.std(rocs)
        pm,ps=np.mean(prs),np.std(prs)
        gm,gs=np.mean(ginis),np.std(ginis)
        km,kstd=np.mean(kss),np.std(kss)
        print(f"  ◆ 5-seed   "
              f"ROC {rm:.4f}±{rs:.4f}  PR {pm:.4f}±{ps:.4f}  "
              f"Gini {gm:.4f}±{gs:.4f}  KS {km:.4f}±{kstd:.4f}")
        return dict(roc_mean=rm,roc_std=rs,pr_mean=pm,pr_std=ps,
                    gini_mean=gm,gini_std=gs,ks_mean=km,ks_std=kstd,
                    pr_all=prs,roc_all=rocs)

    print("\n" + "="*64)
    print("GraphSAGE + 증강 기법  5-seed 비교  (3 인코더 × 4 증강)")
    print("="*64)

    all_rows=[]
    for ename,EncCls in ENCODERS.items():
        print(f"\n{'━'*64}")
        print(f"  인코더: {ename}+SAGE")
        print(f"{'━'*64}")
        base_pr=None
        for aug_name in AUG_NAMES:
            label=f"{ename}+SAGE | {aug_name}"
            r=run_multi(EncCls,aug_name,label)
            if base_pr is None:
                base_pr=r['pr_mean']
            else:
                delta=r['pr_mean']-base_pr
                mark="↑" if delta>0 else "↓"
                print(f"          vs Base: PR Δ{delta:+.4f} {mark}")
            all_rows.append((label,r))

    # ===== 최종 결과표 =====
    print("\n" + "="*84)
    print("최종 결과표 (test, 5-seed mean±std)")
    print(f"{'모델':26s} {'ROC mean±std':>17s} {'PR mean±std':>16s} {'Gini mean±std':>17s} {'KS mean±std':>15s}")
    print("─"*84)
    for name,r in all_rows:
        print(f"{name:26s} "
              f"{r['roc_mean']:.4f}±{r['roc_std']:.4f}   "
              f"{r['pr_mean']:.4f}±{r['pr_std']:.4f}   "
              f"{r['gini_mean']:.4f}±{r['gini_std']:.4f}   "
              f"{r['ks_mean']:.4f}±{r['ks_std']:.4f}")
    best=max(all_rows,key=lambda x:x[1]['pr_mean'])
    print("─"*84)
    print(f"\n>> 베스트(PR mean): {best[0]}")
    print(f"   PR {best[1]['pr_mean']:.4f}±{best[1]['pr_std']:.4f}  "
          f"ROC {best[1]['roc_mean']:.4f}±{best[1]['roc_std']:.4f}  "
          f"KS {best[1]['ks_mean']:.4f}±{best[1]['ks_std']:.4f}")


if __name__=="__main__":
    main()
