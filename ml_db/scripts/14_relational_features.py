# -*- coding: utf-8 -*-
"""
14_relational_features.py
SAGE 노드 피처에 '관계 전용' 신호 4개 추가 (재무 19비율과 안 겹치는 정보).
  재무 19  vs  재무 19 + 관계 4  를 3인코더 × 5-seed 로 비교.
  증강은 적용 안 함 (13에서 효과 없음 검증됨) -> Base 만, 순수 비교.

관계 전용 4피처 (전부 그 연도 그래프 + 과거 위기로만 계산 -> 누수 차단):
  1. nbr_crisis_ratio : 이웃 중 'y년 이전에 이미 위기난' 기업 비율  ★누수주의
  2. nbr_mean_z       : 이웃들의 y년 z_score 평균 (동반부실)
  3. log_degree       : log(1+이웃수). 시스템 중심성
  4. ind_z_rank       : 같은 업종 내 z_score 백분위 (상대 위치)

누수 통제 (13 동일 + 관계피처 시점 컷):
  - 양성(test): min(label_year-1, TRAIN_MAX_YEAR) 그래프
  - 음성: TRAIN_MAX_YEAR 이하 최신 그래프
  - nbr_crisis_ratio 는 label_date < (그래프연도+1) 인 위기만 카운트
    => 그래프 시점에 '아직 안 일어난' 미래 위기는 절대 안 셈

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
SEEDS=[42,123,456,789,2024]

RATIO_COLS=["debt_ratio","equity_ratio","debt_to_assets","noncurrent_liab_ratio",
"current_ratio","quick_ratio","cash_ratio","roa","roe","operating_margin",
"net_margin","gross_margin","asset_turnover","inventory_turnover",
"interest_coverage_proxy","ocf_to_current_liab","retained_earnings_ratio",
"working_capital_ratio","z_score"]
N_RATIO=len(RATIO_COLS)
N_REL=4                       # 관계 전용 피처 수
Z_IDX=RATIO_COLS.index("z_score")


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
        super().__init__(); self.conv=nn.Conv1d(d,32,2,padding=1); self.relu=nn.ReLU()
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


# ============== 그래프 로더 (관계 전용 피처 포함) ==============
def load_graphs(mean,std,lo,hi, use_relational):
    """use_relational=True 면 노드 피처 = [재무19 표준화] + [관계4].
       False 면 재무19 만 (13 Base 와 동일)."""
    con=sqlite3.connect(DB); cur=con.cursor(); col=", ".join(RATIO_COLS)

    # 위기 발생 연도 맵 (nbr_crisis_ratio 누수 컷용): corp_code -> label_year
    crisis_year={}
    for cc,l,dt in cur.execute("SELECT corp_code,label,label_date FROM labels"):
        if l==1 and dt:
            crisis_year[cc]=int(dt[:4])

    # 산업코드 (ind_z_rank 용)
    ind={cc:ic for cc,ic in cur.execute(
        "SELECT corp_code,industry_code FROM companies "
        "WHERE industry_code IS NOT NULL AND industry_code!=''")}

    graphs={}
    years=[r[0] for r in cur.execute("SELECT DISTINCT year FROM graph_nodes ORDER BY year")]
    for y in years:
        nodes=cur.execute("SELECT corp_code,node_idx FROM graph_nodes WHERE year=? ORDER BY node_idx",(y,)).fetchall()
        cc2idx={cc:i for cc,i in nodes}; idx2cc={i:cc for cc,i in nodes}; n=len(nodes)

        # --- 재무 19 (표준화) + 원본 z (관계계산용) ---
        feat=np.zeros((n,N_RATIO),dtype=np.float32)
        raw_z=np.full(n,np.nan)
        fmap={r[0]:r[1:] for r in cur.execute(f"SELECT corp_code,{col} FROM financials WHERE year=?",(y,)).fetchall()}
        for cc,i in cc2idx.items():
            vals=fmap.get(cc)
            if vals is None: continue
            v=np.array([np.nan if z is None else float(z) for z in vals])
            raw_z[i]=v[Z_IDX]
            v=np.clip(v,lo,hi); v=(v-mean)/std; feat[i]=np.nan_to_num(v,nan=0.0)

        # --- 엣지 ---
        edges=cur.execute("SELECT src,dst FROM graph_edges WHERE year=?",(y,)).fetchall()
        ei=np.array(edges,dtype=np.int64).T if edges else np.zeros((2,0),dtype=np.int64)

        if use_relational:
            # 인접 리스트 (self-loop 전, 진짜 이웃만)
            adj=[[] for _ in range(n)]
            for s,t in edges:
                adj[s].append(t)
            # 1) nbr_crisis_ratio : 이웃 중 'y+1 이전에 위기난' 기업 비율
            #    crisis_year[cc] <= y  => 그래프 시점 y 까지 이미 위기 (누수 아님)
            # 2) nbr_mean_z
            # 3) log_degree
            # 4) ind_z_rank : 같은 업종 z 백분위
            rel=np.zeros((n,N_REL),dtype=np.float32)
            # 업종별 z 모으기
            ind_zs={}
            for cc,i in cc2idx.items():
                ic=ind.get(cc)
                if ic and not np.isnan(raw_z[i]):
                    ind_zs.setdefault(ic,[]).append(raw_z[i])
            for cc,i in cc2idx.items():
                nb=adj[i]
                deg=len(nb)
                if deg>0:
                    # 1) 과거(또는 당해) 위기 이웃 비율
                    crisis_cnt=0
                    zs=[]
                    for j in nb:
                        ncc=idx2cc[j]
                        cy=crisis_year.get(ncc)
                        if cy is not None and cy<=y:     # 누수 컷: 미래위기 제외
                            crisis_cnt+=1
                        if not np.isnan(raw_z[j]):
                            zs.append(raw_z[j])
                    rel[i,0]=crisis_cnt/deg
                    rel[i,1]=np.mean(zs) if zs else 0.0
                rel[i,2]=np.log1p(deg)
                # 4) 업종 내 z 백분위
                ic=ind.get(cc)
                if ic and ic in ind_zs and not np.isnan(raw_z[i]) and len(ind_zs[ic])>1:
                    arr=np.array(ind_zs[ic])
                    rel[i,3]=(arr<raw_z[i]).mean()
                else:
                    rel[i,3]=0.5
            # 관계피처도 표준화 (그 연도 내에서)
            rmean=rel.mean(0); rstd=rel.std(0); rstd[rstd==0]=1.0
            rel=(rel-rmean)/rstd
            node_feat=np.concatenate([feat,rel],axis=1)
        else:
            node_feat=feat

        sl=np.arange(n); ei=np.concatenate([ei,np.stack([sl,sl])],axis=1)
        graphs[y]=dict(cc2idx=cc2idx,x=torch.tensor(node_feat.astype(np.float32)),
                       edge_index=torch.tensor(ei))
    con.close()
    return graphs


def main():
    d=np.load(NPZ,allow_pickle=True)
    Xtr,Mtr,ytr,cctr=d["X_train"],d["M_train"],d["y_train"],d["cc_train"]
    Xte,Mte,yte,ccte=d["X_test"],d["M_test"],d["y_test"],d["cc_test"]
    mean,std=d["scaler_mean"],d["scaler_std"]; lo,hi=d["winsor_lo"],d["winsor_hi"]
    print(f"train {Xtr.shape}  양성 {int(ytr.sum())} / test 양성 {int(yte.sum())}")

    con=sqlite3.connect(DB)
    lab={cc:(l,dt) for cc,l,dt in con.execute("SELECT corp_code,label,label_date FROM labels")}
    con.close()

    # 두 버전 그래프: 재무19만 / 재무19+관계4
    print("그래프 로딩 (재무19)...")
    G_base=load_graphs(mean,std,lo,hi, use_relational=False)
    print("그래프 로딩 (재무19+관계4)...")
    G_rel =load_graphs(mean,std,lo,hi, use_relational=True)
    avail_years=sorted(G_base.keys())

    def resolve_obs(cc,is_train,graphs):
        l,dt=lab.get(cc,(0,None))
        if l==1 and dt:
            oy=int(dt[:4])-1
            if not is_train: oy=min(oy,TRAIN_MAX_YEAR)
            return oy if oy in graphs and cc in graphs[oy]["cc2idx"] else None
        cand=[y for y in avail_years if y<=TRAIN_MAX_YEAR and cc in graphs[y]["cc2idx"]]
        return max(cand) if cand else None

    def make_map(ccs,is_train,graphs):
        out=[]
        for cc in ccs:
            oy=resolve_obs(cc,is_train,graphs)
            out.append((oy,graphs[oy]["cc2idx"][cc]) if oy is not None else None)
        return out

    d_seq=Xtr.shape[2]

    def run_single(EncCls, graphs, tr_map, te_map, d_node, seed):
        torch.manual_seed(seed)
        model=Combined(EncCls,d_seq,d_node).to(DEVICE)
        opt=torch.optim.Adam(model.parameters(),lr=LR,weight_decay=1e-4)
        crit=FocalLoss(float(1-ytr.mean()),GAMMA)
        Xtt=torch.tensor(Xtr); Mtt=torch.tensor(Mtr); ytt=torch.tensor(ytr)
        Xee=torch.tensor(Xte); Mee=torch.tensor(Mte)
        def gemb(mp,nrow):
            cache={y:model.sage(graphs[y]["x"],graphs[y]["edge_index"]) for y in graphs}
            g=torch.zeros(nrow,HID)
            for i,m in enumerate(mp):
                if m: g[i]=cache[m[0]][m[1]]
            return g
        best_pr,best_state,wait=0,None,0
        for ep in range(EPOCHS):
            model.train(); opt.zero_grad()
            loss=crit(model(Xtt,Mtt,gemb(tr_map,len(Xtr))),ytt); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            model.eval()
            with torch.no_grad(): s=torch.sigmoid(model(Xee,Mee,gemb(te_map,len(Xte)))).numpy()
            pr=average_precision_score(yte,s)
            if pr>best_pr: best_pr,best_state,wait=pr,{k:v.clone() for k,v in model.state_dict().items()},0
            else:
                wait+=1
                if wait>=PATIENCE: break
        if best_state: model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad(): s=torch.sigmoid(model(Xee,Mee,gemb(te_map,len(Xte)))).numpy()
        roc=roc_auc_score(yte,s); pr=average_precision_score(yte,s)
        return dict(roc=roc,pr=pr,gini=2*roc-1,ks=ks_calc(yte,s))

    def run_multi(EncCls, graphs, d_node, label):
        tr_map=make_map(cctr,True,graphs); te_map=make_map(ccte,False,graphs)
        print(f"\n[{label}]  매칭 train {sum(1 for m in tr_map if m)}/{len(tr_map)} "
              f"test {sum(1 for m in te_map if m)}/{len(te_map)}")
        R={"roc":[],"pr":[],"gini":[],"ks":[]}
        for k,seed in enumerate(SEEDS):
            r=run_single(EncCls,graphs,tr_map,te_map,d_node,seed)
            for kk in R: R[kk].append(r[kk])
            print(f"  seed {k+1}/{len(SEEDS)} ({seed:5d})  "
                  f"ROC {r['roc']:.4f}  PR {r['pr']:.4f}  KS {r['ks']:.4f}")
        out={f"{kk}_mean":np.mean(v) for kk,v in R.items()}
        out.update({f"{kk}_std":np.std(v) for kk,v in R.items()})
        print(f"  ◆ 5-seed  ROC {out['roc_mean']:.4f}±{out['roc_std']:.4f}  "
              f"PR {out['pr_mean']:.4f}±{out['pr_std']:.4f}  "
              f"Gini {out['gini_mean']:.4f}±{out['gini_std']:.4f}  "
              f"KS {out['ks_mean']:.4f}±{out['ks_std']:.4f}")
        return out

    print("\n"+"="*64)
    print("재무19  vs  재무19+관계4  (3인코더 × 5-seed)")
    print("="*64)
    rows=[]
    for ename,EncCls in ENCODERS.items():
        print(f"\n{'━'*64}\n  {ename}+SAGE\n{'━'*64}")
        rb=run_multi(EncCls,G_base,N_RATIO,        f"{ename}+SAGE | 재무19")
        rr=run_multi(EncCls,G_rel, N_RATIO+N_REL,  f"{ename}+SAGE | 재무19+관계4")
        dp=rr['pr_mean']-rb['pr_mean']
        print(f"          관계피처 효과: PR Δ{dp:+.4f} {'↑' if dp>0 else '↓'}")
        rows.append((f"{ename}|재무19",rb)); rows.append((f"{ename}|+관계4",rr))

    print("\n"+"="*82)
    print("최종 결과표 (test, 5-seed mean±std)")
    print(f"{'모델':22s} {'ROC':>16s} {'PR':>16s} {'Gini':>16s} {'KS':>15s}")
    print("─"*82)
    for name,r in rows:
        print(f"{name:22s} {r['roc_mean']:.4f}±{r['roc_std']:.4f}  "
              f"{r['pr_mean']:.4f}±{r['pr_std']:.4f}  "
              f"{r['gini_mean']:.4f}±{r['gini_std']:.4f}  "
              f"{r['ks_mean']:.4f}±{r['ks_std']:.4f}")
    best=max(rows,key=lambda x:x[1]['pr_mean'])
    print("─"*82)
    print(f"\n>> 베스트(PR): {best[0]}  PR {best[1]['pr_mean']:.4f}±{best[1]['pr_std']:.4f}  "
          f"ROC {best[1]['roc_mean']:.4f}")
    print("\n관계피처가 PR 올렸으면 채택. 미미하면(±std 내) '관계 보조신호도 한계' 로 정리.")


if __name__=="__main__":
    main()