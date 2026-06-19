# -*- coding: utf-8 -*-
"""
12b_14b_graphsage_hy.py
반기 파이프라인 GraphSAGE 통합 실험:
  PART A (=12b): 3인코더 × (단독 / +SAGE) ablation, 5-seed
  PART B (=13b): 베스트 인코더에 증강(Noise/Mixup) 5-seed   [MODE='aug']
  PART C (=14b): 관계전용 노드피처 재무19 vs 재무19+관계4    [MODE='rel']
MODE 로 선택 실행 (기본 'ablation'). 한 파일로 12b/13b/14b 다 커버.

==== 누수 통제 (공시일 기준) ====
  - 각 샘플은 자기 시점(year,period)의 그래프에만 매칭
  - 그 시점이 train 경계(2023) 이전인지로 train/test 결정 (09b 와 동일 obs_date)
  - 양성: label_date 이전 시점만 (09b 에서 이미 컷된 시퀀스의 마지막 시점 사용)
  - 관계피처 nbr_crisis_ratio: 그 시점까지 이미 위기난 이웃만 (미래위기 제외)

전제: 09b(sequences_hy.npz), 08b(graph_*_hy) 완료.
"""
import numpy as np, sqlite3
from pathlib import Path
from collections import defaultdict
import warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import torch, torch.nn as nn
from torch_geometric.nn import SAGEConv

# ===== 실행 모드 =====
MODE = "ablation"     # 'ablation' | 'aug' | 'rel'

_DB_DIR = Path(__file__).resolve().parent.parent / "db"
NPZ=_DB_DIR / "sequences_hy.npz"
DB =_DB_DIR / "dart_v2.db"
EPOCHS=80; LR=1e-3; GAMMA=2.0; PATIENCE=12; HID=64
TRAIN_CUT="20230101"
SEEDS=[42,123,456,789,2024]

RATIO_COLS=["debt_ratio","equity_ratio","debt_to_assets","noncurrent_liab_ratio",
"current_ratio","quick_ratio","cash_ratio","roa","roe","operating_margin",
"net_margin","gross_margin","asset_turnover","inventory_turnover",
"interest_coverage_proxy","ocf_to_current_liab","retained_earnings_ratio",
"working_capital_ratio","z_score"]
N_RATIO=len(RATIO_COLS); N_REL=4; Z_IDX=RATIO_COLS.index("z_score")

class FocalLoss(nn.Module):
    def __init__(s,a,g=2.0): super().__init__(); s.a,s.g=a,g
    def forward(s,l,t):
        p=torch.sigmoid(l); ce=nn.functional.binary_cross_entropy_with_logits(l,t,reduction="none")
        pt=p*t+(1-p)*(1-t); at=s.a*t+(1-s.a)*(1-t); return (at*(1-pt)**s.g*ce).mean()
class LSTMEnc(nn.Module):
    def __init__(s,d,h=HID): super().__init__(); s.lstm=nn.LSTM(d,h,batch_first=True)
    def forward(s,x,m): o,_=s.lstm(x); i=(m.sum(1)-1).long().clamp(min=0); return o[torch.arange(o.size(0)),i]
class CNNLSTMEnc(nn.Module):
    def __init__(s,d,h=HID): super().__init__(); s.conv=nn.Conv1d(d,32,2,padding=1); s.relu=nn.ReLU(); s.lstm=nn.LSTM(32,h,batch_first=True)
    def forward(s,x,m): c=s.relu(s.conv(x.transpose(1,2))).transpose(1,2)[:,:x.size(1),:]; o,_=s.lstm(c); i=(m.sum(1)-1).long().clamp(min=0); return o[torch.arange(o.size(0)),i]
class TFTEnc(nn.Module):
    def __init__(s,d,h=HID,heads=4): super().__init__(); s.proj=nn.Linear(d,h); s.grn=nn.Sequential(nn.Linear(h,h),nn.ELU(),nn.Linear(h,h),nn.Dropout(0.2)); s.attn=nn.MultiheadAttention(h,heads,batch_first=True); s.norm=nn.LayerNorm(h)
    def forward(s,x,m): h=s.proj(x); h=h+s.grn(h); a,_=s.attn(h,h,h,key_padding_mask=(m==0)); h=s.norm(h+a); i=(m.sum(1)-1).long().clamp(min=0); return h[torch.arange(h.size(0)),i]
ENCODERS={"LSTM":LSTMEnc,"CNN+LSTM":CNNLSTMEnc,"TFT":TFTEnc}

class SAGEEnc(nn.Module):
    def __init__(s,d,h=HID): super().__init__(); s.c1=SAGEConv(d,h); s.c2=SAGEConv(h,h); s.relu=nn.ReLU(); s.drop=nn.Dropout(0.3)
    def forward(s,x,ei): h=s.relu(s.c1(x,ei)); h=s.drop(h); return s.c2(h,ei)
class Combined(nn.Module):
    def __init__(s,Enc,d_seq,d_node,use_graph=True):
        super().__init__(); s.use_graph=use_graph; s.enc=Enc(d_seq)
        if use_graph:
            s.sage=SAGEEnc(d_node); s.head=nn.Sequential(nn.Linear(HID*2,32),nn.ReLU(),nn.Dropout(0.3),nn.Linear(32,1))
        else:
            s.head=nn.Sequential(nn.Linear(HID,32),nn.ReLU(),nn.Dropout(0.3),nn.Linear(32,1))
    def forward(s,xb,mb,geb=None):
        et=s.enc(xb,mb); z=torch.cat([et,geb],1) if s.use_graph else et; return s.head(z).squeeze(-1)

def ks(y,sc): fpr,tpr,_=roc_curve(y,sc); return float(np.max(tpr-fpr))

# ===== 반기 그래프 로드 (관계피처 옵션) =====
def load_graphs_hy(mean,std,lo,hi, use_rel):
    con=sqlite3.connect(DB); cur=con.cursor(); col=", ".join(RATIO_COLS)
    crisis_year={cc:int(dt[:4]) for cc,l,dt in cur.execute("SELECT corp_code,label,label_date FROM labels") if l==1 and dt}
    ind={cc:ic for cc,ic in cur.execute("SELECT corp_code,industry_code FROM companies WHERE industry_code IS NOT NULL AND industry_code!=''")}
    graphs={}
    keys=cur.execute("SELECT DISTINCT year,period FROM graph_nodes_hy ORDER BY year,period").fetchall()
    for year,period in keys:
        nodes=cur.execute("SELECT corp_code,node_idx FROM graph_nodes_hy WHERE year=? AND period=? ORDER BY node_idx",(year,period)).fetchall()
        cc2idx={cc:i for cc,i in nodes}; idx2cc={i:cc for cc,i in nodes}; n=len(nodes)
        feat=np.zeros((n,N_RATIO),np.float32); raw_z=np.full(n,np.nan)
        fmap={r[0]:r[1:] for r in cur.execute(f"SELECT corp_code,{col} FROM financials_hy WHERE year=? AND period=?",(year,period)).fetchall()}
        for cc,i in cc2idx.items():
            vals=fmap.get(cc)
            if vals is None: continue
            v=np.array([np.nan if z is None else float(z) for z in vals]); raw_z[i]=v[Z_IDX]
            v=np.clip(v,lo,hi); v=(v-mean)/std; feat[i]=np.nan_to_num(v,nan=0.0)
        edges=cur.execute("SELECT src,dst FROM graph_edges_hy WHERE year=? AND period=?",(year,period)).fetchall()
        ei=np.array(edges,dtype=np.int64).T if edges else np.zeros((2,0),np.int64)
        if use_rel:
            adj=[[] for _ in range(n)]
            for sname,t in edges: adj[sname].append(t)
            rel=np.zeros((n,N_REL),np.float32); ind_zs={}
            for cc,i in cc2idx.items():
                ic=ind.get(cc)
                if ic and not np.isnan(raw_z[i]): ind_zs.setdefault(ic,[]).append(raw_z[i])
            for cc,i in cc2idx.items():
                nb=adj[i]; deg=len(nb)
                if deg>0:
                    cc_cnt=0; zs=[]
                    for j in nb:
                        ncc=idx2cc[j]; cy=crisis_year.get(ncc)
                        if cy is not None and cy<=year: cc_cnt+=1   # 누수컷: 미래위기 제외
                        if not np.isnan(raw_z[j]): zs.append(raw_z[j])
                    rel[i,0]=cc_cnt/deg; rel[i,1]=np.mean(zs) if zs else 0.0
                rel[i,2]=np.log1p(deg)
                ic=ind.get(cc)
                if ic and ic in ind_zs and not np.isnan(raw_z[i]) and len(ind_zs[ic])>1:
                    rel[i,3]=(np.array(ind_zs[ic])<raw_z[i]).mean()
                else: rel[i,3]=0.5
            rm=rel.mean(0); rs=rel.std(0); rs[rs==0]=1; rel=(rel-rm)/rs
            node_feat=np.concatenate([feat,rel],1)
        else:
            node_feat=feat
        sl=np.arange(n); ei=np.concatenate([ei,np.stack([sl,sl])],1)
        graphs[(year,period)]=dict(cc2idx=cc2idx,x=torch.tensor(node_feat.astype(np.float32)),edge_index=torch.tensor(ei))
    con.close(); return graphs

def main():
    d=np.load(NPZ,allow_pickle=True)
    Xtr,Mtr,ytr,cctr=d["X_train"],d["M_train"],d["y_train"],d["cc_train"]
    Xte,Mte,yte,ccte=d["X_test"],d["M_test"],d["y_test"],d["cc_test"]
    mean,std=d["scaler_mean"],d["scaler_std"]; lo,hi=d["winsor_lo"],d["winsor_hi"]
    print(f"[MODE={MODE}] train {Xtr.shape} 양성 {int(ytr.sum())} / test 양성 {int(yte.sum())}")

    con=sqlite3.connect(DB)
    # 각 기업의 시점(year,period) 복원: 09b 가 obs_date 기준이므로, 그 기업의
    # '<label_date 최신 시점'을 financials_hy 에서 다시 찾음 (시퀀스 마지막 시점과 일치)
    lab={cc:(l,dt.replace("-","") if dt else None) for cc,l,dt in con.execute("SELECT corp_code,label,label_date FROM labels")}
    fh=con.execute("SELECT corp_code,year,period,obs_date FROM financials_hy WHERE data_quality!='financial_sector'").fetchall()
    by_cc={}
    for cc,y,p,od in fh: by_cc.setdefault(cc,[]).append((od,y,p))
    con.close()

    use_rel=(MODE=="rel")
    print("그래프 로딩...(관계피처:%s)"%use_rel)
    graphs=load_graphs_hy(mean,std,lo,hi,use_rel)
    if MODE=="rel":
        graphs_base=load_graphs_hy(mean,std,lo,hi,False)

    def last_node(cc, is_train):
        """그 기업의 사용 시점(year,period) -> graph key. 누수: label_date 이전 최신."""
        if cc not in by_cc: return None
        l,ld=lab.get(cc,(0,None))
        cut=ld if (l==1 and ld) else TRAIN_CUT  # 음성은 보수적으로 train_cut 이전
        cands=sorted([(od,y,p) for (od,y,p) in by_cc[cc] if (ld is None or od<cut)], key=lambda x:x[0])
        if not cands: return None
        od,y,p=cands[-1]
        return (y,p)

    def make_map(ccs,is_train,G):
        out=[]
        for cc in ccs:
            key=last_node(cc,is_train)
            if key and key in G and cc in G[key]["cc2idx"]:
                out.append((key,G[key]["cc2idx"][cc]))
            else: out.append(None)
        return out

    d_seq=Xtr.shape[2]

    def run(Enc,use_graph,G,d_node,seed,tr_map,te_map,Xt=None,Mt=None,yt=None):
        Xt=Xtr if Xt is None else Xt; Mt=Mtr if Mt is None else Mt; yt=ytr if yt is None else yt
        torch.manual_seed(seed)
        model=Combined(Enc,d_seq,d_node,use_graph)
        opt=torch.optim.Adam(model.parameters(),lr=LR,weight_decay=1e-4); crit=FocalLoss(float(1-yt.mean()),GAMMA)
        Xtt=torch.tensor(Xt); Mtt=torch.tensor(Mt); ytt=torch.tensor(yt)
        Xee=torch.tensor(Xte); Mee=torch.tensor(Mte)
        def gemb(mp,nrow):
            if not use_graph: return None
            cache={k:model.sage(G[k]["x"],G[k]["edge_index"]) for k in G}
            g=torch.zeros(nrow,HID)
            for i,m in enumerate(mp):
                if m: g[i]=cache[m[0]][m[1]]
            return g
        bp,bs,w=0,None,0
        for ep in range(EPOCHS):
            model.train(); opt.zero_grad()
            loss=crit(model(Xtt,Mtt,gemb(tr_map,len(Xt))),ytt); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            model.eval()
            with torch.no_grad(): sc=torch.sigmoid(model(Xee,Mee,gemb(te_map,len(Xte)))).numpy()
            pr=average_precision_score(yte,sc)
            if pr>bp: bp,bs,w=pr,{k:v.clone() for k,v in model.state_dict().items()},0
            else:
                w+=1
                if w>=PATIENCE: break
        if bs: model.load_state_dict(bs)
        model.eval()
        with torch.no_grad(): sc=torch.sigmoid(model(Xee,Mee,gemb(te_map,len(Xte)))).numpy()
        roc=roc_auc_score(yte,sc); pr=average_precision_score(yte,sc)
        return dict(roc=roc,pr=pr,gini=2*roc-1,ks=ks(yte,sc))

    def multi(Enc,use_graph,G,d_node,label,aug=None):
        tr_map=make_map(cctr,True,G); te_map=make_map(ccte,False,G)
        R=defaultdict(list)
        for seed in SEEDS:
            if aug is None:
                r=run(Enc,use_graph,G,d_node,seed,tr_map,te_map)
            else:
                Xa,Ma,ya,tma=aug(seed,tr_map)
                r=run(Enc,use_graph,G,d_node,seed,tma,te_map,Xa,Ma,ya)
            for k in ("roc","pr","gini","ks"): R[k].append(r[k])
        out={f"{k}_m":np.mean(v) for k,v in R.items()}; out.update({f"{k}_s":np.std(v) for k,v in R.items()})
        print(f"  {label:28s} ROC {out['roc_m']:.4f}±{out['roc_s']:.4f}  PR {out['pr_m']:.4f}±{out['pr_s']:.4f}  "
              f"Gini {out['gini_m']:.4f}  KS {out['ks_m']:.4f}")
        return out

    rows=[]
    if MODE=="ablation":
        print("\n=== PART A: 3인코더 × (단독/+SAGE), 5-seed ===")
        for en,Enc in ENCODERS.items():
            r0=multi(Enc,False,graphs,N_RATIO,f"{en} 단독")
            r1=multi(Enc,True ,graphs,N_RATIO,f"{en}+SAGE")
            print(f"     -> PR Δ{r1['pr_m']-r0['pr_m']:+.4f}")
            rows+=[(en+" 단독",r0),(en+"+SAGE",r1)]
    elif MODE=="rel":
        print("\n=== PART C: 재무19 vs 재무19+관계4 (+SAGE), 5-seed ===")
        for en,Enc in ENCODERS.items():
            rb=multi(Enc,True,graphs_base,N_RATIO,f"{en}+SAGE|재무19")
            rr=multi(Enc,True,graphs,N_RATIO+N_REL,f"{en}+SAGE|+관계4")
            print(f"     -> 관계피처 PR Δ{rr['pr_m']-rb['pr_m']:+.4f}")
            rows+=[(en+"|재무19",rb),(en+"|+관계4",rr)]
    elif MODE=="aug":
        print("\n=== PART B: 증강 (Noise/Mixup), 5-seed ===")
        def noise_aug(seed,tmap):
            rng=np.random.default_rng(seed); pos=[i for i in range(len(ytr)) if ytr[i]==1 and tmap[i]]
            aX,aM,ay,am=[],[],[],[]
            for _ in range(2):
                for i in pos:
                    nz=rng.normal(0,0.05,Xtr[i].shape).astype(np.float32)
                    aX.append(Xtr[i]+nz*Mtr[i,:,None]); aM.append(Mtr[i]); ay.append(1.0); am.append(tmap[i])
            if not aX: return Xtr,Mtr,ytr,tmap
            Xa=np.concatenate([Xtr,np.array(aX,np.float32)],0); Ma=np.concatenate([Mtr,np.array(aM,np.float32)],0)
            ya=np.concatenate([ytr,np.array(ay,np.float32)],0); ma=list(tmap)+am
            perm=np.random.default_rng(seed+1).permutation(len(ya))
            return Xa[perm],Ma[perm],ya[perm],[ma[k] for k in perm]
        for en,Enc in ENCODERS.items():
            rb=multi(Enc,True,graphs,N_RATIO,f"{en}+SAGE|Base")
            ra=multi(Enc,True,graphs,N_RATIO,f"{en}+SAGE|Noise",aug=noise_aug)
            print(f"     -> 증강 PR Δ{ra['pr_m']-rb['pr_m']:+.4f}")
            rows+=[(en+"|Base",rb),(en+"|Noise",ra)]

    print("\n"+"="*70+"\n최종 (반기 TTM, 5-seed mean±std)")
    print(f"{'모델':24s}{'ROC':>9s}{'PR':>9s}{'Gini':>9s}{'KS':>9s}")
    for nm,r in rows: print(f"{nm:24s}{r['roc_m']:>9.4f}{r['pr_m']:>9.4f}{r['gini_m']:>9.4f}{r['ks_m']:>9.4f}")
    best=max(rows,key=lambda x:x[1]['pr_m'])
    print(f"\n>> 베스트(PR): {best[0]} PR {best[1]['pr_m']:.4f}±{best[1]['pr_s']:.4f} ROC {best[1]['roc_m']:.4f}")

if __name__=="__main__":
    main()
