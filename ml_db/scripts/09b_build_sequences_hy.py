# -*- coding: utf-8 -*-
"""
09b_build_sequences_hy.py
financials_hy -> 반기 간격 시퀀스 텐서 + split (sequences_hy.npz)

시점 정렬: (year, period) 를 시간순 정렬 -> 'time_key'
  2015FY < 2016H1 < 2016FY < 2017H1 < ...
  (H1 = 그해 8월, FY = 다음해 3월 공시. 시간축은 obs_date 로 정렬)

==== 누수 통제 (공시일 기준, 절대 위반 금지) ====
  - 양성: obs_date < label_date 인 시점만 사용 (일 단위 컷)
  - 음성: pseudo label_date 를 양성 label_date 분포에서 샘플 -> obs_date < pseudo
          (양성/음성 관측창 분포 일치)
  - firm-level split + temporal split(아래 TEMPORAL_CUT 기준 obs_date)
  - 표준화: train 시퀀스 값으로만 fit

윈도우: 최대 MAX_T=6 시점(=3년치 반기), 최소 MIN_T=2, 좌측 패딩+mask
"""
import sqlite3
from pathlib import Path
from collections import defaultdict
import numpy as np

_DB_DIR = Path(__file__).resolve().parent.parent / "db"
DB=_DB_DIR / "dart_v2.db"
OUT=_DB_DIR / "sequences_hy.npz"
MAX_T=6; MIN_T=2; SEED=42
TEMPORAL_CUT="20230101"      # obs_date >= 이면 test

RATIO_COLS=["debt_ratio","equity_ratio","debt_to_assets","noncurrent_liab_ratio",
"current_ratio","quick_ratio","cash_ratio","roa","roe","operating_margin",
"net_margin","gross_margin","asset_turnover","inventory_turnover",
"interest_coverage_proxy","ocf_to_current_liab","retained_earnings_ratio",
"working_capital_ratio","z_score"]
FLAG_COLS=["is_capital_impaired"]
FEATURES=RATIO_COLS+FLAG_COLS
N_RATIO=len(RATIO_COLS)

def main():
    rng=np.random.default_rng(SEED)
    con=sqlite3.connect(DB); cur=con.cursor()

    labels={}
    for cc,l,dt in cur.execute("SELECT corp_code,label,label_date FROM labels"):
        labels[cc]=(l, dt.replace("-","") if dt else None)
    pos_dates=[d for (l,d) in labels.values() if l==1 and d]
    print(f"[1] 양성 {sum(1 for v in labels.values() if v[0]==1)} / 음성 {sum(1 for v in labels.values() if v[0]==0)}")

    col=", ".join(FEATURES)
    rows=cur.execute(
        f"SELECT corp_code,year,period,obs_date,{col} FROM financials_hy "
        f"WHERE data_quality!='financial_sector' ORDER BY corp_code,obs_date").fetchall()
    # corp -> list of (obs_date, vec)  (시간순)
    seqmap=defaultdict(list)
    for r in rows:
        cc,year,period,odate=r[0],r[1],r[2],r[3]
        vec=[np.nan if v is None else float(v) for v in r[4:]]
        seqmap[cc].append((odate,vec))
    print(f"[2] financials_hy 로드: {len(seqmap):,} 기업")

    samples=[]; drop_short=drop_nohist=0
    for cc,(l,ld) in labels.items():
        if cc not in seqmap: continue
        seq_all=sorted(seqmap[cc], key=lambda x:x[0])   # obs_date 오름차순
        if l==1:
            if ld is None: continue
            cut=ld
        else:
            cut=rng.choice(pos_dates)        # pseudo label_date
        use=[(od,v) for (od,v) in seq_all if od<cut]    # 공시일 < label_date (일단위)
        if len(use)==0: drop_nohist+=1; continue
        use=use[-MAX_T:]
        if len(use)<MIN_T: drop_short+=1; continue
        samples.append(dict(cc=cc,label=l,obs=use[-1][0],seq=use))

    npos=sum(1 for s in samples if s["label"]==1)
    print(f"[3] 시퀀스: {len(samples):,} (양성 {npos} / 음성 {len(samples)-npos})")
    print(f"    제외 시점부족 {drop_short} / 이력없음 {drop_nohist}")

    train=[s for s in samples if s["obs"]<TEMPORAL_CUT]
    test =[s for s in samples if s["obs"]>=TEMPORAL_CUT]
    print(f"[4] split train {len(train)} (양성 {sum(1 for s in train if s['label']==1)}) "
          f"/ test {len(test)} (양성 {sum(1 for s in test if s['label']==1)})")

    stacked=[]
    for s in train:
        for (_,v) in s["seq"]: stacked.append(v[:N_RATIO])
    stacked=np.array(stacked,float)
    lo=np.nanpercentile(stacked,1,axis=0); hi=np.nanpercentile(stacked,99,axis=0)
    stacked=np.clip(stacked,lo,hi)
    mean=np.nanmean(stacked,0); std=np.nanstd(stacked,0); std[std==0]=1
    print(f"[5] 표준화 fit(train only): {stacked.shape[0]:,} 시점-행")

    def tens(sl):
        N=len(sl); D=len(FEATURES)
        X=np.zeros((N,MAX_T,D),np.float32); M=np.zeros((N,MAX_T),np.float32)
        y=np.zeros(N,np.float32); ccs=[]
        for i,s in enumerate(sl):
            seq=s["seq"]; t=len(seq); off=MAX_T-t
            for k,(_,v) in enumerate(seq):
                v=np.array(v,float)
                rr=np.clip(v[:N_RATIO],lo,hi); rr=(rr-mean)/std; rr=np.nan_to_num(rr,nan=0.0)
                ff=np.nan_to_num(v[N_RATIO:],nan=0.0)
                X[i,off+k,:]=np.concatenate([rr,ff]); M[i,off+k]=1
            y[i]=s["label"]; ccs.append(s["cc"])
        return X,M,y,np.array(ccs)

    Xtr,Mtr,ytr,cctr=tens(train); Xte,Mte,yte,ccte=tens(test)
    ov=set(cctr)&set(ccte)
    assert len(ov)==0, f"firm 누수 {len(ov)}"
    print(f"[6] 텐서 train {Xtr.shape} / test {Xte.shape} | firm 겹침 0")

    np.savez_compressed(OUT,
        X_train=Xtr,M_train=Mtr,y_train=ytr,cc_train=cctr,
        X_test=Xte,M_test=Mte,y_test=yte,cc_test=ccte,
        feature_names=np.array(FEATURES),scaler_mean=mean,scaler_std=std,
        winsor_lo=lo,winsor_hi=hi)
    print(f"[7] 저장 {OUT}")
    print(f"    양성비율 train {ytr.mean()*100:.1f}% / test {yte.mean()*100:.1f}%")
    print("\n다음: verify_no_leak.py 재실행(이제 [C][D] 검사됨) -> 10b")
    con.close()

if __name__=="__main__":
    main()
