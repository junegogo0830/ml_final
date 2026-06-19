# -*- coding: utf-8 -*-
"""
08b_build_edges_hy.py
반기 포함 시점(period) 스냅샷 그래프.
  시점 키 = (year, period) 예: (2018,'H1'), (2018,'FY')
  노드 = 그 시점 financials_hy 보유 기업 (financial_sector 제외)
  엣지 3종: same_industry / financial_similarity(top-k=10) / size_proximity(top-k=10)
  표준화/유사도는 '그 시점 노드'로만 (누수 차단)

저장: graph_edges_hy(year,period,src,dst,edge_type,weight)
      graph_nodes_hy(year,period,corp_code,node_idx)
"""
import sqlite3, math
from pathlib import Path
from collections import defaultdict
import numpy as np

DB=Path(__file__).resolve().parent.parent / "db" / "dart_v2.db"
TOPK_FIN=10; TOPK_SIZE=10; TOPK_IND=10
RATIO_COLS=["debt_ratio","equity_ratio","debt_to_assets","noncurrent_liab_ratio",
"current_ratio","quick_ratio","cash_ratio","roa","roe","operating_margin",
"net_margin","gross_margin","asset_turnover","inventory_turnover",
"interest_coverage_proxy","ocf_to_current_liab","retained_earnings_ratio",
"working_capital_ratio"]   # z_score 제외(합성지표)

def winsor(a,p=0.01):
    lo=np.nanpercentile(a,p*100,axis=0); hi=np.nanpercentile(a,(1-p)*100,axis=0)
    return np.clip(a,lo,hi)

def topk_cos(mat,k):
    nrm=np.linalg.norm(mat,axis=1,keepdims=True); nrm[nrm==0]=1
    u=mat/nrm; sim=u@u.T; np.fill_diagonal(sim,-np.inf)
    n=mat.shape[0]; kk=min(k,n-1); E=set()
    for i in range(n):
        for j in np.argpartition(sim[i],-kk)[-kk:]:
            if sim[i,j]==-np.inf: continue
            a,b=(i,int(j)) if i<int(j) else (int(j),i)
            E.add((a,b,float(sim[i,j])))
    return E

def topk_size(la,k):
    n=len(la); order=np.argsort(la); kk=min(k,n-1); E=set()
    for pos in range(n):
        i=order[pos]
        for p2 in range(max(0,pos-kk),min(n,pos+kk+1)):
            if p2==pos: continue
            j=order[p2]; a,b=(int(i),int(j)) if i<j else (int(j),int(i))
            E.add((a,b,float(1/(1+abs(la[i]-la[j])))))
    return E

def main():
    con=sqlite3.connect(DB); cur=con.cursor()
    ind={cc:ic for cc,ic in cur.execute(
        "SELECT corp_code,industry_code FROM companies "
        "WHERE industry_code IS NOT NULL AND industry_code!=''")}
    cur.execute("DROP TABLE IF EXISTS graph_edges_hy")
    cur.execute("DROP TABLE IF EXISTS graph_nodes_hy")
    cur.execute("CREATE TABLE graph_edges_hy(year INT,period TEXT,src INT,dst INT,edge_type TEXT,weight REAL)")
    cur.execute("CREATE TABLE graph_nodes_hy(year INT,period TEXT,corp_code TEXT,node_idx INT,PRIMARY KEY(year,period,corp_code))")
    col=", ".join(RATIO_COLS)

    periods=cur.execute("SELECT DISTINCT year,period FROM financials_hy ORDER BY year,period").fetchall()
    tot=defaultdict(int)
    for year,period in periods:
        rows=cur.execute(
            f"SELECT corp_code,{col} FROM financials_hy "
            f"WHERE year=? AND period=? AND data_quality!='financial_sector'",
            (year,period)).fetchall()
        if len(rows)<5: continue
        corps=[r[0] for r in rows]; n=len(corps); D=len(RATIO_COLS)
        mat=np.full((n,D),np.nan)
        for i,r in enumerate(rows):
            for dd in range(D):
                if r[1+dd] is not None: mat[i,dd]=r[1+dd]
        mat=winsor(mat); cm=np.nanmean(mat,0); cs=np.nanstd(mat,0); cs[cs==0]=1
        nanidx=np.where(np.isnan(mat)); mat[nanidx]=np.take(cm,nanidx[1])
        ms=(mat-cm)/cs
        cur.executemany("INSERT INTO graph_nodes_hy VALUES(?,?,?,?)",
                        [(year,period,corps[i],i) for i in range(n)])
        # financial_similarity
        fe=topk_cos(ms,TOPK_FIN)
        for a,b,w in fe:
            cur.execute("INSERT INTO graph_edges_hy VALUES(?,?,?,?,?,?)",(year,period,a,b,"financial_similarity",w))
            cur.execute("INSERT INTO graph_edges_hy VALUES(?,?,?,?,?,?)",(year,period,b,a,"financial_similarity",w))
        tot["fin"]+=len(fe)
        # size
        amap={}
        for cc,amt in cur.execute(
            "SELECT corp_code,thstrm_amount FROM financial_raw "
            "WHERE bsns_year=? AND reprt_code=? AND (account_id='ifrs-full_Assets' OR account_nm='자산총계')",
            (year,'11011' if period=='FY' else '11012')).fetchall():
            if cc in amap: continue
            try:
                a=float(str(amt).replace(",",""))
                if a>0: amap[cc]=math.log(a)
            except: pass
        sidx=[i for i in range(n) if corps[i] in amap]
        if len(sidx)>=5:
            la=np.array([amap[corps[i]] for i in sidx])
            for pa,pb,w in topk_size(la,TOPK_SIZE):
                a,b=sidx[pa],sidx[pb]
                cur.execute("INSERT INTO graph_edges_hy VALUES(?,?,?,?,?,?)",(year,period,a,b,"size_proximity",w))
                cur.execute("INSERT INTO graph_edges_hy VALUES(?,?,?,?,?,?)",(year,period,b,a,"size_proximity",w))
            tot["size"]+=1
        # same_industry
        byi=defaultdict(list)
        for i in range(n):
            ic=ind.get(corps[i])
            if ic: byi[ic].append(i)
        ie=set()
        for ic,mem in byi.items():
            m=len(mem)
            if m<2: continue
            if m<=TOPK_IND+1:
                for x in range(m):
                    for y2 in range(x+1,m): ie.add((mem[x],mem[y2]))
            else:
                for x in range(m):
                    for off in range(1,TOPK_IND+1):
                        yv=mem[(x+off)%m]; a,b=(mem[x],yv) if mem[x]<yv else (yv,mem[x]); ie.add((a,b))
        for a,b in ie:
            cur.execute("INSERT INTO graph_edges_hy VALUES(?,?,?,?,?,?)",(year,period,a,b,"same_industry",1.0))
            cur.execute("INSERT INTO graph_edges_hy VALUES(?,?,?,?,?,?)",(year,period,b,a,"same_industry",1.0))
        tot["ind"]+=len(ie)
        con.commit()
        print(f"  {year}{period}: 노드 {n:,} fin {len(fe):,} ind {len(ie):,}")

    cur.execute("CREATE INDEX idx_hy_edges ON graph_edges_hy(year,period)")
    con.commit()
    print(f"\n엣지 총계: fin~{tot['fin']:,} ind~{tot['ind']:,}")
    iso=cur.execute("""SELECT COUNT(*) FROM graph_nodes_hy gn WHERE NOT EXISTS(
        SELECT 1 FROM graph_edges_hy ge WHERE ge.year=gn.year AND ge.period=gn.period AND ge.src=gn.node_idx)""").fetchone()[0]
    print(f"고립 노드: {iso:,}")
    con.close()
    print("다음: 09b_build_sequences_hy.py")

if __name__=="__main__":
    main()
