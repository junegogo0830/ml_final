# -*- coding: utf-8 -*-
"""
06b_financial_calc_hy.py
반기 포함 재무비율 (TTM 방식) -> financials_hy (corp_code, year, period)

period:
  'FY' = 연말 시점 (12개월 = 연간 보고서 그대로)
  'H1' = 상반기 시점 (6월 말). IS 는 TTM(최근 12개월), BS 는 6월 말 스냅샷

TTM (IS 항목, 매출/영업이익/순이익 등 flow):
  H1 시점 TTM = 당해 상반기(11012) + 전년 하반기
             = 당해 상반기 + (전년 연간 11011 - 전년 상반기 11012)
  FY 시점 TTM = 당해 연간 11011 그대로

BS 항목 (자산/부채/자본 등 stock): 해당 시점 보고서 값 그대로

==== 누수 통제 (절대 위반 금지) ====
  - 각 시점의 공시일 = substr(rcept_no,1,8) (YYYYMMDD)
  - 시점 공시일(obs_date):
      FY(year)  -> 그 연간보고서 공시일 (보통 year+1년 3월)
      H1(year)  -> 그 반기보고서 공시일 (보통 year년 8월)
  - TTM 구성에 쓰는 모든 보고서의 공시일 <= 해당 시점 obs_date
  - 정정공시 중복: (corp,year,reprt,fs_div) 별로 obs 시점 이전 '최신 rcept' 1개만
  - 라벨 누수컷은 09b 에서 obs_date < label_date 로 최종 차단

financials_hy 컬럼: 06 과 동일 19비율 + 플래그 + obs_date(YYYYMMDD)
"""
import sqlite3
from pathlib import Path
from collections import defaultdict

DB = Path(__file__).resolve().parent.parent / "db" / "dart_v2.db"
FS_PRIORITY="CFS"
FIN_CODE_PREFIX=("64","65","66")
FIN_NAME_KW=("은행","보험","증권","금융","캐피탈","저축은행","카드","자산운용","신용","상호저축","여신")

CANON={
 "total_assets":{"ids":{"ifrs-full_Assets","ifrs_Assets"},"names":{"자산총계"},"type":"BS"},
 "total_liabilities":{"ids":{"ifrs-full_Liabilities","ifrs_Liabilities"},"names":{"부채총계"},"type":"BS"},
 "total_equity":{"ids":{"ifrs-full_Equity","ifrs_Equity"},"names":{"자본총계"},"type":"BS"},
 "current_assets":{"ids":{"ifrs-full_CurrentAssets"},"names":{"유동자산"},"type":"BS"},
 "current_liabilities":{"ids":{"ifrs-full_CurrentLiabilities"},"names":{"유동부채"},"type":"BS"},
 "noncurrent_liabilities":{"ids":{"ifrs-full_NoncurrentLiabilities"},"names":{"비유동부채"},"type":"BS"},
 "inventories":{"ids":{"ifrs-full_Inventories"},"names":{"재고자산"},"type":"BS"},
 "cash":{"ids":{"ifrs-full_CashAndCashEquivalents"},"names":{"현금및현금성자산"},"type":"BS"},
 "retained_earnings":{"ids":{"ifrs-full_RetainedEarnings"},"names":{"이익잉여금","이익잉여금(결손금)","이익잉여금(결손)"},"type":"BS"},
 "revenue":{"ids":{"ifrs-full_Revenue","dart_Revenue"},"names":{"매출액","수익(매출액)","영업수익"},"type":"IS"},
 "cost_of_sales":{"ids":{"ifrs-full_CostOfSales"},"names":{"매출원가"},"type":"IS"},
 "gross_profit":{"ids":{"ifrs-full_GrossProfit"},"names":{"매출총이익","매출총이익(손실)"},"type":"IS"},
 "operating_income":{"ids":{"dart_OperatingIncomeLoss","ifrs-full_ProfitLossFromOperatingActivities"},"names":{"영업이익","영업이익(손실)"},"type":"IS"},
 "net_income":{"ids":{"ifrs-full_ProfitLoss"},"names":{"당기순이익","당기순이익(손실)"},"type":"IS"},
 "finance_costs":{"ids":{"ifrs-full_FinanceCosts"},"names":{"금융원가","금융비용"},"type":"IS"},
 "ocf":{"ids":{"ifrs-full_CashFlowsFromUsedInOperatingActivities"},"names":{"영업활동현금흐름","영업활동으로인한현금흐름"},"type":"IS"},
}
def _norm(s): return "".join(str(s).split()) if s is not None else ""
ID2C={}; NM2C={}; ACCTYPE={}
for c,d in CANON.items():
    ACCTYPE[c]=d["type"]
    for i in d["ids"]: ID2C[i]=c
    for n in d["names"]: NM2C[_norm(n)]=c
IS_ACCTS={c for c,t in ACCTYPE.items() if t=="IS"}
CORE=("total_assets","total_liabilities","total_equity","current_assets",
      "current_liabilities","revenue","operating_income","net_income")

def parse_amount(s):
    if s is None: return None
    s=str(s).strip().replace(",","")
    if s in ("","-","N/A","nan","None"): return None
    try: return float(s)
    except ValueError: return None
def sd(a,b):  return None if (a is None or b is None or b==0) else a/b
def sdp(a,b): return None if (a is None or b is None or b<=0) else a/b
def is_financial(code,name):
    code=(code or "").strip(); name=name or ""
    if code[:2] in FIN_CODE_PREFIX: return True
    return any(k in name for k in FIN_NAME_KW)


def collect_reports(cur):
    """(corp,year,reprt,fs_div) -> [(공시일, {canon:val}), ...] 공시일 오름차순 인덱스.
       pick_latest 가 전체스캔 없이 해당 키만 보게 하기 위함 (O(n^2) -> O(n))."""
    rows=cur.execute(
        "SELECT corp_code,bsns_year,reprt_code,fs_div,account_id,account_nm,"
        "thstrm_amount,rcept_no FROM financial_raw "
        "WHERE reprt_code IN ('11011','11012')")
    # 1차: rcept 단위로 값 모으기
    tmp=defaultdict(dict); rdate={}
    while True:
        chunk=rows.fetchmany(50000)
        if not chunk: break
        for cc,yr,rc,fsd,aid,anm,amt,rno in chunk:
            canon=ID2C.get(aid) or NM2C.get(_norm(anm))
            if canon is None: continue
            val=parse_amount(amt)
            if val is None: continue
            rno=str(rno); k=(cc,int(yr),rc,fsd,rno)
            tmp[k].setdefault(canon,val); rdate[k]=rno[:8]
    # 2차: (corp,year,reprt,fsd) -> [(공시일, vals)] 정렬 리스트
    idx=defaultdict(list)
    for (cc,yr,rc,fsd,rno),vals in tmp.items():
        idx[(cc,yr,rc,fsd)].append((rdate[(cc,yr,rc,fsd,rno)], vals))
    for key in idx:
        idx[key].sort(key=lambda x:x[0])
    return idx


def pick_latest(idx, cc, year, reprt, fsd, on_or_before):
    """인덱스에서 공시일<=on_or_before 인 것 중 최신 1개. 전체스캔 없음."""
    lst=idx.get((cc,year,reprt,fsd))
    if not lst: return None, None
    best=None; best_date=None
    for dt,vals in lst:            # 그 키만, 이미 정렬됨
        if dt<=on_or_before:
            best=vals; best_date=dt
        else:
            break
    return best, best_date


def main():
    con=sqlite3.connect(DB); cur=con.cursor()

    comp={cc:(ic,nm) for cc,ic,nm in cur.execute(
        "SELECT corp_code,industry_code,industry_name FROM companies")}

    # 양성 기업 label_date (YYYYMMDD): obs_date >= 이 날짜면 누수 → 제외
    pos_label_date={cc:ldt.replace("-","")
        for cc,ldt in cur.execute(
            "SELECT corp_code,label_date FROM labels WHERE label=1 AND label_date IS NOT NULL")}

    # 기업별 fs_div (연간 기준 CFS 우선)
    has=defaultdict(dict)
    for cc,fsd,ny in cur.execute(
        "SELECT corp_code,fs_div,COUNT(DISTINCT bsns_year) FROM financial_raw "
        "WHERE reprt_code='11011' GROUP BY corp_code,fs_div"):
        has[cc][fsd]=ny
    company_fs={cc:(FS_PRIORITY if FS_PRIORITY in m else max(m,key=m.get))
                for cc,m in has.items()}

    print("[1] financial_raw 수집/인덱싱 중...")
    idx=collect_reports(cur)
    print(f"    보고서 키 {len(idx):,}개 (인덱싱 완료)")

    # 시점 목록: 각 기업 x year x {FY,H1}
    years=sorted({y for (_,y,_,_) in idx.keys()})
    cur.execute("DROP TABLE IF EXISTS financials_hy")
    cur.execute("""CREATE TABLE financials_hy(
        corp_code TEXT, year INTEGER, period TEXT, obs_date TEXT, fs_div TEXT,
        debt_ratio REAL,equity_ratio REAL,debt_to_assets REAL,noncurrent_liab_ratio REAL,
        current_ratio REAL,quick_ratio REAL,cash_ratio REAL,roa REAL,roe REAL,
        operating_margin REAL,net_margin REAL,gross_margin REAL,asset_turnover REAL,
        inventory_turnover REAL,interest_coverage_proxy REAL,ocf_to_current_liab REAL,
        retained_earnings_ratio REAL,working_capital_ratio REAL,z_score REAL,
        is_capital_impaired INTEGER,is_financial_sector INTEGER,
        n_core_found INTEGER,data_quality TEXT,
        PRIMARY KEY(corp_code,year,period))""")

    def build_ratios(bs, isd):
        """bs: BS항목 dict, isd: IS항목 TTM dict -> 19비율."""
        g=lambda k: bs.get(k) if k in bs else isd.get(k)
        TA,TL,TE=g("total_assets"),g("total_liabilities"),g("total_equity")
        CA,CL=g("current_assets"),g("current_liabilities")
        NCL,INV,CASH=g("noncurrent_liabilities"),g("inventories"),g("cash")
        RE=g("retained_earnings")
        REV,COGS,GP=isd.get("revenue"),isd.get("cost_of_sales"),isd.get("gross_profit")
        OI,NI,FC,OCF=isd.get("operating_income"),isd.get("net_income"),isd.get("finance_costs"),isd.get("ocf")
        wc=(CA-CL) if (CA is not None and CL is not None) else None
        qa=(CA-INV) if (CA is not None and INV is not None) else None
        r=dict(
            debt_ratio=sdp(TL,TE),equity_ratio=sdp(TE,TA),debt_to_assets=sdp(TL,TA),
            noncurrent_liab_ratio=sdp(NCL,TE),current_ratio=sd(CA,CL),
            quick_ratio=sd(qa,CL),cash_ratio=sd(CASH,CL),roa=sdp(NI,TA),roe=sdp(NI,TE),
            operating_margin=sdp(OI,REV),net_margin=sdp(NI,REV),gross_margin=sdp(GP,REV),
            asset_turnover=sdp(REV,TA),inventory_turnover=sdp(COGS,INV),
            interest_coverage_proxy=sdp(OI,FC),ocf_to_current_liab=sd(OCF,CL),
            retained_earnings_ratio=sdp(RE,TA),working_capital_ratio=sdp(wc,TA))
        X1=r["working_capital_ratio"]; X2=r["retained_earnings_ratio"]
        X3=sdp(OI,TA); X4=sdp(TE,TL)
        r["z_score"]=(6.56*X1+3.26*X2+6.72*X3+1.05*X4) if None not in (X1,X2,X3,X4) else None
        r["_TE"]=TE
        r["_core"]={k:(bs.get(k) if ACCTYPE.get(k)=="BS" else isd.get(k)) for k in CORE}
        return r

    out=[]; n_fy=n_h1=0
    for cc in company_fs:
        fsd=company_fs[cc]
        for year in years:
            # ---- FY 시점 ----
            fy_vals,fy_date=pick_latest(idx,cc,year,'11011',fsd,'99999999')
            if fy_vals:
                lcut=pos_label_date.get(cc)
                if lcut and fy_date and fy_date>=lcut:
                    pass  # 양성 기업: 공시일 >= label_date → 누수 제외
                else:
                    bs={k:v for k,v in fy_vals.items() if ACCTYPE.get(k)=="BS"}
                    isd={k:v for k,v in fy_vals.items() if ACCTYPE.get(k)=="IS"}
                    rec=build_ratios(bs,isd)
                    out.append(("FY",cc,year,fy_date,fsd,rec)); n_fy+=1

            # ---- H1 시점 (TTM = 당해상반기 + 전년하반기) ----
            h1_vals,h1_date=pick_latest(idx,cc,year,'11012',fsd,'99999999')
            if h1_vals and h1_date:
                lcut=pos_label_date.get(cc)
                if lcut and h1_date>=lcut:
                    pass  # 양성 기업: 공시일 >= label_date → 누수 제외
                else:
                    bs={k:v for k,v in h1_vals.items() if ACCTYPE.get(k)=="BS"}
                    cur_h1_is={k:v for k,v in h1_vals.items() if ACCTYPE.get(k)=="IS"}
                    py_fy,_=pick_latest(idx,cc,year-1,'11011',fsd,h1_date)
                    py_h1,_=pick_latest(idx,cc,year-1,'11012',fsd,h1_date)
                    ttm_is={}
                    if py_fy and py_h1:
                        for k in IS_ACCTS:
                            a=cur_h1_is.get(k); pf=py_fy.get(k); ph=py_h1.get(k)
                            if a is not None and pf is not None and ph is not None:
                                ttm_is[k]=a+(pf-ph)
                    rec=build_ratios(bs,ttm_is)
                    out.append(("H1",cc,year,h1_date,fsd,rec)); n_h1+=1

    print(f"[2] 시점 생성: FY {n_fy:,} / H1 {n_h1:,}")

    ins="""INSERT OR REPLACE INTO financials_hy VALUES
        (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"""
    rows_out=[]
    for period,cc,year,odate,fsd,rec in out:
        TE=rec["_TE"]; imp=1 if (TE is not None and TE<=0) else 0
        ic,nm=comp.get(cc,(None,None)); fin=1 if is_financial(ic,nm) else 0
        ncore=sum(1 for k in CORE if rec["_core"].get(k) is not None)
        if fin: dq="financial_sector"
        elif imp: dq="distressed"
        elif ncore==len(CORE): dq="good"
        else: dq="partial"
        rows_out.append((cc,year,period,odate,fsd,
            rec["debt_ratio"],rec["equity_ratio"],rec["debt_to_assets"],
            rec["noncurrent_liab_ratio"],rec["current_ratio"],rec["quick_ratio"],
            rec["cash_ratio"],rec["roa"],rec["roe"],rec["operating_margin"],
            rec["net_margin"],rec["gross_margin"],rec["asset_turnover"],
            rec["inventory_turnover"],rec["interest_coverage_proxy"],
            rec["ocf_to_current_liab"],rec["retained_earnings_ratio"],
            rec["working_capital_ratio"],rec["z_score"],imp,fin,ncore,dq))
    cur.executemany(ins,rows_out); con.commit()
    print(f"[3] financials_hy 적재: {len(rows_out):,} 행")

    print("\ndata_quality 분포:")
    for dq,n in cur.execute("SELECT data_quality,COUNT(*) FROM financials_hy GROUP BY data_quality ORDER BY COUNT(*) DESC"):
        print(f"  {dq:18s}: {n:,}")
    print("\nperiod 분포:")
    for p,n in cur.execute("SELECT period,COUNT(*) FROM financials_hy GROUP BY period"):
        print(f"  {p}: {n:,}")
    zt=cur.execute("SELECT COUNT(*) FROM financials_hy").fetchone()[0]
    zn=cur.execute("SELECT COUNT(*) FROM financials_hy WHERE z_score IS NOT NULL").fetchone()[0]
    print(f"\nz_score 채움률: {zn:,}/{zt:,} ({zn/zt*100:.1f}%)")
    con.close()
    print("\n다음: verify_no_leak.py 로 누수검증 -> 08b/09b")


if __name__=="__main__":
    main()
