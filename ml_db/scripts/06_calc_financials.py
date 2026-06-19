# -*- coding: utf-8 -*-
"""
06_financial_calc.py
financial_raw -> financials (19개 재무비율 + 판단 플래그)

설계 결정 반영:
  - 19개 비율: 새 표준안 (재무구조/유동성/수익성/활동성/현금흐름/복합)
  - account_id 우선 -> account_nm fallback 이중 매칭 (DART SME 적재율 보전)
  - fs_div: 기업별 CFS 우선, CFS 없으면 OFS. 시계열 안에서 안 섞음
  - 사업보고서(11011)만 사용. 반기(11012)는 누계 IS 문제로 보류 (옵션)
  - Altman Z'' (1995, 사모/비상장 호환, 장부가 기반)
  - interest_coverage_proxy (주석 미수집 -> 금융원가로 근사)
  - 금융업(은행/보험/증권) data_quality='financial_sector'
  - 자본잠식(자본총계<=0) data_quality='distressed' + is_capital_impaired 플래그
"""

import sqlite3
from pathlib import Path
from collections import defaultdict

# ============================ CONFIG ============================
DB = Path(__file__).resolve().parent.parent / "db" / "dart_v2.db"
REPRT_CODE = "11011"          # 사업보고서만. 반기 추가하려면 여기 손대지 말고 별도 처리
FS_PRIORITY = "CFS"           # CFS 우선, 없으면 OFS

# 금융업 KSIC 접두 + 키워드 (둘 중 하나라도 걸리면 금융업)
FIN_CODE_PREFIX = ("64", "65", "66")
FIN_NAME_KW = ("은행", "보험", "증권", "금융", "캐피탈", "저축은행",
               "카드", "자산운용", "신용", "상호저축", "여신")

# ====================== ACCOUNT 매핑 (canonical) ======================
# 각 canonical 계정에 대해 account_id 집합 + account_nm 집합 (공백 제거 비교)
CANON = {
    "total_assets":            {"ids": {"ifrs-full_Assets", "ifrs_Assets"},
                                "names": {"자산총계"}},
    "total_liabilities":       {"ids": {"ifrs-full_Liabilities", "ifrs_Liabilities"},
                                "names": {"부채총계"}},
    "total_equity":            {"ids": {"ifrs-full_Equity", "ifrs_Equity"},
                                "names": {"자본총계"}},
    "current_assets":          {"ids": {"ifrs-full_CurrentAssets"},
                                "names": {"유동자산"}},
    "current_liabilities":     {"ids": {"ifrs-full_CurrentLiabilities"},
                                "names": {"유동부채"}},
    "noncurrent_liabilities":  {"ids": {"ifrs-full_NoncurrentLiabilities"},
                                "names": {"비유동부채"}},
    "inventories":             {"ids": {"ifrs-full_Inventories"},
                                "names": {"재고자산"}},
    "cash":                    {"ids": {"ifrs-full_CashAndCashEquivalents"},
                                "names": {"현금및현금성자산"}},
    "retained_earnings":       {"ids": {"ifrs-full_RetainedEarnings"},
                                "names": {"이익잉여금", "이익잉여금(결손금)", "이익잉여금(결손)"}},
    "revenue":                 {"ids": {"ifrs-full_Revenue", "dart_Revenue"},
                                "names": {"매출액", "수익(매출액)", "영업수익"}},
    "cost_of_sales":           {"ids": {"ifrs-full_CostOfSales"},
                                "names": {"매출원가"}},
    "gross_profit":            {"ids": {"ifrs-full_GrossProfit"},
                                "names": {"매출총이익", "매출총이익(손실)"}},
    "operating_income":        {"ids": {"dart_OperatingIncomeLoss",
                                        "ifrs-full_ProfitLossFromOperatingActivities"},
                                "names": {"영업이익", "영업이익(손실)"}},
    # 주의: net_income 은 ifrs-full_ProfitLoss 만. 총포괄손익(ComprehensiveIncome) 절대 매칭 금지
    "net_income":              {"ids": {"ifrs-full_ProfitLoss"},
                                "names": {"당기순이익", "당기순이익(손실)"}},
    "finance_costs":           {"ids": {"ifrs-full_FinanceCosts"},
                                "names": {"금융원가", "금융비용"}},
    "ocf":                     {"ids": {"ifrs-full_CashFlowsFromUsedInOperatingActivities"},
                                "names": {"영업활동현금흐름", "영업활동으로인한현금흐름"}},
}

# 역방향 조회 테이블
def _norm(s):
    return "".join(str(s).split()) if s is not None else ""

ID2CANON = {}
NM2CANON = {}
for canon, d in CANON.items():
    for i in d["ids"]:
        ID2CANON[i] = canon
    for n in d["names"]:
        NM2CANON[_norm(n)] = canon

# "good" 판정에 필요한 핵심 계정
CORE = ("total_assets", "total_liabilities", "total_equity",
        "current_assets", "current_liabilities",
        "revenue", "operating_income", "net_income")

# ====================== 헬퍼 ======================
def parse_amount(s):
    if s is None:
        return None
    s = str(s).strip().replace(",", "")
    if s in ("", "-", "N/A", "nan", "None"):
        return None
    try:
        return float(s)
    except ValueError:
        return None

def sd(a, b):
    """분모 0/None 만 차단 (분모 음수 허용 X -> 비율 의미 깨짐이라 차단)"""
    if a is None or b is None or b == 0:
        return None
    return a / b

def sdp(a, b):
    """분모 양수일 때만 (자본/매출/자산 등 음수 분모면 비율 무의미)"""
    if a is None or b is None or b <= 0:
        return None
    return a / b

def is_financial(code, name):
    code = (code or "").strip()
    name = name or ""
    if code[:2] in FIN_CODE_PREFIX:
        return True
    return any(k in name for k in FIN_NAME_KW)

# ====================== 실행 ======================
def main():
    con = sqlite3.connect(DB)
    cur = con.cursor()

    # 1) 기업 산업정보 로드 (금융업 판정용)
    comp = {}
    for cc, icode, iname in cur.execute(
            "SELECT corp_code, industry_code, industry_name FROM companies"):
        comp[cc] = (icode, iname)
    print(f"[1] 기업 마스터 로드: {len(comp):,}")

    # 2) 기업별 fs_div 결정 (CFS 우선, 없으면 OFS)
    fs_rows = cur.execute(
        "SELECT corp_code, fs_div, COUNT(DISTINCT bsns_year) "
        "FROM financial_raw WHERE reprt_code=? GROUP BY corp_code, fs_div",
        (REPRT_CODE,)).fetchall()
    has = defaultdict(dict)
    for cc, fsd, ny in fs_rows:
        has[cc][fsd] = ny
    company_fs = {}
    for cc, m in has.items():
        if FS_PRIORITY in m:
            company_fs[cc] = FS_PRIORITY
        else:
            # CFS 없으면 OFS (혹은 존재하는 것 중 연도 많은 것)
            company_fs[cc] = max(m, key=m.get)
    n_cfs = sum(1 for v in company_fs.values() if v == "CFS")
    print(f"[2] fs_div 결정: 기업 {len(company_fs):,} (CFS {n_cfs:,} / OFS {len(company_fs)-n_cfs:,})")

    # 3) financial_raw 스트리밍 -> (corp, year) 별 canonical 계정 dict
    data = defaultdict(dict)   # (corp_code, year) -> {canon: value}
    cur.execute(
        "SELECT corp_code, bsns_year, fs_div, account_id, account_nm, thstrm_amount "
        "FROM financial_raw WHERE reprt_code=?", (REPRT_CODE,))
    n_seen = n_mapped = 0
    while True:
        chunk = cur.fetchmany(50000)
        if not chunk:
            break
        for cc, yr, fsd, aid, anm, amt in chunk:
            n_seen += 1
            if company_fs.get(cc) != fsd:        # 선택된 fs_div 만
                continue
            canon = ID2CANON.get(aid)
            if canon is None:
                canon = NM2CANON.get(_norm(anm))
            if canon is None:
                continue
            val = parse_amount(amt)
            if val is None:
                continue
            key = (cc, int(yr))
            data[key].setdefault(canon, val)     # 첫 값 유지 (id/nm 중복은 동일값)
            n_mapped += 1
    print(f"[3] financial_raw 스캔: {n_seen:,} 행 / 매핑 {n_mapped:,} / 조합 {len(data):,}")

    # 4) financials 테이블 재생성
    cur.execute("DROP TABLE IF EXISTS financials")
    cur.execute("""
        CREATE TABLE financials (
            corp_code TEXT, year INTEGER, fs_div TEXT,
            debt_ratio REAL, equity_ratio REAL, debt_to_assets REAL,
            noncurrent_liab_ratio REAL, current_ratio REAL, quick_ratio REAL,
            cash_ratio REAL, roa REAL, roe REAL, operating_margin REAL,
            net_margin REAL, gross_margin REAL, asset_turnover REAL,
            inventory_turnover REAL, interest_coverage_proxy REAL,
            ocf_to_current_liab REAL, retained_earnings_ratio REAL,
            working_capital_ratio REAL, z_score REAL,
            is_capital_impaired INTEGER, is_financial_sector INTEGER,
            n_core_found INTEGER, data_quality TEXT,
            PRIMARY KEY (corp_code, year)
        )""")

    # 5) 비율 계산 + 적재
    insert_sql = """INSERT OR REPLACE INTO financials VALUES
        (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"""
    rows_out = []
    for (cc, yr), d in data.items():
        g = d.get
        TA, TL, TE = g("total_assets"), g("total_liabilities"), g("total_equity")
        CA, CL = g("current_assets"), g("current_liabilities")
        NCL, INV, CASH = g("noncurrent_liabilities"), g("inventories"), g("cash")
        RE, REV, COGS = g("retained_earnings"), g("revenue"), g("cost_of_sales")
        GP, OI, NI = g("gross_profit"), g("operating_income"), g("net_income")
        FC, OCF = g("finance_costs"), g("ocf")

        wc = (CA - CL) if (CA is not None and CL is not None) else None   # 운전자본
        qa = (CA - INV) if (CA is not None and INV is not None) else None # 당좌자산

        debt_ratio            = sdp(TL, TE)
        equity_ratio          = sdp(TE, TA)
        debt_to_assets        = sdp(TL, TA)
        noncurrent_liab_ratio = sdp(NCL, TE)
        current_ratio         = sd(CA, CL)
        quick_ratio           = sd(qa, CL)
        cash_ratio            = sd(CASH, CL)
        roa                   = sdp(NI, TA)
        roe                   = sdp(NI, TE)
        operating_margin      = sdp(OI, REV)
        net_margin            = sdp(NI, REV)
        gross_margin          = sdp(GP, REV)
        asset_turnover        = sdp(REV, TA)
        inventory_turnover    = sdp(COGS, INV)
        interest_coverage     = sdp(OI, FC)
        ocf_to_cl             = sd(OCF, CL)
        re_ratio              = sdp(RE, TA)
        wc_ratio              = sdp(wc, TA)

        # Altman Z'' = 6.56 X1 + 3.26 X2 + 6.72 X3 + 1.05 X4
        X1 = wc_ratio
        X2 = re_ratio
        X3 = sdp(OI, TA)
        X4 = sdp(TE, TL)              # 장부가 자본 / 총부채
        if None not in (X1, X2, X3, X4):
            z_score = 6.56*X1 + 3.26*X2 + 6.72*X3 + 1.05*X4
        else:
            z_score = None

        is_impaired = 1 if (TE is not None and TE <= 0) else 0
        icode, iname = comp.get(cc, (None, None))
        is_fin = 1 if is_financial(icode, iname) else 0
        n_core = sum(1 for k in CORE if d.get(k) is not None)

        if is_fin:
            dq = "financial_sector"
        elif is_impaired:
            dq = "distressed"
        elif n_core == len(CORE):
            dq = "good"
        else:
            dq = "partial"

        rows_out.append((
            cc, yr, company_fs.get(cc),
            debt_ratio, equity_ratio, debt_to_assets, noncurrent_liab_ratio,
            current_ratio, quick_ratio, cash_ratio, roa, roe, operating_margin,
            net_margin, gross_margin, asset_turnover, inventory_turnover,
            interest_coverage, ocf_to_cl, re_ratio, wc_ratio, z_score,
            is_impaired, is_fin, n_core, dq))

    cur.executemany(insert_sql, rows_out)
    con.commit()
    print(f"[4] financials 적재: {len(rows_out):,} 행")

    # 6) 요약 리포트
    print("\n" + "=" * 60)
    print("data_quality 분포")
    for dq, n in cur.execute(
            "SELECT data_quality, COUNT(*) FROM financials "
            "GROUP BY data_quality ORDER BY COUNT(*) DESC"):
        print(f"  {dq:18s}: {n:,}")

    fin = cur.execute("SELECT COUNT(DISTINCT corp_code) FROM financials "
                      "WHERE is_financial_sector=1").fetchone()[0]
    imp = cur.execute("SELECT COUNT(*) FROM financials "
                      "WHERE is_capital_impaired=1").fetchone()[0]
    print(f"\n  금융업 기업 수      : {fin:,}  (시계열 학습에서 제외 권장)")
    print(f"  자본잠식 기업-연도  : {imp:,}  (is_capital_impaired=1, 강신호)")

    # 라벨 커버리지
    print("\n" + "=" * 60)
    print("labels 매칭 (financials 보유 기업)")
    for lab, name in ((1, "양성(위기)"), (0, "음성(정상)")):
        n = cur.execute(
            "SELECT COUNT(DISTINCT f.corp_code) FROM financials f "
            "JOIN labels l ON l.corp_code=f.corp_code WHERE l.label=?",
            (lab,)).fetchone()[0]
        print(f"  {name}: {n:,} 기업")

    # z_score 채움률 (핵심 baseline)
    zt = cur.execute("SELECT COUNT(*) FROM financials").fetchone()[0]
    zn = cur.execute("SELECT COUNT(*) FROM financials "
                     "WHERE z_score IS NOT NULL").fetchone()[0]
    print(f"\n  z_score 채움률: {zn:,}/{zt:,} ({zn/zt*100:.1f}%)")

    con.close()
    print("\n완료. 다음: financials 분포 EDA + 엣지 구성.")


if __name__ == "__main__":
    main()