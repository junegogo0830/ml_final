# -*- coding: utf-8 -*-
"""
새 버전 학습 - 1단계: DART API에서 최신 회계연도 재무제표 수집
실행: python scripts/dart_fetch_latest.py  (프로젝트 루트에서)

동작:
  - financials 테이블의 최대 연도 + 1 을 수집 대상 연도로 설정
  - 코스피(Y)/코스닥(K) 기업 중 해당 연도 데이터가 없는 기업만 수집
  - DART 단일회사 전체 재무제표 API(CFS 우선, 없으면 OFS)로 financial_raw 저장
  - financial_raw → financials 재무비율 계산 후 저장
  - collection_log_v2, companies.collected_at 갱신
  - models/saved/pipeline_status.json 에 진행 상황 기록
"""

import json
import os
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT        = Path(__file__).resolve().parent.parent
DB_PATH     = ROOT / "db" / "dart_v2.db"
STATUS_PATH = ROOT / "models" / "saved" / "pipeline_status.json"

API_KEY = os.environ.get("DART_API_KEY", "")
if not API_KEY:
    raise RuntimeError("환경변수 DART_API_KEY가 설정되지 않았습니다. "
                       "set DART_API_KEY=발급받은_키  (Windows) 후 다시 실행하세요.")

REPORT_CODE = "11011"   # 사업보고서 (연간)
DELAY       = 0.4       # API 호출 간 대기(초)

FINANCIAL_COLS = [
    "debt_ratio", "equity_ratio", "debt_to_assets", "noncurrent_liab_ratio",
    "current_ratio", "quick_ratio", "cash_ratio", "roa", "roe",
    "operating_margin", "net_margin", "gross_margin", "asset_turnover",
    "inventory_turnover", "interest_coverage_proxy", "ocf_to_current_liab",
    "retained_earnings_ratio", "working_capital_ratio", "z_score",
    "is_capital_impaired", "is_financial_sector", "n_core_found", "data_quality",
]

CLIP_COLS = [
    "debt_ratio", "equity_ratio", "debt_to_assets", "noncurrent_liab_ratio",
    "current_ratio", "quick_ratio", "cash_ratio", "roa", "roe",
    "operating_margin", "net_margin", "gross_margin", "asset_turnover",
    "inventory_turnover", "interest_coverage_proxy", "ocf_to_current_liab",
    "retained_earnings_ratio", "working_capital_ratio", "z_score",
]

FINANCE_INDUSTRY_KEYWORDS = ["금융", "보험", "은행", "증권", "캐피탈", "저축은행"]


# ── 상태 파일 ────────────────────────────────────────────────────────
def write_status(**kw):
    cur = {}
    if STATUS_PATH.exists():
        try:
            cur = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        except Exception:
            cur = {}
    cur.update(kw)
    cur["updated_at"] = datetime.now().isoformat()
    STATUS_PATH.write_text(json.dumps(cur, ensure_ascii=False, indent=2), encoding="utf-8")


# ── DART API ─────────────────────────────────────────────────────────
def fetch_financial(corp_code, year, fs_div):
    url = "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json"
    params = {
        "crtfc_key":  API_KEY,
        "corp_code":  corp_code,
        "bsns_year":  str(year),
        "reprt_code": REPORT_CODE,
        "fs_div":     fs_div,
    }
    try:
        res = requests.get(url, params=params, timeout=15)
        return res.json()
    except Exception as e:
        return {"status": "ERR", "message": str(e)}


# ── 재무비율 계산 ────────────────────────────────────────────────────
def get_amount(df, keywords):
    for kw in keywords:
        m = df[df["account_nm"].str.contains(kw, na=False, regex=False)]
        if len(m) > 0:
            try:
                return float(str(m.iloc[0]["thstrm_amount"]).replace(",", "").replace(" ", ""))
            except Exception:
                continue
    return np.nan


def ratio(a, b):
    return a / b if not np.isnan(a) and not np.isnan(b) and b != 0 else np.nan


def calc_ratios(raw_df, is_financial_sector=0):
    g = lambda *kws: get_amount(raw_df, list(kws))

    total_assets  = g("자산총계")
    total_debt    = g("부채총계")
    total_equity  = g("자본총계", "자기자본")
    revenue       = g("매출액", "영업수익")
    op_income     = g("영업이익")
    net_income    = g("당기순이익")
    interest_exp  = g("이자비용")
    op_cf         = g("영업활동현금흐름", "영업활동으로인한현금흐름")
    current_assets = g("유동자산")
    current_liab   = g("유동부채")
    inventory      = g("재고자산")
    cash           = g("현금및현금성자산", "현금및단기금융상품")
    retained_earn  = g("이익잉여금")
    cogs           = g("매출원가")

    noncurrent_liab = (total_debt - current_liab
                       if not np.isnan(total_debt) and not np.isnan(current_liab) else np.nan)
    quick_assets    = (current_assets - inventory
                       if not np.isnan(current_assets) and not np.isnan(inventory) else np.nan)
    gross_profit    = (revenue - cogs
                       if not np.isnan(revenue) and not np.isnan(cogs) else np.nan)
    wc              = (current_assets - current_liab
                       if not np.isnan(current_assets) and not np.isnan(current_liab) else np.nan)

    z_components = []
    if not np.isnan(total_assets) and total_assets != 0:
        if not np.isnan(wc):             z_components.append(1.2 * wc / total_assets)
        if not np.isnan(retained_earn):  z_components.append(1.4 * retained_earn / total_assets)
        if not np.isnan(op_income):      z_components.append(3.3 * op_income / total_assets)
        if not np.isnan(revenue):        z_components.append(1.0 * revenue / total_assets)
    z_score = sum(z_components) if len(z_components) == 4 else np.nan

    is_capital_impaired = int(not np.isnan(total_equity) and total_equity <= 0)

    core_vals = [total_assets, total_debt, total_equity, revenue, op_income, net_income, op_cf, current_assets]
    n_core_found = sum(1 for v in core_vals if not np.isnan(v))
    data_quality = "good" if n_core_found >= 6 else ("fair" if n_core_found >= 3 else "poor")

    r = dict(
        debt_ratio=ratio(total_debt, total_equity),
        equity_ratio=ratio(total_equity, total_assets),
        debt_to_assets=ratio(total_debt, total_assets),
        noncurrent_liab_ratio=ratio(noncurrent_liab, total_assets),
        current_ratio=ratio(current_assets, current_liab),
        quick_ratio=ratio(quick_assets, current_liab),
        cash_ratio=ratio(cash, current_liab),
        roa=ratio(net_income, total_assets),
        roe=ratio(net_income, total_equity),
        operating_margin=ratio(op_income, revenue),
        net_margin=ratio(net_income, revenue),
        gross_margin=ratio(gross_profit, revenue),
        asset_turnover=ratio(revenue, total_assets),
        inventory_turnover=ratio(revenue, inventory),
        interest_coverage_proxy=ratio(op_income, interest_exp),
        ocf_to_current_liab=ratio(op_cf, current_liab),
        retained_earnings_ratio=ratio(retained_earn, total_assets),
        working_capital_ratio=ratio(wc, total_assets),
        z_score=z_score,
        is_capital_impaired=is_capital_impaired,
        is_financial_sector=is_financial_sector,
        n_core_found=n_core_found,
        data_quality=data_quality,
    )

    for c in CLIP_COLS:
        v = r[c]
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            r[c] = float(np.clip(v, -10, 10))
    return r


# ── DB 헬퍼 ──────────────────────────────────────────────────────────
def insert_raw(cur, corp_code, year, fs_div, data_list):
    cur.execute("DELETE FROM financial_raw WHERE corp_code=? AND bsns_year=? AND fs_div=?",
                 (corp_code, year, fs_div))
    rows = []
    now = datetime.now().isoformat()
    for item in data_list:
        rows.append((
            corp_code, year, REPORT_CODE, fs_div,
            item.get("sj_div", ""), item.get("account_id", ""), item.get("account_nm", ""),
            item.get("thstrm_nm", ""), item.get("thstrm_amount", ""),
            item.get("frmtrm_nm", ""), item.get("frmtrm_amount", ""),
            item.get("bfefrmtrm_nm", ""), item.get("bfefrmtrm_amount", ""),
            item.get("ord", 0), item.get("currency", "KRW"), item.get("rcept_no", ""), now,
        ))
    cur.executemany("""
        INSERT INTO financial_raw
        (corp_code, bsns_year, reprt_code, fs_div, sj_div, account_id, account_nm,
         thstrm_nm, thstrm_amount, frmtrm_nm, frmtrm_amount,
         bfefrmtrm_nm, bfefrmtrm_amount, ord, currency, rcept_no, collected_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)


def insert_financials(cur, corp_code, year, fs_div, ratios):
    cur.execute("DELETE FROM financials WHERE corp_code=? AND year=? AND fs_div=?",
                 (corp_code, year, fs_div))
    cols = ["corp_code", "year", "fs_div"] + FINANCIAL_COLS
    placeholders = ",".join(["?"] * len(cols))
    values = [corp_code, year, fs_div] + [ratios[c] for c in FINANCIAL_COLS]
    values = [None if isinstance(v, float) and np.isnan(v) else v for v in values]
    cur.execute(f"INSERT INTO financials ({','.join(cols)}) VALUES ({placeholders})", values)


def log_result(cur, target, status, error=""):
    now = datetime.now().isoformat()
    cur.execute("""
        INSERT INTO collection_log_v2 (step, target, status, started_at, completed_at, error)
        VALUES (?, ?, ?, ?, ?, ?)
    """, ("05_financial_raw", target, status, now, now, (error or "")[:200]))


# ── 메인 ─────────────────────────────────────────────────────────────
def main():
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()

    max_year = cur.execute("SELECT MAX(year) FROM financials").fetchone()[0] or 2024
    target_year = max_year + 1

    companies = pd.read_sql("""
        SELECT corp_code, corp_name, industry_name FROM companies
        WHERE corp_cls IN ('Y','K')
    """, conn)
    done_codes = set(r[0] for r in cur.execute(
        "SELECT corp_code FROM financials WHERE year=?", (target_year,)).fetchall())
    todo = companies[~companies["corp_code"].isin(done_codes)]

    total = len(todo)
    print(f"[dart_fetch_latest] 대상 연도: {target_year} / 수집 대상: {total}건 "
          f"(전체 {len(companies)}건 중 {len(done_codes)}건 기수집)")

    write_status(running=True, step="fetch", message=f"{target_year}년 재무제표 수집 중",
                  progress={"done": 0, "total": total, "year": target_year})

    success, no_data = 0, 0
    for i, row in enumerate(todo.itertuples(index=False)):
        corp_code, corp_name, industry_name = row.corp_code, row.corp_name, row.industry_name
        is_fin = int(any(k in (industry_name or "") for k in FINANCE_INDUSTRY_KEYWORDS))
        target = f"{corp_code}-{target_year}-{REPORT_CODE}"

        data = fetch_financial(corp_code, target_year, "CFS")
        time.sleep(DELAY)
        fs_div = "CFS"
        if data.get("status") != "000" or not data.get("list"):
            data = fetch_financial(corp_code, target_year, "OFS")
            time.sleep(DELAY)
            fs_div = "OFS"

        if data.get("status") == "000" and data.get("list"):
            insert_raw(cur, corp_code, target_year, fs_div, data["list"])
            raw_df = pd.DataFrame(data["list"])
            ratios = calc_ratios(raw_df, is_financial_sector=is_fin)
            insert_financials(cur, corp_code, target_year, fs_div, ratios)
            log_result(cur, target, "done", f"rows={len(data['list'])}")
            cur.execute("UPDATE companies SET collected_at=? WHERE corp_code=?",
                        (datetime.now().isoformat(), corp_code))
            success += 1
        else:
            log_result(cur, target, "no_data", data.get("message", ""))
            no_data += 1

        if (i + 1) % 10 == 0 or (i + 1) == total:
            conn.commit()
            write_status(progress={"done": i + 1, "total": total, "year": target_year},
                          message=f"{target_year}년 재무제표 수집 중 ({i+1}/{total}, "
                                  f"성공 {success} / 데이터없음 {no_data})")
            print(f"  진행: {i+1}/{total}  성공:{success}  없음:{no_data}")

    conn.commit()
    conn.close()
    print(f"[dart_fetch_latest] 완료 — 성공 {success}건 / 데이터없음 {no_data}건")
    write_status(message=f"{target_year}년 재무제표 수집 완료 (성공 {success} / 데이터없음 {no_data})")


if __name__ == "__main__":
    main()
