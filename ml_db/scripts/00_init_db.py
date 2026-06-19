"""
00_init_db.py: 새 SQLite DB 생성 + 스키마 초기화.

기존 dart_v2.db가 있으면 그대로 두고 테이블만 IF NOT EXISTS로 생성.
완전히 새로 만들고 싶으면 db/dart_v2.db 파일 수동 삭제.

실행:
    cd ml_db
    python scripts/00_init_db.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.db_helper import connect
import config


SCHEMA = """
-- ==========================================
-- 1. 기업 마스터 (활성 + 폐업 통합)
-- ==========================================
CREATE TABLE IF NOT EXISTS companies (
    corp_code     TEXT PRIMARY KEY,    -- DART 8자리
    corp_name     TEXT NOT NULL,
    corp_eng_name TEXT,
    stock_code    TEXT,                -- 6자리 (상장사만)
    corp_cls      TEXT,                -- Y(유가) / K(코스닥) / N(코넥스) / E(기타)
    modify_date   TEXT,                -- corp_code.xml의 최종변경일자
    is_active     INTEGER DEFAULT 1,   -- 1=현재 등록, 0=폐업/말소
    industry_code TEXT,                -- KSIC (02_companies.py에서 보강)
    industry_name TEXT,
    found_date    TEXT,
    ceo_name      TEXT,
    collected_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_companies_name ON companies(corp_name);
CREATE INDEX IF NOT EXISTS idx_companies_stock ON companies(stock_code);
CREATE INDEX IF NOT EXISTS idx_companies_active ON companies(is_active);

-- ==========================================
-- 2. 위기 이벤트 (파산 라벨 소스)
-- ==========================================
CREATE TABLE IF NOT EXISTS crisis_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    corp_code     TEXT NOT NULL,
    event_type    TEXT NOT NULL,
        -- 'default'        : 부도발생 (DS005 부도발생 API)
        -- 'rehab'          : 회생절차 개시신청 (DS005)
        -- 'dissolution'    : 해산사유 발생 (DS005)
        -- 'suspension'     : 영업정지 (DS005)
        -- 'workout'        : 채권은행 관리절차 개시 (DS005)
        -- 'workout_end'    : 채권은행 관리절차 중단 (DS005)
        -- 'audit_qualified': 감사의견 한정/부적정/거절 (F유형, 04에서)
        -- 'delisting'      : 상장폐지 (I유형, 03에서)
    event_date    TEXT NOT NULL,       -- YYYY-MM-DD
    rcept_no      TEXT,                -- 공시 접수번호 (DART 추적용)
    rcept_dt      TEXT,                -- 공시 접수일자
    report_nm     TEXT,                -- 보고서명 원문
    severity      INTEGER,             -- 1~5 (라벨링 시 가중치)
    raw_response  TEXT,                -- API 응답 JSON (감사 추적)
    collected_at  TEXT,
    FOREIGN KEY (corp_code) REFERENCES companies(corp_code),
    UNIQUE(corp_code, event_type, event_date, rcept_no)
);
CREATE INDEX IF NOT EXISTS idx_ce_corp ON crisis_events(corp_code);
CREATE INDEX IF NOT EXISTS idx_ce_date ON crisis_events(event_date);
CREATE INDEX IF NOT EXISTS idx_ce_type ON crisis_events(event_type);

-- ==========================================
-- 3. 감사의견 (F유형)
-- ==========================================
CREATE TABLE IF NOT EXISTS audit_opinions (
    corp_code     TEXT NOT NULL,
    bsns_year     TEXT NOT NULL,       -- '2020' 등
    reprt_code    TEXT NOT NULL,       -- 11011(사업) / 11012(반기) / 11013(1Q) / 11014(3Q)
    audit_opinion TEXT,                -- '적정' / '한정' / '부적정' / '의견거절'
    auditor       TEXT,
    rcept_no      TEXT,
    raw_response  TEXT,
    collected_at  TEXT,
    PRIMARY KEY (corp_code, bsns_year, reprt_code),
    FOREIGN KEY (corp_code) REFERENCES companies(corp_code)
);

-- ==========================================
-- 4. 재무제표 원본 (fnlttSinglAcntAll 결과)
-- ==========================================
CREATE TABLE IF NOT EXISTS financial_raw (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    corp_code     TEXT NOT NULL,
    bsns_year     INTEGER NOT NULL,
    reprt_code    TEXT NOT NULL,       -- '11011' (사업보고서) 기본
    fs_div        TEXT,                -- 'CFS'(연결) / 'OFS'(별도)
    sj_div        TEXT,                -- 'BS'(재무상태표) / 'IS'(손익) / 'CIS' / 'CF'(현금흐름)
    account_id    TEXT,                -- 표준계정ID (예: ifrs-full_Assets)
    account_nm    TEXT,                -- 계정명 (예: 자산총계)
    thstrm_nm     TEXT,                -- 당기명
    thstrm_amount TEXT,                -- 당기금액 (원 단위, 문자열로 보존)
    frmtrm_nm     TEXT,
    frmtrm_amount TEXT,
    bfefrmtrm_nm  TEXT,
    bfefrmtrm_amount TEXT,
    ord           INTEGER,             -- 정렬순서
    currency      TEXT,
    rcept_no      TEXT,
    collected_at  TEXT,
    FOREIGN KEY (corp_code) REFERENCES companies(corp_code)
);
CREATE INDEX IF NOT EXISTS idx_fr_corp_year ON financial_raw(corp_code, bsns_year);
CREATE INDEX IF NOT EXISTS idx_fr_account ON financial_raw(account_id);

-- ==========================================
-- 5. 재무비율 (계산 결과, 19개 + α)
-- ==========================================
CREATE TABLE IF NOT EXISTS financials (
    corp_code     TEXT NOT NULL,
    year          INTEGER NOT NULL,
    -- 원본 계정 (계산용)
    total_assets  REAL,
    total_debt    REAL,
    total_equity  REAL,
    revenue       REAL,
    operating_income REAL,
    net_income    REAL,
    interest_expense REAL,
    operating_cf  REAL,
    capex         REAL,
    current_assets REAL,
    current_liabilities REAL,
    retained_earnings REAL,
    ebit          REAL,
    -- 19개 비율
    debt_ratio    REAL,
    current_ratio REAL,
    interest_coverage REAL,
    net_debt_ratio REAL,
    equity_ratio  REAL,
    roa           REAL,
    roe           REAL,
    op_margin     REAL,
    net_margin    REAL,
    cfo_to_debt   REAL,
    fcf           REAL,
    revenue_growth REAL,
    op_income_growth REAL,
    asset_growth  REAL,
    interest_cov_yoy REAL,
    debt_ratio_trend REAL,
    cf_volatility REAL,
    consecutive_loss INTEGER,
    z_score       REAL,
    -- 메타
    data_quality  TEXT,                -- 'good' / 'partial' / 'imputed'
    fs_div        TEXT,                -- CFS / OFS (어느 거 썼는지)
    calculated_at TEXT,
    PRIMARY KEY (corp_code, year),
    FOREIGN KEY (corp_code) REFERENCES companies(corp_code)
);
CREATE INDEX IF NOT EXISTS idx_fin_year ON financials(year);

-- ==========================================
-- 6. 수집 진행 로그 (재시작 지원)
-- ==========================================
CREATE TABLE IF NOT EXISTS collection_log_v2 (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    step          TEXT NOT NULL,        -- '01_corp_codes', '03_crisis_events' 등
    target        TEXT NOT NULL,        -- corp_code 또는 date range 등
    status        TEXT NOT NULL,        -- 'pending' / 'in_progress' / 'done' / 'failed' / 'no_data'
    started_at    TEXT,
    completed_at  TEXT,
    error         TEXT
);
CREATE INDEX IF NOT EXISTS idx_cl_step ON collection_log_v2(step);
CREATE INDEX IF NOT EXISTS idx_cl_status ON collection_log_v2(status);
"""


def main():
    print(f"DB 경로: {config.DB_PATH}")
    print(f"DB 존재 여부: {config.DB_PATH.exists()}")

    with connect() as conn:
        for stmt in SCHEMA.split(";"):
            if stmt.strip():
                conn.execute(stmt)

    print("\n✅ 스키마 초기화 완료")

    # 생성된 테이블 확인
    with connect() as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    print(f"\n생성된 테이블 ({len(tables)}개):")
    for (t,) in tables:
        print(f"  - {t}")


if __name__ == "__main__":
    main()
