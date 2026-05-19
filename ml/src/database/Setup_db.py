"""
기업 파산 예측 프로젝트 — DB 구축 스크립트
실행: python setup_db.py
생성 파일: bankruptcy_prediction.db
"""

import sqlite3
import os

DB_PATH = "db/bankruptcy_prediction.db"

def create_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # ── 1. 기업 기본정보 ────────────────────────────────────────────
    cur.execute("""
    CREATE TABLE IF NOT EXISTS companies (
        corp_code       TEXT PRIMARY KEY,   -- DART 고유번호 (8자리)
        corp_name       TEXT NOT NULL,      -- 기업명
        stock_code      TEXT,               -- 종목코드 (6자리)
        market          TEXT,               -- 코스피 / 코스닥 / 코넥스
        industry_code   TEXT,               -- KSIC 업종코드
        industry_name   TEXT,               -- 업종명
        ceo_name        TEXT,               -- 대표자명
        found_date      TEXT,               -- 설립일
        created_at      TEXT DEFAULT (datetime('now'))
    )
    """)

    # ── 2. 파산 이력 ─────────────────────────────────────────────────
    cur.execute("""
    CREATE TABLE IF NOT EXISTS bankruptcy (
        corp_code           TEXT PRIMARY KEY,
        stock_code          TEXT,
        corp_name           TEXT,
        delist_date         TEXT,           -- 상장폐지일
        delist_reason       TEXT,           -- 폐지사유 원문
        reason_category     TEXT,           -- 감사의견거절 / 자본잠식 등
        label               INTEGER DEFAULT 1,  -- 파산=1
        FOREIGN KEY (corp_code) REFERENCES companies(corp_code)
    )
    """)

    # ── 3. 연도별 재무제표 원본 ──────────────────────────────────────
    # DART API 응답을 그대로 저장 (원본 보존용)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS financial_raw (
        corp_code       TEXT,
        corp_name       TEXT,
        year            INTEGER,
        report_code     TEXT,               -- 11011=사업보고서
        fs_div          TEXT,               -- CFS=연결 / OFS=별도
        account_nm      TEXT,               -- 계정과목명
        account_id      TEXT,               -- IFRS 계정 ID
        thstrm_amount   TEXT,               -- 당기 금액 (문자열 그대로)
        frmtrm_amount   TEXT,               -- 전기 금액
        currency        TEXT DEFAULT 'KRW',
        created_at      TEXT DEFAULT (datetime('now')),
        PRIMARY KEY (corp_code, year, fs_div, account_nm)
    )
    """)

    # ── 4. 연도별 재무비율 (피처 테이블) ────────────────────────────
    # LSTM 입력으로 직접 사용하는 테이블
    cur.execute("""
    CREATE TABLE IF NOT EXISTS financials (
        corp_code           TEXT,
        year                INTEGER,

        -- 원본 수치 (억원)
        total_assets        REAL,           -- 자산총계
        total_debt          REAL,           -- 부채총계
        total_equity        REAL,           -- 자본총계
        revenue             REAL,           -- 매출액
        operating_income    REAL,           -- 영업이익
        net_income          REAL,           -- 당기순이익
        interest_expense    REAL,           -- 이자비용
        operating_cf        REAL,           -- 영업활동현금흐름
        capex               REAL,           -- 설비투자(CAPEX)
        current_assets      REAL,           -- 유동자산
        current_liabilities REAL,           -- 유동부채
        retained_earnings   REAL,           -- 이익잉여금
        ebit                REAL,           -- EBIT

        -- 안정성 비율
        debt_ratio          REAL,           -- 부채비율 = 총부채/자기자본
        current_ratio       REAL,           -- 유동비율 = 유동자산/유동부채
        interest_coverage   REAL,           -- 이자보상배율 = 영업이익/이자비용
        net_debt_ratio      REAL,           -- 순부채비율
        equity_ratio        REAL,           -- 자기자본비율

        -- 수익성 비율
        roa                 REAL,           -- ROA = 순이익/자산
        roe                 REAL,           -- ROE = 순이익/자기자본
        op_margin           REAL,           -- 영업이익률
        net_margin          REAL,           -- 순이익률

        -- 현금흐름 비율
        cfo_to_debt         REAL,           -- 영업CF/총부채
        fcf                 REAL,           -- 잉여현금흐름 = 영업CF - CAPEX

        -- 성장성 비율 (전년 대비)
        revenue_growth      REAL,           -- 매출성장률
        op_income_growth    REAL,           -- 영업이익성장률
        asset_growth        REAL,           -- 자산성장률

        -- 시계열 파생변수 (악화 속도)
        interest_cov_yoy    REAL,           -- 이자보상배율 전년 대비 변화
        debt_ratio_trend    REAL,           -- 부채비율 3년 추세 (기울기)
        cf_volatility       REAL,           -- 현금흐름 변동성
        consecutive_loss    INTEGER,        -- 연속 적자 횟수

        -- Altman Z-Score
        z_score             REAL,

        -- 데이터 품질
        data_quality        TEXT DEFAULT 'OK',  -- OK / PARTIAL / MISSING
        created_at          TEXT DEFAULT (datetime('now')),

        PRIMARY KEY (corp_code, year),
        FOREIGN KEY (corp_code) REFERENCES companies(corp_code)
    )
    """)

    # ── 5. 기업 간 관계 (그래프 엣지) ───────────────────────────────
    cur.execute("""
    CREATE TABLE IF NOT EXISTS relationships (
        corp_code_from  TEXT,
        corp_code_to    TEXT,
        edge_type       TEXT,       -- industry / financial_sim / bank
        weight          REAL,       -- 엣지 가중치 (유사도 or 1.0)
        year            INTEGER,    -- 해당 연도 기준
        detail          TEXT,       -- 추가 정보 (업종코드, 은행명 등)
        created_at      TEXT DEFAULT (datetime('now')),
        PRIMARY KEY (corp_code_from, corp_code_to, edge_type, year)
    )
    """)

    # ── 6. 수집 로그 (API 호출 추적) ────────────────────────────────
    # 수집 도중 끊겨도 어디까지 됐는지 알 수 있음
    cur.execute("""
    CREATE TABLE IF NOT EXISTS collection_log (
        corp_code       TEXT,
        year            INTEGER,
        status          TEXT,       -- SUCCESS / FAIL / NO_DATA
        error_message   TEXT,
        collected_at    TEXT DEFAULT (datetime('now')),
        PRIMARY KEY (corp_code, year)
    )
    """)

    # ── 인덱스 생성 (쿼리 속도 향상) ────────────────────────────────
    cur.execute("CREATE INDEX IF NOT EXISTS idx_financials_year ON financials(year)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_financials_corp ON financials(corp_code)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_companies_market ON companies(market)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_companies_industry ON companies(industry_code)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_relationships_from ON relationships(corp_code_from)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_relationships_type ON relationships(edge_type, year)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_log_status ON collection_log(status)")

    conn.commit()
    conn.close()
    print(f"DB 생성 완료: {DB_PATH}")
    print_schema()


def print_schema():
    """생성된 테이블 목록과 컬럼 수 출력"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = cur.fetchall()

    print("\n=== 생성된 테이블 ===")
    for (table,) in tables:
        cur.execute(f"PRAGMA table_info({table})")
        cols = cur.fetchall()
        print(f"  {table:<25} ({len(cols)}개 컬럼)")

    conn.close()


def verify_db():
    """DB 정상 작동 확인용 테스트 INSERT/SELECT"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 테스트 데이터 삽입
    cur.execute("""
        INSERT OR IGNORE INTO companies
        (corp_code, corp_name, stock_code, market, industry_code)
        VALUES (?, ?, ?, ?, ?)
    """, ("00000001", "테스트기업", "000001", "코스닥", "J5811"))

    cur.execute("""
        INSERT OR IGNORE INTO financials
        (corp_code, year, total_assets, debt_ratio, roa, interest_coverage, z_score)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, ("00000001", 2023, 5000.0, 150.0, 3.2, 2.1, 2.5))

    conn.commit()

    # 조회 확인
    cur.execute("""
        SELECT c.corp_name, f.year, f.debt_ratio, f.roa, f.z_score
        FROM companies c
        JOIN financials f ON c.corp_code = f.corp_code
        WHERE c.corp_code = '00000001'
    """)
    row = cur.fetchone()
    print(f"\n=== 테스트 조회 결과 ===")
    print(f"  기업명: {row[0]}, 연도: {row[1]}, "
          f"부채비율: {row[2]}, ROA: {row[3]}, Z-Score: {row[4]}")

    # 테스트 데이터 삭제
    cur.execute("DELETE FROM financials WHERE corp_code = '00000001'")
    cur.execute("DELETE FROM companies WHERE corp_code = '00000001'")
    conn.commit()
    conn.close()
    print("  테스트 완료 — DB 정상 작동 확인")


if __name__ == "__main__":
    if os.path.exists(DB_PATH):
        print(f"기존 DB 발견: {DB_PATH}")
        ans = input("덮어쓰시겠습니까? (y/n): ")
        if ans.lower() == 'y':
            os.remove(DB_PATH)
        else:
            print("취소됨")
            exit()

    create_db()
    verify_db()