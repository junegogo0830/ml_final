"""
3단계: DART API → companies 테이블 INSERT
실행: python collect_companies.py

사전 준비:
  1. dart.fss.or.kr 회원가입 후 API 키 발급
  2. API_KEY 변수에 발급받은 키 입력

수집 대상: 코스피 + 코스닥 전체 상장사
출력: bankruptcy_prediction.db의 companies 테이블
"""

import sqlite3
import requests
import zipfile
import io
import xml.etree.ElementTree as ET
import pandas as pd
import time

DB_PATH = "db/bankruptcy_prediction.db"
API_KEY = "926f5be1959376e863235e6be6868be04c55b1eb"  # ← 필수 수정


# ── 1. 전체 기업 고유번호 목록 수집 ─────────────────────────────────
def get_corp_list():
    """
    DART에서 전체 기업 목록을 ZIP 파일로 다운로드
    corp_code(고유번호), corp_name, stock_code, market 포함
    """
    print("전체 기업 목록 다운로드 중...")
    url = "https://opendart.fss.or.kr/api/corpCode.xml"
    params = {"crtfc_key": API_KEY}

    res = requests.get(url, params=params, timeout=30)

    if res.status_code != 200:
        raise Exception(f"API 호출 실패: {res.status_code}")

    # ZIP 파일 안에 XML이 있음
    zf = zipfile.ZipFile(io.BytesIO(res.content))
    xml_data = zf.read("CORPCODE.xml")

    # XML 파싱
    root = ET.fromstring(xml_data)
    companies = []

    for item in root.findall("list"):
        corp_code  = item.findtext("corp_code", "")
        corp_name  = item.findtext("corp_name", "")
        stock_code = item.findtext("stock_code", "").strip()
        modify_date = item.findtext("modify_date", "")

        # 상장사만 (stock_code가 있는 기업)
        if stock_code:
            companies.append({
                "corp_code":  corp_code,
                "corp_name":  corp_name,
                "stock_code": stock_code,
                "modify_date": modify_date,
            })

    df = pd.DataFrame(companies)
    print(f"상장사 목록 수집 완료: {len(df)}개 기업")
    return df


# ── 2. 기업 상세정보 수집 (업종코드, 시장구분 등) ────────────────────
def get_corp_detail(corp_code):
    """
    개별 기업의 상세정보 조회
    업종코드, 대표자명, 설립일, 시장구분 포함
    """
    url = "https://opendart.fss.or.kr/api/company.json"
    params = {
        "crtfc_key": API_KEY,
        "corp_code": corp_code,
    }

    try:
        res = requests.get(url, params=params, timeout=10)
        data = res.json()

        if data.get("status") == "000":
            return {
                "market":         data.get("corp_cls", ""),     # 유가/코스닥
                "industry_code":  data.get("induty_code", ""),    # 업종코드
                "industry_name":  data.get("induty_code", ""),      # 업종명
                "ceo_name":       data.get("ceo_nm", ""),         # 대표자
                "found_date":     data.get("est_dt", ""),         # 설립일
            }
    except Exception:
        pass

    return {}


# ── 3. DB INSERT ─────────────────────────────────────────────────────
def insert_companies(corp_df):
    """
    수집한 기업 목록을 companies 테이블에 INSERT
    상세정보(업종코드 등)는 배치로 추가 수집
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    total = len(corp_df)
    inserted, skipped = 0, 0

    print(f"\n기업 정보 수집 및 INSERT 시작 ({total}개)")
    print("업종코드 수집을 위해 기업당 API 1회 추가 호출")
    print("예상 소요 시간: 약 30~40분 (API 딜레이 포함)\n")

    for i, row in corp_df.iterrows():
        corp_code  = row["corp_code"]
        corp_name  = row["corp_name"]
        stock_code = row["stock_code"]

        # 진행 상황 출력 (100개마다)
        if (inserted + skipped) % 100 == 0:
            print(f"  진행중: {inserted + skipped}/{total} "
                  f"({(inserted+skipped)/total*100:.1f}%)")

        # 상세정보 조회
        detail = get_corp_detail(corp_code)

        try:
            cur.execute("""
                INSERT OR IGNORE INTO companies
                (corp_code, corp_name, stock_code, market,
                 industry_code, industry_name, ceo_name, found_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                corp_code,
                corp_name,
                stock_code.zfill(6),
                detail.get("market", ""),
                detail.get("industry_code", ""),
                detail.get("industry_name", ""),
                detail.get("ceo_name", ""),
                detail.get("found_date", ""),
            ))
            inserted += 1
        except Exception as e:
            print(f"  INSERT 실패: {corp_name} — {e}")
            skipped += 1

        # API 호출 제한 준수 (초당 2회)
        time.sleep(0.5)

        # 100개마다 중간 저장
        if inserted % 100 == 0:
            conn.commit()

    conn.commit()
    conn.close()

    print(f"\n=== INSERT 완료 ===")
    print(f"  삽입: {inserted}건 / 스킵: {skipped}건")


# ── 4. bankruptcy 테이블 corp_code 연결 ─────────────────────────────
def link_bankruptcy_corp_code():
    """
    bankruptcy 테이블의 stock_code를 기준으로
    companies 테이블의 corp_code를 연결
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # stock_code로 corp_code 매칭
    cur.execute("""
        UPDATE bankruptcy
        SET corp_code = (
            SELECT corp_code FROM companies
            WHERE companies.stock_code = bankruptcy.stock_code
            LIMIT 1
        )
        WHERE corp_code IS NULL
    """)

    updated = cur.rowcount
    conn.commit()

    # 매칭 결과 확인
    cur.execute("""
        SELECT
            COUNT(*) as total,
            COUNT(corp_code) as matched,
            COUNT(*) - COUNT(corp_code) as unmatched
        FROM bankruptcy
    """)
    result = cur.fetchone()
    conn.close()

    print(f"\n=== bankruptcy ↔ companies 연결 ===")
    print(f"  전체: {result[0]}건")
    print(f"  연결 성공: {result[1]}건")
    print(f"  연결 실패: {result[2]}건 (상폐 후 DART에서 삭제된 기업)")


# ── 5. 수집 결과 확인 ────────────────────────────────────────────────
def verify_result():
    conn = sqlite3.connect(DB_PATH)

    # 시장별 기업 수
    df_market = pd.read_sql("""
        SELECT market, COUNT(*) as cnt
        FROM companies
        GROUP BY market
        ORDER BY cnt DESC
    """, conn)

    # 업종별 상위 10개
    df_industry = pd.read_sql("""
        SELECT industry_name, COUNT(*) as cnt
        FROM companies
        WHERE industry_name != ''
        GROUP BY industry_name
        ORDER BY cnt DESC
        LIMIT 10
    """, conn)

    # 파산 기업 중 DART 매칭 현황
    df_bankrupt = pd.read_sql("""
        SELECT
            b.reason_category,
            COUNT(*) as total,
            COUNT(c.corp_code) as dart_matched
        FROM bankruptcy b
        LEFT JOIN companies c ON b.corp_code = c.corp_code
        GROUP BY b.reason_category
        ORDER BY total DESC
    """, conn)

    conn.close()

    print("\n=== 시장별 기업 수 ===")
    print(df_market.to_string(index=False))

    print("\n=== 업종별 상위 10개 ===")
    print(df_industry.to_string(index=False))

    print("\n=== 파산 기업 DART 매칭 현황 ===")
    print(df_bankrupt.to_string(index=False))


# ── 메인 ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("3단계: 기업 목록 수집")
    print("=" * 50)

    if API_KEY == "nk":
        print("\n❌ API_KEY를 입력해주세요.")
        print("   dart.fss.or.kr → 회원가입 → API 신청 → 키 발급")
        exit()

    # 1. 전체 기업 목록 수집
    corp_df = get_corp_list()

    # 2. 상세정보 포함 DB INSERT
    insert_companies(corp_df)

    # 3. bankruptcy 테이블 corp_code 연결
    link_bankruptcy_corp_code()

    # 4. 결과 확인
    verify_result()

    print("\n완료. 다음 단계: 재무제표 수집 (04_collect_financials.py)")