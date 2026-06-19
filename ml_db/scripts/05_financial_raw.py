"""
05_financial_raw.py: 재무제표 원본 수집.

대상:
- 학습 대상(is_active=1) 4,657개 기업
- 2015~2025 (11년)
- 사업보고서(11011) + 반기보고서(11012)
- fs_div: CFS 우선, 013(없음)이면 OFS

API: fnlttSinglAcntAll
- 한 호출 = 한 기업의 한 사업연도/한 보고서 = 모든 계정 (BS/IS/CIS/CF)
- 응답은 list 형태로 계정별 행

호출 단위: (corp_code, bsns_year, reprt_code)
- 한 단위마다 CFS 시도, 없으면 OFS 시도
- 둘 다 없으면 no_data로 로그

재시작: collection_log_v2의 step='05_financial_raw',
        target='{corp_code}-{year}-{reprt_code}'

실행:
    python scripts/05_financial_raw.py
"""

import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib import dart_client
from lib.dart_client import DartNoData, DartAPIError, DartQuotaExceeded, log
from lib.db_helper import connect
import config


STEP_NAME = "05_financial_raw"

REPORT_CODES = [
    ("11011", "사업보고서"),
    ("11012", "반기보고서"),
]


def get_active_corps() -> list[str]:
    """학습 대상 corp_code 목록."""
    with connect() as conn:
        return [r[0] for r in conn.execute(
            "SELECT corp_code FROM companies WHERE is_active=1 ORDER BY corp_code"
        ).fetchall()]


def get_processed_targets() -> set[str]:
    """이미 처리한 target 집합. (status=done 또는 no_data 둘 다 스킵)"""
    with connect() as conn:
        rows = conn.execute(
            "SELECT target FROM collection_log_v2 WHERE step=? AND status IN ('done','no_data')",
            (STEP_NAME,),
        ).fetchall()
    return {r[0] for r in rows}


def log_target(target: str, status: str, count: int = 0, error: str = None):
    now = datetime.now().isoformat()
    with connect() as conn:
        conn.execute(
            """INSERT INTO collection_log_v2
               (step, target, status, started_at, completed_at, error)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (STEP_NAME, target, status, now, now,
             f"rows={count}" + (f"; {error}" if error else "")),
        )


def fetch_financial(corp_code: str, year: int, reprt_code: str,
                    fs_div: str) -> list[dict] | None:
    """
    fnlttSinglAcntAll 호출. 성공 시 row list 반환, 데이터 없으면 None.
    DartQuotaExceeded는 상위로 전파.
    """
    params = {
        "corp_code": corp_code,
        "bsns_year": str(year),
        "reprt_code": reprt_code,
        "fs_div": fs_div,
    }
    try:
        data = dart_client.call("fnlttSinglAcntAll", params=params)
    except DartNoData:
        return None
    except DartQuotaExceeded:
        raise
    except DartAPIError as e:
        # 'list' 키가 없거나 다른 에러는 None 처리
        log.debug(f"  {corp_code}-{year}-{reprt_code}-{fs_div}: {e}")
        return None

    rows = data.get("list", []) or []
    return rows if rows else None


def insert_rows(corp_code: str, year: int, reprt_code: str,
                fs_div: str, rows: list[dict]) -> int:
    """financial_raw에 일괄 적재. return: 적재된 행 수."""
    if not rows:
        return 0
    now = datetime.now().isoformat()
    inserted = 0
    with connect() as conn:
        for r in rows:
            # currency 컬럼은 응답에 'currency' 또는 'corp_currency'
            currency = r.get("currency") or r.get("corp_currency")
            conn.execute("""
                INSERT INTO financial_raw
                (corp_code, bsns_year, reprt_code, fs_div, sj_div,
                 account_id, account_nm,
                 thstrm_nm, thstrm_amount,
                 frmtrm_nm, frmtrm_amount,
                 bfefrmtrm_nm, bfefrmtrm_amount,
                 ord, currency, rcept_no, collected_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                corp_code, year, reprt_code, fs_div,
                r.get("sj_div"),
                r.get("account_id"), r.get("account_nm"),
                r.get("thstrm_nm"), r.get("thstrm_amount"),
                r.get("frmtrm_nm"), r.get("frmtrm_amount"),
                r.get("bfefrmtrm_nm"), r.get("bfefrmtrm_amount"),
                int(r["ord"]) if r.get("ord", "").isdigit() else None,
                currency, r.get("rcept_no"), now
            ))
            inserted += 1
    return inserted


def process_one(corp_code: str, year: int, reprt_code: str) -> tuple[str, int]:
    """
    한 단위 처리. CFS 먼저 시도, 없으면 OFS.
    return: (status, row_count)
        status: 'done' (data 있고 적재) / 'no_data' (둘 다 없음)
    """
    # CFS 시도
    rows = fetch_financial(corp_code, year, reprt_code, "CFS")
    fs_div_used = "CFS"
    if rows is None:
        # OFS 시도
        rows = fetch_financial(corp_code, year, reprt_code, "OFS")
        fs_div_used = "OFS"

    if rows is None:
        return "no_data", 0

    cnt = insert_rows(corp_code, year, reprt_code, fs_div_used, rows)
    return "done", cnt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year-from", type=int, default=config.YEAR_FROM)
    parser.add_argument("--year-to", type=int, default=config.YEAR_TO)
    parser.add_argument("--reprt", choices=["11011", "11012", "both"],
                        default="both",
                        help="11011=사업, 11012=반기, both=둘다(기본)")
    args = parser.parse_args()

    print(f"📊 05 재무제표 수집 시작")
    print(f"   기간: {args.year_from} ~ {args.year_to}")
    print(f"   보고서: {args.reprt}")

    if args.reprt == "both":
        reprt_codes = [c for c, _ in REPORT_CODES]
    else:
        reprt_codes = [args.reprt]

    corps = get_active_corps()
    print(f"   학습 대상: {len(corps):,}개 기업")

    # 작업 목록 생성
    targets = []
    for cc in corps:
        for year in range(args.year_from, args.year_to + 1):
            for rc in reprt_codes:
                key = f"{cc}-{year}-{rc}"
                targets.append((cc, year, rc, key))

    total = len(targets)
    processed = get_processed_targets()
    pending = [(cc, y, rc, k) for cc, y, rc, k in targets if k not in processed]
    print(f"   전체 작업: {total:,} | 이미 처리: {len(processed):,} | 남은 작업: {len(pending):,}")

    if not pending:
        print_summary()
        return

    # 메인 루프
    done_count = 0
    no_data_count = 0
    total_rows = 0
    last_progress_time = datetime.now()

    for i, (cc, year, rc, key) in enumerate(pending, 1):
        try:
            status, cnt = process_one(cc, year, rc)
            log_target(key, status, count=cnt)
            if status == "done":
                done_count += 1
                total_rows += cnt
            else:
                no_data_count += 1

        except DartQuotaExceeded:
            print(f"\n⛔ 일일 호출 한도 도달")
            print(f"   이번 실행: {i-1:,} / {len(pending):,}")
            print(f"   data 적재: {done_count:,} 단위, 행 {total_rows:,}")
            print(f"   no_data: {no_data_count:,}")
            print(f"   자정 이후 똑같이 실행하면 이어서 진행됨")
            return

        except Exception as e:
            log.error(f"  {key}: 예외 {e}")
            log_target(key, "failed", error=str(e))

        # 진행 출력 (50개마다)
        if i % 50 == 0:
            elapsed = (datetime.now() - last_progress_time).total_seconds()
            rate = 50 / elapsed if elapsed > 0 else 0
            eta_min = (len(pending) - i) / rate / 60 if rate > 0 else 0
            print(f"  [{i:>6,} / {len(pending):,}] "
                  f"done {done_count:,} | no_data {no_data_count:,} | "
                  f"rows {total_rows:,} | "
                  f"{rate:.1f} req/s | ETA {eta_min/60:.1f}h")
            last_progress_time = datetime.now()

    print_summary()


def print_summary():
    """최종 통계."""
    print(f"\n✅ 05_financial_raw.py 완료\n")
    print("=" * 70)
    print("📊 financial_raw 통계")
    print("=" * 70)
    with connect() as conn:
        total = conn.execute("SELECT COUNT(*) FROM financial_raw").fetchone()[0]
        print(f"  전체 행 수: {total:,}")

        unique_corp_year = conn.execute(
            "SELECT COUNT(DISTINCT corp_code || '-' || bsns_year || '-' || reprt_code) "
            "FROM financial_raw"
        ).fetchone()[0]
        print(f"  (기업, 연도, 보고서) 조합: {unique_corp_year:,}")

        print()
        print("  보고서 종류별:")
        rows = conn.execute(
            "SELECT reprt_code, COUNT(DISTINCT corp_code||bsns_year), COUNT(*) "
            "FROM financial_raw GROUP BY reprt_code"
        ).fetchall()
        for rc, units, rows_cnt in rows:
            print(f"    {rc}: {units:,} 단위, {rows_cnt:,} 행")

        print()
        print("  fs_div(연결/별도) 분포:")
        rows = conn.execute(
            "SELECT fs_div, COUNT(*) FROM financial_raw GROUP BY fs_div"
        ).fetchall()
        for fs, cnt in rows:
            print(f"    {fs}: {cnt:,}")

        print()
        print("  연도별 (사업보고서):")
        rows = conn.execute(
            "SELECT bsns_year, COUNT(DISTINCT corp_code) "
            "FROM financial_raw WHERE reprt_code='11011' "
            "GROUP BY bsns_year ORDER BY bsns_year"
        ).fetchall()
        for y, cnt in rows:
            print(f"    {y}: {cnt:,} 기업")

        # 라벨 매칭
        print()
        print("  labels 매칭:")
        rows = conn.execute("""
            SELECT l.label, COUNT(DISTINCT fr.corp_code)
            FROM financial_raw fr
            JOIN labels l ON fr.corp_code = l.corp_code
            GROUP BY l.label
        """).fetchall()
        for lbl, cnt in rows:
            tag = "양성(위기)" if lbl == 1 else "음성(정상)"
            print(f"    {tag}: {cnt:,} 기업이 재무제표 보유")


if __name__ == "__main__":
    main()