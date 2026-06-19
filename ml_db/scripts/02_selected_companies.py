"""
02_companies.py: companies 테이블 보강.

목적:
- 118,179개 entity 중 "사업보고서 제출 의무 + 실제로 제출한" 기업만 식별
- 학습 대상으로 마킹: is_active = 1
- 나머지는 is_active = 0 (수집 제외)
- corp_cls (Y/K/N/E) 정확히 분류

전략:
1) 상장사 (stock_code 있음): 자동 통과
   - corp_cls는 기업개황 API 또는 list 응답으로 보강
2) 비상장: 2015~2025 사이 사업보고서(A001) 제출 이력 체크
   - list.json?corp_code=X&bgn_de=20150101&end_de=20251231&pblntf_detail_ty=A001
   - 결과 있으면 → 학습 대상, 없으면 → 제외

호출 횟수: 비상장 약 114,212회 (4~5시간)
재시작: collection_log_v2에 corp_code별 기록 → 끊겨도 이어서

실행:
    python scripts/02_companies.py
    python scripts/02_companies.py --resume   (끊긴 후 이어서)
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


STEP_NAME = "02_companies"


def get_all_corp_codes() -> list[tuple[str, str | None]]:
    """companies 테이블에서 (corp_code, stock_code) 전체 반환."""
    with connect() as conn:
        return conn.execute(
            "SELECT corp_code, stock_code FROM companies ORDER BY corp_code"
        ).fetchall()


def get_processed_corp_codes() -> set[str]:
    """이미 02에서 처리한 corp_code 집합 (재시작용)."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT target FROM collection_log_v2 WHERE step=? AND status IN ('done', 'no_data')",
            (STEP_NAME,),
        ).fetchall()
    return {r[0] for r in rows}


def has_business_report(corp_code: str) -> tuple[bool, str | None]:
    """
    해당 corp_code가 2015~2025 사이 사업보고서(A001)를 제출했는지 확인.

    return: (제출 여부, corp_cls)
    """
    params = {
        "corp_code": corp_code,
        "bgn_de": f"{config.YEAR_FROM}0101",
        "end_de": f"{config.YEAR_TO}1231",
        "pblntf_detail_ty": "A001",  # 사업보고서
        "page_count": "10",
        "page_no": "1",
    }
    try:
        data = dart_client.call("list", params=params)
    except DartNoData:
        return False, None
    except DartQuotaExceeded:
        raise  # 한도 초과는 메인 루프에서 중단 처리
    except DartAPIError as e:
        log.warning(f"  {corp_code}: API 에러 {e}")
        return False, None

    rows = data.get("list", []) or []
    if not rows:
        return False, None

    # 첫 응답에서 corp_cls 추출
    corp_cls = rows[0].get("corp_cls")
    return True, corp_cls


def mark_company(corp_code: str, is_active: int, corp_cls: str | None):
    """companies 업데이트."""
    with connect() as conn:
        conn.execute(
            "UPDATE companies SET is_active=?, corp_cls=? WHERE corp_code=?",
            (is_active, corp_cls, corp_code),
        )


def log_target(corp_code: str, status: str, error: str = None):
    """진행 로그 기록."""
    now = datetime.now().isoformat()
    with connect() as conn:
        conn.execute(
            """INSERT INTO collection_log_v2
               (step, target, status, started_at, completed_at, error)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (STEP_NAME, corp_code, status, now, now, error),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="(더 이상 불필요 - 항상 자동으로 이어서 진행함)")
    args = parser.parse_args()

    all_companies = get_all_corp_codes()
    print(f"전체 entity: {len(all_companies):,}")

    # 상장사 (stock_code 있음): 자동 통과
    listed = [(cc, sc) for cc, sc in all_companies if sc]
    unlisted = [(cc, sc) for cc, sc in all_companies if not sc]
    print(f"  - 상장사 (즉시 통과): {len(listed):,}")
    print(f"  - 비상장 (체크 필요): {len(unlisted):,}")

    # 상장사 일괄 처리 — is_active=1, corp_cls는 별도 호출 필요
    # 일단 단순화: stock_code 있으면 is_active=1, corp_cls는 다음 단계에서 보강
    print(f"\n📌 1단계: 상장사 일괄 마킹")
    with connect() as conn:
        conn.execute(
            "UPDATE companies SET is_active=1 WHERE stock_code IS NOT NULL AND stock_code != ''"
        )
    print(f"  ✅ {len(listed):,}개 상장사 is_active=1 마킹 완료")

    # 비상장 체크
    print(f"\n📌 2단계: 비상장 사업보고서 제출 이력 체크")

    # 항상 로그 보고 이어서 진행 (resume이 기본). 이미 처리한 건 무조건 건너뜀.
    processed = get_processed_corp_codes()
    unlisted = [(cc, sc) for cc, sc in unlisted if cc not in processed]
    print(f"  이미 처리: {len(processed):,}개 | 남은 작업: {len(unlisted):,}개")
    print(f"  (예상 호출 횟수: {len(unlisted):,} / 분당 약 450회 / 약 {len(unlisted)/450:.0f}분)")

    if not unlisted:
        print("  ✅ 비상장 체크 이미 전부 완료됨. 통계만 출력.")

    found_count = 0
    total_checked = 0
    last_progress_time = datetime.now()

    for i, (corp_code, _) in enumerate(unlisted, 1):
        try:
            has_report, corp_cls = has_business_report(corp_code)
            total_checked += 1

            if has_report:
                mark_company(corp_code, is_active=1, corp_cls=corp_cls or "E")
                log_target(corp_code, "done")
                found_count += 1
            else:
                # 이미 is_active=0 상태, 로그만 남김
                log_target(corp_code, "no_data")

        except DartQuotaExceeded:
            # 일일 한도 초과: 즉시 중단. 이 corp_code는 로그 안 남김(미처리로 둠)
            print(f"\n⛔ 일일 호출 한도 도달 (이번 실행 처리: {i-1:,} / {len(unlisted):,})")
            print(f"   이번 실행 발견: {found_count:,}")
            print(f"   자정(KST) 이후 똑같이 다시 실행하면 이어서 진행됨:")
            print(f"       python scripts/02_companies.py")
            return

        except Exception as e:
            log.error(f"  {corp_code}: 예외 {e}")
            log_target(corp_code, "failed", error=str(e))

        # 진행 상황 출력 (100개마다)
        if i % 100 == 0:
            elapsed = (datetime.now() - last_progress_time).total_seconds()
            rate = 100 / elapsed if elapsed > 0 else 0
            eta_sec = (len(unlisted) - i) / rate if rate > 0 else 0
            print(f"  [{i:>6,} / {len(unlisted):,}] "
                  f"발견: {found_count:,} | "
                  f"속도: {rate:.1f} req/s | "
                  f"ETA: {eta_sec/60:.1f}분")
            last_progress_time = datetime.now()

    # 최종 통계
    print(f"\n✅ 02_companies.py 완료")
    print(f"\n📊 최종 결과:")
    with connect() as conn:
        active = conn.execute("SELECT COUNT(*) FROM companies WHERE is_active=1").fetchone()[0]
        inactive = conn.execute("SELECT COUNT(*) FROM companies WHERE is_active=0").fetchone()[0]
        listed_active = conn.execute(
            "SELECT COUNT(*) FROM companies WHERE is_active=1 AND stock_code IS NOT NULL AND stock_code!=''"
        ).fetchone()[0]
        unlisted_active = active - listed_active

        # corp_cls 분포
        cls_dist = conn.execute(
            "SELECT corp_cls, COUNT(*) FROM companies WHERE is_active=1 GROUP BY corp_cls"
        ).fetchall()

    print(f"  - 학습 대상 (is_active=1): {active:,}")
    print(f"    └ 상장사: {listed_active:,}")
    print(f"    └ 비상장 (사업보고서 제출): {unlisted_active:,}")
    print(f"  - 제외 (is_active=0): {inactive:,}")
    print(f"\n  corp_cls 분포 (학습 대상):")
    for cls, cnt in cls_dist:
        cls_name = {"Y": "유가증권", "K": "코스닥", "N": "코넥스", "E": "기타"}.get(cls, "?")
        print(f"    {cls or 'NULL'} ({cls_name}): {cnt:,}")


if __name__ == "__main__":
    main()