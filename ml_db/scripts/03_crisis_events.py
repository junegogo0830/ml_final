"""
03_crisis_events.py: 위기 이벤트(부도/회생/해산/영업정지/상장폐지) 수집.

전략:
- list API로 2015~2025 기간의 B001(주요사항보고서) 전체 스캔
- 보고서명(report_nm) 키워드 매칭으로 위기 이벤트만 추출
- 추가: I유형(거래소공시)에서 상장폐지 관련 보고서
- crisis_events 테이블에 적재

호출 횟수: 약 500~1,500회 (페이지당 100건, 11년치)
실행 시간: 30분~1시간

재시작: collection_log_v2에 (step, target=YYYYMM) 기록 → 월 단위 재개

실행:
    python scripts/03_crisis_events.py
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


STEP_NAME = "03_crisis_events"


# ==========================================
# 이벤트 분류 규칙: 보고서명 키워드 → event_type
# ==========================================
# 우선순위 순서대로 매칭 (앞에 매칭되면 그것으로 확정)
EVENT_PATTERNS = [
    # 부도 (가장 강한 신호)
    ("default", ["부도발생", "부도 발생"]),
    # 회생절차
    ("rehab", ["회생절차 개시신청", "회생절차개시신청",
               "회생절차개시 신청", "회생절차개시결정", "회생절차 개시결정",
               "회생절차종결", "회생절차 종결", "회생절차폐지", "회생절차 폐지",
               "간이회생절차"]),
    # 해산
    ("dissolution", ["해산사유발생", "해산사유 발생", "해산결정"]),
    # 영업정지
    ("suspension", ["영업정지"]),
    # 채권은행 관리 (워크아웃)
    ("workout", ["채권은행 등의 관리절차 개시",
                 "채권금융기관 공동관리절차 개시",
                 "채권은행등의관리절차개시",
                 "채권금융기관공동관리절차개시",
                 "공동관리절차 개시",
                 "관리절차개시 신청",
                 "관리절차 개시신청",
                 "주채권은행 관리절차"]),
    ("workout_end", ["관리절차 중단", "관리절차중단", "관리절차 종료"]),
    # 상장폐지 (I유형에서만)
    ("delisting", ["상장폐지결정", "상장폐지 결정", "상장폐지", "상장폐지사유"]),
]


def classify_event(report_nm: str) -> str | None:
    """보고서명 → event_type. 매칭 안 되면 None."""
    if not report_nm:
        return None
    nm = report_nm.strip()
    for event_type, keywords in EVENT_PATTERNS:
        for kw in keywords:
            if kw in nm:
                return event_type
    return None


def get_processed_months() -> set[str]:
    """이미 처리한 YYYYMM 집합."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT target FROM collection_log_v2 WHERE step=? AND status='done'",
            (STEP_NAME,),
        ).fetchall()
    return {r[0] for r in rows}


def log_month(month_key: str, status: str, count: int = 0, error: str = None):
    """월 단위 진행 로그."""
    now = datetime.now().isoformat()
    with connect() as conn:
        conn.execute(
            """INSERT INTO collection_log_v2
               (step, target, status, started_at, completed_at, error)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (STEP_NAME, month_key, status, now, now,
             f"events={count}" + (f"; {error}" if error else "")),
        )


def insert_event(corp_code: str, event_type: str, event_date: str,
                 rcept_no: str, rcept_dt: str, report_nm: str,
                 raw: dict) -> bool:
    """crisis_events 삽입. UNIQUE 충돌 시 무시. return: 새로 들어갔는지."""
    now = datetime.now().isoformat()
    severity_map = {
        "default": 5, "dissolution": 5, "rehab": 4,
        "workout": 3, "suspension": 3, "delisting": 4,
        "workout_end": 2, "audit_qualified": 3,
    }
    severity = severity_map.get(event_type, 2)

    with connect() as conn:
        cur = conn.execute(
            """INSERT OR IGNORE INTO crisis_events
               (corp_code, event_type, event_date, rcept_no, rcept_dt,
                report_nm, severity, raw_response, collected_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (corp_code, event_type, event_date, rcept_no, rcept_dt,
             report_nm, severity, json.dumps(raw, ensure_ascii=False), now),
        )
        return cur.rowcount > 0


def fetch_one_month(year: int, month: int, pblntf_ty: str) -> int:
    """
    특정 월(YYYYMM)의 공시를 list API로 페이지네이션하며 가져와
    위기 이벤트만 필터링해서 적재. return: 적재된 이벤트 수.
    """
    bgn_de = f"{year}{month:02d}01"
    # 월말 계산 (단순)
    if month == 12:
        end_de = f"{year}1231"
    else:
        from calendar import monthrange
        last_day = monthrange(year, month)[1]
        end_de = f"{year}{month:02d}{last_day:02d}"

    inserted = 0
    page_no = 1
    page_count = 100  # 페이지당 최대

    while True:
        params = {
            "bgn_de": bgn_de,
            "end_de": end_de,
            "pblntf_ty": pblntf_ty,   # 'B' or 'I'
            "page_count": str(page_count),
            "page_no": str(page_no),
        }
        try:
            data = dart_client.call("list", params=params)
        except DartNoData:
            break
        except DartQuotaExceeded:
            raise

        rows = data.get("list", []) or []
        if not rows:
            break

        for row in rows:
            report_nm = row.get("report_nm", "")
            event_type = classify_event(report_nm)
            if event_type is None:
                continue

            # 상장폐지는 I유형에서만 인정, B유형의 "상장폐지"는 거름
            # (B유형에서 상장폐지 키워드는 거의 안 나오지만 안전장치)
            if event_type == "delisting" and pblntf_ty == "B":
                continue
            # B 유형은 부도/회생/해산/영업정지/워크아웃만
            if pblntf_ty == "I" and event_type not in ("delisting",):
                continue

            corp_code = row.get("corp_code", "")
            rcept_no = row.get("rcept_no", "")
            rcept_dt = row.get("rcept_dt", "")  # YYYYMMDD

            # event_date를 ISO 형식(YYYY-MM-DD)으로 변환
            if rcept_dt and len(rcept_dt) == 8:
                event_date = f"{rcept_dt[:4]}-{rcept_dt[4:6]}-{rcept_dt[6:8]}"
            else:
                event_date = rcept_dt

            if insert_event(corp_code, event_type, event_date,
                            rcept_no, rcept_dt, report_nm, row):
                inserted += 1

        total_count = int(data.get("total_count", 0))
        total_page = int(data.get("total_page", 1))
        if page_no >= total_page:
            break
        page_no += 1

    return inserted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year-from", type=int, default=config.YEAR_FROM)
    parser.add_argument("--year-to", type=int, default=config.YEAR_TO)
    args = parser.parse_args()

    print(f"📊 03 crisis_events 수집 시작")
    print(f"   기간: {args.year_from} ~ {args.year_to}")
    print(f"   대상: B(주요사항보고서) + I(거래소공시)")

    # 작업 목록: (year, month, pblntf_ty)
    targets = []
    for year in range(args.year_from, args.year_to + 1):
        for month in range(1, 13):
            # 미래 월 스킵
            now = datetime.now()
            if year > now.year or (year == now.year and month > now.month):
                continue
            for ty in ["B", "I"]:
                targets.append((year, month, ty))

    # 이미 처리한 월 제외
    processed = get_processed_months()
    pending = [(y, m, ty) for y, m, ty in targets
               if f"{y}{m:02d}-{ty}" not in processed]
    print(f"   전체 작업 단위: {len(targets)} | 남은 작업: {len(pending)}")

    if not pending:
        print("  ✅ 이미 전부 완료됨.")

    total_inserted = 0
    for i, (year, month, ty) in enumerate(pending, 1):
        month_key = f"{year}{month:02d}-{ty}"
        try:
            inserted = fetch_one_month(year, month, ty)
            log_month(month_key, "done", count=inserted)
            total_inserted += inserted
            print(f"  [{i:>3}/{len(pending)}] {year}-{month:02d} ({ty}): "
                  f"새 이벤트 {inserted:>3}개 (누적 {total_inserted:,})")

        except DartQuotaExceeded:
            print(f"\n⛔ 일일 호출 한도 도달")
            print(f"   처리 완료: {i-1}/{len(pending)} 월")
            print(f"   누적 이벤트: {total_inserted:,}")
            print(f"   자정 이후 똑같이 다시 실행하면 이어서 진행됨")
            return

        except Exception as e:
            log.error(f"  {month_key}: 예외 {e}")
            log_month(month_key, "failed", error=str(e))

    # 최종 통계
    print(f"\n✅ 03_crisis_events.py 완료")
    print(f"\n📊 crisis_events 테이블 통계:")
    with connect() as conn:
        total = conn.execute("SELECT COUNT(*) FROM crisis_events").fetchone()[0]
        print(f"  전체 이벤트: {total:,}")

        print(f"\n  event_type 분포:")
        rows = conn.execute(
            "SELECT event_type, COUNT(*) FROM crisis_events GROUP BY event_type ORDER BY COUNT(*) DESC"
        ).fetchall()
        for et, cnt in rows:
            print(f"    {et:20s} {cnt:>6,}")

        print(f"\n  연도별 분포:")
        rows = conn.execute(
            "SELECT SUBSTR(event_date, 1, 4) AS y, COUNT(*) "
            "FROM crisis_events GROUP BY y ORDER BY y"
        ).fetchall()
        for y, cnt in rows:
            print(f"    {y}: {cnt:>5,}")

        unique_corps = conn.execute(
            "SELECT COUNT(DISTINCT corp_code) FROM crisis_events"
        ).fetchone()[0]
        print(f"\n  영향받은 고유 기업 수: {unique_corps:,}")

        # 학습 대상(is_active=1)과 매칭되는 corp_code 수
        in_universe = conn.execute("""
            SELECT COUNT(DISTINCT ce.corp_code)
            FROM crisis_events ce
            JOIN companies c ON ce.corp_code = c.corp_code
            WHERE c.is_active = 1
        """).fetchone()[0]
        print(f"  그 중 학습 대상(is_active=1) 기업: {in_universe:,}")


if __name__ == "__main__":
    main()