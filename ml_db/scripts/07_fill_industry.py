# -*- coding: utf-8 -*-
"""
07_fill_industry.py
DART company.json 으로 companies 테이블의
  industry_code / industry_name / corp_cls 채우기.

- 호출 대상: is_active=1 (학습 대상) 만. 기업당 1콜.
- 이미 채워진 기업은 건너뜀 (재시작 안전).
- DART 한도(020) 만나면 깔끔히 멈춤 -> 자정 후 같은 명령어로 재개.
- 약 4,657콜 / 0.1초 간격 -> 10~15분 예상.
"""

import sqlite3
import time
import sys
from pathlib import Path

import requests

# ============================ CONFIG ============================
# config.py 가 같은 lib 경로에 있으면 거기서 키 가져오고, 없으면 여기 직접 넣어
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import DART_API_KEY  # noqa
except Exception:
    DART_API_KEY = "여기에_DART_API_KEY_입력"   # config import 실패 시 직접 기입

DB = Path(__file__).resolve().parent.parent / "db" / "dart_v2.db"
URL = "https://opendart.fss.or.kr/api/company.json"
INTERVAL = 0.10          # 초당 약 10콜 (config 의 REQUEST_INTERVAL_SEC 와 동일)
TIMEOUT = 10
MAX_RETRY = 3            # 일시적 네트워크 오류 재시도


def fetch_company(corp_code):
    """company.json 1건 호출. (status, induty_code, induty, corp_cls) 반환."""
    params = {"crtfc_key": DART_API_KEY, "corp_code": corp_code}
    for attempt in range(MAX_RETRY):
        try:
            r = requests.get(URL, params=params, timeout=TIMEOUT)
            j = r.json()
        except Exception as e:
            if attempt == MAX_RETRY - 1:
                return ("EXC", None, None, None)
            time.sleep(0.5)
            continue
        status = j.get("status")
        if status == "000":
            return ("000",
                    j.get("induty_code"),   # KSIC 코드
                    j.get("induty"),        # 업종명 (없을 수 있음)
                    j.get("corp_cls"))      # Y/K/N/E
        # 013 = 조회된 데이터 없음 (비상장 등) -> 정상적으로 빈 응답
        return (status, None, None, None)
    return ("EXC", None, None, None)


def main():
    if "여기에" in DART_API_KEY:
        print("!! DART_API_KEY 가 설정 안 됨. config.py import 실패 시 코드 상단에 직접 입력.")
        return

    con = sqlite3.connect(DB)
    cur = con.cursor()

    # 채울 대상: is_active=1 이면서 industry_code 가 비어있는 기업
    targets = cur.execute(
        "SELECT corp_code, corp_name FROM companies "
        "WHERE is_active=1 "
        "  AND (industry_code IS NULL OR industry_code='')"
    ).fetchall()

    total_active = cur.execute(
        "SELECT COUNT(*) FROM companies WHERE is_active=1").fetchone()[0]
    print(f"학습 대상(is_active=1): {total_active:,}")
    print(f"채울 대상(미충족)     : {len(targets):,}")
    if not targets:
        print("이미 전부 채워짐. 종료.")
        con.close()
        return

    done = 0
    no_data = 0       # status 013 등 데이터 없음
    err = 0
    t0 = time.time()

    for i, (cc, name) in enumerate(targets, 1):
        status, icode, iname, ccls = fetch_company(cc)

        if status == "020":
            # 일일 한도 -> 여기까지 커밋하고 깔끔히 멈춤
            con.commit()
            print(f"\n[중단] DART 일일 한도(020) 도달. {done:,}건 저장됨.")
            print("자정(KST) 이후 같은 명령어로 재개하세요.")
            con.close()
            return

        if status == "000":
            cur.execute(
                "UPDATE companies SET industry_code=?, industry_name=?, corp_cls=? "
                "WHERE corp_code=?",
                (icode, iname, ccls, cc))
            done += 1
        elif status in ("013",):     # 데이터 없음 (정상) -> 빈 값으로 마킹해 재호출 방지
            cur.execute(
                "UPDATE companies SET industry_code='', industry_name='' "
                "WHERE corp_code=? AND (industry_code IS NULL)",
                (cc,))
            no_data += 1
        else:
            err += 1

        if i % 200 == 0:
            con.commit()
            rate = i / (time.time() - t0)
            eta = (len(targets) - i) / rate if rate > 0 else 0
            print(f"  {i:,}/{len(targets):,} | done {done:,} "
                  f"no_data {no_data:,} err {err:,} | "
                  f"{rate:.1f}콜/s | ETA {eta/60:.1f}분")

        time.sleep(INTERVAL)

    con.commit()

    # 최종 요약
    print("\n" + "=" * 55)
    print(f"완료: 호출 {len(targets):,} | 성공 {done:,} | "
          f"데이터없음 {no_data:,} | 에러 {err:,}")

    filled = cur.execute(
        "SELECT COUNT(*) FROM companies "
        "WHERE is_active=1 AND industry_code IS NOT NULL AND industry_code!=''"
    ).fetchone()[0]
    print(f"산업코드 채워진 학습대상: {filled:,}/{total_active:,}")

    # corp_cls 분포
    print("\ncorp_cls 분포 (학습대상):")
    for ccls, n in cur.execute(
            "SELECT corp_cls, COUNT(*) FROM companies "
            "WHERE is_active=1 GROUP BY corp_cls ORDER BY COUNT(*) DESC"):
        print(f"  {ccls}: {n:,}")

    # 금융업 미리보기 (KSIC 64/65/66)
    fin = cur.execute(
        "SELECT COUNT(*) FROM companies WHERE is_active=1 "
        "AND substr(industry_code,1,2) IN ('64','65','66')").fetchone()[0]
    print(f"\n금융업(KSIC 64/65/66) 추정: {fin:,}  <- 06 재실행 시 제대로 잡힘")

    con.close()
    print("\n다음: python 06_financial_calc.py  재실행 -> 금융업 분류 반영")


if __name__ == "__main__":
    main()