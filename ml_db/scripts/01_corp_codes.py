"""
01_corp_codes.py: DART 등록 전체 기업 corp_code 수집.

DART API의 corpCode.xml은 ZIP 파일로 응답. 압축 해제 후 XML 파싱.
활성 기업뿐 아니라 폐업/말소 기업도 포함됨 (이게 우리가 원하는 것).

수집 항목:
- corp_code (8자리 고유번호)
- corp_name
- corp_eng_name
- stock_code (상장사만)
- modify_date

is_active 판정:
- corp_code.xml은 폐업 여부를 직접 알려주지 않음
- modify_date가 매우 오래된 entity는 폐업 가능성 높음
- 정확한 판정은 02_companies.py에서 기업개황 API로 추가 검증

실행:
    python scripts/01_corp_codes.py
"""

import sys
import io
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib import dart_client
from lib.db_helper import connect, log_step
import config


def download_corp_code_xml() -> bytes:
    """corpCode.xml ZIP 다운로드 → 압축 해제 → XML bytes 반환."""
    print("📥 DART 전체 corp_code 다운로드 중...")
    zip_bytes = dart_client.call("corpCode", params={}, response_type="binary")

    # 응답이 zip인지 확인
    if not zip_bytes[:2] == b"PK":
        # JSON 에러 응답일 수 있음
        try:
            import json
            err = json.loads(zip_bytes.decode("utf-8"))
            raise RuntimeError(f"corpCode 에러: {err}")
        except Exception:
            raise RuntimeError(f"corpCode 응답이 ZIP이 아님: {zip_bytes[:200]}")

    # 캐시로 저장
    cache_path = config.CACHE_DIR / f"corpCode_{datetime.now().strftime('%Y%m%d')}.zip"
    cache_path.write_bytes(zip_bytes)
    print(f"💾 ZIP 캐시 저장: {cache_path}")

    # 압축 해제
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        names = z.namelist()
        print(f"📦 ZIP 내부 파일: {names}")
        xml_name = next(n for n in names if n.lower().endswith(".xml"))
        xml_bytes = z.read(xml_name)

    return xml_bytes


def parse_corp_code_xml(xml_bytes: bytes) -> list[dict]:
    """corpCode.xml 파싱 → list of dict."""
    print("🔍 XML 파싱 중...")
    root = ET.fromstring(xml_bytes)
    # 구조: <result><list><corp_code>...</corp_code><corp_name>...</corp_name>...</list>...</result>
    rows = []
    for item in root.findall("list"):
        rows.append({
            "corp_code": (item.findtext("corp_code") or "").strip(),
            "corp_name": (item.findtext("corp_name") or "").strip(),
            "corp_eng_name": (item.findtext("corp_eng_name") or "").strip() or None,
            "stock_code": (item.findtext("stock_code") or "").strip() or None,
            "modify_date": (item.findtext("modify_date") or "").strip() or None,
        })
    print(f"✅ 파싱 완료: {len(rows):,}개 entity")
    return rows


def insert_companies(rows: list[dict]):
    """companies 테이블에 적재. corp_code 중복은 modify_date 최신으로 갱신."""
    now = datetime.now().isoformat()

    with connect() as conn:
        # 기존 데이터와 비교 위해 dict로 로드
        existing = {
            r[0]: r[1]
            for r in conn.execute("SELECT corp_code, modify_date FROM companies")
        }

        inserted = 0
        updated = 0
        skipped = 0

        for r in rows:
            cc = r["corp_code"]
            if not cc or len(cc) != 8:
                skipped += 1
                continue

            if cc in existing:
                # 기존보다 modify_date가 새로우면 갱신
                old_md = existing[cc]
                new_md = r.get("modify_date") or ""
                if new_md and (not old_md or new_md > old_md):
                    conn.execute(
                        """UPDATE companies
                           SET corp_name=?, corp_eng_name=?, stock_code=?,
                               modify_date=?, collected_at=?
                           WHERE corp_code=?""",
                        (r["corp_name"], r["corp_eng_name"], r["stock_code"],
                         r["modify_date"], now, cc),
                    )
                    updated += 1
            else:
                # corp_cls는 02에서 채움. stock_code 유무로 임시 추정.
                corp_cls = None  # 02에서 보강
                conn.execute(
                    """INSERT INTO companies
                       (corp_code, corp_name, corp_eng_name, stock_code,
                        corp_cls, modify_date, is_active, collected_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (cc, r["corp_name"], r["corp_eng_name"], r["stock_code"],
                     corp_cls, r["modify_date"], 1, now),
                )
                inserted += 1

        print(f"\n📊 적재 결과:")
        print(f"  - 신규 입력: {inserted:,}")
        print(f"  - 갱신: {updated:,}")
        print(f"  - 스킵 (corp_code 형식 오류): {skipped:,}")


def main():
    step = "01_corp_codes"
    target = datetime.now().strftime("%Y%m%d")

    log_step(step, target, "in_progress")
    try:
        xml_bytes = download_corp_code_xml()
        rows = parse_corp_code_xml(xml_bytes)
        insert_companies(rows)

        # 통계
        with connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM companies").fetchone()[0]
            listed = conn.execute(
                "SELECT COUNT(*) FROM companies WHERE stock_code IS NOT NULL AND stock_code != ''"
            ).fetchone()[0]

        print(f"\n📊 최종 companies 테이블:")
        print(f"  - 전체: {total:,}")
        print(f"  - 상장사 (stock_code 보유): {listed:,}")
        print(f"  - 비상장: {total - listed:,}")

        log_step(step, target, "done")
        print("\n✅ 01_corp_codes.py 완료")

    except Exception as e:
        log_step(step, target, "failed", error=str(e))
        print(f"\n❌ 실패: {e}")
        raise


if __name__ == "__main__":
    main()
