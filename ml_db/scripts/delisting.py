"""
labels.py: 라벨 테이블(labels) 생성 + 분포 분석.

라벨 규칙:
- 양성(label=1):
    (A) event_type IN ('default','rehab','dissolution','suspension','workout')
    OR
    (B) event_type='delisting' AND 같은 corp_code가 (A) 이벤트도 있음

- label_date: 한 기업의 모든 양성 이벤트 중 가장 빠른 날짜

- 음성(label=0): is_active=1인데 양성 조건 안 맞는 기업

추가 컬럼:
- primary_event_type: severity 가장 높은 이벤트 종류
- event_count: 위기 이벤트 발생 횟수
- has_delisting: 위기 동반 delisting이 있었는지 (보조 정보)

실행 (API 호출 0):
    python scripts/labels.py
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.db_helper import connect


CORE_CRISIS_TYPES = ("default", "rehab", "dissolution", "suspension", "workout")


def create_labels_table():
    """labels 테이블 생성 (없으면)."""
    with connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS labels (
                corp_code         TEXT PRIMARY KEY,
                label             INTEGER NOT NULL,    -- 0=정상, 1=위기
                label_date        TEXT,                -- 양성일 때 가장 빠른 위기 발생일
                primary_event_type TEXT,               -- severity 최고 이벤트 (default > dissolution > rehab > ...)
                max_severity      INTEGER,
                event_count       INTEGER DEFAULT 0,
                has_delisting     INTEGER DEFAULT 0,   -- 위기 동반 delisting 있었나
                created_at        TEXT,
                FOREIGN KEY (corp_code) REFERENCES companies(corp_code)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_labels_label ON labels(label)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_labels_date ON labels(label_date)")


def build_labels():
    """라벨 적재. 기존 데이터 있으면 다 지우고 재구성."""
    now = datetime.now().isoformat()

    with connect() as conn:
        # 초기화
        conn.execute("DELETE FROM labels")

        # === 1. 양성 결정 (5개 core 위기 이벤트 가진 기업) ===
        positive_corps = set(r[0] for r in conn.execute(f"""
            SELECT DISTINCT corp_code FROM crisis_events
            WHERE event_type IN {CORE_CRISIS_TYPES}
        """).fetchall())

        print(f"양성 후보 (core 위기 이벤트 가진 기업): {len(positive_corps):,}")

        # === 2. 학습 대상(is_active=1)과 교집합 ===
        active_corps = set(r[0] for r in conn.execute(
            "SELECT corp_code FROM companies WHERE is_active=1"
        ).fetchall())
        print(f"학습 대상 기업: {len(active_corps):,}")

        positive_active = positive_corps & active_corps
        print(f"양성 (학습 대상 ∩ 위기 이벤트): {len(positive_active):,}")

        # === 3. 양성 기업의 라벨 정보 계산 ===
        # 각 양성 기업에 대해:
        # - 가장 빠른 core 위기 event_date → label_date
        # - severity 최고 이벤트 종류 → primary_event_type
        # - 위기 동반 delisting 있는지

        if positive_active:
            placeholders = ",".join("?" * len(positive_active))
            corp_list = list(positive_active)

            # 가장 빠른 core 위기 + severity 정보
            rows = conn.execute(f"""
                SELECT corp_code,
                       MIN(event_date) AS first_date,
                       COUNT(*) AS event_count
                FROM crisis_events
                WHERE event_type IN {CORE_CRISIS_TYPES}
                  AND corp_code IN ({placeholders})
                GROUP BY corp_code
            """, corp_list).fetchall()

            # 각 기업의 가장 severity 높은 이벤트 종류
            severity_rows = conn.execute(f"""
                SELECT corp_code, event_type, severity
                FROM crisis_events
                WHERE event_type IN {CORE_CRISIS_TYPES}
                  AND corp_code IN ({placeholders})
            """, corp_list).fetchall()
            best_severity = {}
            for cc, et, sev in severity_rows:
                if cc not in best_severity or sev > best_severity[cc][1]:
                    best_severity[cc] = (et, sev)

            # delisting 동반 여부
            delisting_corps = set(r[0] for r in conn.execute(f"""
                SELECT DISTINCT corp_code FROM crisis_events
                WHERE event_type='delisting'
                  AND corp_code IN ({placeholders})
            """, corp_list).fetchall())

            # 적재
            for cc, first_date, ev_cnt in rows:
                et, sev = best_severity[cc]
                has_del = 1 if cc in delisting_corps else 0
                conn.execute("""
                    INSERT INTO labels
                    (corp_code, label, label_date, primary_event_type,
                     max_severity, event_count, has_delisting, created_at)
                    VALUES (?, 1, ?, ?, ?, ?, ?, ?)
                """, (cc, first_date, et, sev, ev_cnt, has_del, now))

        # === 4. 음성 적재 ===
        negative_active = active_corps - positive_active
        for cc in negative_active:
            conn.execute("""
                INSERT INTO labels
                (corp_code, label, label_date, primary_event_type,
                 max_severity, event_count, has_delisting, created_at)
                VALUES (?, 0, NULL, NULL, 0, 0, 0, ?)
            """, (cc, now))

        print(f"음성: {len(negative_active):,}")


def print_distribution():
    """라벨 분포 + 시각화."""
    with connect() as conn:
        print()
        print("=" * 70)
        print("라벨 분포")
        print("=" * 70)

        rows = conn.execute("""
            SELECT label, COUNT(*) FROM labels GROUP BY label
        """).fetchall()
        total = sum(c for _, c in rows)
        for lbl, cnt in rows:
            tag = "양성(위기)" if lbl == 1 else "음성(정상)"
            print(f"  label={lbl} {tag:12s} {cnt:>5,}  ({cnt/total*100:.2f}%)")
        print(f"  {'합계':18s} {total:>5,}")

        # 클래스 불균형 비율
        pos = next((c for l, c in rows if l == 1), 0)
        neg = next((c for l, c in rows if l == 0), 0)
        if pos > 0:
            print(f"\n  음성:양성 비율 = {neg/pos:.1f} : 1")

        # 양성 기업의 primary_event_type 분포
        print()
        print("=" * 70)
        print("양성 기업의 primary_event_type 분포 (severity 최고 이벤트)")
        print("=" * 70)
        rows = conn.execute("""
            SELECT primary_event_type, COUNT(*) FROM labels
            WHERE label=1 GROUP BY primary_event_type
            ORDER BY COUNT(*) DESC
        """).fetchall()
        for et, cnt in rows:
            print(f"  {et:20s} {cnt:>5,}")

        # 위기 동반 delisting
        print()
        print("=" * 70)
        print("위기 동반 delisting 보유 양성 기업")
        print("=" * 70)
        with_del = conn.execute(
            "SELECT COUNT(*) FROM labels WHERE label=1 AND has_delisting=1"
        ).fetchone()[0]
        pos_total = conn.execute(
            "SELECT COUNT(*) FROM labels WHERE label=1"
        ).fetchone()[0]
        print(f"  {with_del}/{pos_total}  ({with_del/max(pos_total,1)*100:.1f}%)")

        # 양성 라벨 연도별 분포
        print()
        print("=" * 70)
        print("양성 라벨 연도별 분포 (label_date 기준)")
        print("=" * 70)
        rows = conn.execute("""
            SELECT SUBSTR(label_date, 1, 4) AS y, COUNT(*)
            FROM labels WHERE label=1
            GROUP BY y ORDER BY y
        """).fetchall()
        for y, cnt in rows:
            bar = "█" * min(cnt // 5, 50)
            print(f"  {y}  {cnt:>4}  {bar}")

        # event_count 분포 (한 기업이 여러 번 위기 겪었는지)
        print()
        print("=" * 70)
        print("양성 기업의 위기 이벤트 발생 횟수")
        print("=" * 70)
        rows = conn.execute("""
            SELECT event_count, COUNT(*) FROM labels
            WHERE label=1 GROUP BY event_count ORDER BY event_count
        """).fetchall()
        for ec, cnt in rows:
            print(f"  {ec}회: {cnt:>4}")

        # 상장사 vs 비상장 양성 분포
        print()
        print("=" * 70)
        print("상장/비상장 × 라벨 교차")
        print("=" * 70)
        for lbl in [1, 0]:
            tag = "양성" if lbl == 1 else "음성"
            listed = conn.execute("""
                SELECT COUNT(*) FROM labels l
                JOIN companies c ON l.corp_code = c.corp_code
                WHERE l.label=? AND c.stock_code IS NOT NULL AND c.stock_code!=''
            """, (lbl,)).fetchone()[0]
            unlisted = conn.execute("""
                SELECT COUNT(*) FROM labels l
                JOIN companies c ON l.corp_code = c.corp_code
                WHERE l.label=? AND (c.stock_code IS NULL OR c.stock_code='')
            """, (lbl,)).fetchone()[0]
            print(f"  {tag}  상장 {listed:>4,}  비상장 {unlisted:>4,}")


def main():
    print("📊 labels 테이블 생성 시작")
    print()
    create_labels_table()
    build_labels()
    print_distribution()
    print()
    print("✅ labels 테이블 생성 완료")


if __name__ == "__main__":
    main()