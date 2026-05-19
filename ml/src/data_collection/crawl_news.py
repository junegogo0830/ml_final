"""
네이버 뉴스 크롤링 - 하루 25,000건 제한 자동 관리
실행: python crawl_news.py

사전 준비:
  1. https://developers.naver.com 에서 앱 등록
  2. 검색 API 사용 신청
  3. CLIENT_ID, CLIENT_SECRET 입력

중단 후 재실행하면 이어서 수집
"""

import requests
import sqlite3
import json
import time
import os
from datetime import datetime

# ── 설정 ──────────────────────────────────────────────────────────────
CLIENT_ID     = "CamjY8un7n0DeCdWYl7F"      # ← 수정 필수
CLIENT_SECRET = "ihgDPnswTV"  # ← 수정 필수
DB_PATH       = "db/bankruptcy_prediction.db"

ARTICLES_PER_CORP = 100   # 기업당 수집 건수
DAILY_LIMIT       = 24000 # 하루 제한 (25,000에서 여유분 제외)
DELAY             = 0.1   # API 호출 간 딜레이 (초)


# ── DB 초기화 ─────────────────────────────────────────────────────────
def init_news_table():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 뉴스 원본 테이블
    cur.execute("""
        CREATE TABLE IF NOT EXISTS news_raw (
            corp_code   TEXT,
            corp_name   TEXT,
            title       TEXT,
            description TEXT,
            pub_date    TEXT,
            link        TEXT,
            collected_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (corp_code, link)
        )
    """)

    # 수집 로그 테이블
    cur.execute("""
        CREATE TABLE IF NOT EXISTS news_collection_log (
            corp_code    TEXT PRIMARY KEY,
            corp_name    TEXT,
            collected    INTEGER DEFAULT 0,
            status       TEXT DEFAULT 'PENDING',
            collected_at TEXT
        )
    """)

    # 일일 수집량 추적 테이블
    cur.execute("""
        CREATE TABLE IF NOT EXISTS news_daily_log (
            date         TEXT PRIMARY KEY,
            total_count  INTEGER DEFAULT 0
        )
    """)

    conn.commit()
    conn.close()
    print("뉴스 테이블 초기화 완료")


# ── 오늘 수집량 확인 ──────────────────────────────────────────────────
def get_today_count():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    today = datetime.now().strftime("%Y-%m-%d")
    cur.execute("SELECT total_count FROM news_daily_log WHERE date = ?", (today,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else 0


def update_today_count(count):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    today = datetime.now().strftime("%Y-%m-%d")
    cur.execute("""
        INSERT INTO news_daily_log (date, total_count)
        VALUES (?, ?)
        ON CONFLICT(date) DO UPDATE SET total_count = total_count + ?
    """, (today, count, count))
    conn.commit()
    conn.close()


# ── 수집 대상 기업 로드 ───────────────────────────────────────────────
def load_target_corps():
    conn = sqlite3.connect(DB_PATH)

    # 파산 기업 전체
    bankrupt = conn.execute("""
        SELECT DISTINCT c.corp_code, c.corp_name
        FROM companies c
        JOIN bankruptcy b ON c.corp_code = b.corp_code
        WHERE c.corp_name IS NOT NULL
    """).fetchall()

    # 정상 기업 300개 (코스피+코스닥, 재무제표 있는 것)
    normal = conn.execute("""
        SELECT DISTINCT c.corp_code, c.corp_name
        FROM companies c
        LEFT JOIN bankruptcy b ON c.corp_code = b.corp_code
        JOIN financials f ON c.corp_code = f.corp_code
        WHERE b.corp_code IS NULL
        AND c.market IN ('K', 'Y')
        AND c.corp_name IS NOT NULL
        LIMIT 300
    """).fetchall()

    conn.close()

    all_corps = list(set(bankrupt + normal))
    print(f"수집 대상: 파산 {len(bankrupt)}개 + 정상 {len(normal)}개 = {len(all_corps)}개")
    return all_corps


# ── 이미 수집된 기업 확인 ────────────────────────────────────────────
def get_collected_corps():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT corp_code FROM news_collection_log WHERE status = 'DONE'")
    done = set(r[0] for r in cur.fetchall())
    conn.close()
    return done


# ── 네이버 뉴스 API 호출 ─────────────────────────────────────────────
def fetch_news(corp_name, display=100):
    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {
        "X-Naver-Client-Id": CLIENT_ID,
        "X-Naver-Client-Secret": CLIENT_SECRET
    }
    params = {
        "query":   corp_name,
        "display": min(display, 100),  # 최대 100건
        "sort":    "date"
    }
    try:
        res = requests.get(url, headers=headers, params=params, timeout=10)
        if res.status_code == 200:
            return res.json().get("items", [])
        else:
            print(f"  API 오류: {res.status_code}")
            return []
    except Exception as e:
        print(f"  요청 실패: {e}")
        return []


# ── 뉴스 DB 저장 ─────────────────────────────────────────────────────
def save_news(corp_code, corp_name, items):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    saved = 0
    for item in items:
        try:
            # HTML 태그 제거
            title = item.get("title", "").replace("<b>", "").replace("</b>", "")
            desc  = item.get("description", "").replace("<b>", "").replace("</b>", "")
            cur.execute("""
                INSERT OR IGNORE INTO news_raw
                (corp_code, corp_name, title, description, pub_date, link)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                corp_code, corp_name,
                title, desc,
                item.get("pubDate", ""),
                item.get("link", "")
            ))
            if cur.rowcount > 0:
                saved += 1
        except Exception:
            continue
    conn.commit()
    conn.close()
    return saved


def mark_done(corp_code, corp_name, count):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT OR REPLACE INTO news_collection_log
        (corp_code, corp_name, collected, status, collected_at)
        VALUES (?, ?, ?, 'DONE', datetime('now'))
    """, (corp_code, corp_name, count))
    conn.commit()
    conn.close()


# ── 메인 수집 루프 ────────────────────────────────────────────────────
def collect():
    init_news_table()

    # 오늘 이미 수집한 양 확인
    today_count = get_today_count()
    if today_count >= DAILY_LIMIT:
        print(f"오늘 수집량 {today_count}건 — 일일 한도 도달. 내일 다시 실행하세요.")
        return

    print(f"오늘 수집량: {today_count}/{DAILY_LIMIT}건")
    remaining_today = DAILY_LIMIT - today_count

    # 수집 대상 기업 로드
    all_corps  = load_target_corps()
    done_corps = get_collected_corps()
    todo_corps = [(cc, cn) for cc, cn in all_corps if cc not in done_corps]

    print(f"완료: {len(done_corps)}개 / 남은 기업: {len(todo_corps)}개\n")

    if not todo_corps:
        print("모든 기업 수집 완료!")
        return

    total_saved   = 0
    today_saved   = 0

    for i, (corp_code, corp_name) in enumerate(todo_corps):

        # 오늘 한도 체크
        if today_saved >= remaining_today:
            print(f"\n오늘 한도 도달 ({DAILY_LIMIT}건). 내일 다시 실행하세요.")
            print(f"오늘 수집: {today_saved}건 / 전체 누적: {total_saved}건")
            break

        print(f"[{i+1}/{len(todo_corps)}] {corp_name} 수집 중...")

        items = fetch_news(corp_name, ARTICLES_PER_CORP)
        time.sleep(DELAY)

        if items:
            saved = save_news(corp_code, corp_name, items)
            mark_done(corp_code, corp_name, saved)
            total_saved  += saved
            today_saved  += len(items)
            print(f"  → {saved}건 저장")
        else:
            mark_done(corp_code, corp_name, 0)
            print(f"  → 뉴스 없음")

        # 50개마다 진행 상황 출력
        if (i + 1) % 50 == 0:
            update_today_count(today_saved)
            print(f"\n진행: {i+1}/{len(todo_corps)} | "
                  f"오늘 수집: {today_saved}건 | "
                  f"남은 한도: {remaining_today - today_saved}건\n")

    # 최종 저장
    update_today_count(today_saved)

    print(f"\n=== 오늘 수집 완료 ===")
    print(f"오늘 수집: {today_saved}건")
    print(f"전체 저장: {total_saved}건")
    print(f"완료 기업: {len(done_corps) + (i+1)}개 / {len(all_corps)}개")

    # 진행 현황
    conn = sqlite3.connect(DB_PATH)
    done = conn.execute("SELECT COUNT(*) FROM news_collection_log WHERE status='DONE'").fetchone()[0]
    total_news = conn.execute("SELECT COUNT(*) FROM news_raw").fetchone()[0]
    conn.close()
    print(f"누적 완료 기업: {done}개 / 총 뉴스: {total_news}건")
    print(f"\n내일 다시 실행하면 이어서 수집합니다.")


if __name__ == "__main__":
    if CLIENT_ID == "네이버_클라이언트_ID":
        print("CLIENT_ID와 CLIENT_SECRET을 입력해주세요.")
        print("https://developers.naver.com 에서 앱 등록 후 검색 API 신청")
        exit()

    print("=" * 50)
    print("네이버 뉴스 크롤링 시작")
    print("=" * 50)
    collect()