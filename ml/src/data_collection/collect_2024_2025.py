# temp_collect_2024_2025.py
import sqlite3
import requests
import time

DB_PATH = "db/bankruptcy_prediction.db"
API_KEY = "926f5be1959376e863235e6be6868be04c55b1eb"  # ← 수정

START_YEAR = 2024
END_YEAR   = 2025
DELAY      = 0.4

def fetch_financial(corp_code, year, fs_div):
    url = "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json"
    params = {
        "crtfc_key":  API_KEY,
        "corp_code":  corp_code,
        "bsns_year":  str(year),
        "reprt_code": "11011",
        "fs_div":     fs_div,
    }
    try:
        res = requests.get(url, params=params, timeout=15)
        return res.json()
    except:
        return {"status": "ERR"}

def log_result(cur, corp_code, year, status):
    cur.execute("""
        INSERT OR REPLACE INTO collection_log
        (corp_code, year, status, collected_at)
        VALUES (?, ?, ?, datetime('now'))
    """, (corp_code, year, status))

def insert_raw(cur, corp_code, year, fs_div, data_list):
    for item in data_list:
        cur.execute("""
            INSERT OR IGNORE INTO financial_raw
            (corp_code, corp_name, year, report_code, fs_div,
             sj_div, sj_nm, account_nm, account_id,
             thstrm_amount, frmtrm_amount, currency)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            corp_code,
            item.get("corp_name", ""),
            year, "11011", fs_div,
            item.get("sj_div", ""),
            item.get("sj_nm", ""),
            item.get("account_nm", ""),
            item.get("account_id", ""),
            item.get("thstrm_amount", ""),
            item.get("frmtrm_amount", ""),
            item.get("currency", "KRW"),
        ))

conn = sqlite3.connect(DB_PATH)
cur  = conn.cursor()

# 전체 기업 목록
cur.execute("SELECT corp_code FROM companies")
corps = [r[0] for r in cur.fetchall()]

# 이미 수집된 항목 확인
cur.execute("""
    SELECT corp_code, year FROM collection_log
    WHERE status IN ('SUCCESS', 'NO_DATA')
    AND year IN (2024, 2025)
""")
done = set((r[0], r[1]) for r in cur.fetchall())

todo = [(c, y) for c in corps
        for y in range(START_YEAR, END_YEAR + 1)
        if (c, y) not in done]

print(f"수집 대상: {len(todo)}건 (이미 완료: {len(done)}건)")
print(f"예상 시간: 약 {len(todo)*DELAY/60:.0f}분\n")

success, no_data = 0, 0

for i, (corp_code, year) in enumerate(todo):
    if i % 10 == 0:
        print(f"진행: {i}/{len(todo)} ({i/len(todo)*100:.1f}%) "
              f"| 성공: {success} | 없음: {no_data}")
        conn.commit()

    # 연결재무제표 우선
    data = fetch_financial(corp_code, year, "CFS")
    time.sleep(DELAY)

    if data.get("status") == "000" and data.get("list"):
        insert_raw(cur, corp_code, year, "CFS", data["list"])
        log_result(cur, corp_code, year, "SUCCESS")
        success += 1
    else:
        # 별도재무제표 시도
        data2 = fetch_financial(corp_code, year, "OFS")
        time.sleep(DELAY)
        if data2.get("status") == "000" and data2.get("list"):
            insert_raw(cur, corp_code, year, "OFS", data2["list"])
            log_result(cur, corp_code, year, "SUCCESS")
            success += 1
        else:
            log_result(cur, corp_code, year, "NO_DATA")
            no_data += 1

conn.commit()
conn.close()

print(f"\n=== 완료 ===")
print(f"성공: {success}건 / 데이터없음: {no_data}건")
print(f"다음: calc_financials.py 재실행으로 재무비율 계산")