import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "db" / "dart_v2.db"
con = sqlite3.connect(DB)
cur = con.cursor()

print("rcept_no 샘플 (연간/반기 각각):")
print("="*60)

for code, name in [("11011", "연간"), ("11012", "반기")]:
    print(f"\n[{name} 보고서 {code}]")
    rows = cur.execute(
        "SELECT DISTINCT bsns_year, rcept_no FROM financial_raw "
        "WHERE reprt_code=? AND rcept_no IS NOT NULL "
        "ORDER BY bsns_year LIMIT 5", (code,)).fetchall()
    for byear, rno in rows:
        rno = str(rno)
        date_part = rno[:8] if len(rno) >= 8 else "???"
        print(f"  bsns_year={byear}  rcept_no={rno}  "
              f"(길이 {len(rno)})  → 공시일 추정 {date_part}")

# rcept_no 길이 분포 (전부 14자리인지)
print("\n" + "="*60)
print("rcept_no 길이 분포:")
for length, n in cur.execute(
        "SELECT LENGTH(rcept_no), COUNT(*) FROM financial_raw "
        "WHERE rcept_no IS NOT NULL GROUP BY LENGTH(rcept_no)"):
    print(f"  {length}자리: {n:,}개")

# 반기 공시 월 분포 (8월쯤 맞는지)
print("\n" + "="*60)
print("반기보고서 공시 월(MM) 분포:")
for mm, n in cur.execute(
        "SELECT substr(rcept_no,5,2), COUNT(DISTINCT corp_code||bsns_year) "
        "FROM financial_raw WHERE reprt_code='11012' AND rcept_no IS NOT NULL "
        "GROUP BY substr(rcept_no,5,2) ORDER BY substr(rcept_no,5,2)"):
    print(f"  {mm}월: {n:,} 건")

con.close()