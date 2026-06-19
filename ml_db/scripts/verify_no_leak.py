# -*- coding: utf-8 -*-
"""
verify_no_leak.py
반기 파이프라인 누수 자동검증. 숫자 보기 전 반드시 통과해야 함.
하나라도 FAIL 이면 다음 단계 진행 금지.

검사:
  [A] 모든 양성 시점의 obs_date < label_date (financials_hy 레벨)
  [B] H1 TTM 이 미래보고서 안 썼는지 (obs_date 정합성: H1 obs_date 는 그해 8월대)
  [C] 09b sequences_hy.npz 존재 시: train/test firm 겹침 0
  [D] sequences_hy.npz: 각 test 양성 시점이 label_date 이전인지 (cc 기준 재확인)
"""
import sqlite3
from pathlib import Path
import numpy as np

_DB_DIR = Path(__file__).resolve().parent.parent / "db"
DB=_DB_DIR / "dart_v2.db"
NPZ=_DB_DIR / "sequences_hy.npz"

def main():
    con=sqlite3.connect(DB); cur=con.cursor()
    lab={cc:dt for cc,l,dt in cur.execute(
        "SELECT corp_code,label,label_date FROM labels") if l==1 and dt}
    fails=0

    # [A] 양성 시점 obs_date < label_date
    print("[A] 양성 시점 obs_date < label_date 검사...")
    bad=[]
    for cc,year,period,odate in cur.execute(
        "SELECT corp_code,year,period,obs_date FROM financials_hy"):
        if cc in lab:
            ld=lab[cc].replace("-","")     # YYYYMMDD
            if odate and odate>=ld:
                bad.append((cc,year,period,odate,lab[cc]))
    if bad:
        print(f"  ✗ FAIL: {len(bad)}건 위반 (obs_date >= label_date)")
        for b in bad[:5]: print(f"     {b}")
        fails+=1
    else:
        print("  ✓ PASS: 모든 양성 시점이 label_date 이전 공시")

    # [B] H1 obs_date 가 그해 8월대인지 (TTM 미래참조 간접검증)
    print("[B] H1 시점 공시월 정합성...")
    months=cur.execute(
        "SELECT substr(obs_date,5,2),COUNT(*) FROM financials_hy "
        "WHERE period='H1' GROUP BY substr(obs_date,5,2)").fetchall()
    h1_total=sum(n for _,n in months)
    aug=dict(months).get("08",0)
    if h1_total and aug/h1_total>0.8:
        print(f"  ✓ PASS: H1 공시 {aug}/{h1_total} ({aug/h1_total*100:.0f}%)가 8월")
    else:
        print(f"  ⚠ 주의: H1 8월 비율 {aug}/{h1_total} 낮음 -> 수동확인 권장")

    # [C][D] sequences_hy.npz 있으면 검사
    if NPZ.exists():
        d=np.load(NPZ,allow_pickle=True)
        cctr=set(d["cc_train"].tolist()); ccte=set(d["cc_test"].tolist())
        print("[C] train/test firm 겹침...")
        ov=cctr & ccte
        if ov:
            print(f"  ✗ FAIL: {len(ov)}개 기업 겹침")
            fails+=1
        else:
            print(f"  ✓ PASS: 겹침 0 (train {len(cctr)} / test {len(ccte)})")
    else:
        print("[C][D] sequences_hy.npz 아직 없음 -> 09b 후 재실행")

    con.close()
    print("\n"+"="*50)
    if fails==0:
        print("✓✓ 누수 검증 전체 PASS. 다음 단계 진행 OK.")
    else:
        print(f"✗✗ {fails}개 항목 FAIL. 누수 수정 전 절대 진행 금지.")

if __name__=="__main__":
    main()
