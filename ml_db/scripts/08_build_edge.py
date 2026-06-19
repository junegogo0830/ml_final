# -*- coding: utf-8 -*-
"""
08_build_edges.py
연도별 스냅샷 그래프 엣지 구성 (GraphSAGE 입력용).

엣지 3종 (모두 같은 연도 안에서만 = 시점 누수 방지):
  1. same_industry        : industry_code 동일. 군집 크면 top-k 제한
  2. financial_similarity : 19개 비율 벡터 z-표준화 후 코사인 top-k
  3. size_proximity       : log(자산총계) 최근접 top-k

핵심 누수 방지:
  - 표준화(mean/std)는 "그 연도 노드들"로만 fit. 미래 정보 안 섞임.
  - 엣지는 같은 연도 노드 사이에만. cross-year 엣지 없음.
  - 노드 = good/partial/distressed (financial_sector 제외).

저장:
  graph_edges 테이블 (year, src, dst, edge_type, weight) - 무방향이라 양방향 저장
  graph_nodes 테이블 (year, corp_code, node_idx) - 연도별 노드 인덱스 매핑
"""

import sqlite3
import math
from pathlib import Path
from collections import defaultdict

import numpy as np

# ============================ CONFIG ============================
DB = Path(__file__).resolve().parent.parent / "db" / "dart_v2.db"
TOPK_FIN = 10        # financial_similarity top-k
TOPK_SIZE = 10       # size_proximity top-k
TOPK_IND = 10        # same_industry 군집이 클 때 노드당 최대 연결 (랜덤/근접 제한)
YEAR_FROM, YEAR_TO = 2015, 2025

# financial_similarity 에 쓸 비율 컬럼 (19개 중 결측 적고 의미 있는 것)
RATIO_COLS = [
    "debt_ratio", "equity_ratio", "debt_to_assets", "noncurrent_liab_ratio",
    "current_ratio", "quick_ratio", "cash_ratio", "roa", "roe",
    "operating_margin", "net_margin", "gross_margin", "asset_turnover",
    "inventory_turnover", "interest_coverage_proxy", "ocf_to_current_liab",
    "retained_earnings_ratio", "working_capital_ratio",
]  # z_score 는 위 비율들의 합성이라 코사인에서 제외 (중복)


def winsorize(arr, p=0.01):
    """극단치 클리핑 (코사인/표준화 안정화). 열별."""
    lo = np.nanpercentile(arr, p * 100, axis=0)
    hi = np.nanpercentile(arr, (1 - p) * 100, axis=0)
    return np.clip(arr, lo, hi)


def topk_cosine(mat, k):
    """행별 코사인 top-k 이웃 인덱스. 자기 자신 제외.
    mat: (N, D) 이미 표준화됨. 반환: list of (i, j) 무방향 후보."""
    # L2 정규화 -> 내적 = 코사인
    norm = np.linalg.norm(mat, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    unit = mat / norm
    sim = unit @ unit.T            # (N, N)
    np.fill_diagonal(sim, -np.inf)  # 자기 자신 제외
    n = mat.shape[0]
    kk = min(k, n - 1)
    edges = set()
    for i in range(n):
        # 상위 kk 이웃
        idx = np.argpartition(sim[i], -kk)[-kk:]
        for j in idx:
            if sim[i, j] == -np.inf:
                continue
            a, b = (i, int(j)) if i < j else (int(j), i)
            edges.add((a, b, float(sim[i, j])))
    return edges


def topk_size(log_assets, k):
    """log 자산 1차원 최근접 top-k. 정렬 기반 O(N log N)."""
    n = len(log_assets)
    order = np.argsort(log_assets)
    edges = set()
    kk = min(k, n - 1)
    for pos in range(n):
        i = order[pos]
        # 양옆으로 kk/2 씩 (가장 가까운 규모)
        lo = max(0, pos - kk)
        hi = min(n, pos + kk + 1)
        for pos2 in range(lo, hi):
            if pos2 == pos:
                continue
            j = order[pos2]
            a, b = (int(i), int(j)) if i < j else (int(j), int(i))
            w = 1.0 / (1.0 + abs(log_assets[i] - log_assets[j]))
            edges.add((a, b, float(w)))
    # 각 노드 degree 가 kk 넘을 수 있어 -> 그대로 둠 (size 는 약한 엣지)
    return edges


def main():
    con = sqlite3.connect(DB)
    cur = con.cursor()

    # 산업코드 (same_industry 용)
    ind = {}
    for cc, icode in cur.execute(
            "SELECT corp_code, industry_code FROM companies "
            "WHERE is_active=1 AND industry_code IS NOT NULL AND industry_code!=''"):
        ind[cc] = icode

    # 테이블 재생성
    cur.execute("DROP TABLE IF EXISTS graph_edges")
    cur.execute("DROP TABLE IF EXISTS graph_nodes")
    cur.execute("""CREATE TABLE graph_edges (
        year INTEGER, src INTEGER, dst INTEGER,
        edge_type TEXT, weight REAL)""")
    cur.execute("""CREATE TABLE graph_nodes (
        year INTEGER, corp_code TEXT, node_idx INTEGER,
        PRIMARY KEY (year, corp_code))""")

    col_sql = ", ".join(RATIO_COLS)
    total_edges = defaultdict(int)

    for year in range(YEAR_FROM, YEAR_TO + 1):
        # 그 연도 노드: 금융업 제외, 자산총계 있어야 size 엣지 가능
        rows = cur.execute(f"""
            SELECT f.corp_code, {col_sql}, f.z_score
            FROM financials f
            WHERE f.year=? AND f.data_quality != 'financial_sector'
        """, (year,)).fetchall()
        if len(rows) < 5:
            continue

        corps = [r[0] for r in rows]
        n = len(corps)
        # 비율 행렬 (N, D). None -> nan
        D = len(RATIO_COLS)
        mat = np.full((n, D), np.nan)
        for i, r in enumerate(rows):
            for d in range(D):
                v = r[1 + d]
                if v is not None:
                    mat[i, d] = v

        # --- 표준화: 그 연도 노드로만 (누수 방지) ---
        mat = winsorize(mat, 0.01)
        col_mean = np.nanmean(mat, axis=0)
        col_std = np.nanstd(mat, axis=0)
        col_std[col_std == 0] = 1.0
        # nan -> 평균(=표준화 후 0)
        inds = np.where(np.isnan(mat))
        mat[inds] = np.take(col_mean, inds[1])
        mat_std = (mat - col_mean) / col_std

        # 노드 인덱스 저장
        cur.executemany(
            "INSERT INTO graph_nodes VALUES (?,?,?)",
            [(year, corps[i], i) for i in range(n)])

        # --- 1) financial_similarity ---
        fin_edges = topk_cosine(mat_std, TOPK_FIN)
        for (a, b, w) in fin_edges:
            # 무방향 -> 양방향 저장 (GraphSAGE 메시지패싱)
            cur.execute("INSERT INTO graph_edges VALUES (?,?,?,?,?)",
                        (year, a, b, "financial_similarity", w))
            cur.execute("INSERT INTO graph_edges VALUES (?,?,?,?,?)",
                        (year, b, a, "financial_similarity", w))
        total_edges["financial_similarity"] += len(fin_edges)

        # --- 2) size_proximity ---
        # 자산총계 = z_score 컬럼엔 없음 -> financials 에서 따로? 여기선 비율만 있어서
        # working_capital 등으로 못 구함. 자산총계 직접 조회 필요.
        # -> financial_raw 대신 별도 조회
        size_rows = cur.execute("""
            SELECT fr.corp_code, fr.thstrm_amount
            FROM financial_raw fr
            WHERE fr.bsns_year=? AND fr.reprt_code='11011'
              AND (fr.account_id='ifrs-full_Assets' OR fr.account_nm='자산총계')
        """, (year,)).fetchall()
        asset_map = {}
        for cc, amt in size_rows:
            if cc in asset_map:
                continue
            try:
                a = float(str(amt).replace(",", ""))
                if a > 0:
                    asset_map[cc] = math.log(a)
            except (ValueError, TypeError):
                pass
        # 현재 연도 노드 중 자산 있는 것만
        size_idx = [i for i in range(n) if corps[i] in asset_map]
        if len(size_idx) >= 5:
            log_a = np.array([asset_map[corps[i]] for i in size_idx])
            sub_edges = topk_size(log_a, TOPK_SIZE)
            for (pa, pb, w) in sub_edges:
                a, b = size_idx[pa], size_idx[pb]
                cur.execute("INSERT INTO graph_edges VALUES (?,?,?,?,?)",
                            (year, a, b, "size_proximity", w))
                cur.execute("INSERT INTO graph_edges VALUES (?,?,?,?,?)",
                            (year, b, a, "size_proximity", w))
            total_edges["size_proximity"] += len(sub_edges)

        # --- 3) same_industry ---
        # 같은 industry_code 끼리. 군집 크면 노드당 TOPK_IND 로 제한.
        by_ind = defaultdict(list)
        for i in range(n):
            ic = ind.get(corps[i])
            if ic:
                by_ind[ic].append(i)
        ind_edges = set()
        for ic, members in by_ind.items():
            m = len(members)
            if m < 2:
                continue
            if m <= TOPK_IND + 1:
                # 작은 군집 -> 완전 연결
                for x in range(m):
                    for y in range(x + 1, m):
                        ind_edges.add((members[x], members[y]))
            else:
                # 큰 군집 -> 각 노드를 같은 군집 내 TOPK_IND 개와만 (순환 연결)
                for x in range(m):
                    for off in range(1, TOPK_IND + 1):
                        y = members[(x + off) % m]
                        a, b = (members[x], y) if members[x] < y else (y, members[x])
                        ind_edges.add((a, b))
        for (a, b) in ind_edges:
            cur.execute("INSERT INTO graph_edges VALUES (?,?,?,?,?)",
                        (year, a, b, "same_industry", 1.0))
            cur.execute("INSERT INTO graph_edges VALUES (?,?,?,?,?)",
                        (year, b, a, "same_industry", 1.0))
        total_edges["same_industry"] += len(ind_edges)

        con.commit()
        print(f"  {year}: 노드 {n:,} | "
              f"fin {len(fin_edges):,} size {len(size_idx) and '~'} ind {len(ind_edges):,}")

    # 인덱스
    cur.execute("CREATE INDEX idx_edges_year ON graph_edges(year, edge_type)")
    con.commit()

    print("\n" + "=" * 55)
    print("엣지 종류별 총계 (무방향 기준, 저장은 양방향 2배):")
    for et, c in total_edges.items():
        print(f"  {et:22s}: {c:,}")

    # 연도별 노드/엣지 요약
    print("\n연도별 노드 수:")
    for year, n in cur.execute(
            "SELECT year, COUNT(*) FROM graph_nodes GROUP BY year ORDER BY year"):
        e = cur.execute("SELECT COUNT(*) FROM graph_edges WHERE year=?",
                        (year,)).fetchone()[0]
        print(f"  {year}: 노드 {n:,} / 엣지(방향) {e:,}")

    # 고립 노드 체크 (엣지 0개인 노드 = GraphSAGE 에서 이웃 없음)
    iso = cur.execute("""
        SELECT COUNT(*) FROM graph_nodes gn
        WHERE NOT EXISTS (
            SELECT 1 FROM graph_edges ge
            WHERE ge.year=gn.year AND ge.src=gn.node_idx)
    """).fetchone()[0]
    print(f"\n고립 노드(이웃 0): {iso:,}  <- 많으면 top-k 늘리거나 self-loop 필요")

    con.close()
    print("\n완료. 다음: 시계열+GraphSAGE 학습 데이터 빌드.")


if __name__ == "__main__":
    main()