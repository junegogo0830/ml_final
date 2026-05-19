"""
relationships 테이블 구성
실행: python build_relationships.py

엣지 유형:
  1. 동일 업종 (KSIC 코드 앞 3자리)
  2. 재무 유사도 (코사인 유사도 >= 0.8)
  3. 동일 규모 (자산 기준 대/중/소)
"""

import sqlite3
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

DB_PATH = "db/bankruptcy_prediction.db"

FEATURE_COLS = [
    'debt_ratio', 'current_ratio', 'interest_coverage',
    'net_debt_ratio', 'equity_ratio',
    'roa', 'roe', 'op_margin', 'net_margin',
    'cfo_to_debt', 'z_score'
]

SIM_THRESHOLD = 0.8   # 재무 유사도 임계값


# ── 1. 데이터 로드 ────────────────────────────────────────────────────
def load_data():
    conn = sqlite3.connect(DB_PATH)

    # 기업 기본정보 (업종코드, 시장)
    companies = pd.read_sql("""
        SELECT corp_code, industry_code, market
        FROM companies
        WHERE market IN ('K', 'Y')
        AND industry_code IS NOT NULL
        AND industry_code != ''
    """, conn)

    # 재무비율 (기업별 평균)
    financials = pd.read_sql(f"""
        SELECT corp_code,
               AVG(total_assets)        as total_assets,
               {', '.join([f'AVG({c}) as {c}' for c in FEATURE_COLS])}
        FROM financials
        GROUP BY corp_code
    """, conn)

    conn.close()

    print(f"기업 정보: {len(companies)}개")
    print(f"재무 데이터: {len(financials)}개")
    return companies, financials


# ── 2. 기업 규모 분류 ─────────────────────────────────────────────────
def classify_size(total_assets):
    if pd.isna(total_assets):
        return None
    if total_assets >= 5000:    # 5,000억 이상
        return "large"
    elif total_assets >= 1000:  # 1,000억~5,000억
        return "medium"
    else:
        return "small"


# ── 3. 엣지 생성 ─────────────────────────────────────────────────────
def build_edges(companies, financials):
    # 데이터 결합
    df = companies.merge(financials, on='corp_code', how='inner')
    df['size'] = df['total_assets'].apply(classify_size)
    df['industry_3'] = df['industry_code'].str[:3]  # 업종코드 앞 3자리

    print(f"\n엣지 구성 대상: {len(df)}개 기업")

    edges = []

    # ── 엣지 1: 동일 업종 ──────────────────────────────────────────────
    print("\n[1] 동일 업종 엣지 생성 중...")
    industry_groups = df.groupby('industry_3')['corp_code'].apply(list)

    for industry, corps in industry_groups.items():
        if len(corps) < 2:
            continue
        for i in range(len(corps)):
            for j in range(i + 1, len(corps)):
                edges.append({
                    'corp_code_from': corps[i],
                    'corp_code_to':   corps[j],
                    'edge_type':      'industry',
                    'weight':         1.0,
                    'year':           0,
                    'detail':         industry
                })
                edges.append({
                    'corp_code_from': corps[j],
                    'corp_code_to':   corps[i],
                    'edge_type':      'industry',
                    'weight':         1.0,
                    'year':           0,
                    'detail':         industry
                })

    print(f"  동일 업종 엣지: {len(edges)}개")

    # ── 엣지 2: 재무 유사도 ────────────────────────────────────────────
    print("\n[2] 재무 유사도 엣지 생성 중...")

    feat_df = df[['corp_code'] + FEATURE_COLS].copy()
    feat_df = feat_df.dropna(thresh=len(FEATURE_COLS) * 0.7)
    feat_df[FEATURE_COLS] = feat_df[FEATURE_COLS].fillna(feat_df[FEATURE_COLS].median())

    corp_codes = feat_df['corp_code'].tolist()
    features   = feat_df[FEATURE_COLS].values

    # 정규화
    from sklearn.preprocessing import StandardScaler
    scaler   = StandardScaler()
    features = scaler.fit_transform(features)

    # 코사인 유사도 계산 (배치 처리)
    sim_edges_before = len(edges)
    batch_size = 200

    for i in range(0, len(corp_codes), batch_size):
        batch_feats  = features[i:i + batch_size]
        sim_matrix   = cosine_similarity(batch_feats, features)

        for bi, gi in enumerate(range(i, min(i + batch_size, len(corp_codes)))):
            for j in range(gi + 1, len(corp_codes)):
                sim = sim_matrix[bi, j]
                if sim >= SIM_THRESHOLD:
                    edges.append({
                        'corp_code_from': corp_codes[gi],
                        'corp_code_to':   corp_codes[j],
                        'edge_type':      'financial_sim',
                        'weight':         round(float(sim), 4),
                        'year':           0,
                        'detail':         f"sim={sim:.3f}"
                    })
                    edges.append({
                        'corp_code_from': corp_codes[j],
                        'corp_code_to':   corp_codes[gi],
                        'edge_type':      'financial_sim',
                        'weight':         round(float(sim), 4),
                        'year':           0,
                        'detail':         f"sim={sim:.3f}"
                    })

        if (i // batch_size) % 5 == 0:
            print(f"  유사도 계산: {min(i+batch_size, len(corp_codes))}/{len(corp_codes)}")

    print(f"  재무 유사도 엣지: {len(edges) - sim_edges_before}개")

    # ── 엣지 3: 동일 규모 ──────────────────────────────────────────────
    print("\n[3] 동일 규모 엣지 생성 중...")
    size_edges_before = len(edges)

    size_groups = df.groupby('size')['corp_code'].apply(list)

    for size, corps in size_groups.items():
        if size is None or len(corps) < 2:
            continue
        corps = [str(c) for c in corps]
        # 규모 그룹은 너무 커지므로 랜덤 샘플링 (최대 50개 연결)
        if len(corps) > 50:
            np.random.seed(42)
            idx = np.random.choice(len(corps), 50, replace=False)
            sampled = [corps[i] for i in idx]  # ← np.random.choice 대신
        else:
            sampled = corps

        for i in range(len(sampled)):
            for j in range(i + 1, len(sampled)):
                edges.append({
                    'corp_code_from': sampled[i],
                    'corp_code_to':   sampled[j],
                    'edge_type':      'size',
                    'weight':         1.0,
                    'year':           0,
                    'detail':         size
                })
                edges.append({
                    'corp_code_from': sampled[j],
                    'corp_code_to':   sampled[i],
                    'edge_type':      'size',
                    'weight':         1.0,
                    'year':           0,
                    'detail':         size
                })

    print(f"  동일 규모 엣지: {len(edges) - size_edges_before}개")

    return pd.DataFrame(edges)


# ── 4. DB INSERT ─────────────────────────────────────────────────────
def save_edges(edges_df):
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()

    # 기존 데이터 삭제
    cur.execute("DELETE FROM relationships")
    conn.commit()

    edges_df.to_sql('relationships', conn,
                    if_exists='append', index=False)
    conn.commit()

    # 결과 확인
    result = pd.read_sql("""
        SELECT edge_type, COUNT(*) as 엣지수
        FROM relationships
        GROUP BY edge_type
        ORDER BY 엣지수 DESC
    """, conn)

    print("\n=== relationships 테이블 구성 완료 ===")
    print(result.to_string(index=False))
    print(f"\n총 엣지: {len(edges_df)}개")

    conn.close()


# ── 메인 ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("relationships 테이블 구성")
    print("=" * 50)

    companies, financials = load_data()
    edges_df = build_edges(companies, financials)
    save_edges(edges_df)

    print("\n완료. 다음 단계: Baseline 모델 학습")