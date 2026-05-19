import sqlite3
import pandas as pd

conn = sqlite3.connect("db/bankruptcy_prediction.db")

# financials + news_sentiment 결합
print("데이터 결합 중...")
df = pd.read_sql("""
    SELECT 
        f.*,
        COALESCE(n.news_count,        0)    as news_count,
        COALESCE(n.sentiment_avg,     0.0)  as sentiment_avg,
        COALESCE(n.negative_ratio,    0.0)  as negative_ratio,
        COALESCE(n.positive_ratio,    0.0)  as positive_ratio,
        COALESCE(n.news_bankruptcy_flag, 0) as news_bankruptcy_flag,
        COALESCE(n.news_lawsuit_flag,    0) as news_lawsuit_flag,
        COALESCE(n.news_ceo_change_flag, 0) as news_ceo_change_flag,
        CASE WHEN b.corp_code IS NOT NULL THEN 1 ELSE 0 END as label
    FROM financials f
    LEFT JOIN news_sentiment n 
        ON f.corp_code = n.corp_code
        AND f.year = n.year
    LEFT JOIN bankruptcy b ON f.corp_code = b.corp_code
    ORDER BY f.corp_code, f.year
""", conn)

conn.close()

print(f"결합 완료: {len(df)}행 / {df['corp_code'].nunique()}개 기업")
print(f"파산: {df[df['label']==1]['corp_code'].nunique()}개")
print(f"정상: {df[df['label']==0]['corp_code'].nunique()}개")

# 뉴스 피처 결합 현황
print(f"\n뉴스 피처 있는 행: {(df['news_count']>0).sum()}행")
print(f"뉴스 피처 없는 행: {(df['news_count']==0).sum()}행 (결측→0으로 채움)")

# 컬럼 확인
print(f"\n전체 피처 수: {len(df.columns)}개")
print(df.columns.tolist())

# CSV 저장
df.to_csv("data/processed/financials_with_news.csv", index=False)
print("\n저장 완료: financials_with_news.csv")