"""
6단계: 기업 단위 8:2 분리 + CTGAN 증강 (3:1 목표)
실행: python split_and_augment.py

분리 기준: 기업 단위 stratified random split (8:2)
           동일 기업의 모든 연도는 같은 세트에 배치
증강 대상: 학습 데이터의 파산 기업만
목표 비율: 정상:파산 = 3:1

설치 필요:
  pip install ctgan scikit-learn
"""

import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

DB_PATH = "db/bankruptcy_prediction.db"
RANDOM_SEED = 42

FEATURE_COLS = [
    'debt_ratio', 'current_ratio', 'interest_coverage',
    'net_debt_ratio', 'equity_ratio',
    'roa', 'roe', 'op_margin', 'net_margin',
    'cfo_to_debt', 'fcf',
    'revenue_growth', 'op_income_growth', 'asset_growth',
    'interest_cov_yoy', 'debt_ratio_trend', 'cf_volatility',
    'consecutive_loss', 'z_score',
    # 뉴스 피처 추가
    'sentiment_avg', 'negative_ratio',
    'news_count', 'news_bankruptcy_flag', 'news_lawsuit_flag'
]


def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT f.*, c.market, c.industry_code,
               COALESCE(n.sentiment_avg, 0.0)      as sentiment_avg,
               COALESCE(n.negative_ratio, 0.0)     as negative_ratio,
               COALESCE(n.news_count, 0)            as news_count,
               COALESCE(n.news_bankruptcy_flag, 0)  as news_bankruptcy_flag,
               COALESCE(n.news_lawsuit_flag, 0)     as news_lawsuit_flag
        FROM financials f
        JOIN companies c ON f.corp_code = c.corp_code
        LEFT JOIN (
            SELECT corp_code,
                   AVG(sentiment_avg)            as sentiment_avg,
                   AVG(negative_ratio)           as negative_ratio,
                   SUM(news_count)               as news_count,
                   MAX(news_bankruptcy_flag)     as news_bankruptcy_flag,
                   MAX(news_lawsuit_flag)        as news_lawsuit_flag
            FROM news_sentiment
            GROUP BY corp_code
        ) n ON f.corp_code = n.corp_code
    """, conn)
    bankrupt_corps = pd.read_sql("""
        SELECT corp_code FROM bankruptcy
        WHERE corp_code IS NOT NULL
    """, conn)
    conn.close()

    bankrupt_set = set(bankrupt_corps['corp_code'].tolist())
    df['label'] = df['corp_code'].apply(lambda x: 1 if x in bankrupt_set else 0)

    print(f"전체 데이터: {len(df)}행, {df['corp_code'].nunique()}개 기업")
    print(f"  파산 기업: {df[df['label']==1]['corp_code'].nunique()}개")
    print(f"  정상 기업: {df[df['label']==0]['corp_code'].nunique()}개")
    return df


def corp_split(df):
    """기업 단위 stratified 8:2 분리"""
    corp_labels = df.groupby('corp_code')['label'].first().reset_index()

    train_corps, test_corps = train_test_split(
        corp_labels['corp_code'],
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=corp_labels['label']
    )

    train_df = df[df['corp_code'].isin(train_corps)].copy()
    test_df  = df[df['corp_code'].isin(test_corps)].copy()

    print("\n=== 기업 단위 8:2 분리 결과 ===")
    print(f"\n학습 데이터:")
    print(f"  파산 기업: {train_df[train_df['label']==1]['corp_code'].nunique()}개")
    print(f"  정상 기업: {train_df[train_df['label']==0]['corp_code'].nunique()}개")
    print(f"  전체 행수: {len(train_df)}행")
    print(f"\n테스트 데이터:")
    print(f"  파산 기업: {test_df[test_df['label']==1]['corp_code'].nunique()}개")
    print(f"  정상 기업: {test_df[test_df['label']==0]['corp_code'].nunique()}개")
    print(f"  전체 행수: {len(test_df)}행")

    return train_df, test_df


def augment_with_ctgan(train_df):
    """학습 데이터 파산 기업만 증강 (정상:파산 = 3:1, 기업 단위)"""
    try:
        from ctgan import CTGAN
    except ImportError:
        print("\nCTGAN 미설치. 설치 중...")
        import subprocess
        subprocess.run(["pip", "install", "ctgan", "-q"])
        from ctgan import CTGAN

    normal_count   = train_df[train_df['label']==0]['corp_code'].nunique()
    bankrupt_count = train_df[train_df['label']==1]['corp_code'].nunique()
    target_count   = normal_count // 3
    n_new_corps    = max(target_count - bankrupt_count, 0)

    print(f"\n=== CTGAN 증강 (3:1 목표) ===")
    print(f"정상 기업 수:    {normal_count}개")
    print(f"현재 파산 기업:  {bankrupt_count}개")
    print(f"목표 파산 기업:  {target_count}개")
    print(f"신규 생성 기업:  {n_new_corps}개")

    if n_new_corps <= 0:
        print("증강 불필요")
        return train_df

    # 파산 기업 전체 행 사용 (시계열 학습)
    bankrupt_train = train_df[train_df['label'] == 1].copy()

    # 결측치 30% 이하 피처만 사용
    available_features = [
        c for c in FEATURE_COLS
        if c in bankrupt_train.columns and
        bankrupt_train[c].isna().mean() <= 0.3
    ]
    augment_data = bankrupt_train[available_features].copy().dropna()

    print(f"사용 피처: {len(available_features)}개")
    print(f"학습 데이터: {len(augment_data)}행")

    print("\nCTGAN 학습 중... (3~5분 소요)")
    ctgan = CTGAN(epochs=300, verbose=False)
    ctgan.fit(augment_data)

    # 각 합성 기업당 평균 시계열 길이만큼 생성
    avg_years = bankrupt_train.groupby('corp_code').size().mean()
    avg_years = max(int(avg_years), 5)
    print(f"기업당 평균 시계열: {avg_years}년")

    total_rows = n_new_corps * avg_years
    synthetic  = ctgan.sample(total_rows)

    # 기업 코드 부여 (각 기업당 avg_years 행씩)
    corp_codes = []
    years      = []
    base_year  = int(bankrupt_train['year'].median())
    for i in range(n_new_corps):
        for y in range(avg_years):
            corp_codes.append(f'SYNTH_{i:04d}')
            years.append(base_year - (avg_years - 1 - y))

    synthetic['label']     = 1
    synthetic['corp_code'] = corp_codes
    synthetic['year']      = years

    print(f"증강 완료: {n_new_corps}개 기업 / {len(synthetic)}행 생성")

    # 품질 확인
    print(f"\n{'피처':<25} {'원본 평균':>10} {'원본 std':>10} {'합성 평균':>10} {'합성 std':>10}")
    print("-" * 65)
    for col in ['debt_ratio', 'roa', 'z_score', 'interest_coverage']:
        if col in available_features:
            om = augment_data[col].mean()
            os = augment_data[col].std()
            sm = synthetic[col].mean()
            ss = synthetic[col].std()
            print(f"  {col:<23} {om:>10.3f} {os:>10.3f} {sm:>10.3f} {ss:>10.3f}")

    return pd.concat([train_df, synthetic], ignore_index=True)


def save_splits(train_df, test_df, train_augmented):
    save_cols = ['corp_code', 'year', 'label'] + \
                [c for c in FEATURE_COLS if c in train_df.columns]

    train_df[[c for c in save_cols if c in train_df.columns]]\
        .to_csv("data/raw/train_original.csv", index=False)
    train_augmented[[c for c in save_cols if c in train_augmented.columns]]\
        .to_csv("data/processed/train_augmented.csv", index=False)
    test_df[[c for c in save_cols if c in test_df.columns]]\
        .to_csv("data/raw/test.csv", index=False)

    print("\n=== 파일 저장 완료 ===")
    print("  train_original.csv  : 증강 전 학습 데이터")
    print("  train_augmented.csv : 증강 후 학습 데이터 (CTGAN 포함)")
    print("  test.csv            : 테스트 데이터 (실제 데이터만)")


def final_summary(train_df, test_df, train_augmented):
    print("\n=== 최종 데이터셋 구성 ===")
    print(f"{'':20} {'학습(원본)':>12} {'학습(증강)':>12} {'테스트':>12}")
    print("-" * 60)

    for label, name in [(0, '정상'), (1, '파산')]:
        tr = train_df[train_df['label']==label]['corp_code'].nunique()
        ta = len(train_augmented[train_augmented['label']==label])
        te = test_df[test_df['label']==label]['corp_code'].nunique()
        print(f"  {name:<18} {tr:>12} {ta:>12} {te:>12}")

    print("-" * 60)
    print(f"  {'합계':<18} {len(train_df):>12} {len(train_augmented):>12} {len(test_df):>12}")

    n_aug = len(train_augmented[train_augmented['label']==0])
    b_aug = len(train_augmented[train_augmented['label']==1])
    n_te  = test_df[test_df['label']==0]['corp_code'].nunique()
    b_te  = test_df[test_df['label']==1]['corp_code'].nunique()

    print(f"\n  학습 불균형 (증강 후): {n_aug/max(b_aug,1):.1f}:1 (목표 3:1)")
    print(f"  테스트 불균형:         {n_te/max(b_te,1):.1f}:1")
    print(f"\n  ※ 테스트는 실제 데이터만 사용 (합성 데이터 없음)")


if __name__ == "__main__":
    print("=" * 50)
    print("6단계: 데이터 분리 + CTGAN 증강")
    print("=" * 50)

    print("\n[1] 데이터 로드")
    df = load_data()

    print("\n[2] 기업 단위 8:2 분리")
    train_df, test_df = corp_split(df)

    print("\n[3] CTGAN 증강 (학습만)")
    train_augmented = augment_with_ctgan(train_df)

    print("\n[4] 저장")
    save_splits(train_df, test_df, train_augmented)

    final_summary(train_df, test_df, train_augmented)

    print("\n완료. 다음 단계: 모델 학습 (07_baseline_models.py)")