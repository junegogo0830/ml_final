# 기업 파산 예측 기반 대출 심사 시스템

머신러닝 4조 (전서영 · 최병준 · 최영웅 · 최진)

기업의 재무 시계열 정보와 기업 간 관계 정보를 동시에 학습하여 부도 위험을 1~2년 전에 조기 탐지하고, 산출된 부도 확률(PD)을 실제 대출 심사 의사결정에 적용하는 end-to-end 시스템이다.

## 프로젝트 개요

기존 대출 심사는 단일 시점의 재무비율에 의존하여 기업의 재무 악화 추세를 반영하지 못한다. 본 프로젝트는 다음의 한계를 극복하는 것을 목표로 한다.

- 시계열 정보 부재: 단일 시점 평가에서 다년도 시계열 학습으로 전환
- 관계 정보 부재: 동일 업종 기업 간 연쇄 부실(contagion) 효과를 그래프 신경망으로 학습
- 라벨 정의의 모호성: "상장폐지" 대신 DART 공시 기반 5종 실제 위기 이벤트로 라벨 재정의
- 데이터 누수: 4단계 누수 차단 절차를 통해 실제 배포 환경에서 신뢰할 수 있는 성능 확보

## 핵심 결과

- 학습 대상: 4,657개 기업 (상장 3,967 + 비상장 690)
- 분석 기간: 2015 ~ 2025 (11년)
- 재무제표 원본: 약 808만 행
- 양성 라벨: 487건 (기존 84건 대비 5.8배 증가)
- 최종 모델: TFT + GraphSAGE
- 최종 성능: ROC-AUC 0.748, PR-AUC 0.364, Gini 0.496, KS 0.409

## 폴더 구조

```
.
├── ml_db/              # 데이터베이스 구축 관련 코드
│   ├── scripts/        # DART OpenAPI 수집 스크립트 (00~07)
│   ├── lib/            # API 호출 래퍼, DB 헬퍼
│   └── config.py
├── ml/                 # 모델 학습 및 대시보드
│   ├── src/
│   │   ├── models/     # LSTM, CNN+LSTM, TFT, GraphSAGE
│   │   ├── data/       # 전처리, split, augment
│   │   ├── dashboard/  # Streamlit 대시보드
│   │   └── prediction/ # predictor + 비즈니스 룰
│   ├── saved_models/   # 학습된 모델 가중치
│   └── data/           # 전처리된 데이터
├── AI_USAGE_LOG.md     # AI 도구 사용 기록
├── VALIDATION_LOG.md   # AI 결과 검증 기록
└── README.md
```

## 데이터베이스

데이터베이스는 용량이 약 2GB로 GitHub 저장소에 포함하지 않았다. 아래 구글 드라이브 링크를 통해 다운로드 후 `ml_db/db/` 폴더에 배치하면 된다.

https://drive.google.com/file/d/17Ifbyexl6Id7DXmTmQMXYvASm6CXXyft/view?usp=sharing

## 시스템 구조

입력 → 모델 → 출력의 흐름은 다음과 같다.

1. 입력: DART 재무제표 시계열 (19개 재무비율) + 기업 관계망 (업종/규모/유사도 엣지)
2. 모델: TFT (시계열 인코더) + GraphSAGE (관계 인코더) → 임베딩 결합 → 분류기
3. 출력: 부도 확률 PD (0~1)
4. 의사결정:
   - PD < 0.30 → 대출 승인
   - 0.30 ≤ PD ≤ 0.50 → 조건부 승인 (담보/보증 요구)
   - PD > 0.50 → 거절
5. 자동 산출: 적정 금리, 권장 한도, 예상 손실(EL)

## 입력 피처 (19개 재무비율)

| 구분 | 지표 |
|---|---|
| 재무구조 (4) | 부채비율, 자기자본비율, 자산대비부채비율, 비유동부채비율 |
| 유동성 (3) | 유동비율, 당좌비율, 현금비율 |
| 수익성 (5) | ROA, ROE, 영업이익률, 순이익률, 매출총이익률 |
| 활동성 (2) | 자산회전율, 재고회전율 |
| 현금흐름 (1) | 영업현금흐름 / 유동부채 |
| 복합·위험 (4) | 이자보상배율, 이익잉여금, 운전자본, Altman Z-score |

## 데이터 누수 통제

평가의 정직성을 확보하기 위해 4단계 누수 차단 절차를 적용하였다.

1. 라벨 누수: 위기 공시 정보를 입력 피처에서 제외
2. 시점 누수: label_date 이전 재무제표만 학습에 사용
3. 기업 단위 분할: 동일 corp_code가 train/test 양쪽에 포함되지 않도록 강제
4. 시간 분할: 2022년 이전 학습, 2023년 이후 평가 (temporal split)

누수 제거 결과 중간 발표 단계의 ROC-AUC 0.9615는 0.7472로 조정되었으며, 이는 실제 배포 환경에서 기대 가능한 정직한 성능 수치이다.

## 실행 방법

### 1. 데이터베이스 구축 (ml_db)

DART API 키 발급 후 `ml_db/config.py`에 입력한다.

```bash
cd ml_db
python scripts/00_init_db.py        # DB 스키마 생성
python scripts/01_corp_codes.py     # 전체 기업 수집
python scripts/02_companies.py      # 학습 대상 필터링
python scripts/03_crisis_events.py  # 위기 이벤트 수집
python scripts/labels.py            # 라벨 테이블 생성
python scripts/05_financial_raw.py  # 재무제표 원본 수집
python scripts/06_financial_calc.py # 재무비율 계산
```

### 2. 모델 학습 (ml)

```bash
cd ml
python src/data/split_augment.py    # train/test 분할
python src/models/train_all.py      # LSTM, CNN+LSTM, TFT + GraphSAGE 학습
```

### 3. 대시보드 실행

```bash
python -m streamlit run ml/src/dashboard/Dash_board.py
```

## 팀원별 역할

| 팀원 | 담당 역할 |
|---|---|
| 최병준 | 데이터 수집 (DART OpenAPI 호출, 크롤링), 모델 학습, 대시보드 구현 |
| 전서영 | 데이터 수집 (DART OpenAPI 호출, 크롤링), 모델 학습 |
| 최영웅 | 보고서 작성, PPT 제작, 자료 조사 |
| 최진 | 보고서 작성, PPT 제작, 자료 조사 |

## 환경

- Python 3.10
- PyTorch 2.1.0
- SQLite (DART 데이터 저장)
- Streamlit (대시보드)
- 운영체제: Windows
