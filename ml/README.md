# 기업 파산 예측 기반 대출 심사 시스템

> **한국 DART 데이터 기반 LSTM + GraphSAGE 통합 모델**
> 시계열 재무 패턴과 기업 간 관계망을 결합한 파산 예측 시스템
>
> https://drive.google.com/file/d/1BbXufespOSQV6qzLG__A8j-MjQr40Eap/view?usp=sharing
데이터배이스 다운로드

---

## 📌 프로젝트 개요

기업 대출 심사의 한계를 ML로 해결하는 시스템.

- **후행성 극복**: 다년도 시계열 분석으로 단면 분석보다 빠른 위험 신호 포착
- **관계망 분석**: 기업 간 GraphSAGE로 연쇄 부도 위험 반영
- **설명 가능성**: SHAP 기반 거절 사유 자동 생성

---

## 🎯 핵심 차별점

| 항목 | 기존 연구 (GNN-GAT-LSTM, 2025) | 본 프로젝트 |
|------|-------------------------------|-------------|
| 데이터 | 중국 상장사 | **한국 DART 최초 적용** |
| 시계열 | LSTM | LSTM (동일) |
| 그래프 | GAT | **GraphSAGE** |
| 엣지 구성 | 공급망/주주 | **업종/재무유사도/규모** |
| 증강 | 미적용 | **CTGAN** |

---

## 📊 성능 (중간 결과)

| 모델 | ROC-AUC | PR-AUC | F1 | Threshold |
|------|---------|--------|-----|-----------|
| LSTM | 0.9341 | 0.6531 | 0.6452 | 0.884 |
| CNN+LSTM | **0.9615** | **0.7549** | 0.7059 | 0.385 |
| TFT | 0.9133 | 0.7080 | **0.7647** | 0.690 |
| LSTM+GraphSAGE | 진행 중 | - | - | - |

목표 PR-AUC 0.75 ✓ / F1 0.70 ✓ 달성

---

## 🗂 프로젝트 구조

```
ml/
├── data/
│   ├── raw/                  # 원본 train/test
│   ├── processed/            # 전처리·증강 데이터
│   └── predictions/          # 모델별 예측 결과
├── db/
│   └── bankruptcy_prediction.db   # SQLite (11개 테이블)
├── saved_models/             # 학습된 모델 가중치
│   ├── lstm.pth / cnn_lstm.pth / tft.pth
│   ├── *_meta.json           # 메타데이터 (threshold, 성능)
│   └── current_model.json    # 활성 모델 지정
├── src/
│   ├── data_collection/      # DART/뉴스 수집
│   ├── database/             # DB 구축
│   ├── models/               # 모델 학습 스크립트
│   ├── prediction/           # predictor.py (추론 인터페이스)
│   └── dashboard/            # Streamlit 대시보드
└── README.md
```

---

## 🗄 데이터베이스 구조

| 테이블 | 행수 | 역할 |
|--------|------|------|
| companies | 3,962 | 기업 기본정보 |
| bankruptcy | 897 | 파산 이력 (DART 매칭 641건) |
| financial_raw | 910,511 | DART 원본 재무제표 |
| financials | 23,028 | 계산된 재무비율 19종 |
| relationships | 64,464 | 기업 간 엣지 |
| news_raw | 82,386 | 네이버 뉴스 원본 |
| news_sentiment_raw | 82,386 | KoBERT 감성 분석 |
| news_sentiment | 10,932 | 월별 집계 감성 피처 |

---

## 🔬 모델 아키텍처

### Baseline (비교용)

```
Altman Z-Score (1968) → Logistic Regression (1980)
→ XGBoost / LightGBM (2016) → LSTM 단독 (2018)
```

### 제안 모델

```
재무 시계열 (9년)
      ↓
[ LSTM ]  ─ 시계열 임베딩 생성
      ↓
[ GraphSAGE ]  ─ 이웃 기업 임베딩 집계
      ↓
[ Classifier ]  ─ 파산 확률 출력
```

### 그래프 엣지 (relationships 테이블)

```
industry       동일 업종 (KSIC 3자리)    22,860개
financial_sim  재무 유사도 (코사인 ≥0.8) 34,254개
size           동일 규모 (대/중/소)       7,350개
                              총합     64,464개
```

---

## 🧪 데이터 처리

### 입력 피처 (24개)

**재무비율 19개**
- 안정성: debt_ratio, current_ratio, interest_coverage, net_debt_ratio, equity_ratio
- 수익성: roa, roe, op_margin, net_margin
- 현금흐름: cfo_to_debt, fcf
- 성장성: revenue_growth, op_income_growth, asset_growth
- 시계열 파생: interest_cov_yoy, debt_ratio_trend, cf_volatility, consecutive_loss
- 종합: z_score (Altman)

**뉴스 피처 5개 (KoBERT 감성 분석)**
- sentiment_avg, negative_ratio, news_count
- news_bankruptcy_flag, news_lawsuit_flag

### CTGAN 증강

```
파산 기업 부족 문제:
  실제 파산 84개 → 9:1 불균형

해결:
  CTGAN으로 분포 학습 후 합성 샘플 생성
  학습 데이터: 74개 → 4,514개 (행 기준)
  테스트는 실제 데이터만 사용 (누수 방지)
```

---

## 🚀 실행 방법

### 1. 환경 설정

```bash
pip install torch pandas numpy scikit-learn
pip install ctgan transformers sentencepiece
pip install streamlit plotly
```

### 2. 데이터 수집 (이미 완료)

```bash
python src/database/Setup_db.py              # DB 생성
python src/data_collection/collect_companies.py  # 기업 목록
python src/data_collection/collect_financial.py  # 재무제표 (2015~2023)
python src/data_collection/collect_2024_2025.py  # 추가 (2024~2025)
python src/data_collection/calc_finalcials.py    # 재무비율 계산
python src/data_collection/crawl_news.py         # 뉴스 크롤링
python src/models/kobert_sentiment.py            # KoBERT 감성 분석
python src/database/build_relationship.py        # 관계망 구성
```

### 3. 모델 학습

```bash
python src/models/split_augment.py    # 8:2 분리 + CTGAN 증강
python src/models/all_model.py        # 3개 모델 일괄 학습
```

### 4. 대시보드 실행

```bash
python -m streamlit run src/dashboard/Dash_board.py
```

---

## 💼 대시보드 기능

### 1. 기업 검색
기업명 입력 → 일치 기업 표시 → 선택

### 2. 4개 메뉴

**💰 대출 심사**
- 신청 금액, 대출 기간, 업종 리스크, 담보 입력
- 출력: 승인/거절, 적정 금리, 권장 한도, Expected Loss

**📊 재무제표 분석**
- 수익성 / 안정성 / 유동성 / 성장성 4탭
- 최근 9~11년 시계열 차트

**📦 포트폴리오 리스크**
- 여러 기업 동시 평가
- PD 분포 시각화

**🔗 유사 기업 분석**
- relationships 테이블 기반
- 동일 업종 / 재무 유사도 / 동일 규모 필터
- 선택 기업 vs 유사 기업 PD 비교

---

## 📐 비즈니스 룰 (수식 기반)

### 적정 금리 (Basel III)

```
금리 = 기준금리 + 신용 스프레드 + 업종 가산 + 담보 할인

PD 구간별 스프레드:
  0~10%   : +1.5%
  10~30%  : +3.0%
  30~50%  : +5.0%
  50% 초과: 거절
```

### 권장 한도

```
신용 한도 = min(상환능력, 매출×3, 자산×0.5) × (1-PD)
담보 한도 = 담보가치 × 70%
최종 한도 = 신용 한도 + 담보 한도
```

### Expected Loss

```
EL = PD × LGD × EAD

LGD (담보 유형별):
  무담보:     60%
  부동산:     60% × (1 - coverage × 0.7)
  신용보증서: 60% × (1 - coverage × 0.85)
  기계설비:   60% × (1 - coverage × 0.5)
```

---

## 🔄 모델 교체 메커니즘

```python
# saved_models/current_model.json
{
  "active_model": "cnn_lstm",
  "threshold": 0.385
}
```

대시보드 사이드바에서 모델 선택 → `set_active()` 호출 → JSON 자동 갱신
→ 페이지 새로고침 시 새 모델 활성화

신규 모델 추가 시:
1. `saved_models/`에 `.pth`, `_meta.json`, `_scaler.npz` 저장
2. `predictor.py`의 `MODEL_REGISTRY`에 등록
3. 즉시 사용 가능

---

## 🛠 기술 스택

| 영역 | 사용 기술 |
|------|----------|
| 데이터 수집 | DART OpenAPI, 네이버 뉴스 API |
| NLP | KoBERT (snunlp/KR-FinBert-SC) |
| DB | SQLite |
| ML 프레임워크 | PyTorch |
| 데이터 증강 | CTGAN |
| 대시보드 | Streamlit + Plotly |
| 시각화 | Plotly Express |

---

## 📅 진행 현황

- ✅ 1~7단계 (데이터 수집 → 증강) 완료
- ✅ 시계열 모델 3종 (LSTM, CNN+LSTM, TFT) 학습 완료
- ✅ 대시보드 v1 구현 완료
- 🔄 LSTM + GraphSAGE 통합 모델 구현 중
- ⏳ Ablation Study + 최종 평가
- ⏳ FastAPI 백엔드 분리

---

## 📊 평가 지표

핵심 지표는 **PR-AUC** (불균형 데이터 표준):

```
정확도(Accuracy) 사용 X
  → 정상:파산 = 25:1 환경에서
    전부 정상 예측 시 정확도 96% 나옴

PR-AUC 우선:
  → 파산 클래스에만 집중하는 지표
```

목표:
- PR-AUC > 0.75
- F1 > 0.70
- Altman 대비 PR-AUC +15%p 개선

---

## 🏆 핵심 기여

1. **한국 DART 데이터 최초로** LSTM + GNN 결합
2. **CTGAN 기반 증강**으로 파산 샘플 희소성 해결
3. **GraphSAGE 선택 근거 실증** (균일 엣지 환경에서 GAT보다 효율적)
4. **Basel III 기반 비즈니스 룰**과 ML 모델 통합
5. **실시간 모델 교체 가능한** 모듈식 대시보드 구조

---

## 📝 참고 자료

- Altman, E. I. (1968). Financial Ratios, Discriminant Analysis and the Prediction of Corporate Bankruptcy
- Hamilton, W. et al. (2017). Inductive Representation Learning on Large Graphs (GraphSAGE)
- Xu, L. et al. (2019). Modeling Tabular Data using Conditional GAN (CTGAN)
- GNN-GAT-LSTM Bankruptcy Prediction (2025) — 비교 대상 선행 연구


---

## 📄 라이선스

학술 프로젝트 (비상업적 사용)
