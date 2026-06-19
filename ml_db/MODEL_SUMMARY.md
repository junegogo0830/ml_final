# PD 모델 (파산 예측) — 기법·결과 요약

> **데이터**: DART 공시 DB (`dart_v2.db`)  
> **목표**: 재무 비율 시계열 + 기업 관계 그래프로 파산 확률(PD) 예측  
> **평가 기준**: PR-AUC (주지표), ROC-AUC, Gini, KS  
> **분류**: 심각한 클래스 불균형 (train ~10%, test 현실 분포 유지)

---

## 1. 데이터 파이프라인

| 스크립트 | 역할 |
|----------|------|
| `00_init_db.py` | DB 초기화, 테이블 스키마 생성 |
| `01_corp_codes.py` | DART 기업코드 수집 |
| `02_selected_companies.py` | 분석 대상 기업 필터링 |
| `03_crisis_events.py` | 파산·상장폐지 이벤트 수집 → `labels` 테이블 |
| `05_financial_raw.py` | 재무제표 원본 수집 |
| `06_calc_financials.py` | 19개 재무비율 계산 → `financials` 테이블 |
| `07_fill_industry.py` | 산업코드 보완 |
| `08_build_edge.py` | 기업 간 관계 엣지 생성 → `graph_nodes`, `graph_edges` (연도별 스냅샷) |
| `09_build_sequence.py` | MAX_T=5 시계열 시퀀스 생성, Winsorizing + 표준화, `sequences.npz` 출력 |

### 재무비율 19개

```
debt_ratio, equity_ratio, debt_to_assets, noncurrent_liab_ratio,
current_ratio, quick_ratio, cash_ratio,
roa, roe, operating_margin, net_margin, gross_margin,
asset_turnover, inventory_turnover, interest_coverage_proxy,
ocf_to_current_liab, retained_earnings_ratio, working_capital_ratio, z_score
```

### 시계열·분할 설계

- **관측 윈도우**: 최대 5개 연도 (좌측 패딩, 마스크 M 사용)
- **시간 분할**: `obs_year ≤ 2022` → train, `obs_year > 2022` → test
- **양성(파산) obs_year**: `label_year − 1` 그래프 사용
- **Test 누수 방지**: test 양성이라도 `label_year−1 > 2022`이면 2022 그래프로 상한

---

## 2. 모델링 흐름

```
09 sequences.npz
     │
     ├── 10_train_compare.py   시계열 단독 3종 (LSTM / CNN+LSTM / TFT-lite)
     │         + Altman Z-score baseline
     │
     ├── 11_ctgan_augment.py   CTGAN 합성 증강 → 시계열 모델 재비교
     │
     ├── 12_graphsage.py       시계열 + GraphSAGE 결합 (ablation)
     │         LSTM 단독 vs LSTM+SAGE
     │         CNN+LSTM 단독 vs CNN+LSTM+SAGE
     │         TFT 단독 vs TFT+SAGE
     │
     └── 13_window.py          GraphSAGE + 증강 기법 5-seed 비교
               Base / Gaussian Noise / Mixup / Noise+Mixup
               3 인코더 × 4 증강 = 12 구성
```

---

## 3. 적용 기법 정리

### 3-1. 손실 함수: Focal Loss

$$\mathcal{L} = \alpha_t (1 - p_t)^{\gamma} \cdot \text{BCE}$$

- `alpha = 1 − pos_rate` (클래스 불균형 역비례 가중)
- `gamma = 2.0` (쉬운 음성에 집중 억제, 어려운 양성 집중)
- 이진 교차엔트로피 대비 희소 양성 클래스 학습 효율 향상

### 3-2. 시계열 인코더 3종

| 모델 | 구조 | 특징 |
|------|------|------|
| **LSTM** | LSTM(64) → last valid hidden | 기본 순환 구조, 해석 용이 |
| **CNN+LSTM** | Conv1d(32, k=2) → LSTM(64) | 지역 패턴 추출 후 순서 학습 |
| **TFT-lite** | Linear proj → GRN → MultiheadAttn(4) → LayerNorm | Self-attention, 마스크 직접 사용 |

- 마스킹: 패딩 위치(M=0)를 어텐션 key_padding_mask / last-valid-step 추출에 활용
- Early stopping: val PR-AUC 기준, patience=12

### 3-3. 그래프 인코더: GraphSAGE

- **입력**: 연도별 기업 관계 스냅샷 그래프, 노드 피처 = 19개 재무비율
- **구조**: SAGEConv(19→64) → ReLU → Dropout(0.3) → SAGEConv(64→64)
- **Self-loop 추가**: 고립 노드(이웃 없음) 방지
- **결합**: concat([시계열 임베딩, 그래프 임베딩]) → MLP(128→32→1)

**누수 방지 핵심**:
- 양성 기업: `obs_year = label_year − 1` (파산 직전 연도) 그래프
- test 양성: `min(label_year − 1, 2022)` → 미래 그래프 사용 금지
- 음성 기업: `obs_year ≤ 2022` 범위 최신 연도 (train/test 동일 기준)

### 3-4. 데이터 증강 (Script 11, 13)

| 기법 | 방식 | 적용 범위 |
|------|------|-----------|
| **CTGAN** (11) | 생성 모델로 양성 시퀀스 합성 (flatten→CTGAN→reshape) | train 양성 ×3 |
| **Window** (13 구버전) | 양성의 부분 시계열 (MIN_T=2 ~ valid−1) | train 양성 |
| **Gaussian Noise** (13) | 유효 시점 피처에 σ=0.05 노이즈 | train 양성 ×2 |
| **Mixup** (13) | 양성 쌍 λ~Beta(0.4,0.4) 선형 보간, 우세 샘플 마스크 상속 | train 양성 ~×1 |
| **Noise+Mixup** (13) | Noise 증강 후 Mixup 추가 | train 양성 ~×3 |

> **증강 공통 원칙**: test 데이터 불변, 증강 샘플은 원본의 obs_year·node_idx 상속

---

## 4. 평가 지표

| 지표 | 설명 | 주의 |
|------|------|------|
| **PR-AUC** (주지표) | Precision-Recall 곡선 아래 면적, 불균형 데이터에 적합 | test 현실 분포 (~10%)로 평가 |
| **ROC-AUC** | FPR-TPR 기반, 불균형 시 낙관적 | 참고용 |
| **Gini** | `2 × ROC − 1` | 참고용 |
| **KS** | `max(TPR − FPR)`, 신용평가 관행 | 참고용 |
| **Bootstrap 95% CI** | PR-AUC 신뢰구간 (n=1000) | 통계적 유의성 판단 |
| **5-seed mean±std** | 시드별 분산 측정 | 안정성 판단 |

---

## 5. 성능 결과

> 아래 표는 실험 후 채워넣을 것. 각 스크립트 실행 결과를 기록.

### 5-1. Script 10 — 시계열 단독 (단일 시드, seed=42)

| 모델 | ROC | PR | Gini | KS |
|------|-----|----|------|----|
| Altman Z-score baseline | — | — | — | — |
| LSTM | — | — | — | — |
| CNN+LSTM | — | — | — | — |
| TFT-lite | — | — | — | — |

### 5-2. Script 11 — CTGAN 증강 전/후

| 모델 | ROC (전) | PR (전) | ROC (후) | PR (후) | Δ PR |
|------|----------|---------|----------|---------|------|
| LSTM | — | — | — | — | — |
| CNN+LSTM | — | — | — | — | — |
| TFT | — | — | — | — | — |

### 5-3. Script 12 — GraphSAGE ablation (단일 시드)

| 모델 | ROC | PR | Gini | KS | PR 95% CI |
|------|-----|----|------|----|-----------|
| LSTM 단독 | — | — | — | — | — |
| LSTM+SAGE | — | — | — | — | — |
| CNN+LSTM 단독 | — | — | — | — | — |
| CNN+LSTM+SAGE | — | — | — | — | — |
| TFT 단독 | — | — | — | — | — |
| TFT+SAGE | — | — | — | — | — |

### 5-4. Script 13 — 5-seed 증강 비교 (mean±std)

| 모델 | 증강 | ROC | PR | Gini | KS |
|------|------|-----|----|------|----|
| LSTM+SAGE | Base | — | — | — | — |
| LSTM+SAGE | Noise | — | — | — | — |
| LSTM+SAGE | Mixup | — | — | — | — |
| LSTM+SAGE | Noise+Mixup | — | — | — | — |
| CNN+LSTM+SAGE | Base | — | — | — | — |
| CNN+LSTM+SAGE | Noise | — | — | — | — |
| CNN+LSTM+SAGE | Mixup | — | — | — | — |
| CNN+LSTM+SAGE | Noise+Mixup | — | — | — | — |
| TFT+SAGE | Base | — | — | — | — |
| TFT+SAGE | Noise | — | — | — | — |
| TFT+SAGE | Mixup | — | — | — | — |
| TFT+SAGE | Noise+Mixup | — | — | — | — |

---

## 6. 알려진 한계

### 데이터

- **양성 샘플 절대수 부족**: train 양성 ~159개 수준 → PR-AUC 분산 크고 Bootstrap CI 넓음
- **그래프 엣지 품질**: 관계 엣지가 단순 이진(존재/비존재)으로 정의됨; 가중 엣지(거래 금액, 지분율 등)로 고도화 가능
- **관측 윈도우 5년**: 파산까지 장기 추세를 놓칠 수 있음 (일부 기업은 10년 이상 재무악화)

### 모델

- **그래프 노드 피처 = 재무비율**: 시계열 인코더의 입력과 동일 정보원 → 완전히 독립적인 그래프 신호 부재
- **GraphSAGE 깊이 2층**: 2-hop 이웃까지만 집계, 광범위한 전파 효과 반영 안 됨
- **인코더 고정 (HID=64)**: 더 큰 hidden dim이나 bidirectional LSTM 미실험
- **Mixup 정렬 문제**: 서로 다른 유효 길이의 시계열을 단순 선형 보간하면 시간 정렬 불일치 발생 가능

### 평가

- **단일 테스트 셋**: 시간 기준 1회 분할, 교차 검증(시계열 walk-forward) 미적용
- **임계값 최적화 미수행**: 실운용 시 Precision-Recall 균형점(threshold) 재조정 필요
- **확률 보정 미수행**: Focal Loss 출력 확률이 실제 파산 확률과 calibration 되지 않음

---

## 7. 향후 개선 방향

| 영역 | 방향 |
|------|------|
| 그래프 | 가중 엣지 (지분율·거래금액) 도입, GATv2로 교체 |
| 시계열 | Mamba / S4 (상태공간 모델) 실험 |
| 증강 | TimeGAN 기반 합성, Feature Masking (결측 시뮬레이션) |
| 앙상블 | LSTM+SAGE, TFT+SAGE 스택 앙상블 (soft voting) |
| 보정 | Platt Scaling / Isotonic Regression으로 확률 캘리브레이션 |
| 평가 | Walk-forward 교차검증 (연도별 롤링 train/val split) |
