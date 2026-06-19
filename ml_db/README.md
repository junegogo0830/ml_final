# ml_db — 파산 예측 v2 데이터 파이프라인

DART 공시 데이터를 수집해 기업 파산 예측용 DB(`dart_v2.db`)와 학습용 시퀀스를 만드는 파이프라인입니다.
여기서 만든 `dart_v2.db`는 컴패니언 프로젝트 [`ml`](../머신러닝/ml)의 Streamlit 대시보드가 그대로 가져다 씁니다.

## 설계 원칙

- **라벨 정확성**: 상장폐지가 아닌 DART 공시 기반 진짜 부도/회생/해산 이벤트
- **데이터 시점 정합성**: event_date와 financials의 시점 매칭 → 누수 방지 기반
- **재시작 가능**: 모든 수집 단계가 `collection_log_v2`에 진행상태 기록
- **감사 추적**: 모든 API 응답 원본을 `raw_response`에 JSON으로 보존

자세한 모델링 방법론·실험 결과는 [`MODEL_SUMMARY.md`](MODEL_SUMMARY.md) 참고.

## 폴더 구조

```
ml_db/
├── config.py             # API 키(환경변수), 경로, rate limit 설정
├── lib/
│   ├── dart_client.py    # API 호출 (rate limit, retry, 로깅)
│   └── db_helper.py      # SQLite 헬퍼
├── scripts/              # 00~14 순서대로 실행 (아래 "실행 순서" 참고)
├── db/                    # 생성될 SQLite + npz (repo에 포함 안 됨)
├── cache/                 # corp_code.zip 등 임시 파일 (repo에 포함 안 됨)
├── logs/                  # 일별 수집/호출 로그 (API 키 평문 포함 — repo에 포함 안 됨)
└── requirements.txt
```

## 준비

```bash
pip install -r requirements.txt
```

torch-geometric은 torch/CUDA 버전에 따라 설치 방법이 다릅니다.
[공식 설치 가이드](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)를 따라 별도 설치하세요.

DART API 키 발급 후 환경변수로 설정 (코드에 직접 넣지 마세요):

```bash
# Windows cmd
set DART_API_KEY=발급받은_40자리_키

# PowerShell
$env:DART_API_KEY = "발급받은_40자리_키"
```

## 실행 순서

`scripts/` 안의 번호 순서대로 실행합니다. `b` 접미사는 반기(half-year) 데이터를 함께 다루는 병렬 변형, `10~14`는 모델 학습/비교/실험 스크립트입니다.

| 단계 | 스크립트 | 역할 |
|------|----------|------|
| 0 | `00_init_db.py` | DB 스키마 생성 (idempotent) |
| 1 | `01_corp_codes.py` | DART 전체 기업 corp_code 수집 (활성+폐업) |
| 2 | `02_selected_companies.py` | 사업보고서 제출 의무 있는 기업만 `is_active=1`로 마킹, corp_cls 보강 |
| 3 | `03_crisis_events.py` | 부도/회생/해산/영업정지/상장폐지 이벤트 수집 → `crisis_events` |
| — | `delisting.py` | `crisis_events` 기반으로 `labels` 테이블(부도 라벨) 생성 |
| 5 | `05_financial_raw.py` | 재무제표 원본 수집 (사업보고서 11011 + 반기 11012, CFS 우선) |
| 6 | `06_calc_financials.py` / `06b_financial_calc_hy.py` | 19개 재무비율 계산 → `financials` / `financials_hy`(TTM) |
| 7 | `07_fill_industry.py` | 산업코드(KSIC) 보완 |
| 8 | `08_build_edge.py` / `08b_build_edges_hy.py` | 연도별 기업 관계 그래프 → `graph_nodes`, `graph_edges` |
| 9 | `09_build_sequence.py` / `09b_build_sequences_hy.py` | 시계열 시퀀스(MAX_T=5) 생성, Winsorize+표준화 → `sequences.npz` |
| 검증 | `verify_no_leak.py` | 반기 파이프라인 시점 누수 자동검증 (09b 이후, 숫자 보기 전 필수 통과) |
| 10 | `10_train_compare.py` / `10b_train_compare_hy.py` | LSTM/CNN+LSTM/TFT-lite 비교, Altman z-score 베이스라인 대비 |
| 11 | `11_ctgan_augment.py` | CTGAN으로 train 양성 시퀀스 증강 후 재비교 (test는 절대 증강 안 함) |
| 12 | `12_graphsage.py` / `12b_14b_graphsage_hy.py` | LSTM(시계열) + GraphSAGE(관계) 결합 모델 ablation |
| 13 | `13_window.py` | GraphSAGE + 증강(Noise/Mixup) 5-seed 비교 |
| 14 | `14_relational_features.py` | 관계 전용 피처 4개 추가 효과 검증 (누수 차단 설계) |

각 단계는 `collection_log_v2`(수집) 또는 결과 파일 존재 여부로 재시작 가능합니다. 중간에 멈춰도 같은 스크립트를 다시 실행하면 이어서 진행됩니다.

## 핵심 테이블

| 테이블 | 역할 |
|--------|------|
| `companies` | 기업 마스터 (활성+폐업, corp_code/corp_name/stock_code/corp_cls) |
| `crisis_events` | 부도/회생/해산 등 위기 이벤트 원본 (라벨 소스) |
| `labels` | 최종 파산 라벨 (`delisting.py`가 `crisis_events`로부터 생성) |
| `audit_opinions` | 감사의견 (적정/한정/부적정/의견거절) |
| `financial_raw` | DART `fnlttSinglAcntAll` 원본 응답 |
| `financials` / `financials_hy` | 계산된 19개 재무비율 (연간 / 반기 TTM) |
| `graph_nodes` / `graph_edges` | 연도별 기업 관계 그래프 스냅샷 |
| `collection_log_v2` | 수집 단계별 진행 로그 (재시작용) |

## 주의

- `db/*.db`, `db/*.npz`, `cache/`, `logs/`는 `.gitignore`에 포함되어 있습니다 — 용량이 크고(DB 2GB+), `logs/`에는 API 키가 평문으로 기록되므로 절대 커밋하지 마세요.
- 다른 PC에서 다시 만들려면 `scripts/00_init_db.py`부터 순서대로 실행하면 됩니다(전체 수집은 수 시간 소요).
