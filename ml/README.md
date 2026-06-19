# 기업 파산 예측 기반 대출 심사 대시보드

한국 DART 재무제표 시계열(LSTM)로 기업 파산 확률(PD)을 추정하고,
이를 바탕으로 대출 심사·재무분석·유사기업 비교·모델 버전 관리까지
한 화면에서 처리하는 Streamlit 대시보드입니다.

> 데이터/모델을 **새로 구축**하려면 컴패니언 프로젝트 [`ml_db`](../ml_db)를 먼저 실행해
> `db/dart_v2.db`를 만들어야 합니다. 이 저장소는 그 DB를 **서빙(추론/대시보드)**하는 쪽입니다.

---

## 기능

| 탭 | 내용 |
|----|------|
| **심사** | 신청 금액·기간·업종 리스크·담보 입력 → PD 기반 승인/거절, 적정 금리, 여신 한도, Expected Loss 산출. 거절 시 **SHAP(GradientExplainer)** 기반 거절 사유, 승인 시 심사 주요 근거를 카드로 표시 |
| **재무분석** | 수익성/안정성/유동성/기타 그룹별 시계열 차트, Z-Score(Altman), 전체 데이터 테이블 |
| **유사기업** | `graph_edges`/`graph_nodes` 기반 업종·재무유사도·규모 관계망에서 유사 기업 비교 |
| **버전관리** | 현재 재무제표 다운로드 시각, 코스피/코스닥/비상장 등록 기업 수 표시. **"새 버전 학습"** 클릭 시 DART API에서 최신 재무제표를 자동 수집하고 LSTM을 재학습 (기존 모델은 자동 백업) |

---

## 프로젝트 구조

```
ml/
├── backend/
│   ├── app.py          # Streamlit 앱 (UI + 라우팅 전체)
│   └── predictor.py     # 모델 로딩/추론/SHAP 설명 (LSTM·CNN+LSTM·TFT)
├── scripts/
│   ├── dart_fetch_latest.py   # 1단계: DART API에서 최신 회계연도 재무제표 수집
│   ├── retrain_lstm.py        # 2단계: dart_v2.db로 LSTM 재학습
│   └── refresh_pipeline.py    # 1+2단계를 순차 실행 (버전관리 탭의 "새 버전 학습" 버튼이 호출)
├── models/
│   └── saved/            # 활성 모델 체크포인트 (lstm/cnn_lstm/tft .pth + _meta.json + _scaler.npz)
├── db/
│   └── dart_v2.db        # SQLite (저장소에는 포함 안 됨, 아래 "DB 준비" 참고)
├── legacy/                # 구버전(v1) 파이프라인·실험 코드 보관 (현재 앱에서 사용 안 함)
├── requirements.txt
└── .gitignore
```

---

## 실행 방법

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. DB 준비

`db/dart_v2.db`는 용량이 커서(2GB+) 저장소에 포함하지 않습니다. 둘 중 하나로 준비하세요.

- **이미 가지고 있는 경우**: `db/dart_v2.db` 경로에 복사
- **새로 구축하는 경우**: 컴패니언 프로젝트 `ml_db`의 `scripts/00_init_db.py` ~ `09_build_sequence.py`를 순서대로 실행해 생성 후, 해당 `db/dart_v2.db`를 이 프로젝트의 `db/` 폴더로 복사

### 3. DART API 키 설정 (버전관리 탭의 "새 버전 학습" 기능에 필요)

[DART Open API](https://opendart.fss.or.kr)에서 키를 발급받아 환경변수로 설정합니다. (대시보드 조회/심사 자체는 키 없이도 동작합니다 — DB만 있으면 됩니다.)

```bash
# Windows cmd
set DART_API_KEY=발급받은_40자리_키

# PowerShell
$env:DART_API_KEY = "발급받은_40자리_키"
```

### 4. 대시보드 실행

```bash
python -m streamlit run backend/app.py
```

`http://localhost:8501` 접속.

---

## 모델 교체 / 재학습

`models/saved/current_model.json`이 활성 모델을 지정합니다.

```json
{ "active_model": "lstm", "threshold": 0.778 }
```

- **수동 교체**: `predictor.py`의 `Predictor.set_active(model_name)` 호출 (lstm / cnn_lstm / tft 중 `MODEL_REGISTRY`에 등록된 모델)
- **자동 재학습**: 대시보드 "버전관리" 탭 → "새 버전 학습" 클릭 → `refresh_pipeline.py`가 백그라운드로
  1) DART API에서 신규 연도 재무제표 수집 (`dart_fetch_latest.py`)
  2) LSTM 재학습 (`retrain_lstm.py`, 기존 체크포인트는 `models/saved/backup_<timestamp>/`로 자동 백업)

  를 순서대로 실행하고, 진행 상황을 `models/saved/pipeline_status.json`에 기록합니다.

---

## 기술 스택

DART OpenAPI · SQLite · PyTorch (양방향 LSTM+어텐션 / CNN+LSTM / TFT) · SHAP(GradientExplainer) · Streamlit + Plotly

---

## legacy/

CTGAN 증강, KoBERT 뉴스 감성분석, GraphSAGE 단독 학습 등 v1 실험 코드와 구버전 DB(`bankruptcy_prediction.db`)를 보관합니다. 현재 서빙 중인 대시보드(`backend/app.py`)는 이 폴더를 참조하지 않으며, git 추적에서도 제외되어 있습니다(`.gitignore`).
