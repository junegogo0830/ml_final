# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3, os, sys, math, json, subprocess
from datetime import datetime
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predictor import Predictor, calc_interest_rate, calc_loan_limit, calc_expected_loss

DB_PATH      = "db/dart_v2.db"
ACTIVE_MODEL = "lstm"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATUS_PATH  = PROJECT_ROOT / "models" / "saved" / "pipeline_status.json"

# ── 팔레트 (크림 + 틸) ─ 목표 이미지에서 추출
INK    = "#2D3035"   # 텍스트 주색 (차콜)
DARK   = "#3D4045"   # 버튼 / 다크 강조
SUB    = "#8A8175"   # 보조 텍스트 (웜 그레이)
LINE   = "#E6D6C0"   # 경계선 (크림 톤)
CARD   = "#FBF3E8"   # 카드 배경 (연한 크림)
BG     = "#FBE6D1"   # 메인 배경 (크림/살구)
TEAL   = "#5FA6A2"   # 주 액센트 (틸/제이드)
TEAL_D = "#3E8A85"   # 진한 틸 (강조 / hover / 안전)
WARN   = "#C2884B"   # 주의 (머스터드)
ALERT  = "#C2703A"   # 위험 (테라코타)
DANGER = "#B5453E"   # 매우 위험 (벽돌 레드)

# 하위호환 별칭 (기존 코드 {G1}~{G5} 그대로 동작)
G1, G2, G3, G4, G5 = INK, DARK, SUB, LINE, CARD

DART_COL_MAP = {
    "debt_ratio":              "debt_ratio",
    "current_ratio":           "current_ratio",
    "interest_coverage_proxy": "interest_coverage",
    "equity_ratio":            "equity_ratio",
    "roa":                     "roa",
    "roe":                     "roe",
    "operating_margin":        "op_margin",
    "net_margin":              "net_margin",
    "ocf_to_current_liab":     "cfo_to_debt",
    "z_score":                 "z_score",
}

FEATURE_KO = {
    "debt_ratio":           "부채비율",
    "current_ratio":        "유동비율",
    "interest_coverage":    "이자보상배율",
    "net_debt_ratio":       "순부채비율",
    "equity_ratio":         "자기자본비율",
    "roa":                  "총자산이익률(ROA)",
    "roe":                  "자기자본이익률(ROE)",
    "op_margin":            "영업이익률",
    "net_margin":           "순이익률",
    "cfo_to_debt":          "영업CF 대비 부채",
    "fcf":                  "잉여현금흐름",
    "revenue_growth":       "매출 성장률",
    "op_income_growth":     "영업이익 성장률",
    "asset_growth":         "자산 성장률",
    "interest_cov_yoy":     "이자보상배율 변화",
    "debt_ratio_trend":     "부채비율 추세",
    "cf_volatility":        "현금흐름 변동성",
    "consecutive_loss":     "연속 적자 여부",
    "z_score":              "Z-Score",
    "sentiment_avg":        "뉴스 감성 점수",
    "negative_ratio":       "부정 뉴스 비율",
    "positive_ratio":       "긍정 뉴스 비율",
    "news_count":           "뉴스 건수",
    "news_bankruptcy_flag": "파산 관련 뉴스",
    "news_lawsuit_flag":    "소송 관련 뉴스",
    "news_ceo_change_flag": "대표이사 변경 뉴스",
    "neighbor_avg_debt":    "연관기업 평균 부채비율",
    "neighbor_loss_ratio":  "연관기업 손실 비율",
    "industry_risk":        "업종 위험도",
    "related_default_rate": "연관기업 부도율",
}


def cls_name(cls):
    return {"Y": "코스피", "K": "코스닥", "N": "코넥스"}.get(cls or "", "기타")


# ── 페이지 설정
st.set_page_config(page_title="대출 심사", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown(f"""
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.css');

html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
.main, .block-container {{ background:{BG} !important; color:{INK}; }}
html, body, .stApp, [class*="css"] {{
    font-family:'Pretendard', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}}
#MainMenu, footer, header {{ visibility:hidden; }}
section[data-testid="stSidebar"],
[data-testid="collapsedControl"] {{ display:none !important; }}

/* 공통 버튼 */
.stButton > button {{
    border-radius:12px; font-weight:500; transition:all .15s;
    border:1.5px solid {LINE}; background:{CARD}; color:{DARK};
    padding:10px 18px;
}}
.stButton > button:hover {{
    background:#F3E7D3 !important; border-color:{TEAL} !important;
    color:{INK} !important;
}}
.stButton > button[kind="primary"] {{
    background:{DARK} !important; border-color:{DARK} !important;
    color:#F4F1EC !important; font-weight:600;
}}
.stButton > button[kind="primary"]:hover {{
    background:{INK} !important; border-color:{INK} !important;
}}

/* 홈 네비 버튼 — 다크 필 (이미지와 동일) */
.nav-btn > div > button {{
    padding:20px 32px !important; font-size:16px !important;
    font-weight:600 !important; letter-spacing:.02em;
    background:{DARK} !important; color:#F2EFEA !important;
    border:none !important; border-radius:14px !important;
}}
.nav-btn > div > button:hover {{
    background:{INK} !important; color:#FFFFFF !important;
}}

/* 뒤로가기 버튼 작게 */
.back-btn > div > button {{
    background:transparent !important; border:none !important;
    color:{SUB} !important; font-size:13px !important;
    padding:4px 0 !important; box-shadow:none !important;
    border-radius:0 !important;
}}
.back-btn > div > button:hover {{ color:{INK} !important; background:transparent !important; }}

/* 메트릭 — 값은 틸 */
[data-testid="metric-container"] {{
    background:{CARD}; border-radius:14px; padding:14px 18px;
    border-top:2px solid {TEAL};
    box-shadow:0 1px 3px rgba(60,40,20,.06);
}}
[data-testid="stMetricLabel"] {{ color:{SUB} !important; font-size:11px !important; }}
[data-testid="stMetricValue"] {{ color:{TEAL_D} !important; font-weight:700 !important; }}

/* 탭 */
.stTabs [data-baseweb="tab-list"] {{
    background:{CARD}; border-radius:12px 12px 0 0;
    gap:2px; border-bottom:1.5px solid {LINE}; padding:4px 8px 0;
}}
.stTabs [data-baseweb="tab"] {{
    color:{SUB}; font-weight:500; padding:7px 16px; border-radius:10px 10px 0 0;
}}
.stTabs [aria-selected="true"] {{
    color:{INK} !important; background:#F3E7D3 !important;
    border-bottom:2px solid {TEAL} !important;
}}

/* 인풋 */
.stTextInput input, .stNumberInput input {{
    background:{CARD}; border-color:{LINE}; color:{INK};
    border-radius:12px;
}}
.stTextInput input:focus, .stNumberInput input:focus {{
    border-color:{TEAL} !important;
    box-shadow:0 0 0 4px rgba(95,166,162,.18) !important;
}}
div[data-baseweb="select"] > div {{ background:{CARD}; border-color:{LINE}; border-radius:12px; }}

/* 슬라이더 틸 */
.stSlider [data-baseweb="slider"] div[role="slider"] {{ background:{TEAL_D} !important; }}
.stSlider [data-baseweb="slider"] > div > div > div {{ background:{TEAL} !important; }}
</style>
""", unsafe_allow_html=True)


# ── 헬퍼
@st.cache_resource
def get_predictor():
    p = Predictor()
    if p.active_name != ACTIVE_MODEL:
        try: p.set_active(ACTIVE_MODEL)
        except Exception: pass
    return p


def plotly_base(fig, height=320):
    fig.update_layout(
        height=height, margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(size=12, color=INK),
        title_font=dict(size=13, color=INK),
        xaxis=dict(gridcolor="#EFDFC9", linecolor=LINE, showline=True),
        yaxis=dict(gridcolor="#EFDFC9", linecolor=LINE, showline=True),
    )


def risk_badge(pd_score, thr):
    if pd_score < 0.10: return "안전",      TEAL_D, "#E7F1EF"
    if pd_score < 0.20: return "주의",      WARN,   "#F6EDD8"
    if pd_score < thr:  return "위험",      ALERT,  "#F4E2D2"
    return               "매우 위험", DANGER, "#F2DAD3"


def divider():
    st.markdown(f'<hr style="border:none;border-top:1px solid {LINE};margin:20px 0;">', True)


# ── 데이터 로더

@st.cache_data
def search_companies(keyword: str):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT c.corp_code, c.corp_name, c.corp_cls,
               c.industry_code, c.industry_name,
               CASE WHEN l.corp_code IS NOT NULL THEN 1 ELSE 0 END AS is_bankrupt
        FROM companies c
        LEFT JOIN labels l ON c.corp_code = l.corp_code AND l.label = 1
        WHERE c.corp_name LIKE ? AND c.corp_cls IN ('Y','K')
        ORDER BY c.corp_name LIMIT 20
    """, conn, params=(f"%{keyword}%",))
    conn.close()
    return df


@st.cache_data
def load_financials(corp_code: str):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT * FROM financials WHERE corp_code=? AND fs_div='CFS' ORDER BY year",
        conn, params=(corp_code,))
    if len(df) == 0:
        df = pd.read_sql(
            "SELECT * FROM financials WHERE corp_code=? ORDER BY year",
            conn, params=(corp_code,))
    conn.close()
    return df


@st.cache_data
def load_seq(corp_code: str, feature_cols_tuple: tuple):
    feature_cols = list(feature_cols_tuple)
    df = load_financials(corp_code)
    if len(df) == 0:
        return df
    for dc, mc in DART_COL_MAP.items():
        if dc in df.columns and mc not in df.columns:
            df[mc] = df[dc]
    for fc in feature_cols:
        if fc not in df.columns:
            df[fc] = 0.0
    return df


@st.cache_data
def load_similar(corp_code: str):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT c.corp_code, c.corp_name, c.corp_cls, ge.edge_type, ge.weight,
               CASE WHEN l.corp_code IS NOT NULL THEN 1 ELSE 0 END AS is_bankrupt
        FROM graph_edges ge
        JOIN graph_nodes gn_src
             ON ge.year=gn_src.year AND ge.src=gn_src.node_idx
            AND gn_src.corp_code=:cc
            AND gn_src.year=(SELECT MAX(year) FROM graph_nodes WHERE corp_code=:cc)
        JOIN graph_nodes gn_dst
             ON ge.year=gn_dst.year AND ge.dst=gn_dst.node_idx
        JOIN companies c ON gn_dst.corp_code=c.corp_code
        LEFT JOIN labels l ON c.corp_code=l.corp_code AND l.label=1
        ORDER BY ge.weight DESC LIMIT 50
    """, conn, params={"cc": corp_code})
    conn.close()
    return df


def build_seq(corp_code, predictor):
    fc    = predictor.meta["feature_cols"]
    df    = load_seq(corp_code, tuple(fc))
    if len(df) == 0:
        return None
    sl    = predictor.meta["seq_len"]
    feats = df[fc].fillna(0).values.astype("float32")
    return (feats[-sl:] if len(feats) >= sl
            else np.vstack([np.zeros((sl-len(feats), len(fc)), "float32"), feats]))


def predict_pd(corp_code, predictor):
    seq = build_seq(corp_code, predictor)
    if seq is None:
        return None
    try:
        return predictor.predict(seq)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_background_sequences(feature_cols_tuple, seq_len, n=30):
    feature_cols = list(feature_cols_tuple)
    conn  = sqlite3.connect(DB_PATH)
    codes = pd.read_sql(
        "SELECT DISTINCT corp_code FROM financials ORDER BY RANDOM() LIMIT ?",
        conn, params=(n,))
    conn.close()
    seqs = []
    for cc in codes["corp_code"]:
        df = load_seq(cc, tuple(feature_cols))
        if len(df) == 0:
            continue
        feats = df[feature_cols].fillna(0).values.astype("float32")
        seq   = (feats[-seq_len:] if len(feats) >= seq_len
                 else np.vstack([np.zeros((seq_len-len(feats), len(feature_cols)), "float32"), feats]))
        seqs.append(seq)
    return np.array(seqs, dtype="float32")


def explain_factors(corp_code, predictor, top_n=8):
    seq = build_seq(corp_code, predictor)
    if seq is None:
        return []
    fc = predictor.meta["feature_cols"]
    bg = load_background_sequences(tuple(fc), predictor.meta["seq_len"])
    if len(bg) == 0:
        return []
    try:
        return predictor.explain(seq, bg, top_n=top_n)
    except Exception:
        return []


# ════════════════════════════════════════════════════
# 0. 홈
# ════════════════════════════════════════════════════
def page_home():
    st.markdown(f"""
    <div style="text-align:center;padding:72px 0 48px;">
        <h1 style="margin:0;font-size:38px;font-weight:800;letter-spacing:-1px;color:{INK};">
            대출 심사
        </h1>
    </div>
    """, unsafe_allow_html=True)

    _, c, _ = st.columns([2, 1, 2])
    with c:
        for tab in ["심사", "재무분석", "유사기업", "버전관리"]:
            st.markdown('<div class="nav-btn">', unsafe_allow_html=True)
            if st.button(tab, key=f"nav_{tab}", use_container_width=True):
                st.session_state["page"] = tab
                st.session_state["corp"] = None
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════
# 버전 관리 화면
# ════════════════════════════════════════════════════
def load_version_stats():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    last_collected = cur.execute(
        "SELECT MAX(completed_at) FROM collection_log_v2 WHERE step='05_financial_raw'"
    ).fetchone()[0]
    n_kospi = cur.execute("SELECT COUNT(*) FROM companies WHERE corp_cls='Y'").fetchone()[0]
    n_kosdaq = cur.execute("SELECT COUNT(*) FROM companies WHERE corp_cls='K'").fetchone()[0]
    n_unlisted = cur.execute(
        "SELECT COUNT(*) FROM companies WHERE corp_cls IS NULL OR corp_cls NOT IN ('Y','K')"
    ).fetchone()[0]
    max_year = cur.execute("SELECT MAX(year) FROM financials").fetchone()[0]
    conn.close()
    return {
        "last_collected": last_collected,
        "n_kospi": n_kospi,
        "n_kosdaq": n_kosdaq,
        "n_unlisted": n_unlisted,
        "max_year": max_year,
    }


def load_pipeline_status():
    if not STATUS_PATH.exists():
        return None
    try:
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def page_version(predictor):
    st.markdown('<div class="back-btn">', unsafe_allow_html=True)
    if st.button("← 뒤로", key="back_to_home_version"):
        st.session_state["page"] = None
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="text-align:center;padding:40px 0 32px;">
        <h2 style="margin:0;font-size:26px;font-weight:700;color:{INK};">버전 관리</h2>
    </div>
    """, unsafe_allow_html=True)

    stats  = load_version_stats()
    status = load_pipeline_status()

    dl_time = stats["last_collected"]
    if dl_time:
        try:
            dl_time = datetime.fromisoformat(dl_time).strftime("%Y-%m-%d %H:%M")
        except Exception:
            pass
    else:
        dl_time = "기록 없음"

    c1, c2 = st.columns(2)
    with c1:
        st.metric("최근 재무제표 다운로드", dl_time)
    with c2:
        st.metric("최신 데이터 회계연도",
                  f"{stats['max_year']}년" if stats["max_year"] else "-")

    divider()

    st.markdown(f"""
    <div style="font-size:14px;font-weight:600;color:{INK};margin-bottom:8px;">
        등록된 기업 수
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1: st.metric("코스피", f"{stats['n_kospi']:,}개")
    with m2: st.metric("코스닥", f"{stats['n_kosdaq']:,}개")
    with m3: st.metric("비상장", f"{stats['n_unlisted']:,}개")

    divider()

    running = bool(status and status.get("running"))

    if status:
        step_label = {
            "fetch":   "재무제표 수집 중",
            "retrain": "모델 재학습 중",
            "done":    "완료",
            "error":   "오류",
        }.get(status.get("step"), status.get("step", ""))

        box_color = ALERT if status.get("step") == "error" else (TEAL_D if running else SUB)
        st.markdown(f"""
        <div style="background:{CARD};border:1.5px solid {LINE};border-radius:14px;
                    padding:14px 20px;margin:10px 0;">
            <div style="font-size:13px;font-weight:700;color:{box_color};margin-bottom:4px;">
                {step_label}
            </div>
            <div style="font-size:13px;color:{INK};">
                {status.get('message', '')}
            </div>
            <div style="font-size:11px;color:{SUB};margin-top:6px;">
                업데이트: {status.get('updated_at', '-')}
            </div>
        </div>
        """, unsafe_allow_html=True)

        progress = status.get("progress") or {}
        if running and progress.get("total"):
            st.progress(min(progress.get("done", 0) / progress["total"], 1.0))
        elif running and progress.get("epochs"):
            st.progress(min(progress.get("epoch", 0) / progress["epochs"], 1.0))

    b1, b2 = st.columns(2)
    with b1:
        if st.button("새 버전 학습", type="primary", use_container_width=True,
                     disabled=running, key="start_retrain"):
            STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
            subprocess.Popen(
                [sys.executable, str(PROJECT_ROOT / "scripts" / "refresh_pipeline.py")],
                cwd=str(PROJECT_ROOT),
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
            st.rerun()
    with b2:
        if st.button("새로고침", use_container_width=True, key="refresh_status"):
            st.rerun()

    if running:
        st.caption("학습이 진행 중입니다. '새로고침' 버튼으로 진행 상황을 확인하세요.")


# ════════════════════════════════════════════════════
# 1. 탭별 검색 화면
# ════════════════════════════════════════════════════
def page_search(tab: str, predictor):
    # 뒤로
    st.markdown('<div class="back-btn">', unsafe_allow_html=True)
    if st.button("← 뒤로", key="back_to_home"):
        st.session_state["page"] = None
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="text-align:center;padding:40px 0 32px;">
        <h2 style="margin:0;font-size:26px;font-weight:700;color:{INK};">{tab}</h2>
    </div>
    """, unsafe_allow_html=True)

    _, c, _ = st.columns([1, 2, 1])
    with c:
        keyword = st.text_input("기업명", placeholder="기업명을 입력하세요",
                                label_visibility="collapsed", key="search_kw")
        if keyword:
            results = search_companies(keyword)
            if len(results) == 0:
                st.error("검색 결과가 없습니다.")
            else:
                options = [f"{r['corp_name']}  ({cls_name(r['corp_cls'])})"
                           for _, r in results.iterrows()]
                idx = st.selectbox("기업 선택", range(len(options)),
                                   format_func=lambda i: options[i],
                                   label_visibility="collapsed", key="search_sel")
                if st.button("조회", type="primary", use_container_width=True,
                             key="search_go"):
                    st.session_state["corp"] = results.iloc[idx].to_dict()
                    st.rerun()


def render_explanation(corp_code: str, predictor, mode: str = "reject"):
    """SHAP(GradientExplainer) 기반 주요 영향 요인 표시.
    mode='reject' → 거절 사유 / mode='approve' → 심사 주요 근거"""
    with st.spinner("주요 영향 요인 분석 중..."):
        factors = explain_factors(corp_code, predictor, top_n=10)
    if not factors:
        return

    risk_factors = [f for f in factors if f["shap"] > 0][:5]
    safe_factors = sorted([f for f in factors if f["shap"] < 0], key=lambda r: r["shap"])[:3]

    title = "거절 사유" if mode == "reject" else "심사 주요 근거"
    st.markdown(f'<p style="font-size:14px;font-weight:700;color:{INK};margin:14px 0 6px;">{title}</p>',
                unsafe_allow_html=True)

    def _row(f, color, sign):
        name = FEATURE_KO.get(f["feature"], f["feature"])
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;align-items:center;
                    background:{CARD};border:1px solid {LINE};border-radius:10px;
                    padding:8px 14px;margin-bottom:6px;">
            <span style="font-size:13px;color:{INK};">{name}</span>
            <span style="font-size:12px;color:{SUB};">현재값 {f['value']:.3f}</span>
            <span style="font-size:13px;font-weight:700;color:{color};">{sign}{f['shap']*100:.2f}%p</span>
        </div>
        """, unsafe_allow_html=True)

    if risk_factors:
        st.markdown(f'<p style="font-size:12px;color:{SUB};margin:0 0 4px;">파산 확률을 높이는 요인</p>',
                    unsafe_allow_html=True)
        for f in risk_factors:
            _row(f, ALERT, "+")
    else:
        st.caption("파산 확률을 뚜렷하게 높이는 요인이 발견되지 않았습니다.")

    if mode != "reject" and safe_factors:
        st.markdown(f'<p style="font-size:12px;color:{SUB};margin:10px 0 4px;">파산 확률을 낮추는 요인</p>',
                    unsafe_allow_html=True)
        for f in safe_factors:
            _row(f, TEAL_D, "")


# ════════════════════════════════════════════════════
# 결과: 심사
# ════════════════════════════════════════════════════
def render_loan(corp: dict, predictor):
    with st.spinner("파산 확률 분석 중..."):
        pred = predict_pd(corp["corp_code"], predictor)

    if pred is None:
        st.error("재무 데이터가 없는 기업입니다.")
        return

    pd_score = pred["probability"]
    thr      = predictor.meta["threshold"]
    level, lc, lb = risk_badge(pd_score, thr)

    col_gauge, col_form = st.columns([2, 3])

    with col_gauge:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(pd_score * 100, 2),
            number=dict(suffix="%", font=dict(size=32, color=lc)),
            gauge=dict(
                axis=dict(range=[0, 100], tickwidth=1, tickcolor=LINE,
                          tickfont=dict(size=10)),
                bar=dict(color=lc, thickness=0.22),
                bgcolor=CARD, borderwidth=0,
                steps=[
                    dict(range=[0, 10],         color="#DCEDEA"),
                    dict(range=[10, thr * 100], color="#F3E7CF"),
                    dict(range=[thr * 100, 50], color="#F2DCC9"),
                    dict(range=[50, 100],       color="#E9C2B5"),
                ],
                threshold=dict(line=dict(color=DANGER, width=3),
                               thickness=0.8, value=thr * 100),
            ),
        ))
        fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=0),
                          paper_bgcolor=BG)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
        <div style="background:{lb};border:1px solid {lc}30;border-radius:14px;
                    padding:10px;text-align:center;margin-top:-4px;">
            <div style="font-size:16px;font-weight:700;color:{lc};">{level}</div>
            <div style="font-size:11px;color:{SUB};margin-top:2px;">
                임계값 {thr*100:.1f}% · {'파산' if pred['label']==1 else '정상'}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_form:
        fc1, fc2 = st.columns(2)
        with fc1:
            loan_amount   = st.number_input("신청 금액 (억원)", min_value=1.0,
                                            max_value=10000.0, value=100.0, step=10.0)
            loan_period   = st.slider("대출 기간 (년)", 1, 10, 3)
            industry_risk = st.selectbox(
                "업종 위험도", ["low", "medium", "high"], index=1,
                format_func=lambda x: {"low": "저위험", "medium": "보통", "high": "고위험"}[x])
            total_assets  = st.number_input("총자산 (억원)", min_value=1.0,
                                            value=500.0, step=50.0)
        with fc2:
            has_collateral  = st.checkbox("담보 보유")
            collateral_value, collateral_type = 0.0, None
            if has_collateral:
                collateral_type  = st.selectbox("담보 종류",
                                                ["부동산", "유가증권", "보증서"])
                collateral_value = st.number_input("담보 가치 (억원)", min_value=0.0,
                                                   value=50.0, step=10.0)
            operating_cf = st.number_input("영업현금흐름 (억원)", value=30.0, step=10.0)
            revenue      = st.number_input("매출액 (억원)", min_value=0.0,
                                           value=200.0, step=50.0)
            has_prev = st.checkbox("이전 대출 이력")
            loan_penalty = 0.0
            if has_prev:
                ps = st.selectbox("기존 대출 상태", ["정상", "연체 이력", "부실"])
                if ps == "연체 이력":
                    loan_penalty = 0.005
                elif ps == "부실":
                    loan_penalty = 0.015
                    pd_score = min(pd_score * 1.3, 1.0)

    divider()

    rate = calc_interest_rate(pd_score, industry_risk, has_collateral)
    if rate is None or pd_score >= 0.5:
        st.markdown(f"""
        <div style="background:#F2DAD3;border:1px solid #E0B5AC;border-radius:14px;
                    padding:20px;text-align:center;">
            <div style="font-size:20px;font-weight:800;color:{DANGER};margin-bottom:4px;">
                여신 거절
            </div>
            <div style="font-size:13px;color:#8A332E;">
                파산 확률 {pd_score*100:.1f}% — 기준(50%) 초과
            </div>
        </div>
        """, unsafe_allow_html=True)
        render_explanation(corp["corp_code"], predictor, mode="reject")
        return

    rate_total   = rate + loan_penalty
    limit        = calc_loan_limit(operating_cf, total_assets, pd_score,
                                   collateral_value, loan_period, rate_total, revenue)
    approved_amt = min(loan_amount, limit)
    ok           = loan_amount <= limit
    rc           = TEAL_D if ok else WARN
    rbg          = "#E7F1EF" if ok else "#F6EDD8"

    st.markdown(f"""
    <div style="background:{rbg};
                border:1px solid {rc}30;border-radius:14px;
                padding:14px 20px;margin-bottom:12px;">
        <div style="font-size:18px;font-weight:800;color:{rc};margin-bottom:2px;">
            {'승인' if ok else '조건부 승인'}
        </div>
        <div style="font-size:13px;color:{INK};">
            {'신청 금액 '+str(int(loan_amount))+'억 승인'
             if ok else f'신청 {int(loan_amount)}억 → 한도 {limit:.1f}억으로 감액'}
        </div>
    </div>
    """, unsafe_allow_html=True)

    r1, r2, r3, r4 = st.columns(4)
    with r1: st.metric("적용 금리",    f"{rate_total*100:.2f}%")
    with r2: st.metric("여신 한도",    f"{limit:.1f} 억")
    with r3: st.metric("승인 금액",    f"{approved_amt:.1f} 억")
    el, lgd = calc_expected_loss(pd_score, approved_amt, has_collateral,
                                 collateral_type, collateral_value)
    with r4: st.metric("Expected Loss", f"{el:.2f} 억",
                       help=f"EL = PD × LGD({lgd*100:.0f}%) × EAD")

    with st.expander("금리 산출 내역"):
        ind_p  = {"low": 0.0, "medium": 0.003, "high": 0.005}[industry_risk]
        cd     = -0.005 if has_collateral else 0.0
        csp    = rate - 0.035 - ind_p - cd
        st.markdown(f"""
| 항목 | 값 |
|------|-----|
| 기준 금리 | 3.50% |
| 신용 스프레드 | {csp*100:.2f}% |
| 업종 가산 | {ind_p*100:.2f}% |
| 담보 할인 | {cd*100:.2f}% |
| 이전 대출 가산 | {loan_penalty*100:.2f}% |
| **최종 금리** | **{rate_total*100:.2f}%** |
""")

    with st.expander("심사 주요 근거 (SHAP)"):
        render_explanation(corp["corp_code"], predictor, mode="approve")


# ════════════════════════════════════════════════════
# 결과: 재무분석
# ════════════════════════════════════════════════════
def render_financial(corp: dict, predictor):
    fin = load_financials(corp["corp_code"])
    if len(fin) == 0:
        st.error("재무 데이터가 없습니다.")
        return

    latest = fin.iloc[-1]
    sv = lambda v, d=0: (v if v is not None and not pd.isna(v) else d)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1: st.metric("부채비율",  f"{sv(latest.get('debt_ratio'))*100:.1f}%")
    with m2: st.metric("유동비율",  f"{sv(latest.get('current_ratio'))*100:.1f}%")
    with m3: st.metric("ROA",       f"{sv(latest.get('roa'))*100:.2f}%")
    with m4: st.metric("ROE",       f"{sv(latest.get('roe'))*100:.2f}%")
    with m5: st.metric("이자보상",  f"{sv(latest.get('interest_coverage_proxy')):.2f}x")
    with m6: st.metric("Z-Score",   f"{sv(latest.get('z_score')):.2f}")

    divider()

    GROUPS = [
        ("수익성",
         ["roa", "roe", "operating_margin", "net_margin"],
         ["ROA", "ROE", "영업이익률", "순이익률"]),
        ("안정성",
         ["debt_ratio", "equity_ratio", "debt_to_assets", "z_score"],
         ["부채비율", "자기자본비율", "부채/자산", "Z-Score"]),
        ("유동성",
         ["current_ratio", "quick_ratio", "interest_coverage_proxy", "ocf_to_current_liab"],
         ["유동비율", "당좌비율", "이자보상배율", "영업CF/유동부채"]),
        ("기타",
         ["asset_turnover", "retained_earnings_ratio", "working_capital_ratio"],
         ["자산회전율", "이익잉여금비율", "운전자본비율"]),
    ]

    # 크림 배경에 어울리는 틸 앵커 팔레트
    palette = [TEAL_D, "#C2884B", "#7A9E8E", "#9C6B4F", DARK]
    tabs = st.tabs([g[0] for g in GROUPS])
    for tab, (gname, cols, labels) in zip(tabs, GROUPS):
        with tab:
            avail = [(c, l) for c, l in zip(cols, labels) if c in fin.columns]
            if not avail:
                st.info(f"{gname} 데이터 없음")
                continue
            fig = go.Figure()
            for i, (c, l) in enumerate(avail):
                fig.add_trace(go.Scatter(
                    x=fin["year"], y=fin[c], mode="lines+markers", name=l,
                    line=dict(color=palette[i % len(palette)], width=2.5),
                    marker=dict(size=7, line=dict(color="white", width=1.5)),
                ))
            plotly_base(fig, height=300)
            fig.update_xaxes(title="연도")
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("전체 데이터"):
        show = ["year", "debt_ratio", "equity_ratio", "current_ratio",
                "roa", "roe", "operating_margin", "net_margin", "z_score"]
        avail = [c for c in show if c in fin.columns]
        st.dataframe(fin[avail].round(4), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════
# 결과: 유사기업 + 관계망
# ════════════════════════════════════════════════════
def _build_network(center_code, center_name, nbrs, predictor):
    etypes    = nbrs["edge_type"].unique().tolist() or ["unknown"]
    ECOLS     = [TEAL_D, "#C2884B", "#7A9E8E", "#9C6B4F"]
    ET_COLOR  = {et: ECOLS[i % len(ECOLS)] for i, et in enumerate(etypes)}
    ET_KO     = {et: et.replace("_", " ") for et in etypes}
    sec_cnt   = nbrs.groupby("edge_type").size().to_dict()
    sec_start = {et: i * 2 * math.pi / max(len(etypes), 1) for i, et in enumerate(etypes)}

    own_pred = predict_pd(center_code, predictor)
    own_pd   = own_pred["probability"] * 100 if own_pred else 0

    nx = [0]; ny = [0]; npd = [own_pd]; nsz = [20]
    nhv = [f"{center_name}<br>PD: {own_pd:.1f}% (조회 기업)"]
    etr = {et: {"x": [], "y": []} for et in etypes}
    seen = {et: 0 for et in etypes}

    for _, row in nbrs.iterrows():
        et    = row["edge_type"]
        total = sec_cnt.get(et, 1)
        angle = (sec_start.get(et, 0)
                 + 2 * math.pi / max(len(etypes), 1) * (seen[et] + 0.5) / total)
        seen[et] += 1
        x, y = math.cos(angle) * 1.8, math.sin(angle) * 1.8
        nx.append(x); ny.append(y)
        p    = predict_pd(row["corp_code"], predictor)
        pv   = p["probability"] * 100 if p else 0
        npd.append(pv); nsz.append(13)
        nhv.append(f"{row['corp_name']}<br>유사도:{row['weight']:.3f}  PD:{pv:.1f}%"
                   + ("  [위기이력]" if row["is_bankrupt"] else ""))
        etr[et]["x"] += [0, x, None]; etr[et]["y"] += [0, y, None]

    fig = go.Figure()
    for et, d in etr.items():
        if d["x"]:
            fig.add_trace(go.Scatter(x=d["x"], y=d["y"], mode="lines",
                                     name=ET_KO[et],
                                     line=dict(color=ET_COLOR[et], width=1.5),
                                     opacity=0.45, hoverinfo="none"))
    fig.add_trace(go.Scatter(
        x=nx, y=ny, mode="markers+text",
        marker=dict(size=nsz, color=npd,
                    colorscale="RdYlGn_r", cmin=0, cmax=100,
                    colorbar=dict(title="PD (%)", thickness=11, len=0.65),
                    line=dict(color="white", width=2)),
        text=[h.split("<br>")[0] for h in nhv],
        textposition="top center", textfont=dict(size=9, color=DARK),
        hovertext=nhv, hoverinfo="text", showlegend=False,
    ))
    fig.update_layout(
        height=440, showlegend=True,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor=BG, plot_bgcolor=BG,
        xaxis=dict(showgrid=False, zeroline=False,
                   showticklabels=False, range=[-2.6, 2.6]),
        yaxis=dict(showgrid=False, zeroline=False,
                   showticklabels=False, range=[-2.6, 2.6]),
        legend=dict(bgcolor="rgba(251,243,232,.9)",
                    bordercolor=LINE, borderwidth=1),
        font=dict(size=11, color=INK),
    )
    return fig


def render_similar(corp: dict, predictor):
    similar = load_similar(corp["corp_code"])
    if len(similar) == 0:
        st.warning("관계 데이터가 없는 기업입니다.")
        return

    with st.spinner("분석 중..."):
        rows     = []
        own_pred = predict_pd(corp["corp_code"], predictor)
        for _, row in similar.head(20).iterrows():
            p = predict_pd(row["corp_code"], predictor)
            if p is None:
                continue
            rows.append({
                "기업명": row["corp_name"],
                "유사도": round(row["weight"], 3),
                "실제":   "위기" if row["is_bankrupt"] else "정상",
                "PD (%)": round(p["probability"] * 100, 2),
                "예측":   "파산" if p["label"] == 1 else "정상",
            })

    if not rows:
        st.info("분석 가능한 유사기업이 없습니다.")
        return

    sim_df = pd.DataFrame(rows).sort_values("PD (%)", ascending=False)
    own_pd = own_pred["probability"] * 100 if own_pred else 0
    avg_pd = sim_df["PD (%)"].mean()
    diff   = own_pd - avg_pd

    c1, c2, c3 = st.columns(3)
    with c1: st.metric(f"{corp['corp_name']} PD", f"{own_pd:.2f}%")
    with c2: st.metric("유사기업 평균 PD", f"{avg_pd:.2f}%")
    with c3: st.metric("차이", f"{diff:+.2f}%p", delta=f"{diff:+.2f}%p",
                       delta_color="inverse")

    if diff > 5:
        st.warning(f"유사기업 대비 PD가 {diff:.1f}%p 높음 — 추가 심사 필요")
    elif diff < -5:
        st.success(f"유사기업 대비 PD가 {abs(diff):.1f}%p 낮음 — 비교적 안정")

    col1, col2 = st.columns([3, 2])
    with col1:
        def cpd(val):
            if val >= 50:     return f"color:{DANGER};font-weight:bold"
            if val >= own_pd: return f"color:{WARN}"
            return f"color:{TEAL_D}"
        st.dataframe(sim_df.style.applymap(cpd, subset=["PD (%)"]).format({"PD (%)": "{:.2f}"}),
                     use_container_width=True, hide_index=True)

    with col2:
        fig = px.bar(sim_df.head(12), x="PD (%)", y="기업명", orientation="h",
                     color="PD (%)", color_continuous_scale="RdYlGn_r", title="PD 비교")
        fig.add_vline(x=own_pd, line_dash="dash", line_color=DARK)
        plotly_base(fig, height=320)
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    divider()
    st.markdown(f'<p style="font-size:13px;font-weight:600;color:{INK};margin-bottom:8px;">관계망</p>',
                unsafe_allow_html=True)
    max_n = st.slider("최대 이웃 수", 5, 40, 20, key="net_max")
    with st.spinner("네트워크 생성 중..."):
        fig2 = _build_network(corp["corp_code"], corp["corp_name"],
                              similar.head(max_n), predictor)
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("노드 색상: 초록(안전) → 빨강(위험)  ·  노드에 마우스를 올리면 상세 정보")


# ════════════════════════════════════════════════════
# 공통: 결과 화면
# ════════════════════════════════════════════════════
def page_result(tab: str, corp: dict, predictor):
    # 뒤로 (검색 화면으로)
    st.markdown('<div class="back-btn">', unsafe_allow_html=True)
    if st.button("← 뒤로", key="back_to_search"):
        st.session_state["corp"] = None
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # 기업 헤더
    sc = DANGER if corp.get("is_bankrupt") else TEAL_D
    st.markdown(f"""
    <div style="background:{CARD};border:1.5px solid {LINE};border-radius:14px;
                padding:12px 20px;margin:10px 0 20px;
                display:flex;align-items:center;justify-content:space-between;">
        <div>
            <span style="font-size:18px;font-weight:700;color:{INK};">
                {corp['corp_name']}</span>
            <span style="margin-left:10px;font-size:12px;color:{SUB};">
                {cls_name(corp.get('corp_cls',''))} · {corp.get('industry_name','') or '업종 미확인'}
            </span>
        </div>
        <div style="background:{sc}12;color:{sc};padding:3px 12px;
                    border-radius:20px;font-size:12px;font-weight:600;border:1px solid {sc}30;">
            {'위기 이력' if corp.get('is_bankrupt') else '정상'}
        </div>
    </div>
    """, unsafe_allow_html=True)

    if tab == "심사":
        render_loan(corp, predictor)
    elif tab == "재무분석":
        render_financial(corp, predictor)
    elif tab == "유사기업":
        render_similar(corp, predictor)


# ════════════════════════════════════════════════════
# 메인
# ════════════════════════════════════════════════════
def main():
    for k, v in [("page", None), ("corp", None)]:
        if k not in st.session_state:
            st.session_state[k] = v

    predictor = get_predictor()
    page = st.session_state["page"]
    corp = st.session_state.get("corp")

    if page is None:
        page_home()
    elif page == "버전관리":
        page_version(predictor)
    elif corp is None:
        page_search(page, predictor)
    else:
        page_result(page, corp, predictor)


main()