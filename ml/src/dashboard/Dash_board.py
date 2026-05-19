"""
심사관용 대시보드
실행: python -m streamlit run src/dashboard/Dash_board.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import sys, os
import plotly.express as px
import plotly.graph_objects as go

# predictor 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'prediction'))
from predictor import Predictor, calc_interest_rate, calc_loan_limit, calc_expected_loss

DB_PATH = "db/bankruptcy_prediction.db"

st.set_page_config(
    page_title="기업 대출 심사 시스템",
    page_icon="🏦",
    layout="wide"
)

# ── 데이터 / 모델 로드 ────────────────────────────────────────────────
@st.cache_resource
def get_predictor():
    return Predictor()


@st.cache_data
def search_companies(keyword):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT c.corp_code, c.corp_name, c.market, c.industry_code,
               CASE WHEN b.corp_code IS NOT NULL THEN 1 ELSE 0 END as is_bankrupt
        FROM companies c
        LEFT JOIN bankruptcy b ON c.corp_code = b.corp_code
        WHERE c.corp_name LIKE ? AND c.market IN ('K', 'Y')
        ORDER BY c.corp_name
        LIMIT 20
    """, conn, params=(f"%{keyword}%",))
    conn.close()
    return df


@st.cache_data
def load_financial_full(corp_code):
    """기업의 전체 재무 데이터 로드"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT * FROM financials WHERE corp_code = ? ORDER BY year
    """, conn, params=(corp_code,))
    conn.close()
    return df


@st.cache_data
def load_financial_sequence(corp_code, feature_cols):
    """모델 입력용 시퀀스"""
    conn = sqlite3.connect(DB_PATH)

    fin_cols = ['debt_ratio', 'current_ratio', 'interest_coverage',
                'net_debt_ratio', 'equity_ratio',
                'roa', 'roe', 'op_margin', 'net_margin',
                'cfo_to_debt', 'fcf',
                'revenue_growth', 'op_income_growth', 'asset_growth',
                'interest_cov_yoy', 'debt_ratio_trend', 'cf_volatility',
                'consecutive_loss', 'z_score']
    selected_fin = [c for c in feature_cols if c in fin_cols]
    cols = ', '.join(selected_fin)

    df = pd.read_sql(f"""
        SELECT year, {cols}, total_assets, operating_cf
        FROM financials WHERE corp_code = ? ORDER BY year
    """, conn, params=(corp_code,))

    news_df = pd.read_sql("""
        SELECT AVG(sentiment_avg) as sentiment_avg,
               AVG(negative_ratio) as negative_ratio,
               SUM(news_count) as news_count,
               MAX(news_bankruptcy_flag) as news_bankruptcy_flag,
               MAX(news_lawsuit_flag) as news_lawsuit_flag
        FROM news_sentiment WHERE corp_code = ?
    """, conn, params=(corp_code,))
    conn.close()

    news_cols = ['sentiment_avg', 'negative_ratio', 'news_count',
                 'news_bankruptcy_flag', 'news_lawsuit_flag']
    for col in news_cols:
        if col in feature_cols:
            df[col] = news_df[col].iloc[0] if len(news_df) > 0 else 0
            df[col] = df[col].fillna(0)
    return df


@st.cache_data
def load_similar_companies(corp_code):
    """유사 기업 (relationships 테이블에서)"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT r.corp_code_to as corp_code, c.corp_name, c.market,
               r.edge_type, r.weight, r.detail,
               CASE WHEN b.corp_code IS NOT NULL THEN 1 ELSE 0 END as is_bankrupt
        FROM relationships r
        JOIN companies c ON r.corp_code_to = c.corp_code
        LEFT JOIN bankruptcy b ON r.corp_code_to = b.corp_code
        WHERE r.corp_code_from = ?
        ORDER BY r.weight DESC
        LIMIT 50
    """, conn, params=(corp_code,))
    conn.close()
    return df


def predict_pd(corp_code, predictor):
    """파산 확률 예측"""
    feature_cols = predictor.meta['feature_cols']
    fin_df = load_financial_sequence(corp_code, feature_cols)

    if len(fin_df) == 0:
        return None

    seq_len = predictor.meta['seq_len']
    feats = fin_df[feature_cols].fillna(0).values.astype(np.float32)

    if len(feats) >= seq_len:
        sequence = feats[-seq_len:]
    else:
        pad = np.zeros((seq_len - len(feats), len(feature_cols)), dtype=np.float32)
        sequence = np.vstack([pad, feats])

    return predictor.predict(sequence)


# ── 사이드바: 모델 설정 ──────────────────────────────────────────────
st.sidebar.title("⚙️ 시스템 설정")
predictor = get_predictor()
available_models = predictor.list_models()
model_names = [m['model_name'] for m in available_models]

if not model_names:
    st.sidebar.error("학습된 모델이 없습니다.")
    st.stop()

selected_model = st.sidebar.selectbox(
    "예측 모델",
    model_names,
    index=model_names.index(predictor.active_name)
)

current_meta = next((m for m in available_models if m['model_name'] == selected_model), None)
if current_meta:
    custom_thr = st.sidebar.slider(
        "Threshold", 0.0, 1.0,
        float(current_meta['threshold']), 0.01
    )
    if st.sidebar.button("모델 적용"):
        predictor.set_active(selected_model, custom_thr)
        st.cache_resource.clear()
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**활성 모델 성능**")
    st.sidebar.metric("PR-AUC", f"{current_meta['pr_auc']:.4f}")
    st.sidebar.metric("F1",     f"{current_meta['f1']:.4f}")


# ── 세션 상태 ────────────────────────────────────────────────────────
if 'selected_corp' not in st.session_state:
    st.session_state.selected_corp = None
if 'menu' not in st.session_state:
    st.session_state.menu = None


# ── 메인 화면 ────────────────────────────────────────────────────────
st.title("🏦 기업 대출 심사 시스템")
st.caption(f"활성 모델: **{predictor.active_name.upper()}**  |  심사 기준 PD: **{predictor.meta['threshold']*100:.1f}%**")

# 검색 영역
st.markdown("### 🔍 기업 검색")
col1, col2 = st.columns([4, 1])
with col1:
    keyword = st.text_input("기업명 입력", placeholder="예: 삼성전자, LG, 현대...", key="search_input")
with col2:
    st.write("")
    st.write("")
    search_clicked = st.button("검색", use_container_width=True)

# 검색 결과
if keyword and (search_clicked or keyword):
    results = search_companies(keyword)

    if len(results) == 0:
        st.error("❌ 일치하는 기업이 없습니다.")
    else:
        st.success(f"✅ 일치하는 기업이 **{len(results)}개** 있습니다.")

        # 기업 선택
        company_options = [
            f"{row['corp_name']} ({'코스피' if row['market']=='Y' else '코스닥'})"
            for _, row in results.iterrows()
        ]
        idx = st.selectbox("기업 선택", range(len(company_options)),
                            format_func=lambda i: company_options[i])
        selected = results.iloc[idx]
        st.session_state.selected_corp = {
            'corp_code': selected['corp_code'],
            'corp_name': selected['corp_name'],
            'market':    selected['market'],
            'industry_code': selected['industry_code'],
            'is_bankrupt': selected['is_bankrupt'],
        }


# ── 기업 선택 후 메뉴 ────────────────────────────────────────────────
if st.session_state.selected_corp:
    corp = st.session_state.selected_corp
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("선택 기업", corp['corp_name'])
    with col2:
        st.metric("시장", "코스피" if corp['market'] == 'Y' else "코스닥")
    with col3:
        st.metric("업종 코드", corp['industry_code'] or "N/A")
    with col4:
        st.metric("실제 상태",
                  "🔴 파산" if corp['is_bankrupt'] else "🟢 정상")

    st.markdown("### 📋 심사 메뉴")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("💰 대출 심사", use_container_width=True, type="primary"):
            st.session_state.menu = "loan"
    with col2:
        if st.button("📊 재무제표 확인", use_container_width=True):
            st.session_state.menu = "financial"
    with col3:
        if st.button("📦 포트폴리오", use_container_width=True):
            st.session_state.menu = "portfolio"
    with col4:
        if st.button("🔗 유사 기업 분석", use_container_width=True):
            st.session_state.menu = "similar"


# ═══════════════════════════════════════════════════════════════════
# 메뉴 1: 대출 심사
# ═══════════════════════════════════════════════════════════════════
if st.session_state.selected_corp and st.session_state.menu == "loan":
    corp = st.session_state.selected_corp
    st.markdown("---")
    st.subheader(f"💰 대출 심사 — {corp['corp_name']}")

    # PD 예측
    pred = predict_pd(corp['corp_code'], predictor)
    if pred is None:
        st.error("재무 데이터가 없는 기업입니다.")
    else:
        pd_score = pred['probability']

        # 심사 조건 입력
        st.markdown("##### 📝 심사 조건 입력")
        col_a, col_b = st.columns(2)

        with col_a:
            loan_amount = st.number_input(
                "신청 금액 (억원)",
                min_value=1.0, max_value=10000.0,
                value=100.0, step=10.0
            )
            loan_period = st.slider(
                "대출 기간 (년)", 1, 10, 3
            )
            industry_risk = st.selectbox(
                "업종 리스크 등급",
                ["low", "medium", "high"],
                index=1,
                format_func=lambda x: {"low": "저위험", "medium": "보통", "high": "고위험"}[x]
            )

        with col_b:
            has_collateral = st.checkbox("담보 제공")
            collateral_value = 0.0
            if has_collateral:
                collateral_type = st.selectbox(
                    "담보 유형",
                    ["부동산", "기계설비", "신용보증서"]
                )
                collateral_value = st.number_input(
                    "담보 가치 (억원)",
                    min_value=0.0, max_value=10000.0,
                    value=50.0, step=10.0
                )

            has_prev_loan = st.checkbox("이전 대출 이력 있음")
            prev_loan_status = "양호"
            if has_prev_loan:
                prev_loan_status = st.selectbox(
                    "이전 대출 상환 상태",
                    ["양호", "연체 경험", "부실"]
                )

        # 심사 결과
        st.markdown("---")
        st.markdown("##### 🎯 심사 결과")

        col1, col2, col3 = st.columns(3)
        with col1:
            risk_color = "🔴" if pd_score >= 0.5 else "🟡" if pd_score >= 0.2 else "🟢"
            st.metric(f"{risk_color} 파산 확률 (PD)", f"{pd_score*100:.2f}%")
        with col2:
            risk_grade = "고위험" if pd_score >= 0.5 else "주의" if pd_score >= 0.2 else "안전"
            st.metric("위험 등급", risk_grade)
        with col3:
            if not has_collateral:
                collateral_type = None
            el, applied_lgd = calc_expected_loss(
                pd_score, loan_amount,
                has_collateral=has_collateral,
                collateral_type=collateral_type if has_collateral else None,
                collateral_value=collateral_value
            )
            st.metric("Expected Loss", f"{el:.2f} 억",
              help=f"LGD: {applied_lgd*100:.0f}%")
    

        # 이전 대출 이력 페널티
        loan_penalty = 0.0
        if has_prev_loan:
            if prev_loan_status == "연체 경험":
                loan_penalty = 0.005
            elif prev_loan_status == "부실":
                loan_penalty = 0.015
                pd_score = min(pd_score * 1.3, 1.0)  # PD 가산

        # 금리 산출
        rate = calc_interest_rate(pd_score, industry_risk, has_collateral)

        st.markdown("---")
        if rate is None or pd_score >= 0.5:
            st.error("### ⛔ 대출 거절")
            st.markdown(f"""
            **거절 사유:**
            - 파산 확률 {pd_score*100:.1f}%로 위험 임계값 50% 초과
            - 신용 위험이 매우 높음
            """)
        else:
            rate_total = rate + loan_penalty
            conn = sqlite3.connect(DB_PATH)
            latest = conn.execute("""
                SELECT total_assets, operating_cf , revenue FROM financials
                WHERE corp_code = ? ORDER BY year DESC LIMIT 1
            """, (corp['corp_code'],)).fetchone()
            conn.close()

            operating_cf = (latest[1] / 100) if latest and latest[1] else 1.0
            revenue_val  = (latest[2] / 100) if latest and len(latest) > 2 and latest[2] else None
            total_assets_val = (latest[0] / 100) if latest and latest[0] else 10.0

            limit = calc_loan_limit(
                operating_cf=abs(operating_cf),
                total_assets=abs(total_assets_val),
                pd_score=pd_score,
                collateral_value=collateral_value,  # 사용자 입력 (이미 억원)
                loan_period_years=loan_period,
                interest_rate=rate if rate else 0.05,
                revenue=abs(revenue_val) if revenue_val else None
            )

            if loan_amount > limit:
                st.warning(f"### ⚠️ 조건부 승인")
                st.markdown(f"""
                **신청 금액 {loan_amount:.0f}억 → 권장 한도 {limit:.1f}억으로 감액 후 승인 가능**
                """)
            else:
                st.success(f"### ✅ 대출 승인")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("적정 금리", f"{rate_total*100:.2f}%")
            with col2:
                st.metric("권장 한도", f"{limit:.1f} 억")
            with col3:
                st.metric("승인 가능 금액", f"{min(loan_amount, limit):.1f} 억")

            # 금리 구성
            with st.expander("📌 금리 산출 상세"):
                # 변수 미리 계산
                industry_premium = {"low": 0, "medium": 0.003, "high": 0.005}[industry_risk]
                collateral_disc  = -0.005 if has_collateral else 0.0
                credit_spread    = rate - 0.035 - industry_premium - collateral_disc

                st.markdown(f"""
| 항목 | 값 |
|------|-----|
| 기준금리 | 3.50% |
| 신용 스프레드 (PD {pd_score*100:.1f}%) | {credit_spread*100:.2f}% |
| 업종 가산 ({industry_risk}) | {industry_premium*100:.2f}% |
| 담보 할인 | {collateral_disc*100:.2f}% |
| 이전 대출 페널티 | {loan_penalty*100:.2f}% |
| **최종 금리** | **{rate_total*100:.2f}%** |
""")


# ═══════════════════════════════════════════════════════════════════
# 메뉴 2: 재무제표 확인
# ═══════════════════════════════════════════════════════════════════
if st.session_state.selected_corp and st.session_state.menu == "financial":
    corp = st.session_state.selected_corp
    st.markdown("---")
    st.subheader(f"📊 재무제표 분석 — {corp['corp_name']}")

    fin_df = load_financial_full(corp['corp_code'])

    if len(fin_df) == 0:
        st.error("재무 데이터가 없습니다.")
    else:
        # 최신 연도 주요 지표
        latest = fin_df.iloc[-1]
        st.markdown(f"##### 최신 연도: {int(latest['year'])}년")

        def safe_num(val, default=0):
            """None/NaN 안전 처리"""
            if val is None or pd.isna(val):
                return default
            return val

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("총자산", f"{safe_num(latest.get('total_assets'))/100:.1f} 억")
            st.metric("매출액", f"{safe_num(latest.get('revenue'))/100:.1f} 억")
        with col2:
            st.metric("부채비율", f"{safe_num(latest.get('debt_ratio'))*100:.1f}%")
            st.metric("ROA", f"{safe_num(latest.get('roa'))*100:.2f}%")
        with col3:
            st.metric("유동비율", f"{safe_num(latest.get('current_ratio'))*100:.1f}%")
            st.metric("이자보상배율", f"{safe_num(latest.get('interest_coverage')):.2f}")
        with col4:
            st.metric("Altman Z-Score", f"{safe_num(latest.get('z_score')):.2f}")
            st.metric("영업CF", f"{safe_num(latest.get('operating_cf'))/100:.1f} 억")

        st.markdown("---")
        st.markdown("##### 📈 재무 지표 추이")

        # 4개 차트 (탭)
        tab1, tab2, tab3, tab4 = st.tabs(["수익성", "안정성", "유동성", "성장성"])

        with tab1:
            cols = ['roa', 'roe', 'op_margin', 'net_margin']
            available = [c for c in cols if c in fin_df.columns]
            if available:
                chart_df = fin_df[['year'] + available].melt(
                    id_vars='year', var_name='지표', value_name='값'
                )
                fig = px.line(chart_df, x='year', y='값', color='지표',
                              markers=True, height=400)
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            cols = ['debt_ratio', 'net_debt_ratio', 'equity_ratio', 'z_score']
            available = [c for c in cols if c in fin_df.columns]
            if available:
                chart_df = fin_df[['year'] + available].melt(
                    id_vars='year', var_name='지표', value_name='값'
                )
                fig = px.line(chart_df, x='year', y='값', color='지표',
                              markers=True, height=400)
                st.plotly_chart(fig, use_container_width=True)

        with tab3:
            cols = ['current_ratio', 'interest_coverage', 'cfo_to_debt']
            available = [c for c in cols if c in fin_df.columns]
            if available:
                chart_df = fin_df[['year'] + available].melt(
                    id_vars='year', var_name='지표', value_name='값'
                )
                fig = px.line(chart_df, x='year', y='값', color='지표',
                              markers=True, height=400)
                st.plotly_chart(fig, use_container_width=True)

        with tab4:
            cols = ['revenue_growth', 'op_income_growth', 'asset_growth']
            available = [c for c in cols if c in fin_df.columns]
            if available:
                chart_df = fin_df[['year'] + available].melt(
                    id_vars='year', var_name='지표', value_name='값'
                )
                fig = px.line(chart_df, x='year', y='값', color='지표',
                              markers=True, height=400)
                st.plotly_chart(fig, use_container_width=True)

        # 상세 데이터 테이블
        with st.expander("📋 전체 재무 데이터 보기"):
            display_cols = ['year', 'total_assets', 'total_debt', 'revenue',
                            'operating_income', 'net_income', 'debt_ratio',
                            'roa', 'z_score']
            available = [c for c in display_cols if c in fin_df.columns]
            st.dataframe(fin_df[available].round(3), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# 메뉴 3: 포트폴리오
# ═══════════════════════════════════════════════════════════════════
if st.session_state.selected_corp and st.session_state.menu == "portfolio":
    st.markdown("---")
    st.subheader("📦 포트폴리오 리스크 분석")

    st.markdown("##### 여러 기업을 동시에 평가하여 포트폴리오 전체 리스크 산정")

    # 기업 검색 (다중)
    portfolio_keyword = st.text_input("기업 추가 검색")
    if portfolio_keyword:
        port_results = search_companies(portfolio_keyword)
        if len(port_results) > 0:
            port_options = port_results['corp_name'].tolist()
            selected_portfolio = st.multiselect(
                "포트폴리오 기업 선택",
                port_options,
                default=[st.session_state.selected_corp['corp_name']]
                        if st.session_state.selected_corp['corp_name'] in port_options else []
            )

            if selected_portfolio:
                feature_cols = predictor.meta['feature_cols']
                seq_len      = predictor.meta['seq_len']
                results      = []

                for name in selected_portfolio:
                    corp_row = port_results[port_results['corp_name']==name].iloc[0]
                    pred = predict_pd(corp_row['corp_code'], predictor)
                    if pred is None:
                        continue
                    results.append({
                        '기업명':   name,
                        '시장':     '코스피' if corp_row['market']=='Y' else '코스닥',
                        'PD (%)':   round(pred['probability']*100, 2),
                        '위험등급': '🔴 고위험' if pred['probability'] >= 0.5
                                  else '🟡 주의' if pred['probability'] >= 0.2
                                  else '🟢 안전',
                    })

                if results:
                    result_df = pd.DataFrame(results).sort_values('PD (%)', ascending=False)
                    st.dataframe(result_df, use_container_width=True, hide_index=True)

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("총 기업", len(result_df))
                    with col2:
                        st.metric("평균 PD", f"{result_df['PD (%)'].mean():.2f}%")
                    with col3:
                        high = (result_df['PD (%)'] >= 50).sum()
                        st.metric("고위험", f"{high}개")
                    with col4:
                        warn = ((result_df['PD (%)'] >= 20) & (result_df['PD (%)'] < 50)).sum()
                        st.metric("주의", f"{warn}개")

                    fig = px.histogram(result_df, x='PD (%)', nbins=20,
                                        title="포트폴리오 PD 분포",
                                        color_discrete_sequence=['#FF6B6B'])
                    fig.add_vline(x=20, line_dash="dash", line_color="orange")
                    fig.add_vline(x=50, line_dash="dash", line_color="red")
                    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# 메뉴 4: 유사 기업
# ═══════════════════════════════════════════════════════════════════
if st.session_state.selected_corp and st.session_state.menu == "similar":
    corp = st.session_state.selected_corp
    st.markdown("---")
    st.subheader(f"🔗 유사 기업 분석 — {corp['corp_name']}")

    similar = load_similar_companies(corp['corp_code'])

    if len(similar) == 0:
        st.warning("관계망 데이터가 없습니다.")
    else:
        st.markdown(f"##### 유사 기업 총 {len(similar)}개")

        # 엣지 유형별 필터
        edge_types = similar['edge_type'].unique().tolist()
        selected_type = st.radio(
            "관계 유형",
            ['전체'] + edge_types,
            horizontal=True,
            format_func=lambda x: {
                '전체':          '전체',
                'industry':      '동일 업종',
                'financial_sim': '재무 유사도',
                'size':          '동일 규모'
            }.get(x, x)
        )

        if selected_type != '전체':
            filtered = similar[similar['edge_type'] == selected_type]
        else:
            filtered = similar

        # 각 유사 기업의 PD 예측
        st.markdown("##### 유사 기업 파산 위험도")

        with st.spinner("유사 기업 분석 중..."):
            similar_pds = []
            for _, row in filtered.head(20).iterrows():
                pred = predict_pd(row['corp_code'], predictor)
                if pred is not None:
                    similar_pds.append({
                        '기업명':   row['corp_name'],
                        '관계':     {'industry':'동일업종', 'financial_sim':'재무유사', 'size':'동일규모'}.get(row['edge_type'], row['edge_type']),
                        '가중치':   round(row['weight'], 3),
                        '실제':     '🔴 파산' if row['is_bankrupt'] else '🟢 정상',
                        'PD (%)':   round(pred['probability']*100, 2),
                        '예측':     '파산' if pred['label']==1 else '정상',
                    })

        if similar_pds:
            sim_df = pd.DataFrame(similar_pds).sort_values('PD (%)', ascending=False)
            st.dataframe(sim_df, use_container_width=True, hide_index=True)

            # 분석 인사이트
            avg_pd = sim_df['PD (%)'].mean()
            own_pred = predict_pd(corp['corp_code'], predictor)
            own_pd = own_pred['probability']*100 if own_pred else 0

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("선택 기업 PD", f"{own_pd:.2f}%")
            with col2:
                st.metric("유사 기업 평균 PD", f"{avg_pd:.2f}%")
            with col3:
                diff = own_pd - avg_pd
                st.metric("차이", f"{diff:+.2f}%p",
                          delta_color="inverse")

            if own_pd > avg_pd + 5:
                st.warning(f"⚠️ 유사 기업 평균보다 PD가 **{diff:.1f}%p 높음** → 추가 주의 필요")
            elif own_pd < avg_pd - 5:
                st.success(f"✅ 유사 기업 평균보다 PD가 **{abs(diff):.1f}%p 낮음** → 상대적 안정")
            else:
                st.info("유사 기업과 비슷한 위험도 수준")


# ── 푸터 ─────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("기업 대출 심사 시스템 v2.0 | LSTM/CNN-LSTM/TFT + GraphSAGE")