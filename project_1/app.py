"""
시계열 예측 웹앱 — app.py
Streamlit 기반 단변량 시계열 CSV 업로드 → 모델 학습 → 결과 대시보드
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

from utils.data_loader import (
    load_csv,
    detect_date_columns,
    detect_value_columns,
    preprocess,
    train_test_split_series,
)
from utils.window_strategy import evaluate_model, run_horizon_sensitivity
from utils.visualizer import (
    plot_series,
    plot_forecast,
    plot_metrics_comparison,
    plot_horizon_sensitivity,
    plot_window_comparison,
)
from models.arima_model import forecast_arima
from models.sarima_model import forecast_sarima
from models.prophet_model import forecast_prophet
from models.lstm_model import forecast_lstm

# ── 페이지 설정 ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="시계열 예측 웹앱",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 시계열 예측 웹앱")
st.caption("단변량 시계열 CSV를 업로드하면 ARIMA / SARIMA / Prophet / LSTM으로 자동 예측합니다.")

# ─────────────────────────────────────────────────────────────────────────────
# 사이드바: 데이터 업로드 & 컬럼 선택
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("① 데이터 업로드")
    uploaded_file = st.file_uploader("CSV 파일을 선택하세요", type=["csv"])

    series: pd.Series | None = None
    date_col = value_col = None

    if uploaded_file:
        df_raw = load_csv(uploaded_file)

        if df_raw is not None:
            st.success(f"로드 완료: {df_raw.shape[0]}행 × {df_raw.shape[1]}열")

            # ── 컬럼 선택 ──────────────────────────────────────────────────
            st.subheader("컬럼 선택")
            date_candidates = detect_date_columns(df_raw)
            value_candidates = detect_value_columns(
                df_raw, date_candidates[0] if date_candidates else ""
            )

            if not date_candidates:
                date_candidates = df_raw.columns.tolist()
            if not value_candidates:
                value_candidates = df_raw.columns.tolist()

            date_col = st.selectbox(
                "날짜 컬럼",
                date_candidates,
                index=0,
                help="자동 감지된 날짜/시간 컬럼",
            )
            value_col = st.selectbox(
                "값 컬럼",
                value_candidates,
                index=0,
                help="예측 대상 수치 컬럼",
            )

            series = preprocess(df_raw, date_col, value_col)

    # ── 예측 설정 ──────────────────────────────────────────────────────────
    st.header("② 예측 설정")

    train_ratio = st.slider(
        "학습 / 테스트 분할 비율",
        min_value=0.5,
        max_value=0.95,
        value=0.8,
        step=0.05,
        help="전체 데이터 중 학습에 사용할 비율",
    )

    horizon = st.number_input(
        "예측 시평 (Forecast Horizon)",
        min_value=1,
        max_value=200,
        value=12,
        step=1,
        help="매 예측 스텝에서 앞을 내다볼 시점 수",
    )

    window_type = st.radio(
        "윈도우 방식",
        options=["expanding", "rolling"],
        index=0,
        format_func=lambda x: "Expanding Window (누적)" if x == "expanding" else "Rolling Window (고정)",
        help="학습 데이터 구성 방식",
    )

    rolling_window_size: int | None = None
    if window_type == "rolling":
        rolling_window_size = st.number_input(
            "Rolling 윈도우 크기",
            min_value=10,
            max_value=500,
            value=50,
            step=5,
            help="Rolling 방식에서 사용할 최근 데이터 포인트 수",
        )

    # ── 모델 선택 ──────────────────────────────────────────────────────────
    st.header("③ 모델 선택")
    use_arima = st.checkbox("ARIMA", value=True)
    use_sarima = st.checkbox("SARIMA", value=False)
    use_prophet = st.checkbox("Prophet", value=False)
    use_lstm = st.checkbox("LSTM", value=False)

    # ── 모델 파라미터 ──────────────────────────────────────────────────────
    st.header("④ 모델 파라미터")

    arima_params: dict = {}
    if use_arima:
        with st.expander("ARIMA 파라미터", expanded=False):
            arima_params = {
                "p": st.number_input("p (AR 차수)", 0, 10, 1, key="a_p"),
                "d": st.number_input("d (차분 차수)", 0, 3, 1, key="a_d"),
                "q": st.number_input("q (MA 차수)", 0, 10, 1, key="a_q"),
            }

    sarima_params: dict = {}
    if use_sarima:
        with st.expander("SARIMA 파라미터", expanded=False):
            sarima_params = {
                "p": st.number_input("p", 0, 5, 1, key="s_p"),
                "d": st.number_input("d", 0, 3, 1, key="s_d"),
                "q": st.number_input("q", 0, 5, 1, key="s_q"),
                "P": st.number_input("P (계절 AR)", 0, 3, 1, key="s_P"),
                "D": st.number_input("D (계절 차분)", 0, 2, 1, key="s_D"),
                "Q": st.number_input("Q (계절 MA)", 0, 3, 1, key="s_Q"),
                "s": st.number_input("s (계절 주기)", 2, 365, 12, key="s_s"),
            }

    prophet_params: dict = {}
    if use_prophet:
        with st.expander("Prophet 파라미터", expanded=False):
            prophet_params = {
                "changepoint_prior_scale": st.slider(
                    "changepoint_prior_scale", 0.001, 0.5, 0.05, 0.001, key="p_cps"
                ),
                "seasonality_mode": st.selectbox(
                    "seasonality_mode",
                    ["additive", "multiplicative"],
                    key="p_sm",
                ),
            }

    lstm_params: dict = {}
    if use_lstm:
        with st.expander("LSTM 파라미터", expanded=False):
            lstm_params = {
                "window_size": st.number_input("window_size (입력 시퀀스 길이)", 2, 100, 10, key="l_ws"),
                "hidden_units": st.number_input("hidden_units", 8, 512, 64, key="l_hu"),
                "epochs": st.number_input("epochs", 1, 200, 20, key="l_ep"),
                "batch_size": st.number_input("batch_size", 4, 256, 32, key="l_bs"),
            }

    # ── 시평 민감도 분석 설정 ───────────────────────────────────────────────
    st.header("⑤ 시평 민감도 분석")
    run_sensitivity = st.checkbox("시평 민감도 분석 실행", value=False,
                                  help="여러 Horizon에 대해 성능을 계산합니다. 시간이 추가로 소요됩니다.")
    if run_sensitivity:
        h_min = st.number_input("최소 Horizon", 1, 50, 1, key="hs_min")
        h_max = st.number_input("최대 Horizon", 2, 200, 24, key="hs_max")
        h_step = st.number_input("Horizon 간격", 1, 50, 4, key="hs_step")

    # ── 실행 버튼 ──────────────────────────────────────────────────────────
    st.divider()
    run_btn = st.button("🚀 예측 실행", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# 메인 화면
# ─────────────────────────────────────────────────────────────────────────────
if series is None:
    st.info("왼쪽 사이드바에서 CSV 파일을 업로드하면 분석을 시작할 수 있습니다.")
    st.stop()

# ── 탭 구성 ───────────────────────────────────────────────────────────────────
tab_eda, tab_result, tab_compare, tab_horizon = st.tabs(
    ["📊 데이터 탐색 (EDA)", "🔮 예측 결과", "📋 성능 비교", "📉 시평 분석"]
)

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 1: EDA
# ═══════════════════════════════════════════════════════════════════════════════
with tab_eda:
    st.subheader("데이터 미리보기")
    train_s, test_s = train_test_split_series(series, train_ratio)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("전체 데이터 포인트", len(series))
    col2.metric("학습 데이터", len(train_s))
    col3.metric("테스트 데이터", len(test_s))
    col4.metric("설정 Horizon", horizon)

    st.plotly_chart(
        plot_series(series, title=f"{value_col} 시계열"),
        use_container_width=True,
    )

    st.subheader("기초 통계")
    desc = series.describe().rename("통계값").to_frame()
    desc.index.name = "항목"
    st.dataframe(desc.style.format("{:.4f}"), use_container_width=False)

    st.subheader("원본 데이터 (처음 100행)")
    st.dataframe(
        pd.DataFrame({"날짜": series.index[:100], "값": series.values[:100]}),
        use_container_width=True,
        height=280,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# 예측 실행 (버튼 클릭 시)
# ═══════════════════════════════════════════════════════════════════════════════
if not run_btn:
    with tab_result:
        st.info("사이드바에서 모델을 선택하고 '예측 실행' 버튼을 누르세요.")
    with tab_compare:
        st.info("예측을 먼저 실행하세요.")
    with tab_horizon:
        st.info("예측을 먼저 실행하세요.")
    st.stop()

# 선택된 모델 확인
selected_models: dict[str, tuple] = {}
if use_arima:
    selected_models["ARIMA"] = (forecast_arima, arima_params)
if use_sarima:
    selected_models["SARIMA"] = (forecast_sarima, sarima_params)
if use_prophet:
    selected_models["Prophet"] = (forecast_prophet, prophet_params)
if use_lstm:
    selected_models["LSTM"] = (forecast_lstm, lstm_params)

if not selected_models:
    st.error("최소 하나의 모델을 선택해주세요.")
    st.stop()

# ── 학습 & 평가 ───────────────────────────────────────────────────────────────
pred_results: dict[str, dict] = {}
metrics_expanding: dict[str, dict] = {}
metrics_rolling: dict[str, dict] = {}
train_size = int(len(series) * train_ratio)

progress_bar = st.progress(0, text="모델 학습 중...")
total_models = len(selected_models)

for i, (model_name, (fn, params)) in enumerate(selected_models.items()):
    progress_bar.progress(
        int((i / total_models) * 100),
        text=f"{model_name} 학습 중...",
    )

    with st.spinner(f"{model_name} 평가 중..."):
        # ── Expanding Window 평가 (메인)
        result = evaluate_model(
            series,
            train_ratio,
            int(horizon),
            fn,
            params,
            window_type=window_type,
            rolling_window_size=int(rolling_window_size) if rolling_window_size else None,
        )
        if result:
            pred_results[model_name] = result
            if window_type == "expanding":
                metrics_expanding[model_name] = result["metrics"]
            else:
                metrics_rolling[model_name] = result["metrics"]

        # ── Window 비교를 위해 반대 방식도 실행
        if window_type == "expanding":
            r2 = evaluate_model(
                series, train_ratio, int(horizon), fn, params,
                window_type="rolling",
                rolling_window_size=int(rolling_window_size) if rolling_window_size else None,
            )
            if r2:
                metrics_rolling[model_name] = r2["metrics"]
        else:
            r2 = evaluate_model(
                series, train_ratio, int(horizon), fn, params,
                window_type="expanding",
            )
            if r2:
                metrics_expanding[model_name] = r2["metrics"]

progress_bar.progress(100, text="완료!")

if not pred_results:
    st.error("모든 모델 평가에 실패했습니다. 데이터 크기나 파라미터를 확인해주세요.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 2: 예측 결과
# ═══════════════════════════════════════════════════════════════════════════════
with tab_result:
    st.subheader("실제값 vs 예측값")
    st.plotly_chart(
        plot_forecast(series, train_size, pred_results),
        use_container_width=True,
    )

    for model_name, res in pred_results.items():
        with st.expander(f"{model_name} 예측 상세"):
            detail_df = pd.DataFrame(
                {
                    "날짜": res["pred_index"],
                    "실제값": res["actuals"],
                    "예측값": res["predictions"],
                    "오차": res["actuals"] - res["predictions"],
                }
            )
            st.dataframe(detail_df.style.format({"실제값": "{:.4f}", "예측값": "{:.4f}", "오차": "{:.4f}"}),
                         use_container_width=True, height=250)

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 3: 성능 비교
# ═══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    all_metrics = {m: r["metrics"] for m, r in pred_results.items()}

    st.subheader("성능 지표 테이블")
    metrics_df = pd.DataFrame(all_metrics).T.rename_axis("모델")
    st.dataframe(
        metrics_df.style.format("{:.4f}").highlight_min(axis=0, color="#d4edda"),
        use_container_width=True,
    )

    st.subheader("성능 지표 바 차트")
    st.plotly_chart(plot_metrics_comparison(all_metrics), use_container_width=True)

    st.subheader("Rolling vs Expanding Window 비교")
    if metrics_rolling and metrics_expanding:
        metric_sel = st.selectbox("비교 지표 선택", ["MAE", "RMSE", "SMAPE", "MASE"], key="wc_metric")
        st.plotly_chart(
            plot_window_comparison(metrics_rolling, metrics_expanding, metric=metric_sel),
            use_container_width=True,
        )
    else:
        st.info("Rolling 및 Expanding 두 방식 모두의 결과가 있을 때 비교 차트가 표시됩니다.")

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 4: 시평 민감도 분석
# ═══════════════════════════════════════════════════════════════════════════════
with tab_horizon:
    if not run_sensitivity:
        st.info("사이드바에서 '시평 민감도 분석 실행'을 체크하고 예측을 다시 실행하세요.")
    else:
        horizons_to_test = list(range(int(h_min), int(h_max) + 1, int(h_step)))
        if len(horizons_to_test) < 2:
            st.warning("최소 2개 이상의 Horizon이 필요합니다. 간격을 줄여보세요.")
        else:
            horizon_metrics_all: dict[str, dict[int, dict[str, float]]] = {}

            hs_bar = st.progress(0, text="시평 민감도 분석 중...")
            for i, (model_name, (fn, params)) in enumerate(selected_models.items()):
                hs_bar.progress(
                    int((i / total_models) * 100),
                    text=f"{model_name} 시평 분석 중...",
                )
                with st.spinner(f"{model_name} 시평 민감도 계산 중..."):
                    hm = run_horizon_sensitivity(
                        series,
                        train_ratio,
                        horizons_to_test,
                        fn,
                        params,
                        window_type=window_type,
                        rolling_window_size=int(rolling_window_size) if rolling_window_size else None,
                    )
                    if hm:
                        horizon_metrics_all[model_name] = hm

            hs_bar.progress(100, text="완료!")

            if horizon_metrics_all:
                st.subheader("시평에 따른 성능 변화")
                st.plotly_chart(
                    plot_horizon_sensitivity(horizon_metrics_all),
                    use_container_width=True,
                )

                # 수치 테이블
                st.subheader("시평별 성능 수치")
                rows = []
                for model_name, hm in horizon_metrics_all.items():
                    for h_val, m in hm.items():
                        rows.append({"모델": model_name, "Horizon": h_val, **m})
                if rows:
                    hs_df = pd.DataFrame(rows).set_index(["모델", "Horizon"])
                    st.dataframe(hs_df.style.format("{:.4f}"), use_container_width=True)
            else:
                st.warning("시평 민감도 분석 결과가 없습니다. 데이터 크기를 확인하세요.")
