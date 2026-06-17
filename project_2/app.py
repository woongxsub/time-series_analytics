import hashlib
import io

import numpy as np
import pandas as pd
import streamlit as st

from detector import detect_anomalies
from preprocessor import extract_label_series, preprocess, stl_decompose
from visualizer import (
    display_anomaly_detail_stats,
    display_anomaly_table,
    display_metrics,
    plot_anomaly_detail,
    plot_roc_curves,
    plot_score_histogram,
    plot_scorer_comparison,
    plot_series_with_anomalies,
    plot_stl_decomposition,
)

st.set_page_config(
    page_title="시계열 이상탐지 대시보드",
    layout="wide",
    page_icon="📊",
)

st.title("📊 시계열 이상탐지 대시보드")

for key, default in [("results", None), ("file_hash", None)]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── 헬퍼 함수 ────────────────────────────────────────────────────────────────

def _md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _recommendation_message(ratio: float) -> str:
    if ratio == 0:
        return (
            "이상이 탐지되지 않았습니다.  \n"
            "임계치를 낮추거나 다른 Aggregator로 재분석을 권장합니다."
        )
    elif ratio < 1:
        return (
            f"탐지된 이상 비율이 **{ratio:.1f}%** 로 매우 낮습니다.  \n"
            "실제 이상이 탐지되지 않았을 가능성이 있으므로  \n"
            "임계치 하향 또는 Scorer 변경 후 재분석을 권장합니다."
        )
    elif ratio < 3:
        return (
            f"탐지된 이상 비율이 **{ratio:.1f}%** 로 보수적인 수준입니다.  \n"
            "중요도가 높은 이상만 탐지되고 있습니다.  \n"
            "더 많은 이상을 탐지하려면 임계치 하향을 고려하세요."
        )
    elif ratio <= 10:
        return (
            f"탐지된 이상 비율이 **{ratio:.1f}%** 로 적절한 수준입니다.  \n"
            "현재 임계치 설정을 유지하고  \n"
            "탐지된 구간에 대한 도메인 검토를 권장합니다."
        )
    else:
        return (
            f"탐지된 이상 비율이 **{ratio:.1f}%** 로 높은 수준입니다.  \n"
            "정상 패턴과 유사한 구간이 이상으로 탐지되었을 가능성이 있으므로  \n"
            "임계치 상향 또는 추가 검토를 권장합니다."
        )


def _realtime_feedback(results_dict: dict, high_quantile: float):
    """기존 raw scores에 새 임계치를 적용해 예상 탐지 수를 즉시 표시."""
    lines = []
    total_n, total_pts = 0, 0

    for col, res in results_dict.items():
        if res.get("strategy") != "without_label":
            continue
        scores = res.get("scores")
        if not scores:
            continue
        vals = scores[0].univariate_values()
        clean = vals[~np.isnan(vals)]
        if len(clean) == 0:
            continue
        thr = float(np.quantile(clean, high_quantile))
        n_anom = int(np.sum(clean > thr))
        ratio = n_anom / len(clean) * 100
        lines.append(f"- `{col}`: {n_anom}개 ({ratio:.1f}%)")
        total_n += n_anom
        total_pts += len(clean)

    if not lines:
        return

    st.markdown("**📊 현재 임계치 예상 탐지**")
    for l in lines:
        st.markdown(l)

    if total_pts > 0:
        total_ratio = total_n / total_pts * 100
        st.info(_recommendation_message(total_ratio))


def run_column_analysis(df, time_col, col, label_col, aggregator_type, high_quantile):
    """단일 타겟 컬럼에 대한 전체 분석 파이프라인."""
    try:
        train, test, full_series, freq, lags = preprocess(df, time_col, col)
    except ValueError as e:
        st.error(f"[{col}] 전처리 오류: {e}")
        return None

    label_series = None
    if label_col:
        try:
            label_series = extract_label_series(df, time_col, label_col)
        except ValueError as e:
            st.error(f"[{col}] 레이블 전처리 오류: {e}")
            return None

    try:
        result = detect_anomalies(
            train=train, test=test, lags=lags,
            label_series=label_series,
            aggregator_type=aggregator_type,
            high_quantile=high_quantile,
        )
    except Exception as e:
        st.error(f"[{col}] 이상탐지 오류: {e}")
        return None

    # scores는 lags 워밍업 때문에 test보다 늦게 시작 → 공통 시작점으로 맞춤
    if result.get("scores"):
        common_start = max(s.start_time() for s in result["scores"])
        test_display = test.slice(common_start, test.end_time())
    else:
        test_display = test

    result.update({"test_series": test_display, "full_series": full_series, "freq": freq, "lags": lags})

    # STL 분해 + 잔차 이상탐지
    try:
        trend, seasonal, residual = stl_decompose(full_series, freq)
        result["stl"] = {"trend": trend, "seasonal": seasonal, "residual": residual}

        split_idx = int(len(residual) * 0.8)
        train_res = residual[:split_idx]
        test_res = residual[split_idx:]

        res_result = detect_anomalies(
            train=train_res, test=test_res, lags=lags,
            label_series=None,
            aggregator_type=aggregator_type,
            high_quantile=high_quantile,
        )
        if res_result.get("scores"):
            res_common_start = max(s.start_time() for s in res_result["scores"])
            test_res_display = test_res.slice(res_common_start, test_res.end_time())
        else:
            test_res_display = test_res
        res_result["test_series"] = test_res_display
        result["residual_result"] = res_result
    except Exception as e:
        result["stl"] = None
        result["residual_result"] = None
        result["stl_error"] = str(e)

    return result


def _display_column_tab(col: str, res: dict):
    """컬럼별 탭 안에 전체 대시보드 렌더링."""
    test_series = res["test_series"]
    anomaly_segments = res["anomaly_segments"]
    scores = res["scores"]
    scorer_names = res["scorer_names"]
    best_idx = res.get("best_scorer_idx")

    # ── 전략 배너 ─────────────────────────────────────────────────────────────
    if res["strategy"] == "with_label":
        st.success(
            f"✅ **{res['best_scorer_name']}** 가 AUC **{res['best_auc']:.4f}** 로 자동 선택됨"
            f"  |  최적 임계치: `{res['optimal_threshold']:.4f}`"
        )
    else:
        st.info("레이블 없음 모드 — Aggregator + 분위수 임계치 적용")

    # ── 요약 메트릭 ───────────────────────────────────────────────────────────
    st.subheader("📈 분석 요약")
    display_metrics(test_series, anomaly_segments)
    st.markdown("---")

    # ── 그래프 1: 원본 시계열 + 이상 구간 ────────────────────────────────────
    st.subheader("그래프 1. 원본 시계열 + 이상 구간 하이라이트")
    st.plotly_chart(
        plot_series_with_anomalies(test_series, anomaly_segments),
        use_container_width=True, key=f"g1_{col}",
    )

    # ── 그래프 2: Scorer 점수 비교 ────────────────────────────────────────────
    st.subheader("그래프 2. Scorer별 이상 점수 비교")
    thresholds = res.get("thresholds")
    if thresholds is None and res["strategy"] == "with_label":
        opt = res.get("optimal_threshold")
        thresholds = [opt if i == best_idx else None for i in range(len(scores))]
    st.plotly_chart(
        plot_scorer_comparison(scores, scorer_names, thresholds, best_idx),
        use_container_width=True, key=f"g2_{col}",
    )

    # ── 그래프 3: 이상 점수 히스토그램 ───────────────────────────────────────
    st.subheader("그래프 3. 이상 점수 분포 히스토그램")
    if res["strategy"] == "without_label":
        hist_score = scores[0]
        hist_threshold = res["thresholds"][0]
        hist_scorer_name = "NormScorer"
    else:
        hist_score = scores[best_idx]
        hist_threshold = res["optimal_threshold"]
        hist_scorer_name = res["best_scorer_name"]

    st.plotly_chart(
        plot_score_histogram(hist_score, hist_threshold, res["final_anomalies"], hist_scorer_name),
        use_container_width=True, key=f"g3_{col}",
    )
    st.caption(f"임계치 **{hist_threshold:.4f}** 기준으로 왼쪽은 정상 구간, 오른쪽은 이상 구간")

    # ── 그래프 4: AUC-ROC (레이블 있을 때만) ─────────────────────────────────
    if res["strategy"] == "with_label":
        st.subheader("그래프 4. AUC-ROC 커브")
        st.info(
            f"📌 **{res['best_scorer_name']}** (AUC={res['best_auc']:.4f}) 자동 선택  |  "
            f"ROC 최적 임계치: {res['optimal_threshold']:.4f}"
        )
        st.plotly_chart(
            plot_roc_curves(res["roc_data"], scorer_names, res["auc_scores"], best_idx),
            use_container_width=True, key=f"g4_{col}",
        )

    # ── STL 분해 + 잔차 이상탐지 ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔬 STL 분해 + 잔차 이상탐지")
    st.info(
        "**원본 이상탐지**는 전역 이상(Global Anomaly)을 탐지합니다. "
        "추세와 계절성이 강한 데이터에서는 정상적인 계절 피크도 이상으로 오탐될 수 있습니다.  \n"
        "**잔차 이상탐지**는 추세와 계절성을 제거한 순수 잔차에서 이상을 탐지하여 "
        "맥락 이상(Contextual Anomaly)을 포착합니다. "
        "두 결과를 함께 확인하면 더 완전한 이상탐지가 가능합니다."
    )

    if res.get("stl") is not None:
        stl = res["stl"]
        with st.spinner("STL 분해 시각화 중..."):
            st.plotly_chart(
                plot_stl_decomposition(
                    res["full_series"], stl["trend"], stl["seasonal"], stl["residual"]
                ),
                use_container_width=True, key=f"stl_{col}",
            )

        res_result = res.get("residual_result")
        if res_result is not None:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**📊 원본 이상탐지 결과**")
                st.plotly_chart(
                    plot_series_with_anomalies(test_series, anomaly_segments),
                    use_container_width=True, key=f"stl_orig_{col}",
                )
            with c2:
                st.markdown("**📊 잔차 이상탐지 결과**")
                st.plotly_chart(
                    plot_series_with_anomalies(
                        res_result["test_series"], res_result["anomaly_segments"]
                    ),
                    use_container_width=True, key=f"stl_res_{col}",
                )
    else:
        st.warning(f"STL 분해 실패: {res.get('stl_error', '알 수 없는 오류')}")

    # ── 이상 구간 테이블 ──────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📋 이상 구간 테이블")
    st.caption("점이상: 주황색 배경 | 패턴이상: 빨간색 배경")
    df_anomalies = display_anomaly_table(anomaly_segments)

    if df_anomalies is not None:
        csv_bytes = df_anomalies.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            label=f"📥 {col} 이상 구간 CSV 다운로드",
            data=csv_bytes,
            file_name=f"anomaly_{col}.csv",
            mime="text/csv",
            key=f"dl_{col}",
        )

    # ── 이상 구간 상세 분석 ───────────────────────────────────────────────────
    if anomaly_segments:
        st.subheader("🔍 이상 구간 상세 분석")
        seg_labels = [
            f"구간 {i+1}: {s['start']} ~ {s['end']} ({s['type']})"
            for i, s in enumerate(anomaly_segments)
        ]
        selected_idx = st.selectbox(
            "분석할 이상 구간 선택",
            range(len(seg_labels)),
            format_func=lambda i: seg_labels[i],
            key=f"seg_{col}",
        )
        selected_seg = anomaly_segments[selected_idx]

        st.plotly_chart(
            plot_anomaly_detail(test_series, selected_seg),
            use_container_width=True, key=f"detail_{col}",
        )
        st.markdown("**구간 통계**")
        display_anomaly_detail_stats(test_series, anomaly_segments, selected_seg)


# ── 사이드바 ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 데이터 업로드")
    uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])

    df = None
    current_hash = None
    target_cols = []
    time_col = None
    label_col = None
    aggregator_type = "or"
    high_quantile = 0.95

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        current_hash = _md5(file_bytes)

        try:
            df = pd.read_csv(io.BytesIO(file_bytes))
        except Exception as e:
            st.error(f"CSV 파일 형식 오류: {e}")

        if df is not None and len(df) < 100:
            st.error("데이터가 너무 적습니다. 최소 100행 이상이 필요합니다.")
            df = None

        if df is not None:
            st.success(f"✅ 로드 완료: {len(df):,}행 × {len(df.columns)}열")
            cols = list(df.columns)

            st.markdown("**컬럼 설정**")
            time_col = st.selectbox("시간 컬럼", cols, index=0)
            remaining = [c for c in cols if c != time_col]

            target_cols = st.multiselect(
                "타겟 컬럼 (복수 선택 가능)",
                remaining,
                default=[remaining[0]] if remaining else [],
                help="선택된 컬럼마다 독립적으로 이상탐지 수행",
            )

            label_candidates = [c for c in remaining if c not in target_cols]
            label_col_options = ["없음 (레이블 미사용)"] + label_candidates
            label_col_choice = st.selectbox(
                "레이블 컬럼 (0/1, 선택사항)",
                label_col_options,
                index=0,
                help="이상=1, 정상=0 컬럼 선택 시 AUC 기반 탐지 활성화",
            )
            label_col = None if label_col_choice == "없음 (레이블 미사용)" else label_col_choice
            has_label = label_col is not None

            st.markdown("---")
            st.header("⚙️ 이상탐지 설정")

            if not has_label:
                agg_choice = st.selectbox(
                    "Aggregator 선택",
                    ["OrAggregator", "AndAggregator"],
                    help="Or: 하나라도 이상 → 탐지 / And: 모두 이상이어야 탐지",
                )
                aggregator_type = "or" if agg_choice == "OrAggregator" else "and"
                high_quantile = st.slider(
                    "임계치 (high_quantile)",
                    min_value=0.80, max_value=0.99, value=0.95, step=0.01,
                    help="높을수록 극단적인 값만 이상으로 탐지",
                )
                if st.session_state.results:
                    st.markdown("---")
                    _realtime_feedback(st.session_state.results, high_quantile)
            else:
                st.info("레이블 컬럼이 지정되었으므로 AUC 기반으로 Scorer를 자동 선택합니다.")

            st.markdown("---")
            run_button = st.button(
                "🚀 분석 시작", type="primary", use_container_width=True,
                disabled=len(target_cols) == 0,
            )

            file_changed = current_hash != st.session_state.file_hash
            should_run = run_button or (file_changed and st.session_state.results is not None)

            if should_run and target_cols:
                all_results = {}
                for col in target_cols:
                    with st.spinner(f"'{col}' 컬럼 분석 중..."):
                        col_result = run_column_analysis(
                            df, time_col, col, label_col, aggregator_type, high_quantile
                        )
                        if col_result is not None:
                            all_results[col] = col_result

                if all_results:
                    st.session_state.results = all_results
                    st.session_state.file_hash = current_hash
                    st.rerun()
    else:
        st.info("먼저 CSV 파일을 업로드하세요.")


# ── 메인 대시보드 ─────────────────────────────────────────────────────────────
if st.session_state.results is None:
    st.markdown("---")
    st.markdown("### 👈 사이드바에서 CSV 파일을 업로드하고 분석을 시작하세요.")
    with st.expander("📖 사용 방법"):
        st.markdown("""
**공통**: CSV 파일 하나만 업로드합니다.

**레이블 없는 경우 (비지도 탐지)**
- 타겟 컬럼 복수 선택 → 각 컬럼 독립 분석 후 탭으로 표시
- Aggregator + 임계치 설정 → 슬라이더 조정 시 예상 탐지 수 실시간 표시

**레이블 있는 경우 (지도 평가)**
- 0/1 레이블 컬럼 선택 → AUC-ROC 기반 최적 Scorer 자동 선택

**추가 기능**
- STL 분해 (추세·계절성·잔차) + 잔차 이상탐지 나란히 비교
- 이상 점수 분포 히스토그램 (정상 vs 이상 구간)
- 이상 구간 클릭 선택 → 확대 그래프 + 5개 통계 지표
""")
else:
    results = st.session_state.results
    col_names = list(results.keys())
    tabs = st.tabs([f"📊 {c}" for c in col_names])

    for tab, col in zip(tabs, col_names):
        with tab:
            _display_column_tab(col, results[col])
