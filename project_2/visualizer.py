import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SCORER_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]


def display_metrics(test_series, anomaly_segments):
    """상단 요약 메트릭 카드 4개."""
    total_pts = len(test_series)
    n_segs = len(anomaly_segments)
    n_pts = sum(s["length"] for s in anomaly_segments)
    ratio = n_pts / total_pts * 100 if total_pts > 0 else 0.0
    n_point = sum(1 for s in anomaly_segments if s["type"] == "점이상")
    n_pattern = sum(1 for s in anomaly_segments if s["type"] == "패턴이상")

    start = test_series.time_index[0]
    end = test_series.time_index[-1]

    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        st.metric("분석 기간", f"{start.date()}  ~  {end.date()}")
    with c2:
        st.metric("이상 구간 수", f"{n_segs}개")
    with c3:
        st.metric("이상 비율", f"{ratio:.2f}%")
    with c4:
        st.metric("점이상 / 패턴이상", f"{n_point} / {n_pattern}")


def plot_series_with_anomalies(test_series, anomaly_segments):
    """원본 시계열 + 이상 구간 하이라이트."""
    time_idx = test_series.time_index
    vals = test_series.univariate_values()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=time_idx,
            y=vals,
            mode="lines",
            name="원본 시계열",
            line=dict(color="#6464ff", width=1),
        )
    )

    added_point_legend = False
    added_pattern_legend = False

    for seg in anomaly_segments:
        if seg["type"] == "점이상":
            mask = (time_idx >= seg["start"]) & (time_idx <= seg["end"])
            fig.add_trace(
                go.Scatter(
                    x=time_idx[mask],
                    y=vals[mask],
                    mode="markers",
                    name="점이상" if not added_point_legend else None,
                    showlegend=not added_point_legend,
                    marker=dict(color="red", size=8, symbol="circle"),
                    legendgroup="point",
                )
            )
            added_point_legend = True
        else:
            fig.add_vrect(
                x0=seg["start"],
                x1=seg["end"],
                fillcolor="red",
                opacity=0.18,
                layer="below",
                line_width=0,
                annotation_text="패턴이상" if not added_pattern_legend else "",
                annotation_position="top left",
            )
            if not added_pattern_legend:
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="lines",
                        name="패턴이상 구간",
                        line=dict(color="rgba(255,0,0,0.4)", width=10),
                        legendgroup="pattern",
                    )
                )
                added_pattern_legend = True

    fig.update_layout(
        title="원본 시계열 + 이상 구간 하이라이트",
        xaxis_title="시간",
        yaxis_title="값",
        height=420,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plot_scorer_comparison(scores, scorer_names, thresholds=None, best_scorer_idx=None):
    """Scorer별 이상 점수 비교 (3개 라인 + 임계치)."""
    fig = go.Figure()

    for i, (score, name) in enumerate(zip(scores, scorer_names)):
        is_best = best_scorer_idx is not None and i == best_scorer_idx
        display_name = f"★ {name} (선택됨)" if is_best else name

        fig.add_trace(
            go.Scatter(
                x=score.time_index,
                y=score.univariate_values(),
                mode="lines",
                name=display_name,
                line=dict(
                    color=SCORER_COLORS[i],
                    width=2.5 if is_best else 1.2,
                ),
                opacity=1.0 if (best_scorer_idx is None or is_best) else 0.45,
            )
        )

        if thresholds and i < len(thresholds) and thresholds[i] is not None:
            fig.add_hline(
                y=thresholds[i],
                line_dash="dash",
                line_color=SCORER_COLORS[i],
                opacity=0.6,
                annotation_text=f"{name} 임계치 ({thresholds[i]:.3f})",
                annotation_position="right",
            )

    fig.update_layout(
        title="Scorer별 이상 점수 비교",
        xaxis_title="시간",
        yaxis_title="이상 점수",
        height=420,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plot_roc_curves(roc_data, scorer_names, auc_scores, best_scorer_idx):
    """Scorer별 ROC 커브."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="랜덤 기준선 (AUC=0.50)",
            line=dict(color="gray", dash="dash", width=1),
        )
    )

    for i, (roc, name, auc) in enumerate(zip(roc_data, scorer_names, auc_scores)):
        is_best = i == best_scorer_idx
        display_name = (
            f"★ {name} (AUC={auc:.4f}, 선택됨)" if is_best else f"{name} (AUC={auc:.4f})"
        )
        fig.add_trace(
            go.Scatter(
                x=roc["fpr"],
                y=roc["tpr"],
                mode="lines",
                name=display_name,
                line=dict(color=SCORER_COLORS[i], width=3 if is_best else 1.5),
                opacity=1.0 if is_best else 0.6,
            )
        )

    fig.update_layout(
        title="Scorer별 AUC-ROC 커브",
        xaxis_title="FPR (False Positive Rate)",
        yaxis_title="TPR (True Positive Rate)",
        height=460,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    return fig


def plot_stl_decomposition(original, trend, seasonal, residual):
    """STL 분해 결과: 원본·추세·계절성·잔차 4행 서브플롯."""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        subplot_titles=["원본 (Original)", "추세 (Trend)", "계절성 (Seasonal)", "잔차 (Residual)"],
        vertical_spacing=0.07,
    )
    components = [
        (original, "#6464ff"),
        (trend, "#ff7f0e"),
        (seasonal, "#2ca02c"),
        (residual, "#d62728"),
    ]
    for row, (ts, color) in enumerate(components, 1):
        fig.add_trace(
            go.Scatter(
                x=ts.time_index,
                y=ts.univariate_values(),
                mode="lines",
                line=dict(color=color, width=1),
                showlegend=False,
            ),
            row=row, col=1,
        )
    fig.update_layout(height=680, title="STL 분해 결과")
    return fig


def plot_score_histogram(score, threshold, final_anomalies, scorer_name="NormScorer"):
    """이상 점수 분포 히스토그램 — 정상(파란색) vs 이상(빨간색) + 임계치 수직선."""
    score_pd = score.to_series()
    final_pd = final_anomalies.to_series()
    common_idx = score_pd.index.intersection(final_pd.index)

    if len(common_idx) > 0:
        score_vals = score_pd.loc[common_idx].values
        binary_vals = final_pd.loc[common_idx].values
    else:
        min_len = min(len(score_pd), len(final_pd))
        score_vals = score_pd.values[:min_len]
        binary_vals = final_pd.values[:min_len]

    normal_scores = score_vals[binary_vals == 0]
    anomaly_scores = score_vals[binary_vals == 1]

    fig = go.Figure()
    if len(normal_scores) > 0:
        fig.add_trace(go.Histogram(
            x=normal_scores, name="정상 구간",
            opacity=0.65, marker_color="#1f77b4", nbinsx=60,
        ))
    if len(anomaly_scores) > 0:
        fig.add_trace(go.Histogram(
            x=anomaly_scores, name="이상 구간",
            opacity=0.65, marker_color="red", nbinsx=60,
        ))
    fig.add_vline(
        x=threshold, line_dash="dash", line_color="black", line_width=2,
        annotation_text=f"임계치 ({threshold:.4f})",
        annotation_position="top right",
    )
    fig.update_layout(
        barmode="overlay",
        title=f"{scorer_name} 이상 점수 분포",
        xaxis_title="이상 점수",
        yaxis_title="빈도",
        height=360,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plot_anomaly_detail(test_series, segment, context_ratio=0.1):
    """선택된 이상 구간 확대 그래프 (전후 10% 맥락 포함)."""
    time_idx = test_series.time_index
    vals = test_series.univariate_values()
    n = len(time_idx)
    context_steps = max(3, int(n * context_ratio))

    start_pos = max(0, int(np.searchsorted(time_idx, segment["start"])) - context_steps)
    end_pos = min(n, int(np.searchsorted(time_idx, segment["end"], side="right")) + context_steps)

    sliced_time = time_idx[start_pos:end_pos]
    sliced_vals = vals[start_pos:end_pos]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sliced_time, y=sliced_vals,
        mode="lines", name="시계열",
        line=dict(color="#6464ff", width=1.5),
    ))
    fig.add_vrect(
        x0=segment["start"], x1=segment["end"],
        fillcolor="red", opacity=0.18, layer="below", line_width=0,
        annotation_text=segment["type"], annotation_position="top left",
    )
    fig.update_layout(
        title=f"이상 구간 확대: {segment['start']} ~ {segment['end']}",
        xaxis_title="시간", yaxis_title="값",
        height=320, hovermode="x unified",
    )
    return fig


def display_anomaly_detail_stats(test_series, anomaly_segments, selected_segment):
    """선택 구간의 통계 + 정상 평균 대비 편차."""
    time_idx = test_series.time_index
    vals = test_series.univariate_values()

    seg_mask = (time_idx >= selected_segment["start"]) & (time_idx <= selected_segment["end"])
    seg_vals = vals[seg_mask]

    anomaly_mask = np.zeros(len(vals), dtype=bool)
    for seg in anomaly_segments:
        m = (time_idx >= seg["start"]) & (time_idx <= seg["end"])
        anomaly_mask |= m
    normal_vals = vals[~anomaly_mask]

    if len(seg_vals) == 0:
        st.warning("해당 구간의 데이터를 찾을 수 없습니다.")
        return

    seg_mean = float(np.mean(seg_vals))
    seg_max = float(np.max(seg_vals))
    seg_min = float(np.min(seg_vals))
    seg_std = float(np.std(seg_vals))
    normal_mean = float(np.mean(normal_vals)) if len(normal_vals) > 0 else None
    deviation = ((seg_mean - normal_mean) / abs(normal_mean) * 100) if normal_mean else None

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("평균", f"{seg_mean:.2f}")
    c2.metric("최대", f"{seg_max:.2f}")
    c3.metric("최소", f"{seg_min:.2f}")
    c4.metric("표준편차", f"{seg_std:.2f}")
    if deviation is not None:
        c5.metric("정상 평균 대비", f"{deviation:+.1f}%")


def display_anomaly_table(anomaly_segments):
    """이상 구간 테이블 표시 및 DataFrame 반환."""
    if not anomaly_segments:
        st.info("탐지된 이상 구간이 없습니다.")
        return None

    rows = [
        {
            "시작 시점": seg["start"],
            "종료 시점": seg["end"],
            "지속 시간": seg["duration"],
            "이상 유형": seg["type"],
            "최대 이상 점수": round(seg["max_score"], 4),
        }
        for seg in anomaly_segments
    ]
    df = pd.DataFrame(rows)

    def _color_row(row):
        color = "#fff3cd" if row["이상 유형"] == "점이상" else "#f8d7da"
        return [f"background-color: {color}"] * len(row)

    st.dataframe(df.style.apply(_color_row, axis=1), use_container_width=True)
    return df
