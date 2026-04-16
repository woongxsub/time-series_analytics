"""
Plotly 기반 시각화 모듈.

plot_series              — 전체 시계열 라인 차트
plot_forecast            — 실제값 vs 예측값 비교
plot_metrics_comparison  — 모델별 성능 지표 바 차트 (2×2 서브플롯)
plot_horizon_sensitivity — 시평에 따른 성능 변화 라인 차트
plot_window_comparison   — Rolling vs Expanding 윈도우 비교 바 차트
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

_PALETTE = px.colors.qualitative.Set1
METRIC_NAMES = ["MAE", "RMSE", "SMAPE", "MASE"]


# ──────────────────────────────────────────────────────────────────────────────
def plot_series(series: pd.Series, title: str = "시계열 데이터") -> go.Figure:
    """전체 시계열을 단순 라인 차트로 시각화."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines",
            name="실제값",
            line=dict(color="royalblue", width=1.5),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="날짜",
        yaxis_title="값",
        hovermode="x unified",
        height=350,
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
def plot_forecast(
    series: pd.Series,
    train_size: int,
    pred_results: dict[str, dict],
) -> go.Figure:
    """
    실제값 전체와 각 모델의 예측값을 겹쳐서 시각화.

    pred_results : {model_name: {'predictions': np.ndarray, 'pred_index': list}}
    """
    fig = go.Figure()

    # 전체 실제값 (회색 배경)
    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines",
            name="실제값",
            line=dict(color="rgba(100,100,100,0.4)", width=1),
        )
    )

    # 테스트 구간 실제값 (진하게)
    if train_size < len(series):
        test_series = series.iloc[train_size:]
        fig.add_trace(
            go.Scatter(
                x=test_series.index,
                y=test_series.values,
                mode="lines",
                name="실제값 (테스트)",
                line=dict(color="black", width=2),
            )
        )

    # 모델별 예측값
    for i, (model_name, res) in enumerate(pred_results.items()):
        color = _PALETTE[i % len(_PALETTE)]
        fig.add_trace(
            go.Scatter(
                x=res["pred_index"],
                y=res["predictions"],
                mode="lines",
                name=f"{model_name} 예측",
                line=dict(color=color, width=2, dash="dash"),
            )
        )

    # Train / Test 경계선
    if 0 < train_size < len(series):
        train_end_date = str(series.index[train_size - 1])
        fig.add_shape(
            type="line",
            x0=train_end_date, x1=train_end_date,
            y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(dash="dot", color="gray"),
        )
        fig.add_annotation(
            x=train_end_date,
            y=1,
            xref="x", yref="paper",
            text="Train | Test",
            showarrow=False,
            xanchor="right",
            yanchor="top",
        )

    fig.update_layout(
        title="실제값 vs 예측값 비교",
        xaxis_title="날짜",
        yaxis_title="값",
        hovermode="x unified",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
def plot_metrics_comparison(metrics_dict: dict[str, dict[str, float]]) -> go.Figure:
    """
    2×2 서브플롯으로 모델별 MAE / RMSE / SMAPE / MASE 비교.

    metrics_dict : {model_name: {metric: value}}
    """
    model_names = list(metrics_dict.keys())
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(model_names))]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=METRIC_NAMES,
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for idx, metric in enumerate(METRIC_NAMES):
        row, col = positions[idx]
        values = [metrics_dict[m].get(metric, np.nan) for m in model_names]
        fig.add_trace(
            go.Bar(
                x=model_names,
                y=values,
                marker_color=colors,
                name=metric,
                showlegend=False,
                text=[f"{v:.4f}" if not np.isnan(v) else "N/A" for v in values],
                textposition="outside",
            ),
            row=row,
            col=col,
        )

    fig.update_layout(title="모델별 성능 지표 비교", height=520)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
def plot_horizon_sensitivity(
    horizon_metrics: dict[str, dict[int, dict[str, float]]],
) -> go.Figure:
    """
    horizon 변화에 따른 성능 변화 라인 차트 (2×2 서브플롯).

    horizon_metrics : {model_name: {horizon: {metric: value}}}
    """
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=METRIC_NAMES,
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for idx, metric in enumerate(METRIC_NAMES):
        row, col = positions[idx]
        for i, (model_name, hm) in enumerate(horizon_metrics.items()):
            horizons_sorted = sorted(hm.keys())
            values = [hm[h].get(metric, np.nan) for h in horizons_sorted]
            color = _PALETTE[i % len(_PALETTE)]
            fig.add_trace(
                go.Scatter(
                    x=horizons_sorted,
                    y=values,
                    mode="lines+markers",
                    name=model_name,
                    line=dict(color=color),
                    showlegend=(idx == 0),
                ),
                row=row,
                col=col,
            )

    fig.update_layout(
        title="시평(Horizon) 변화에 따른 성능 변화",
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
def plot_window_comparison(
    rolling_metrics: dict[str, dict[str, float]],
    expanding_metrics: dict[str, dict[str, float]],
    metric: str = "MAE",
) -> go.Figure:
    """Rolling vs Expanding 윈도우 비교 그룹 바 차트."""
    model_names = sorted(set(rolling_metrics) | set(expanding_metrics))
    rolling_vals = [rolling_metrics.get(m, {}).get(metric, np.nan) for m in model_names]
    expanding_vals = [expanding_metrics.get(m, {}).get(metric, np.nan) for m in model_names]

    fig = go.Figure(
        data=[
            go.Bar(name="Rolling Window", x=model_names, y=rolling_vals, marker_color="#2196F3"),
            go.Bar(name="Expanding Window", x=model_names, y=expanding_vals, marker_color="#FF9800"),
        ]
    )
    fig.update_layout(
        barmode="group",
        title=f"Rolling vs Expanding Window 비교 ({metric})",
        xaxis_title="모델",
        yaxis_title=metric,
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
