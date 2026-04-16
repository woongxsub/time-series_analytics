"""
Rolling / Expanding Window 평가 전략 모듈.

evaluate_model
    - Rolling Horizon 방식으로 테스트 구간을 horizon 단위로 순회하며 예측 수집
    - window_type='expanding' → 처음부터 현재까지 누적 학습
    - window_type='rolling'   → 고정 크기(rolling_window_size)의 최근 구간만 학습

run_horizon_sensitivity
    - 여러 horizon 값에 대해 evaluate_model 을 반복 실행해 성능 변화 추적
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable

from evaluation.metrics import compute_all_metrics


def evaluate_model(
    series: pd.Series,
    train_ratio: float,
    horizon: int,
    model_fn: Callable,
    model_params: dict,
    window_type: str = "expanding",
    rolling_window_size: int | None = None,
) -> dict | None:
    """
    Rolling Horizon 평가.

    Parameters
    ----------
    series : pd.Series — datetime index를 가진 시계열
    train_ratio : float — 초기 학습 비율
    horizon : int — 매 스텝 예측할 미래 시점 수
    model_fn : callable(train_series, horizon, **params) → np.ndarray[horizon]
    model_params : dict — model_fn 에 전달할 파라미터
    window_type : 'expanding' | 'rolling'
    rolling_window_size : Rolling 방식일 때 학습 윈도우 크기 (None 이면 train_size 사용)

    Returns
    -------
    dict with keys: predictions, actuals, pred_index, metrics
    None if no evaluation steps could be performed
    """
    n = len(series)
    initial_train_size = int(n * train_ratio)

    if initial_train_size < 2:
        return None

    all_predictions: list[float] = []
    all_actuals: list[float] = []
    all_indices: list = []

    cursor = initial_train_size

    while cursor + horizon <= n:
        # ── 학습 데이터 슬라이싱 ──────────────────────────────────────────
        if window_type == "expanding":
            train = series.iloc[:cursor]
        else:
            ws = rolling_window_size if rolling_window_size else initial_train_size
            start = max(0, cursor - ws)
            train = series.iloc[start:cursor]

        actuals = series.iloc[cursor : cursor + horizon].values

        # ── 예측 ──────────────────────────────────────────────────────────
        try:
            preds = model_fn(train, horizon, **model_params)
            preds = np.asarray(preds, dtype=float)
            if len(preds) != horizon:
                preds = np.full(horizon, float(train.iloc[-1]))
        except Exception:
            preds = np.full(horizon, float(train.iloc[-1]))

        all_predictions.extend(preds.tolist())
        all_actuals.extend(actuals.tolist())
        all_indices.extend(series.index[cursor : cursor + horizon].tolist())

        cursor += horizon

    if not all_actuals:
        return None

    predictions = np.array(all_predictions, dtype=float)
    actuals = np.array(all_actuals, dtype=float)
    train_values = series.iloc[:initial_train_size].values

    return {
        "predictions": predictions,
        "actuals": actuals,
        "pred_index": all_indices,
        "metrics": compute_all_metrics(actuals, predictions, train_values),
    }


def run_horizon_sensitivity(
    series: pd.Series,
    train_ratio: float,
    horizons: list[int],
    model_fn: Callable,
    model_params: dict,
    window_type: str = "expanding",
    rolling_window_size: int | None = None,
) -> dict[int, dict[str, float]]:
    """
    여러 horizon 값에 대해 evaluate_model 을 실행해
    {horizon: metrics_dict} 형태로 반환.
    """
    results: dict[int, dict[str, float]] = {}
    for h in horizons:
        res = evaluate_model(
            series,
            train_ratio,
            h,
            model_fn,
            model_params,
            window_type,
            rolling_window_size,
        )
        if res is not None:
            results[h] = res["metrics"]
    return results
