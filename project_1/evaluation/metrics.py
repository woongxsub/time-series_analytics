import numpy as np


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """평균 절대 오차 (Mean Absolute Error)."""
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """평균 제곱근 오차 (Root Mean Squared Error)."""
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    대칭 평균 절대 퍼센트 오차 (Symmetric MAPE, %).
    분모가 0인 경우를 안전하게 처리.
    """
    denom = (np.abs(actual) + np.abs(predicted)) / 2.0
    mask = denom != 0
    ratio = np.zeros_like(actual, dtype=float)
    ratio[mask] = np.abs(actual[mask] - predicted[mask]) / denom[mask]
    return float(np.mean(ratio) * 100)


def mase(actual: np.ndarray, predicted: np.ndarray, train_series: np.ndarray) -> float:
    """
    Naive forecast(lag-1) 대비 상대 성능 (Mean Absolute Scaled Error).
    train_series: 학습 데이터 값 배열 (MASE 스케일 계산용)
    """
    naive_errors = np.abs(np.diff(train_series))
    scale = np.mean(naive_errors)
    if scale == 0 or np.isnan(scale):
        return float("nan")
    return float(mae(actual, predicted) / scale)


def compute_all_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    train_series: np.ndarray,
) -> dict[str, float]:
    """네 가지 지표를 한 번에 계산해 dict로 반환."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    train_series = np.asarray(train_series, dtype=float)
    return {
        "MAE": mae(actual, predicted),
        "RMSE": rmse(actual, predicted),
        "SMAPE": smape(actual, predicted),
        "MASE": mase(actual, predicted, train_series),
    }
