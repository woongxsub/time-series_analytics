import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def forecast_arima(
    train_series: pd.Series,
    horizon: int,
    p: int = 1,
    d: int = 1,
    q: int = 1,
) -> np.ndarray:
    """
    train_series에 ARIMA(p,d,q)를 적합하고 horizon 스텝 예측값 반환.
    적합 실패 시 마지막 관측값으로 채운 배열 반환.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = ARIMA(train_series, order=(p, d, q))
            result = model.fit()
            forecast = result.forecast(steps=horizon)
            return np.asarray(forecast, dtype=float)
        except Exception:
            return np.full(horizon, float(train_series.iloc[-1]))
