import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def forecast_sarima(
    train_series: pd.Series,
    horizon: int,
    p: int = 1,
    d: int = 1,
    q: int = 1,
    P: int = 1,
    D: int = 1,
    Q: int = 1,
    s: int = 12,
) -> np.ndarray:
    """
    train_series에 SARIMA(p,d,q)(P,D,Q,s)를 적합하고 horizon 스텝 예측값 반환.
    적합 실패 시 마지막 관측값으로 채운 배열 반환.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = SARIMAX(
                train_series,
                order=(p, d, q),
                seasonal_order=(P, D, Q, s),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            result = model.fit(disp=False, maxiter=200)
            forecast = result.forecast(steps=horizon)
            return np.asarray(forecast, dtype=float)
        except Exception:
            return np.full(horizon, float(train_series.iloc[-1]))
