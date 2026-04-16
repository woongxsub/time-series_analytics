import warnings
import logging
import numpy as np
import pandas as pd

# Prophet 로그 억제
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)


def forecast_prophet(
    train_series: pd.Series,
    horizon: int,
    changepoint_prior_scale: float = 0.05,
    seasonality_mode: str = "additive",
) -> np.ndarray:
    """
    train_series(datetime index)에 Prophet을 적합하고 horizon 스텝 예측값 반환.
    내부적으로 컬럼명을 ds/y로 자동 변환.
    적합 실패 시 마지막 관측값으로 채운 배열 반환.
    """
    try:
        from prophet import Prophet
    except ImportError:
        try:
            from fbprophet import Prophet  # type: ignore
        except ImportError:
            return np.full(horizon, float(train_series.iloc[-1]))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            df = pd.DataFrame({"ds": train_series.index, "y": train_series.values})

            model = Prophet(
                changepoint_prior_scale=changepoint_prior_scale,
                seasonality_mode=seasonality_mode,
            )
            model.fit(df)

            # 날짜 인덱스 주파수 추론
            freq = pd.infer_freq(train_series.index)
            if freq is None:
                # 평균 간격을 days 단위로 추정
                deltas = pd.Series(train_series.index).diff().dropna()
                avg_days = deltas.median().days if hasattr(deltas.median(), "days") else 1
                freq = f"{avg_days}D"

            future = model.make_future_dataframe(periods=horizon, freq=freq)
            forecast = model.predict(future)
            return forecast["yhat"].iloc[-horizon:].to_numpy(dtype=float)
        except Exception:
            return np.full(horizon, float(train_series.iloc[-1]))
