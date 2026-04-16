import warnings
import numpy as np
import pandas as pd

os_env_set = False


def _suppress_tf_logs():
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def _build_sequences(values: np.ndarray, window_size: int):
    """슬라이딩 윈도우로 (X, y) 시퀀스 생성."""
    X, y = [], []
    for i in range(len(values) - window_size):
        X.append(values[i : i + window_size])
        y.append(values[i + window_size])
    return np.array(X, dtype=float), np.array(y, dtype=float)


def forecast_lstm(
    train_series: pd.Series,
    horizon: int,
    window_size: int = 10,
    hidden_units: int = 64,
    epochs: int = 20,
    batch_size: int = 32,
) -> np.ndarray:
    """
    train_series에 단층 LSTM을 적합하고 horizon 스텝을 재귀적으로 예측.
    TensorFlow 미설치 또는 학습 데이터 부족 시 마지막 관측값 반환.
    """
    _suppress_tf_logs()

    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense

        tf.get_logger().setLevel("ERROR")
    except ImportError:
        return np.full(horizon, float(train_series.iloc[-1]))

    values = train_series.values.astype(float)

    if len(values) <= window_size:
        return np.full(horizon, float(values[-1]))

    # Min-Max 정규화
    v_min, v_max = values.min(), values.max()
    scale = v_max - v_min
    if scale == 0:
        return np.full(horizon, float(v_min))
    normalized = (values - v_min) / scale

    X, y = _build_sequences(normalized, window_size)
    if len(X) == 0:
        return np.full(horizon, float(train_series.iloc[-1]))

    X = X.reshape(-1, window_size, 1)

    model = Sequential(
        [
            LSTM(hidden_units, input_shape=(window_size, 1)),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(
            X,
            y,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
        )

    # 재귀적 예측
    window = normalized[-window_size:].tolist()
    preds_norm = []
    for _ in range(horizon):
        x_in = np.array(window[-window_size:], dtype=float).reshape(1, window_size, 1)
        pred = float(model.predict(x_in, verbose=0)[0][0])
        preds_norm.append(pred)
        window.append(pred)

    # 역정규화
    predictions = np.array(preds_norm) * scale + v_min
    return predictions.astype(float)
