import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.dataprocessing.transformers import MissingValuesFiller


def detect_frequency(df, time_col):
    time_diffs = df[time_col].diff().dropna()
    if len(time_diffs) == 0:
        return "hourly", 24
    median_diff = time_diffs.median()
    if median_diff >= pd.Timedelta(days=1):
        return "daily", 7
    elif median_diff >= pd.Timedelta(hours=1):
        return "hourly", 24
    else:
        return "minute", 60


def _format_duration(delta):
    total_seconds = int(delta.total_seconds())
    if total_seconds < 3600:
        return f"{total_seconds // 60}분"
    elif total_seconds < 86400:
        h, m = divmod(total_seconds, 3600)
        return f"{h}시간 {m // 60}분"
    else:
        d, rem = divmod(total_seconds, 86400)
        return f"{d}일 {rem // 3600}시간"


def preprocess(df, time_col, target_col):
    """
    전처리 파이프라인.
    Returns: (train, test, full_series, freq, lags)
    """
    if len(df) < 100:
        raise ValueError("데이터가 너무 적습니다. 최소 100행 이상이 필요합니다.")

    df = df.copy()

    try:
        df[time_col] = pd.to_datetime(df[time_col])
    except Exception as e:
        raise ValueError(f"시간 컬럼 파싱 실패: {e}")

    df = df.sort_values(time_col).reset_index(drop=True)

    try:
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    except Exception as e:
        raise ValueError(f"타겟 컬럼 숫자 변환 실패: {e}")

    if df[target_col].isna().all():
        raise ValueError("타겟 컬럼에 유효한 숫자 값이 없습니다.")

    freq, lags = detect_frequency(df, time_col)

    sub = df[[time_col, target_col]].copy()
    attempts = [
        dict(fill_missing_dates=True, freq=None),
        dict(fill_missing_dates=True, freq="B"),   # 영업일 (주식 데이터)
        dict(fill_missing_dates=True, freq="D"),   # 달력일
        dict(),                                     # freq 자동
    ]
    series = None
    last_err = None
    for kwargs in attempts:
        try:
            series = TimeSeries.from_dataframe(
                sub, time_col=time_col, value_cols=target_col, **kwargs
            )
            break
        except Exception as e:
            last_err = e
    if series is None:
        raise ValueError(f"TimeSeries 변환 실패: {last_err}")

    filler = MissingValuesFiller()
    series = filler.transform(series)

    split_idx = int(len(series) * 0.8)
    train = series[:split_idx]
    test = series[split_idx:]

    return train, test, series, freq, lags


def stl_decompose(full_series, freq):
    """
    STL 분해: 추세 + 계절성 + 잔차 반환 (각각 Darts TimeSeries).
    """
    from statsmodels.tsa.seasonal import STL

    period_map = {"daily": 7, "hourly": 24, "minute": 60}
    period = period_map.get(freq, 7)

    pd_s = full_series.to_series()

    if len(pd_s) < 2 * period:
        raise ValueError(f"STL 분해에 필요한 최소 데이터({2 * period}행)가 부족합니다.")

    stl = STL(pd_s, period=period, robust=True)
    fit = stl.fit()

    def _ts(arr):
        return TimeSeries.from_series(pd.Series(arr, index=pd_s.index, name=pd_s.name))

    return _ts(fit.trend), _ts(fit.seasonal), _ts(fit.resid)


def extract_label_series(df, time_col, label_col):
    """
    업로드된 단일 DataFrame에서 레이블 컬럼을 Darts TimeSeries로 추출.
    """
    sub = df[[time_col, label_col]].copy()

    try:
        sub[time_col] = pd.to_datetime(sub[time_col])
    except Exception as e:
        raise ValueError(f"시간 컬럼 파싱 실패: {e}")

    sub = sub.sort_values(time_col).reset_index(drop=True)
    sub[label_col] = pd.to_numeric(sub[label_col], errors="coerce").fillna(0)

    attempts = [
        dict(fill_missing_dates=True, freq=None),
        dict(fill_missing_dates=True, freq="B"),
        dict(fill_missing_dates=True, freq="D"),
        dict(),
    ]
    last_err = None
    for kwargs in attempts:
        try:
            return TimeSeries.from_dataframe(sub, time_col=time_col, value_cols=label_col, **kwargs)
        except Exception as e:
            last_err = e
    raise ValueError(f"레이블 TimeSeries 변환 실패: {last_err}")
