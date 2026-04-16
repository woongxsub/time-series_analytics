import pandas as pd
import numpy as np
import streamlit as st


def load_csv(uploaded_file) -> pd.DataFrame | None:
    """Streamlit UploadedFile로부터 CSV를 읽어 DataFrame 반환."""
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"CSV 로딩 오류: {e}")
        return None


def detect_date_columns(df: pd.DataFrame) -> list[str]:
    """날짜/시간형으로 파싱 가능한 컬럼 목록 반환."""
    candidates = []
    for col in df.columns:
        if "datetime" in str(df[col].dtype):
            candidates.append(col)
        elif df[col].dtype == object:
            try:
                parsed = pd.to_datetime(df[col].dropna().head(20), infer_datetime_format=True)
                if len(parsed) > 0:
                    candidates.append(col)
            except Exception:
                pass
    return candidates


def detect_value_columns(df: pd.DataFrame, date_col: str) -> list[str]:
    """숫자형 컬럼 중 날짜 컬럼을 제외한 목록 반환."""
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric if c != date_col]


def preprocess(df: pd.DataFrame, date_col: str, value_col: str) -> pd.Series:
    """
    날짜 파싱, 정렬, 결측치 처리 후 pd.Series(datetime index) 반환.
    결측치가 있으면 선형 보간 후 Streamlit warning 표시.
    """
    data = df[[date_col, value_col]].copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values(date_col).reset_index(drop=True)
    data = data.set_index(date_col)
    series = data[value_col].astype(float)

    missing = series.isna().sum()
    if missing > 0:
        series = series.interpolate(method="linear").ffill().bfill()
        st.warning(f"결측치 {missing}개를 선형 보간(linear interpolation)으로 처리했습니다.")

    return series


def train_test_split_series(series: pd.Series, train_ratio: float) -> tuple[pd.Series, pd.Series]:
    """시계열을 train/test 로 분할."""
    split = int(len(series) * train_ratio)
    return series.iloc[:split], series.iloc[split:]
