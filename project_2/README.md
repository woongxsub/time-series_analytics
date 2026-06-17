# 시계열 이상탐지 Streamlit 웹앱

Darts 라이브러리 기반 시계열 이상탐지 대시보드.  
NormScorer / KMeansScorer / WassersteinScorer 3종 조합, 레이블 유무에 따른 자동 전략 분기.

---

## 설치

```bash
pip install -r requirements.txt
```

> PyTorch는 OS/CUDA 환경에 따라 별도 설치가 필요할 수 있습니다.  
> CPU 전용: `pip install torch --index-url https://download.pytorch.org/whl/cpu`

---

## 실행

```bash
cd anomaly_detection_app
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 접속.

---

## 테스트 데이터 준비

### 레이블 없는 케이스 (TaxiNewYork)

```python
from darts.datasets import TaxiNewYorkDataset
import pandas as pd

series = TaxiNewYorkDataset().load()
df = series.to_dataframe().reset_index()
df.columns = ["timestamp", "value"]
df.to_csv("taxi_ny.csv", index=False)
```

→ `taxi_ny.csv`를 업로드, 시간 컬럼: `timestamp`, 타겟 컬럼: `value`

### 레이블 있는 케이스

```python
import pandas as pd

anomalies = {
    "NYC Marathon":     ("2014-11-02 00:00", "2014-11-02 23:30"),
    "Thanksgiving":     ("2014-11-27 00:00", "2014-11-28 23:30"),
    "Christmas":        ("2014-12-25 00:00", "2014-12-26 23:30"),
    "New Year":         ("2014-12-31 00:00", "2015-01-01 23:30"),
    "Blizzard":         ("2015-01-26 00:00", "2015-01-27 23:30"),
}

df = pd.read_csv("taxi_ny.csv")
df["is_anomaly"] = 0
df["timestamp"] = pd.to_datetime(df["timestamp"])

for start, end in anomalies.values():
    mask = (df["timestamp"] >= start) & (df["timestamp"] < end)
    df.loc[mask, "is_anomaly"] = 1

label_df = df[["timestamp", "is_anomaly"]]
label_df.to_csv("taxi_ny_labels.csv", index=False)
```

→ `taxi_ny_labels.csv`를 레이블 CSV로 추가 업로드

---

## 기능 요약

| 기능 | 설명 |
|---|---|
| 레이블 없음 | ThresholdDetector + Or/AndAggregator |
| 레이블 있음 | AUC-ROC 기반 최적 Scorer 자동 선택 |
| 이상 유형 분류 | 1-2포인트: 점이상, 3+포인트: 패턴이상 |
| 시각화 | Plotly 인터랙티브 (확대/축소, 호버) |
| 자동 재분석 | 파일 교체 시 session_state 해시 감지 후 자동 재실행 |
| CSV 다운로드 | 탐지된 이상 구간 테이블 저장 |
