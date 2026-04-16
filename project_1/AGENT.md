# AGENT.md — 시계열 예측 웹앱 프로젝트

## 프로젝트 목표

단변량 시계열 CSV 파일을 업로드하면 자동으로 데이터를 분석하고,
사용자가 설정한 예측 기간(시평, Forecast Horizon)에 따라 시계열 예측을 수행하는 Streamlit 기반 웹 애플리케이션을 개발한다.

---

## 기술 스택

- **Frontend/UI**: Streamlit
- **언어**: Python 3.10+
- **핵심 라이브러리**:
  - 데이터 처리: `pandas`, `numpy`
  - 시각화: `plotly`, `matplotlib`
  - ARIMA/SARIMA: `statsmodels`
  - Prophet: `prophet`
  - LSTM: `tensorflow` / `keras`
  - 평가 지표: 직접 구현 (sklearn 보조)
- **배포**: Streamlit Community Cloud

---

## 프로젝트 구조

---

## 주요 기능 명세

### 1. 데이터 입력
- CSV 파일 업로드 (날짜 컬럼 + 값 컬럼 포함 단변량 형식)
- 날짜 컬럼 및 값 컬럼 자동 감지 + 수동 선택 옵션
- 데이터 미리보기, 기초 통계, 시계열 시각화 제공

### 2. 예측 설정
| 설정 항목 | 설명 |
|-----------|------|
| 예측 모형 | ARIMA, SARIMA, Prophet, LSTM (중복 선택 가능) |
| 시평 (Forecast Horizon) | 예측할 미래 시점 수 |
| 학습/테스트 분할 비율 | 슬라이더로 조정 |
| 윈도우 방식 | Rolling Window / Expanding Window 선택 |

### 3. 모델별 파라미터 설정
| 모델 | 사용자 설정 파라미터 |
|------|----------------------|
| ARIMA | p, d, q |
| SARIMA | p, d, q, P, D, Q, s |
| Prophet | changepoint_prior_scale, seasonality_mode (additive/multiplicative) |
| LSTM | window_size, hidden_units, epochs, batch_size |

### 4. 학습 및 평가 방식
- **Rolling Window**: 고정된 크기의 최근 구간만 사용하여 학습
- **Expanding Window**: 처음부터 현재까지 누적하여 학습
- **Rolling Horizon 평가**: 테스트 구간 > 시평일 경우, 반복 예측 후 평균 성능 계산

### 5. 성능 지표

| 지표 | 설명 |
|------|------|
| MAE | 평균 절대 오차 |
| RMSE | 평균 제곱근 오차 (큰 오차에 민감) |
| SMAPE | 대칭 평균 절대 퍼센트 오차 (스케일 불변) |
| MASE | Naive forecast 대비 상대 성능 |

### 6. 대시보드 출력
- 실제값 vs 예측값 비교 그래프 (모델별)
- 모델별 성능 지표 비교 테이블
- 시평 변화에 따른 성능 변화 그래프
- Rolling vs Expanding window 결과 비교

---

## 구현 순서 (권장)

1. `utils/data_loader.py` — CSV 업로드 및 전처리
2. `evaluation/metrics.py` — MAE, RMSE, SMAPE, MASE 구현
3. `models/arima_model.py` — ARIMA 기본 구현
4. `models/sarima_model.py` — SARIMA 구현
5. `models/prophet_model.py` — Prophet 구현
6. `models/lstm_model.py` — LSTM 구현
7. `utils/window_strategy.py` — Rolling/Expanding window 로직
8. `utils/visualizer.py` — Plotly 시각화
9. `app.py` — Streamlit UI 통합
10. 배포 (`requirements.txt` 정리 후 Streamlit Cloud 연결)

---

## 배포 요구사항

- Streamlit Community Cloud (https://streamlit.io/cloud) 를 통한 온라인 배포
- GitHub 레포지토리 연결 필요
- `requirements.txt`에 모든 의존성 명시
- 배포 완료 후 URL 제출

---

## 제약 조건 및 주의사항

- 입력 CSV는 **단변량** (날짜 1개 + 값 1개 컬럼) 형식을 기준으로 한다
- LSTM은 학습 시간이 길 수 있으므로 epoch 기본값을 낮게 설정 (예: 20)
- Prophet은 날짜 컬럼명이 `ds`, 값 컬럼명이 `y`여야 하므로 내부에서 자동 변환 처리
- 결측치가 있을 경우 선형 보간(interpolation) 기본 적용, 사용자에게 알림
- 모든 예측 구간은 테스트 셋 기준이며, 미래 예측(out-of-sample)도 선택적으로 제공