import numpy as np
from darts import TimeSeries, concatenate
from darts.models import SKLearnModel
from darts.ad import (
    ForecastingAnomalyModel,
    NormScorer,
    KMeansScorer,
    WassersteinScorer,
    ThresholdDetector,
    OrAggregator,
    AndAggregator,
)
from sklearn.metrics import roc_auc_score, roc_curve

SCORER_NAMES = ["NormScorer", "KMeansScorer", "WassersteinScorer"]


def _build_forecasting_model(lags):
    return SKLearnModel(
        lags=lags,
        lags_future_covariates=[0],
        output_chunk_length=1,
        add_encoders={"cyclic": {"future": ["hour", "dayofweek"]}},
    )


def _trim_to_common_index(series_list):
    """여러 TimeSeries를 공통 시간 인덱스(교집합)로 잘라 반환."""
    common_start = max(s.start_time() for s in series_list)
    common_end = min(s.end_time() for s in series_list)
    return [s.slice(common_start, common_end) for s in series_list]


def _align_by_time(score_series, label_series):
    """두 시리즈를 공통 시간 인덱스로 정렬하여 numpy array 반환."""
    score_pd = score_series.to_series()
    label_pd = label_series.to_series()
    common_idx = score_pd.index.intersection(label_pd.index)
    if len(common_idx) == 0:
        raise ValueError("이상 점수 시리즈와 레이블 시리즈 간 겹치는 시간 인덱스가 없습니다.")
    return score_pd.loc[common_idx].values, label_pd.loc[common_idx].values, common_idx


def _format_duration(delta):
    total_seconds = int(abs(delta.total_seconds()))
    if total_seconds < 3600:
        return f"{total_seconds // 60}분"
    elif total_seconds < 86400:
        h = total_seconds // 3600
        m = (total_seconds % 3600) // 60
        return f"{h}시간 {m}분"
    else:
        d = total_seconds // 86400
        h = (total_seconds % 86400) // 3600
        return f"{d}일 {h}시간"


def classify_anomaly_types(binary_anomalies, score_series):
    """연속 이상 구간을 점이상/패턴이상으로 분류."""
    if binary_anomalies is None:
        return []

    binary_vals = binary_anomalies.univariate_values()
    score_vals = score_series.univariate_values()
    time_index = binary_anomalies.time_index

    min_len = min(len(binary_vals), len(score_vals), len(time_index))
    binary_vals = binary_vals[:min_len]
    score_vals = score_vals[:min_len]
    time_index = time_index[:min_len]

    segments = []
    i = 0
    while i < min_len:
        if binary_vals[i] == 1:
            start_idx = i
            while i < min_len and binary_vals[i] == 1:
                i += 1
            end_idx = i - 1

            length = end_idx - start_idx + 1
            anomaly_type = "점이상" if length <= 2 else "패턴이상"

            seg_scores = score_vals[start_idx : end_idx + 1]
            valid = seg_scores[~np.isnan(seg_scores)]
            max_score = float(np.max(valid)) if len(valid) > 0 else 0.0

            delta = time_index[end_idx] - time_index[start_idx]
            segments.append(
                {
                    "start": time_index[start_idx],
                    "end": time_index[end_idx],
                    "length": length,
                    "duration": _format_duration(delta),
                    "type": anomaly_type,
                    "max_score": max_score,
                }
            )
        else:
            i += 1

    return segments


def detect_anomalies(
    train,
    test,
    lags,
    label_series=None,
    aggregator_type="or",
    high_quantile=0.95,
):
    """
    이상탐지 메인 파이프라인.

    Returns dict:
        scores, scorer_names, strategy, final_anomalies, anomaly_segments
        (with_label만) auc_scores, best_scorer_idx, best_scorer_name, best_auc, optimal_threshold, roc_data
        (without_label만) thresholds
    """
    forecasting_model = _build_forecasting_model(lags)
    forecasting_model.fit(train)

    scorers = [NormScorer(), KMeansScorer(k=2), WassersteinScorer()]

    anomaly_model = ForecastingAnomalyModel(
        model=forecasting_model,
        scorer=scorers,
    )
    anomaly_model.fit(train, allow_model_training=False)

    raw_scores = anomaly_model.score(test)
    # score()는 단일 Scorer면 TimeSeries, 복수 Scorer면 tuple을 반환
    if isinstance(raw_scores, TimeSeries):
        raw_scores = [raw_scores]
    else:
        raw_scores = list(raw_scores)

    result = {
        "scores": raw_scores,  # 원본 길이 유지 (시각화용)
        "scorer_names": SCORER_NAMES,
        "strategy": None,
        "final_anomalies": None,
        "anomaly_segments": [],
    }

    if label_series is not None:
        result = _strategy_with_label(result, raw_scores, label_series)
    else:
        result = _strategy_without_label(result, raw_scores, aggregator_type, high_quantile)

    # final_anomalies와 score를 공통 구간으로 맞춰 classify
    aligned = _trim_to_common_index([result["final_anomalies"], raw_scores[0]])
    result["anomaly_segments"] = classify_anomaly_types(aligned[0], aligned[1])
    return result


def _strategy_with_label(result, raw_scores, label_series):
    """전략 2: 레이블 있는 경우 — AUC 기반 Scorer 자동 선택."""
    auc_scores = []
    roc_data = []

    for score in raw_scores:
        try:
            score_vals, label_vals, common_idx = _align_by_time(score, label_series)
            mask = ~(np.isnan(score_vals) | np.isnan(label_vals))
            s, l = score_vals[mask], label_vals[mask]

            if len(np.unique(l)) < 2:
                auc_scores.append(0.5)
                roc_data.append({"fpr": [0, 1], "tpr": [0, 1], "thresholds": [1, 0]})
                continue

            auc = roc_auc_score(l, s)
            fpr, tpr, thresholds = roc_curve(l, s)
            auc_scores.append(float(auc))
            roc_data.append({"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thresholds.tolist()})
        except Exception:
            auc_scores.append(0.5)
            roc_data.append({"fpr": [0, 1], "tpr": [0, 1], "thresholds": [1, 0]})

    best_idx = int(np.argmax(auc_scores))
    best_auc = auc_scores[best_idx]

    fpr_arr = np.array(roc_data[best_idx]["fpr"])
    tpr_arr = np.array(roc_data[best_idx]["tpr"])
    thr_arr = np.array(roc_data[best_idx]["thresholds"])

    opt_idx = int(np.argmax(tpr_arr - fpr_arr))
    optimal_threshold = float(thr_arr[opt_idx])

    best_score = raw_scores[best_idx]
    score_vals_all = best_score.univariate_values()
    binary_vals = (score_vals_all >= optimal_threshold).astype(float)
    final_anomalies = TimeSeries.from_times_and_values(
        best_score.time_index, binary_vals
    )

    result.update(
        {
            "strategy": "with_label",
            "auc_scores": auc_scores,
            "best_scorer_idx": best_idx,
            "best_scorer_name": SCORER_NAMES[best_idx],
            "best_auc": best_auc,
            "optimal_threshold": optimal_threshold,
            "roc_data": roc_data,
            "final_anomalies": final_anomalies,
        }
    )
    return result


def _strategy_without_label(result, raw_scores, aggregator_type, high_quantile):
    """전략 1: 레이블 없는 경우 — 분위수 임계치 + Aggregator."""
    binary_results = []
    thresholds = []

    for score in raw_scores:
        vals = score.univariate_values()
        clean = vals[~np.isnan(vals)]
        threshold = float(np.quantile(clean, high_quantile)) if len(clean) > 0 else 0.0
        thresholds.append(threshold)

        detector = ThresholdDetector(high_threshold=threshold)
        binary = detector.detect(score)
        binary_results.append(binary)

    # Scorer마다 윈도우 크기가 달라 길이가 다를 수 있으므로 공통 구간으로 정렬
    binary_results = _trim_to_common_index(binary_results)

    # Aggregator는 다변량 TimeSeries(컬럼 = Scorer 수) 하나를 받음
    combined = concatenate(binary_results, axis=1)
    aggregator = OrAggregator() if aggregator_type == "or" else AndAggregator()
    final_anomalies = aggregator.predict(combined)

    result.update(
        {
            "strategy": "without_label",
            "final_anomalies": final_anomalies,
            "binary_results": binary_results,
            "thresholds": thresholds,
        }
    )
    return result
