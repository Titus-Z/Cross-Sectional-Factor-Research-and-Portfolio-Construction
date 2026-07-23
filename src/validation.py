"""Walk-forward validation 模块。

这个模块负责把“单次 train/test 切分”升级成更接近实战的验证方式：

- 使用 `TimeSeriesSplit` 在训练期内部继续做多折时间序列验证；
- 每个 fold 都单独拟合特征筛选器和模型；
- 每个模型在每个 fold 上都记录完整指标；
- 最后再根据平均验证表现给模型分配集成权重。
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.feature_selector import FeatureSelector
from src.model import build_model
from src.model_params import model_param_candidates_for_model
from src.progress import create_progress_bar, format_duration
from src.reporting import calculate_prediction_metrics
from src.time_series_pipeline import purge_training_label_overlap


def generate_walk_forward_folds(
    data: pd.DataFrame,
    n_splits: int = 5,
    date_column: str = "date",
    purge_days: int = 0,
) -> list[tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]]:
    """按日期生成 walk-forward folds。

    这里不是直接对“总行数”做 TimeSeriesSplit，而是先对唯一日期做 split，
    然后再把日期映射回完整的横截面样本。

    这样可以保证：

    - 同一天的所有股票都在同一个 fold 里；
    - 不会出现同一天一半在训练、一半在验证的情况；
    - 更贴近量化横截面建模的真实使用方式。
    """

    unique_dates = np.array(sorted(pd.to_datetime(data[date_column]).dropna().unique()))
    if len(unique_dates) <= n_splits:
        raise ValueError("Number of unique dates must be greater than n_splits.")

    splitter = TimeSeriesSplit(n_splits=n_splits)
    folds: list[tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]] = []

    for fold_number, (train_index, valid_index) in enumerate(splitter.split(unique_dates), start=1):
        train_dates_before_purge = unique_dates[train_index]
        valid_dates = unique_dates[valid_index]

        # Forward-return labels are shifted within each instrument. Purging the
        # final N global dates is insufficient when a stock has missing dates,
        # so the shared helper removes the final N observations per instrument.
        if purge_days < 0:
            raise ValueError("purge_days must be non-negative.")
        train_fold_before_purge = data[
            pd.to_datetime(data[date_column]).isin(train_dates_before_purge)
        ].copy().reset_index(drop=True)
        if purge_days > 0:
            train_fold, purge_summary = purge_training_label_overlap(
                train_fold_before_purge,
                target_horizon=purge_days,
            )
        else:
            train_fold = train_fold_before_purge
            purge_summary = {
                "purged_date_count": 0,
                "purged_row_count": 0,
                "purge_policy": "disabled",
            }

        valid_fold = data[pd.to_datetime(data[date_column]).isin(valid_dates)].copy().reset_index(drop=True)
        train_dates_after_purge = pd.Index(sorted(pd.to_datetime(train_fold[date_column]).unique()))

        fold_summary = {
            "fold": int(fold_number),
            "train_min_date": str(pd.Timestamp(train_dates_after_purge[0]).date()),
            "train_max_date": str(pd.Timestamp(train_dates_after_purge[-1]).date()),
            "valid_min_date": str(pd.Timestamp(valid_dates[0]).date()),
            "valid_max_date": str(pd.Timestamp(valid_dates[-1]).date()),
            "train_rows": int(len(train_fold)),
            "valid_rows": int(len(valid_fold)),
            "purge_days": int(purge_days),
            "purged_train_date_count": int(purge_summary["purged_date_count"]),
            "purged_train_row_count": int(purge_summary["purged_row_count"]),
            "purge_policy": str(purge_summary["purge_policy"]),
            "train_max_date_before_purge": str(pd.Timestamp(train_dates_before_purge[-1]).date()),
        }
        folds.append((train_fold, valid_fold, fold_summary))

    return folds


def calculate_model_weights(
    model_summary_df: pd.DataFrame,
    score_metric: str = "pearson_ic_mean",
) -> dict[str, float]:
    """根据平均验证表现计算模型权重。

    这里使用一个很简单但实用的思路：

    1. 先取每个模型的平均验证分数；
    2. 负 IC 不获得额外表现权重；
    3. 将正分归一化后，与等权组合做 50/50 收缩。

    收缩的目的，是防止两个非常接近的弱 IC 因为简单平移而变成近似
    100% / 0% 的极端权重。最终权重仍反映验证表现，但不会把验证噪声
    误当成确定性排序。
    """

    if model_summary_df.empty:
        return {}

    score_series = model_summary_df.set_index("model")[score_metric].astype(float)
    score_series = score_series.replace([np.inf, -np.inf], np.nan)

    # A fold can legitimately have no cross-sectional dispersion, leaving every
    # model score undefined. In that case the validation layer has no evidence
    # for preferring one model, so equal weights are the only defensible fallback.
    # Calling `fillna(score_series.min())` directly would keep all values as NaN
    # when the entire column is missing and could contaminate the final ensemble.
    finite_scores = score_series.dropna()
    if finite_scores.empty:
        uniform_weight = 1.0 / max(len(score_series), 1)
        return {model_name: uniform_weight for model_name in score_series.index}
    score_series = score_series.fillna(0.0)
    uniform_weight = 1.0 / max(len(score_series), 1)
    positive_scores = score_series.clip(lower=0.0)
    if float(positive_scores.sum()) <= 1e-12:
        return {model_name: uniform_weight for model_name in score_series.index}

    performance_weights = positive_scores / positive_scores.sum()
    shrinkage_to_equal = 0.5
    final_weights = (
        (1.0 - shrinkage_to_equal) * performance_weights
        + shrinkage_to_equal * uniform_weight
    )
    return {model_name: float(weight) for model_name, weight in final_weights.items()}


def run_walk_forward_validation(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    model_names: list[str],
    selector_config: dict[str, Any],
    model_params_by_name: dict[str, dict[str, Any]] | None = None,
    hyperparameter_grid_by_name: dict[str, list[dict[str, Any]]] | None = None,
    random_state: int = 42,
    n_splits: int = 5,
    purge_days: int = 0,
    score_metric: str = "pearson_ic_mean",
    show_progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """执行完整的 walk-forward validation。

    返回三个结果：

    1. 每个 fold、每个模型的详细指标；
    2. 按模型聚合后的平均指标；
    3. 根据平均验证分数算出的模型权重。
    """

    fold_records: list[dict[str, Any]] = []
    folds = generate_walk_forward_folds(train_df, n_splits=n_splits, purge_days=purge_days)
    overall_start_time = time.perf_counter()
    fold_progress = create_progress_bar(
        total=len(folds),
        description="Walk-forward folds",
        enabled=show_progress,
    )

    for fold_index, (train_fold, valid_fold, fold_summary) in enumerate(folds, start=1):
        fold_start_time = time.perf_counter()
        # 注意：特征筛选器也必须在每个 fold 内部单独拟合，
        # 不能拿全训练期的 y 提前帮未来验证折选择特征。
        selector = FeatureSelector(**selector_config)
        selector.fit(
            train_fold[feature_columns],
            train_fold["y"],
            dates=train_fold["date"],
        )

        X_train = selector.transform(train_fold[feature_columns])
        y_train = train_fold["y"].reset_index(drop=True)
        X_valid = selector.transform(valid_fold[feature_columns])

        model_candidate_counts = {
            model_name: len(
                model_param_candidates_for_model(model_name, model_params_by_name, hyperparameter_grid_by_name)
            )
            for model_name in model_names
        }
        model_progress = create_progress_bar(
            total=sum(model_candidate_counts.values()),
            description=f"Fold {fold_summary['fold']} models",
            enabled=show_progress,
            leave=False,
        )

        for model_name in model_names:
            candidates = model_param_candidates_for_model(model_name, model_params_by_name, hyperparameter_grid_by_name)
            for candidate_index, candidate_params in enumerate(candidates, start=1):
                model_start_time = time.perf_counter()
                model_wrapper = build_model(
                    model_name=model_name,
                    random_state=random_state,
                    params=candidate_params,
                )
                model_wrapper.fit(X_train, y_train)

                valid_predictions = model_wrapper.predict(X_valid)
                valid_result_df = valid_fold[["date", "instrument_id", "y"]].copy()
                valid_result_df["predicted_y"] = valid_predictions
                model_elapsed = time.perf_counter() - model_start_time
                model_params_json = json.dumps(candidate_params or {}, ensure_ascii=True, sort_keys=True)
                param_set_id = f"grid_{candidate_index}" if len(candidates) > 1 else "default"

                fold_metrics = calculate_prediction_metrics(valid_result_df)
                fold_metrics.update(
                    {
                        "fold": fold_summary["fold"],
                        "model": model_name,
                        "param_set_id": param_set_id,
                        "model_params_json": model_params_json,
                        "train_min_date": fold_summary["train_min_date"],
                        "train_max_date": fold_summary["train_max_date"],
                        "valid_min_date": fold_summary["valid_min_date"],
                        "valid_max_date": fold_summary["valid_max_date"],
                        "train_rows": fold_summary["train_rows"],
                        "valid_rows": fold_summary["valid_rows"],
                        "purge_days": fold_summary["purge_days"],
                        "purged_train_date_count": fold_summary["purged_train_date_count"],
                        "purged_train_row_count": fold_summary["purged_train_row_count"],
                        "purge_policy": fold_summary["purge_policy"],
                        "train_max_date_before_purge": fold_summary["train_max_date_before_purge"],
                        "selected_feature_count": int(len(selector.selected_features_)),
                        "fit_predict_time_sec": float(model_elapsed),
                    }
                )
                fold_records.append(fold_metrics)
                model_progress.update(1)
                model_progress.set_postfix_str(
                    f"{model_name}:{param_set_id} | {format_duration(model_elapsed)} / candidate"
                )

        model_progress.close()
        fold_elapsed = time.perf_counter() - fold_start_time
        total_elapsed = time.perf_counter() - overall_start_time
        average_fold_seconds = total_elapsed / max(fold_index, 1)
        remaining_fold_count = len(folds) - fold_index
        estimated_remaining = average_fold_seconds * remaining_fold_count
        fold_progress.update(1)
        fold_progress.set_postfix_str(
            (
                f"latest=Fold {fold_summary['fold']} {fold_summary['valid_min_date']}→"
                f"{fold_summary['valid_max_date']} | fold {format_duration(fold_elapsed)} | "
                f"est left {format_duration(estimated_remaining)}"
            )
        )

    fold_progress.close()

    fold_metrics_df = pd.DataFrame(fold_records)
    if fold_metrics_df.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            {},
        )

    numeric_columns = [
        "pearson_corr",
        "spearman_corr",
        "rmse",
        "mae",
        "pearson_ic_mean",
        "pearson_ic_median",
        "pearson_ic_std",
        "pearson_ic_icir",
        "pearson_ic_positive_ratio",
        "spearman_ic_mean",
        "spearman_ic_median",
        "spearman_ic_std",
        "spearman_ic_icir",
        "spearman_ic_positive_ratio",
        "long_short_spread",
        "group_monotonic_spearman",
        "prediction_coverage_ratio",
        "evaluation_date_count",
        "selected_feature_count",
        "fit_predict_time_sec",
    ]
    param_group_columns = ["model", "param_set_id", "model_params_json"]
    param_summary_df = (
        fold_metrics_df.groupby(param_group_columns, as_index=False)[numeric_columns]
        .mean()
        .sort_values(score_metric, ascending=False)
        .reset_index(drop=True)
    )
    model_summary_df = (
        param_summary_df.sort_values(score_metric, ascending=False)
        .groupby("model", as_index=False)
        .head(1)
        .sort_values(score_metric, ascending=False)
        .reset_index(drop=True)
    )
    model_weights = calculate_model_weights(model_summary_df, score_metric=score_metric)
    return fold_metrics_df, model_summary_df, model_weights
