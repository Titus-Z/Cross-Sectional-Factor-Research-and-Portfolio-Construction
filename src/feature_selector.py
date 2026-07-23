"""特征筛选模块。"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

from src.provenance import dumps_strict_json


class FeatureSelector:
    """一个简化但完整的特征筛选流水线。

    处理顺序与题目要求一致：

    1. 删除缺失率过高的特征
    2. 用中位数填充剩余缺失值
    3. 删除低方差特征
    4. 删除高相关特征
    5. 基于相关性或互信息选取 Top N
    """

    def __init__(
        self,
        missing_threshold: float = 0.5,
        variance_threshold: float = 0.001,
        correlation_threshold: float = 0.95,
        top_n: int = 50,
        score_method: str = "correlation",
        random_state: int = 42,
    ) -> None:
        self.missing_threshold = missing_threshold
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.top_n = top_n
        self.score_method = score_method
        self.random_state = int(random_state)

        self.columns_after_missing_: list[str] = []
        self.columns_after_variance_: list[str] = []
        self.columns_after_correlation_: list[str] = []
        self.selected_features_: list[str] = []
        self.initial_feature_count_: int = 0
        self.fill_values_: dict[str, float] = {}
        self.feature_scores_: dict[str, float] = {}
        self.stage_feature_counts_: dict[str, int] = {}
        self.score_basis_: str = "unfitted"

    @staticmethod
    def _coerce_finite_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
        """Convert candidates to numeric values and treat infinities as missing.

        Formula features can produce infinities after a near-zero denominator.
        Leaving them in the matrix makes missing-rate, median, variance, and
        correlation calculations inconsistent across pandas versions.
        """

        return frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

    def _passes_variance_threshold(self, series: pd.Series) -> bool:
        """Compare variance with the configured threshold without squaring huge values.

        Dividing a column by a positive constant preserves whether it is constant.
        The original variance equals ``scaled_variance * scale**2``. Comparing the
        logarithms avoids overflow when a formula emits very large finite values.
        """

        numeric = pd.to_numeric(series, errors="coerce")
        scale = float(numeric.abs().max()) if numeric.notna().any() else 0.0
        if not np.isfinite(scale) or scale <= 0.0:
            return False
        scaled_variance = float((numeric / scale).var())
        if not np.isfinite(scaled_variance) or scaled_variance <= 0.0:
            return False
        if self.variance_threshold <= 0.0:
            return True
        log_variance = math.log(scaled_variance) + 2.0 * math.log(scale)
        return bool(log_variance >= math.log(self.variance_threshold))

    @staticmethod
    def _mean_daily_cross_sectional_correlation(
        feature: pd.Series,
        target: pd.Series,
        dates: pd.Series,
    ) -> float:
        """Score one feature using the same daily cross-sectional IC as evaluation."""

        score_frame = pd.DataFrame(
            {
                "feature": pd.to_numeric(feature, errors="coerce").reset_index(drop=True),
                "target": pd.to_numeric(target, errors="coerce").reset_index(drop=True),
                "date": pd.to_datetime(dates, errors="coerce").reset_index(drop=True),
            }
        )
        daily_values: list[float] = []
        for _, date_frame in score_frame.groupby("date", sort=True):
            valid = date_frame.dropna(subset=["feature", "target"])
            if len(valid) < 2:
                continue
            if valid["feature"].nunique() <= 1 or valid["target"].nunique() <= 1:
                continue
            correlation = valid["feature"].corr(valid["target"], method="pearson")
            if pd.notna(correlation):
                daily_values.append(float(correlation))
        if not daily_values:
            return 0.0
        return float(abs(np.mean(daily_values)))

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dates: pd.Series | None = None,
    ) -> "FeatureSelector":
        """在训练集上拟合特征筛选器。"""

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows.")
        if dates is not None and len(X) != len(dates):
            raise ValueError("X and dates must have the same number of rows.")

        X_numeric = self._coerce_finite_numeric_frame(X)
        self.initial_feature_count_ = int(X_numeric.shape[1])

        # 步骤 1：按缺失率删除特征。
        missing_ratio = X_numeric.isna().mean()
        self.columns_after_missing_ = missing_ratio[missing_ratio <= self.missing_threshold].index.tolist()
        X_stage_1 = X_numeric[self.columns_after_missing_].copy()

        if not self.columns_after_missing_:
            raise ValueError("No features remain after missing-ratio filtering.")

        # 步骤 2：用训练集的中位数填充缺失值，并把这些填充值记录下来，
        # 这样后续对验证集和测试集也能保持一致处理。
        self.fill_values_ = X_stage_1.median(numeric_only=True).fillna(0.0).to_dict()
        X_stage_2 = X_stage_1.fillna(self.fill_values_)

        # 步骤 3：删除低方差特征。
        self.columns_after_variance_ = [
            column
            for column in X_stage_2.columns
            if self._passes_variance_threshold(X_stage_2[column])
        ]
        X_stage_3 = X_stage_2[self.columns_after_variance_].copy()

        if not self.columns_after_variance_:
            raise ValueError("No features remain after low-variance filtering.")

        # 步骤 4：删除高相关特征。
        # 这里用上三角矩阵逐列检查，是一个简单直观的实现方式。
        # Pearson correlation is unchanged by positive column scaling. Scaling
        # every column to max-absolute value near one avoids covariance overflow.
        correlation_scale = X_stage_3.abs().max().replace(0.0, 1.0)
        correlation_matrix = X_stage_3.divide(correlation_scale, axis=1).corr().abs()
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        columns_to_drop = [
            column for column in upper_triangle.columns if (upper_triangle[column] > self.correlation_threshold).any()
        ]
        self.columns_after_correlation_ = [column for column in X_stage_3.columns if column not in columns_to_drop]
        X_stage_4 = X_stage_3[self.columns_after_correlation_].copy()

        if not self.columns_after_correlation_:
            raise ValueError("No features remain after high-correlation filtering.")

        # 步骤 5：按相关系数或互信息对特征打分，再保留 Top N。
        # 单变量打分同样使用正比例缩放后的列，避免极端公式值在相关性
        # 内部协方差计算中溢出。缩放不改变 Pearson 方向或样本排序。
        score_scale = X_stage_4.abs().max().replace(0.0, 1.0)
        X_stage_4_scaled = X_stage_4.divide(score_scale, axis=1)
        if self.score_method == "mutual_info":
            mi_scores = mutual_info_regression(
                X_stage_4_scaled,
                y,
                random_state=self.random_state,
            )
            scores = pd.Series(mi_scores, index=X_stage_4_scaled.columns).fillna(0.0)
            self.score_basis_ = "pooled_mutual_information_train_only"
        elif dates is not None:
            date_series = pd.Series(dates).reset_index(drop=True)
            target_series = pd.Series(y).reset_index(drop=True)
            scores = pd.Series(
                {
                    column: self._mean_daily_cross_sectional_correlation(
                        X_stage_4_scaled[column],
                        target_series,
                        date_series,
                    )
                    for column in X_stage_4_scaled.columns
                }
            ).fillna(0.0)
            self.score_basis_ = "absolute_mean_daily_cross_sectional_pearson_ic_train_only"
        else:
            scores = X_stage_4_scaled.apply(lambda series: abs(series.corr(y))).fillna(0.0)
            self.score_basis_ = "absolute_pooled_pearson_correlation_train_only_compatibility"

        scores = scores.sort_values(ascending=False)
        keep_count = min(self.top_n, len(scores))
        self.selected_features_ = scores.head(keep_count).index.tolist()
        self.feature_scores_ = scores.to_dict()

        self.stage_feature_counts_ = {
            "initial": self.initial_feature_count_,
            "after_missing_filter": len(self.columns_after_missing_),
            "after_variance_filter": len(self.columns_after_variance_),
            "after_correlation_filter": len(self.columns_after_correlation_),
            "after_top_n_selection": len(self.selected_features_),
        }

        if not self.selected_features_:
            raise ValueError("No features remain after Top N selection.")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """将训练时学到的筛选规则应用到新数据上。"""

        if not self.selected_features_:
            raise RuntimeError("FeatureSelector has not been fitted yet.")

        X_transformed = self._coerce_finite_numeric_frame(X[self.columns_after_missing_].copy())
        X_transformed = X_transformed.fillna(self.fill_values_)
        X_transformed = X_transformed[self.columns_after_variance_]
        X_transformed = X_transformed[self.columns_after_correlation_]
        X_transformed = X_transformed[self.selected_features_]
        return X_transformed.reset_index(drop=True)

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dates: pd.Series | None = None,
    ) -> pd.DataFrame:
        """先拟合再转换，便于在训练集上一行调用。"""

        return self.fit(X, y, dates=dates).transform(X)

    def to_dict(self) -> dict:
        """导出筛选器配置和结果，便于保存。"""

        return {
            "missing_threshold": self.missing_threshold,
            "variance_threshold": self.variance_threshold,
            "correlation_threshold": self.correlation_threshold,
            "top_n": self.top_n,
            "score_method": self.score_method,
            "random_state": self.random_state,
            "score_basis": self.score_basis_,
            "initial_feature_count": self.initial_feature_count_,
            "stage_feature_counts": self.stage_feature_counts_,
            "columns_after_missing": self.columns_after_missing_,
            "columns_after_variance": self.columns_after_variance_,
            "columns_after_correlation": self.columns_after_correlation_,
            "selected_features": self.selected_features_,
            "fill_values": {key: float(value) for key, value in self.fill_values_.items()},
            "feature_scores": {key: float(value) for key, value in self.feature_scores_.items()},
        }

    def get_top_features(self, top_k: int = 10) -> pd.DataFrame:
        """返回按筛选得分排序的前若干个特征。"""

        score_series = pd.Series(self.feature_scores_, name="score").sort_values(ascending=False)
        top_features = score_series.head(top_k).reset_index()
        top_features.columns = ["feature", "score"]
        return top_features

    def save(self, output_path: str | Path) -> None:
        """保存筛选器元数据到 JSON 文件。"""

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(dumps_strict_json(self.to_dict()), encoding="utf-8")
