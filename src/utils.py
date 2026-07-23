"""常用工具函数模块。

这里放的是项目中多个模块都会重复使用的基础函数，
例如滚动均值、滚动标准差、延迟、差分、横截面排序等。

随着 Alpha191 因子数量增加，这个模块也承担了更多“公式算子库”的职责，
例如：

- `SMA` 对应的中国式平滑移动平均
- `WMA` / `DECAYLINEAR` 对应的加权平均
- `TSMAX` / `TSMIN` / `HIGHDAY` / `LOWDAY`
- `REGBETA` 对应的滚动回归斜率
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def safe_divide(numerator, denominator, eps: float = 1e-8):
    """安全除法。

    在量化因子计算中，分母为 0 的情况很常见，例如：

    - 某天成交量为 0
    - 某个价格区间宽度为 0
    - 某个滚动均值暂时不可用

    接近 0 的分母会被转成缺失值，正常分母不做任何加法修正。
固定给分母加 `eps` 会改变比例的经济含义，并且使原本应当对价格缩放不变的因子受到股价单位影响。
    """

    if isinstance(denominator, (pd.Series, pd.DataFrame)):
        numeric_denominator = denominator.astype(float)
        valid_denominator = numeric_denominator.where(numeric_denominator.abs() > eps)
        return numerator / valid_denominator

    if np.isscalar(denominator):
        try:
            denominator_value = float(denominator)
        except (TypeError, ValueError):
            denominator_value = float("nan")
        if not np.isfinite(denominator_value) or abs(denominator_value) <= eps:
            # Multiplication preserves a Series/DataFrame index when the numerator
            # is pandas-backed, while scalar numerators naturally become NaN.
            return numerator * np.nan
        return numerator / denominator_value

    denominator_array = np.asarray(denominator, dtype=float)
    valid_denominator = np.where(
        np.isfinite(denominator_array) & (np.abs(denominator_array) > eps),
        denominator_array,
        np.nan,
    )
    return np.asarray(numerator) / valid_denominator


def rolling_mean(series: pd.Series, window: int, min_periods: int = 1) -> pd.Series:
    """计算滚动均值。"""

    return series.rolling(window=window, min_periods=min_periods).mean()


def rolling_sum(series: pd.Series, window: int, min_periods: int = 1) -> pd.Series:
    """计算滚动求和。"""

    return series.rolling(window=window, min_periods=min_periods).sum()


def rolling_std(series: pd.Series, window: int, min_periods: int = 1) -> pd.Series:
    """计算滚动标准差。"""

    return series.rolling(window=window, min_periods=min_periods).std()


def rolling_min(series: pd.Series, window: int, min_periods: int = 1) -> pd.Series:
    """计算滚动最小值。"""

    return series.rolling(window=window, min_periods=min_periods).min()


def rolling_max(series: pd.Series, window: int, min_periods: int = 1) -> pd.Series:
    """计算滚动最大值。"""

    return series.rolling(window=window, min_periods=min_periods).max()


def rolling_prod(series: pd.Series, window: int, min_periods: int = 1) -> pd.Series:
    """计算滚动连乘积。"""

    return series.rolling(window=window, min_periods=min_periods).apply(np.prod, raw=True)


def delay(series: pd.Series, periods: int = 1) -> pd.Series:
    """将序列向后移动若干期。

    在因子研究中，延迟操作非常常见，用于表达“前一天值”“前 N 天值”等概念。
    """

    return series.shift(periods)


def delta(series: pd.Series, periods: int = 1) -> pd.Series:
    """计算当前值与若干期之前值的差。"""

    return series.diff(periods)


def rolling_corr(series_a: pd.Series, series_b: pd.Series, window: int, min_periods: int = 2) -> pd.Series:
    """计算两个序列的滚动相关系数。"""

    return series_a.rolling(window=window, min_periods=min_periods).corr(series_b)


def rolling_cov(series_a: pd.Series, series_b: pd.Series, window: int, min_periods: int = 2) -> pd.Series:
    """计算两个序列的滚动协方差。"""

    return series_a.rolling(window=window, min_periods=min_periods).cov(series_b)


def cross_sectional_rank(series: pd.Series) -> pd.Series:
    """横截面百分位排名。

    这个函数通常搭配 `groupby("date").transform(...)` 使用，
    表示“在同一天所有股票中，这个值排在什么位置”。
    """

    return series.rank(method="average", pct=True)


def ts_rank(series: pd.Series, window: int) -> pd.Series:
    """时间序列排名。

    对于每一个时点，我们取最近 `window` 个样本，
    然后看当前值在这个滚动窗口里的相对排名。
    """

    def _rank_last(window_series: pd.Series) -> float:
        valid = window_series.dropna()
        if valid.empty:
            return float("nan")
        return float(valid.rank(method="average", pct=True).iloc[-1])

    return series.rolling(window=window, min_periods=1).apply(_rank_last, raw=False)


def ts_argmax(series: pd.Series, window: int) -> pd.Series:
    """返回窗口内最高值距离当前时点的“日序号”。

    这里的返回值遵循很多量化公式里的约定：

    - 窗口最早的位置记为 1
    - 窗口最后也就是当前的位置记为 `window`

    因此如果结果越接近 `window`，表示高点越靠近当前时点。
    """

    def _argmax(window_values: np.ndarray) -> float:
        if len(window_values) == 0 or np.all(np.isnan(window_values)):
            return float("nan")
        return float(np.nanargmax(window_values) + 1)

    return series.rolling(window=window, min_periods=1).apply(_argmax, raw=True)


def ts_argmin(series: pd.Series, window: int) -> pd.Series:
    """返回窗口内最低值距离当前时点的“日序号”。"""

    def _argmin(window_values: np.ndarray) -> float:
        if len(window_values) == 0 or np.all(np.isnan(window_values)):
            return float("nan")
        return float(np.nanargmin(window_values) + 1)

    return series.rolling(window=window, min_periods=1).apply(_argmin, raw=True)


def decay_linear(series: pd.Series, window: int) -> pd.Series:
    """线性衰减加权平均。

    最近的数据给予更大权重，较早的数据给予更小权重，
    这是很多 Alpha 因子里常见的平滑方式。
    """

    weights = np.arange(1, window + 1, dtype=float)

    def _apply(window_series: pd.Series) -> float:
        # 缺失市场观测不应被前后填充成一条伪造路径。
        # 这里只使用窗口内真实存在的数值，并保留它们原本的时间权重位置：
        # 早期值权重较小，越接近当前日权重越大。
        numeric_values = pd.to_numeric(window_series, errors="coerce").to_numpy(dtype=float)
        valid_mask = np.isfinite(numeric_values)
        if not valid_mask.any():
            return float("nan")
        positional_weights = weights[-len(numeric_values) :]
        valid_values = numeric_values[valid_mask]
        valid_weights = positional_weights[valid_mask]
        return float(np.dot(valid_values, valid_weights) / valid_weights.sum())

    return series.rolling(window=window, min_periods=1).apply(_apply, raw=False)


def wma(series: pd.Series, window: int) -> pd.Series:
    """线性加权移动平均。

    这个函数与 `decay_linear` 十分接近，但命名上更贴近许多技术指标和 Alpha 公式中的 `WMA`。
    """

    return decay_linear(series, window=window)


def sma_cn(series: pd.Series, n: int, m: int) -> pd.Series:
    """实现国内技术分析公式中常见的 `SMA(X, N, M)`。

    这是一个递推平滑序列，其计算方式为：

    `Y_t = (M * X_t + (N - M) * Y_{t-1}) / N`

    在 pandas 中可以用 `ewm(alpha=M/N)` 近似等价实现。
    """

    alpha = m / float(n)
    return series.ewm(alpha=alpha, adjust=False).mean()


def rolling_count(condition: pd.Series, window: int) -> pd.Series:
    """计算滚动窗口内条件成立的次数。"""

    return condition.astype(float).rolling(window=window, min_periods=1).sum()


def rolling_sum_if(values: pd.Series, condition: pd.Series, window: int) -> pd.Series:
    """在滚动窗口内，对满足条件的位置做求和。"""

    masked_values = values.where(condition, 0.0)
    return rolling_sum(masked_values, window=window)


def rolling_mean_abs_dev(series: pd.Series, window: int) -> pd.Series:
    """计算滚动平均绝对离差。

    这个统计量在 CCI、某些震荡类因子中很常见。
    """

    def _mad(window_values: pd.Series) -> float:
        valid = pd.Series(window_values).dropna()
        if valid.empty:
            return float("nan")
        mean_value = valid.mean()
        return float(np.mean(np.abs(valid - mean_value)))

    return series.rolling(window=window, min_periods=1).apply(_mad, raw=False)


def rolling_regression_beta(series: pd.Series, window: int) -> pd.Series:
    """计算序列对时间序列 `1..window` 的滚动线性回归斜率。

    这可以近似对应 Alpha191 里若干 `REGBETA(..., SEQUENCE(window))` 类型公式。
    """

    def _beta(window_values: pd.Series) -> float:
        valid = pd.Series(window_values).dropna()
        if len(valid) < 2:
            return float("nan")
        x = np.arange(1, len(valid) + 1, dtype=float)
        y = valid.to_numpy(dtype=float)
        x_mean = x.mean()
        y_mean = y.mean()
        denominator = np.sum((x - x_mean) ** 2)
        if denominator == 0:
            return float("nan")
        numerator = np.sum((x - x_mean) * (y - y_mean))
        return float(numerator / denominator)

    return series.rolling(window=window, min_periods=2).apply(_beta, raw=False)


def ema(series: pd.Series, span: int) -> pd.Series:
    """计算指数移动平均线。"""

    return series.ewm(span=span, adjust=False).mean()
