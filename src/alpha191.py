"""国泰君安 Alpha191 因子模块。

本模块在原来“5 个教学示例因子”的基础上，扩展为一个更接近
国泰君安 Alpha191 风格的实现框架。

当前版本特点：

- 已实现一批可直接基于 OHLCV / VWAP 数据计算的 Alpha191 公式
- 使用统一的底层算子封装滚动窗口、横截面排名、相关系数、SMA/WMA 等操作
- 支持批量因子生成，并可选显示进度条
- 对暂未实现的因子给出明确说明，避免误导使用者

重要说明：

- 由于用户给出的 Alpha191 公式文本中包含部分排版问题、拼写问题和少量歧义，
  本文件在实现时对个别公式做了保守且常见的工程化解释，并在代码注释中说明。
- 一些依赖基准指数、Fama-French 因子、`SELF` 递归状态或复杂自定义运算符的公式，
  暂时没有强行写成“看起来能跑”的版本，而是明确标记为未实现。
- Plain `RANK(...)` 按 Alpha191 常见约定处理为“同一交易日横截面排名”；
  `TSRANK(...)` 则处理为“单只股票时间序列窗口排名”。
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable

import numpy as np
import pandas as pd

from src.utils import (
    cross_sectional_rank,
    decay_linear,
    delay,
    delta,
    rolling_corr,
    rolling_count,
    rolling_cov,
    rolling_max,
    rolling_mean,
    rolling_mean_abs_dev,
    rolling_min,
    rolling_regression_beta,
    rolling_std,
    rolling_sum,
    safe_divide,
    sma_cn,
    ts_rank,
    wma,
)

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


# Yahoo-style back-adjustment can rescale one stock's entire historical price
# path after a later split or distribution.  A formula that uses an absolute
# price level or absolute price difference can therefore acquire a scale that
# was not observable at the historical signal date.  The public baseline uses
# only formulas whose output is unchanged when every OHLC/VWAP observation of
# one instrument is multiplied by an arbitrary positive constant.  Alpha011 is
# deliberately excluded because its absolute-volume multiplier is not invariant
# to a vendor changing historical share units around a split.
CANONICAL_SCALE_INVARIANT_ALPHA_FACTORS = (
    "alpha001",
    "alpha002",
    "alpha004",
    "alpha005",
    "alpha006",
    "alpha015",
    "alpha018",
    "alpha019",
    "alpha020",
    "alpha022",
    "alpha023",
)


def _iter_with_progress(items: Iterable, description: str, show_progress: bool):
    """在批量生成因子时提供可选进度条。"""

    if not show_progress or tqdm is None:
        return items
    return tqdm(items, desc=description)


def _to_series(value, index: pd.Index) -> pd.Series:
    """把标量或数组统一转成与原始数据对齐的 Series。"""

    if isinstance(value, pd.Series):
        return value
    return pd.Series(value, index=index)


def _cs_rank(date_index: pd.Series, values: pd.Series) -> pd.Series:
    """对同一交易日的股票做横截面排名。"""

    return values.groupby(date_index).transform(cross_sectional_rank)


def _group_transform(group_index: pd.Series, values: pd.Series, func: Callable[[pd.Series], pd.Series]) -> pd.Series:
    """对每只股票分别执行时序变换。"""

    return values.groupby(group_index).transform(func)


def _group_binary_rolling(
    group_index: pd.Series,
    left: pd.Series,
    right: pd.Series,
    operator: Callable[[pd.Series, pd.Series], pd.Series],
) -> pd.Series:
    """对两列序列按股票分组后执行双输入滚动运算。"""

    temp = pd.DataFrame(
        {
            "group": group_index,
            "left": left,
            "right": right,
        },
        index=left.index,
    )
    results = []
    for _, group in temp.groupby("group", sort=False):
        result = operator(group["left"], group["right"])
        result.index = group.index
        results.append(result)
    return pd.concat(results).sort_index()


def _group_corr(group_index: pd.Series, left: pd.Series, right: pd.Series, window: int) -> pd.Series:
    """按股票分组计算滚动相关系数。"""

    return _group_binary_rolling(
        group_index,
        left,
        right,
        operator=lambda a, b: rolling_corr(a, b, window=window),
    )


def _group_cov(group_index: pd.Series, left: pd.Series, right: pd.Series, window: int) -> pd.Series:
    """按股票分组计算滚动协方差。"""

    return _group_binary_rolling(
        group_index,
        left,
        right,
        operator=lambda a, b: rolling_cov(a, b, window=window),
    )


def _days_since_argmax(series: pd.Series, window: int) -> pd.Series:
    """返回窗口内最高值距离“今天”的天数。

    约定：

    - 如果最高值就是今天，返回 0
    - 如果最高值出现在 3 天前，返回 3

    这个定义与很多技术指标里 `HIGHDAY` / `LOWDAY` 的习惯更一致。
    """

    def _apply(window_values: np.ndarray) -> float:
        if len(window_values) == 0 or np.all(np.isnan(window_values)):
            return float("nan")
        return float(len(window_values) - 1 - np.nanargmax(window_values))

    return series.rolling(window=window, min_periods=1).apply(_apply, raw=True)


def _days_since_argmin(series: pd.Series, window: int) -> pd.Series:
    """返回窗口内最低值距离“今天”的天数。"""

    def _apply(window_values: np.ndarray) -> float:
        if len(window_values) == 0 or np.all(np.isnan(window_values)):
            return float("nan")
        return float(len(window_values) - 1 - np.nanargmin(window_values))

    return series.rolling(window=window, min_periods=1).apply(_apply, raw=True)


def _where(condition: pd.Series, left, right, index: pd.Index) -> pd.Series:
    """对 Series 做向量化条件分支。"""

    left_series = _to_series(left, index=index)
    right_series = _to_series(right, index=index)
    return pd.Series(np.where(condition, left_series, right_series), index=index)


def _signed_volume(close: pd.Series) -> pd.Series:
    """构造涨跌方向签名量。"""

    prev_close = delay(close, 1)
    return pd.Series(
        np.where(
            close > prev_close,
            1.0,
            np.where(close < prev_close, -1.0, 0.0),
        ),
        index=close.index,
    )


def _prepare_base_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """为 Alpha191 计算准备基础字段。

    之所以单独做这一步，是因为许多公式会重复使用同一批中间量：

    - 收益率 `ret`
    - 成交额 `amount`
    - 前一日 OHLC
    - 典型价格 `hlc3`
    """

    df = data.copy()
    instrument_index = df["instrument_id"]

    df["ret"] = _group_transform(instrument_index, df["close"], lambda s: safe_divide(s, delay(s, 1)) - 1.0)
    df["amount"] = df["turnover"] if "turnover" in df.columns else df["close"] * df["volume"]
    df["prev_close"] = _group_transform(instrument_index, df["close"], lambda s: delay(s, 1))
    df["prev_open"] = _group_transform(instrument_index, df["open"], lambda s: delay(s, 1))
    df["prev_high"] = _group_transform(instrument_index, df["high"], lambda s: delay(s, 1))
    df["prev_low"] = _group_transform(instrument_index, df["low"], lambda s: delay(s, 1))
    df["hlc3"] = (df["high"] + df["low"] + df["close"]) / 3.0
    return df.replace([np.inf, -np.inf], np.nan)


def _rsi_style(group_index: pd.Series, base_series: pd.Series, window: int) -> pd.Series:
    """实现 Alpha191 中多次出现的 RSI 风格比值公式。"""

    delta_value = _group_transform(group_index, base_series, lambda s: delta(s, 1))
    positive_part = pd.Series(np.maximum(delta_value, 0.0), index=base_series.index)
    absolute_part = pd.Series(np.abs(delta_value), index=base_series.index)
    positive_sma = _group_transform(group_index, positive_part, lambda s: sma_cn(s, window, 1))
    absolute_sma = _group_transform(group_index, absolute_part, lambda s: sma_cn(s, window, 1))
    return safe_divide(positive_sma, absolute_sma) * 100.0


def _stochastic_rsv(df: pd.DataFrame, window: int) -> pd.Series:
    """计算 RSV 类随机指标核心值。"""

    low_n = _group_transform(df["instrument_id"], df["low"], lambda s: rolling_min(s, window))
    high_n = _group_transform(df["instrument_id"], df["high"], lambda s: rolling_max(s, window))
    return safe_divide(df["close"] - low_n, high_n - low_n) * 100.0


def _signed_volume_sum(df: pd.DataFrame, window: int) -> pd.Series:
    """计算涨跌方向签名量的滚动求和。"""

    signed_volume = _signed_volume(df["close"]) * df["volume"]
    return _group_transform(df["instrument_id"], signed_volume, lambda s: rolling_sum(s, window))


def _true_range(df: pd.DataFrame) -> pd.Series:
    """计算经典 True Range。"""

    return pd.Series(
        np.maximum(
            np.maximum(df["high"] - df["low"], np.abs(df["high"] - df["prev_close"])),
            np.abs(df["low"] - df["prev_close"]),
        ),
        index=df.index,
    )


def _dtm_dbm(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """计算 DTM / DBM。

    这是一些国泰君安 Alpha191 公式里常用的中间量，定义采用常见技术分析版本：

    - `DTM = open <= prev_open ? 0 : max(high - open, open - prev_open)`
    - `DBM = open >= prev_open ? 0 : max(open - low, open - prev_open)`
    """

    dtm = _where(
        df["open"] <= df["prev_open"],
        0.0,
        pd.Series(np.maximum(df["high"] - df["open"], df["open"] - df["prev_open"]), index=df.index),
        df.index,
    )
    dbm = _where(
        df["open"] >= df["prev_open"],
        0.0,
        pd.Series(np.maximum(df["open"] - df["low"], df["open"] - df["prev_open"]), index=df.index),
        df.index,
    )
    return dtm, dbm


def _directional_movement(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """计算方向运动指标里的 HD / LD。"""

    hd = df["high"] - df["prev_high"]
    ld = df["prev_low"] - df["low"]
    return hd, ld


def _alpha001(df: pd.DataFrame) -> pd.Series:
    """Alpha001。"""

    delta_log_volume = _group_transform(df["instrument_id"], np.log(df["volume"].clip(lower=1.0)), lambda s: delta(s, 1))
    intraday_return = safe_divide(df["close"] - df["open"], df["open"])
    rank_volume = _cs_rank(df["date"], delta_log_volume)
    rank_intraday = _cs_rank(df["date"], intraday_return)
    return -_group_corr(df["instrument_id"], rank_volume, rank_intraday, window=6)


def _alpha002(df: pd.DataFrame) -> pd.Series:
    """Alpha002。"""

    core = safe_divide((df["close"] - df["low"]) - (df["high"] - df["close"]), df["high"] - df["low"])
    return -_group_transform(df["instrument_id"], core, lambda s: delta(s, 1))


def _alpha003(df: pd.DataFrame) -> pd.Series:
    """Alpha003。"""

    previous_close = df["prev_close"]
    min_low = np.minimum(df["low"], previous_close)
    max_high = np.maximum(df["high"], previous_close)
    adjusted_move = pd.Series(
        np.where(
            df["close"] == previous_close,
            0.0,
            df["close"] - np.where(df["close"] > previous_close, min_low, max_high),
        ),
        index=df.index,
    )
    return _group_transform(df["instrument_id"], adjusted_move, lambda s: rolling_sum(s, 6))


def _alpha004(df: pd.DataFrame) -> pd.Series:
    """Alpha004。"""

    mean_close_8 = _group_transform(df["instrument_id"], df["close"], lambda s: rolling_mean(s, 8))
    std_close_8 = _group_transform(df["instrument_id"], df["close"], lambda s: rolling_std(s, 8))
    mean_close_2 = _group_transform(df["instrument_id"], df["close"], lambda s: rolling_mean(s, 2))
    volume_ratio = safe_divide(
        df["volume"],
        _group_transform(df["instrument_id"], df["volume"], lambda s: rolling_mean(s, 20)),
    )

    return pd.Series(
        np.where(
            (mean_close_8 + std_close_8) < mean_close_2,
            -1.0,
            np.where(
                mean_close_2 < (mean_close_8 - std_close_8),
                1.0,
                np.where(volume_ratio >= 1.0, 1.0, -1.0),
            ),
        ),
        index=df.index,
    )


def _alpha005(df: pd.DataFrame) -> pd.Series:
    """Alpha005。"""

    tsrank_volume = _group_transform(df["instrument_id"], df["volume"], lambda s: ts_rank(s, 5))
    tsrank_high = _group_transform(df["instrument_id"], df["high"], lambda s: ts_rank(s, 5))
    corr_value = _group_corr(df["instrument_id"], tsrank_volume, tsrank_high, window=5)
    return -_group_transform(df["instrument_id"], corr_value, lambda s: rolling_max(s, 3))


def _alpha006(df: pd.DataFrame) -> pd.Series:
    """Alpha006。"""

    blended_open_high = df["open"] * 0.85 + df["high"] * 0.15
    delta_value = _group_transform(df["instrument_id"], blended_open_high, lambda s: delta(s, 4))
    return -_cs_rank(df["date"], np.sign(delta_value))


def _alpha007(df: pd.DataFrame) -> pd.Series:
    """Alpha007。"""

    gap = df["vwap"] - df["close"]
    max_gap = _group_transform(df["instrument_id"], gap, lambda s: rolling_max(s, 3))
    min_gap = _group_transform(df["instrument_id"], gap, lambda s: rolling_min(s, 3))
    delta_volume = _group_transform(df["instrument_id"], df["volume"], lambda s: delta(s, 3))
    return (_cs_rank(df["date"], max_gap) + _cs_rank(df["date"], min_gap)) * _cs_rank(df["date"], delta_volume)


def _alpha008(df: pd.DataFrame) -> pd.Series:
    """Alpha008。"""

    blended_price = ((df["high"] + df["low"]) / 2.0) * 0.2 + df["vwap"] * 0.8
    delta_value = _group_transform(df["instrument_id"], blended_price, lambda s: delta(s, 4))
    return _cs_rank(df["date"], -delta_value)


def _alpha009(df: pd.DataFrame) -> pd.Series:
    """Alpha009。"""

    midpoint_move = (df["high"] + df["low"]) / 2.0 - (df["prev_high"] + df["prev_low"]) / 2.0
    core = safe_divide(midpoint_move * (df["high"] - df["low"]), df["volume"])
    return _group_transform(df["instrument_id"], core, lambda s: sma_cn(s, 7, 2))


def _alpha010(df: pd.DataFrame) -> pd.Series:
    """Alpha010。"""

    std_ret_20 = _group_transform(df["instrument_id"], df["ret"], lambda s: rolling_std(s, 20))
    conditional_value = _where(df["ret"] < 0, std_ret_20, df["close"], df.index)
    squared_value = conditional_value.pow(2)
    tsmax_value = _group_transform(df["instrument_id"], squared_value, lambda s: rolling_max(s, 5))
    return _cs_rank(df["date"], tsmax_value)


def _alpha011(df: pd.DataFrame) -> pd.Series:
    """Alpha011。"""

    core = safe_divide((df["close"] - df["low"]) - (df["high"] - df["close"]), df["high"] - df["low"]) * df["volume"]
    return _group_transform(df["instrument_id"], core, lambda s: rolling_sum(s, 6))


def _alpha012(df: pd.DataFrame) -> pd.Series:
    """Alpha012。"""

    open_gap = df["open"] - _group_transform(df["instrument_id"], df["vwap"], lambda s: rolling_mean(s, 10))
    close_vwap_gap = np.abs(df["close"] - df["vwap"])
    return -_cs_rank(df["date"], open_gap) * _cs_rank(df["date"], close_vwap_gap)


def _alpha013(df: pd.DataFrame) -> pd.Series:
    """Alpha013。"""

    return np.sqrt(df["high"] * df["low"]) - df["vwap"]


def _alpha014(df: pd.DataFrame) -> pd.Series:
    """Alpha014。"""

    return _group_transform(df["instrument_id"], df["close"], lambda s: delta(s, 5))


def _alpha015(df: pd.DataFrame) -> pd.Series:
    """Alpha015。"""

    return safe_divide(df["open"], df["prev_close"]) - 1.0


def _alpha016(df: pd.DataFrame) -> pd.Series:
    """Alpha016。"""

    rank_volume = _cs_rank(df["date"], df["volume"])
    rank_vwap = _cs_rank(df["date"], df["vwap"])
    corr_value = _group_corr(df["instrument_id"], rank_volume, rank_vwap, window=5)
    ranked_corr = _cs_rank(df["date"], corr_value)
    return -_group_transform(df["instrument_id"], ranked_corr, lambda s: rolling_max(s, 5))


def _alpha017(df: pd.DataFrame) -> pd.Series:
    """Alpha017。"""

    rank_value = _cs_rank(
        df["date"],
        df["vwap"] - _group_transform(df["instrument_id"], df["vwap"], lambda s: rolling_max(s, 15)),
    ).clip(lower=1e-6)
    delta_close = _group_transform(df["instrument_id"], df["close"], lambda s: delta(s, 5)).fillna(0.0)
    return pd.Series(np.power(rank_value, delta_close), index=df.index)


def _alpha018(df: pd.DataFrame) -> pd.Series:
    """Alpha018。"""

    delayed_close = _group_transform(df["instrument_id"], df["close"], lambda s: delay(s, 5))
    return safe_divide(df["close"], delayed_close)


def _alpha019(df: pd.DataFrame) -> pd.Series:
    """Alpha019。"""

    delayed_close = _group_transform(df["instrument_id"], df["close"], lambda s: delay(s, 5))
    rise_value = safe_divide(df["close"] - delayed_close, df["close"])
    fall_value = safe_divide(df["close"] - delayed_close, delayed_close)
    return pd.Series(
        np.where(
            df["close"] < delayed_close,
            fall_value,
            np.where(df["close"] == delayed_close, 0.0, rise_value),
        ),
        index=df.index,
    )


def _alpha020(df: pd.DataFrame) -> pd.Series:
    """Alpha020。"""

    delayed_close = _group_transform(df["instrument_id"], df["close"], lambda s: delay(s, 6))
    return safe_divide(df["close"] - delayed_close, delayed_close) * 100.0


def _alpha021(df: pd.DataFrame) -> pd.Series:
    """Alpha021。"""

    mean_close_6 = _group_transform(df["instrument_id"], df["close"], lambda s: rolling_mean(s, 6))
    return _group_transform(df["instrument_id"], mean_close_6, lambda s: rolling_regression_beta(s, 6))


def _alpha022(df: pd.DataFrame) -> pd.Series:
    """Alpha022。

    原始文本中写的是 `SMEAN`，这里按国内技术分析常见写法解释为 `SMA` 风格平滑。
    """

    mean_close_6 = _group_transform(df["instrument_id"], df["close"], lambda s: rolling_mean(s, 6))
    ratio = safe_divide(df["close"] - mean_close_6, mean_close_6)
    delta_ratio = ratio - _group_transform(df["instrument_id"], ratio, lambda s: delay(s, 3))
    return _group_transform(df["instrument_id"], delta_ratio, lambda s: sma_cn(s, 12, 1))


def _alpha023(df: pd.DataFrame) -> pd.Series:
    """Alpha023。"""

    std_close_20 = _group_transform(df["instrument_id"], df["close"], lambda s: rolling_std(s, 20)).fillna(0.0)
    up_component = _where(df["close"] > df["prev_close"], std_close_20, 0.0, df.index)
    down_component = _where(df["close"] <= df["prev_close"], std_close_20, 0.0, df.index)
    up_sma = _group_transform(df["instrument_id"], up_component, lambda s: sma_cn(s, 20, 1))
    down_sma = _group_transform(df["instrument_id"], down_component, lambda s: sma_cn(s, 20, 1))
    return safe_divide(up_sma, up_sma + down_sma) * 100.0


def _alpha024(df: pd.DataFrame) -> pd.Series:
    """Alpha024。"""

    close_delta_5 = _group_transform(df["instrument_id"], df["close"], lambda s: delta(s, 5))
    return _group_transform(df["instrument_id"], close_delta_5, lambda s: sma_cn(s, 5, 1))


def _alpha025(df: pd.DataFrame) -> pd.Series:
    """Alpha025。"""

    delta_close_7 = _group_transform(df["instrument_id"], df["close"], lambda s: delta(s, 7))
    volume_ratio = safe_divide(
        df["volume"],
        _group_transform(df["instrument_id"], df["volume"], lambda s: rolling_mean(s, 20)),
    )
    decay_volume = _group_transform(df["instrument_id"], volume_ratio, lambda s: decay_linear(s, 9))
    ret_sum_250 = _group_transform(df["instrument_id"], df["ret"], lambda s: rolling_sum(s, 250))
    return -_cs_rank(df["date"], delta_close_7 * (1.0 - _cs_rank(df["date"], decay_volume))) * (
        1.0 + _cs_rank(df["date"], ret_sum_250)
    )


def _alpha026(df: pd.DataFrame) -> pd.Series:
    """Alpha026。"""

    mean_close_7 = _group_transform(df["instrument_id"], df["close"], lambda s: rolling_mean(s, 7))
    delayed_close_5 = _group_transform(df["instrument_id"], df["close"], lambda s: delay(s, 5))
    corr_part = _group_corr(df["instrument_id"], df["vwap"], delayed_close_5, window=230)
    return (mean_close_7 - df["close"]) + corr_part


def _alpha027(df: pd.DataFrame) -> pd.Series:
    """Alpha027。"""

    momentum_3 = safe_divide(
        _group_transform(df["instrument_id"], df["close"], lambda s: delta(s, 3)),
        _group_transform(df["instrument_id"], df["close"], lambda s: delay(s, 3)),
    ) * 100.0
    momentum_6 = safe_divide(
        _group_transform(df["instrument_id"], df["close"], lambda s: delta(s, 6)),
        _group_transform(df["instrument_id"], df["close"], lambda s: delay(s, 6)),
    ) * 100.0
    return _group_transform(df["instrument_id"], momentum_3 + momentum_6, lambda s: wma(s, 12))


def _alpha028(df: pd.DataFrame) -> pd.Series:
    """Alpha028。

    用户给出的原始公式中第二层分母存在明显排版问题，这里采用 KDJ / RSV 类公式中最常见的形式：
    `TSMAX(HIGH, 9) - TSMIN(LOW, 9)`。
    """

    low_9 = _group_transform(df["instrument_id"], df["low"], lambda s: rolling_min(s, 9))
    high_9 = _group_transform(df["instrument_id"], df["high"], lambda s: rolling_max(s, 9))
    rsv = safe_divide(df["close"] - low_9, high_9 - low_9) * 100.0
    first_sma = _group_transform(df["instrument_id"], rsv, lambda s: sma_cn(s, 3, 1))
    second_sma = _group_transform(df["instrument_id"], first_sma, lambda s: sma_cn(s, 3, 1))
    return 3.0 * first_sma - 2.0 * second_sma


def _alpha029(df: pd.DataFrame) -> pd.Series:
    """Alpha029。"""

    delayed_close_6 = _group_transform(df["instrument_id"], df["close"], lambda s: delay(s, 6))
    return safe_divide(df["close"] - delayed_close_6, delayed_close_6) * df["volume"]


def _alpha031(df: pd.DataFrame) -> pd.Series:
    """Alpha031。"""

    mean_close_12 = _group_transform(df["instrument_id"], df["close"], lambda s: rolling_mean(s, 12))
    return safe_divide(df["close"] - mean_close_12, mean_close_12) * 100.0


def _alpha032(df: pd.DataFrame) -> pd.Series:
    """Alpha032。"""

    rank_high = _cs_rank(df["date"], df["high"])
    rank_volume = _cs_rank(df["date"], df["volume"])
    corr_value = _group_corr(df["instrument_id"], rank_high, rank_volume, window=3)
    ranked_corr = _cs_rank(df["date"], corr_value)
    return -_group_transform(df["instrument_id"], ranked_corr, lambda s: rolling_sum(s, 3))


def _alpha033(df: pd.DataFrame) -> pd.Series:
    """Alpha033。"""

    low_min_5 = _group_transform(df["instrument_id"], df["low"], lambda s: rolling_min(s, 5))
    delayed_low_min_5 = _group_transform(df["instrument_id"], low_min_5, lambda s: delay(s, 5))
    ret_sum_240 = _group_transform(df["instrument_id"], df["ret"], lambda s: rolling_sum(s, 240))
    ret_sum_20 = _group_transform(df["instrument_id"], df["ret"], lambda s: rolling_sum(s, 20))
    return (-low_min_5 + delayed_low_min_5) * _cs_rank(df["date"], safe_divide(ret_sum_240 - ret_sum_20, 220.0)) * (
        _group_transform(df["instrument_id"], df["volume"], lambda s: ts_rank(s, 5))
    )


def _alpha034(df: pd.DataFrame) -> pd.Series:
    """Alpha034。"""

    return safe_divide(_group_transform(df["instrument_id"], df["close"], lambda s: rolling_mean(s, 12)), df["close"])


def _alpha035(df: pd.DataFrame) -> pd.Series:
    """Alpha035。"""

    decay_open = _group_transform(
        df["instrument_id"],
        _group_transform(df["instrument_id"], df["open"], lambda s: delta(s, 1)),
        lambda s: decay_linear(s, 15),
    )
    corr_open = _group_corr(df["instrument_id"], df["volume"], df["open"], window=17)
    decay_corr = _group_transform(df["instrument_id"], corr_open, lambda s: decay_linear(s, 7))
    return -pd.Series(
        np.minimum(_cs_rank(df["date"], decay_open), _cs_rank(df["date"], decay_corr)),
        index=df.index,
    )


def _alpha037(df: pd.DataFrame) -> pd.Series:
    """Alpha037。"""

    open_sum_5 = _group_transform(df["instrument_id"], df["open"], lambda s: rolling_sum(s, 5))
    ret_sum_5 = _group_transform(df["instrument_id"], df["ret"], lambda s: rolling_sum(s, 5))
    combined = open_sum_5 * ret_sum_5
    delayed_combined = _group_transform(df["instrument_id"], combined, lambda s: delay(s, 10))
    return -_cs_rank(df["date"], combined - delayed_combined)


def _alpha038(df: pd.DataFrame) -> pd.Series:
    """Alpha038。"""

    mean_high_20 = _group_transform(df["instrument_id"], df["high"], lambda s: rolling_mean(s, 20))
    high_delta_2 = _group_transform(df["instrument_id"], df["high"], lambda s: delta(s, 2))
    return _where(mean_high_20 < df["high"], -high_delta_2, 0.0, df.index)


def _alpha039(df: pd.DataFrame) -> pd.Series:
    """Alpha039。"""

    decay_close = _group_transform(
        df["instrument_id"],
        _group_transform(df["instrument_id"], df["close"], lambda s: delta(s, 2)),
        lambda s: decay_linear(s, 8),
    )
    mean_volume_180 = _group_transform(df["instrument_id"], df["volume"], lambda s: rolling_mean(s, 180))
    volume_sum_37 = _group_transform(df["instrument_id"], mean_volume_180, lambda s: rolling_sum(s, 37))
    blended_price = df["vwap"] * 0.3 + df["open"] * 0.7
    corr_value = _group_corr(df["instrument_id"], blended_price, volume_sum_37, window=14)
    decay_corr = _group_transform(df["instrument_id"], corr_value, lambda s: decay_linear(s, 12))
    return -(_cs_rank(df["date"], decay_close) - _cs_rank(df["date"], decay_corr))


def _alpha040(df: pd.DataFrame) -> pd.Series:
    """Alpha040。"""

    up_volume = _where(df["close"] > df["prev_close"], df["volume"], 0.0, df.index)
    down_volume = _where(df["close"] <= df["prev_close"], df["volume"], 0.0, df.index)
    up_sum = _group_transform(df["instrument_id"], up_volume, lambda s: rolling_sum(s, 26))
    down_sum = _group_transform(df["instrument_id"], down_volume, lambda s: rolling_sum(s, 26))
    return safe_divide(up_sum, down_sum) * 100.0


def _alpha041(df: pd.DataFrame) -> pd.Series:
    """Alpha041。"""

    delta_vwap_3 = _group_transform(df["instrument_id"], df["vwap"], lambda s: delta(s, 3))
    return -_cs_rank(df["date"], _group_transform(df["instrument_id"], delta_vwap_3, lambda s: rolling_max(s, 5)))


def _alpha042(df: pd.DataFrame) -> pd.Series:
    """Alpha042。"""

    std_high_10 = _group_transform(df["instrument_id"], df["high"], lambda s: rolling_std(s, 10))
    corr_value = _group_corr(df["instrument_id"], df["high"], df["volume"], window=10)
    return -_cs_rank(df["date"], std_high_10) * corr_value


def _alpha043(df: pd.DataFrame) -> pd.Series:
    """Alpha043。"""

    signed_volume = _signed_volume(df["close"]) * df["volume"]
    return _group_transform(df["instrument_id"], signed_volume, lambda s: rolling_sum(s, 6))


def _alpha044(df: pd.DataFrame) -> pd.Series:
    """Alpha044。"""

    mean_volume_10 = _group_transform(df["instrument_id"], df["volume"], lambda s: rolling_mean(s, 10))
    corr_low_volume = _group_corr(df["instrument_id"], df["low"], mean_volume_10, window=7)
    first_part = _group_transform(
        df["instrument_id"],
        _group_transform(df["instrument_id"], corr_low_volume, lambda s: decay_linear(s, 6)),
        lambda s: ts_rank(s, 4),
    )
    second_part = _group_transform(
        df["instrument_id"],
        _group_transform(
            df["instrument_id"],
            _group_transform(df["instrument_id"], df["vwap"], lambda s: delta(s, 3)),
            lambda s: decay_linear(s, 10),
        ),
        lambda s: ts_rank(s, 15),
    )
    return first_part + second_part


def _alpha045(df: pd.DataFrame) -> pd.Series:
    """Alpha045。"""

    blended_price = df["close"] * 0.6 + df["open"] * 0.4
    delta_blended = _group_transform(df["instrument_id"], blended_price, lambda s: delta(s, 1))
    mean_volume_150 = _group_transform(df["instrument_id"], df["volume"], lambda s: rolling_mean(s, 150))
    corr_value = _group_corr(df["instrument_id"], df["vwap"], mean_volume_150, window=15)
    return _cs_rank(df["date"], delta_blended) * _cs_rank(df["date"], corr_value)


def _alpha046(df: pd.DataFrame) -> pd.Series:
    """Alpha046。"""

    return (
        _group_transform(df["instrument_id"], df["close"], lambda s: rolling_mean(s, 3))
        + _group_transform(df["instrument_id"], df["close"], lambda s: rolling_mean(s, 6))
        + _group_transform(df["instrument_id"], df["close"], lambda s: rolling_mean(s, 12))
        + _group_transform(df["instrument_id"], df["close"], lambda s: rolling_mean(s, 24))
    ) / (4.0 * df["close"])


def _alpha047(df: pd.DataFrame) -> pd.Series:
    """Alpha047。"""

    high_6 = _group_transform(df["instrument_id"], df["high"], lambda s: rolling_max(s, 6))
    low_6 = _group_transform(df["instrument_id"], df["low"], lambda s: rolling_min(s, 6))
    ratio = safe_divide(high_6 - df["close"], high_6 - low_6) * 100.0
    return _group_transform(df["instrument_id"], ratio, lambda s: sma_cn(s, 9, 1))


def _alpha048(df: pd.DataFrame) -> pd.Series:
    """Alpha048。"""

    sign_sum = (
        np.sign(df["close"] - df["prev_close"])
        + np.sign(df["prev_close"] - _group_transform(df["instrument_id"], df["close"], lambda s: delay(s, 2)))
        + np.sign(
            _group_transform(df["instrument_id"], df["close"], lambda s: delay(s, 2))
            - _group_transform(df["instrument_id"], df["close"], lambda s: delay(s, 3))
        )
    )
    volume_sum_5 = _group_transform(df["instrument_id"], df["volume"], lambda s: rolling_sum(s, 5))
    volume_sum_20 = _group_transform(df["instrument_id"], df["volume"], lambda s: rolling_sum(s, 20))
    return -safe_divide(_cs_rank(df["date"], sign_sum) * volume_sum_5, volume_sum_20)


def _alpha049(df: pd.DataFrame) -> pd.Series:
    """Alpha049。"""

    reference = np.maximum(np.abs(df["high"] - df["prev_high"]), np.abs(df["low"] - df["prev_low"]))
    down_move = _where((df["high"] + df["low"]) >= (df["prev_high"] + df["prev_low"]), 0.0, reference, df.index)
    up_move = _where((df["high"] + df["low"]) <= (df["prev_high"] + df["prev_low"]), 0.0, reference, df.index)
    down_sum = _group_transform(df["instrument_id"], down_move, lambda s: rolling_sum(s, 12))
    up_sum = _group_transform(df["instrument_id"], up_move, lambda s: rolling_sum(s, 12))
    return safe_divide(down_sum, down_sum + up_sum)


def _alpha050(df: pd.DataFrame) -> pd.Series:
    """Alpha050。"""

    reference = np.maximum(np.abs(df["high"] - df["prev_high"]), np.abs(df["low"] - df["prev_low"]))
    down_move = _where((df["high"] + df["low"]) >= (df["prev_high"] + df["prev_low"]), 0.0, reference, df.index)
    up_move = _where((df["high"] + df["low"]) <= (df["prev_high"] + df["prev_low"]), 0.0, reference, df.index)
    down_sum = _group_transform(df["instrument_id"], down_move, lambda s: rolling_sum(s, 12))
    up_sum = _group_transform(df["instrument_id"], up_move, lambda s: rolling_sum(s, 12))
    return safe_divide(up_sum, up_sum + down_sum) - safe_divide(down_sum, down_sum + up_sum)


def _alpha051(df: pd.DataFrame) -> pd.Series:
    """Alpha051。"""

    reference = np.maximum(np.abs(df["high"] - df["prev_high"]), np.abs(df["low"] - df["prev_low"]))
    up_move = _where((df["high"] + df["low"]) <= (df["prev_high"] + df["prev_low"]), 0.0, reference, df.index)
    down_move = _where((df["high"] + df["low"]) >= (df["prev_high"] + df["prev_low"]), 0.0, reference, df.index)
    up_sum = _group_transform(df["instrument_id"], up_move, lambda s: rolling_sum(s, 12))
    down_sum = _group_transform(df["instrument_id"], down_move, lambda s: rolling_sum(s, 12))
    return safe_divide(up_sum, up_sum + down_sum)


def _alpha052(df: pd.DataFrame) -> pd.Series:
    """Alpha052。"""

    prev_hlc3 = _group_transform(df["instrument_id"], df["hlc3"], lambda s: delay(s, 1))
    numerator = _group_transform(
        df["instrument_id"],
        pd.Series(np.maximum(0.0, df["high"] - prev_hlc3), index=df.index),
        lambda s: rolling_sum(s, 26),
    )
    denominator = _group_transform(
        df["instrument_id"],
        pd.Series(np.maximum(0.0, prev_hlc3 - df["low"]), index=df.index),
        lambda s: rolling_sum(s, 26),
    )
    return safe_divide(numerator, denominator) * 100.0


def _alpha053(df: pd.DataFrame) -> pd.Series:
    """Alpha053。"""

    return _group_transform(
        df["instrument_id"],
        (df["close"] > df["prev_close"]).astype(float),
        lambda s: rolling_sum(s, 12),
    ) / 12.0 * 100.0


def _alpha054(df: pd.DataFrame) -> pd.Series:
    """Alpha054。

    原始文本里 `STD(ABS(CLOSE - OPEN))` 未给出窗口长度；
    这里采用与同式中相关项一致、也最常见的 `10` 日窗口。
    """

    std_open_close_gap = _group_transform(
        df["instrument_id"],
        np.abs(df["close"] - df["open"]),
        lambda s: rolling_std(s, 10),
    )
    corr_value = _group_corr(df["instrument_id"], df["close"], df["open"], window=10)
    return -_cs_rank(df["date"], std_open_close_gap + (df["close"] - df["open"]) + corr_value)


def _alpha057(df: pd.DataFrame) -> pd.Series:
    """Alpha057。"""

    low_9 = _group_transform(df["instrument_id"], df["low"], lambda s: rolling_min(s, 9))
    high_9 = _group_transform(df["instrument_id"], df["high"], lambda s: rolling_max(s, 9))
    rsv = safe_divide(df["close"] - low_9, high_9 - low_9) * 100.0
    return _group_transform(df["instrument_id"], rsv, lambda s: sma_cn(s, 3, 1))


def _alpha058(df: pd.DataFrame) -> pd.Series:
    """Alpha058。"""

    return _group_transform(
        df["instrument_id"],
        (df["close"] > df["prev_close"]).astype(float),
        lambda s: rolling_sum(s, 20),
    ) / 20.0 * 100.0


def _alpha059(df: pd.DataFrame) -> pd.Series:
    """Alpha059。"""

    previous_close = df["prev_close"]
    min_low = np.minimum(df["low"], previous_close)
    max_high = np.maximum(df["high"], previous_close)
    adjusted_move = pd.Series(
        np.where(
            df["close"] == previous_close,
            0.0,
            df["close"] - np.where(df["close"] > previous_close, min_low, max_high),
        ),
        index=df.index,
    )
    return _group_transform(df["instrument_id"], adjusted_move, lambda s: rolling_sum(s, 20))


def _alpha060(df: pd.DataFrame) -> pd.Series:
    """Alpha060。"""

    core = safe_divide((df["close"] - df["low"]) - (df["high"] - df["close"]), df["high"] - df["low"]) * df["volume"]
    return _group_transform(df["instrument_id"], core, lambda s: rolling_sum(s, 20))


def _alpha061(df: pd.DataFrame) -> pd.Series:
    """Alpha061。"""

    decay_delta_vwap = _group_transform(
        df["instrument_id"],
        _group_transform(df["instrument_id"], df["vwap"], lambda s: delta(s, 1)),
        lambda s: decay_linear(s, 12),
    )
    mean_volume_80 = _group_transform(df["instrument_id"], df["volume"], lambda s: rolling_mean(s, 80))
    corr_low_volume = _group_corr(df["instrument_id"], df["low"], mean_volume_80, window=8)
    ranked_corr = _cs_rank(df["date"], corr_low_volume)
    decay_ranked_corr = _group_transform(df["instrument_id"], ranked_corr, lambda s: decay_linear(s, 17))
    return -pd.Series(
        np.maximum(_cs_rank(df["date"], decay_delta_vwap), _cs_rank(df["date"], decay_ranked_corr)),
        index=df.index,
    )


def _alpha062(df: pd.DataFrame) -> pd.Series:
    """Alpha062。"""

    rank_volume = _cs_rank(df["date"], df["volume"])
    return -_group_corr(df["instrument_id"], df["high"], rank_volume, window=5)


def _alpha063(df: pd.DataFrame) -> pd.Series:
    """Alpha063。"""

    return _rsi_style(df["instrument_id"], df["close"], window=6)


def _alpha064(df: pd.DataFrame) -> pd.Series:
    """Alpha064。"""

    rank_vwap = _cs_rank(df["date"], df["vwap"])
    rank_volume = _cs_rank(df["date"], df["volume"])
    corr_rank_vwap_volume = _group_corr(df["instrument_id"], rank_vwap, rank_volume, window=4)
    decay_corr_1 = _group_transform(df["instrument_id"], corr_rank_vwap_volume, lambda s: decay_linear(s, 4))

    rank_close = _cs_rank(df["date"], df["close"])
    mean_volume_60 = _group_transform(df["instrument_id"], df["volume"], lambda s: rolling_mean(s, 60))
    rank_mean_volume_60 = _cs_rank(df["date"], mean_volume_60)
    corr_rank_close_volume = _group_corr(df["instrument_id"], rank_close, rank_mean_volume_60, window=4)
    max_corr_13 = _group_transform(df["instrument_id"], corr_rank_close_volume, lambda s: rolling_max(s, 13))
    decay_corr_2 = _group_transform(df["instrument_id"], max_corr_13, lambda s: decay_linear(s, 14))

    return -pd.Series(
        np.maximum(_cs_rank(df["date"], decay_corr_1), _cs_rank(df["date"], decay_corr_2)),
        index=df.index,
    )


def _alpha065(df: pd.DataFrame) -> pd.Series:
    """Alpha065。"""

    mean_close_6 = _group_transform(df["instrument_id"], df["close"], lambda s: rolling_mean(s, 6))
    return safe_divide(mean_close_6, df["close"])


def _alpha066(df: pd.DataFrame) -> pd.Series:
    """Alpha066。"""

    mean_close_6 = _group_transform(df["instrument_id"], df["close"], lambda s: rolling_mean(s, 6))
    return safe_divide(df["close"] - mean_close_6, mean_close_6) * 100.0


def _alpha067(df: pd.DataFrame) -> pd.Series:
    """Alpha067。"""

    return _rsi_style(df["instrument_id"], df["close"], window=24)


def _alpha068(df: pd.DataFrame) -> pd.Series:
    """Alpha068。"""

    midpoint_move = (df["high"] + df["low"]) / 2.0 - (df["prev_high"] + df["prev_low"]) / 2.0
    core = safe_divide(midpoint_move * (df["high"] - df["low"]), df["volume"])
    return _group_transform(df["instrument_id"], core, lambda s: sma_cn(s, 15, 2))


def _alpha069(df: pd.DataFrame) -> pd.Series:
    """Alpha069。"""

    dtm, dbm = _dtm_dbm(df)
    dtm_sum = _group_transform(df["instrument_id"], dtm, lambda s: rolling_sum(s, 20))
    dbm_sum = _group_transform(df["instrument_id"], dbm, lambda s: rolling_sum(s, 20))
    positive_branch = safe_divide(dtm_sum - dbm_sum, dtm_sum)
    negative_branch = safe_divide(dtm_sum - dbm_sum, dbm_sum)
    return pd.Series(
        np.where(
            dtm_sum > dbm_sum,
            positive_branch,
            np.where(dtm_sum == dbm_sum, 0.0, negative_branch),
        ),
        index=df.index,
    )


def _alpha070(df: pd.DataFrame) -> pd.Series:
    """Alpha070。"""

    return _group_transform(df["instrument_id"], df["amount"], lambda s: rolling_std(s, 6))


def _alpha071(df: pd.DataFrame) -> pd.Series:
    """Alpha071。"""

    mean_close_24 = _group_transform(df["instrument_id"], df["close"], lambda s: rolling_mean(s, 24))
    return safe_divide(df["close"] - mean_close_24, mean_close_24) * 100.0


def _alpha072(df: pd.DataFrame) -> pd.Series:
    """Alpha072。"""

    high_6 = _group_transform(df["instrument_id"], df["high"], lambda s: rolling_max(s, 6))
    low_6 = _group_transform(df["instrument_id"], df["low"], lambda s: rolling_min(s, 6))
    ratio = safe_divide(high_6 - df["close"], high_6 - low_6) * 100.0
    return _group_transform(df["instrument_id"], ratio, lambda s: sma_cn(s, 15, 1))


def _alpha073(df: pd.DataFrame) -> pd.Series:
    """Alpha073。"""

    corr_close_volume = _group_corr(df["instrument_id"], df["close"], df["volume"], window=10)
    decay_corr_16 = _group_transform(df["instrument_id"], corr_close_volume, lambda s: decay_linear(s, 16))
    decay_corr_4 = _group_transform(df["instrument_id"], decay_corr_16, lambda s: decay_linear(s, 4))
    first_part = _group_transform(df["instrument_id"], decay_corr_4, lambda s: ts_rank(s, 5))

    mean_volume_30 = _group_transform(df["instrument_id"], df["volume"], lambda s: rolling_mean(s, 30))
    corr_vwap_volume = _group_corr(df["instrument_id"], df["vwap"], mean_volume_30, window=4)
    decay_vwap_corr = _group_transform(df["instrument_id"], corr_vwap_volume, lambda s: decay_linear(s, 3))
    second_part = _cs_rank(df["date"], decay_vwap_corr)
    return -(first_part - second_part)


def _alpha074(df: pd.DataFrame) -> pd.Series:
    """Alpha074。"""

    blended_low_vwap = df["low"] * 0.35 + df["vwap"] * 0.65
    sum_price_20 = _group_transform(df["instrument_id"], blended_low_vwap, lambda s: rolling_sum(s, 20))
    mean_volume_40 = _group_transform(df["instrument_id"], df["volume"], lambda s: rolling_mean(s, 40))
    sum_volume_20 = _group_transform(df["instrument_id"], mean_volume_40, lambda s: rolling_sum(s, 20))
    corr_part_1 = _group_corr(df["instrument_id"], sum_price_20, sum_volume_20, window=7)

    rank_vwap = _cs_rank(df["date"], df["vwap"])
    rank_volume = _cs_rank(df["date"], df["volume"])
    corr_part_2 = _group_corr(df["instrument_id"], rank_vwap, rank_volume, window=6)
    return _cs_rank(df["date"], corr_part_1) + _cs_rank(df["date"], corr_part_2)


def _alpha076(df: pd.DataFrame) -> pd.Series:
    """Alpha076。"""

    turnover_return = safe_divide(np.abs(df["ret"]), df["volume"])
    std_part = _group_transform(df["instrument_id"], turnover_return, lambda s: rolling_std(s, 20))
    mean_part = _group_transform(df["instrument_id"], turnover_return, lambda s: rolling_mean(s, 20))
    return safe_divide(std_part, mean_part)


def _alpha077(df: pd.DataFrame) -> pd.Series:
    """Alpha077。"""

    first_signal = ((df["high"] + df["low"]) / 2.0) - df["vwap"]
    decay_first = _group_transform(df["instrument_id"], first_signal, lambda s: decay_linear(s, 20))

    mid_price = (df["high"] + df["low"]) / 2.0
    mean_volume_40 = _group_transform(df["instrument_id"], df["volume"], lambda s: rolling_mean(s, 40))
    corr_second = _group_corr(df["instrument_id"], mid_price, mean_volume_40, window=3)
    decay_second = _group_transform(df["instrument_id"], corr_second, lambda s: decay_linear(s, 6))
    return pd.Series(
        np.minimum(_cs_rank(df["date"], decay_first), _cs_rank(df["date"], decay_second)),
        index=df.index,
    )


def _alpha078(df: pd.DataFrame) -> pd.Series:
    """Alpha078。"""

    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    ma_tp_12 = _group_transform(df["instrument_id"], typical_price, lambda s: rolling_mean(s, 12))
    mad_part = _group_transform(
        df["instrument_id"],
        np.abs(df["close"] - ma_tp_12),
        lambda s: rolling_mean(s, 12),
    )
    return safe_divide(typical_price - ma_tp_12, 0.015 * mad_part)


def _alpha079(df: pd.DataFrame) -> pd.Series:
    """Alpha079。"""

    return _rsi_style(df["instrument_id"], df["close"], window=12)


def _alpha080(df: pd.DataFrame) -> pd.Series:
    """Alpha080。"""

    delayed_volume_5 = _group_transform(df["instrument_id"], df["volume"], lambda s: delay(s, 5))
    return safe_divide(df["volume"] - delayed_volume_5, delayed_volume_5) * 100.0


def _alpha081(df: pd.DataFrame) -> pd.Series:
    """Alpha081。"""

    return _group_transform(df["instrument_id"], df["volume"], lambda s: sma_cn(s, 21, 2))


def _alpha082(df: pd.DataFrame) -> pd.Series:
    """Alpha082。"""

    high_6 = _group_transform(df["instrument_id"], df["high"], lambda s: rolling_max(s, 6))
    low_6 = _group_transform(df["instrument_id"], df["low"], lambda s: rolling_min(s, 6))
    ratio = safe_divide(high_6 - df["close"], high_6 - low_6) * 100.0
    return _group_transform(df["instrument_id"], ratio, lambda s: sma_cn(s, 20, 1))


def _alpha083(df: pd.DataFrame) -> pd.Series:
    """Alpha083。"""

    rank_high = _cs_rank(df["date"], df["high"])
    rank_volume = _cs_rank(df["date"], df["volume"])
    covariance = _group_cov(df["instrument_id"], rank_high, rank_volume, window=5)
    return -_cs_rank(df["date"], covariance)


def _alpha084(df: pd.DataFrame) -> pd.Series:
    """Alpha084。"""

    return _signed_volume_sum(df, 20)


def _alpha085(df: pd.DataFrame) -> pd.Series:
    """Alpha085。"""

    volume_ratio = safe_divide(
        df["volume"],
        _group_transform(df["instrument_id"], df["volume"], lambda s: rolling_mean(s, 20)),
    )
    close_delta_7 = _group_transform(df["instrument_id"], df["close"], lambda s: delta(s, 7))
    return _group_transform(df["instrument_id"], volume_ratio, lambda s: ts_rank(s, 20)) * _group_transform(
        df["instrument_id"],
        -close_delta_7,
        lambda s: ts_rank(s, 8),
    )


def _alpha086(df: pd.DataFrame) -> pd.Series:
    """Alpha086。"""

    trend_difference = (
        safe_divide(
            _group_transform(df["instrument_id"], df["close"], lambda s: delay(s, 20))
            - _group_transform(df["instrument_id"], df["close"], lambda s: delay(s, 10)),
            10.0,
        )
        - safe_divide(_group_transform(df["instrument_id"], df["close"], lambda s: delay(s, 10)) - df["close"], 10.0)
    )
    close_delta_1 = _group_transform(df["instrument_id"], df["close"], lambda s: delta(s, 1))
    return pd.Series(
        np.where(
            trend_difference > 0.25,
            -1.0,
            np.where(trend_difference < 0.0, 1.0, -close_delta_1),
        ),
        index=df.index,
    )


def _alpha087(df: pd.DataFrame) -> pd.Series:
    """Alpha087。"""

    decay_vwap = _group_transform(
        df["instrument_id"],
        _group_transform(df["instrument_id"], df["vwap"], lambda s: delta(s, 4)),
        lambda s: decay_linear(s, 7),
    )
    low_center_gap = safe_divide(df["low"] - df["vwap"], df["open"] - (df["high"] + df["low"]) / 2.0)
    decay_gap = _group_transform(df["instrument_id"], low_center_gap, lambda s: decay_linear(s, 11))
    tsrank_gap = _group_transform(df["instrument_id"], decay_gap, lambda s: ts_rank(s, 7))
    return -(_cs_rank(df["date"], decay_vwap) + tsrank_gap)


def _alpha088(df: pd.DataFrame) -> pd.Series:
    """Alpha088。"""

    delayed_close_20 = _group_transform(df["instrument_id"], df["close"], lambda s: delay(s, 20))
    return safe_divide(df["close"] - delayed_close_20, delayed_close_20) * 100.0


def _alpha089(df: pd.DataFrame) -> pd.Series:
    """Alpha089。"""

    sma_close_13 = _group_transform(df["instrument_id"], df["close"], lambda s: sma_cn(s, 13, 2))
    sma_close_27 = _group_transform(df["instrument_id"], df["close"], lambda s: sma_cn(s, 27, 2))
    diff_part = sma_close_13 - sma_close_27
    return 2.0 * (diff_part - _group_transform(df["instrument_id"], diff_part, lambda s: sma_cn(s, 10, 2)))


def _alpha090(df: pd.DataFrame) -> pd.Series:
    """Alpha090。"""

    rank_vwap = _cs_rank(df["date"], df["vwap"])
    rank_volume = _cs_rank(df["date"], df["volume"])
    return -_cs_rank(df["date"], _group_corr(df["instrument_id"], rank_vwap, rank_volume, window=5))


def _alpha091(df: pd.DataFrame) -> pd.Series:
    """Alpha091。"""

    close_gap = df["close"] - _group_transform(df["instrument_id"], df["close"], lambda s: rolling_max(s, 5))
    mean_volume_40 = _group_transform(df["instrument_id"], df["volume"], lambda s: rolling_mean(s, 40))
    corr_value = _group_corr(df["instrument_id"], mean_volume_40, df["low"], window=5)
    return -(_cs_rank(df["date"], close_gap) * _cs_rank(df["date"], corr_value))


def _alpha092(df: pd.DataFrame) -> pd.Series:
    """Alpha092。"""

    blended_price = df["close"] * 0.35 + df["vwap"] * 0.65
    decay_delta_price = _group_transform(
        df["instrument_id"],
        _group_transform(df["instrument_id"], blended_price, lambda s: delta(s, 2)),
        lambda s: decay_linear(s, 3),
    )
    mean_volume_180 = _group_transform(df["instrument_id"], df["volume"], lambda s: rolling_mean(s, 180))
    corr_value = _group_corr(df["instrument_id"], mean_volume_180, df["close"], window=13)
    abs_decay_corr = _group_transform(
        df["instrument_id"],
        np.abs(corr_value),
        lambda s: decay_linear(s, 5),
    )
    tsrank_corr = _group_transform(df["instrument_id"], abs_decay_corr, lambda s: ts_rank(s, 15))
    return -pd.Series(
        np.maximum(_cs_rank(df["date"], decay_delta_price), tsrank_corr),
        index=df.index,
    )


def _alpha093(df: pd.DataFrame) -> pd.Series:
    """Alpha093。"""

    open_down_move = _where(
        df["open"] >= df["prev_open"],
        0.0,
        pd.Series(np.maximum(df["open"] - df["low"], df["open"] - df["prev_open"]), index=df.index),
        df.index,
    )
    return _group_transform(df["instrument_id"], open_down_move, lambda s: rolling_sum(s, 20))


def _alpha094(df: pd.DataFrame) -> pd.Series:
    """Alpha094。"""

    return _signed_volume_sum(df, 30)


def _alpha095(df: pd.DataFrame) -> pd.Series:
    """Alpha095。"""

    return _group_transform(df["instrument_id"], df["amount"], lambda s: rolling_std(s, 20))


def _alpha096(df: pd.DataFrame) -> pd.Series:
    """Alpha096。"""

    first_sma = _group_transform(df["instrument_id"], _stochastic_rsv(df, 9), lambda s: sma_cn(s, 3, 1))
    return _group_transform(df["instrument_id"], first_sma, lambda s: sma_cn(s, 3, 1))


def _alpha097(df: pd.DataFrame) -> pd.Series:
    """Alpha097。"""

    return _group_transform(df["instrument_id"], df["volume"], lambda s: rolling_std(s, 10))


def _alpha098(df: pd.DataFrame) -> pd.Series:
    """Alpha098。"""

    mean_close_100 = _group_transform(df["instrument_id"], df["close"], lambda s: rolling_mean(s, 100))
    delta_mean_100 = _group_transform(df["instrument_id"], mean_close_100, lambda s: delta(s, 100))
    delayed_close_100 = _group_transform(df["instrument_id"], df["close"], lambda s: delay(s, 100))
    signal = safe_divide(delta_mean_100, delayed_close_100)
    min_close_100 = _group_transform(df["instrument_id"], df["close"], lambda s: rolling_min(s, 100))
    delta_close_3 = _group_transform(df["instrument_id"], df["close"], lambda s: delta(s, 3))
    return pd.Series(
        np.where(signal <= 0.05, -(df["close"] - min_close_100), -delta_close_3),
        index=df.index,
    )


def _alpha099(df: pd.DataFrame) -> pd.Series:
    """Alpha099。"""

    rank_close = _cs_rank(df["date"], df["close"])
    rank_volume = _cs_rank(df["date"], df["volume"])
    covariance = _group_cov(df["instrument_id"], rank_close, rank_volume, window=5)
    return -_cs_rank(df["date"], covariance)


def _alpha100(df: pd.DataFrame) -> pd.Series:
    """Alpha100。"""

    return _group_transform(df["instrument_id"], df["volume"], lambda s: rolling_std(s, 20))


def _alpha101(df: pd.DataFrame) -> pd.Series:
    """Alpha101。"""

    mean_volume_30 = _group_transform(df["instrument_id"], df["volume"], lambda s: rolling_mean(s, 30))
    sum_mean_volume_37 = _group_transform(df["instrument_id"], mean_volume_30, lambda s: rolling_sum(s, 37))
    corr_close_volume = _group_corr(df["instrument_id"], df["close"], sum_mean_volume_37, window=15)

    blended_price = df["high"] * 0.1 + df["vwap"] * 0.9
    rank_blended_price = _cs_rank(df["date"], blended_price)
    rank_volume = _cs_rank(df["date"], df["volume"])
    corr_rank_part = _group_corr(df["instrument_id"], rank_blended_price, rank_volume, window=11)
    return pd.Series(
        np.where(
            _cs_rank(df["date"], corr_close_volume) < _cs_rank(df["date"], corr_rank_part),
            -1.0,
            0.0,
        ),
        index=df.index,
    )


def _alpha102(df: pd.DataFrame) -> pd.Series:
    """Alpha102。"""

    return _rsi_style(df["instrument_id"], df["volume"], window=6)


def _alpha103(df: pd.DataFrame) -> pd.Series:
    """Alpha103。"""

    low_day = _group_transform(df["instrument_id"], df["low"], lambda s: _days_since_argmin(s, 20))
    return safe_divide(20.0 - low_day, 20.0) * 100.0


def _alpha104(df: pd.DataFrame) -> pd.Series:
    """Alpha104。"""

    corr_high_volume = _group_corr(df["instrument_id"], df["high"], df["volume"], window=5)
    delta_corr_5 = _group_transform(df["instrument_id"], corr_high_volume, lambda s: delta(s, 5))
    std_close_20 = _group_transform(df["instrument_id"], df["close"], lambda s: rolling_std(s, 20))
    return -(delta_corr_5 * _cs_rank(df["date"], std_close_20))


def _alpha105(df: pd.DataFrame) -> pd.Series:
    """Alpha105。"""

    rank_open = _cs_rank(df["date"], df["open"])
    rank_volume = _cs_rank(df["date"], df["volume"])
    return -_group_corr(df["instrument_id"], rank_open, rank_volume, window=10)


def _alpha106(df: pd.DataFrame) -> pd.Series:
    """Alpha106。"""

    return _group_transform(df["instrument_id"], df["close"], lambda s: delta(s, 20))


def _alpha107(df: pd.DataFrame) -> pd.Series:
    """Alpha107。"""

    open_prev_high = df["open"] - df["prev_high"]
    open_prev_close = df["open"] - df["prev_close"]
    open_prev_low = df["open"] - df["prev_low"]
    return (-_cs_rank(df["date"], open_prev_high)) * _cs_rank(df["date"], open_prev_close) * _cs_rank(
        df["date"], open_prev_low
    )


def _alpha108(df: pd.DataFrame) -> pd.Series:
    """Alpha108。"""

    high_min_2 = _group_transform(df["instrument_id"], df["high"], lambda s: rolling_min(s, 2))
    rank_base = _cs_rank(df["date"], df["high"] - high_min_2).clip(lower=1e-6)
    mean_volume_120 = _group_transform(df["instrument_id"], df["volume"], lambda s: rolling_mean(s, 120))
    corr_vwap_volume = _group_corr(df["instrument_id"], df["vwap"], mean_volume_120, window=6)
    exponent = _cs_rank(df["date"], corr_vwap_volume)
    return -pd.Series(np.power(rank_base, exponent), index=df.index)


def _alpha109(df: pd.DataFrame) -> pd.Series:
    """Alpha109。"""

    price_range = df["high"] - df["low"]
    sma_range_10 = _group_transform(df["instrument_id"], price_range, lambda s: sma_cn(s, 10, 2))
    return safe_divide(sma_range_10, _group_transform(df["instrument_id"], sma_range_10, lambda s: sma_cn(s, 10, 2)))


def _alpha110(df: pd.DataFrame) -> pd.Series:
    """Alpha110。"""

    up_gap = pd.Series(np.maximum(0.0, df["high"] - df["prev_close"]), index=df.index)
    down_gap = pd.Series(np.maximum(0.0, df["prev_close"] - df["low"]), index=df.index)
    up_sum = _group_transform(df["instrument_id"], up_gap, lambda s: rolling_sum(s, 20))
    down_sum = _group_transform(df["instrument_id"], down_gap, lambda s: rolling_sum(s, 20))
    return safe_divide(up_sum, down_sum) * 100.0


def _alpha111(df: pd.DataFrame) -> pd.Series:
    """Alpha111。"""

    volume_flow = df["volume"] * safe_divide((df["close"] - df["low"]) - (df["high"] - df["close"]), df["high"] - df["low"])
    fast = _group_transform(df["instrument_id"], volume_flow, lambda s: sma_cn(s, 4, 2))
    slow = _group_transform(df["instrument_id"], volume_flow, lambda s: sma_cn(s, 11, 2))
    return slow - fast


def _alpha112(df: pd.DataFrame) -> pd.Series:
    """Alpha112。"""

    close_delta = df["close"] - df["prev_close"]
    up_sum = _group_transform(
        df["instrument_id"],
        _where(close_delta > 0, close_delta, 0.0, df.index),
        lambda s: rolling_sum(s, 12),
    )
    down_sum = _group_transform(
        df["instrument_id"],
        _where(close_delta < 0, np.abs(close_delta), 0.0, df.index),
        lambda s: rolling_sum(s, 12),
    )
    return safe_divide(up_sum - down_sum, up_sum + down_sum) * 100.0


def _alpha113(df: pd.DataFrame) -> pd.Series:
    """Alpha113。"""

    delayed_close_5 = _group_transform(df["instrument_id"], df["close"], lambda s: delay(s, 5))
    delayed_close_mean_20 = _group_transform(df["instrument_id"], delayed_close_5, lambda s: rolling_mean(s, 20))
    corr_close_volume = _group_corr(df["instrument_id"], df["close"], df["volume"], window=2)
    sum_close_5 = _group_transform(df["instrument_id"], df["close"], lambda s: rolling_sum(s, 5))
    sum_close_20 = _group_transform(df["instrument_id"], df["close"], lambda s: rolling_sum(s, 20))
    corr_sum_close = _group_corr(df["instrument_id"], sum_close_5, sum_close_20, window=2)
    return -(_cs_rank(df["date"], delayed_close_mean_20) * corr_close_volume * _cs_rank(df["date"], corr_sum_close))


def _alpha114(df: pd.DataFrame) -> pd.Series:
    """Alpha114。"""

    close_mean_5 = _group_transform(df["instrument_id"], df["close"], lambda s: rolling_mean(s, 5))
    normalized_range = safe_divide(df["high"] - df["low"], close_mean_5)
    delayed_range = _group_transform(df["instrument_id"], normalized_range, lambda s: delay(s, 2))
    numerator = _cs_rank(df["date"], delayed_range) * _cs_rank(df["date"], _cs_rank(df["date"], df["volume"]))
    denominator = safe_divide(normalized_range, df["vwap"] - df["close"])
    return safe_divide(numerator, denominator)


def _alpha115(df: pd.DataFrame) -> pd.Series:
    """Alpha115。"""

    blended_price = df["high"] * 0.9 + df["close"] * 0.1
    mean_volume_30 = _group_transform(df["instrument_id"], df["volume"], lambda s: rolling_mean(s, 30))
    corr_part_1 = _group_corr(df["instrument_id"], blended_price, mean_volume_30, window=10)
    rank_part_1 = _cs_rank(df["date"], corr_part_1).clip(lower=1e-6)

    midpoint = (df["high"] + df["low"]) / 2.0
    tsrank_midpoint = _group_transform(df["instrument_id"], midpoint, lambda s: ts_rank(s, 4))
    tsrank_volume_10 = _group_transform(df["instrument_id"], df["volume"], lambda s: ts_rank(s, 10))
    corr_part_2 = _group_corr(df["instrument_id"], tsrank_midpoint, tsrank_volume_10, window=7)
    exponent = _cs_rank(df["date"], corr_part_2)
    return pd.Series(np.power(rank_part_1, exponent), index=df.index)


def _alpha116(df: pd.DataFrame) -> pd.Series:
    """Alpha116。"""

    return _group_transform(df["instrument_id"], df["close"], lambda s: rolling_regression_beta(s, 20))


def _alpha117(df: pd.DataFrame) -> pd.Series:
    """Alpha117。"""

    part_1 = _group_transform(df["instrument_id"], df["volume"], lambda s: ts_rank(s, 32))
    part_2 = 1.0 - _group_transform(df["instrument_id"], (df["close"] + df["high"]) - df["low"], lambda s: ts_rank(s, 16))
    part_3 = 1.0 - _group_transform(df["instrument_id"], df["ret"], lambda s: ts_rank(s, 32))
    return part_1 * part_2 * part_3


def _alpha118(df: pd.DataFrame) -> pd.Series:
    """Alpha118。"""

    numerator = _group_transform(df["instrument_id"], df["high"] - df["open"], lambda s: rolling_sum(s, 20))
    denominator = _group_transform(df["instrument_id"], df["open"] - df["low"], lambda s: rolling_sum(s, 20))
    return safe_divide(numerator, denominator) * 100.0


def _alpha119(df: pd.DataFrame) -> pd.Series:
    """Alpha119。"""

    mean_volume_5 = _group_transform(df["instrument_id"], df["volume"], lambda s: rolling_mean(s, 5))
    sum_mean_volume_26 = _group_transform(df["instrument_id"], mean_volume_5, lambda s: rolling_sum(s, 26))
    corr_part_1 = _group_corr(df["instrument_id"], df["vwap"], sum_mean_volume_26, window=5)
    rank_part_1 = _cs_rank(df["date"], _group_transform(df["instrument_id"], corr_part_1, lambda s: decay_linear(s, 7)))

    mean_volume_15 = _group_transform(df["instrument_id"], df["volume"], lambda s: rolling_mean(s, 15))
    rank_open = _cs_rank(df["date"], df["open"])
    rank_mean_volume_15 = _cs_rank(df["date"], mean_volume_15)
    corr_part_2 = _group_corr(df["instrument_id"], rank_open, rank_mean_volume_15, window=21)
    min_corr_9 = _group_transform(df["instrument_id"], corr_part_2, lambda s: rolling_min(s, 9))
    tsrank_min_corr = _group_transform(df["instrument_id"], min_corr_9, lambda s: ts_rank(s, 7))
    rank_part_2 = _cs_rank(df["date"], _group_transform(df["instrument_id"], tsrank_min_corr, lambda s: decay_linear(s, 8)))
    return rank_part_1 - rank_part_2


def _alpha120(df: pd.DataFrame) -> pd.Series:
    """Alpha120。"""

    return safe_divide(_cs_rank(df["date"], df["vwap"] - df["close"]), _cs_rank(df["date"], df["vwap"] + df["close"]))


def _alpha121(df: pd.DataFrame) -> pd.Series:
    """Alpha121。"""

    vwap_gap = df["vwap"] - _group_transform(df["instrument_id"], df["vwap"], lambda s: rolling_min(s, 12))
    rank_base = _cs_rank(df["date"], vwap_gap).clip(lower=1e-6)
    tsrank_vwap_20 = _group_transform(df["instrument_id"], df["vwap"], lambda s: ts_rank(s, 20))
    mean_volume_60 = _group_transform(df["instrument_id"], df["volume"], lambda s: rolling_mean(s, 60))
    tsrank_volume_2 = _group_transform(df["instrument_id"], mean_volume_60, lambda s: ts_rank(s, 2))
    corr_value = _group_corr(df["instrument_id"], tsrank_vwap_20, tsrank_volume_2, window=18)
    exponent = _group_transform(df["instrument_id"], corr_value, lambda s: ts_rank(s, 3))
    return -pd.Series(np.power(rank_base, exponent), index=df.index)


def _alpha122(df: pd.DataFrame) -> pd.Series:
    """Alpha122。"""

    log_close = np.log(df["close"].clip(lower=1e-8))
    sma_1 = _group_transform(df["instrument_id"], log_close, lambda s: sma_cn(s, 13, 2))
    sma_2 = _group_transform(df["instrument_id"], sma_1, lambda s: sma_cn(s, 13, 2))
    sma_3 = _group_transform(df["instrument_id"], sma_2, lambda s: sma_cn(s, 13, 2))
    delayed_sma = _group_transform(df["instrument_id"], sma_3, lambda s: delay(s, 1))
    return safe_divide(sma_3 - delayed_sma, delayed_sma)


def _alpha123(df: pd.DataFrame) -> pd.Series:
    """Alpha123。"""

    midpoint_sum_20 = _group_transform(df["instrument_id"], (df["high"] + df["low"]) / 2.0, lambda s: rolling_sum(s, 20))
    mean_volume_60 = _group_transform(df["instrument_id"], df["volume"], lambda s: rolling_mean(s, 60))
    volume_sum_20 = _group_transform(df["instrument_id"], mean_volume_60, lambda s: rolling_sum(s, 20))
    corr_part_1 = _group_corr(df["instrument_id"], midpoint_sum_20, volume_sum_20, window=9)
    corr_part_2 = _group_corr(df["instrument_id"], df["low"], df["volume"], window=6)
    return pd.Series(
        np.where(_cs_rank(df["date"], corr_part_1) < _cs_rank(df["date"], corr_part_2), -1.0, 0.0),
        index=df.index,
    )


def _alpha124(df: pd.DataFrame) -> pd.Series:
    """Alpha124。"""

    close_max_30 = _group_transform(df["instrument_id"], df["close"], lambda s: rolling_max(s, 30))
    rank_close_max_30 = _cs_rank(df["date"], close_max_30)
    decay_rank = _group_transform(df["instrument_id"], rank_close_max_30, lambda s: decay_linear(s, 2))
    return safe_divide(df["close"] - df["vwap"], decay_rank)


def _alpha125(df: pd.DataFrame) -> pd.Series:
    """Alpha125。"""

    mean_volume_80 = _group_transform(df["instrument_id"], df["volume"], lambda s: rolling_mean(s, 80))
    corr_vwap_volume = _group_corr(df["instrument_id"], df["vwap"], mean_volume_80, window=17)
    decay_corr = _group_transform(df["instrument_id"], corr_vwap_volume, lambda s: decay_linear(s, 20))

    blended_price = (df["close"] + df["vwap"]) * 0.5
    delta_blended = _group_transform(df["instrument_id"], blended_price, lambda s: delta(s, 3))
    decay_delta = _group_transform(df["instrument_id"], delta_blended, lambda s: decay_linear(s, 16))
    return safe_divide(_cs_rank(df["date"], decay_corr), _cs_rank(df["date"], decay_delta))


def _alpha126(df: pd.DataFrame) -> pd.Series:
    """Alpha126。"""

    return (df["close"] + df["high"] + df["low"]) / 3.0


def _alpha127(df: pd.DataFrame) -> pd.Series:
    """Alpha127。

    原始文本中 `MEAN(...)` 未写窗口，这里按常见实现解释为 12 日滚动均值。
    """

    close_max_12 = _group_transform(df["instrument_id"], df["close"], lambda s: rolling_max(s, 12))
    normalized = 100.0 * safe_divide(df["close"] - close_max_12, close_max_12)
    mean_square = _group_transform(df["instrument_id"], normalized.pow(2), lambda s: rolling_mean(s, 12))
    return np.sqrt(mean_square)


def _alpha128(df: pd.DataFrame) -> pd.Series:
    """Alpha128。"""

    prev_hlc3 = _group_transform(df["instrument_id"], df["hlc3"], lambda s: delay(s, 1))
    positive_flow = _where(df["hlc3"] > prev_hlc3, df["hlc3"] * df["volume"], 0.0, df.index)
    negative_flow = _where(df["hlc3"] < prev_hlc3, df["hlc3"] * df["volume"], 0.0, df.index)
    pos_sum = _group_transform(df["instrument_id"], positive_flow, lambda s: rolling_sum(s, 14))
    neg_sum = _group_transform(df["instrument_id"], negative_flow, lambda s: rolling_sum(s, 14))
    money_ratio = safe_divide(pos_sum, neg_sum)
    return 100.0 - safe_divide(100.0, 1.0 + money_ratio)


def _alpha129(df: pd.DataFrame) -> pd.Series:
    """Alpha129。"""

    negative_change = _where(df["close"] - df["prev_close"] < 0, np.abs(df["close"] - df["prev_close"]), 0.0, df.index)
    return _group_transform(df["instrument_id"], negative_change, lambda s: rolling_sum(s, 12))


def _alpha130(df: pd.DataFrame) -> pd.Series:
    """Alpha130。"""

    mid_price = (df["high"] + df["low"]) / 2.0
    mean_volume_40 = _group_transform(df["instrument_id"], df["volume"], lambda s: rolling_mean(s, 40))
    corr_part_1 = _group_corr(df["instrument_id"], mid_price, mean_volume_40, window=9)
    decay_part_1 = _group_transform(df["instrument_id"], corr_part_1, lambda s: decay_linear(s, 10))

    rank_vwap = _cs_rank(df["date"], df["vwap"])
    rank_volume = _cs_rank(df["date"], df["volume"])
    corr_part_2 = _group_corr(df["instrument_id"], rank_vwap, rank_volume, window=7)
    decay_part_2 = _group_transform(df["instrument_id"], corr_part_2, lambda s: decay_linear(s, 3))
    return safe_divide(_cs_rank(df["date"], decay_part_1), _cs_rank(df["date"], decay_part_2))


def _alpha131(df: pd.DataFrame) -> pd.Series:
    """Alpha131。"""

    rank_delta_vwap = _cs_rank(df["date"], _group_transform(df["instrument_id"], df["vwap"], lambda s: delta(s, 1))).clip(
        lower=1e-6
    )
    mean_volume_50 = _group_transform(df["instrument_id"], df["volume"], lambda s: rolling_mean(s, 50))
    corr_value = _group_corr(df["instrument_id"], df["close"], mean_volume_50, window=18)
    exponent = _group_transform(df["instrument_id"], corr_value, lambda s: ts_rank(s, 18))
    return pd.Series(np.power(rank_delta_vwap, exponent), index=df.index)


def _alpha132(df: pd.DataFrame) -> pd.Series:
    """Alpha132。"""

    return _group_transform(df["instrument_id"], df["amount"], lambda s: rolling_mean(s, 20))


def _alpha133(df: pd.DataFrame) -> pd.Series:
    """Alpha133。"""

    high_day = _group_transform(df["instrument_id"], df["high"], lambda s: _days_since_argmax(s, 20))
    low_day = _group_transform(df["instrument_id"], df["low"], lambda s: _days_since_argmin(s, 20))
    return safe_divide(20.0 - high_day, 20.0) * 100.0 - safe_divide(20.0 - low_day, 20.0) * 100.0


def _alpha134(df: pd.DataFrame) -> pd.Series:
    """Alpha134。"""

    delayed_close_12 = _group_transform(df["instrument_id"], df["close"], lambda s: delay(s, 12))
    return safe_divide(df["close"] - delayed_close_12, delayed_close_12) * df["volume"]


def _alpha135(df: pd.DataFrame) -> pd.Series:
    """Alpha135。"""

    close_ratio_20 = safe_divide(df["close"], _group_transform(df["instrument_id"], df["close"], lambda s: delay(s, 20)))
    delayed_ratio = _group_transform(df["instrument_id"], close_ratio_20, lambda s: delay(s, 1))
    return _group_transform(df["instrument_id"], delayed_ratio, lambda s: sma_cn(s, 20, 1))


def _alpha136(df: pd.DataFrame) -> pd.Series:
    """Alpha136。"""

    delta_ret_3 = _group_transform(df["instrument_id"], df["ret"], lambda s: delta(s, 3))
    corr_open_volume = _group_corr(df["instrument_id"], df["open"], df["volume"], window=10)
    return -_cs_rank(df["date"], delta_ret_3) * corr_open_volume


def _alpha138(df: pd.DataFrame) -> pd.Series:
    """Alpha138。"""

    blended_low_vwap = df["low"] * 0.7 + df["vwap"] * 0.3
    decay_delta = _group_transform(
        df["instrument_id"],
        _group_transform(df["instrument_id"], blended_low_vwap, lambda s: delta(s, 3)),
        lambda s: decay_linear(s, 20),
    )

    tsrank_low_8 = _group_transform(df["instrument_id"], df["low"], lambda s: ts_rank(s, 8))
    mean_volume_60 = _group_transform(df["instrument_id"], df["volume"], lambda s: rolling_mean(s, 60))
    tsrank_volume_17 = _group_transform(df["instrument_id"], mean_volume_60, lambda s: ts_rank(s, 17))
    corr_inner = _group_corr(df["instrument_id"], tsrank_low_8, tsrank_volume_17, window=5)
    tsrank_corr_19 = _group_transform(df["instrument_id"], corr_inner, lambda s: ts_rank(s, 19))
    decay_tsrank = _group_transform(df["instrument_id"], tsrank_corr_19, lambda s: decay_linear(s, 16))
    tsrank_decay = _group_transform(df["instrument_id"], decay_tsrank, lambda s: ts_rank(s, 7))
    return -(_cs_rank(df["date"], decay_delta) - tsrank_decay)


def _alpha139(df: pd.DataFrame) -> pd.Series:
    """Alpha139。"""

    return -_group_corr(df["instrument_id"], df["open"], df["volume"], window=10)


def _alpha140(df: pd.DataFrame) -> pd.Series:
    """Alpha140。"""

    rank_combo = (_cs_rank(df["date"], df["open"]) + _cs_rank(df["date"], df["low"])) - (
        _cs_rank(df["date"], df["high"]) + _cs_rank(df["date"], df["close"])
    )
    decay_rank_combo = _group_transform(df["instrument_id"], rank_combo, lambda s: decay_linear(s, 8))

    tsrank_close_8 = _group_transform(df["instrument_id"], df["close"], lambda s: ts_rank(s, 8))
    mean_volume_60 = _group_transform(df["instrument_id"], df["volume"], lambda s: rolling_mean(s, 60))
    tsrank_volume_20 = _group_transform(df["instrument_id"], mean_volume_60, lambda s: ts_rank(s, 20))
    corr_value = _group_corr(df["instrument_id"], tsrank_close_8, tsrank_volume_20, window=8)
    decay_corr = _group_transform(df["instrument_id"], corr_value, lambda s: decay_linear(s, 7))
    tsrank_corr = _group_transform(df["instrument_id"], decay_corr, lambda s: ts_rank(s, 3))
    return pd.Series(np.minimum(_cs_rank(df["date"], decay_rank_combo), tsrank_corr), index=df.index)


def _alpha141(df: pd.DataFrame) -> pd.Series:
    """Alpha141。"""

    rank_high = _cs_rank(df["date"], df["high"])
    mean_volume_15 = _group_transform(df["instrument_id"], df["volume"], lambda s: rolling_mean(s, 15))
    rank_mean_volume_15 = _cs_rank(df["date"], mean_volume_15)
    return -_cs_rank(df["date"], _group_corr(df["instrument_id"], rank_high, rank_mean_volume_15, window=9))


def _alpha142(df: pd.DataFrame) -> pd.Series:
    """Alpha142。"""

    part_1 = -_cs_rank(df["date"], _group_transform(df["instrument_id"], df["close"], lambda s: ts_rank(s, 10)))
    delta_delta_close = _group_transform(
        df["instrument_id"],
        _group_transform(df["instrument_id"], df["close"], lambda s: delta(s, 1)),
        lambda s: delta(s, 1),
    )
    part_2 = _cs_rank(df["date"], delta_delta_close)
    volume_ratio = safe_divide(
        df["volume"],
        _group_transform(df["instrument_id"], df["volume"], lambda s: rolling_mean(s, 20)),
    )
    part_3 = _cs_rank(df["date"], _group_transform(df["instrument_id"], volume_ratio, lambda s: ts_rank(s, 5)))
    return part_1 * part_2 * part_3


def _alpha144(df: pd.DataFrame) -> pd.Series:
    """Alpha144。"""

    normalized_return = safe_divide(np.abs(df["ret"]), df["amount"])
    negative_mask = df["close"] < df["prev_close"]
    sum_if = _group_transform(
        df["instrument_id"],
        _where(negative_mask, normalized_return, 0.0, df.index),
        lambda s: rolling_sum(s, 20),
    )
    count_if = _group_transform(df["instrument_id"], negative_mask.astype(float), lambda s: rolling_sum(s, 20))
    return safe_divide(sum_if, count_if)


def _alpha145(df: pd.DataFrame) -> pd.Series:
    """Alpha145。"""

    mean_9 = _group_transform(df["instrument_id"], df["volume"], lambda s: rolling_mean(s, 9))
    mean_26 = _group_transform(df["instrument_id"], df["volume"], lambda s: rolling_mean(s, 26))
    mean_12 = _group_transform(df["instrument_id"], df["volume"], lambda s: rolling_mean(s, 12))
    return safe_divide(mean_9 - mean_26, mean_12) * 100.0


def _alpha146(df: pd.DataFrame) -> pd.Series:
    """Alpha146。"""

    ret_series = df["ret"]
    sma_ret_61 = _group_transform(df["instrument_id"], ret_series, lambda s: sma_cn(s, 61, 2))
    diff_part = ret_series - sma_ret_61
    mean_diff_20 = _group_transform(df["instrument_id"], diff_part, lambda s: rolling_mean(s, 20))
    denominator = _group_transform(
        df["instrument_id"],
        (ret_series - diff_part).pow(2),
        lambda s: sma_cn(s, 60, 1),
    )
    return safe_divide(mean_diff_20 * diff_part, denominator)


def _alpha147(df: pd.DataFrame) -> pd.Series:
    """Alpha147。"""

    mean_close_12 = _group_transform(df["instrument_id"], df["close"], lambda s: rolling_mean(s, 12))
    return _group_transform(df["instrument_id"], mean_close_12, lambda s: rolling_regression_beta(s, 12))


def _alpha148(df: pd.DataFrame) -> pd.Series:
    """Alpha148。"""

    mean_volume_60 = _group_transform(df["instrument_id"], df["volume"], lambda s: rolling_mean(s, 60))
    sum_mean_volume_9 = _group_transform(df["instrument_id"], mean_volume_60, lambda s: rolling_sum(s, 9))
    corr_value = _group_corr(df["instrument_id"], df["open"], sum_mean_volume_9, window=6)
    open_min_14 = _group_transform(df["instrument_id"], df["open"], lambda s: rolling_min(s, 14))
    return pd.Series(
        np.where(_cs_rank(df["date"], corr_value) < _cs_rank(df["date"], df["open"] - open_min_14), -1.0, 0.0),
        index=df.index,
    )


def _alpha150(df: pd.DataFrame) -> pd.Series:
    """Alpha150。"""

    return df["hlc3"] * df["volume"]


def _alpha151(df: pd.DataFrame) -> pd.Series:
    """Alpha151。"""

    close_delta_20 = _group_transform(df["instrument_id"], df["close"], lambda s: delta(s, 20))
    return _group_transform(df["instrument_id"], close_delta_20, lambda s: sma_cn(s, 20, 1))


def _alpha152(df: pd.DataFrame) -> pd.Series:
    """Alpha152。"""

    close_ratio_9 = safe_divide(df["close"], _group_transform(df["instrument_id"], df["close"], lambda s: delay(s, 9)))
    delayed_ratio_1 = _group_transform(df["instrument_id"], close_ratio_9, lambda s: delay(s, 1))
    sma_ratio_9 = _group_transform(df["instrument_id"], delayed_ratio_1, lambda s: sma_cn(s, 9, 1))
    delayed_sma_1 = _group_transform(df["instrument_id"], sma_ratio_9, lambda s: delay(s, 1))
    mean_12 = _group_transform(df["instrument_id"], delayed_sma_1, lambda s: rolling_mean(s, 12))
    mean_26 = _group_transform(df["instrument_id"], delayed_sma_1, lambda s: rolling_mean(s, 26))
    return _group_transform(df["instrument_id"], mean_12 - mean_26, lambda s: sma_cn(s, 9, 1))


def _alpha153(df: pd.DataFrame) -> pd.Series:
    """Alpha153。"""

    return (
        _group_transform(df["instrument_id"], df["close"], lambda s: rolling_mean(s, 3))
        + _group_transform(df["instrument_id"], df["close"], lambda s: rolling_mean(s, 6))
        + _group_transform(df["instrument_id"], df["close"], lambda s: rolling_mean(s, 12))
        + _group_transform(df["instrument_id"], df["close"], lambda s: rolling_mean(s, 24))
    ) / 4.0


def _alpha154(df: pd.DataFrame) -> pd.Series:
    """Alpha154。"""

    vwap_min_16 = _group_transform(df["instrument_id"], df["vwap"], lambda s: rolling_min(s, 16))
    mean_volume_180 = _group_transform(df["instrument_id"], df["volume"], lambda s: rolling_mean(s, 180))
    corr_value = _group_corr(df["instrument_id"], df["vwap"], mean_volume_180, window=18)
    return pd.Series(np.where((df["vwap"] - vwap_min_16) < corr_value, 1.0, 0.0), index=df.index)


def _alpha155(df: pd.DataFrame) -> pd.Series:
    """Alpha155。"""

    sma_volume_13 = _group_transform(df["instrument_id"], df["volume"], lambda s: sma_cn(s, 13, 2))
    sma_volume_27 = _group_transform(df["instrument_id"], df["volume"], lambda s: sma_cn(s, 27, 2))
    diff_part = sma_volume_13 - sma_volume_27
    return diff_part - _group_transform(df["instrument_id"], diff_part, lambda s: sma_cn(s, 10, 2))


def _alpha156(df: pd.DataFrame) -> pd.Series:
    """Alpha156。"""

    decay_vwap = _group_transform(
        df["instrument_id"],
        _group_transform(df["instrument_id"], df["vwap"], lambda s: delta(s, 5)),
        lambda s: decay_linear(s, 3),
    )
    blended_open_low = df["open"] * 0.15 + df["low"] * 0.85
    ratio_change = safe_divide(
        _group_transform(df["instrument_id"], blended_open_low, lambda s: delta(s, 2)),
        blended_open_low,
    )
    decay_ratio = _group_transform(df["instrument_id"], -ratio_change, lambda s: decay_linear(s, 3))
    return -pd.Series(
        np.maximum(_cs_rank(df["date"], decay_vwap), _cs_rank(df["date"], decay_ratio)),
        index=df.index,
    )


def _alpha158(df: pd.DataFrame) -> pd.Series:
    """Alpha158。"""

    sma_close_15 = _group_transform(df["instrument_id"], df["close"], lambda s: sma_cn(s, 15, 2))
    return safe_divide((df["high"] - sma_close_15) - (df["low"] - sma_close_15), df["close"])


def _alpha159(df: pd.DataFrame) -> pd.Series:
    """Alpha159。"""

    min_low_close = pd.Series(np.minimum(df["low"], df["prev_close"]), index=df.index)
    max_high_close = pd.Series(np.maximum(df["high"], df["prev_close"]), index=df.index)
    range_base = max_high_close - min_low_close

    def _part(window: int, weight: float) -> pd.Series:
        numerator = df["close"] - _group_transform(df["instrument_id"], min_low_close, lambda s: rolling_sum(s, window))
        denominator = _group_transform(df["instrument_id"], range_base, lambda s: rolling_sum(s, window))
        return safe_divide(numerator, denominator) * weight

    total = _part(6, 12 * 24) + _part(12, 6 * 24) + _part(24, 6 * 24)
    return safe_divide(total * 100.0, (6 * 12) + (6 * 24) + (12 * 24))


def _alpha160(df: pd.DataFrame) -> pd.Series:
    """Alpha160。"""

    std_close_20 = _group_transform(df["instrument_id"], df["close"], lambda s: rolling_std(s, 20))
    negative_std = _where(df["close"] <= df["prev_close"], std_close_20, 0.0, df.index)
    return _group_transform(df["instrument_id"], negative_std, lambda s: sma_cn(s, 20, 1))


def _alpha161(df: pd.DataFrame) -> pd.Series:
    """Alpha161。"""

    true_range = _true_range(df)
    return _group_transform(df["instrument_id"], true_range, lambda s: rolling_mean(s, 12))


def _alpha162(df: pd.DataFrame) -> pd.Series:
    """Alpha162。"""

    rsi_12 = _rsi_style(df["instrument_id"], df["close"], window=12)
    min_rsi_12 = _group_transform(df["instrument_id"], rsi_12, lambda s: rolling_min(s, 12))
    max_rsi_12 = _group_transform(df["instrument_id"], rsi_12, lambda s: rolling_max(s, 12))
    return safe_divide(rsi_12 - min_rsi_12, max_rsi_12 - min_rsi_12)


def _alpha163(df: pd.DataFrame) -> pd.Series:
    """Alpha163。"""

    mean_volume_20 = _group_transform(df["instrument_id"], df["volume"], lambda s: rolling_mean(s, 20))
    return _cs_rank(df["date"], (((-df["ret"]) * mean_volume_20) * df["vwap"]) * (df["high"] - df["close"]))


def _alpha164(df: pd.DataFrame) -> pd.Series:
    """Alpha164。"""

    inverse_move = _where(
        df["close"] > df["prev_close"],
        safe_divide(1.0, df["close"] - df["prev_close"]),
        1.0,
        df.index,
    )
    min_inverse = _group_transform(df["instrument_id"], inverse_move, lambda s: rolling_min(s, 12))
    normalized = safe_divide(inverse_move - min_inverse, df["high"] - df["low"]) * 100.0
    return _group_transform(df["instrument_id"], normalized, lambda s: sma_cn(s, 13, 2))


def _alpha167(df: pd.DataFrame) -> pd.Series:
    """Alpha167。"""

    positive_change = _where(df["close"] - df["prev_close"] > 0, df["close"] - df["prev_close"], 0.0, df.index)
    return _group_transform(df["instrument_id"], positive_change, lambda s: rolling_sum(s, 12))


def _alpha168(df: pd.DataFrame) -> pd.Series:
    """Alpha168。"""

    mean_volume_20 = _group_transform(df["instrument_id"], df["volume"], lambda s: rolling_mean(s, 20))
    return -safe_divide(df["volume"], mean_volume_20)


def _alpha169(df: pd.DataFrame) -> pd.Series:
    """Alpha169。"""

    close_delta = df["close"] - df["prev_close"]
    sma_delta_9 = _group_transform(df["instrument_id"], close_delta, lambda s: sma_cn(s, 9, 1))
    delayed_sma = _group_transform(df["instrument_id"], sma_delta_9, lambda s: delay(s, 1))
    mean_12 = _group_transform(df["instrument_id"], delayed_sma, lambda s: rolling_mean(s, 12))
    mean_26 = _group_transform(df["instrument_id"], delayed_sma, lambda s: rolling_mean(s, 26))
    return _group_transform(df["instrument_id"], mean_12 - mean_26, lambda s: sma_cn(s, 10, 1))


def _alpha170(df: pd.DataFrame) -> pd.Series:
    """Alpha170。"""

    mean_volume_20 = _group_transform(df["instrument_id"], df["volume"], lambda s: rolling_mean(s, 20))
    high_sum_5 = _group_transform(df["instrument_id"], df["high"], lambda s: rolling_sum(s, 5))
    first_term = safe_divide(_cs_rank(df["date"], safe_divide(1.0, df["close"])) * df["volume"], mean_volume_20)
    second_term = safe_divide(df["high"] * _cs_rank(df["date"], df["high"] - df["close"]), high_sum_5 / 5.0)
    third_term = _cs_rank(df["date"], df["vwap"] - _group_transform(df["instrument_id"], df["vwap"], lambda s: delay(s, 5)))
    return first_term * second_term - third_term


def _alpha171(df: pd.DataFrame) -> pd.Series:
    """Alpha171。"""

    numerator = -((df["low"] - df["close"]) * np.power(df["open"], 5))
    denominator = (df["close"] - df["high"]) * np.power(df["close"], 5)
    return safe_divide(numerator, denominator)


def _alpha172(df: pd.DataFrame) -> pd.Series:
    """Alpha172。"""

    hd, ld = _directional_movement(df)
    tr = _true_range(df)
    ld_component = _where((ld > 0) & (ld > hd), ld, 0.0, df.index)
    hd_component = _where((hd > 0) & (hd > ld), hd, 0.0, df.index)
    ld_ratio = safe_divide(
        _group_transform(df["instrument_id"], ld_component, lambda s: rolling_sum(s, 14)) * 100.0,
        _group_transform(df["instrument_id"], tr, lambda s: rolling_sum(s, 14)),
    )
    hd_ratio = safe_divide(
        _group_transform(df["instrument_id"], hd_component, lambda s: rolling_sum(s, 14)) * 100.0,
        _group_transform(df["instrument_id"], tr, lambda s: rolling_sum(s, 14)),
    )
    directional_gap = safe_divide(np.abs(ld_ratio - hd_ratio), ld_ratio + hd_ratio) * 100.0
    return _group_transform(df["instrument_id"], directional_gap, lambda s: rolling_mean(s, 6))


def _alpha173(df: pd.DataFrame) -> pd.Series:
    """Alpha173。"""

    sma_close_13 = _group_transform(df["instrument_id"], df["close"], lambda s: sma_cn(s, 13, 2))
    double_sma = _group_transform(df["instrument_id"], sma_close_13, lambda s: sma_cn(s, 13, 2))
    log_close = np.log(df["close"].clip(lower=1e-8))
    triple_log_sma = _group_transform(
        df["instrument_id"],
        _group_transform(df["instrument_id"], _group_transform(df["instrument_id"], log_close, lambda s: sma_cn(s, 13, 2)), lambda s: sma_cn(s, 13, 2)),
        lambda s: sma_cn(s, 13, 2),
    )
    return 3.0 * sma_close_13 - 2.0 * double_sma + triple_log_sma


def _alpha174(df: pd.DataFrame) -> pd.Series:
    """Alpha174。"""

    std_close_20 = _group_transform(df["instrument_id"], df["close"], lambda s: rolling_std(s, 20))
    positive_std = _where(df["close"] > df["prev_close"], std_close_20, 0.0, df.index)
    return _group_transform(df["instrument_id"], positive_std, lambda s: sma_cn(s, 20, 1))


def _alpha175(df: pd.DataFrame) -> pd.Series:
    """Alpha175。"""

    return _group_transform(df["instrument_id"], _true_range(df), lambda s: rolling_mean(s, 6))


def _alpha176(df: pd.DataFrame) -> pd.Series:
    """Alpha176。"""

    close_low_ratio = safe_divide(
        df["close"] - _group_transform(df["instrument_id"], df["low"], lambda s: rolling_min(s, 12)),
        _group_transform(df["instrument_id"], df["high"], lambda s: rolling_max(s, 12))
        - _group_transform(df["instrument_id"], df["low"], lambda s: rolling_min(s, 12)),
    )
    rank_ratio = _cs_rank(df["date"], close_low_ratio)
    rank_volume = _cs_rank(df["date"], df["volume"])
    return _group_corr(df["instrument_id"], rank_ratio, rank_volume, window=6)


def _alpha177(df: pd.DataFrame) -> pd.Series:
    """Alpha177。"""

    high_day = _group_transform(df["instrument_id"], df["high"], lambda s: _days_since_argmax(s, 20))
    return safe_divide(20.0 - high_day, 20.0) * 100.0


def _alpha178(df: pd.DataFrame) -> pd.Series:
    """Alpha178。"""

    return safe_divide(df["close"] - df["prev_close"], df["prev_close"]) * df["volume"]


def _alpha179(df: pd.DataFrame) -> pd.Series:
    """Alpha179。"""

    corr_vwap_volume = _group_corr(df["instrument_id"], df["vwap"], df["volume"], window=4)
    rank_low = _cs_rank(df["date"], df["low"])
    mean_volume_50 = _group_transform(df["instrument_id"], df["volume"], lambda s: rolling_mean(s, 50))
    rank_mean_volume_50 = _cs_rank(df["date"], mean_volume_50)
    corr_low_volume = _group_corr(df["instrument_id"], rank_low, rank_mean_volume_50, window=12)
    return _cs_rank(df["date"], corr_vwap_volume) * _cs_rank(df["date"], corr_low_volume)


def _alpha180(df: pd.DataFrame) -> pd.Series:
    """Alpha180。"""

    mean_volume_20 = _group_transform(df["instrument_id"], df["volume"], lambda s: rolling_mean(s, 20))
    abs_delta_close_7 = np.abs(_group_transform(df["instrument_id"], df["close"], lambda s: delta(s, 7)))
    tsrank_abs_delta = _group_transform(df["instrument_id"], abs_delta_close_7, lambda s: ts_rank(s, 60))
    sign_delta_close = np.sign(_group_transform(df["instrument_id"], df["close"], lambda s: delta(s, 7)))
    conditional_value = -tsrank_abs_delta * sign_delta_close
    return pd.Series(np.where(mean_volume_20 < df["volume"], conditional_value, -df["volume"]), index=df.index)


def _alpha184(df: pd.DataFrame) -> pd.Series:
    """Alpha184。"""

    delayed_open_close_gap = _group_transform(df["instrument_id"], df["open"] - df["close"], lambda s: delay(s, 1))
    corr_value = _group_corr(df["instrument_id"], delayed_open_close_gap, df["close"], window=200)
    return _cs_rank(df["date"], corr_value) + _cs_rank(df["date"], df["open"] - df["close"])


def _alpha185(df: pd.DataFrame) -> pd.Series:
    """Alpha185。"""

    return _cs_rank(df["date"], -np.power(1.0 - safe_divide(df["open"], df["close"]), 2))


def _alpha186(df: pd.DataFrame) -> pd.Series:
    """Alpha186。"""

    current = _alpha172(df)
    delayed = _group_transform(df["instrument_id"], current, lambda s: delay(s, 6))
    return (current + delayed) / 2.0


def _alpha187(df: pd.DataFrame) -> pd.Series:
    """Alpha187。"""

    open_up_move = _where(
        df["open"] <= df["prev_open"],
        0.0,
        pd.Series(np.maximum(df["high"] - df["open"], df["open"] - df["prev_open"]), index=df.index),
        df.index,
    )
    return _group_transform(df["instrument_id"], open_up_move, lambda s: rolling_sum(s, 20))


def _alpha188(df: pd.DataFrame) -> pd.Series:
    """Alpha188。"""

    range_series = df["high"] - df["low"]
    sma_range = _group_transform(df["instrument_id"], range_series, lambda s: sma_cn(s, 11, 2))
    return safe_divide(range_series - sma_range, sma_range) * 100.0


def _alpha189(df: pd.DataFrame) -> pd.Series:
    """Alpha189。"""

    mean_close_6 = _group_transform(df["instrument_id"], df["close"], lambda s: rolling_mean(s, 6))
    abs_deviation = np.abs(df["close"] - mean_close_6)
    return _group_transform(df["instrument_id"], abs_deviation, lambda s: rolling_mean(s, 6))


def _alpha191(df: pd.DataFrame) -> pd.Series:
    """Alpha191。"""

    mean_volume_20 = _group_transform(df["instrument_id"], df["volume"], lambda s: rolling_mean(s, 20))
    corr_value = _group_corr(df["instrument_id"], mean_volume_20, df["low"], window=5)
    return corr_value + (df["high"] + df["low"]) / 2.0 - df["close"]


SUPPORTED_ALPHA_FUNCTIONS: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "alpha001": _alpha001,
    "alpha002": _alpha002,
    "alpha003": _alpha003,
    "alpha004": _alpha004,
    "alpha005": _alpha005,
    "alpha006": _alpha006,
    "alpha007": _alpha007,
    "alpha008": _alpha008,
    "alpha009": _alpha009,
    "alpha010": _alpha010,
    "alpha011": _alpha011,
    "alpha012": _alpha012,
    "alpha013": _alpha013,
    "alpha014": _alpha014,
    "alpha015": _alpha015,
    "alpha016": _alpha016,
    "alpha017": _alpha017,
    "alpha018": _alpha018,
    "alpha019": _alpha019,
    "alpha020": _alpha020,
    "alpha021": _alpha021,
    "alpha022": _alpha022,
    "alpha023": _alpha023,
    "alpha024": _alpha024,
    "alpha025": _alpha025,
    "alpha026": _alpha026,
    "alpha027": _alpha027,
    "alpha028": _alpha028,
    "alpha029": _alpha029,
    "alpha031": _alpha031,
    "alpha032": _alpha032,
    "alpha033": _alpha033,
    "alpha034": _alpha034,
    "alpha035": _alpha035,
    "alpha037": _alpha037,
    "alpha038": _alpha038,
    "alpha039": _alpha039,
    "alpha040": _alpha040,
    "alpha041": _alpha041,
    "alpha042": _alpha042,
    "alpha043": _alpha043,
    "alpha044": _alpha044,
    "alpha045": _alpha045,
    "alpha046": _alpha046,
    "alpha047": _alpha047,
    "alpha048": _alpha048,
    "alpha049": _alpha049,
    "alpha050": _alpha050,
    "alpha051": _alpha051,
    "alpha052": _alpha052,
    "alpha053": _alpha053,
    "alpha054": _alpha054,
    "alpha057": _alpha057,
    "alpha058": _alpha058,
    "alpha059": _alpha059,
    "alpha060": _alpha060,
    "alpha061": _alpha061,
    "alpha062": _alpha062,
    "alpha063": _alpha063,
    "alpha064": _alpha064,
    "alpha065": _alpha065,
    "alpha066": _alpha066,
    "alpha067": _alpha067,
    "alpha068": _alpha068,
    "alpha069": _alpha069,
    "alpha070": _alpha070,
    "alpha071": _alpha071,
    "alpha072": _alpha072,
    "alpha073": _alpha073,
    "alpha074": _alpha074,
    "alpha076": _alpha076,
    "alpha077": _alpha077,
    "alpha078": _alpha078,
    "alpha079": _alpha079,
    "alpha080": _alpha080,
    "alpha081": _alpha081,
    "alpha082": _alpha082,
    "alpha083": _alpha083,
    "alpha084": _alpha084,
    "alpha085": _alpha085,
    "alpha086": _alpha086,
    "alpha087": _alpha087,
    "alpha088": _alpha088,
    "alpha089": _alpha089,
    "alpha090": _alpha090,
    "alpha091": _alpha091,
    "alpha092": _alpha092,
    "alpha093": _alpha093,
    "alpha094": _alpha094,
    "alpha095": _alpha095,
    "alpha096": _alpha096,
    "alpha097": _alpha097,
    "alpha098": _alpha098,
    "alpha099": _alpha099,
    "alpha100": _alpha100,
    "alpha101": _alpha101,
    "alpha102": _alpha102,
    "alpha103": _alpha103,
    "alpha104": _alpha104,
    "alpha105": _alpha105,
    "alpha106": _alpha106,
    "alpha107": _alpha107,
    "alpha108": _alpha108,
    "alpha109": _alpha109,
    "alpha110": _alpha110,
    "alpha111": _alpha111,
    "alpha112": _alpha112,
    "alpha113": _alpha113,
    "alpha114": _alpha114,
    "alpha115": _alpha115,
    "alpha116": _alpha116,
    "alpha117": _alpha117,
    "alpha118": _alpha118,
    "alpha119": _alpha119,
    "alpha120": _alpha120,
    "alpha121": _alpha121,
    "alpha122": _alpha122,
    "alpha123": _alpha123,
    "alpha124": _alpha124,
    "alpha125": _alpha125,
    "alpha126": _alpha126,
    "alpha127": _alpha127,
    "alpha128": _alpha128,
    "alpha129": _alpha129,
    "alpha130": _alpha130,
    "alpha131": _alpha131,
    "alpha132": _alpha132,
    "alpha133": _alpha133,
    "alpha134": _alpha134,
    "alpha135": _alpha135,
    "alpha136": _alpha136,
    "alpha138": _alpha138,
    "alpha139": _alpha139,
    "alpha140": _alpha140,
    "alpha141": _alpha141,
    "alpha142": _alpha142,
    "alpha144": _alpha144,
    "alpha145": _alpha145,
    "alpha146": _alpha146,
    "alpha147": _alpha147,
    "alpha148": _alpha148,
    "alpha150": _alpha150,
    "alpha151": _alpha151,
    "alpha152": _alpha152,
    "alpha153": _alpha153,
    "alpha154": _alpha154,
    "alpha155": _alpha155,
    "alpha156": _alpha156,
    "alpha158": _alpha158,
    "alpha159": _alpha159,
    "alpha160": _alpha160,
    "alpha161": _alpha161,
    "alpha162": _alpha162,
    "alpha163": _alpha163,
    "alpha164": _alpha164,
    "alpha167": _alpha167,
    "alpha168": _alpha168,
    "alpha169": _alpha169,
    "alpha170": _alpha170,
    "alpha171": _alpha171,
    "alpha172": _alpha172,
    "alpha173": _alpha173,
    "alpha174": _alpha174,
    "alpha175": _alpha175,
    "alpha176": _alpha176,
    "alpha177": _alpha177,
    "alpha178": _alpha178,
    "alpha179": _alpha179,
    "alpha180": _alpha180,
    "alpha184": _alpha184,
    "alpha185": _alpha185,
    "alpha186": _alpha186,
    "alpha187": _alpha187,
    "alpha188": _alpha188,
    "alpha189": _alpha189,
    "alpha191": _alpha191,
}


UNSUPPORTED_ALPHA_FACTORS = {
    "alpha030": "需要市场因子 MKT / SMB / HML 和回归残差，当前项目输入数据里没有这些字段。",
    "alpha036": "原始文本排版存在歧义，`RANK(SUM(CORR(...), 6), 2)` 的括号关系不明确。",
    "alpha055": "公式结构较复杂且原始文本存在排版问题，本版本暂未实现。",
    "alpha056": "包含多层嵌套比较与幂运算，且原始文本可读性较差，本版本暂未实现。",
    "alpha075": "依赖基准指数开收盘价 `BANCHMARKINDEXCLOSE / OPEN`，当前项目输入数据中没有这些字段。",
    "alpha137": "原始公式结构复杂且排版问题较多，本版本暂未实现。",
    "alpha143": "依赖递归状态变量 `SELF`，当前项目没有该类状态型公式引擎。",
    "alpha149": "依赖基准指数收益过滤后的滚动回归，当前项目输入数据中没有基准指数字段。",
    "alpha157": "原始公式包含多层嵌套 `PROD / LOG / TSRANK` 组合且排版歧义较大，本版本暂未实现。",
    "alpha165": "依赖 `SUMAC` 这类累计震荡路径算子，当前版本暂未实现。",
    "alpha166": "原始公式文本存在明显排版与括号问题，本版本暂未实现。",
    "alpha181": "依赖基准指数序列，当前项目输入数据中没有这些字段。",
    "alpha182": "依赖基准指数开收盘价，当前项目输入数据中没有这些字段。",
    "alpha183": "依赖 `SUMAC` 累积路径算子，当前版本暂未实现。",
    "alpha190": "原始公式包含复杂 `COUNT / SUMIF / LOG` 组合并存在排版歧义，本版本暂未实现。",
}


def list_supported_alpha_factors() -> list[str]:
    """返回当前版本已实现的 Alpha 因子名称列表。"""

    return list(SUPPORTED_ALPHA_FUNCTIONS.keys())


def generate_alpha191_features(
    data: pd.DataFrame,
    factor_names: list[str] | None = None,
    show_progress: bool = False,
) -> pd.DataFrame:
    """批量生成 Alpha191 因子。

    参数：

    - `data`：已经清洗好的日频数据
    - `factor_names`：需要生成的因子名列表，默认生成当前已实现的全部因子
    - `show_progress`：是否显示进度条，适合在因子较多时开启
    """

    prepared_df = _prepare_base_dataframe(data)
    # `None` 表示“使用默认全量 Alpha”；
    # 空列表 `[]` 表示“这次明确不生成 Alpha”。
    # 这两个含义必须区分开，否则 `technical_only` 这类消融实验
    # 会在后台偷偷生成全部 Alpha191，浪费大量时间。
    selected_factors = list_supported_alpha_factors() if factor_names is None else list(factor_names)
    generated_factor_map: dict[str, pd.Series] = {}

    for factor_name in _iter_with_progress(selected_factors, description="Generating Alpha191 factors", show_progress=show_progress):
        if factor_name in SUPPORTED_ALPHA_FUNCTIONS:
            factor_values = SUPPORTED_ALPHA_FUNCTIONS[factor_name](prepared_df)

            # 这里不要像之前那样在循环里反复执行：
            # `feature_df[factor_name] = ...`
            #
            # 原因是当 Alpha 因子很多时，这会让 pandas DataFrame 变得高度碎片化，
            # 进而触发 `DataFrame is highly fragmented` 的 PerformanceWarning，
            # 同时还会让内存访问和后续运算都变慢。
            #
            # 更稳妥的做法是：
            # 1. 先把每个因子的结果收集到字典里；
            # 2. 最后一次性构造成完整 DataFrame。
            #
            # 另外，这里统一把结果转成与原始索引对齐的 Series，
            # 这样即使某些因子函数返回的是 numpy 数组，也能保持行索引一致。
            generated_factor_map[factor_name] = pd.Series(factor_values, index=prepared_df.index)
            continue

        if factor_name in UNSUPPORTED_ALPHA_FACTORS:
            raise ValueError(f"{factor_name} is recognized but not implemented yet: {UNSUPPORTED_ALPHA_FACTORS[factor_name]}")

        raise ValueError(f"Unknown alpha factor: {factor_name}")

    if not generated_factor_map:
        return pd.DataFrame(index=prepared_df.index)

    # 一次性把全部因子列拼成结果表，再做一次 copy，
    # 可以显著减少 DataFrame 内部碎片化问题。
    feature_df = pd.DataFrame(generated_factor_map, index=prepared_df.index).copy()
    return feature_df.replace([np.inf, -np.inf], np.nan)
