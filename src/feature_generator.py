"""特征生成模块。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.alpha191 import generate_alpha191_features
from src.utils import (
    cross_sectional_rank,
    delay,
    ema,
    rolling_max,
    rolling_mean,
    rolling_min,
    rolling_std,
    rolling_sum,
    safe_divide,
    sma_cn,
)


OPTIONAL_FUNDAMENTAL_COLUMNS = [
    "eps",
    "pe",
    "pb",
    "ps",
    "roe",
    "roa",
    "yoy",
    "qoq",
]

OPTIONAL_MACRO_COLUMNS = [
    "vix",
    "sp500_return",
    "sp500_close",
    "nasdaq_return",
    "treasury_10y",
    "treasury_2y",
    "treasury_3m",
    "yield_curve_10y_2y",
    "yield_curve_10y_3m",
    "fed_funds_rate",
    "cpi_yoy",
    "unemployment_rate",
    "dollar_index",
    "oil_price",
]

FORBIDDEN_MODEL_FEATURE_NAMES = {
    "y",
    "next_open",
    "predicted_y",
    "adjustment",
}
FORBIDDEN_MODEL_FEATURE_PREFIXES = ("y_", "future_", "next_", "target_")


def _assert_feature_allowlist(feature_columns: list[str]) -> None:
    """Fail closed if a target, future field, or adjustment metadata leaks in."""

    forbidden = []
    for feature_name in feature_columns:
        normalized = str(feature_name).strip().lower()
        if normalized in FORBIDDEN_MODEL_FEATURE_NAMES or any(
            normalized.startswith(prefix) for prefix in FORBIDDEN_MODEL_FEATURE_PREFIXES
        ):
            forbidden.append(str(feature_name))
    if forbidden:
        raise ValueError(f"Forbidden model feature columns detected: {sorted(set(forbidden))}")


def _group_transform(data: pd.DataFrame, column: str, func) -> pd.Series:
    """按股票分组执行时序变换。"""

    return data.groupby("instrument_id")[column].transform(func)


def _group_series_transform(data: pd.DataFrame, series: pd.Series, func) -> pd.Series:
    """对临时序列按股票分组执行时序变换。"""

    return series.groupby(data["instrument_id"]).transform(func)


def _generate_base_features(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """生成基础技术指标与量价特征。

    重要说明：

    这里大量使用了“当前交易日”的 OHLCV 字段，例如 `close`、`high`、`low`、`volume`。
    这在“收盘后生成信号、预测下一期收益”的设定下是合理的；
    但如果你的真实业务是“收盘前预测当日收盘”或“盘中预测当天收益”，
    那么这些当前日字段就需要整体再滞后至少 1 根 bar。
    """

    feature_df = data.copy()
    created_features: list[str] = []

    feature_df["price_range"] = safe_divide(feature_df["high"] - feature_df["low"], feature_df["close"])
    feature_df["price_position"] = safe_divide(
        feature_df["close"] - feature_df["low"],
        feature_df["high"] - feature_df["low"],
    )
    feature_df["vwap_gap"] = safe_divide(feature_df["close"] - feature_df["vwap"], feature_df["vwap"])
    created_features.extend(["price_range", "price_position", "vwap_gap"])

    if "turnover" not in feature_df.columns:
        feature_df["turnover"] = feature_df["close"] * feature_df["volume"]

    if "market_cap" not in feature_df.columns:
        feature_df["market_cap"] = np.nan

    has_market_cap = pd.to_numeric(feature_df["market_cap"], errors="coerce").notna().any()
    if has_market_cap:
        # Dollar turnover / market cap is a scale-free liquidity proxy. Deriving
        # shares outstanding from a back-adjusted historical close would import
        # the vendor's later split scale into an old observation.
        feature_df["turnover_rate_proxy"] = safe_divide(
            feature_df["turnover"], feature_df["market_cap"]
        )
        created_features.append("turnover_rate_proxy")

    for window in [5, 10, 20]:
        close_ma_column = f"close_ma_{window}"
        volume_ma_column = f"volume_ma_{window}"
        return_std_column = f"return_std_{window}"
        momentum_column = f"momentum_{window}"
        close_to_ma_column = f"close_to_ma_{window}"
        volume_to_ma_column = f"volume_to_ma_{window}"

        feature_df[close_ma_column] = _group_transform(feature_df, "close", lambda series: rolling_mean(series, window))
        feature_df[volume_ma_column] = _group_transform(feature_df, "volume", lambda series: rolling_mean(series, window))
        feature_df[return_std_column] = _group_transform(feature_df, "log_return", lambda series: rolling_std(series, window))
        feature_df[momentum_column] = _group_transform(
            feature_df,
            "close",
            lambda series: safe_divide(series, delay(series, window)) - 1.0,
        )
        feature_df[close_to_ma_column] = safe_divide(
            feature_df["close"] - feature_df[close_ma_column],
            feature_df[close_ma_column],
        )
        feature_df[volume_to_ma_column] = safe_divide(
            feature_df["volume"] - feature_df[volume_ma_column],
            feature_df[volume_ma_column],
        )

        # Moving averages remain in feature_df as intermediate values. Absolute
        # back-adjusted price/volume levels are excluded from the candidate list;
        # only scale-free ratios, returns, and volatility enter the model.
        created_features.extend(
            [return_std_column, momentum_column, close_to_ma_column, volume_to_ma_column]
        )

    feature_df["volume_rank"] = feature_df.groupby("date")["volume"].transform(cross_sectional_rank)
    feature_df["turnover_rank"] = feature_df.groupby("date")["turnover"].transform(cross_sectional_rank)
    created_features.extend(["volume_rank", "turnover_rank"])
    if has_market_cap:
        feature_df["market_cap_rank"] = feature_df.groupby("date")["market_cap"].transform(cross_sectional_rank)
        feature_df["turnover_rate_rank"] = feature_df.groupby("date")["turnover_rate_proxy"].transform(cross_sectional_rank)
        created_features.extend(["market_cap_rank", "turnover_rate_rank"])

    return feature_df, created_features


def _generate_advanced_technical_features(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """生成更丰富的经典技术指标。

    这里加入了用户特别提到、并且能够直接从 OHLCV 数据推导的指标：

    - `EMA`
    - `MACD`
    - `DMA`
    - `RSI`
    - `WR`
    - `RSV`
    - `KDJ`
    - `UOS`
    - `BOLL`
    - `MIKE`
    - `XSChannel`
    - `OBV`
    - 成交额类特征
    """

    feature_df = data.copy()
    created_features: list[str] = []
    fundamental_rank_features: list[str] = []

    # 成交额相关特征：这部分可视为用户提到的 `AMT` 类指标。
    feature_df["amt_ma_5"] = _group_transform(feature_df, "turnover", lambda series: rolling_mean(series, 5))
    feature_df["amt_ma_20"] = _group_transform(feature_df, "turnover", lambda series: rolling_mean(series, 20))
    feature_df["amt_ratio_20"] = safe_divide(feature_df["turnover"], feature_df["amt_ma_20"])
    feature_df["amt_range_20"] = safe_divide(
        _group_transform(feature_df, "turnover", lambda series: rolling_max(series, 20))
        - _group_transform(feature_df, "turnover", lambda series: rolling_min(series, 20)),
        feature_df["amt_ma_20"],
    )
    created_features.extend(["amt_ratio_20", "amt_range_20"])

    # EMA / MACD
    feature_df["ema_close_12"] = _group_transform(feature_df, "close", lambda series: ema(series, 12))
    feature_df["ema_close_26"] = _group_transform(feature_df, "close", lambda series: ema(series, 26))
    feature_df["macd_dif"] = feature_df["ema_close_12"] - feature_df["ema_close_26"]
    feature_df["macd_dea"] = _group_series_transform(feature_df, feature_df["macd_dif"], lambda series: ema(series, 9))
    feature_df["macd_hist"] = 2.0 * (feature_df["macd_dif"] - feature_df["macd_dea"])
    feature_df["macd_dif_pct"] = safe_divide(feature_df["macd_dif"], feature_df["close"])
    feature_df["macd_dea_pct"] = safe_divide(feature_df["macd_dea"], feature_df["close"])
    feature_df["macd_hist_pct"] = safe_divide(feature_df["macd_hist"], feature_df["close"])
    created_features.extend(["macd_dif_pct", "macd_dea_pct", "macd_hist_pct"])

    # DMA
    feature_df["close_ma_50"] = _group_transform(feature_df, "close", lambda series: rolling_mean(series, 50))
    feature_df["dma_10_50"] = feature_df["close_ma_10"] - feature_df["close_ma_50"]
    feature_df["dma_signal_10"] = _group_series_transform(feature_df, feature_df["dma_10_50"], lambda series: rolling_mean(series, 10))
    feature_df["dma_10_50_pct"] = safe_divide(feature_df["dma_10_50"], feature_df["close"])
    feature_df["dma_signal_10_pct"] = safe_divide(feature_df["dma_signal_10"], feature_df["close"])
    created_features.extend(["dma_10_50_pct", "dma_signal_10_pct"])

    # VMA / VOL
    feature_df["vma_5"] = feature_df["volume_ma_5"]
    feature_df["vma_20"] = feature_df["volume_ma_20"]
    feature_df["volatility_20"] = feature_df["return_std_20"]
    created_features.append("volatility_20")

    # RSI
    close_delta = _group_transform(feature_df, "close", lambda series: series.diff())
    up_move = pd.Series(np.maximum(close_delta, 0.0), index=feature_df.index)
    down_move = pd.Series(np.maximum(-close_delta, 0.0), index=feature_df.index)
    for window in [6, 14]:
        avg_up = _group_series_transform(feature_df, up_move, lambda series: rolling_mean(series, window))
        avg_down = _group_series_transform(feature_df, down_move, lambda series: rolling_mean(series, window))
        rsi_column = f"rsi_{window}"
        feature_df[rsi_column] = 100.0 - 100.0 / (1.0 + safe_divide(avg_up, avg_down))
        created_features.append(rsi_column)

    # WR / RSV / KDJ
    feature_df["high_14"] = _group_transform(feature_df, "high", lambda series: rolling_max(series, 14))
    feature_df["low_14"] = _group_transform(feature_df, "low", lambda series: rolling_min(series, 14))
    feature_df["wr_14"] = -100.0 * safe_divide(feature_df["high_14"] - feature_df["close"], feature_df["high_14"] - feature_df["low_14"])

    feature_df["high_9"] = _group_transform(feature_df, "high", lambda series: rolling_max(series, 9))
    feature_df["low_9"] = _group_transform(feature_df, "low", lambda series: rolling_min(series, 9))
    feature_df["rsv_9"] = 100.0 * safe_divide(feature_df["close"] - feature_df["low_9"], feature_df["high_9"] - feature_df["low_9"])
    feature_df["kdj_k_9"] = _group_series_transform(feature_df, feature_df["rsv_9"], lambda series: sma_cn(series, 3, 1))
    feature_df["kdj_d_9"] = _group_series_transform(feature_df, feature_df["kdj_k_9"], lambda series: sma_cn(series, 3, 1))
    feature_df["kdj_j_9"] = 3.0 * feature_df["kdj_k_9"] - 2.0 * feature_df["kdj_d_9"]
    created_features.extend(["wr_14", "rsv_9", "kdj_k_9", "kdj_d_9", "kdj_j_9"])

    # UOS
    prev_close = _group_transform(feature_df, "close", lambda series: delay(series, 1))
    bp = feature_df["close"] - pd.Series(np.minimum(feature_df["low"], prev_close), index=feature_df.index)
    tr = pd.Series(np.maximum(feature_df["high"], prev_close), index=feature_df.index) - pd.Series(
        np.minimum(feature_df["low"], prev_close), index=feature_df.index
    )
    avg_7 = safe_divide(_group_series_transform(feature_df, bp, lambda series: rolling_sum(series, 7)), _group_series_transform(feature_df, tr, lambda series: rolling_sum(series, 7)))
    avg_14 = safe_divide(_group_series_transform(feature_df, bp, lambda series: rolling_sum(series, 14)), _group_series_transform(feature_df, tr, lambda series: rolling_sum(series, 14)))
    avg_28 = safe_divide(_group_series_transform(feature_df, bp, lambda series: rolling_sum(series, 28)), _group_series_transform(feature_df, tr, lambda series: rolling_sum(series, 28)))
    feature_df["uos_7_14_28"] = 100.0 * (4.0 * avg_7 + 2.0 * avg_14 + avg_28) / 7.0
    created_features.append("uos_7_14_28")

    # BOLL
    feature_df["boll_mid_20"] = _group_transform(feature_df, "close", lambda series: rolling_mean(series, 20))
    feature_df["boll_std_20"] = _group_transform(feature_df, "close", lambda series: rolling_std(series, 20))
    feature_df["boll_upper_20"] = feature_df["boll_mid_20"] + 2.0 * feature_df["boll_std_20"]
    feature_df["boll_lower_20"] = feature_df["boll_mid_20"] - 2.0 * feature_df["boll_std_20"]
    feature_df["boll_width_20"] = safe_divide(feature_df["boll_upper_20"] - feature_df["boll_lower_20"], feature_df["boll_mid_20"])
    feature_df["boll_zscore_20"] = safe_divide(feature_df["close"] - feature_df["boll_mid_20"], feature_df["boll_std_20"])
    created_features.extend(["boll_width_20", "boll_zscore_20"])

    # MIKE
    typical_price = (feature_df["high"] + feature_df["low"] + feature_df["close"]) / 3.0
    feature_df["mike_wr"] = 2.0 * typical_price - feature_df["low"]
    feature_df["mike_mr"] = typical_price + (feature_df["high"] - feature_df["low"])
    feature_df["mike_sr"] = 2.0 * feature_df["high"] - feature_df["low"]
    feature_df["mike_ws"] = 2.0 * typical_price - feature_df["high"]
    feature_df["mike_ms"] = typical_price - (feature_df["high"] - feature_df["low"])
    feature_df["mike_ss"] = 2.0 * feature_df["low"] - feature_df["high"]
    for mike_column in ["mike_wr", "mike_mr", "mike_sr", "mike_ws", "mike_ms", "mike_ss"]:
        relative_column = f"{mike_column}_to_close"
        feature_df[relative_column] = safe_divide(feature_df[mike_column], feature_df["close"]) - 1.0
        created_features.append(relative_column)

    # XSChannel
    feature_df["xschannel_upper_20"] = _group_transform(feature_df, "high", lambda series: rolling_max(series, 20))
    feature_df["xschannel_lower_20"] = _group_transform(feature_df, "low", lambda series: rolling_min(series, 20))
    feature_df["xschannel_width_20"] = safe_divide(
        feature_df["xschannel_upper_20"] - feature_df["xschannel_lower_20"],
        feature_df["close"],
    )
    feature_df["xschannel_pos_20"] = safe_divide(
        feature_df["close"] - feature_df["xschannel_lower_20"],
        feature_df["xschannel_upper_20"] - feature_df["xschannel_lower_20"],
    )
    created_features.extend(["xschannel_width_20", "xschannel_pos_20"])

    # OBV
    signed_volume = pd.Series(
        np.where(
            feature_df["close"] > prev_close,
            feature_df["volume"],
            np.where(feature_df["close"] < prev_close, -feature_df["volume"], 0.0),
        ),
        index=feature_df.index,
    )
    feature_df["obv"] = _group_series_transform(feature_df, signed_volume, lambda series: series.cumsum())
    feature_df["obv_ma_20"] = _group_series_transform(feature_df, feature_df["obv"], lambda series: rolling_mean(series, 20))
    feature_df["obv_ratio_20"] = safe_divide(feature_df["obv"], feature_df["obv_ma_20"].abs())
    created_features.append("obv_ratio_20")

    # 如果输入数据已经包含一些基本面列，就一并纳入模型，并且额外构造横截面排名。
    for column in OPTIONAL_FUNDAMENTAL_COLUMNS:
        if column in feature_df.columns:
            rank_column = f"{column}_rank"
            feature_df[rank_column] = feature_df.groupby("date")[column].transform(cross_sectional_rank)
            fundamental_rank_features.append(rank_column)

    created_features.extend(fundamental_rank_features)
    return feature_df, created_features, fundamental_rank_features


def _broadcast_date_series(data: pd.DataFrame, date_series: pd.Series, column_name: str) -> pd.Series:
    """把按日期索引的市场/宏观序列映射回股票日频面板。

    例如市场 20 日波动率本质上每天只有一个值，
    但模型输入是 `date x instrument_id` 的面板，所以需要按 `date` 广播到每只股票。
    """

    mapping = pd.Series(date_series).copy()
    mapping.index = pd.to_datetime(mapping.index)
    return pd.to_datetime(data["date"]).map(mapping).rename(column_name)


def _date_level_numeric_series(data: pd.DataFrame, column: str) -> pd.Series:
    """把可能重复在每只股票上的宏观列压缩成日频序列。

    如果一个宏观变量已经被 merge 到股票面板里，通常同一天每只股票的值相同。
    这里按日期取均值，能容忍少量缺失或重复。
    """

    return pd.to_numeric(data[column], errors="coerce").groupby(pd.to_datetime(data["date"])).mean().sort_index()


def _generate_context_features(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, list[str]]]:
    """生成非纯技术类上下文因子。

    这一层的目标是把用户提到的几类“实战里更像研究项目”的数据统一抽象成因子：

    - 基本面：value / profitability / growth；
    - 行业：股票相对行业、行业相对市场；
    - 市场状态：市场收益、波动率、横截面离散度、上涨股票比例；
    - 宏观变量：如果 CSV 已经包含 VIX、利率、通胀等列，就生成变化率和 z-score。

    注意：
    这里不会凭空下载或伪造宏观数据。只有输入 CSV 里真实存在的列才会进入特征。
    """

    feature_df = data.copy()
    created_features: list[str] = []
    family_map: dict[str, list[str]] = {
        "fundamental_context": [],
        "sector_context": [],
        "market_state": [],
        "macro_context": [],
    }

    # 1. 基本面上下文：把 PE/PB/PS 转成更直观的收益率或价值暴露，再做横截面排名。
    if "pe" in feature_df.columns:
        feature_df["earnings_yield"] = safe_divide(1.0, feature_df["pe"])
        family_map["fundamental_context"].append("earnings_yield")
    elif "eps" in feature_df.columns:
        feature_df["earnings_yield"] = safe_divide(feature_df["eps"], feature_df["close"])
        family_map["fundamental_context"].append("earnings_yield")

    if "pb" in feature_df.columns:
        feature_df["book_to_price"] = safe_divide(1.0, feature_df["pb"])
        family_map["fundamental_context"].append("book_to_price")
    if "ps" in feature_df.columns:
        feature_df["sales_to_price"] = safe_divide(1.0, feature_df["ps"])
        family_map["fundamental_context"].append("sales_to_price")

    profitability_parts = [column for column in ["roe", "roa"] if column in feature_df.columns]
    if profitability_parts:
        feature_df["profitability_combo"] = feature_df[profitability_parts].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        family_map["fundamental_context"].append("profitability_combo")

    growth_parts = [column for column in ["yoy", "qoq"] if column in feature_df.columns]
    if growth_parts:
        feature_df["growth_combo"] = feature_df[growth_parts].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        family_map["fundamental_context"].append("growth_combo")

    for column in list(family_map["fundamental_context"]):
        rank_column = f"{column}_rank"
        feature_df[rank_column] = feature_df.groupby("date")[column].transform(cross_sectional_rank)
        family_map["fundamental_context"].append(rank_column)

    # 2. 行业上下文：构造股票相对行业、行业相对市场的特征。
    if "sector" in feature_df.columns:
        market_return_by_date = feature_df.groupby("date")["log_return"].transform("mean")
        sector_return_by_date = feature_df.groupby(["date", "sector"])["log_return"].transform("mean")
        feature_df["market_equal_weight_return_1d"] = market_return_by_date
        feature_df["sector_equal_weight_return_1d"] = sector_return_by_date
        feature_df["stock_excess_market_return_1d"] = feature_df["log_return"] - market_return_by_date
        feature_df["stock_excess_sector_return_1d"] = feature_df["log_return"] - sector_return_by_date
        feature_df["sector_excess_market_return_1d"] = sector_return_by_date - market_return_by_date
        family_map["sector_context"].extend(
            [
                "sector_equal_weight_return_1d",
                "stock_excess_market_return_1d",
                "stock_excess_sector_return_1d",
                "sector_excess_market_return_1d",
            ]
        )

        for column in ["momentum_20", "return_std_20", "volume_to_ma_20", "turnover_rate_proxy"]:
            if column not in feature_df.columns:
                continue
            sector_average = feature_df.groupby(["date", "sector"])[column].transform("mean")
            relative_column = f"sector_relative_{column}"
            feature_df[relative_column] = feature_df[column] - sector_average
            family_map["sector_context"].append(relative_column)

        feature_df["sector_volume_rank"] = feature_df.groupby(["date", "sector"])["volume"].transform(cross_sectional_rank)
        family_map["sector_context"].append("sector_volume_rank")
        if pd.to_numeric(feature_df["market_cap"], errors="coerce").notna().any():
            feature_df["sector_market_cap_rank"] = feature_df.groupby(["date", "sector"])["market_cap"].transform(cross_sectional_rank)
            family_map["sector_context"].append("sector_market_cap_rank")

    # 3. 市场状态：每天一个值，广播到所有股票。
    date_indexed_return = _date_level_numeric_series(feature_df, "log_return")
    market_return_5d = date_indexed_return.rolling(5, min_periods=2).sum()
    market_return_20d = date_indexed_return.rolling(20, min_periods=5).sum()
    market_volatility_20d = date_indexed_return.rolling(20, min_periods=5).std()
    market_volatility_median_252d = market_volatility_20d.rolling(252, min_periods=20).median()

    dispersion_by_date = feature_df.groupby("date")["log_return"].std().sort_index()
    breadth_by_date = feature_df.assign(_positive_return=feature_df["log_return"] > 0).groupby("date")["_positive_return"].mean().sort_index()

    market_state_series = {
        "market_return_5d": market_return_5d,
        "market_return_20d": market_return_20d,
        "market_volatility_20d": market_volatility_20d,
        "market_high_vol_regime": (market_volatility_20d > market_volatility_median_252d).astype(float),
        "market_cross_sectional_dispersion": dispersion_by_date,
        "market_breadth": breadth_by_date,
    }
    for column_name, series in market_state_series.items():
        feature_df[column_name] = _broadcast_date_series(feature_df, series, column_name)
        family_map["market_state"].append(column_name)

    # 4. 宏观上下文：只有输入 CSV 中真实存在的宏观列才会生成。
    for macro_column in OPTIONAL_MACRO_COLUMNS:
        if macro_column not in feature_df.columns:
            continue
        macro_series = _date_level_numeric_series(feature_df, macro_column).ffill()
        raw_column = f"macro_{macro_column}"
        change_5d_column = f"macro_{macro_column}_change_5d"
        zscore_252d_column = f"macro_{macro_column}_zscore_252d"
        rolling_mean_252d = macro_series.rolling(252, min_periods=20).mean()
        rolling_std_252d = macro_series.rolling(252, min_periods=20).std()

        feature_df[raw_column] = _broadcast_date_series(feature_df, macro_series, raw_column)
        feature_df[change_5d_column] = _broadcast_date_series(feature_df, macro_series.diff(5), change_5d_column)
        feature_df[zscore_252d_column] = _broadcast_date_series(
            feature_df,
            safe_divide(macro_series - rolling_mean_252d, rolling_std_252d),
            zscore_252d_column,
        )
        family_map["macro_context"].extend([raw_column, change_5d_column, zscore_252d_column])

    for feature_list in family_map.values():
        created_features.extend(feature_list)
    created_features = list(dict.fromkeys(created_features))
    return feature_df, created_features, family_map


def generate_feature_matrix(
    data: pd.DataFrame,
    alpha_factor_names: list[str] | None = None,
    show_progress: bool = False,
) -> tuple[pd.DataFrame, list[str], dict]:
    """生成完整特征矩阵与特征元数据。

    这个函数本身只负责“如何从一段给定历史生成特征”，
    不负责决定 train / test 该怎么切。

    也正因为如此，真正调用它时最好遵循下面的顺序：

    1. 先对原始数据做时间切分；
    2. 再分别对训练期 / 测试期生成特征；
    3. 如果测试期需要滚动窗口，再额外补一段来自训练期的历史上下文。

    当前项目已经把这套更安全的顺序封装到 `src/time_series_pipeline.py` 中。
    """

    # `log_return` 属于特征，因此在 raw train/OOS 切分之后才构造。
    # OOS 调用方会先拼接训练期尾部历史，所以测试第一天仍能得到只依赖
    # 已知价格的收益率，同时避免 data loader 在切分前提前制造特征。
    feature_input = data.sort_values(["instrument_id", "date"]).reset_index(drop=True).copy()
    feature_input["log_return"] = feature_input.groupby("instrument_id")["close"].transform(
        lambda series: np.log(pd.to_numeric(series, errors="coerce").clip(lower=1e-8)).diff()
    )

    feature_df, base_feature_columns = _generate_base_features(feature_input)
    feature_df, advanced_feature_columns, fundamental_rank_columns = _generate_advanced_technical_features(feature_df)
    feature_df, context_feature_columns, context_family_map = _generate_context_features(feature_df)
    alpha_feature_df = generate_alpha191_features(
        feature_df,
        factor_names=alpha_factor_names,
        show_progress=show_progress,
    )

    # 这里不要在循环里一列一列地往 feature_df 里插入 Alpha 因子。
    # 原因是当前项目里 Alpha 列很多，如果反复执行 `feature_df[column] = ...`，
    # DataFrame 会逐渐变得高度碎片化（fragmented），从而触发 pandas 的
    # PerformanceWarning，并让后续计算变慢、占用更多内存。
    #
    # 更好的做法是一次性按列拼接：
    # - 语义上更清晰：把“已有特征”和“Alpha 因子块”合并成一个新表；
    # - 性能上更稳定：避免成百上千次列插入；
    # - `.copy()` 可以顺手做一次内存整理，进一步减少碎片化。
    if not alpha_feature_df.empty:
        feature_df = pd.concat([feature_df, alpha_feature_df], axis=1).copy()

    # Absolute OHLCV/VWAP levels are deliberately not direct candidates. Vendor
    # back-adjustment can use later split information to rescale historical price
    # levels. The values are still available to construct scale-free technical and
    # Alpha formulas after the raw time split.
    raw_feature_columns = [
        column
        for column in [
            "market_cap",
            "turnover",
            "log_return",
        ]
        if column in feature_df.columns and not feature_df[column].isna().all()
    ]

    # `adjustment` 这里故意不再放进原始训练特征。
    # 原因是当数据来自 Yahoo Finance 等来源时，`adjustment` 往往和
    # 复权处理、分红、拆股等公司行为有关，里面可能隐含“事后才知道”的信息。
    # 为了降低潜在数据泄露风险，我们保留原字段用于数据记录，但不喂给模型。

    # 只把确实有观测值的基本面字段加入候选列表。全 NaN 的 `market_cap`、
    # `pb` 或 `ps` 仍可保留在数据表中用于审计，但不应虚增候选特征数量。
    fundamental_raw_columns = [
        column
        for column in OPTIONAL_FUNDAMENTAL_COLUMNS
        if column in feature_df.columns and not feature_df[column].isna().all()
    ]

    feature_columns = (
        raw_feature_columns
        + fundamental_raw_columns
        + base_feature_columns
        + advanced_feature_columns
        + context_feature_columns
        + list(alpha_feature_df.columns)
    )
    feature_columns = list(dict.fromkeys(feature_columns))
    _assert_feature_allowlist(feature_columns)

    feature_metadata = {
        "raw_feature_columns": raw_feature_columns,
        "fundamental_raw_columns": fundamental_raw_columns,
        "base_feature_columns": base_feature_columns,
        "advanced_feature_columns": advanced_feature_columns,
        "context_feature_columns": context_feature_columns,
        "context_feature_family_map": context_family_map,
        "fundamental_rank_columns": fundamental_rank_columns,
        "alpha_feature_columns": list(alpha_feature_df.columns),
        "feature_counts": {
            "raw_feature_count": len(raw_feature_columns),
            "fundamental_raw_count": len(fundamental_raw_columns),
            "base_feature_count": len(base_feature_columns),
            "advanced_feature_count": len(advanced_feature_columns),
            "context_feature_count": len(context_feature_columns),
            "alpha_feature_count": len(alpha_feature_df.columns),
            "candidate_feature_count": len(feature_columns),
        },
    }

    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    return feature_df, feature_columns, feature_metadata
