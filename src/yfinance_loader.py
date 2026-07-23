"""Yahoo Finance 数据下载模块。

说明：

- 用户消息中写的是 “yhfinance”，这里按常见用法实现为 `yfinance`
- 该模块负责把 Yahoo Finance 的原始日线数据转换成项目统一字段
- 同时提供保存到本地 CSV 的能力，便于后续重复训练或离线分析
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.data_loader import DEFAULT_TARGET_COLUMN, add_forward_return_targets
from src.universe import get_symbol_sector_map

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def _iter_with_progress(items: Iterable, description: str):
    """为循环提供可选进度条。

    如果环境里已经安装了 `tqdm`，就显示进度条；
    如果没有安装，也不会报错，而是退化成普通循环。
    """

    if tqdm is None:
        return items
    return tqdm(items, desc=description)


def _normalize_yfinance_history(history: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """把单个股票的 Yahoo Finance 数据转换为项目标准列。"""

    # 新版 yfinance 在只有单个 ticker 时，也可能返回 MultiIndex 列：
    # 例如 `('Open', 'AAPL')`、`('Close', 'AAPL')`。
    # 由于当前函数本来就是“一次只处理一只股票”，这里可以安全地把列压平成
    # 第一层价格字段，恢复成更常见的 `Open / High / Low / Close / Volume` 结构。
    normalized = history.copy()
    if isinstance(normalized.columns, pd.MultiIndex):
        normalized.columns = normalized.columns.get_level_values(0)

    normalized = normalized.reset_index().copy()
    normalized.columns = [str(column).strip() for column in normalized.columns]

    # yfinance 的列名通常是首字母大写，这里统一转成项目内部的小写格式。
    rename_map = {
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    normalized = normalized.rename(columns=rename_map)
    normalized["instrument_id"] = symbol

    # 日频接口通常不直接提供 VWAP，因此这里用 OHLC 均值构造一个近似版本，
    # 目的是保证下游特征工程可以直接复用统一字段。
    normalized["vwap"] = (
        normalized["open"] + normalized["high"] + normalized["low"] + normalized["close"]
    ) / 4.0

    if "adj_close" in normalized.columns:
        raw_close = pd.to_numeric(normalized["close"], errors="coerce")
        normalized["adjustment"] = pd.to_numeric(
            normalized["adj_close"], errors="coerce"
        ) / raw_close.where(raw_close.ne(0.0), np.nan)
    else:
        normalized["adjustment"] = 1.0

    normalized["turnover"] = normalized["close"] * normalized["volume"]
    normalized["next_open"] = normalized["open"].shift(-1)

    # Yahoo Finance 日频免费接口通常不提供可按历史日期对齐的真实市值。
    # 这里保留空值；data_loader 和特征层都不会再用价格×成交量伪造市值。
    normalized["market_cap"] = np.nan

    standard_columns = [
        "instrument_id",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vwap",
        "adjustment",
        "next_open",
        "market_cap",
        "turnover",
    ]
    return normalized[standard_columns].copy()


def download_yfinance_to_csv(
    symbols: list[str],
    output_path: str | Path,
    start_date: str,
    end_date: str | None = None,
    auto_adjust: bool = False,
) -> pd.DataFrame:
    """下载多个股票的 Yahoo Finance 日线数据并保存为 CSV。

    参数说明：

    - `symbols`：股票代码列表，例如 `["AAPL", "MSFT"]`
    - `output_path`：保存到本地的 CSV 路径
    - `start_date` / `end_date`：下载区间
    - `auto_adjust`：是否让 yfinance 自动复权
    """

    if yf is None:
        raise ImportError("yfinance is not installed. Please install requirements.txt first.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_frames: list[pd.DataFrame] = []

    for symbol in _iter_with_progress(symbols, description="Downloading Yahoo Finance data"):
        history = yf.download(
            tickers=symbol,
            start=start_date,
            end=end_date,
            auto_adjust=auto_adjust,
            progress=False,
            interval="1d",
            threads=False,
        )

        if history is None or history.empty:
            continue

        normalized = _normalize_yfinance_history(history, symbol=symbol)
        all_frames.append(normalized)

    if not all_frames:
        raise ValueError("No price data was downloaded. Please check symbols and date range.")

    merged = pd.concat(all_frames, ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"]).dt.tz_localize(None)
    merged = merged.sort_values(["instrument_id", "date"]).reset_index(drop=True)

    # 给真实数据补上静态板块标签。这个映射不是 point-in-time 行业历史：
    # 它适合做受限的行业内排名诊断，但必须在报告里披露分类变更风险。
    # 自定义 ticker 不在内置映射时标记为 Unknown。
    sector_map = get_symbol_sector_map(symbols)
    merged["sector"] = merged["instrument_id"].map(sector_map).fillna("Unknown")

    # 在下载阶段就把多周期目标列落到 CSV，
    # 这样你直接打开原始数据文件时，也能清楚看到项目到底在预测哪些 horizon。
    merged = add_forward_return_targets(merged, price_column="close")
    merged["y"] = merged[DEFAULT_TARGET_COLUMN]
    merged.to_csv(output_path, index=False)
    return merged
