"""宏观与市场代理变量下载 / 合并模块。

这个模块解决一个具体问题：

主项目的 `feature_generator.py` 已经能把 `vix`、`sp500_return`、
`treasury_10y` 等列转换成 `macro_*` 因子，但原始股票日线 CSV
默认不一定包含这些列。这里提供一个独立入口，把 Yahoo Finance 上
可免费获取的宏观/市场代理序列下载下来，再按日期合并进股票面板。

重要边界：

- 这些变量是“宏观代理变量”，不是完整宏观数据库；
- `VIX`、指数收益、收益率曲线、美元指数、油价能增强市场状态描述；
- CPI、失业率、联邦基金利率等月度/政策数据更适合以后接 FRED；
- 合并时只做按日期前向填充，含义是“某天股票信号最多使用当日或更早
  已经观察到的宏观市场代理值”，不做后向填充，避免未来信息泄露。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - 运行环境缺依赖时给出清晰错误
    yf = None

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm 是可选体验增强
    tqdm = None


@dataclass(frozen=True)
class MacroProxySpec:
    """定义一个可下载的宏观/市场代理序列。

    `ticker` 是 Yahoo Finance 代码，`column_name` 是合并到项目 CSV 后的列名。
    `scale` 用来把 Yahoo 的特殊报价转成更直观的量级。例如 `^TNX` 通常是
    10 年美债收益率乘以 10，所以这里除以 10。
    """

    ticker: str
    column_name: str
    scale: float = 1.0


DEFAULT_MACRO_PROXY_SPECS = [
    MacroProxySpec("^VIX", "vix"),
    MacroProxySpec("^GSPC", "sp500_close"),
    MacroProxySpec("^IXIC", "nasdaq_close"),
    MacroProxySpec("^TNX", "treasury_10y", scale=0.1),
    # Yahoo 免费接口里 2Y ticker 不稳定；这里用 3M T-bill 作为短端利率代理。
    MacroProxySpec("^IRX", "treasury_3m", scale=0.1),
    MacroProxySpec("DX-Y.NYB", "dollar_index"),
    MacroProxySpec("CL=F", "oil_price"),
]


def _iter_with_progress(items: Iterable[MacroProxySpec], description: str, enabled: bool = True):
    """给下载循环加进度条；没有 tqdm 时退化为普通迭代。"""

    if not enabled or tqdm is None:
        return items
    return tqdm(list(items), desc=description)


def _extract_close_series(history: pd.DataFrame, spec: MacroProxySpec) -> pd.Series:
    """从 yfinance 返回结果中提取 close 序列。

    yfinance 在单 ticker / 多 ticker、不同版本下可能返回普通列或 MultiIndex 列。
    这里统一压成一条以日期为 index 的数值序列。
    """

    if history is None or history.empty:
        return pd.Series(dtype=float, name=spec.column_name)

    normalized = history.copy()
    if isinstance(normalized.columns, pd.MultiIndex):
        normalized.columns = normalized.columns.get_level_values(0)

    close_column = "Close" if "Close" in normalized.columns else "Adj Close"
    if close_column not in normalized.columns:
        return pd.Series(dtype=float, name=spec.column_name)

    close_series = pd.to_numeric(normalized[close_column], errors="coerce") * float(spec.scale)
    close_series.index = pd.to_datetime(close_series.index).tz_localize(None)
    close_series.name = spec.column_name
    return close_series.dropna()


def download_macro_proxy_data(
    start_date: str,
    end_date: str | None = None,
    specs: list[MacroProxySpec] | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """下载宏观/市场代理变量并整理成 date-level 表。

    返回表是一行一个交易日，不包含股票代码。后续会按 `date` 广播到每只股票。
    """

    if yf is None:
        raise ImportError("yfinance is not installed. Please install requirements.txt first.")

    specs = specs or DEFAULT_MACRO_PROXY_SPECS
    series_by_name: dict[str, pd.Series] = {}

    for spec in _iter_with_progress(specs, "Downloading macro proxy data", enabled=show_progress):
        try:
            history = yf.download(
                tickers=spec.ticker,
                start=start_date,
                end=end_date,
                auto_adjust=False,
                progress=False,
                interval="1d",
                threads=False,
            )
        except Exception:
            # 免费数据源偶尔会单 ticker 失败。这里跳过失败项，保留其他宏观代理。
            continue

        close_series = _extract_close_series(history, spec)
        if close_series.empty:
            continue
        series_by_name[spec.column_name] = close_series

    if not series_by_name:
        raise ValueError("No macro proxy data was downloaded. Check network, tickers, or date range.")

    macro_df = pd.DataFrame(series_by_name).sort_index()
    macro_df.index.name = "date"
    macro_df = macro_df.reset_index()

    # 派生收益率类变量。收益率比指数点位更接近可比较的市场状态特征。
    if "sp500_close" in macro_df.columns:
        macro_df["sp500_return"] = pd.to_numeric(macro_df["sp500_close"], errors="coerce").pct_change(fill_method=None)
    if "nasdaq_close" in macro_df.columns:
        macro_df["nasdaq_return"] = pd.to_numeric(macro_df["nasdaq_close"], errors="coerce").pct_change(fill_method=None)

    # 收益率曲线是常用宏观状态变量。若 2Y 没下载成功，则不强行生成。
    if {"treasury_10y", "treasury_2y"}.issubset(macro_df.columns):
        macro_df["yield_curve_10y_2y"] = macro_df["treasury_10y"] - macro_df["treasury_2y"]
    if {"treasury_10y", "treasury_3m"}.issubset(macro_df.columns):
        macro_df["yield_curve_10y_3m"] = macro_df["treasury_10y"] - macro_df["treasury_3m"]

    macro_df = macro_df.replace([np.inf, -np.inf], np.nan)
    return macro_df.sort_values("date").reset_index(drop=True)


def save_macro_proxy_data(macro_df: pd.DataFrame, output_path: str | Path) -> Path:
    """保存宏观代理变量到本地 CSV，方便复用和排查。"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    macro_df.to_csv(output_path, index=False)
    return output_path


def merge_macro_proxy_into_daily_data(
    daily_data_path: str | Path,
    macro_data_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """把 date-level 宏观代理变量合并进股票日线面板。

    合并流程：

    1. 读取股票 daily CSV 和宏观 CSV；
    2. 用股票交易日作为主时间轴；
    3. 将宏观变量按日期 reindex 到股票交易日；
    4. 对宏观变量做 forward-fill，只传播过去已经观察到的值；
    5. 按日期左连接到每只股票。
    """

    daily_data_path = Path(daily_data_path)
    macro_data_path = Path(macro_data_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    daily_df = pd.read_csv(daily_data_path)
    macro_df = pd.read_csv(macro_data_path)
    if "date" not in daily_df.columns or "date" not in macro_df.columns:
        raise ValueError("Both daily data and macro data must contain a `date` column.")

    daily_df["date"] = pd.to_datetime(daily_df["date"]).dt.tz_localize(None)
    macro_df["date"] = pd.to_datetime(macro_df["date"]).dt.tz_localize(None)

    macro_columns = [column for column in macro_df.columns if column != "date"]
    if not macro_columns:
        raise ValueError("Macro data does not contain any usable macro columns.")

    stock_dates = pd.Index(sorted(daily_df["date"].dropna().unique()), name="date")
    macro_by_date = macro_df.sort_values("date").drop_duplicates("date").set_index("date")
    aligned_macro = macro_by_date.reindex(stock_dates).ffill().reset_index()

    merged = daily_df.merge(aligned_macro, how="left", on="date", suffixes=("", "_macro"))

    # 如果原 daily CSV 已经有同名宏观列，优先用新下载的列覆盖旧列，避免历史实验残留。
    for column in macro_columns:
        duplicate_column = f"{column}_macro"
        if duplicate_column in merged.columns:
            merged[column] = merged[duplicate_column].combine_first(merged[column])
            merged = merged.drop(columns=[duplicate_column])

    merged = merged.sort_values(["instrument_id", "date"]).reset_index(drop=True)
    merged.to_csv(output_path, index=False)
    return merged
