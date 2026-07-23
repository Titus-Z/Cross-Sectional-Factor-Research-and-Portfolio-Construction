"""FMP 基本面数据下载与按披露日期合并模块。

这个模块只做两件事：

1. 从 Financial Modeling Prep 下载季度基本面数据；
2. 把季度基本面面板按“披露日期”安全地 merge 到日频价格数据。

为什么一定要按披露日期 merge，而不是按财报期末日期直接回填？

- 因为模型在某个交易日只能看到“当天之前已经公开披露”的信息；
- 如果你拿 2025Q1 的财报数据，按 2025-03-31 就开始回填到日频价格表里，
  但这份财报实际是 2025-05-02 才披露，那就等于偷偷用了未来信息；
- 这种错误在量化里非常常见，而且会直接把结果做假。

因此，这里会优先使用：

- `acceptedDate`
- 其次 `fillingDate` / `filingDate`
- 最后才退回 `date + 90 天`（对缺失披露日期的保守可用日假设）

这并不完美，但已经比“按财报期末直接回填”安全得多。
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from src.progress import optional_progress


FMP_BASE_URL = "https://financialmodelingprep.com/stable"
FUNDAMENTAL_REPORT_DATE_FALLBACK_LAG_DAYS = 90

# 当前项目最关心的是这几个字段。
# 它们和 feature_generator.py 里的 OPTIONAL_FUNDAMENTAL_COLUMNS 对齐，
# 这样合并后的数据可以直接被现有特征工程读取。
#
# 说明：
# - `eps / roe / roa / yoy / qoq` 会从免费可用的财报表里直接推导；
# - `pe / pb / ps` 会在 merge 到日频价格表之后，再结合当天价格和市值计算。
STANDARD_FUNDAMENTAL_COLUMNS = [
    "eps",
    "pe",
    "pb",
    "ps",
    "roe",
    "roa",
    "yoy",
    "qoq",
]


def _sanitize_request_error(error_text: str) -> str:
    """移除 HTTP 异常字符串里的敏感 query 参数。

    `requests.HTTPError` 的默认字符串会把完整 URL 打印出来。如果 URL 里
    带有 `apikey=...`，直接写入日志或 CSV 就会把密钥落盘。

    这里保留足够排查问题的信息：
    - HTTP 状态；
    - endpoint 和 symbol；
    - 是否是权限/额度问题；

    同时把 `apikey` 统一替换成 `<redacted>`。
    """

    return re.sub(r"apikey=[^&\\s]+", "apikey=<redacted>", str(error_text))


def _resolve_api_key(api_key: str | None = None) -> str:
    """解析 FMP API key。

    优先级：
    1. 显式传入的 `api_key`
    2. 环境变量 `FMP_API_KEY`
    """

    resolved = api_key or os.getenv("FMP_API_KEY")
    if not resolved:
        raise ValueError(
            "FMP API key is required. Please pass --api-key or set environment variable FMP_API_KEY."
        )
    return resolved


def _request_json(
    endpoint: str,
    params: dict,
    api_key: str,
    timeout: int = 30,
) -> list[dict]:
    """请求 FMP JSON 数据。

    这里统一走 stable base url，并且把异常处理集中起来，
    避免上层下载流程被大量重复的 HTTP 细节干扰。
    """

    request_params = dict(params)
    request_params["apikey"] = api_key
    response = requests.get(f"{FMP_BASE_URL}/{endpoint}", params=request_params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()

    # FMP 多数 endpoint 会返回 list[dict]。
    # 但某些异常情况下可能返回单 dict，这里统一包装成 list。
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return payload
    return []


def _pick_first(record: dict, candidate_keys: list[str]) -> float | str | None:
    """从多个候选字段名里取第一个非空值。

    FMP 不同 endpoint、不同公司、甚至不同版本文档里，
    字段命名可能会有细小差异。这里做一层温和的字段兼容，
    让下载脚本更稳一些。
    """

    for key in candidate_keys:
        if key in record and pd.notna(record[key]):
            return record[key]
    return None


def _normalize_statement_frame(records: list[dict], symbol: str, source_name: str) -> pd.DataFrame:
    """把单个 endpoint 返回的记录规范成 DataFrame。"""

    if not records:
        return pd.DataFrame()

    frame = pd.DataFrame(records).copy()
    if frame.empty:
        return frame

    frame["instrument_id"] = symbol
    frame["source_name"] = source_name

    # FMP 里常见的是 acceptedDate / fillingDate / filingDate / date。
    # 我们统一生成一个 `effective_date`，后面 merge 时用它表示“最晚不超过该交易日可见”的日期。
    accepted_date = None
    for candidate in ["acceptedDate", "accepted_date"]:
        if candidate in frame.columns:
            accepted_date = pd.to_datetime(frame[candidate], errors="coerce")
            break

    filing_date = None
    for candidate in ["filingDate", "fillingDate", "filing_date", "filling_date"]:
        if candidate in frame.columns:
            filing_date = pd.to_datetime(frame[candidate], errors="coerce")
            break

    report_date = None
    for candidate in ["date", "reportDate", "reportedDate"]:
        if candidate in frame.columns:
            report_date = pd.to_datetime(frame[candidate], errors="coerce")
            break

    if report_date is None:
        report_date = pd.Series(pd.NaT, index=frame.index)
    if filing_date is None:
        filing_date = pd.Series(pd.NaT, index=frame.index)
    if accepted_date is None:
        accepted_date = pd.Series(pd.NaT, index=frame.index)

    frame["report_date"] = report_date.dt.tz_localize(None)
    frame["filing_date"] = filing_date.dt.tz_localize(None)
    frame["accepted_date"] = accepted_date.dt.tz_localize(None)
    # 财报期末本身不是市场可见日期。缺少 accepted/filing 时间戳时，
    # 使用 report_date + 90 天作为保守近似，避免把尚未披露的信息提前合并。
    fallback_available_date = frame["report_date"] + pd.to_timedelta(
        FUNDAMENTAL_REPORT_DATE_FALLBACK_LAG_DAYS,
        unit="D",
    )
    frame["effective_date"] = (
        frame["accepted_date"]
        .fillna(frame["filing_date"])
        .fillna(fallback_available_date)
    )

    return frame


def _build_quarterly_fundamental_panel(
    symbol: str,
    income_df: pd.DataFrame,
    balance_df: pd.DataFrame,
) -> pd.DataFrame:
    """把多个 endpoint 组装成统一的季度基本面面板。

    这里不追求把 FMP 的所有字段都搬进来，只取当前项目最可能有用、
    并且适合先接入机器学习管线的少量字段。
    """

    merge_keys = ["instrument_id", "report_date"]
    base_frames = [frame for frame in [income_df, balance_df] if not frame.empty]
    if not base_frames:
        return pd.DataFrame()

    panel = base_frames[0].copy()
    for frame in base_frames[1:]:
        candidate_columns = [
            column
            for column in frame.columns
            if column not in merge_keys and column not in {"source_name"}
        ]
        panel = panel.merge(
            frame[merge_keys + candidate_columns],
            on=merge_keys,
            how="outer",
            suffixes=("", "_dup"),
        )

    # 合并多个报表来源时，同一财季的收入表和资产负债表可能具有不同的
    # accepted/filing 时间。整行基本面只有等到所有被使用字段都已公开后才可见，
    # 因此 availability 类日期必须取最晚值。过去用 bfill 取第一列，可能让较晚
    # 披露的另一张报表字段提前进入日频样本。
    for base_name in ["filing_date", "accepted_date", "effective_date", "period", "calendarYear"]:
        duplicate_columns = [column for column in panel.columns if column == base_name or column.startswith(f"{base_name}_")]
        if not duplicate_columns:
            continue
        if base_name in {"filing_date", "accepted_date", "effective_date"}:
            date_candidates = panel[duplicate_columns].apply(pd.to_datetime, errors="coerce")
            panel[base_name] = date_candidates.max(axis=1)
        else:
            panel[base_name] = panel[duplicate_columns].bfill(axis=1).iloc[:, 0]
        drop_columns = [column for column in duplicate_columns if column != base_name]
        if drop_columns:
            panel = panel.drop(columns=drop_columns)

    # 统一抽取当前项目关心的字段。
    extracted_records: list[dict] = []
    panel = panel.sort_values("report_date").reset_index(drop=True)

    for _, record in panel.iterrows():
        record_dict = record.to_dict()

        revenue = _pick_first(
            record_dict,
            [
                "revenue",
                "revenue_dup",
            ],
        )
        net_income = _pick_first(record_dict, ["netIncome", "netIncome_dup"])
        total_assets = _pick_first(record_dict, ["totalAssets", "totalAssets_dup"])
        total_equity = _pick_first(
            record_dict,
            [
                "totalStockholdersEquity",
                "totalEquity",
                "totalStockholdersEquity_dup",
                "totalEquity_dup",
            ],
        )
        eps = _pick_first(record_dict, ["eps", "epsDiluted"])
        extracted_records.append(
            {
                "instrument_id": symbol,
                "report_date": pd.to_datetime(record_dict.get("report_date"), errors="coerce"),
                "filing_date": pd.to_datetime(record_dict.get("filing_date"), errors="coerce"),
                "accepted_date": pd.to_datetime(record_dict.get("accepted_date"), errors="coerce"),
                "effective_date": pd.to_datetime(record_dict.get("effective_date"), errors="coerce"),
                "fiscal_period": record_dict.get("period"),
                "fiscal_year": record_dict.get("calendarYear"),
                "revenue": revenue,
                "net_income": net_income,
                "total_assets": total_assets,
                "total_equity": total_equity,
                "eps": eps,
            }
        )

    fundamentals_df = pd.DataFrame(extracted_records).copy()
    if fundamentals_df.empty:
        return fundamentals_df

    numeric_columns = [
        "revenue",
        "net_income",
        "total_assets",
        "total_equity",
        "eps",
    ]
    for column in numeric_columns:
        fundamentals_df[column] = pd.to_numeric(fundamentals_df[column], errors="coerce")

    fundamentals_df = fundamentals_df.sort_values(["instrument_id", "report_date"]).reset_index(drop=True)

    # 免费计划拿不到季度 ratios endpoint，因此这里自己推导质量类指标。
    fundamentals_df["revenue_ttm"] = fundamentals_df.groupby("instrument_id")["revenue"].transform(
        lambda series: series.rolling(window=4, min_periods=4).sum()
    )
    fundamentals_df["net_income_ttm"] = fundamentals_df.groupby("instrument_id")["net_income"].transform(
        lambda series: series.rolling(window=4, min_periods=4).sum()
    )
    fundamentals_df["avg_assets_4q"] = fundamentals_df.groupby("instrument_id")["total_assets"].transform(
        lambda series: series.rolling(window=4, min_periods=2).mean()
    )
    fundamentals_df["avg_equity_4q"] = fundamentals_df.groupby("instrument_id")["total_equity"].transform(
        lambda series: series.rolling(window=4, min_periods=2).mean()
    )
    fundamentals_df["roe"] = fundamentals_df["net_income_ttm"] / fundamentals_df["avg_equity_4q"]
    fundamentals_df["roa"] = fundamentals_df["net_income_ttm"] / fundamentals_df["avg_assets_4q"]

    # 这里把 `yoy` / `qoq` 明确定义成 revenue growth。
    # 原因很简单：先用收入增长率做一个稳定、容易解释的版本。
    fundamentals_df["yoy"] = fundamentals_df.groupby("instrument_id")["revenue"].transform(
        lambda series: series / series.shift(4) - 1.0
    )
    fundamentals_df["qoq"] = fundamentals_df.groupby("instrument_id")["revenue"].transform(
        lambda series: series / series.shift(1) - 1.0
    )

    fundamentals_df = fundamentals_df.replace([np.inf, -np.inf], np.nan)
    return fundamentals_df


def download_fmp_quarterly_fundamentals(
    symbols: list[str],
    api_key: str | None = None,
    limit: int = 5,
    period: str = "quarter",
    sleep_sec: float = 0.2,
    show_progress: bool = True,
) -> pd.DataFrame:
    """下载多个股票的季度基本面数据。

    下载策略：
    - 每个 symbol 请求 2 个免费可用 endpoint：income / balance
    - 再从这两张表推导出当前项目需要的关键字段

    之所以这样做，是因为 FMP 免费计划下：
    - `ratios` 和 `key-metrics` 的季度接口会返回 402；
    - 但 `income-statement` 和 `balance-sheet-statement` 仍然可用。
    """

    resolved_api_key = _resolve_api_key(api_key)
    all_panels: list[pd.DataFrame] = []
    skipped_symbols: list[dict[str, str | int]] = []

    for symbol in optional_progress(symbols, description="Downloading FMP fundamentals", enabled=show_progress):
        try:
            income_records = _request_json(
                endpoint="income-statement",
                params={"symbol": symbol, "period": period, "limit": limit},
                api_key=resolved_api_key,
            )
            balance_records = _request_json(
                endpoint="balance-sheet-statement",
                params={"symbol": symbol, "period": period, "limit": limit},
                api_key=resolved_api_key,
            )
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else -1
            skipped_symbols.append(
                {
                    "instrument_id": symbol,
                    "status_code": int(status_code),
                    "reason": _sanitize_request_error(str(exc)),
                }
            )
            print(f"[Warn] Skip {symbol} due to FMP HTTP error: {status_code}")
            if sleep_sec > 0:
                time.sleep(sleep_sec)
            continue
        except requests.RequestException as exc:
            skipped_symbols.append(
                {
                    "instrument_id": symbol,
                    "status_code": -1,
                    "reason": _sanitize_request_error(str(exc)),
                }
            )
            print(f"[Warn] Skip {symbol} due to network/API error.")
            if sleep_sec > 0:
                time.sleep(sleep_sec)
            continue
        income_df = _normalize_statement_frame(income_records, symbol=symbol, source_name="income_statement")
        balance_df = _normalize_statement_frame(balance_records, symbol=symbol, source_name="balance_sheet")
        panel = _build_quarterly_fundamental_panel(symbol=symbol, income_df=income_df, balance_df=balance_df)
        if not panel.empty:
            all_panels.append(panel)

        # FMP 文档明确提到 429 是频率限制，因此这里主动 sleep 一下。
        # 这个停顿很便宜，但能显著降低批量抓 300 只股票时触发限频的概率。
        if sleep_sec > 0:
            time.sleep(sleep_sec)

    if not all_panels:
        raise ValueError("No FMP fundamentals were downloaded. Please check symbols, API key, and rate limits.")

    merged = pd.concat(all_panels, ignore_index=True)
    merged = merged.sort_values(["instrument_id", "effective_date", "report_date"]).reset_index(drop=True)
    merged.attrs["skipped_symbols"] = skipped_symbols
    return merged


def save_fundamentals_to_csv(fundamentals_df: pd.DataFrame, output_path: str | Path) -> Path:
    """保存季度基本面面板到本地 CSV。"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fundamentals_df.to_csv(output_path, index=False)
    skipped_symbols = fundamentals_df.attrs.get("skipped_symbols", [])
    if skipped_symbols:
        skipped_path = output_path.with_name(f"{output_path.stem}_skipped_symbols.csv")
        pd.DataFrame(skipped_symbols).to_csv(skipped_path, index=False)
    return output_path


def infer_symbols_from_daily_data(daily_data_path: str | Path) -> list[str]:
    """从现有日频价格文件中推断股票池。"""

    daily_df = pd.read_csv(daily_data_path, usecols=["instrument_id"])
    symbols = sorted(daily_df["instrument_id"].dropna().astype(str).unique().tolist())
    if not symbols:
        raise ValueError("No symbols were found in daily price data.")
    return symbols


def merge_fundamentals_into_daily_data(
    daily_df: pd.DataFrame,
    fundamentals_df: pd.DataFrame,
) -> pd.DataFrame:
    """把季度基本面按披露日期安全合并到日频数据上。

    合并规则：
    - 以日频 `date` 为左表时间轴；
    - 以基本面 `effective_date` 为右表时间轴；
    - 对每只股票单独做 backward asof merge；
    - 也就是“给某一天配上当时最新已披露的基本面信息”。
    """

    if daily_df.empty:
        return daily_df.copy()
    if fundamentals_df.empty:
        return daily_df.copy()

    left = daily_df.copy()
    right = fundamentals_df.copy()

    left["date"] = pd.to_datetime(left["date"], errors="coerce")
    right["effective_date"] = pd.to_datetime(right["effective_date"], errors="coerce")
    right["report_date"] = pd.to_datetime(right["report_date"], errors="coerce")
    right["filing_date"] = pd.to_datetime(right["filing_date"], errors="coerce")
    right["accepted_date"] = pd.to_datetime(right["accepted_date"], errors="coerce")

    right = right.dropna(subset=["instrument_id", "effective_date"]).copy()
    right = right.sort_values(["instrument_id", "effective_date"]).reset_index(drop=True)
    left = left.sort_values(["instrument_id", "date"]).reset_index(drop=True)

    # merge_asof 要求按 key 排序。
    merged_frames: list[pd.DataFrame] = []
    fundamental_columns = [
        "effective_date",
        "report_date",
        "filing_date",
        "accepted_date",
        "fiscal_period",
        "fiscal_year",
        "revenue",
        "revenue_ttm",
        "net_income",
        "net_income_ttm",
        "total_assets",
        "total_equity",
        "avg_assets_4q",
        "avg_equity_4q",
        "eps",
        "roe",
        "roa",
        "yoy",
        "qoq",
    ]

    for symbol, left_symbol_df in left.groupby("instrument_id", sort=False):
        right_symbol_df = right[right["instrument_id"] == symbol].copy()
        if right_symbol_df.empty:
            merged_frames.append(left_symbol_df.copy())
            continue

        merged_symbol_df = pd.merge_asof(
            left_symbol_df.sort_values("date"),
            right_symbol_df[["instrument_id", *fundamental_columns]].sort_values("effective_date"),
            left_on="date",
            right_on="effective_date",
            by="instrument_id",
            direction="backward",
        )
        merged_frames.append(merged_symbol_df)

    merged_df = pd.concat(merged_frames, ignore_index=True)
    merged_df = merged_df.sort_values(["instrument_id", "date"]).reset_index(drop=True)

    # 市值不能由价格和成交量可靠推导。缺少真实 point-in-time 市值时，
    # `pb` / `ps` 必须保留为缺失值，不能用流动性代理伪造估值比率。
    if "market_cap" not in merged_df.columns:
        merged_df["market_cap"] = np.nan
    merged_df["market_cap_source"] = (
        "provided" if merged_df["market_cap"].notna().any() else "missing"
    )

    if "eps" in merged_df.columns:
        merged_df["pe"] = merged_df["close"] / merged_df["eps"]
    if "total_equity" in merged_df.columns and "market_cap" in merged_df.columns:
        merged_df["pb"] = merged_df["market_cap"] / merged_df["total_equity"]
    if "revenue_ttm" in merged_df.columns and "market_cap" in merged_df.columns:
        merged_df["ps"] = merged_df["market_cap"] / merged_df["revenue_ttm"]
    merged_df = merged_df.replace([np.inf, -np.inf], np.nan)
    return merged_df


def merge_fundamentals_csv_into_daily_csv(
    daily_data_path: str | Path,
    fundamentals_path: str | Path,
    output_path: str | Path,
) -> Path:
    """从两个 CSV 读取后执行合并，并写回新的 CSV。"""

    daily_df = pd.read_csv(daily_data_path)
    fundamentals_df = pd.read_csv(fundamentals_path)
    merged_df = merge_fundamentals_into_daily_data(daily_df=daily_df, fundamentals_df=fundamentals_df)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    return output_path
