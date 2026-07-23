"""构建统一的 context 数据集。

这个入口把项目里已经有的三类“非纯价量信息”合到一份 CSV：

1. 日频股票数据：OHLCV、VWAP、市值、行业、标签；
2. FMP 基本面数据：EPS、PE、PB、PS、ROE、ROA、YoY、QoQ；
3. 宏观 / 市场代理变量：VIX、指数收益、利率、美元、油价等。

为什么要单独做这个入口？

- `main_fmp_fundamentals.py` 只负责基本面；
- `main_macro_data.py` 只负责宏观；
- 真实实验时模型只应该读取一份清楚的数据源，否则很容易出现“这次实验到底用了哪些上下文变量”
  说不清楚的问题。

这个脚本默认不重新下载 FMP 基本面，因为免费 API 有每日调用限制。
它会优先复用你已经保存到本地的 fundamentals CSV。
宏观代理变量来自 Yahoo Finance，调用成本低，可以按需重新下载。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from src.fmp_fundamentals import merge_fundamentals_csv_into_daily_csv
from src.macro_data import (
    download_macro_proxy_data,
    merge_macro_proxy_into_daily_data,
    save_macro_proxy_data,
)
from src.progress import create_progress_bar, format_duration
from src.project_paths import resolve_project_path
from src.runtime_config import DEFAULT_PRIMARY_DATA_PATH, DEFAULT_SAMPLE_START_DATE


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    这里故意把基本面和宏观拆成可选项：
    - 没有 FMP quota 时，仍可只构建 macro context 数据；
    - 已经有本地 fundamentals CSV 时，可以无 API 成本地合并；
    - 宏观数据可重新下载，也可复用本地 CSV。
    """

    parser = argparse.ArgumentParser(description="Build daily price + fundamentals + macro context CSV.")
    parser.add_argument("--daily-data-path", default=DEFAULT_PRIMARY_DATA_PATH, help="基础日频股票 CSV。")
    parser.add_argument(
        "--fundamentals-path",
        default=None,
        help="本地 FMP 季度基本面 CSV；不传则跳过基本面合并。",
    )
    parser.add_argument(
        "--macro-output-path",
        default="data/macro/macro_proxy_daily.csv",
        help="宏观代理变量 CSV 输出或读取路径。",
    )
    parser.add_argument(
        "--output-path",
        default="data/us_large_cap_300_context.csv",
        help="最终统一 context CSV 输出路径。",
    )
    parser.add_argument("--start-date", default=DEFAULT_SAMPLE_START_DATE, help="宏观数据下载起点。")
    parser.add_argument("--end-date", default=None, help="宏观数据下载终点。")
    parser.add_argument("--skip-macro-download", action="store_true", help="复用已有 macro-output-path，不重新下载。")
    parser.add_argument("--skip-macro", action="store_true", help="完全跳过宏观合并。")
    parser.add_argument("--no-progress", action="store_true", help="关闭进度条。")
    return parser.parse_args()


def summarize_context_file(output_path: Path) -> None:
    """输出最小摘要，确认哪些上下文字段真的进入了结果文件。"""

    data = pd.read_csv(output_path)
    fundamental_columns = [column for column in ["eps", "pe", "pb", "ps", "roe", "roa", "yoy", "qoq"] if column in data.columns]
    macro_columns = [
        column
        for column in [
            "vix",
            "sp500_close",
            "sp500_return",
            "nasdaq_close",
            "nasdaq_return",
            "treasury_10y",
            "treasury_3m",
            "yield_curve_10y_3m",
            "dollar_index",
            "oil_price",
        ]
        if column in data.columns
    ]
    sector_columns = [column for column in ["sector", "industry"] if column in data.columns]

    print(f"[Info] Context CSV: {output_path}")
    print(f"[Info] Rows: {len(data):,}")
    if "instrument_id" in data.columns:
        print(f"[Info] Instruments: {data['instrument_id'].nunique():,}")
    if "date" in data.columns:
        date_series = pd.to_datetime(data["date"], errors="coerce")
        print(f"[Info] Date range: {date_series.min().date()} to {date_series.max().date()}")

    print(f"[Info] Sector columns: {', '.join(sector_columns) if sector_columns else 'none'}")
    print(f"[Info] Fundamental columns: {', '.join(fundamental_columns) if fundamental_columns else 'none'}")
    print(f"[Info] Macro columns: {', '.join(macro_columns) if macro_columns else 'none'}")

    if fundamental_columns:
        coverage = {column: float(data[column].notna().mean()) for column in fundamental_columns}
        print(f"[Info] Fundamental non-null coverage: {coverage}")
    if macro_columns:
        coverage = {column: float(data[column].notna().mean()) for column in macro_columns}
        print(f"[Info] Macro non-null coverage: {coverage}")


def main() -> None:
    """执行统一 context 数据集构建流程。"""

    args = parse_args()
    start_time = time.perf_counter()

    daily_data_path = resolve_project_path(args.daily_data_path)
    fundamentals_path = resolve_project_path(args.fundamentals_path) if args.fundamentals_path else None
    macro_output_path = resolve_project_path(args.macro_output_path)
    output_path = resolve_project_path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_steps = 1
    if fundamentals_path is not None:
        total_steps += 1
    if not args.skip_macro:
        total_steps += 1
    progress = create_progress_bar(total=total_steps, description="Context data workflow", enabled=not args.no_progress)

    working_path = daily_data_path

    # 1. 可选：先把本地 FMP 基本面按披露日期合并到日频数据。
    if fundamentals_path is not None:
        if not fundamentals_path.exists():
            raise FileNotFoundError(f"Fundamentals CSV does not exist: {fundamentals_path}")
        stage_start = time.perf_counter()
        temporary_fundamental_path = output_path.with_name(f"{output_path.stem}__fundamental_tmp.csv")
        merge_fundamentals_csv_into_daily_csv(
            daily_data_path=working_path,
            fundamentals_path=fundamentals_path,
            output_path=temporary_fundamental_path,
        )
        working_path = temporary_fundamental_path
        progress.update(1)
        progress.set_postfix_str(f"fundamentals {format_duration(time.perf_counter() - stage_start)}")

    # 2. 可选：下载或复用宏观代理变量，再按日期 forward-fill 合并。
    if not args.skip_macro:
        stage_start = time.perf_counter()
        if not args.skip_macro_download:
            macro_df = download_macro_proxy_data(
                start_date=args.start_date,
                end_date=args.end_date,
                show_progress=not args.no_progress,
            )
            save_macro_proxy_data(macro_df, macro_output_path)
        elif not macro_output_path.exists():
            raise FileNotFoundError(f"Macro CSV does not exist: {macro_output_path}")

        merge_macro_proxy_into_daily_data(
            daily_data_path=working_path,
            macro_data_path=macro_output_path,
            output_path=output_path,
        )
        working_path = output_path
        progress.update(1)
        progress.set_postfix_str(f"macro {format_duration(time.perf_counter() - stage_start)}")

    # 3. 如果用户只做基本面、不做宏观，需要把中间文件落到最终 output。
    if args.skip_macro:
        pd.read_csv(working_path).to_csv(output_path, index=False)
        progress.update(1)
        progress.set_postfix_str("final copy")
    else:
        progress.update(1)
        progress.set_postfix_str("final ready")

    progress.close()
    summarize_context_file(output_path)
    print(f"[Info] Context data workflow finished in {format_duration(time.perf_counter() - start_time)}")


if __name__ == "__main__":
    main()
