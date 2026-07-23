"""FMP 基本面下载与合并入口。

这个入口故意不碰现有训练脚本，只负责两件独立任务：

1. 下载季度基本面面板；
2. 把基本面按披露日期 merge 到现有日频价格 CSV。

这样你可以先确认：
- 数据到底有没有下对；
- merge 之后列名是否符合预期；
- 是否真的把基本面字段接进了价格面板；

然后再决定什么时候把这份新 CSV 接入训练。
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from src.fmp_fundamentals import (
    download_fmp_quarterly_fundamentals,
    infer_symbols_from_daily_data,
    merge_fundamentals_csv_into_daily_csv,
    save_fundamentals_to_csv,
)
from src.progress import create_progress_bar, format_duration
from src.project_paths import resolve_project_path


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Download FMP fundamentals and merge them into daily price CSV.")
    parser.add_argument(
        "--daily-data-path",
        type=str,
        default="data/us_large_cap_300_daily.csv",
        help="现有日频价格数据 CSV。默认从这个文件推断股票池。",
    )
    parser.add_argument(
        "--fundamentals-output-path",
        type=str,
        default="data/fmp/fundamentals_quarterly.csv",
        help="季度基本面面板输出路径。",
    )
    parser.add_argument(
        "--merged-output-path",
        type=str,
        default="data/us_large_cap_300_with_fundamentals.csv",
        help="合并后的日频价格 + 基本面输出路径。",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="显式指定 symbol 列表；不传则从 daily-data-path 推断。",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="FMP API key。若不传，则读取环境变量 FMP_API_KEY。",
    )
    parser.add_argument("--limit", type=int, default=5, help="每只股票下载多少个季度记录。免费计划下建议不超过 5。")
    parser.add_argument("--sleep-sec", type=float, default=0.2, help="两只股票之间主动 sleep 的秒数。")
    parser.add_argument("--download-only", action="store_true", help="只下载基本面，不执行 merge。")
    parser.add_argument("--merge-only", action="store_true", help="只做 merge，默认读取 fundamentals-output-path。")
    return parser.parse_args()


def print_summary(fundamentals_path, merged_path) -> None:
    """打印简短结果摘要。"""

    if fundamentals_path and fundamentals_path.exists():
        fundamentals_df = pd.read_csv(fundamentals_path)
        print(f"[Info] Fundamentals panel saved to: {fundamentals_path}")
        print(f"[Info] Fundamentals rows: {len(fundamentals_df):,}")
        print(f"[Info] Fundamentals symbols: {fundamentals_df['instrument_id'].nunique() if 'instrument_id' in fundamentals_df.columns else 'N/A'}")

    if merged_path and merged_path.exists():
        merged_df = pd.read_csv(merged_path, nrows=5)
        print(f"[Info] Merged daily file saved to: {merged_path}")
        print(f"[Info] Example merged columns: {list(merged_df.columns)}")


def main() -> None:
    """执行下载与合并流程。"""

    args = parse_args()

    if args.download_only and args.merge_only:
        raise ValueError("--download-only and --merge-only cannot be used together.")

    daily_data_path = resolve_project_path(args.daily_data_path)
    fundamentals_output_path = resolve_project_path(args.fundamentals_output_path)
    merged_output_path = resolve_project_path(args.merged_output_path)

    workflow_start = time.perf_counter()
    progress_bar = create_progress_bar(total=2, description="FMP fundamentals workflow", enabled=True)

    if args.symbols:
        symbols = sorted(dict.fromkeys(args.symbols))
    else:
        symbols = infer_symbols_from_daily_data(daily_data_path)

    print(f"[Info] Symbols to process: {len(symbols)}")
    print(f"[Info] Daily data source: {daily_data_path}")

    if not args.merge_only:
        stage_start = time.perf_counter()
        fundamentals_df = download_fmp_quarterly_fundamentals(
            symbols=symbols,
            api_key=args.api_key,
            limit=args.limit,
            sleep_sec=args.sleep_sec,
            show_progress=True,
        )
        save_fundamentals_to_csv(fundamentals_df, fundamentals_output_path)
        stage_elapsed = time.perf_counter() - stage_start
        progress_bar.update(1)
        progress_bar.set_postfix_str(f"download {format_duration(stage_elapsed)}")
    else:
        progress_bar.update(1)
        progress_bar.set_postfix_str("download skipped")

    if not args.download_only:
        stage_start = time.perf_counter()
        merge_fundamentals_csv_into_daily_csv(
            daily_data_path=daily_data_path,
            fundamentals_path=fundamentals_output_path,
            output_path=merged_output_path,
        )
        stage_elapsed = time.perf_counter() - stage_start
        progress_bar.update(1)
        progress_bar.set_postfix_str(f"merge {format_duration(stage_elapsed)}")
    else:
        progress_bar.update(1)
        progress_bar.set_postfix_str("merge skipped")

    progress_bar.close()
    total_elapsed = time.perf_counter() - workflow_start
    print(f"[Info] FMP workflow finished in {format_duration(total_elapsed)}")
    print_summary(
        fundamentals_path=fundamentals_output_path if not args.merge_only else None,
        merged_path=merged_output_path if not args.download_only else None,
    )


if __name__ == "__main__":
    main()
