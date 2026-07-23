"""宏观 / 市场代理数据下载与合并入口。

运行这个脚本后，会得到一个带宏观代理列的新 daily CSV，例如：

```bash
python main_macro_data.py \
  --data-path data/us_large_cap_300_daily.csv \
  --merged-output-path data/us_large_cap_300_with_macro.csv
```

然后主训练脚本只要读取这个新 CSV，`src/feature_generator.py` 就会自动把
`vix`、`sp500_return`、`treasury_10y` 等列转换成 `macro_*` 因子。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from src.macro_data import (
    download_macro_proxy_data,
    merge_macro_proxy_into_daily_data,
    save_macro_proxy_data,
)
from src.project_paths import resolve_project_path
from src.progress import create_progress_bar, format_duration
from src.runtime_config import DEFAULT_PRIMARY_DATA_PATH, DEFAULT_SAMPLE_START_DATE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download macro proxy data and merge it into daily stock CSV.")
    parser.add_argument("--data-path", default=DEFAULT_PRIMARY_DATA_PATH, help="原始股票日线 CSV。")
    parser.add_argument("--macro-output-path", default="data/macro/macro_proxy_daily.csv", help="宏观代理变量 CSV 输出路径。")
    parser.add_argument("--merged-output-path", default="data/us_large_cap_300_with_macro.csv", help="合并后股票面板输出路径。")
    parser.add_argument("--start-date", default=DEFAULT_SAMPLE_START_DATE, help="宏观数据下载起点。")
    parser.add_argument("--end-date", default=None, help="宏观数据下载终点；默认下载到 yfinance 可用最新日期。")
    parser.add_argument("--merge-only", action="store_true", help="只做合并，不重新下载宏观数据。")
    parser.add_argument("--no-progress", action="store_true", help="关闭进度条。")
    return parser.parse_args()


def summarize_outputs(macro_path: Path, merged_path: Path) -> None:
    """打印最小摘要，确认下载/合并是否真的产生了可用列。"""

    macro_df = pd.read_csv(macro_path)
    merged_df = pd.read_csv(merged_path)
    macro_columns = [column for column in macro_df.columns if column != "date"]
    merged_macro_columns = [column for column in macro_columns if column in merged_df.columns]

    print(f"[Info] Macro proxy CSV: {macro_path}")
    print(f"[Info] Merged daily CSV: {merged_path}")
    print(f"[Info] Macro rows: {len(macro_df):,}")
    print(f"[Info] Macro columns: {', '.join(macro_columns)}")
    print(f"[Info] Merged rows: {len(merged_df):,}")
    print(f"[Info] Macro columns merged into daily panel: {', '.join(merged_macro_columns)}")
    if "date" in merged_df.columns and len(merged_df) > 0:
        date_series = pd.to_datetime(merged_df["date"])
        print(f"[Info] Merged date range: {date_series.min().date()} to {date_series.max().date()}")


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()

    data_path = resolve_project_path(args.data_path)
    macro_output_path = resolve_project_path(args.macro_output_path)
    merged_output_path = resolve_project_path(args.merged_output_path)

    progress = create_progress_bar(total=2, description="Macro data workflow", enabled=not args.no_progress)

    if not args.merge_only:
        stage_start = time.perf_counter()
        macro_df = download_macro_proxy_data(
            start_date=args.start_date,
            end_date=args.end_date,
            show_progress=not args.no_progress,
        )
        save_macro_proxy_data(macro_df, macro_output_path)
        progress.set_postfix_str(
            f"download_macro done | stage={format_duration(time.perf_counter() - stage_start)} "
            f"| total={format_duration(time.perf_counter() - start_time)}"
        )
        progress.update(1)
    else:
        if not macro_output_path.exists():
            raise FileNotFoundError(f"Macro CSV does not exist: {macro_output_path}")
        progress.set_postfix_str("download skipped")
        progress.update(1)

    stage_start = time.perf_counter()
    merge_macro_proxy_into_daily_data(
        daily_data_path=data_path,
        macro_data_path=macro_output_path,
        output_path=merged_output_path,
    )
    progress.set_postfix_str(
        f"merge_macro done | stage={format_duration(time.perf_counter() - stage_start)} "
        f"| total={format_duration(time.perf_counter() - start_time)}"
    )
    progress.update(1)
    if progress is not None:
        progress.close()

    summarize_outputs(macro_output_path, merged_output_path)
    print(f"[Info] Macro data workflow finished in {format_duration(time.perf_counter() - start_time)}")


if __name__ == "__main__":
    main()
