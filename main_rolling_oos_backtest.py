"""Run rolling OOS long-short backtests.

这个脚本是 `main_long_short_backtest.py` 的外层包装。

单窗口回测回答：

```text
在一个 OOS 区间里，Top-K long / Bottom-K short 组合表现如何？
```

滚动 OOS 回测回答：

```text
这个组合在不同 OOS 子窗口里是否稳定？
```

默认公开窗口：

- full: 2025-06-01 到最新可评估日期；
- 3m: 3 个月窗口，按月滚动；
- 6m: 6 个月窗口，每 2 个月滚动；
- 12m: 12 个月窗口。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.long_short_backtest import (
    LongShortBacktestConfig,
    dataframe_to_markdown,
    run_long_short_backtest,
    slugify_name,
)
from src.pdf_report import PdfSection, write_pdf_report
from src.portfolio import load_market_snapshot_frame, load_prediction_frame
from src.project_paths import resolve_project_path
from src.runtime_config import DEFAULT_OOS_START_DATE, DEFAULT_PRIMARY_DATA_PATH

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm 只影响显示，不影响计算。
    tqdm = None


DEFAULT_PREDICTIONS_PATH = (
    "outputs/public_us300_release_v1/"
    "test_predictions_with_actual.csv"
)


@dataclass(frozen=True)
class OOSWindow:
    """一个 OOS 子窗口。"""

    window_id: str
    window_mode: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    calendar_months: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full and rolling OOS long-short backtests.")
    parser.add_argument("--predictions-path", default=DEFAULT_PREDICTIONS_PATH, help="预测结果 CSV。")
    parser.add_argument("--data-path", default=DEFAULT_PRIMARY_DATA_PATH, help="行情数据 CSV。")
    parser.add_argument("--output-root-dir", default="outputs/rolling_oos_backtest_us300", help="输出目录。")
    parser.add_argument("--oos-start-date", default=DEFAULT_OOS_START_DATE, help="OOS 起始日期。")
    parser.add_argument(
        "--window-modes",
        nargs="+",
        choices=["full", "3m", "6m", "12m"],
        default=["full", "3m", "6m", "12m"],
        help="要运行的 OOS 窗口类型。",
    )
    parser.add_argument("--include-partial-final-window", action="store_true", help="是否保留最后一个不满窗口。")
    parser.add_argument("--base-run-name", default=None, help="用于输出文件夹的基础运行名。")
    parser.add_argument(
        "--hold-days-list",
        nargs="+",
        type=int,
        default=[10, 20],
        help="信号日 forward-return horizon 网格；实际可执行持有天数单独输出。",
    )
    parser.add_argument(
        "--step-days-list",
        nargs="+",
        type=int,
        default=None,
        help=(
            "信号日调仓步长网格。默认等于 hold_days；"
            "signal_horizon 且存在执行延迟时，实际持有天数会更短。"
        ),
    )
    parser.add_argument("--top-k-list", nargs="+", type=int, default=[10, 20, 30, 50], help="Top-K 网格。")
    parser.add_argument(
        "--cost-bps-list",
        nargs="+",
        type=float,
        default=[5.0, 10.0, 20.0, 50.0],
        help="单边交易成本网格，单位 bps。",
    )
    parser.add_argument(
        "--neutral-modes",
        nargs="+",
        choices=["unconstrained", "sector_neutral"],
        default=["unconstrained", "sector_neutral"],
        help="组合构建方式。",
    )
    parser.add_argument("--signal-delay-days", type=int, default=1, help="信号日后延迟几个交易日执行。")
    parser.add_argument(
        "--holding-clock",
        choices=["signal_horizon", "execution_horizon"],
        default="signal_horizon",
        help="默认从信号日计算目标 horizon；旧执行日时钟只用于敏感性对照。",
    )
    parser.add_argument("--borrow-cost-bps", type=float, default=0.0, help="做空借券成本占位。")
    return parser.parse_args()


def progress_iter(iterable: Iterable, *, total: int, desc: str) -> Iterable:
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc)


def next_window_start(current_start: pd.Timestamp, step_months: int) -> pd.Timestamp:
    """按日历月移动窗口起点。"""

    return pd.Timestamp(current_start) + pd.DateOffset(months=step_months)


def window_end_from_months(start_date: pd.Timestamp, months: int) -> pd.Timestamp:
    """计算固定月数窗口的自然结束日。"""

    return pd.Timestamp(start_date) + pd.DateOffset(months=months) - pd.Timedelta(days=1)


def build_windows(
    *,
    min_start: pd.Timestamp,
    max_date: pd.Timestamp,
    window_modes: list[str],
    include_partial_final_window: bool,
) -> list[OOSWindow]:
    """根据用户指定的 rolling 规则生成窗口列表。"""

    windows: list[OOSWindow] = []

    if "full" in window_modes:
        windows.append(
            OOSWindow(
                window_id=f"full_{min_start:%Y%m%d}_{max_date:%Y%m%d}",
                window_mode="full",
                start_date=min_start,
                end_date=max_date,
                calendar_months=None,
            )
        )

    rolling_specs = {
        "3m": {"months": 3, "step_months": 1},
        "6m": {"months": 6, "step_months": 2},
        "12m": {"months": 12, "step_months": 12},
    }

    for mode in ["3m", "6m", "12m"]:
        if mode not in window_modes:
            continue
        months = int(rolling_specs[mode]["months"])
        step_months = int(rolling_specs[mode]["step_months"])
        start_date = min_start
        while start_date <= max_date:
            natural_end = window_end_from_months(start_date, months)
            if natural_end <= max_date:
                end_date = natural_end
            elif (include_partial_final_window or (mode == "12m" and start_date == min_start)) and start_date < max_date:
                end_date = max_date
            else:
                break

            windows.append(
                OOSWindow(
                    window_id=f"{mode}_{start_date:%Y%m%d}_{end_date:%Y%m%d}",
                    window_mode=mode,
                    start_date=start_date,
                    end_date=end_date,
                    calendar_months=months,
                )
            )
            start_date = next_window_start(start_date, step_months)

    return windows


def format_grid_value(value: float | int) -> str:
    if isinstance(value, int) or float(value).is_integer():
        return str(int(value))
    return f"{float(value):.2f}".replace(".", "p")


def build_run_dir_name(
    *,
    base_run_name: str,
    window: OOSWindow,
    hold_days: int,
    step_days: int,
    top_k: int,
    cost_bps: float,
    neutral_mode: str,
    holding_clock: str,
) -> str:
    return slugify_name(
        f"{base_run_name}__{window.window_id}"
        f"__hold{hold_days}d__step{step_days}d__top{top_k}"
        f"__cost{format_grid_value(cost_bps)}bps__{neutral_mode}__{holding_clock}"
    )


def write_window_prediction_file(
    output_dir: Path,
    prediction_df: pd.DataFrame,
    window: OOSWindow,
) -> Path:
    """把某个窗口内的预测结果落盘，供单窗口回测函数读取。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    window_prediction = prediction_df[
        (prediction_df["date"] >= window.start_date) & (prediction_df["date"] <= window.end_date)
    ].copy()
    path = output_dir / f"{window.window_id}_predictions.csv"
    window_prediction.to_csv(path, index=False)
    return path


def summarize_window_predictions(prediction_df: pd.DataFrame, window: OOSWindow) -> dict[str, Any]:
    window_prediction = prediction_df[
        (prediction_df["date"] >= window.start_date) & (prediction_df["date"] <= window.end_date)
    ].copy()
    if window_prediction.empty:
        return {
            "window_prediction_rows": 0,
            "window_instruments": 0,
            "prediction_min_date": "",
            "prediction_max_date": "",
        }
    return {
        "window_prediction_rows": int(len(window_prediction)),
        "window_instruments": int(window_prediction["instrument_id"].nunique()),
        "prediction_min_date": str(window_prediction["date"].min().date()),
        "prediction_max_date": str(window_prediction["date"].max().date()),
    }


def run_rolling_grid(args: argparse.Namespace) -> pd.DataFrame:
    predictions_path = resolve_project_path(args.predictions_path)
    data_path = resolve_project_path(args.data_path)
    output_root_dir = resolve_project_path(args.output_root_dir)
    output_root_dir.mkdir(parents=True, exist_ok=True)

    if args.signal_delay_days < 0:
        raise ValueError("signal-delay-days must be non-negative.")
    if args.holding_clock == "signal_horizon" and any(
        args.signal_delay_days >= hold_days for hold_days in args.hold_days_list
    ):
        raise ValueError(
            "signal-delay-days must be smaller than every hold-days-list value "
            "when holding-clock=signal_horizon."
        )

    prediction_df = load_prediction_frame(predictions_path)
    market_snapshot_df = load_market_snapshot_frame(data_path)

    oos_start = pd.Timestamp(args.oos_start_date)
    prediction_df = prediction_df[prediction_df["date"] >= oos_start].copy()
    if prediction_df.empty:
        raise ValueError(f"No prediction rows on or after OOS start date: {oos_start.date()}")

    max_prediction_date = pd.Timestamp(prediction_df["date"].max())
    windows = build_windows(
        min_start=oos_start,
        max_date=max_prediction_date,
        window_modes=list(args.window_modes),
        include_partial_final_window=bool(args.include_partial_final_window),
    )
    if not windows:
        raise ValueError("No OOS windows were generated. Check date range and window mode settings.")

    base_run_name = args.base_run_name or slugify_name(predictions_path.parent.name)
    window_prediction_dir = output_root_dir / "_window_predictions"

    grid_items: list[tuple[OOSWindow, int, int, int, float, str]] = []
    for window in windows:
        for hold_days in args.hold_days_list:
            step_days_values = args.step_days_list if args.step_days_list is not None else [hold_days]
            for step_days in step_days_values:
                if step_days > hold_days:
                    # step > hold 会导致持仓之间有空档。这里跳过，保持研究口径清晰。
                    continue
                for top_k in args.top_k_list:
                    for cost_bps in args.cost_bps_list:
                        for neutral_mode in args.neutral_modes:
                            grid_items.append((window, hold_days, step_days, top_k, cost_bps, neutral_mode))

    summary_rows: list[dict[str, Any]] = []
    window_prediction_paths: dict[str, Path] = {}
    for window, hold_days, step_days, top_k, cost_bps, neutral_mode in progress_iter(
        grid_items,
        total=len(grid_items),
        desc="Rolling OOS long-short grid",
    ):
        if window.window_id not in window_prediction_paths:
            window_prediction_paths[window.window_id] = write_window_prediction_file(
                window_prediction_dir,
                prediction_df,
                window,
            )

        run_dir_name = build_run_dir_name(
            base_run_name=base_run_name,
            window=window,
            hold_days=hold_days,
            step_days=step_days,
            top_k=top_k,
            cost_bps=cost_bps,
            neutral_mode=neutral_mode,
            holding_clock=args.holding_clock,
        )

        # 关键防线：market data 也裁到当前窗口结束日。
        # 这样单窗口回测不会把持仓收益算到窗口外面。
        window_market_df = market_snapshot_df[
            (market_snapshot_df["date"] >= window.start_date) & (market_snapshot_df["date"] <= window.end_date)
        ].copy()

        config = LongShortBacktestConfig(
            run_name=run_dir_name,
            predictions_path=window_prediction_paths[window.window_id],
            data_path=data_path,
            output_dir=output_root_dir / window.window_id / run_dir_name,
            hold_days=int(hold_days),
            step_days=int(step_days),
            top_k=int(top_k),
            cost_bps=float(cost_bps),
            neutral_mode=str(neutral_mode),
            signal_delay_days=int(args.signal_delay_days),
            holding_clock=args.holding_clock,
            borrow_cost_bps=float(args.borrow_cost_bps),
        )

        row: dict[str, Any] = {
            "window_id": window.window_id,
            "window_mode": window.window_mode,
            "window_start": str(window.start_date.date()),
            "window_end": str(window.end_date.date()),
            "calendar_months": window.calendar_months,
            "run_name": run_dir_name,
            "status": "ok",
            **summarize_window_predictions(prediction_df, window),
        }
        try:
            result = run_long_short_backtest(config=config, market_snapshot_df=window_market_df)
            metrics = result["metrics"]
            row.update(
                {
                    "hold_days": metrics.get("hold_days"),
                    "holding_clock": metrics.get("holding_clock"),
                    "effective_holding_days": metrics.get("effective_holding_days"),
                    "step_days": metrics.get("step_days"),
                    "top_k": metrics.get("top_k"),
                    "cost_bps": metrics.get("cost_bps"),
                    "neutral_mode": metrics.get("neutral_mode"),
                    "daily_count": metrics.get("daily_count"),
                    "rebalance_count": metrics.get("rebalance_count"),
                    "portfolio_total_return": metrics.get("portfolio_total_return"),
                    "portfolio_annualized_return": metrics.get("portfolio_annualized_return"),
                    "portfolio_annualized_vol": metrics.get("portfolio_annualized_vol"),
                    "portfolio_sharpe": metrics.get("portfolio_sharpe"),
                    "portfolio_max_drawdown": metrics.get("portfolio_max_drawdown"),
                    "portfolio_calmar": metrics.get("portfolio_calmar"),
                    "hit_ratio": metrics.get("hit_ratio"),
                    "average_gross_turnover": metrics.get("average_gross_turnover"),
                    "average_turnover_cost_bps": metrics.get("average_turnover_cost_bps"),
                    "average_max_abs_sector_net_weight": metrics.get("average_max_abs_sector_net_weight"),
                    "benchmark_total_return": metrics.get("benchmark_total_return"),
                    "excess_total_return_vs_benchmark": metrics.get("excess_total_return_vs_benchmark"),
                    "is_short_sample_warning": metrics.get("is_short_sample_warning"),
                    "error": "",
                }
            )
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = str(exc)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_root_dir / "rolling_oos_grid_summary.csv", index=False)
    return summary_df


def write_reports(output_root_dir: Path, summary_df: pd.DataFrame, args: argparse.Namespace) -> Path:
    ok_df = summary_df[summary_df["status"] == "ok"].copy()
    failed_df = summary_df[summary_df["status"] != "ok"].copy()

    if not ok_df.empty:
        top_by_sharpe = ok_df.sort_values("portfolio_sharpe", ascending=False).head(20)
        window_best = (
            ok_df.sort_values("portfolio_sharpe", ascending=False)
            .groupby("window_id", as_index=False)
            .head(1)
            .sort_values(["window_mode", "window_start"])
        )
    else:
        top_by_sharpe = pd.DataFrame()
        window_best = pd.DataFrame()

    warning_count = int(ok_df["is_short_sample_warning"].fillna(False).sum()) if not ok_df.empty else 0
    report_text = f"""# Rolling OOS Long-Short Backtest Summary

## Scope

- Predictions path: `{args.predictions_path}`
- Market data path: `{args.data_path}`
- OOS start date: `{args.oos_start_date}`
- Window modes: `{", ".join(args.window_modes)}`
- Holding clock: `{args.holding_clock}`
- Signal delay days: `{args.signal_delay_days}`
- Total grid rows: `{len(summary_df)}`
- Successful rows: `{len(ok_df)}`
- Failed rows: `{len(failed_df)}`
- Short-sample warning rows: `{warning_count}`

## Best Config Per Window

{dataframe_to_markdown(window_best)}

## Top 20 By Sharpe

{dataframe_to_markdown(top_by_sharpe)}

## Failed Rows

{dataframe_to_markdown(failed_df.head(30))}

## Interpretation

- `portfolio_total_return` 是窗口内实际累计收益，不是年化收益。
- `portfolio_sharpe` 和 `portfolio_calmar` 在短窗口里会非常不稳定，必须结合 `rebalance_count` 看。
- 3M 窗口按月滚动，6M 窗口每两个月滚动，12M 窗口用于观察更长 OOS 的稳定性。
- 每个窗口都会限制持仓结束日不能超过窗口结束日，避免窗口外收益混入。
- `signal_horizon` 从信号日计算终点；10 日目标加 1 日 close 执行延迟时，实际收益覆盖 9 个交易日。
"""
    (output_root_dir / "rolling_oos_grid_summary.md").write_text(report_text, encoding="utf-8")

    pdf_path = output_root_dir / "rolling_oos_grid_summary.pdf"
    sections = [
        PdfSection(
            "Scope",
            body=(
                f"Predictions: {args.predictions_path}\n"
                f"Data: {args.data_path}\n"
                f"OOS start: {args.oos_start_date}\n"
                f"Window modes: {', '.join(args.window_modes)}\n"
                f"Successful rows: {len(ok_df)} / {len(summary_df)}\n"
                f"Short-sample warning rows: {warning_count}"
            ),
        ),
        PdfSection("Best Config Per Window", table=window_best, max_table_rows=24),
        PdfSection("Top 20 By Sharpe", table=top_by_sharpe, max_table_rows=20),
        PdfSection("Failed Rows", table=failed_df.head(30), max_table_rows=30),
        PdfSection(
            "Reading Notes",
            body=(
                "Total return is cumulative return inside the window. Sharpe and Calmar are annualized diagnostics "
                "and become unstable when rebalance_count is small. The script clips market data to each window end, "
                "so a holding period cannot earn returns outside its OOS window."
            ),
        ),
    ]
    write_pdf_report(
        pdf_path,
        title="MyQuant Rolling OOS Long-Short Backtest",
        subtitle=f"Generated from {Path(args.predictions_path).name}",
        sections=sections,
    )
    return pdf_path


def main() -> None:
    args = parse_args()
    output_root_dir = resolve_project_path(args.output_root_dir)
    summary_df = run_rolling_grid(args)
    pdf_path = write_reports(output_root_dir, summary_df, args)
    print(f"[RollingOOS] summary_csv={output_root_dir / 'rolling_oos_grid_summary.csv'}")
    print(f"[RollingOOS] summary_pdf={pdf_path}")
    if not summary_df.empty:
        status_counts = summary_df["status"].value_counts().to_dict()
        print(f"[RollingOOS] status_counts={json.dumps(status_counts, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
