"""Run long-short portfolio backtest grid.

这个入口只读取已经保存好的预测结果，不重新训练模型。

公开默认主线：

- 数据：`data/us_large_cap_300_daily.csv`
- 信号：`outputs/public_us300_release_v1/test_predictions_with_actual.csv`
- 组合：Top-K long / Bottom-K short
- 网格：10/20 天持有、Top-K 10/20/30/50、成本 5/10/20/50 bps、行业中性开/关

这样设计的原因是：训练层回答“模型有没有预测能力”，这个脚本回答
“预测分数扣掉成本后能不能转成组合收益”。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.long_short_backtest import LongShortBacktestConfig, dataframe_to_markdown, run_long_short_backtest, slugify_name
from src.data_loader import PRICE_ADJUSTMENT_MODES
from src.portfolio import load_market_snapshot_frame
from src.project_paths import PROJECT_ROOT, resolve_project_path
from src.provenance import (
    build_data_fingerprint,
    collect_environment,
    project_relative_path,
    sanitize_arguments,
    sanitize_command,
    utc_now_iso,
    write_run_manifest,
)
from src.runtime_config import DEFAULT_PRIMARY_DATA_PATH

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional for the CLI display only.
    tqdm = None


DEFAULT_PREDICTIONS_PATH = "outputs/public_us300_release_v1/test_predictions_with_actual.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Top-K long / Bottom-K short portfolio backtest grid.")
    parser.add_argument(
        "--predictions-paths",
        nargs="+",
        default=[DEFAULT_PREDICTIONS_PATH],
        help="一个或多个包含 date/instrument_id/y/predicted_y 的预测结果文件。",
    )
    parser.add_argument(
        "--run-names",
        nargs="+",
        default=None,
        help="可选运行名，数量必须和 predictions-paths 一致。",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=DEFAULT_PRIMARY_DATA_PATH,
        help="用于读取 close、sector、market_cap 的市场数据文件。",
    )
    parser.add_argument(
        "--output-root-dir",
        type=str,
        default="outputs/long_short_backtest",
        help="多空组合回测输出目录。",
    )
    parser.add_argument(
        "--hold-days-list",
        nargs="+",
        type=int,
        default=[10, 20],
        help="信号日 forward-return horizon 网格，例如 10 20；实际持有天数另行报告。",
    )
    parser.add_argument(
        "--top-k-list",
        nargs="+",
        type=int,
        default=[10, 20, 30, 50],
        help="每侧持仓数量网格，例如 10 20 30 50。",
    )
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
        help="组合构建方式：全市场排序或行业内排序。",
    )
    parser.add_argument(
        "--step-days",
        type=int,
        default=None,
        help="信号日之间的调仓间隔。默认等于 hold_days；signal_horizon 且有延迟时可能留出空仓日。",
    )
    parser.add_argument(
        "--signal-delay-days",
        type=int,
        default=1,
        help="信号日之后延迟几个交易日执行。默认 1，避免当天收盘信号当天成交的假设。",
    )
    parser.add_argument(
        "--holding-clock",
        choices=["signal_horizon", "execution_horizon"],
        default="signal_horizon",
        help=(
            "持有期时钟。signal_horizon 从信号日计算并与 y_10d 终点对齐；"
            "execution_horizon 表示延迟执行后再完整持有 N 日，只用于历史敏感性对照。"
        ),
    )
    parser.add_argument(
        "--borrow-cost-bps",
        type=float,
        default=0.0,
        help="简化借券费敏感性，按空头名义金额和持有天数线性计提的年化 bps；默认 0。",
    )
    parser.add_argument(
        "--price-adjustment-mode",
        choices=list(PRICE_ADJUSTMENT_MODES),
        default="vendor_adjusted",
        help="组合收益使用 vendor-adjusted close 或原始 raw close；必须与训练口径一致。",
    )
    return parser.parse_args()


def progress_iter(iterable: Iterable, *, total: int, description: str) -> Iterable:
    """给网格运行加进度条；如果 tqdm 不可用则退回普通迭代。"""

    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=description)


def infer_run_name(predictions_path: Path) -> str:
    """从预测文件路径推断运行名。"""

    parent = predictions_path.parent.name
    if parent:
        return slugify_name(parent)
    return slugify_name(predictions_path.stem)


def format_grid_value(value: float | int) -> str:
    """把网格参数转成稳定的文件夹片段。"""

    if isinstance(value, int) or float(value).is_integer():
        return str(int(value))
    return f"{float(value):.2f}".replace(".", "p")


def build_run_dir_name(
    base_run_name: str,
    *,
    hold_days: int,
    top_k: int,
    cost_bps: float,
    neutral_mode: str,
    holding_clock: str,
) -> str:
    """构造单个网格配置的输出文件夹名。"""

    return slugify_name(
        f"{base_run_name}__ls__hold{hold_days}d__top{top_k}"
        f"__cost{format_grid_value(cost_bps)}bps__{neutral_mode}__{holding_clock}"
    )


def write_grid_summary(output_root_dir: Path, summary_df: pd.DataFrame) -> None:
    """写出完整网格汇总。"""

    output_root_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_root_dir / "portfolio_grid_summary.csv", index=False)

    if summary_df.empty:
        text = "# Long-Short Portfolio Grid Summary\n\n_No runs were completed._\n"
    else:
        sort_columns = ["portfolio_sharpe", "portfolio_calmar", "portfolio_max_drawdown"]
        ranked = summary_df.sort_values(sort_columns, ascending=[False, False, False]).reset_index(drop=True)
        short_sample_count = int(summary_df["is_short_sample_warning"].fillna(False).sum())
        text = f"""# Long-Short Portfolio Grid Summary

## 1. Scope

This report compares Top-K long / Bottom-K short portfolio diagnostics across holding periods,
Top-K values, transaction costs, and neutralization modes.

- Total runs: `{len(summary_df)}`
- Short-sample warning runs: `{short_sample_count}`
- Main caveat: current OOS window is short, so Sharpe and Calmar should be treated as diagnostics.

## 2. Top Runs By Sharpe

{dataframe_to_markdown(ranked.head(20))}

## 3. Full Grid

{dataframe_to_markdown(summary_df)}
"""

    (output_root_dir / "portfolio_grid_summary.md").write_text(text, encoding="utf-8")


def write_best_config_report(output_root_dir: Path, summary_df: pd.DataFrame) -> None:
    """写出最佳配置报告。

    这里不会只按收益排序。组合回测至少要同时看收益、风险、回撤和成本。
    """

    if summary_df.empty:
        (output_root_dir / "best_config_report.md").write_text(
            "# Best Long-Short Config Report\n\n_No completed runs._\n",
            encoding="utf-8",
        )
        return

    by_sharpe = summary_df.sort_values("portfolio_sharpe", ascending=False).head(10)
    by_calmar = summary_df.sort_values("portfolio_calmar", ascending=False).head(10)
    by_drawdown = summary_df.sort_values("portfolio_max_drawdown", ascending=False).head(10)
    by_cost = summary_df.sort_values("average_turnover_cost_bps", ascending=True).head(10)

    report = f"""# Best Long-Short Config Report

## 1. How To Read This

This report intentionally ranks the grid from multiple angles.
If one configuration has high Sharpe but extreme turnover or very few rebalance periods,
it should not be treated as robust evidence.

## 2. Best By Sharpe

{dataframe_to_markdown(by_sharpe)}

## 3. Best By Calmar

{dataframe_to_markdown(by_calmar)}

## 4. Lowest Drawdown

{dataframe_to_markdown(by_drawdown)}

## 5. Lowest Turnover Cost

{dataframe_to_markdown(by_cost)}

## 6. Required Caveat

Current OOS data mostly covers early 2026. For 20-day holding windows,
the number of independent rebalance periods can be very small.
The right interpretation is portfolio feasibility diagnostic, not live trading proof.
"""

    (output_root_dir / "best_config_report.md").write_text(report, encoding="utf-8")


def validate_args(args: argparse.Namespace, predictions_paths: list[Path]) -> None:
    if args.run_names is not None and len(args.run_names) != len(predictions_paths):
        raise ValueError("run_names length must match predictions_paths length.")
    for hold_days in args.hold_days_list:
        if hold_days <= 0:
            raise ValueError("hold-days-list values must be positive.")
    for top_k in args.top_k_list:
        if top_k <= 0:
            raise ValueError("top-k-list values must be positive.")
    for cost_bps in args.cost_bps_list:
        if cost_bps < 0:
            raise ValueError("cost-bps-list values must be non-negative.")
    if args.step_days is not None and args.step_days <= 0:
        raise ValueError("step-days must be positive when provided.")
    if args.signal_delay_days < 0:
        raise ValueError("signal-delay-days must be non-negative.")
    if args.holding_clock == "signal_horizon" and any(
        args.signal_delay_days >= hold_days for hold_days in args.hold_days_list
    ):
        raise ValueError(
            "signal-delay-days must be smaller than every hold-days-list value "
            "when holding-clock=signal_horizon."
        )
    if args.borrow_cost_bps < 0:
        raise ValueError("borrow-cost-bps must be non-negative.")


def main() -> None:
    run_started_at_utc = utc_now_iso()
    run_started_at_perf = time.perf_counter()
    args = parse_args()
    predictions_paths = [resolve_project_path(path_like) for path_like in args.predictions_paths]
    validate_args(args, predictions_paths)

    data_path = resolve_project_path(args.data_path)
    output_root_dir = resolve_project_path(args.output_root_dir)
    market_snapshot_df = load_market_snapshot_frame(
        data_path,
        price_adjustment_mode=args.price_adjustment_mode,
    )

    grid_items: list[tuple[Path, str, int, int, float, str]] = []
    for index, predictions_path in enumerate(predictions_paths):
        base_run_name = args.run_names[index] if args.run_names is not None else infer_run_name(predictions_path)
        for hold_days in args.hold_days_list:
            for top_k in args.top_k_list:
                for cost_bps in args.cost_bps_list:
                    for neutral_mode in args.neutral_modes:
                        grid_items.append((predictions_path, base_run_name, hold_days, top_k, cost_bps, neutral_mode))

    summary_rows: list[dict[str, Any]] = []
    detailed_run_artifacts: dict[str, dict[str, Any]] = {}
    for predictions_path, base_run_name, hold_days, top_k, cost_bps, neutral_mode in progress_iter(
        grid_items,
        total=len(grid_items),
        description="Long-short backtest grid",
    ):
        step_days = args.step_days if args.step_days is not None else hold_days
        run_dir_name = build_run_dir_name(
            base_run_name,
            hold_days=hold_days,
            top_k=top_k,
            cost_bps=cost_bps,
            neutral_mode=neutral_mode,
            holding_clock=args.holding_clock,
        )
        config = LongShortBacktestConfig(
            run_name=run_dir_name,
            predictions_path=predictions_path,
            data_path=data_path,
            output_dir=output_root_dir / run_dir_name,
            hold_days=hold_days,
            step_days=step_days,
            top_k=top_k,
            cost_bps=cost_bps,
            neutral_mode=neutral_mode,
            signal_delay_days=args.signal_delay_days,
            holding_clock=args.holding_clock,
            borrow_cost_bps=args.borrow_cost_bps,
            price_adjustment_mode=args.price_adjustment_mode,
        )
        result = run_long_short_backtest(config=config, market_snapshot_df=market_snapshot_df)
        metrics = result["metrics"]
        detailed_run_artifacts[run_dir_name] = {
            filename: build_data_fingerprint(config.output_dir / filename, PROJECT_ROOT)
            for filename in [
                "daily_returns.csv",
                "portfolio_weights.csv",
                "turnover_cost.csv",
                "skipped_trades.csv",
                "sector_exposure.csv",
                "extreme_return_days.csv",
                "position_daily_contributions.csv",
                "instrument_return_attribution.csv",
                "portfolio_metrics.json",
                "portfolio_report.md",
            ]
        }
        summary_rows.append(
            {
                "run_name": run_dir_name,
                "base_run_name": base_run_name,
                "predictions_path": project_relative_path(predictions_path, PROJECT_ROOT),
                "hold_days": metrics.get("hold_days"),
                "step_days": metrics.get("step_days"),
                "signal_delay_days": metrics.get("signal_delay_days"),
                "holding_clock": metrics.get("holding_clock"),
                "effective_holding_days": metrics.get("effective_holding_days"),
                "top_k": metrics.get("top_k"),
                "cost_bps": metrics.get("cost_bps"),
                "borrow_cost_bps": metrics.get("borrow_cost_bps"),
                "price_adjustment_mode": metrics.get("price_adjustment_mode"),
                "neutral_mode": metrics.get("neutral_mode"),
                "daily_count": metrics.get("daily_count"),
                "invested_day_count": metrics.get("invested_day_count"),
                "cash_day_count": metrics.get("cash_day_count"),
                "rebalance_count": metrics.get("rebalance_count"),
                "portfolio_total_return": metrics.get("portfolio_total_return"),
                "portfolio_annualized_return": metrics.get("portfolio_annualized_return"),
                "portfolio_annualized_vol": metrics.get("portfolio_annualized_vol"),
                "portfolio_sharpe": metrics.get("portfolio_sharpe"),
                "portfolio_max_drawdown": metrics.get("portfolio_max_drawdown"),
                "portfolio_calmar": metrics.get("portfolio_calmar"),
                "sharpe_definition": metrics.get("sharpe_definition"),
                "hit_ratio": metrics.get("hit_ratio"),
                "first_half_total_return": metrics.get("first_half_total_return"),
                "second_half_total_return": metrics.get("second_half_total_return"),
                "monthly_period_count": metrics.get("monthly_period_count"),
                "positive_month_ratio": metrics.get("positive_month_ratio"),
                "best_month_return": metrics.get("best_month_return"),
                "worst_month_return": metrics.get("worst_month_return"),
                "top_5_net_return_days_simple_sum": metrics.get("top_5_net_return_days_simple_sum"),
                "bottom_5_net_return_days_simple_sum": metrics.get("bottom_5_net_return_days_simple_sum"),
                "top_5_instrument_abs_contribution_share": metrics.get(
                    "top_5_instrument_abs_contribution_share"
                ),
                "max_abs_selected_stock_daily_return": metrics.get(
                    "max_abs_selected_stock_daily_return"
                ),
                "selected_stock_return_abs_gt_20pct_count": metrics.get(
                    "selected_stock_return_abs_gt_20pct_count"
                ),
                "selected_stock_return_abs_gt_50pct_count": metrics.get(
                    "selected_stock_return_abs_gt_50pct_count"
                ),
                "average_gross_turnover": metrics.get("average_gross_turnover"),
                "average_net_turnover": metrics.get("average_net_turnover"),
                "average_turnover_cost_bps": metrics.get("average_turnover_cost_bps"),
                "total_turnover_cost": metrics.get("total_turnover_cost"),
                "turnover_accounting": metrics.get("turnover_accounting"),
                "skipped_incomplete_return_path_count": metrics.get(
                    "skipped_incomplete_return_path_count"
                ),
                "total_borrow_cost": metrics.get("total_borrow_cost"),
                "average_gross_exposure": metrics.get("average_gross_exposure"),
                "average_net_exposure": metrics.get("average_net_exposure"),
                "average_max_abs_sector_net_weight": metrics.get("average_max_abs_sector_net_weight"),
                "benchmark_total_return": metrics.get("benchmark_total_return"),
                "relative_wealth_vs_equal_weight_long_only": metrics.get(
                    "relative_wealth_vs_equal_weight_long_only"
                ),
                "is_short_sample_warning": metrics.get("is_short_sample_warning"),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    write_grid_summary(output_root_dir, summary_df)
    write_best_config_report(output_root_dir, summary_df)

    # 组合结果与训练结果属于两层证据，因此单独记录回测输入、假设、代码版本
    # 和运行环境。公开报告可以据此确认预测文件、行情文件和价格口径一致。
    prediction_fingerprints = [
        build_data_fingerprint(path, PROJECT_ROOT)
        for path in predictions_paths
    ]
    backtest_manifest = {
        "schema_version": 1,
        "status": "completed",
        "started_at_utc": run_started_at_utc,
        "finished_at_utc": utc_now_iso(),
        "total_runtime_seconds": time.perf_counter() - run_started_at_perf,
        "command": sanitize_command([Path(sys.executable).name, *sys.argv]),
        "arguments": sanitize_arguments(vars(args)),
        "market_data": build_data_fingerprint(data_path, PROJECT_ROOT),
        "prediction_inputs": prediction_fingerprints,
        "artifacts": {
            "portfolio_grid_summary": build_data_fingerprint(
                output_root_dir / "portfolio_grid_summary.csv",
                PROJECT_ROOT,
            ),
            "detailed_runs": detailed_run_artifacts,
        },
        "price_adjustment_mode": args.price_adjustment_mode,
        "holding_clock": args.holding_clock,
        "effective_holding_days_by_horizon": {
            str(int(hold_days)): int(
                hold_days - args.signal_delay_days
                if args.holding_clock == "signal_horizon"
                else hold_days
            )
            for hold_days in args.hold_days_list
        },
        "grid_run_count": int(len(summary_df)),
        "metric_definitions": {
            "portfolio_sharpe": (
                "mean daily net return divided by sample standard deviation of daily net return, "
                "multiplied by sqrt(252), with zero risk-free rate"
            ),
            "portfolio_total_return": "compound net return over the observed backtest dates",
            "daily_count": (
                "all market dates from first execution through final liquidation, including cash dates"
            ),
            "relative_wealth_vs_equal_weight_long_only": (
                "portfolio_nav / equal_weight_long_only_nav - 1; context only, not matched-risk alpha"
            ),
            "cost_bps": "all-in proportional friction sensitivity per traded notional",
            "average_gross_turnover": (
                "mean complete sleeve round-trip traded notional (entry plus liquidation) after "
                "scaling each overlapping sleeve by its portfolio capital allocation; cross-sleeve "
                "netting is not assumed"
            ),
            "total_turnover_cost": (
                "sum of transaction costs actually deducted from the daily portfolio return ledger"
            ),
            "turnover_accounting": (
                "each independent sleeve pays entry and liquidation turnover, scaled by sleeve capital; "
                "cross-sleeve order netting is not assumed"
            ),
            "skipped_incomplete_return_path_count": (
                "number of candidate sleeves excluded because at least one selected stock lacked the "
                "complete requested close-return path"
            ),
            "holding_clock": (
                "signal_horizon measures the endpoint from the signal date; execution_horizon "
                "starts a full holding period after delayed execution and is sensitivity-only"
            ),
            "top_5_instrument_abs_contribution_share": (
                "share of total absolute gross position-day contribution attributable to the five "
                "largest instruments; concentration diagnostic only"
            ),
        },
        "environment": collect_environment(PROJECT_ROOT),
    }
    write_run_manifest(output_root_dir / "backtest_run_manifest.json", backtest_manifest)

    if not summary_df.empty:
        best = summary_df.sort_values("portfolio_sharpe", ascending=False).iloc[0]
        print(
            "[LongShort] best_by_sharpe="
            f"{best['run_name']} sharpe={best['portfolio_sharpe']:.4f} "
            f"calmar={best['portfolio_calmar']:.4f} "
            f"max_dd={best['portfolio_max_drawdown']:.4f} "
            f"turnover_cost_bps={best['average_turnover_cost_bps']:.4f}"
        )
        print(f"[LongShort] summary: {output_root_dir / 'portfolio_grid_summary.csv'}")
        print(f"[LongShort] manifest: {output_root_dir / 'backtest_run_manifest.json'}")


if __name__ == "__main__":
    main()
