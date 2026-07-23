"""组合诊断入口。

这个入口和训练主流程分层：

- `main.py` / `main_experiments.py` 负责生成预测信号；
- 这个脚本负责把已经生成的预测结果转成最小可解释组合；
- 当前重点是：权重、换手、成本、size/sector 暴露、成本后净值指标。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.portfolio import infer_hold_days, load_market_snapshot_frame, run_portfolio_diagnostic
from src.project_paths import resolve_project_path
from src.runtime_config import DEFAULT_PRIMARY_DATA_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minimal portfolio diagnostics from saved prediction files.")
    parser.add_argument(
        "--predictions-paths",
        nargs="+",
        required=True,
        help="一个或多个 `test_predictions_with_actual.csv` 路径。",
    )
    parser.add_argument(
        "--run-names",
        nargs="+",
        default=None,
        help="可选的运行名字列表，数量需要和 predictions-paths 一致。",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=DEFAULT_PRIMARY_DATA_PATH,
        help="用于读取 sector / market_cap 暴露的市场数据文件。",
    )
    parser.add_argument(
        "--output-root-dir",
        type=str,
        default="outputs/portfolio_experiments",
        help="组合诊断输出根目录。",
    )
    parser.add_argument(
        "--hold-days",
        type=int,
        default=None,
        help="持有周期。若不传，则从训练报告或路径名推断。",
    )
    parser.add_argument(
        "--step-days",
        type=int,
        default=None,
        help="调仓步长。默认与 hold-days 相同。",
    )
    parser.add_argument("--top-n", type=int, default=10, help="每次持有的股票数。")
    parser.add_argument(
        "--constraint-modes",
        nargs="+",
        choices=["unconstrained", "sector_size_constrained"],
        default=["unconstrained"],
        help="组合约束模式，可同时传多个以做对比。",
    )
    parser.add_argument(
        "--weighting-scheme",
        type=str,
        choices=["equal", "rank"],
        default="equal",
        help="权重方案：等权或按排名加权。",
    )
    parser.add_argument(
        "--max-weight",
        type=float,
        default=0.15,
        help="单票最大权重约束。若不想启用可传 1.0。",
    )
    parser.add_argument(
        "--sector-active-tolerance",
        type=float,
        default=0.10,
        help="轻约束模式下允许的行业超配容忍度，例如 0.10 表示相对基准最多超配 10%%。",
    )
    parser.add_argument(
        "--size-exposure-limit",
        type=float,
        default=0.20,
        help="轻约束模式下允许的 size 暴露绝对值上限。",
    )
    parser.add_argument(
        "--sector-active-tolerances",
        nargs="+",
        type=float,
        default=None,
        help="可选：轻约束模式下批量扫描多个行业超配容忍度。",
    )
    parser.add_argument(
        "--size-exposure-limits",
        nargs="+",
        type=float,
        default=None,
        help="可选：轻约束模式下批量扫描多个 size 暴露上限。",
    )
    parser.add_argument("--buy-cost-bps", type=float, default=5.0, help="买入基础成本（bps）。")
    parser.add_argument("--sell-cost-bps", type=float, default=5.0, help="卖出基础成本（bps）。")
    parser.add_argument("--slippage-bps", type=float, default=0.0, help="单边滑点假设（bps），会同时加到买入和卖出。")
    parser.add_argument("--commission-bps", type=float, default=0.0, help="单边佣金假设（bps），会同时加到买入和卖出。")
    parser.add_argument("--stamp-tax-bps", type=float, default=0.0, help="卖出印花税或交易税假设（bps），只加到卖出。")
    parser.add_argument(
        "--signal-delay-days",
        type=int,
        default=1,
        help="信号生成后延迟多少个交易日执行。默认 1，表示下一交易日收盘执行。",
    )
    parser.add_argument(
        "--holding-clock",
        choices=["signal_horizon", "execution_horizon"],
        default="signal_horizon",
        help="默认从信号日计算标签 horizon；旧执行日口径只用于敏感性对照。",
    )
    return parser.parse_args()


def infer_run_name(predictions_path: Path) -> str:
    parent_name = predictions_path.parent.name
    if parent_name:
        return parent_name
    return predictions_path.stem


def format_param_tag(value: float) -> str:
    text = f"{value:.2f}"
    return text.replace("-", "m").replace(".", "p")


def build_constraint_param_grid(args: argparse.Namespace) -> list[tuple[float, float]]:
    sector_values = (
        args.sector_active_tolerances
        if args.sector_active_tolerances is not None
        else [args.sector_active_tolerance]
    )
    size_values = (
        args.size_exposure_limits
        if args.size_exposure_limits is not None
        else [args.size_exposure_limit]
    )
    return [(float(sector_value), float(size_value)) for sector_value in sector_values for size_value in size_values]


def resolve_effective_cost_bps(args: argparse.Namespace) -> tuple[float, float]:
    """把更真实的交易成本拆分折算成买卖两侧 bps。

    `src.portfolio` 内部已经支持买入成本和卖出成本。
    为了不破坏已有回测逻辑，这里只在入口层做拆分：

    - buy effective cost = 基础买入成本 + 滑点 + 佣金；
    - sell effective cost = 基础卖出成本 + 滑点 + 佣金 + 印花税/交易税。

    美国股票通常没有中国 A 股式印花税，但保留这个参数有利于做跨市场假设。
    """

    buy_cost_bps = float(args.buy_cost_bps) + float(args.slippage_bps) + float(args.commission_bps)
    sell_cost_bps = (
        float(args.sell_cost_bps)
        + float(args.slippage_bps)
        + float(args.commission_bps)
        + float(args.stamp_tax_bps)
    )
    return buy_cost_bps, sell_cost_bps


def write_summary_artifacts(output_root_dir: Path, summary_df: pd.DataFrame) -> None:
    output_root_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_root_dir / "portfolio_summary.csv", index=False)

    if summary_df.empty:
        markdown_text = "# Portfolio Summary\n\n_No portfolio diagnostic runs were executed._\n"
    else:
        markdown_text = (
            "# Portfolio Summary\n\n"
            "这份摘要表用于快速比较不同预测结果文件转成组合后的表现、换手和暴露指标。\n\n"
            + summary_df.to_markdown(index=False)
            + "\n"
        )

    (output_root_dir / "portfolio_summary.md").write_text(markdown_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    predictions_paths = [resolve_project_path(path_like) for path_like in args.predictions_paths]

    if args.run_names is not None and len(args.run_names) != len(predictions_paths):
        raise ValueError("run_names length must match predictions_paths length.")

    data_path = resolve_project_path(args.data_path)
    output_root_dir = resolve_project_path(args.output_root_dir)
    market_snapshot_df = load_market_snapshot_frame(data_path)
    constrained_param_grid = build_constraint_param_grid(args)
    effective_buy_cost_bps, effective_sell_cost_bps = resolve_effective_cost_bps(args)

    summary_rows: list[dict[str, object]] = []

    for index, predictions_path in enumerate(predictions_paths):
        run_name = args.run_names[index] if args.run_names is not None else infer_run_name(predictions_path)
        hold_days = infer_hold_days(predictions_path=predictions_path, explicit_hold_days=args.hold_days)
        step_days = args.step_days if args.step_days is not None else hold_days
        for constraint_mode in args.constraint_modes:
            if constraint_mode == "sector_size_constrained":
                mode_param_grid: list[tuple[float | None, float | None]] = constrained_param_grid
            else:
                mode_param_grid = [(None, None)]

            for sector_active_tolerance, size_exposure_limit in mode_param_grid:
                mode_run_name = f"{run_name}__{constraint_mode}__{args.holding_clock}"
                if sector_active_tolerance is not None and size_exposure_limit is not None:
                    mode_run_name += (
                        f"__sat{format_param_tag(sector_active_tolerance)}"
                        f"__sel{format_param_tag(size_exposure_limit)}"
                    )

                output_dir = output_root_dir / mode_run_name
                result = run_portfolio_diagnostic(
                    predictions_path=predictions_path,
                    market_snapshot_df=market_snapshot_df,
                    data_path=data_path,
                    output_dir=output_dir,
                    hold_days=hold_days,
                    step_days=step_days,
                    top_n=args.top_n,
                    weighting_scheme=args.weighting_scheme,
                    max_weight=args.max_weight,
                    constraint_mode=constraint_mode,
                    sector_active_tolerance=sector_active_tolerance,
                    size_exposure_limit=size_exposure_limit,
                    buy_cost_bps=effective_buy_cost_bps,
                    sell_cost_bps=effective_sell_cost_bps,
                    run_name=mode_run_name,
                    signal_delay_days=args.signal_delay_days,
                    holding_clock=args.holding_clock,
                )
                metrics = result["metrics"]
                summary_rows.append(
                    {
                        "run_name": mode_run_name,
                        "base_run_name": run_name,
                        "constraint_mode": metrics.get("constraint_mode"),
                        "sector_active_tolerance": metrics.get("sector_active_tolerance"),
                        "size_exposure_limit": metrics.get("size_exposure_limit"),
                        "hold_days": metrics.get("hold_days"),
                        "rebalance_step_days": metrics.get("rebalance_step_days"),
                        "signal_delay_days": metrics.get("signal_delay_days"),
                        "holding_clock": metrics.get("holding_clock"),
                        "effective_holding_days": metrics.get("effective_holding_days"),
                        "top_n": metrics.get("top_n"),
                        "weighting_scheme": metrics.get("weighting_scheme"),
                        "base_buy_cost_bps": args.buy_cost_bps,
                        "base_sell_cost_bps": args.sell_cost_bps,
                        "slippage_bps": args.slippage_bps,
                        "commission_bps": args.commission_bps,
                        "stamp_tax_bps": args.stamp_tax_bps,
                        "effective_buy_cost_bps": effective_buy_cost_bps,
                        "effective_sell_cost_bps": effective_sell_cost_bps,
                        "portfolio_total_return": metrics.get("portfolio_total_return"),
                        "portfolio_annualized_return": metrics.get("portfolio_annualized_return"),
                        "portfolio_annualized_vol": metrics.get("portfolio_annualized_vol"),
                        "portfolio_sharpe": metrics.get("portfolio_sharpe"),
                        "sharpe_definition": metrics.get("sharpe_definition"),
                        "portfolio_max_drawdown": metrics.get("portfolio_max_drawdown"),
                        "benchmark_total_return": metrics.get("benchmark_total_return"),
                        "excess_total_return_vs_benchmark": metrics.get("excess_total_return_vs_benchmark"),
                        "average_turnover": metrics.get("average_turnover"),
                        "average_transaction_cost_bps": metrics.get("average_transaction_cost_bps"),
                        "total_transaction_cost": metrics.get("total_transaction_cost"),
                        "average_abs_active_size_exposure": metrics.get("average_abs_active_size_exposure"),
                        "average_max_abs_sector_active_weight": metrics.get("average_max_abs_sector_active_weight"),
                        "positive_period_ratio": metrics.get("positive_period_ratio"),
                        "period_count": metrics.get("period_count"),
                    }
                )

                print(
                    f"[Portfolio] {mode_run_name}: total_return={metrics.get('portfolio_total_return'):.4f}, "
                    f"sharpe={metrics.get('portfolio_sharpe'):.4f}, "
                    f"size_abs={metrics.get('average_abs_active_size_exposure'):.4f}, "
                    f"sector_abs={metrics.get('average_max_abs_sector_active_weight'):.4f}, "
                    f"excess_vs_benchmark={metrics.get('excess_total_return_vs_benchmark'):.4f}"
                )

    summary_df = pd.DataFrame(summary_rows)
    write_summary_artifacts(output_root_dir=output_root_dir, summary_df=summary_df)


if __name__ == "__main__":
    main()
