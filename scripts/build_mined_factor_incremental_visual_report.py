"""Build a teacher-facing visual report for the mined-factor experiment.

这个脚本只做报告，不重新训练模型。

输入：
    outputs/mined_factor_incremental_experiment_oos202506/*.csv

输出：
    outputs/mined_factor_incremental_experiment_oos202506/visual_report/
        myquant_mined_factor_incremental_report.html
        myquant_mined_factor_incremental_report.pdf
        figures/*.png

报告的写作原则：

1. 不只展示最好结果，也展示负结果和限制；
2. 把“预测层”和“组合层”分开；
3. 把 96 个汇总视角和 336 条明细回测分开；
4. 用图形回答“这是不是成熟项目”，避免只贴一堆 CSV 表。
"""

from __future__ import annotations

import base64
import html
import json
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "outputs/mined_factor_incremental_experiment_oos202506"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "visual_report"


EXPERIMENT_ORDER = [
    "baseline_linear",
    "warm_gp_linear",
    "ppo_linear",
    "warm_gp_ppo_linear",
    "baseline_nonlinear",
    "warm_gp_ppo_nonlinear",
]

EXPERIMENT_LABELS = {
    "baseline_linear": "Baseline Linear",
    "warm_gp_linear": "Warm-GP Linear",
    "ppo_linear": "PPO Linear",
    "warm_gp_ppo_linear": "Warm-GP+PPO Linear",
    "baseline_nonlinear": "Baseline Nonlinear",
    "warm_gp_ppo_nonlinear": "Warm-GP+PPO Nonlinear",
}

STRATEGY_ORDER = ["hold10_step10", "hold10_step5", "hold20_step20", "hold20_step10"]
WINDOW_ORDER = ["full", "3m", "6m", "12m"]


@dataclass(frozen=True)
class FigureRecord:
    """记录报告中一张图的路径和解释。"""

    title: str
    path: Path
    caption: str


def find_readable_font() -> FontProperties:
    """尽量使用 macOS 上能显示中文的字体。"""

    candidate_paths = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    for font_path in candidate_paths:
        if Path(font_path).exists():
            font_manager.fontManager.addfont(font_path)
            return FontProperties(fname=font_path)
    return FontProperties()


FONT_PROP = find_readable_font()


def configure_style() -> None:
    """统一图表风格。"""

    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["axes.unicode_minus"] = False
    if FONT_PROP.get_file():
        plt.rcParams["font.family"] = FONT_PROP.get_name()


def percent_axis(ax: plt.Axes) -> None:
    """把 y 轴显示为百分比。"""

    ax.yaxis.set_major_formatter(lambda value, _: f"{value * 100:.1f}%")


def safe_savefig(fig: plt.Figure, path: Path) -> Path:
    """保存图片并关闭 figure，避免批量生成时占用内存。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def load_inputs(input_dir: Path) -> dict[str, pd.DataFrame | dict]:
    """读取本次报告需要的所有结果表。"""

    required_files = {
        "model_metrics": "model_metrics.csv",
        "model_delta": "model_metric_delta.csv",
        "view_96": "view_96_summary.csv",
        "portfolio": "portfolio_metrics.csv",
        "portfolio_delta": "portfolio_view_delta.csv",
        "runtime": "runtime.csv",
        "feature_counts": "feature_counts.csv",
        "factor_zoo_summary": "factor_zoo_summary.csv",
        "config": "config.json",
    }
    loaded: dict[str, pd.DataFrame | dict] = {}
    for key, filename in required_files.items():
        path = input_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing required report input: {path}")
        if filename.endswith(".json"):
            loaded[key] = json.loads(path.read_text(encoding="utf-8"))
        else:
            loaded[key] = pd.read_csv(path)
    return loaded


def ordered_experiment_series(values: pd.Series) -> pd.Series:
    """按固定实验顺序重排 Series。"""

    return values.reindex([name for name in EXPERIMENT_ORDER if name in values.index])


def add_value_labels(ax: plt.Axes, *, fmt: str = "{:.3f}", rotation: int = 0) -> None:
    """给柱状图添加数值标签。"""

    for container in ax.containers:
        labels = []
        for value in container.datavalues:
            if pd.isna(value):
                labels.append("")
            else:
                labels.append(fmt.format(value))
        ax.bar_label(container, labels=labels, fontsize=8, rotation=rotation, padding=2)


def plot_model_ic(model_metrics: pd.DataFrame, fig_dir: Path) -> FigureRecord:
    metrics = model_metrics.copy()
    metrics["experiment_label"] = metrics["experiment"].map(EXPERIMENT_LABELS)
    metrics = metrics.set_index("experiment").reindex(EXPERIMENT_ORDER).dropna(subset=["experiment_label"])
    long_df = metrics.reset_index().melt(
        id_vars=["experiment_label"],
        value_vars=["pearson_ic_mean", "spearman_ic_mean"],
        var_name="metric",
        value_name="value",
    )
    long_df["metric"] = long_df["metric"].map({"pearson_ic_mean": "Pearson IC", "spearman_ic_mean": "Rank IC"})

    fig, ax = plt.subplots(figsize=(13.5, 6.5))
    sns.barplot(data=long_df, x="experiment_label", y="value", hue="metric", ax=ax, palette="Set2")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Model-Layer IC Comparison")
    ax.set_xlabel("")
    ax.set_ylabel("Mean Daily Cross-Sectional IC")
    ax.tick_params(axis="x", rotation=25)
    add_value_labels(ax, fmt="{:.4f}", rotation=90)
    path = safe_savefig(fig, fig_dir / "01_model_ic_comparison.png")
    return FigureRecord(
        "模型层 IC 对比",
        path,
        "PPO 线性组只比 baseline_linear 略高；Warm-GP 单独加入后 IC 下降；Warm-GP+PPO 在非线性组上提升最明显。",
    )


def plot_model_metric_delta(model_delta: pd.DataFrame, fig_dir: Path) -> FigureRecord:
    delta = model_delta.copy()
    heat_cols = [
        "delta_pearson_corr",
        "delta_spearman_corr",
        "delta_pearson_ic_mean",
        "delta_spearman_ic_mean",
        "delta_long_short_return",
        "delta_rmse",
        "delta_mae",
    ]
    available = [column for column in heat_cols if column in delta.columns]
    heat = delta.set_index("experiment")[available].copy()
    heat = heat.rename(
        columns={
            "delta_pearson_corr": "Pearson Corr",
            "delta_spearman_corr": "Spearman Corr",
            "delta_pearson_ic_mean": "Pearson IC",
            "delta_spearman_ic_mean": "Rank IC",
            "delta_long_short_return": "LS Proxy",
            "delta_rmse": "RMSE",
            "delta_mae": "MAE",
        }
    )
    heat.index = [EXPERIMENT_LABELS.get(index, index) for index in heat.index]

    fig, ax = plt.subplots(figsize=(12.5, 5.5))
    sns.heatmap(heat, annot=True, fmt=".4f", cmap="RdYlGn", center=0, ax=ax, linewidths=0.4)
    ax.set_title("Model-Layer Delta vs Same-Family Baseline")
    ax.set_xlabel("Metric Delta")
    ax.set_ylabel("")
    path = safe_savefig(fig, fig_dir / "02_model_metric_delta_heatmap.png")
    return FigureRecord(
        "模型层增量热力图",
        path,
        "绿色表示相对同类 baseline 改善。注意 RMSE/MAE 的正值代表误差增加，所以这两列不能按绿色机械解读。",
    )


def plot_feature_counts(feature_counts: pd.DataFrame, fig_dir: Path) -> FigureRecord:
    df = feature_counts.copy()
    df["experiment_label"] = df["experiment"].map(EXPERIMENT_LABELS)
    df = df.set_index("experiment").reindex(EXPERIMENT_ORDER).reset_index()

    fig, ax = plt.subplots(figsize=(12.5, 5.5))
    ax.bar(df["experiment_label"], df["baseline_numeric_feature_count"], label="Original features", color="#4C78A8")
    ax.bar(
        df["experiment_label"],
        df["mined_feature_count"],
        bottom=df["baseline_numeric_feature_count"],
        label="Mined factors",
        color="#F58518",
    )
    ax.set_title("Feature Count by Experiment Group")
    ax.set_ylabel("Feature Count")
    ax.tick_params(axis="x", rotation=25)
    ax.legend()
    add_value_labels(ax, fmt="{:.0f}")
    path = safe_savefig(fig, fig_dir / "03_feature_counts.png")
    return FigureRecord(
        "特征数量结构",
        path,
        "原始特征为 275 个；Warm-GP 加 8 个，PPO 加 10 个，合并组加 18 个。",
    )


def plot_runtime(runtime: pd.DataFrame, fig_dir: Path) -> FigureRecord:
    stage_summary = runtime.groupby("stage", as_index=False)["runtime_seconds"].sum()
    stage_summary = stage_summary[stage_summary["stage"].str.startswith("model_") | (stage_summary["stage"] == "materialize_feature_group")]
    stage_summary["stage"] = stage_summary["stage"].str.replace("model_", "", regex=False)
    stage_summary = stage_summary.sort_values("runtime_seconds", ascending=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=stage_summary, x="runtime_seconds", y="stage", ax=ax, color="#4C78A8")
    ax.set_title("Runtime Bottleneck by Stage")
    ax.set_xlabel("Runtime Seconds")
    ax.set_ylabel("")
    for patch in ax.patches:
        width = patch.get_width()
        ax.text(width + 8, patch.get_y() + patch.get_height() / 2, f"{width:.0f}s", va="center", fontsize=9)
    path = safe_savefig(fig, fig_dir / "04_runtime_bottleneck.png")
    return FigureRecord(
        "运行耗时瓶颈",
        path,
        "ExtraTrees 是最大耗时来源，约 26 分钟；后续主实验应把它降级为低频对照组。",
    )


def plot_96_average_by_experiment(view_96: pd.DataFrame, fig_dir: Path) -> FigureRecord:
    summary = (
        view_96.groupby("experiment", as_index=False)
        .agg(
            avg_excess=("avg_excess_return", "mean"),
            avg_total=("avg_total_return", "mean"),
            avg_sharpe=("avg_sharpe", "mean"),
            positive_excess=("positive_excess_windows", "sum"),
        )
        .set_index("experiment")
        .reindex(EXPERIMENT_ORDER)
        .reset_index()
    )
    summary["experiment_label"] = summary["experiment"].map(EXPERIMENT_LABELS)
    long_df = summary.melt(
        id_vars=["experiment_label"],
        value_vars=["avg_excess", "avg_total"],
        var_name="metric",
        value_name="value",
    )
    long_df["metric"] = long_df["metric"].map({"avg_excess": "Avg Excess", "avg_total": "Avg Total"})

    fig, ax = plt.subplots(figsize=(13.5, 6.5))
    sns.barplot(data=long_df, x="experiment_label", y="value", hue="metric", ax=ax, palette=["#E45756", "#54A24B"])
    ax.axhline(0, color="black", linewidth=1)
    percent_axis(ax)
    ax.set_title("Average Across 96 Views")
    ax.set_xlabel("")
    ax.set_ylabel("Average Return")
    ax.tick_params(axis="x", rotation=25)
    ax.legend()
    path = safe_savefig(fig, fig_dir / "05_96_average_by_experiment.png")
    return FigureRecord(
        "96 视角平均表现",
        path,
        "没有任何实验组的 96 视角平均超额收益为正；这说明项目有完整验证闭环，但 alpha 稳定性还不够强。",
    )


def plot_positive_window_counts(view_96: pd.DataFrame, fig_dir: Path) -> FigureRecord:
    summary = (
        view_96.groupby("experiment", as_index=False)
        .agg(
            positive_excess=("positive_excess_windows", "sum"),
            total_windows=("window_count", "sum"),
        )
        .set_index("experiment")
        .reindex(EXPERIMENT_ORDER)
        .reset_index()
    )
    summary["positive_ratio"] = summary["positive_excess"] / summary["total_windows"].replace(0, np.nan)
    summary["experiment_label"] = summary["experiment"].map(EXPERIMENT_LABELS)

    fig, ax = plt.subplots(figsize=(12.5, 5.5))
    sns.barplot(data=summary, x="experiment_label", y="positive_ratio", ax=ax, color="#72B7B2")
    percent_axis(ax)
    ax.set_ylim(0, max(0.55, float(summary["positive_ratio"].max()) + 0.08))
    ax.set_title("Positive Excess Window Ratio")
    ax.set_xlabel("")
    ax.set_ylabel("Positive Excess Window Ratio")
    ax.tick_params(axis="x", rotation=25)
    add_value_labels(ax, fmt="{:.2f}")
    path = safe_savefig(fig, fig_dir / "06_positive_excess_window_ratio.png")
    return FigureRecord(
        "正超额窗口比例",
        path,
        "PPO 线性组没有明显提高赢 benchmark 的窗口数量；非线性合并组平均收益更好，但正窗口比例不是压倒性优势。",
    )


def plot_excess_heatmaps(view_96: pd.DataFrame, fig_dir: Path) -> list[FigureRecord]:
    records: list[FigureRecord] = []
    for window_mode in WINDOW_ORDER:
        subset = view_96[view_96["window_mode"] == window_mode].copy()
        if subset.empty:
            continue
        pivot = subset.pivot_table(
            index="experiment",
            columns="strategy_name",
            values="avg_excess_return",
            aggfunc="mean",
        ).reindex(EXPERIMENT_ORDER)[STRATEGY_ORDER]
        pivot.index = [EXPERIMENT_LABELS.get(index, index) for index in pivot.index]

        fig, ax = plt.subplots(figsize=(11, 6.5))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2%",
            cmap="RdYlGn",
            center=0,
            ax=ax,
            linewidths=0.5,
            cbar_kws={"label": "Avg Excess Return"},
        )
        ax.set_title(f"Average Excess Return Heatmap - {window_mode.upper()} View")
        ax.set_xlabel("Strategy")
        ax.set_ylabel("")
        path = safe_savefig(fig, fig_dir / f"07_excess_heatmap_{window_mode}.png")
        records.append(
            FigureRecord(
                f"{window_mode.upper()} 视角超额收益热力图",
                path,
                "绿色表示该策略和窗口类型下平均超额收益更高。full/12m 只有一个窗口，3m/6m 更能观察稳定性。",
            )
        )
    return records


def plot_top_bottom_views(view_96: pd.DataFrame, fig_dir: Path) -> list[FigureRecord]:
    records: list[FigureRecord] = []
    for name, ascending, filename, title in [
        ("top", False, "08_top_20_views.png", "Top 20 Views by Average Excess Return"),
        ("bottom", True, "09_bottom_20_views.png", "Bottom 20 Views by Average Excess Return"),
    ]:
        df = view_96.sort_values("avg_excess_return", ascending=ascending).head(20).copy()
        df["view"] = df["experiment"].map(EXPERIMENT_LABELS) + " | " + df["strategy_name"] + " | " + df["window_mode"]
        fig, ax = plt.subplots(figsize=(13, 9))
        sns.barplot(data=df, y="view", x="avg_excess_return", ax=ax, color="#4C78A8" if not ascending else "#E45756")
        ax.axvline(0, color="black", linewidth=1)
        ax.xaxis.set_major_formatter(lambda value, _: f"{value * 100:.1f}%")
        ax.set_title(title)
        ax.set_xlabel("Average Excess Return")
        ax.set_ylabel("")
        path = safe_savefig(fig, fig_dir / filename)
        records.append(
            FigureRecord(
                "最好/最差视角对照" if name == "top" else "最差视角风险提示",
                path,
                "只看最好视角会高估项目成熟度；最差视角展示了模型和策略对窗口、模型族、调仓规则的敏感性。",
            )
        )
    return records


def plot_sharpe_vs_excess(view_96: pd.DataFrame, fig_dir: Path) -> FigureRecord:
    df = view_96.copy()
    df["experiment_label"] = df["experiment"].map(EXPERIMENT_LABELS)

    fig, ax = plt.subplots(figsize=(11.5, 7))
    sns.scatterplot(
        data=df,
        x="avg_excess_return",
        y="avg_sharpe",
        hue="experiment_label",
        style="window_mode",
        size="window_count",
        sizes=(60, 220),
        ax=ax,
        alpha=0.85,
    )
    ax.axvline(0, color="black", linewidth=1)
    ax.axhline(0, color="black", linewidth=1)
    ax.xaxis.set_major_formatter(lambda value, _: f"{value * 100:.1f}%")
    ax.set_title("Sharpe vs Average Excess Return Across 96 Views")
    ax.set_xlabel("Average Excess Return")
    ax.set_ylabel("Average Sharpe")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    path = safe_savefig(fig, fig_dir / "10_sharpe_vs_excess_scatter.png")
    return FigureRecord(
        "Sharpe 与超额收益关系",
        path,
        "高 Sharpe 不一定代表稳定超额收益，尤其在短窗口和少量调仓次数下更容易失真。",
    )


def plot_portfolio_detail_box(portfolio: pd.DataFrame, fig_dir: Path) -> FigureRecord:
    df = portfolio[portfolio["status"] == "ok"].copy()
    df["experiment_label"] = df["experiment"].map(EXPERIMENT_LABELS)
    df["experiment_label"] = pd.Categorical(df["experiment_label"], [EXPERIMENT_LABELS[x] for x in EXPERIMENT_ORDER])

    fig, ax = plt.subplots(figsize=(13, 7))
    sns.boxplot(data=df, x="experiment_label", y="excess_total_return_vs_benchmark", ax=ax, color="#9ecae9")
    sns.stripplot(data=df, x="experiment_label", y="excess_total_return_vs_benchmark", ax=ax, color="black", alpha=0.25, size=3)
    ax.axhline(0, color="black", linewidth=1)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value * 100:.0f}%")
    ax.set_title("Distribution of Detailed Portfolio Excess Returns")
    ax.set_xlabel("")
    ax.set_ylabel("Excess Return vs Benchmark")
    ax.tick_params(axis="x", rotation=25)
    path = safe_savefig(fig, fig_dir / "11_portfolio_excess_distribution.png")
    return FigureRecord(
        "组合明细超额收益分布",
        path,
        "336 条组合明细显示：收益分布离散明显，项目已经有组合验证层，但信号稳定性还没有达到强结论。",
    )


def plot_strategy_window_matrix(view_96: pd.DataFrame, fig_dir: Path) -> FigureRecord:
    df = view_96.copy()
    df["strategy_window"] = df["strategy_name"] + " | " + df["window_mode"]
    pivot = df.pivot_table(
        index="strategy_window",
        columns="experiment",
        values="avg_excess_return",
        aggfunc="mean",
    )[EXPERIMENT_ORDER]
    pivot.columns = [EXPERIMENT_LABELS.get(column, column) for column in pivot.columns]

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(pivot, annot=True, fmt=".1%", cmap="RdYlGn", center=0, ax=ax, linewidths=0.4)
    ax.set_title("Strategy x Window x Experiment: Average Excess Return")
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Strategy | Window")
    path = safe_savefig(fig, fig_dir / "12_strategy_window_matrix.png")
    return FigureRecord(
        "策略-窗口-实验矩阵",
        path,
        "这张图直接回答策略结果是否稳定：如果一列只有少数格子为绿，说明提升依赖特定窗口或持有规则。",
    )


def load_daily_returns_for_view(input_dir: Path, row: pd.Series) -> pd.DataFrame | None:
    """读取某个组合明细对应的 daily_returns.csv。"""

    experiment = str(row["experiment"])
    window_id = str(row["window_id"])
    hold_days = int(row["hold_days"])
    step_days = int(row["step_days"])
    root = input_dir / "portfolio_runs" / experiment / window_id
    if not root.exists():
        return None
    matches = sorted(root.glob(f"*hold{hold_days}d_step{step_days}d*/daily_returns.csv"))
    if not matches:
        return None
    daily = pd.read_csv(matches[0])
    daily["date"] = pd.to_datetime(daily["date"])
    daily["experiment"] = experiment
    daily["strategy_name"] = row["strategy_name"]
    daily["window_id"] = window_id
    return daily


def plot_nav_and_drawdown(input_dir: Path, portfolio: pd.DataFrame, fig_dir: Path) -> list[FigureRecord]:
    """画代表性 full-window 净值和回撤曲线。"""

    full = portfolio[(portfolio["status"] == "ok") & (portfolio["window_mode"] == "full")].copy()
    selected_keys = [
        ("baseline_linear", "hold20_step20"),
        ("ppo_linear", "hold20_step20"),
        ("warm_gp_ppo_linear", "hold20_step20"),
        ("warm_gp_ppo_nonlinear", "hold20_step20"),
    ]
    daily_frames: list[pd.DataFrame] = []
    for experiment, strategy in selected_keys:
        matched = full[(full["experiment"] == experiment) & (full["strategy_name"] == strategy)]
        if matched.empty:
            continue
        daily = load_daily_returns_for_view(input_dir, matched.iloc[0])
        if daily is not None and not daily.empty:
            daily["label"] = EXPERIMENT_LABELS.get(experiment, experiment) + f" | {strategy}"
            daily_frames.append(daily)

    if not daily_frames:
        return []
    daily_df = pd.concat(daily_frames, ignore_index=True)

    records: list[FigureRecord] = []
    fig, ax = plt.subplots(figsize=(13, 6.5))
    for label, frame in daily_df.groupby("label"):
        ax.plot(frame["date"], frame["portfolio_nav"], label=label, linewidth=2)
    benchmark = daily_df.groupby("date", as_index=False)["benchmark_nav"].mean()
    ax.plot(benchmark["date"], benchmark["benchmark_nav"], label="Equal-weight benchmark", color="black", linestyle="--")
    ax.set_title("Full-OOS Portfolio NAV Curves")
    ax.set_xlabel("Date")
    ax.set_ylabel("NAV")
    ax.legend(fontsize=8)
    records.append(
        FigureRecord(
            "代表性组合净值曲线",
            safe_savefig(fig, fig_dir / "13_full_oos_nav_curves.png"),
            "净值曲线展示 absolute return；是否成熟还要看是否稳定跑赢 benchmark。",
        )
    )

    fig, ax = plt.subplots(figsize=(13, 6.5))
    for label, frame in daily_df.groupby("label"):
        nav = frame["portfolio_nav"].astype(float)
        drawdown = nav / nav.cummax() - 1.0
        ax.plot(frame["date"], drawdown, label=label, linewidth=2)
    ax.axhline(0, color="black", linewidth=1)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value * 100:.0f}%")
    ax.set_title("Full-OOS Drawdown Curves")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.legend(fontsize=8)
    records.append(
        FigureRecord(
            "代表性组合回撤曲线",
            safe_savefig(fig, fig_dir / "14_full_oos_drawdown_curves.png"),
            "回撤曲线更接近真实投资视角。即使总收益为正，深回撤也会削弱项目可交易性。",
        )
    )
    return records


def plot_prediction_correlation(input_dir: Path, fig_dir: Path) -> FigureRecord | None:
    """计算不同实验组预测分数之间的相关性。"""

    prediction_dir = input_dir / "predictions"
    frames: list[pd.DataFrame] = []
    for experiment in EXPERIMENT_ORDER:
        path = prediction_dir / f"{experiment}_predictions.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, usecols=["date", "instrument_id", "predicted_y"])
        df = df.rename(columns={"predicted_y": experiment})
        frames.append(df)
    if not frames:
        return None
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["date", "instrument_id"], how="inner")
    corr = merged[[column for column in EXPERIMENT_ORDER if column in merged.columns]].corr()
    corr.index = [EXPERIMENT_LABELS.get(index, index) for index in corr.index]
    corr.columns = [EXPERIMENT_LABELS.get(column, column) for column in corr.columns]

    fig, ax = plt.subplots(figsize=(9.5, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", vmin=-1, vmax=1, ax=ax, linewidths=0.4)
    ax.set_title("Prediction Correlation Across Experiment Groups")
    path = safe_savefig(fig, fig_dir / "15_prediction_correlation.png")
    return FigureRecord(
        "不同实验组预测相关性",
        path,
        "预测相关性越高，说明新因子或新模型带来的排序变化越有限；这有助于判断增量是否真实。",
    )


def plot_3m_stability(portfolio: pd.DataFrame, fig_dir: Path) -> FigureRecord:
    df = portfolio[
        (portfolio["status"] == "ok")
        & (portfolio["window_mode"] == "3m")
        & (portfolio["strategy_name"].isin(["hold20_step20", "hold20_step10"]))
        & (portfolio["experiment"].isin(["baseline_linear", "ppo_linear", "warm_gp_ppo_nonlinear"]))
    ].copy()
    df["label"] = df["experiment"].map(EXPERIMENT_LABELS) + " | " + df["strategy_name"]
    df["window_start"] = pd.to_datetime(df["window_start"])
    df = df.sort_values("window_start")

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.lineplot(data=df, x="window_start", y="excess_total_return_vs_benchmark", hue="label", marker="o", ax=ax)
    ax.axhline(0, color="black", linewidth=1)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value * 100:.0f}%")
    ax.set_title("3M Rolling OOS Stability: Selected Experiments")
    ax.set_xlabel("Window Start")
    ax.set_ylabel("Excess Return")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    path = safe_savefig(fig, fig_dir / "16_3m_rolling_stability.png")
    return FigureRecord(
        "3M rolling OOS 稳定性",
        path,
        "3M 子窗口能暴露时间稳定性问题。成熟项目需要说明哪些窗口有效、哪些窗口失效。",
    )


def dataframe_to_html(df: pd.DataFrame, *, max_rows: int = 20, percent_cols: Iterable[str] = ()) -> str:
    """把表格格式化成 HTML。"""

    preview = df.head(max_rows).copy()
    percent_set = set(percent_cols)
    for column in preview.columns:
        if column in percent_set:
            preview[column] = pd.to_numeric(preview[column], errors="coerce").map(
                lambda value: f"{value * 100:.2f}%" if pd.notna(value) else ""
            )
        elif pd.api.types.is_float_dtype(preview[column]):
            preview[column] = preview[column].map(lambda value: f"{value:.4f}" if pd.notna(value) else "")
    return preview.to_html(index=False, escape=True, classes="data-table")


def image_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def build_html_report(
    *,
    output_path: Path,
    input_dir: Path,
    figure_records: list[FigureRecord],
    data: dict[str, pd.DataFrame | dict],
) -> None:
    """生成单文件 HTML，图片以 base64 嵌入，方便邮件附件传递。"""

    model_metrics = data["model_metrics"]  # type: ignore[assignment]
    model_delta = data["model_delta"]  # type: ignore[assignment]
    view_96 = data["view_96"]  # type: ignore[assignment]
    portfolio_delta = data["portfolio_delta"]  # type: ignore[assignment]
    runtime = data["runtime"]  # type: ignore[assignment]
    feature_counts = data["feature_counts"]  # type: ignore[assignment]
    factor_zoo_summary = data["factor_zoo_summary"]  # type: ignore[assignment]
    config = data["config"]  # type: ignore[assignment]

    best_views = view_96.sort_values("avg_excess_return", ascending=False).head(10)
    worst_views = view_96.sort_values("avg_excess_return", ascending=True).head(10)
    top_delta = portfolio_delta.sort_values("delta_avg_excess_return", ascending=False).head(10)
    runtime_summary = runtime.groupby("stage", as_index=False)["runtime_seconds"].sum().sort_values(
        "runtime_seconds", ascending=False
    )

    image_sections = []
    for record in figure_records:
        encoded = image_to_base64(record.path)
        image_sections.append(
            f"""
            <section class="figure-card">
              <h3>{html.escape(record.title)}</h3>
              <img src="data:image/png;base64,{encoded}" alt="{html.escape(record.title)}" />
              <p>{html.escape(record.caption)}</p>
            </section>
            """
        )

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>MyQuant Mined Factor Incremental Experiment Report</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Hiragino Sans GB", "Helvetica Neue", Arial, sans-serif;
      margin: 0;
      background: #f4f0e8;
      color: #1f2933;
      line-height: 1.55;
    }}
    .hero {{
      background: linear-gradient(135deg, #102a43 0%, #243b53 48%, #486581 100%);
      color: white;
      padding: 46px 64px;
    }}
    .hero h1 {{ margin: 0 0 12px; font-size: 34px; }}
    .hero p {{ max-width: 980px; font-size: 16px; opacity: 0.95; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 32px 28px 80px; }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 14px;
      margin: 22px 0 28px;
    }}
    .metric-card {{
      background: #fff;
      border-radius: 16px;
      padding: 18px;
      box-shadow: 0 8px 28px rgba(16, 42, 67, 0.08);
      border: 1px solid rgba(16, 42, 67, 0.08);
    }}
    .metric-card .label {{ color: #627d98; font-size: 13px; }}
    .metric-card .value {{ font-size: 24px; font-weight: 700; margin-top: 5px; }}
    section {{
      background: rgba(255, 255, 255, 0.92);
      margin: 24px 0;
      padding: 26px;
      border-radius: 18px;
      box-shadow: 0 10px 30px rgba(16, 42, 67, 0.08);
    }}
    h2 {{ margin-top: 0; color: #102a43; }}
    h3 {{ margin-top: 0; color: #243b53; }}
    .verdict {{
      border-left: 6px solid #d64545;
      background: #fff8f4;
    }}
    .good {{
      border-left: 6px solid #2f855a;
      background: #f0fff4;
    }}
    .figure-card img {{
      width: 100%;
      height: auto;
      border-radius: 12px;
      border: 1px solid #d9e2ec;
      background: white;
    }}
    .figure-card p {{ color: #52606d; }}
    .data-table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 13px;
      margin-top: 12px;
    }}
    .data-table th, .data-table td {{
      border: 1px solid #d9e2ec;
      padding: 8px 10px;
      text-align: right;
    }}
    .data-table th:first-child, .data-table td:first-child {{
      text-align: left;
    }}
    .data-table th {{
      background: #d9e2ec;
      color: #102a43;
    }}
    code {{
      background: #e4e7eb;
      padding: 2px 6px;
      border-radius: 6px;
    }}
    .two-col {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
    }}
    @media (max-width: 900px) {{
      .summary-grid, .two-col {{ grid-template-columns: 1fr; }}
      .hero {{ padding: 32px 24px; }}
      main {{ padding: 20px 14px; }}
    }}
  </style>
</head>
<body>
  <div class="hero">
    <h1>MyQuant 自挖因子增量实验报告</h1>
    <p>
      本报告基于 2025-06-01 起始 OOS、y_10d 标签、us300 股票池，对原始 275 个特征、
      Warm-GP 因子、PPO 因子以及非线性模型组合进行了 96 个结果视角的模型层和组合层检验。
    </p>
  </div>
  <main>
    <div class="summary-grid">
      <div class="metric-card"><div class="label">96-view rows</div><div class="value">{len(view_96)}</div></div>
      <div class="metric-card"><div class="label">Portfolio detail rows</div><div class="value">{len(data["portfolio"])}</div></div>
      <div class="metric-card"><div class="label">Baseline features</div><div class="value">{int(feature_counts["baseline_numeric_feature_count"].max())}</div></div>
      <div class="metric-card"><div class="label">Mined factors max</div><div class="value">{int(feature_counts["mined_feature_count"].max())}</div></div>
    </div>

    <section class="verdict">
      <h2>结论先行</h2>
      <p>
        这个项目已经具备比较完整的研究闭环：真实数据、严格 OOS、Alpha191/技术指标、自挖因子、模型消融、
        多空组合回测、成本和调仓规则、滚动窗口稳定性分析。工程和研究流程是成熟雏形。
      </p>
      <p>
        但从结果看，还不能宣称“自挖因子已经形成稳定可交易 alpha”。PPO 因子在线性模型里只有弱增量；
        Warm-GP 单独加入后变差；Warm-GP+PPO 对非线性模型有明显提升，但 96 视角平均超额收益仍未转正。
      </p>
    </section>

    <section class="good">
      <h2>实验设置</h2>
      <p><b>OOS start:</b> <code>{html.escape(str(config.get("oos_start_date", "2025-06-01")))}</code></p>
      <p><b>Target horizon:</b> <code>{html.escape(str(config.get("target_horizon", 10)))}</code></p>
      <p><b>Top-K:</b> <code>{html.escape(str(config.get("top_k", 20)))}</code>, <b>Cost:</b> <code>{html.escape(str(config.get("cost_bps", 5.0)))} bps</code></p>
      <div class="two-col">
        <div>
          <h3>Factor zoo summary</h3>
          {dataframe_to_html(factor_zoo_summary, max_rows=10)}
        </div>
        <div>
          <h3>Feature counts</h3>
          {dataframe_to_html(feature_counts, max_rows=10)}
        </div>
      </div>
    </section>

    <section>
      <h2>模型层指标</h2>
      {dataframe_to_html(model_metrics.sort_values("pearson_ic_mean", ascending=False), max_rows=10)}
    </section>

    <section>
      <h2>模型层增量</h2>
      {dataframe_to_html(model_delta, max_rows=10)}
    </section>

    <section>
      <h2>最好与最差视角</h2>
      <div class="two-col">
        <div>
          <h3>Top 10 by average excess return</h3>
          {dataframe_to_html(best_views, max_rows=10, percent_cols=["avg_total_return", "avg_excess_return", "min_excess_return", "worst_max_drawdown"])}
        </div>
        <div>
          <h3>Bottom 10 by average excess return</h3>
          {dataframe_to_html(worst_views, max_rows=10, percent_cols=["avg_total_return", "avg_excess_return", "min_excess_return", "worst_max_drawdown"])}
        </div>
      </div>
    </section>

    <section>
      <h2>组合层增量 Top 10</h2>
      {dataframe_to_html(top_delta, max_rows=10)}
    </section>

    <section>
      <h2>运行耗时</h2>
      {dataframe_to_html(runtime_summary, max_rows=15)}
    </section>

    {''.join(image_sections)}

    <section class="verdict">
      <h2>给老师看的判断</h2>
      <p>
        如果评价“项目是否成熟”，答案应该拆开：
      </p>
      <ul>
        <li><b>研究流程覆盖：</b>包含特征挖掘、模型消融和组合验证；其有效性仍由严格 OOS 结果决定。</li>
        <li><b>交易信号成熟度：</b>不足。自挖因子目前更像候选信号库，还没有稳定超额收益证据。</li>
        <li><b>下一步优先级：</b>把 PPO reward 改为模型增量分数；做 residual target mining；减少 ExtraTrees 默认运行；扩大稳定性检验。</li>
      </ul>
    </section>
  </main>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")


def wrap_lines(text: str, width: int = 88) -> list[str]:
    lines: list[str] = []
    for raw in str(text).splitlines():
        if not raw.strip():
            lines.append("")
        else:
            lines.extend(textwrap.wrap(raw, width=width, replace_whitespace=False))
    return lines


def add_text_page(pdf: PdfPages, title: str, paragraphs: list[str]) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    y = 0.95
    ax.text(0.07, y, title, fontproperties=FONT_PROP, fontsize=18, weight="bold", va="top")
    y -= 0.06
    for paragraph in paragraphs:
        for line in wrap_lines(paragraph, width=86):
            if y < 0.08:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                fig = plt.figure(figsize=(8.27, 11.69))
                ax = fig.add_axes([0, 0, 1, 1])
                ax.axis("off")
                y = 0.95
            ax.text(0.07, y, line, fontproperties=FONT_PROP, fontsize=10.5, va="top")
            y -= 0.022
        y -= 0.018
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_figure_page(pdf: PdfPages, record: FigureRecord) -> None:
    image = plt.imread(record.path)
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_axes([0.04, 0.16, 0.92, 0.74])
    ax.imshow(image)
    ax.axis("off")
    fig.text(0.05, 0.94, record.title, fontproperties=FONT_PROP, fontsize=16, weight="bold", va="top")
    caption = "\n".join(wrap_lines(record.caption, width=110))
    fig.text(0.05, 0.08, caption, fontproperties=FONT_PROP, fontsize=10, va="bottom")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def build_pdf_report(output_path: Path, figure_records: list[FigureRecord]) -> None:
    """用 matplotlib 生成图文 PDF。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        add_text_page(
            pdf,
            "MyQuant 自挖因子增量实验报告",
            [
                "本报告基于 2025-06-01 起始 OOS、y_10d 标签、us300 股票池，对原始 275 个特征、Warm-GP 因子、PPO 因子以及非线性模型组合进行了 96 个结果视角的模型层和组合层检验。",
                "结论：项目的工程闭环和研究验证层已经比较完整，但自挖因子目前还不能被表述为稳定可交易 alpha。PPO 因子只有弱增量，Warm-GP 单独加入效果较差，Warm-GP+PPO 对非线性模型有改善但仍需要更严格的稳定性验证。",
                "阅读方法：先看模型层 IC 和增量，再看组合层 rolling OOS 稳定性，最后看回撤和运行成本。不要只看单一最好窗口。",
            ],
        )
        for record in figure_records:
            add_figure_page(pdf, record)
        add_text_page(
            pdf,
            "Final Assessment",
            [
                "研究流程成熟度：较高。项目已经覆盖真实数据、严格时间切分、Alpha191/技术指标、自挖因子、模型消融、组合回测、成本、调仓频率、滚动窗口稳定性。",
                "交易信号成熟度：不足。96 视角平均超额收益未转正，说明当前信号还不能作为稳定策略证明。",
                "下一步优先级：1. PPO reward 改成模型增量分数；2. residual target mining；3. 降低 ExtraTrees 默认运行频率；4. 对新因子做多窗口稳定性筛选。",
            ],
        )


def build_all_reports(input_dir: Path = DEFAULT_INPUT_DIR, output_dir: Path = DEFAULT_OUTPUT_DIR) -> tuple[Path, Path]:
    configure_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    data = load_inputs(input_dir)
    model_metrics = data["model_metrics"]  # type: ignore[assignment]
    model_delta = data["model_delta"]  # type: ignore[assignment]
    view_96 = data["view_96"]  # type: ignore[assignment]
    portfolio = data["portfolio"]  # type: ignore[assignment]
    portfolio_delta = data["portfolio_delta"]  # type: ignore[assignment]
    runtime = data["runtime"]  # type: ignore[assignment]
    feature_counts = data["feature_counts"]  # type: ignore[assignment]

    figure_records: list[FigureRecord] = []
    figure_records.append(plot_model_ic(model_metrics, fig_dir))
    figure_records.append(plot_model_metric_delta(model_delta, fig_dir))
    figure_records.append(plot_feature_counts(feature_counts, fig_dir))
    figure_records.append(plot_runtime(runtime, fig_dir))
    figure_records.append(plot_96_average_by_experiment(view_96, fig_dir))
    figure_records.append(plot_positive_window_counts(view_96, fig_dir))
    figure_records.extend(plot_excess_heatmaps(view_96, fig_dir))
    figure_records.extend(plot_top_bottom_views(view_96, fig_dir))
    figure_records.append(plot_sharpe_vs_excess(view_96, fig_dir))
    figure_records.append(plot_portfolio_detail_box(portfolio, fig_dir))
    figure_records.append(plot_strategy_window_matrix(view_96, fig_dir))
    figure_records.extend(plot_nav_and_drawdown(input_dir, portfolio, fig_dir))
    prediction_corr = plot_prediction_correlation(input_dir, fig_dir)
    if prediction_corr is not None:
        figure_records.append(prediction_corr)
    figure_records.append(plot_3m_stability(portfolio, fig_dir))

    html_path = output_dir / "myquant_mined_factor_incremental_report.html"
    pdf_path = output_dir / "myquant_mined_factor_incremental_report.pdf"
    build_html_report(output_path=html_path, input_dir=input_dir, figure_records=figure_records, data=data)
    build_pdf_report(pdf_path, figure_records)
    return html_path, pdf_path


def main() -> None:
    html_path, pdf_path = build_all_reports()
    print(f"[VisualReport] html={html_path}")
    print(f"[VisualReport] pdf={pdf_path}")


if __name__ == "__main__":
    main()
