"""Alpha191 专项诊断模块。

这个模块服务于一个更明确的项目升级目标：

- 不再只说“我实现了很多 Alpha191 因子”；
- 而是说明这些 Alpha 属于哪些经济/价量信号家族；
- 哪些 Alpha 在不同持有期上有 IC；
- 哪些 Alpha 高度重复；
- 哪些 Alpha 换手可能过高；
- 哪些 Alpha 在年份维度上出现衰减。

这些诊断结果比“Top 10 特征重要性”更适合写进简历，因为它们能体现：

1. 你不是盲目堆因子；
2. 你会检查因子的冗余、稳定性、持有期匹配和可交易性；
3. 你已经为后续自动挖掘因子建立了统一评价标准。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.progress import optional_progress


ALPHA_FAMILIES = [
    "momentum",
    "reversal",
    "volatility",
    "liquidity",
    "volume_price",
    "vwap_deviation",
    "range_position",
    "complex_mixed",
]


# Alpha191 里有少数公式会产生极端大数，例如除数非常接近 0 时。
# 这些极端值如果直接参与 `std` 或 `corr`，可能触发 overflow warning，
# 也可能让一个异常点主导 Pearson IC。
# 这里做的是诊断层面的数值保护：先把 inf 变成 NaN，再把极端值截断到一个很宽的范围。
# RankIC 基本不受这个截断影响，因为排名只关心顺序。
ALPHA_VALUE_CLIP = 1e12


def sanitize_alpha_values(values: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """清理 Alpha 数值，避免极端值破坏相关性、标准差和热力图计算。"""

    numeric = values.apply(pd.to_numeric, errors="coerce") if isinstance(values, pd.DataFrame) else pd.to_numeric(values, errors="coerce")
    return numeric.replace([np.inf, -np.inf], np.nan).clip(lower=-ALPHA_VALUE_CLIP, upper=ALPHA_VALUE_CLIP)


# 这份映射是人工规则，不声称是唯一标准。
# Alpha191 很多公式本身是混合型信号，例如同时包含 volume、rank、corr、decay、price range。
# 这里为项目报告提供稳定、可解释的分组口径，不声称是学术级分类标准。
ALPHA_FAMILY_BY_NUMBER: dict[int, str] = {
    # 动量 / 趋势延续：以 close 差分、过去收益、均线偏离等为核心。
    6: "momentum",
    8: "momentum",
    14: "momentum",
    18: "momentum",
    20: "momentum",
    24: "momentum",
    29: "momentum",
    31: "momentum",
    34: "momentum",
    65: "momentum",
    66: "momentum",
    71: "momentum",
    88: "momentum",
    106: "momentum",
    134: "momentum",
    151: "momentum",
    153: "momentum",
    167: "momentum",
    178: "momentum",
    # 反转 / 超买超卖：以过去涨跌、RSI/KDJ/回撤或短期反转结构为核心。
    3: "reversal",
    19: "reversal",
    28: "reversal",
    33: "reversal",
    47: "reversal",
    49: "reversal",
    50: "reversal",
    51: "reversal",
    53: "reversal",
    57: "reversal",
    58: "reversal",
    59: "reversal",
    63: "reversal",
    67: "reversal",
    79: "reversal",
    82: "reversal",
    96: "reversal",
    103: "reversal",
    112: "reversal",
    129: "reversal",
    160: "reversal",
    162: "reversal",
    164: "reversal",
    172: "reversal",
    174: "reversal",
    177: "reversal",
    186: "reversal",
    187: "reversal",
    # 波动率 / 风险结构：以标准差、区间波动、协方差或真实波幅为核心。
    10: "volatility",
    42: "volatility",
    54: "volatility",
    70: "volatility",
    76: "volatility",
    83: "volatility",
    95: "volatility",
    97: "volatility",
    100: "volatility",
    104: "volatility",
    109: "volatility",
    127: "volatility",
    158: "volatility",
    161: "volatility",
    175: "volatility",
    188: "volatility",
    189: "volatility",
    # 流动性 / 成交活跃度：以 volume、turnover、amount 变化本身为核心。
    40: "liquidity",
    43: "liquidity",
    80: "liquidity",
    81: "liquidity",
    84: "liquidity",
    94: "liquidity",
    102: "liquidity",
    145: "liquidity",
    155: "liquidity",
    168: "liquidity",
    # 价量关系：价格、收益、high/low 与 volume 的相关、协方差或条件成交量。
    1: "volume_price",
    5: "volume_price",
    9: "volume_price",
    11: "volume_price",
    16: "volume_price",
    32: "volume_price",
    36: "volume_price",
    44: "volume_price",
    45: "volume_price",
    48: "volume_price",
    52: "volume_price",
    60: "volume_price",
    62: "volume_price",
    68: "volume_price",
    74: "volume_price",
    90: "volume_price",
    99: "volume_price",
    101: "volume_price",
    105: "volume_price",
    108: "volume_price",
    110: "volume_price",
    111: "volume_price",
    113: "volume_price",
    115: "volume_price",
    117: "volume_price",
    118: "volume_price",
    119: "volume_price",
    121: "volume_price",
    123: "volume_price",
    128: "volume_price",
    130: "volume_price",
    136: "volume_price",
    139: "volume_price",
    141: "volume_price",
    142: "volume_price",
    150: "volume_price",
    163: "volume_price",
    170: "volume_price",
    176: "volume_price",
    179: "volume_price",
    180: "volume_price",
    191: "volume_price",
    # VWAP 偏离：以 VWAP-close、VWAP-open 或 VWAP 变化为核心。
    7: "vwap_deviation",
    12: "vwap_deviation",
    13: "vwap_deviation",
    17: "vwap_deviation",
    26: "vwap_deviation",
    41: "vwap_deviation",
    61: "vwap_deviation",
    73: "vwap_deviation",
    77: "vwap_deviation",
    87: "vwap_deviation",
    92: "vwap_deviation",
    120: "vwap_deviation",
    124: "vwap_deviation",
    125: "vwap_deviation",
    156: "vwap_deviation",
    # 区间位置 / 通道结构：以 high-low、close 在区间内的位置、通道宽度等为核心。
    21: "range_position",
    38: "range_position",
    46: "range_position",
    72: "range_position",
    78: "range_position",
    93: "range_position",
    133: "range_position",
    159: "range_position",
    173: "range_position",
    184: "range_position",
    185: "range_position",
}


def extract_alpha_number(alpha_name: str) -> int | None:
    """从 `alpha001` 这类列名里提取数字编号。"""

    lowered = alpha_name.lower().strip()
    if not lowered.startswith("alpha"):
        return None
    suffix = lowered.replace("alpha", "", 1)
    if not suffix.isdigit():
        return None
    return int(suffix)


def classify_alpha_family(alpha_name: str) -> str:
    """给单个 Alpha 分配一个主家族。"""

    alpha_number = extract_alpha_number(alpha_name)
    if alpha_number is None:
        return "complex_mixed"
    return ALPHA_FAMILY_BY_NUMBER.get(alpha_number, "complex_mixed")


def build_alpha_family_map(alpha_columns: list[str]) -> pd.DataFrame:
    """构造 Alpha 家族映射表。"""

    records = []
    for alpha_name in alpha_columns:
        alpha_number = extract_alpha_number(alpha_name)
        family = classify_alpha_family(alpha_name)
        records.append(
            {
                "alpha_name": alpha_name,
                "alpha_number": alpha_number,
                "family": family,
                "is_manually_classified": bool(alpha_number in ALPHA_FAMILY_BY_NUMBER) if alpha_number is not None else False,
            }
        )
    return pd.DataFrame(records).sort_values(["family", "alpha_number", "alpha_name"]).reset_index(drop=True)


def _safe_ic_summary(ic_series: pd.Series, prefix: str) -> dict[str, Any]:
    """把逐日 IC 序列压缩成摘要指标。"""

    valid = pd.to_numeric(ic_series, errors="coerce").dropna()
    if valid.empty:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_icir": np.nan,
            f"{prefix}_positive_ratio": np.nan,
            f"{prefix}_days": 0,
        }

    ic_mean = float(valid.mean())
    ic_std = float(valid.std(ddof=0))
    return {
        f"{prefix}_mean": ic_mean,
        f"{prefix}_median": float(valid.median()),
        f"{prefix}_std": ic_std,
        f"{prefix}_icir": float(ic_mean / ic_std) if abs(ic_std) > 1e-12 else np.nan,
        f"{prefix}_positive_ratio": float((valid > 0).mean()),
        f"{prefix}_days": int(valid.shape[0]),
    }


def compute_daily_alpha_ic(
    data: pd.DataFrame,
    alpha_columns: list[str],
    target_columns: dict[int, str],
    subset_label: str,
    min_cross_section: int = 30,
    show_progress: bool = True,
) -> pd.DataFrame:
    """按日期批量计算 Alpha IC。

    这里不用逐个 Alpha 循环算相关性，而是在每个日期里用 `corrwith`
    一次性计算所有 Alpha 和目标列的横截面相关。
    这样在 176 个 Alpha 上会明显更快。
    """

    records: list[dict[str, Any]] = []
    grouped_items = list(data.groupby("date", sort=True))

    for current_date, date_slice in optional_progress(
        grouped_items,
        description=f"Daily alpha IC ({subset_label})",
        enabled=show_progress,
        total=len(grouped_items),
    ):
        alpha_block = sanitize_alpha_values(date_slice[alpha_columns])
        if alpha_block.notna().sum(axis=0).max() < min_cross_section:
            continue

        for horizon, target_column in target_columns.items():
            if target_column not in date_slice.columns:
                continue

            target = pd.to_numeric(date_slice[target_column], errors="coerce")
            target = target.replace([np.inf, -np.inf], np.nan)
            valid_target = target.notna()
            if int(valid_target.sum()) < min_cross_section:
                continue

            target_valid = target.loc[valid_target]
            if float(target_valid.std(ddof=0)) <= 1e-12:
                # 单日目标值没有横截面差异时，IC 没有定义。
                continue

            alpha_valid = alpha_block.loc[valid_target]
            alpha_std = alpha_valid.std(axis=0, ddof=0).replace([np.inf, -np.inf], np.nan)
            enough_alpha = (alpha_valid.notna().sum(axis=0) >= min_cross_section) & (alpha_std > 1e-12)
            alpha_valid = alpha_valid.loc[:, enough_alpha]
            if alpha_valid.empty:
                continue

            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                pearson_values = alpha_valid.corrwith(target_valid, axis=0, method="pearson")
                rank_values = alpha_valid.rank(axis=0, pct=True).corrwith(target_valid.rank(pct=True), axis=0, method="pearson")

            for alpha_name, ic_value in pearson_values.dropna().items():
                records.append(
                    {
                        "subset": subset_label,
                        "date": pd.Timestamp(current_date),
                        "target_horizon": int(horizon),
                        "target_column": target_column,
                        "alpha_name": alpha_name,
                        "method": "pearson",
                        "ic": float(ic_value),
                    }
                )

            for alpha_name, ic_value in rank_values.dropna().items():
                records.append(
                    {
                        "subset": subset_label,
                        "date": pd.Timestamp(current_date),
                        "target_horizon": int(horizon),
                        "target_column": target_column,
                        "alpha_name": alpha_name,
                        "method": "rank",
                        "ic": float(ic_value),
                    }
                )

    return pd.DataFrame(records)


def summarize_alpha_ic(
    daily_ic_df: pd.DataFrame,
    alpha_family_map: pd.DataFrame,
) -> pd.DataFrame:
    """汇总 Alpha 在每个 horizon / subset 上的 IC 表现。"""

    if daily_ic_df.empty:
        return pd.DataFrame()

    summary_records: list[dict[str, Any]] = []
    for (subset_label, horizon, alpha_name), group in daily_ic_df.groupby(["subset", "target_horizon", "alpha_name"]):
        pearson_series = group.loc[group["method"] == "pearson", "ic"]
        rank_series = group.loc[group["method"] == "rank", "ic"]
        summary_records.append(
            {
                "subset": subset_label,
                "target_horizon": int(horizon),
                "alpha_name": alpha_name,
                **_safe_ic_summary(pearson_series, "pearson_ic"),
                **_safe_ic_summary(rank_series, "rank_ic"),
            }
        )

    summary_df = pd.DataFrame(summary_records)
    summary_df = summary_df.merge(alpha_family_map[["alpha_name", "family"]], on="alpha_name", how="left")
    return summary_df.sort_values(["subset", "target_horizon", "rank_ic_mean"], ascending=[True, True, False]).reset_index(drop=True)


def build_horizon_match_table(summary_df: pd.DataFrame, subset_label: str = "train") -> pd.DataFrame:
    """把不同 horizon 的表现拼到一张宽表里，找每个 Alpha 最匹配的持有期。"""

    if summary_df.empty:
        return pd.DataFrame()

    subset_df = summary_df[summary_df["subset"] == subset_label].copy()
    if subset_df.empty:
        return pd.DataFrame()

    pivot = subset_df.pivot_table(
        index=["alpha_name", "family"],
        columns="target_horizon",
        values="rank_ic_mean",
        aggfunc="mean",
    ).reset_index()
    pivot.columns = [f"IC_{column}d" if isinstance(column, int) else column for column in pivot.columns]

    horizon_columns = [column for column in pivot.columns if column.startswith("IC_")]
    if horizon_columns:
        best_column = pivot[horizon_columns].idxmax(axis=1)
        pivot["best_horizon"] = best_column.str.extract(r"IC_(\d+)d").astype(float).astype("Int64")
        pivot["best_rank_ic"] = pivot[horizon_columns].max(axis=1)
    return pivot.sort_values("best_rank_ic", ascending=False, na_position="last").reset_index(drop=True)


def compute_alpha_turnover_proxy(
    data: pd.DataFrame,
    alpha_columns: list[str],
    top_fraction: float = 0.2,
    show_progress: bool = True,
) -> pd.DataFrame:
    """用排名变化和 Top 组留存率估计 Alpha 的换手压力。

    这里不是实际交易换手，只是一个因子层面的 proxy：

    - `rank_turnover` 越高，说明 Alpha 排名每天变化越大；
    - `top_retention` 越低，说明 Top 20% 股票留存越差，组合调仓压力越大。
    """

    if not 0.0 < top_fraction < 1.0:
        raise ValueError("top_fraction must be inside (0, 1).")

    sorted_data = data.sort_values(["instrument_id", "date"]).copy()
    turnover_records: list[dict[str, Any]] = []
    date_group_indices = list(sorted_data.groupby("date", sort=True).indices.items())

    for alpha_name in optional_progress(
        alpha_columns,
        description="Alpha turnover proxy",
        enabled=show_progress,
        total=len(alpha_columns),
    ):
        alpha_series = sanitize_alpha_values(sorted_data[alpha_name])
        rank_series = alpha_series.groupby(sorted_data["date"]).rank(pct=True)
        rank_turnover = rank_series.groupby(sorted_data["instrument_id"]).diff().abs().mean()

        top_sets: list[set[str]] = []
        for _, row_index in date_group_indices:
            # 这里使用 groupby 预先给出的行号，而不是在循环里反复执行
            # `sorted_data[sorted_data["date"] == current_date]`。
            # 后者会在每个 Alpha、每个日期都扫描整张表，数据稍大就会非常慢。
            date_slice = sorted_data.iloc[row_index]
            valid = date_slice[["instrument_id"]].copy()
            valid[alpha_name] = sanitize_alpha_values(date_slice[alpha_name])
            valid = valid.dropna()
            if valid.empty:
                top_sets.append(set())
                continue
            top_count = max(1, int(np.ceil(len(valid) * top_fraction)))
            top_names = set(valid.nlargest(top_count, alpha_name)["instrument_id"].astype(str))
            top_sets.append(top_names)

        retention_values = []
        for previous, current in zip(top_sets[:-1], top_sets[1:]):
            if not previous or not current:
                continue
            retention_values.append(len(previous.intersection(current)) / max(len(previous), 1))

        turnover_records.append(
            {
                "alpha_name": alpha_name,
                "family": classify_alpha_family(alpha_name),
                "rank_turnover": float(rank_turnover) if pd.notna(rank_turnover) else np.nan,
                "top_retention": float(np.mean(retention_values)) if retention_values else np.nan,
                "top_fraction": float(top_fraction),
                "dates_used": int(len(date_group_indices)),
            }
        )

    return pd.DataFrame(turnover_records).sort_values("rank_turnover", ascending=True, na_position="last").reset_index(drop=True)


def compute_alpha_correlation(
    data: pd.DataFrame,
    alpha_columns: list[str],
    max_rows: int = 120_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """计算训练期 Alpha 相关性矩阵。

    为了避免不同时期分布变化污染相关性，先在每个日期横截面内做 z-score。
    当样本行数过多时，再抽样计算相关性，避免内存过高。
    """

    alpha_block = data[["date", *alpha_columns]].copy()
    zscore_frames: list[pd.DataFrame] = []

    for _, date_slice in alpha_block.groupby("date", sort=True):
        values = sanitize_alpha_values(date_slice[alpha_columns])
        means = values.mean(axis=0)
        stds = values.std(axis=0, ddof=0).replace(0.0, np.nan)
        zscore_frames.append((values - means) / stds)

    zscore_df = pd.concat(zscore_frames, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    if len(zscore_df) > max_rows:
        zscore_df = zscore_df.sample(n=max_rows, random_state=random_state)

    return zscore_df.corr(method="pearson")


def build_correlation_clusters(
    correlation_df: pd.DataFrame,
    alpha_family_map: pd.DataFrame,
    ic_summary_df: pd.DataFrame | None = None,
    subset_label: str = "train",
    target_horizon: int = 10,
    threshold: float = 0.9,
) -> pd.DataFrame:
    """用简单贪心规则把高相关 Alpha 聚类。

    这里不用复杂层次聚类，原因是报告目标是找重复因子，
    贪心聚类已经足够回答“哪些 Alpha 大概率是同一类信号”。
    """

    if correlation_df.empty:
        return pd.DataFrame()

    score_map: dict[str, float] = {}
    if ic_summary_df is not None and not ic_summary_df.empty:
        score_source = ic_summary_df[
            (ic_summary_df["subset"] == subset_label)
            & (ic_summary_df["target_horizon"] == target_horizon)
        ]
        score_map = dict(zip(score_source["alpha_name"], score_source["rank_ic_mean"]))

    family_map = dict(zip(alpha_family_map["alpha_name"], alpha_family_map["family"]))
    unassigned = set(correlation_df.columns)
    cluster_records: list[dict[str, Any]] = []
    cluster_id = 1

    while unassigned:
        seed = sorted(unassigned)[0]
        related = set(correlation_df.index[correlation_df.loc[seed].abs() >= threshold]).intersection(unassigned)
        if not related:
            related = {seed}

        representative = max(related, key=lambda name: score_map.get(name, -np.inf))
        family_counter = Counter(family_map.get(name, "complex_mixed") for name in related)
        dominant_family = family_counter.most_common(1)[0][0] if family_counter else "complex_mixed"

        for alpha_name in sorted(related):
            cluster_records.append(
                {
                    "cluster_id": cluster_id,
                    "alpha_name": alpha_name,
                    "family": family_map.get(alpha_name, "complex_mixed"),
                    "cluster_size": len(related),
                    "representative_alpha": representative,
                    "dominant_family": dominant_family,
                    "representative_rank_ic": score_map.get(representative, np.nan),
                    "max_abs_corr_to_representative": float(abs(correlation_df.loc[alpha_name, representative])),
                    "threshold": float(threshold),
                }
            )

        unassigned -= related
        cluster_id += 1

    return pd.DataFrame(cluster_records).sort_values(["cluster_size", "cluster_id"], ascending=[False, True]).reset_index(drop=True)


def plot_alpha_correlation_heatmap(
    correlation_df: pd.DataFrame,
    output_path: str | Path,
    alpha_order: list[str] | None = None,
    max_alpha_count: int = 60,
) -> None:
    """保存 Alpha 相关性热力图。"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if correlation_df.empty:
        return

    if alpha_order:
        ordered_names = [name for name in alpha_order if name in correlation_df.columns][:max_alpha_count]
    else:
        ordered_names = list(correlation_df.columns[:max_alpha_count])

    if not ordered_names:
        return

    heatmap_data = correlation_df.loc[ordered_names, ordered_names]
    fig_size = max(8, min(18, len(ordered_names) * 0.22))
    plt.figure(figsize=(fig_size, fig_size))
    plt.imshow(heatmap_data, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(ordered_names)), ordered_names, rotation=90, fontsize=6)
    plt.yticks(range(len(ordered_names)), ordered_names, fontsize=6)
    plt.title("Alpha Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def build_decay_table(
    daily_ic_df: pd.DataFrame,
    target_horizon: int,
    method: str = "rank",
    period: str = "Y",
) -> pd.DataFrame:
    """按年份或季度汇总 Alpha IC 衰减。"""

    if daily_ic_df.empty:
        return pd.DataFrame()

    filtered = daily_ic_df[
        (daily_ic_df["target_horizon"] == target_horizon)
        & (daily_ic_df["method"] == method)
    ].copy()
    if filtered.empty:
        return pd.DataFrame()

    filtered["date"] = pd.to_datetime(filtered["date"])
    if period.upper() == "Q":
        filtered["period"] = filtered["date"].dt.to_period("Q").astype(str)
    else:
        filtered["period"] = filtered["date"].dt.year.astype(str)

    decay_df = (
        filtered.groupby(["subset", "period", "alpha_name"], as_index=False)["ic"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "ic_mean", "median": "ic_median", "std": "ic_std", "count": "ic_days"})
    )
    decay_df["family"] = decay_df["alpha_name"].map(classify_alpha_family)
    return decay_df


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    """把小表格转成 Markdown，避免报告里塞入过长表格。"""

    if df.empty:
        return "_No data available._"

    clipped = df.head(max_rows).copy()
    headers = " | ".join(clipped.columns)
    separators = " | ".join(["---"] * len(clipped.columns))
    rows = [" | ".join(str(value) for value in row) for row in clipped.astype(str).itertuples(index=False, name=None)]
    return "\n".join([f"| {headers} |", f"| {separators} |"] + [f"| {row} |" for row in rows])


def write_alpha_diagnostics_report(
    output_path: str | Path,
    dataset_summary: dict[str, Any],
    alpha_family_map: pd.DataFrame,
    ic_summary_df: pd.DataFrame,
    horizon_match_df: pd.DataFrame,
    turnover_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    decay_df: pd.DataFrame,
    main_horizon: int,
) -> None:
    """写出一个面向项目展示和简历表达的 Alpha 诊断报告。"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    family_counts = alpha_family_map["family"].value_counts().rename_axis("family").reset_index(name="alpha_count")

    train_main = ic_summary_df[
        (ic_summary_df["subset"] == "train")
        & (ic_summary_df["target_horizon"] == main_horizon)
    ].copy()
    oos_main = ic_summary_df[
        (ic_summary_df["subset"] == "oos")
        & (ic_summary_df["target_horizon"] == main_horizon)
    ].copy()

    top_train = train_main.sort_values("rank_ic_mean", ascending=False)[
        ["alpha_name", "family", "rank_ic_mean", "rank_ic_positive_ratio", "rank_ic_icir"]
    ].head(15)
    top_oos = oos_main.sort_values("rank_ic_mean", ascending=False)[
        ["alpha_name", "family", "rank_ic_mean", "rank_ic_positive_ratio", "rank_ic_icir"]
    ].head(15)

    family_performance = (
        train_main.groupby("family", as_index=False)
        .agg(
            alpha_count=("alpha_name", "count"),
            mean_rank_ic=("rank_ic_mean", "mean"),
            median_rank_ic=("rank_ic_mean", "median"),
            positive_alpha_ratio=("rank_ic_mean", lambda values: float((values > 0).mean())),
        )
        .sort_values("mean_rank_ic", ascending=False)
    )

    large_clusters = cluster_df[cluster_df["cluster_size"] >= 2][
        ["cluster_id", "cluster_size", "dominant_family", "representative_alpha", "representative_rank_ic"]
    ].drop_duplicates("cluster_id").sort_values("cluster_size", ascending=False)

    low_turnover = turnover_df.sort_values(["rank_turnover", "top_retention"], ascending=[True, False])[
        ["alpha_name", "family", "rank_turnover", "top_retention"]
    ].head(15)

    recent_decay = decay_df[decay_df["subset"] == "oos"].sort_values(["period", "ic_mean"], ascending=[False, False])
    if recent_decay.empty:
        recent_decay = decay_df.sort_values(["period", "ic_mean"], ascending=[False, False])
    recent_decay = recent_decay[["period", "alpha_name", "family", "ic_mean", "ic_days"]].head(20)

    report_text = f"""# Alpha191 Diagnostics Report

## 1. Resume-Ready Bullets

- Built an Alpha191 diagnostic framework that classifies formulaic alphas into signal families, evaluates horizon-specific IC/RankIC, detects redundant alpha clusters, and estimates turnover pressure using rank-change proxies.
- Added train/OOS alpha decay checks to test whether classic formulaic alphas remain effective on the current US large-cap universe instead of assuming historical Alpha101/Alpha191 formulas still work.
- Connected alpha diagnostics to future automated factor mining by defining reusable evaluation gates: IC stability, horizon match, redundancy, family contribution, and turnover proxy.

## 2. Dataset And Scope

- Data path: `{dataset_summary.get("data_path")}`
- Date range: `{dataset_summary.get("min_date")}` to `{dataset_summary.get("max_date")}`
- Train rows: `{dataset_summary.get("train_rows")}`
- OOS rows: `{dataset_summary.get("oos_rows")}`
- Train date range: `{dataset_summary.get("train_min_date")}` to `{dataset_summary.get("train_max_date")}`
- OOS date range: `{dataset_summary.get("oos_min_date")}` to `{dataset_summary.get("oos_max_date")}`
- Target horizons: `{dataset_summary.get("target_horizons")}`
- Alpha count diagnosed: `{dataset_summary.get("alpha_count")}`
- Main horizon for ranking: `{main_horizon}d`

## 3. Alpha Family Map

{dataframe_to_markdown(family_counts, max_rows=20)}

## 4. Train IC Distribution By Family

{dataframe_to_markdown(family_performance, max_rows=20)}

## 5. Best Train Alpha By {main_horizon}d RankIC

{dataframe_to_markdown(top_train, max_rows=15)}

## 6. Best OOS Alpha By {main_horizon}d RankIC

{dataframe_to_markdown(top_oos, max_rows=15)}

## 7. Horizon Match

{dataframe_to_markdown(horizon_match_df[["alpha_name", "family", "best_horizon", "best_rank_ic"]], max_rows=20)}

## 8. Redundancy Clusters

{dataframe_to_markdown(large_clusters, max_rows=20)}

## 9. Turnover Proxy

{dataframe_to_markdown(low_turnover, max_rows=15)}

## 10. Recent Decay Snapshot

{dataframe_to_markdown(recent_decay, max_rows=20)}

## 11. Reading Order

1. 先看 `Alpha Family Map`：确认 Alpha191 不再是黑箱公式堆，而是有可解释家族。
2. 再看 `Train IC Distribution By Family`：判断哪类 Alpha 在训练期整体更有信号。
3. 再看 `Best OOS Alpha`：确认训练期强信号是否能延续到 OOS。
4. 再看 `Horizon Match`：不要把短周期 Alpha 强行塞进 10d 或 20d 标签。
5. 再看 `Redundancy Clusters`：高相关 Alpha 只保留代表项，为后续 factor synergy 做准备。
6. 最后看 `Turnover Proxy`：排名变化太快的 Alpha 即使 IC 好，也可能在成本后失效。

## 12. Recommended Follow-up

建议继续运行 `Alpha family ablation`：

```text
baseline technical indicators
baseline + selected momentum alpha
baseline + selected reversal alpha
baseline + selected volatility alpha
baseline + selected volume_price alpha
baseline + all selected non-redundant alpha
```

这个实验用于回答一个产品内研究问题：

```text
这些 Alpha191 公式到底给模型带来了多少增量贡献？
```
"""

    output_path.write_text(report_text, encoding="utf-8")


def write_resume_bullet_report(output_path: str | Path) -> None:
    """单独写一份可直接改进简历的 bullet 草稿。"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        """# Resume Bullets - Alpha Diagnostics

- Built an Alpha191 diagnostic framework for a US large-cap return prediction pipeline, classifying formulaic alphas into signal families and evaluating horizon-specific IC, RankIC, redundancy clusters, decay, and turnover proxies.
- Designed an AlphaEval-style gating process for formulaic factors, filtering classic and newly mined alpha candidates by OOS stability, horizon match, correlation redundancy, and rank-turnover pressure before model inclusion.
- Extended the factor research workflow from raw Alpha191 replication to reusable alpha validation, enabling family-level ablation and automated factor mining with consistent evaluation standards.
""",
        encoding="utf-8",
    )
