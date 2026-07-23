"""时间序列训练流水线辅助模块。

这个模块专门解决量化项目里最常见、也最隐蔽的一类问题：

1. 先在全量数据上生成滚动特征；
2. 再切 train / test；
3. 表面上看是时间切分，实际上却很难证明流程没有信息泄露。

为了让项目更“真实可信”，这里把最关键的约束写成明确代码：

- 必须先按日期切分原始数据；
- 训练集特征只能由训练期原始数据生成；
- 测试集特征只能由“训练期历史上下文 + 测试期当前及过去数据”生成；
- 绝不允许为了算测试期特征而把未来测试日期之后的数据混进来。
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.data_loader import time_based_train_test_split
from src.feature_generator import generate_feature_matrix
from src.progress import create_progress_bar


# 这里给一个比较保守的历史缓冲窗口。
# 原因是 Alpha191 和部分技术指标里已经出现了 180 / 230 / 250 这类长窗口。
# 如果测试集开头没有足够的训练期历史，滚动特征就会在测试期前几天退化得很严重。
DEFAULT_HISTORY_WINDOW = 260


def _sort_market_data(data: pd.DataFrame) -> pd.DataFrame:
    """按股票和日期排序，确保后续滚动计算方向一致。"""

    return data.sort_values(["instrument_id", "date"]).reset_index(drop=True)


def append_history_context(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    history_window: int = DEFAULT_HISTORY_WINDOW,
) -> pd.DataFrame:
    """为测试集拼接一段仅来自训练期的历史上下文。

    为什么需要这一步：

    - 如果直接只拿测试集去算滚动均值 / Alpha191，测试期开头一段会缺历史；
    - 如果把“整个全量数据”都拿来算，又会让流程变得不够透明，难以证明没有 leakage；
    - 所以这里折中成“只拿训练期最后 `history_window` 条历史 + 完整测试期”。

    这样测试样本在每个时点只能看到：

    - 自己当前时点；
    - 更早的历史；
    - 绝看不到未来测试日期之后的数据。
    """

    if train_raw.empty or test_raw.empty:
        raise ValueError("Both train_raw and test_raw must contain data.")

    if history_window <= 0:
        raise ValueError("history_window must be a positive integer.")

    sorted_train = _sort_market_data(train_raw)
    sorted_test = _sort_market_data(test_raw)

    history_context = (
        sorted_train.groupby("instrument_id", group_keys=False)
        .tail(history_window)
        .reset_index(drop=True)
    )

    context_df = pd.concat([history_context, sorted_test], ignore_index=True)
    context_df = (
        context_df.sort_values(["instrument_id", "date"])
        .drop_duplicates(subset=["instrument_id", "date"], keep="last")
        .reset_index(drop=True)
    )
    return context_df


def purge_training_label_overlap(
    train_raw: pd.DataFrame,
    target_horizon: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """按股票移除训练期末尾会跨越下一时间块的监督样本。

    ``y_10d[t]`` 使用 ``t+10`` 的收盘价。如果保留切分点前最后 10 个
    交易日作为训练样本，它们的标签会读取 OOS 内价格。即使特征完全只看
    历史，这仍属于 label overlap leakage。

    标签使用统一市场交易日历，但按股票删除训练块末尾 N 条记录仍是一条
    保守边界：活跃股票恰好删除最后 N 个全市场日期，缺行股票可能多删除
    少量更早样本，却不会把跨界标签留在模型拟合中。

    被 purge 的记录仍可作为 OOS 滚动特征的历史上下文，因为当天 OHLCV
    在预测时已经可见；它们只是不能带着跨边界标签进入模型拟合。
    """

    if target_horizon < 0:
        raise ValueError("target_horizon must be non-negative.")
    if target_horizon == 0:
        return train_raw.copy(), {"purged_date_count": 0, "purged_row_count": 0}

    sorted_train = _sort_market_data(train_raw)
    purge_index = (
        sorted_train.groupby("instrument_id", group_keys=False)
        .tail(target_horizon)
        .index
    )
    keep_mask = ~sorted_train.index.isin(purge_index)
    purged_rows = sorted_train.loc[~keep_mask].copy()
    purged_train = sorted_train.loc[keep_mask].copy().reset_index(drop=True)
    if purged_train.empty:
        raise ValueError("Training period is too short for the requested per-instrument purge.")

    purged_dates = pd.Index(sorted(pd.to_datetime(purged_rows["date"]).dropna().unique()))
    return purged_train, {
        "purged_date_count": int(len(purged_dates)),
        "purged_row_count": int((~keep_mask).sum()),
        "first_purged_date": str(pd.Timestamp(purged_dates[0]).date()),
        "last_purged_date": str(pd.Timestamp(purged_dates[-1]).date()),
        "train_max_date_after_purge": str(pd.to_datetime(purged_train["date"]).max().date()),
        "purge_policy": "last_target_horizon_rows_per_instrument",
    }


def strict_time_split_feature_engineering(
    raw_data: pd.DataFrame,
    test_size: float = 0.2,
    history_window: int = DEFAULT_HISTORY_WINDOW,
    test_start_date: str | pd.Timestamp | None = None,
    target_horizon: int = 0,
    alpha_factor_names: list[str] | None = None,
    show_progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    """先切原始数据，再分别生成训练集和测试集特征。

    这是当前项目最重要的 leakage 防线。

    返回值包含：

    - `train_feature_df`：只基于训练期原始数据生成的训练特征；
    - `test_feature_df`：基于“训练期历史上下文 + 测试期”生成，并裁剪回真实测试日期的测试特征；
    - `feature_columns`：训练时允许进入模型的候选特征列表；
    - `feature_metadata`：特征分组说明。
    """

    stage_progress = create_progress_bar(
        total=5,
        description="Feature engineering stages",
        enabled=show_progress,
    )

    sorted_raw_data = _sort_market_data(raw_data)
    train_raw_with_context, test_raw = time_based_train_test_split(
        sorted_raw_data,
        test_size=test_size,
        test_start_date=test_start_date,
    )
    stage_progress.update(1)
    stage_progress.set_postfix_str("raw train/test split finished")

    if train_raw_with_context.empty or test_raw.empty:
        raise ValueError("Time-based split produced an empty train or test set.")

    # 训练样本需要额外 purge 一个标签周期，阻止 y_10d 等标签跨过 OOS
    # 边界。完整的切分前训练尾部仍保留给测试滚动特征作为历史上下文。
    train_raw, purge_summary = purge_training_label_overlap(
        train_raw_with_context,
        target_horizon=target_horizon,
    )

    # 训练集特征严格只由训练期原始数据生成。
    train_feature_df, feature_columns, feature_metadata = generate_feature_matrix(
        train_raw,
        alpha_factor_names=alpha_factor_names,
        show_progress=show_progress,
    )
    stage_progress.update(1)
    stage_progress.set_postfix_str("train features finished")

    # 测试集特征只允许看到：
    # 1. 训练期最后一段历史；
    # 2. 测试期自己及其过去。
    test_with_context = append_history_context(
        train_raw=train_raw_with_context,
        test_raw=test_raw,
        history_window=history_window,
    )
    stage_progress.update(1)
    stage_progress.set_postfix_str("history context prepared")
    test_feature_context_df, _, _ = generate_feature_matrix(
        test_with_context,
        alpha_factor_names=alpha_factor_names,
        show_progress=show_progress,
    )
    stage_progress.update(1)
    stage_progress.set_postfix_str("test/context features finished")

    # 这里务必把历史上下文部分裁掉，只保留真实测试日期对应的样本。
    # 否则后续评估时会把训练期上下文样本也混进去。
    test_dates = pd.Index(pd.to_datetime(test_raw["date"]).unique())
    test_feature_df = test_feature_context_df[pd.to_datetime(test_feature_context_df["date"]).isin(test_dates)].copy()

    # 删除无法监督训练 / 评估的尾部无标签样本。
    train_feature_df = train_feature_df.dropna(subset=["y"]).reset_index(drop=True)
    test_feature_df = test_feature_df.dropna(subset=["y"]).reset_index(drop=True)

    # 为了防止训练集和测试集列集合意外不一致，这里显式对齐候选特征。
    for feature_name in feature_columns:
        if feature_name not in test_feature_df.columns:
            test_feature_df[feature_name] = pd.NA

    stage_progress.update(1)
    stage_progress.set_postfix_str("feature alignment finished")
    stage_progress.close()

    feature_metadata = dict(feature_metadata)
    feature_metadata["temporal_purge"] = {
        "target_horizon": int(target_horizon),
        **purge_summary,
    }

    return train_feature_df, test_feature_df, feature_columns, feature_metadata
