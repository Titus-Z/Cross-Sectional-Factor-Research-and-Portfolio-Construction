from __future__ import annotations

import math
from typing import Any, Mapping


REWARD_MODES = ("predictive_ic", "incremental_proxy")


def finite_float(value: object, default: float = 0.0) -> float:
    """把任意对象安全转成有限浮点数。

    自动因子挖掘会产生很多边界情况：某些日期横截面不足、某些分组没有股票、
    某些公式全是 NaN。reward 函数必须先消化这些异常值，不能把 NaN 传给排序器
    或 PPO policy。
    """

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(numeric):
        return float(default)
    return float(numeric)


def read_metric(metrics: Mapping[str, Any], name: str, prefix: str = "") -> float:
    """按统一规则读取指标列。

    同一个指标在不同脚本里可能叫 `pearson_ic_mean`、`train_pearson_ic_mean`
    或 `validation_pearson_ic_mean`。这里用 prefix 明确告诉 reward 函数读取哪一段
    数据，避免把 train、validation、OOS 混在一起。
    """

    return finite_float(metrics.get(f"{prefix}{name}"), 0.0)


def clipped_positive_ratio(value: float) -> float:
    """把正 IC 日期比例转换成 reward 中的小幅稳定性奖励。

    0.5 可以近似理解成随机方向；只有超过 0.5 的部分才给奖励。
    """

    return max(finite_float(value, 0.5) - 0.5, 0.0)


def compute_predictive_ic_reward(
    metrics: Mapping[str, Any],
    *,
    prefix: str = "",
    financial_logic_score: float = 0.0,
    complexity: float = 1.0,
    max_signal_corr_abs: float = 0.0,
) -> float:
    """第一套 reward：单因子预测力口径。

    这套 reward 用来回答：
    `这个公式本身是否能把未来收益排出来？`

    权重设计偏向 Pearson IC，同时保留 RankIC、long-short、单调性和正 IC
    日期比例。相关性和复杂度惩罚较轻，因为这套口径主要评估单因子强弱。
    """

    pearson_ic = read_metric(metrics, "pearson_ic_mean", prefix)
    spearman_ic = read_metric(metrics, "spearman_ic_mean", prefix)
    long_short = read_metric(metrics, "long_short_spread", prefix)
    monotonic = read_metric(metrics, "group_monotonic_spearman", prefix)
    pearson_positive = read_metric(metrics, "pearson_ic_positive_ratio", prefix)
    spearman_positive = read_metric(metrics, "spearman_ic_positive_ratio", prefix)

    reward = (
        pearson_ic
        + 0.50 * spearman_ic
        + 0.75 * long_short
        + 0.02 * monotonic
        + 0.05 * clipped_positive_ratio(pearson_positive)
        + 0.03 * clipped_positive_ratio(spearman_positive)
        + 0.02 * finite_float(financial_logic_score, 0.0)
        - 0.003 * max(finite_float(complexity, 1.0) - 1.0, 0.0)
        - 0.03 * finite_float(max_signal_corr_abs, 0.0)
    )
    return finite_float(reward, 0.0)


def compute_incremental_proxy_reward(
    metrics: Mapping[str, Any],
    *,
    prefix: str = "",
    financial_logic_score: float = 0.0,
    complexity: float = 1.0,
    max_signal_corr_abs: float = 0.0,
) -> float:
    """第二套 reward：模型/组合增量代理口径。

    这套 reward 用来回答：
    `这个公式是否更像能进入模型和组合层的增量信号？`

    它比 `predictive_ic` 更重视 RankIC、long-short、非重叠组合 Sharpe、
    去冗余和复杂度控制。它仍然是 proxy，因为真正的增量必须通过
    `baseline vs baseline + mined factors` 消融验证。
    """

    pearson_ic = read_metric(metrics, "pearson_ic_mean", prefix)
    spearman_ic = read_metric(metrics, "spearman_ic_mean", prefix)
    long_short = read_metric(metrics, "long_short_spread", prefix)
    monotonic = read_metric(metrics, "group_monotonic_spearman", prefix)
    pearson_ir = read_metric(metrics, "pearson_ic_ir", prefix)
    spearman_ir = read_metric(metrics, "spearman_ic_ir", prefix)
    pearson_positive = read_metric(metrics, "pearson_ic_positive_ratio", prefix)
    spearman_positive = read_metric(metrics, "spearman_ic_positive_ratio", prefix)
    non_overlap_sharpe = read_metric(metrics, "non_overlap_sharpe_horizon_adj", prefix)

    # Sharpe 的数值尺度可能远大于 IC，所以只给很小权重，并做截断。
    bounded_sharpe = max(min(non_overlap_sharpe, 5.0), -5.0)
    bounded_pearson_ir = max(min(pearson_ir, 3.0), -3.0)
    bounded_spearman_ir = max(min(spearman_ir, 3.0), -3.0)

    reward = (
        0.25 * pearson_ic
        + 0.75 * spearman_ic
        + 1.20 * long_short
        + 0.05 * monotonic
        + 0.025 * bounded_pearson_ir
        + 0.035 * bounded_spearman_ir
        + 0.010 * bounded_sharpe
        + 0.04 * clipped_positive_ratio(pearson_positive)
        + 0.06 * clipped_positive_ratio(spearman_positive)
        + 0.04 * finite_float(financial_logic_score, 0.0)
        - 0.012 * max(finite_float(complexity, 1.0) - 1.0, 0.0)
        - 0.12 * finite_float(max_signal_corr_abs, 0.0)
    )
    return finite_float(reward, 0.0)


def compute_reward(
    metrics: Mapping[str, Any],
    *,
    reward_mode: str,
    prefix: str = "",
    financial_logic_score: float = 0.0,
    complexity: float = 1.0,
    max_signal_corr_abs: float = 0.0,
) -> float:
    """统一 reward 入口。

    `predictive_ic` 和 `incremental_proxy` 应该并排保留。以后对老师或面试官
    可以说：我没有只用一种主观打分，而是比较了“单因子预测力 reward”和
    “模型/组合增量代理 reward”两套选择逻辑。
    """

    if reward_mode == "predictive_ic":
        return compute_predictive_ic_reward(
            metrics,
            prefix=prefix,
            financial_logic_score=financial_logic_score,
            complexity=complexity,
            max_signal_corr_abs=max_signal_corr_abs,
        )
    if reward_mode == "incremental_proxy":
        return compute_incremental_proxy_reward(
            metrics,
            prefix=prefix,
            financial_logic_score=financial_logic_score,
            complexity=complexity,
            max_signal_corr_abs=max_signal_corr_abs,
        )
    raise ValueError(f"Unsupported reward_mode={reward_mode!r}. Expected one of {REWARD_MODES}.")
