"""横截面预处理模块。

这个模块专门处理“同一天、不同股票之间”的特征清洗与标准化问题。

在量化横截面建模里，常见做法不是直接把原始因子值喂给模型，
而是先做几步更稳健的预处理：

1. Winsorize：裁剪极端值，避免少数异常股票把当天分布拖歪；
2. Z-score：把每个日期横截面的因子值标准化到可比较尺度；
3. Neutralize：剥离可用的行业暴露；只有真实市值覆盖充分时才剥离规模暴露。

为什么这些操作按“日期横截面”做，而不是按全样本一起做？

- 因子模型通常在每个交易日里做股票排序；
- 不同日期的市场整体水平不一样，直接跨日期混在一起标准化会失真；
- 同一天你能同时观察到所有股票的特征，因此按当天横截面处理是合理的。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.progress import optional_progress


DEFAULT_WINSORIZE_QUANTILE = 0.01
MIN_MARKET_CAP_COVERAGE_FOR_NEUTRALIZATION = 0.90
# A model feature named ``market_cap`` may itself be standardized. Keep the
# observable raw exposure in a non-feature column so later factor blocks can use
# the same size-neutralization regression instead of a transformed value.
RAW_MARKET_CAP_EXPOSURE_COLUMN = "_raw_market_cap_exposure"


def _safe_zscore(series: pd.Series) -> pd.Series:
    """对单个序列做安全标准化。

    如果当天某个暴露变量完全没有波动，例如：

    - 所有股票的市值代理值意外相同；
    - 只有一只股票有有效值；

    那么标准差会是 0，这时直接除法会产生无穷大或 NaN。
    这里统一做保护：如果标准差不可用，就只做去均值，不再除以标准差。
    """

    valid = pd.to_numeric(series, errors="coerce")
    mean_value = valid.mean()
    std_value = valid.std(ddof=0)

    if pd.isna(std_value) or abs(float(std_value)) < 1e-12:
        return valid - mean_value
    return (valid - mean_value) / std_value


def _winsorize_cross_section(
    feature_block: pd.DataFrame,
    lower_quantile: float = DEFAULT_WINSORIZE_QUANTILE,
) -> pd.DataFrame:
    """对某一天的全部股票特征做分位数裁剪。

    这里是“每列单独裁剪”，而不是把所有特征混成一列一起裁剪。
    原因很简单：不同特征的数量级、分布、经济含义都完全不同，
    必须各自按自己的横截面分布处理。
    """

    upper_quantile = 1.0 - lower_quantile
    lower_bounds = feature_block.quantile(lower_quantile)
    upper_bounds = feature_block.quantile(upper_quantile)
    return feature_block.clip(lower=lower_bounds, upper=upper_bounds, axis=1)


def _zscore_cross_section(feature_block: pd.DataFrame) -> pd.DataFrame:
    """对某一天的横截面特征做 z-score。

    这个函数只应该处理“同一天不同股票之间存在差异”的特征。
    对 VIX、利率、指数收益这类 date-level 宏观变量，同一天所有股票
    的值本来就完全相同，不能用横截面 z-score，否则会被压成 NaN。
    上层会先把这类 date-level 特征拆出去。
    """

    means = feature_block.mean(axis=0)
    stds = feature_block.std(axis=0, ddof=0).replace(0.0, np.nan)
    return (feature_block - means) / stds


def _split_cross_sectional_and_date_level_columns(
    feature_block: pd.DataFrame,
    min_std: float = 1e-12,
) -> tuple[list[str], list[str]]:
    """区分横截面特征和 date-level 特征。

    横截面预处理的前提是：同一天不同股票之间有可比较的分布。

    但宏观变量有不同性质：
    - `vix`
    - `sp500_return`
    - `treasury_10y`
    - `oil_price`

    这些变量在同一天对所有股票相同，表达的是市场状态随时间变化。
    如果把它们放进横截面 z-score，会得到 0 方差，再被后续步骤删掉。

    因此这里按“当天横截面标准差”拆成两组：
    - cross-sectional columns：正常 winsorize / z-score / neutralize；
    - date-level columns：保留原始数值，交给模型学习时间状态。
    """

    numeric_block = feature_block.apply(pd.to_numeric, errors="coerce")
    valid_counts = numeric_block.notna().sum(axis=0)
    std_values = numeric_block.std(axis=0, ddof=0)

    cross_sectional_columns = [
        column
        for column in numeric_block.columns
        if int(valid_counts.get(column, 0)) > 1
        and pd.notna(std_values.get(column))
        and abs(float(std_values.get(column))) > min_std
    ]
    date_level_columns = [column for column in numeric_block.columns if column not in cross_sectional_columns]
    return cross_sectional_columns, date_level_columns


def _build_neutralization_exposure_matrix(date_slice: pd.DataFrame) -> pd.DataFrame:
    """构造用于中性化的横截面暴露矩阵。

    当前版本最多使用两类常见暴露：

    1. `log_market_cap`：只有真实市值覆盖率达到 90% 时才控制大小盘风格；
    2. `sector` dummy：控制行业/板块风格。

    这里加一列常数项，是为了让线性回归可以自动吸收截距。
    """

    exposure_parts: list[pd.DataFrame] = [pd.DataFrame({"intercept": 1.0}, index=date_slice.index)]

    market_cap_column = (
        RAW_MARKET_CAP_EXPOSURE_COLUMN
        if RAW_MARKET_CAP_EXPOSURE_COLUMN in date_slice.columns
        else "market_cap"
    )
    if market_cap_column in date_slice.columns:
        valid_market_cap = pd.to_numeric(date_slice[market_cap_column], errors="coerce")
        positive_market_cap = valid_market_cap.where(valid_market_cap > 0.0)
        coverage = float(positive_market_cap.notna().mean())
        if (
            coverage >= MIN_MARKET_CAP_COVERAGE_FOR_NEUTRALIZATION
            and positive_market_cap.dropna().nunique() > 1
        ):
            exposure_parts.append(
                pd.DataFrame({"log_market_cap": _safe_zscore(np.log(positive_market_cap))})
            )

    if "sector" in date_slice.columns and date_slice["sector"].notna().any():
        sector_dummies = pd.get_dummies(date_slice["sector"].fillna("Unknown"), prefix="sector", drop_first=True)
        if not sector_dummies.empty:
            exposure_parts.append(sector_dummies.astype(float))

    exposure_matrix = pd.concat(exposure_parts, axis=1)
    return exposure_matrix


def _neutralize_feature_block(
    date_slice: pd.DataFrame,
    feature_block: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """对某一天的全部股票特征做条件式市值/行业中性化。

    实现方式是：

    - 对每个特征单独做一次横截面回归；
    - 自变量是可用的真实市值暴露和行业 dummy；
    - 取回归残差作为“去掉公共暴露后的特征值”。

    这一步的直觉可以理解成：

    - 如果某个因子只是因为“大盘股普遍更高”才显得有效，
      那么去掉市值暴露后，它的残差部分会更接近真正的 alpha；
    - 如果某个因子只是“能源股整体比别的行业高”，
      去掉行业暴露后，剩下的是行业内部更可比较的差异。
    """

    exposure_matrix = _build_neutralization_exposure_matrix(date_slice)
    if exposure_matrix.shape[1] <= 1:
        return feature_block.copy(), {"used_size": False, "used_sector": False}

    neutralized_columns: dict[str, pd.Series] = {}

    for feature_name in feature_block.columns:
        y = pd.to_numeric(feature_block[feature_name], errors="coerce")
        valid_mask = y.notna() & exposure_matrix.notna().all(axis=1)

        # 如果有效样本数还不够支撑回归，就不要强行中性化，
        # 直接保留当前处理后的特征值，避免小样本下数值发散。
        if int(valid_mask.sum()) <= exposure_matrix.shape[1]:
            neutralized_columns[feature_name] = y
            continue

        x_valid = exposure_matrix.loc[valid_mask].to_numpy(dtype=float)
        y_valid = y.loc[valid_mask].to_numpy(dtype=float)

        beta, *_ = np.linalg.lstsq(x_valid, y_valid, rcond=None)
        fitted = x_valid @ beta

        residual = pd.Series(np.nan, index=feature_block.index, dtype=float)
        residual.loc[valid_mask] = y_valid - fitted
        neutralized_columns[feature_name] = residual

    summary = {
        "used_size": "log_market_cap" in exposure_matrix.columns,
        "used_sector": any(column.startswith("sector_") for column in exposure_matrix.columns),
    }
    return pd.DataFrame(neutralized_columns, index=feature_block.index), summary


def apply_cross_sectional_preprocessing(
    data: pd.DataFrame,
    feature_columns: list[str],
    winsorize_quantile: float = DEFAULT_WINSORIZE_QUANTILE,
    apply_neutralization: bool = True,
    show_progress: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """对整张特征表做横截面预处理。

    处理顺序固定为：

    1. 每个日期做 winsorize；
    2. 每个日期做 z-score；
    3. 每个日期做行业中性化；真实市值覆盖充分时再做规模中性化；

    为什么顺序这么安排？

    - 先裁剪极端值，避免极端股票污染均值和标准差；
    - 再做标准化，让不同特征进入同一数量级；
    - 最后做中性化，得到尽量剥离公共风格暴露后的残差特征。
    """

    if not feature_columns:
        return data.copy(), {
            "enabled": False,
            "reason": "No feature columns were provided.",
        }

    if not 0.0 < winsorize_quantile < 0.5:
        raise ValueError("winsorize_quantile must fall inside (0, 0.5).")

    processed = data.copy()
    if "market_cap" in processed.columns and RAW_MARKET_CAP_EXPOSURE_COLUMN not in processed.columns:
        processed[RAW_MARKET_CAP_EXPOSURE_COLUMN] = pd.to_numeric(
            processed["market_cap"],
            errors="coerce",
        )
    processed[feature_columns] = processed[feature_columns].apply(pd.to_numeric, errors="coerce").astype(float)
    market_cap_coverage_ratio = 0.0
    market_cap_summary_column = (
        RAW_MARKET_CAP_EXPOSURE_COLUMN
        if RAW_MARKET_CAP_EXPOSURE_COLUMN in processed.columns
        else "market_cap"
    )
    if market_cap_summary_column in processed.columns and len(processed) > 0:
        # A second preprocessing pass may receive a model feature named
        # ``market_cap`` that has already been standardized. Coverage and
        # neutralization must both use the preserved observable exposure.
        market_cap_values = pd.to_numeric(
            processed[market_cap_summary_column],
            errors="coerce",
        )
        market_cap_coverage_ratio = float((market_cap_values > 0.0).mean())
    used_sector_on_any_date = False
    used_size_on_any_date = False
    processed_date_count = 0
    preserved_date_level_assignments = 0

    grouped_items = list(processed.groupby("date").groups.items())

    for current_date, row_index in optional_progress(
        grouped_items,
        description="Cross-sectional preprocessing",
        enabled=show_progress,
        total=len(grouped_items),
    ):
        date_index = pd.Index(row_index)
        date_slice = processed.loc[date_index].copy()

        # 先把要处理的特征块独立拿出来，避免在大表上反复逐列修改。
        raw_feature_block = date_slice[feature_columns].apply(pd.to_numeric, errors="coerce")
        feature_block = raw_feature_block.copy()
        cross_sectional_columns, date_level_columns = _split_cross_sectional_and_date_level_columns(
            raw_feature_block
        )

        # 只对真正有横截面分布的列做 winsorize / z-score。
        # 对 date-level 列保留原始值，否则宏观状态变量会在这里被错误抹掉。
        if cross_sectional_columns:
            cross_sectional_block = raw_feature_block[cross_sectional_columns]
            cross_sectional_block = _winsorize_cross_section(
                cross_sectional_block,
                lower_quantile=winsorize_quantile,
            )
            cross_sectional_block = _zscore_cross_section(cross_sectional_block)
            feature_block[cross_sectional_columns] = cross_sectional_block

        if date_level_columns:
            preserved_date_level_assignments += len(date_level_columns)

        neutralization_summary = {"used_size": False, "used_sector": False}
        if apply_neutralization and cross_sectional_columns:
            # 中性化同样只适合横截面列。
            # 如果把同一天常数的宏观列拿去对 intercept 回归，残差会变成 0，
            # 等价于把宏观变量从模型中删除。
            neutralized_block, neutralization_summary = _neutralize_feature_block(
                date_slice,
                feature_block[cross_sectional_columns],
            )
            feature_block[cross_sectional_columns] = neutralized_block

        processed.loc[date_index, feature_columns] = feature_block.to_numpy()
        used_sector_on_any_date = used_sector_on_any_date or neutralization_summary["used_sector"]
        used_size_on_any_date = used_size_on_any_date or neutralization_summary["used_size"]
        processed_date_count += 1

    processed = processed.replace([np.inf, -np.inf], np.nan)
    summary = {
        "enabled": True,
        "winsorize_quantile": winsorize_quantile,
        "zscore_applied": True,
        "neutralization_applied": apply_neutralization,
        "size_neutralization_used": used_size_on_any_date,
        "market_cap_coverage_ratio": market_cap_coverage_ratio,
        "sector_neutralization_used": used_sector_on_any_date,
        "processed_date_count": processed_date_count,
        "date_level_feature_preservation_used": preserved_date_level_assignments > 0,
        "preserved_date_level_feature_assignments": int(preserved_date_level_assignments),
    }
    return processed, summary
