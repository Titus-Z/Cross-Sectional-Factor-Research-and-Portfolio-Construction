"""数据加载与基础清洗模块。

这个版本已经不再服务于“自动生成 demo 数据”的场景，
而是专门面向真实股票日线数据。

当前模块最重要的职责有两件：

1. 把原始 CSV 清洗成统一、稳定、可建模的表结构；
2. 在这里统一构造多种“未来收益率标签”，避免标签逻辑散落在各个脚本里。

为什么要把标签构造集中在这里？

- 这样 `main.py`、`evaluate.py`、`yfinance_loader.py` 都能共用同一套定义；
- 你后面切换 `1日 / 5日 / 10日` 目标时，不需要担心不同脚本口径不一致；
- 对学习来说，也更容易搞清楚“模型到底在预测什么”。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import safe_divide


REQUIRED_COLUMNS = [
    "instrument_id",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
]

SUPPORTED_TARGET_HORIZONS = (1, 5, 10)
# 公共研究主线预测未来 10 个交易日收益。各实验仍可显式切换到 1d/5d，
# 但兼容列 `y` 的默认含义必须与 README 和直接运行入口保持一致。
DEFAULT_TARGET_HORIZON = 10
TARGET_COLUMN_MAP = {horizon: f"y_{horizon}d" for horizon in SUPPORTED_TARGET_HORIZONS}
DEFAULT_TARGET_COLUMN = TARGET_COLUMN_MAP[DEFAULT_TARGET_HORIZON]
PRICE_ADJUSTMENT_MODES = ("vendor_adjusted", "raw")


def add_forward_return_targets(
    data: pd.DataFrame,
    price_column: str = "close",
    horizons: tuple[int, ...] = SUPPORTED_TARGET_HORIZONS,
) -> pd.DataFrame:
    """基于未来收盘价构造多周期收益率标签。

    这里统一使用“未来第 N 个交易日收盘价 / 当前收盘价 - 1”的定义，
    主要有两个考虑：

    1. 比 `next_open / close - 1` 这种隔夜目标更平滑，噪声更小；
    2. 更容易扩展到 `5日`、`10日` 这样的中短期预测任务。

    例如：

    - `y_1d`：预测未来 1 个交易日的收盘收益
    - `y_5d`：预测未来 5 个交易日的收盘收益
    - `y_10d`：预测未来 10 个交易日的收盘收益

    目标终点使用全市场统一交易日历。某只股票在目标日没有收盘价时，
    标签保持缺失，不能跳到它自己的下一条可用记录。这样停牌或缺行股票
    不会把“10 日收益”悄悄延长成 11 日、20 日甚至更久。
    """

    if price_column not in data.columns:
        raise ValueError(f"price_column '{price_column}' is not present in the input data.")

    enriched = data.copy()
    if "instrument_id" not in enriched.columns or "date" not in enriched.columns:
        raise ValueError("Forward targets require instrument_id and date columns.")
    enriched["date"] = pd.to_datetime(enriched["date"])
    if enriched.duplicated(subset=["instrument_id", "date"], keep=False).any():
        raise ValueError("Forward targets require unique instrument_id/date rows.")

    market_calendar = pd.Index(sorted(enriched["date"].dropna().unique()))
    close_lookup = pd.Series(
        pd.to_numeric(enriched[price_column], errors="coerce").to_numpy(),
        index=pd.MultiIndex.from_arrays(
            [enriched["instrument_id"].astype(str), enriched["date"]],
            names=["instrument_id", "date"],
        ),
    )

    for horizon in horizons:
        if horizon <= 0:
            raise ValueError("Target horizon must be a positive integer.")

        target_column = TARGET_COLUMN_MAP.get(horizon, f"y_{horizon}d")
        # Index is today's common market date; value is the exact t+N common
        # market date.  Building the Series explicitly avoids depending on
        # Index.to_series index/value defaults, which have varied across pandas
        # versions and make this important horizon contract harder to audit.
        future_calendar_dates = pd.Series(
            market_calendar.to_numpy(),
            index=market_calendar,
        ).shift(-horizon)
        target_dates = enriched["date"].map(future_calendar_dates)
        target_index = pd.MultiIndex.from_arrays(
            [enriched["instrument_id"].astype(str), target_dates],
            names=["instrument_id", "date"],
        )
        future_price = pd.Series(
            close_lookup.reindex(target_index).to_numpy(),
            index=enriched.index,
            dtype=float,
        )
        enriched[target_column] = safe_divide(future_price, enriched[price_column]) - 1.0

    return enriched


def activate_target_horizon(
    data: pd.DataFrame,
    target_horizon: int = DEFAULT_TARGET_HORIZON,
) -> tuple[pd.DataFrame, str]:
    """把指定周期的标签映射到统一的 `y` 列。

    整个项目下游大部分模块默认都读 `y` 这一列。
    为了避免在很多地方都传 `y_5d` / `y_10d` 这种列名，
    这里做一个非常简单的适配层：

    - 保留所有原始目标列，例如 `y_1d`、`y_5d`、`y_10d`
    - 只把当前要训练的那个目标复制到统一列名 `y`

    这样既能保留多目标信息，也能让主流程代码继续保持清晰。
    """

    if target_horizon not in TARGET_COLUMN_MAP:
        raise ValueError(
            f"Unsupported target horizon: {target_horizon}. "
            f"Supported horizons: {list(SUPPORTED_TARGET_HORIZONS)}"
        )

    target_column = TARGET_COLUMN_MAP[target_horizon]
    activated = data.copy()

    if target_column not in activated.columns:
        activated = add_forward_return_targets(activated, horizons=(target_horizon,))

    activated["y"] = activated[target_column]
    return activated, target_column


def load_daily_data(
    csv_path: str | Path,
    price_adjustment_mode: str = "vendor_adjusted",
) -> pd.DataFrame:
    """从 CSV 读取并清洗日频数据。

    该函数会完成以下操作：

    - 检查关键字段是否存在；
    - 将日期转成 `datetime`；
    - 按股票和日期排序；
    - 将数值列强制转成数值类型；
    - 保留原始 OHLCV 缺失，避免伪造价格和成交路径；
    - 自动补充 `y_1d` / `y_5d` / `y_10d` 这三种前瞻收益率标签；
    - 如果缺少 `turnover`，用 close × volume 构造成交额代理；
    - 不伪造 `market_cap`，缺失时保留 NaN 并交由特征层跳过市值特征；
    - 自动构造默认目标列。
    """

    if price_adjustment_mode not in PRICE_ADJUSTMENT_MODES:
        raise ValueError(
            f"Unsupported price_adjustment_mode: {price_adjustment_mode}. "
            f"Supported modes: {list(PRICE_ADJUSTMENT_MODES)}"
        )

    csv_path = Path(csv_path)
    data = pd.read_csv(csv_path)

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    data["instrument_id"] = data["instrument_id"].astype(str).str.strip()
    if data["instrument_id"].eq("").any() or data["instrument_id"].eq("nan").any():
        raise ValueError("instrument_id contains empty values.")

    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["instrument_id", "date"]).reset_index(drop=True)

    duplicate_count = int(data.duplicated(subset=["instrument_id", "date"], keep=False).sum())
    if duplicate_count > 0:
        raise ValueError(
            "Duplicate instrument_id/date rows are not allowed because they would "
            f"double-count training and portfolio observations. Duplicate rows: {duplicate_count}"
        )

    # Fundamental availability timestamps are audit metadata. Converting them to
    # numeric would erase the evidence that a quarterly value became available
    # only after its filing/acceptance date. They remain outside the explicit
    # feature allowlist in feature_generator.py.
    metadata_date_columns = {
        "effective_date",
        "report_date",
        "filing_date",
        "accepted_date",
    }
    for column in sorted(metadata_date_columns & set(data.columns)):
        data[column] = pd.to_datetime(data[column], errors="coerce")

    non_numeric_columns = {
        "instrument_id",
        "date",
        "sector",
        "fiscal_period",
        "market_cap_source",
        *metadata_date_columns,
    }
    numeric_columns = [column for column in data.columns if column not in non_numeric_columns]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if (data["close"].dropna() <= 0.0).any():
        invalid_count = int((data["close"].dropna() <= 0.0).sum())
        raise ValueError(f"close contains {invalid_count} nonpositive observations.")
    if (data["volume"].dropna() < 0.0).any():
        invalid_count = int((data["volume"].dropna() < 0.0).sum())
        raise ValueError(f"volume contains {invalid_count} negative observations.")

    # Yahoo 日线或用户自备 OHLCV 可能没有 VWAP 和 adjustment。
    # 先完成数值转换再计算 VWAP，避免字符串价格触发拼接或类型错误。
    # adjustment 只为兼容数据格式保留，不会进入 canonical 模型特征。
    if "vwap" not in data.columns:
        data["vwap"] = (data["high"] + data["low"] + data["close"]) / 3.0
        numeric_columns.append("vwap")
    if "adjustment" not in data.columns:
        data["adjustment"] = 1.0
        numeric_columns.append("adjustment")

    # Preserve the observable raw close for dollar-turnover construction.  Once
    # OHLC is back-adjusted, adjusted close * volume may inherit a later corporate
    # action's rescaling and no longer represent the dollar value traded that day.
    raw_close_for_turnover = pd.to_numeric(data["close"], errors="coerce").copy()

    # 对未复权 Yahoo OHLC 应用同一行的 Adj Close / Close 因子，使拆股、
    # 分红等公司行为不会制造虚假价格跳变。adjustment 本身依旧禁止进入模型。
    # 公开主线要求保留原始 OHLC 和非平凡 adjustment，便于审计公司行为及
    # 原始成交额。全 1 adjustment 只能用于明确标注的兼容实验。
    if price_adjustment_mode == "vendor_adjusted":
        adjustment = pd.to_numeric(data["adjustment"], errors="coerce")
        adjustment = adjustment.where(adjustment > 0.0, np.nan)
        for price_column in ["open", "high", "low", "close", "vwap"]:
            data[price_column] = pd.to_numeric(data[price_column], errors="coerce") * adjustment

    # OHLCV 缺失不在数据加载层填充。即使 ffill 没有使用未来值，
    # 它仍会制造本来没有发生的价格、成交量和日收益路径。因此：
    #
    # - 原始市场字段继续保留 NaN；
    # - 滚动特征由各自的最小窗口规则决定是否可用；
    # - 模型特征缺失只能在训练 fold 内拟合 imputer。
    #
    # 基本面的“持续有效”必须由 FMP 等合并模块根据 effective_date
    # 单独处理，不由这个通用 OHLCV loader 猜测。

    # 市值无法从 OHLCV 可靠推导。过去用 close × volume 制造代理值会把
    # 流动性暴露误称为 size exposure，因此现在缺失时只保留 NaN。
    # 真实市值必须来自具有可用时间戳的 point-in-time 数据源。
    if "market_cap" not in data.columns:
        data["market_cap"] = np.nan
    market_cap_available = pd.to_numeric(data["market_cap"], errors="coerce").notna()
    if "market_cap_source" in data.columns:
        provided_source = data["market_cap_source"].astype("string").str.strip()
        source_missing = provided_source.isna() | provided_source.eq("")
        provided_source = provided_source.mask(
            source_missing & market_cap_available,
            "provided_unspecified",
        )
        data["market_cap_source"] = provided_source.mask(~market_cap_available, "missing")
    else:
        data["market_cap_source"] = np.where(
            market_cap_available,
            "provided_unspecified",
            "missing",
        )

    derived_raw_turnover = raw_close_for_turnover * pd.to_numeric(data["volume"], errors="coerce")
    if "turnover" not in data.columns:
        data["turnover"] = derived_raw_turnover
    else:
        # Keep an explicitly supplied dollar-turnover observation, but fill only
        # rows where it is absent and both raw close and volume are observable.
        supplied_turnover = pd.to_numeric(data["turnover"], errors="coerce")
        data["turnover"] = supplied_turnover.where(supplied_turnover.notna(), derived_raw_turnover)

    # 如果未来开盘价不存在，则尝试用下一期 open 构造一个替代版本。
    # 这个字段现在主要作为原始数据留存和兼容用途；
    # 真正用于训练的默认标签，下面会统一切换到未来收盘收益率。
    if "next_open" not in data.columns:
        data["next_open"] = data.groupby("instrument_id")["open"].shift(-1)

    # 在真实量化研究里，过短的隔夜目标通常噪声很大。
    # 所以这里统一补充 1日 / 5日 / 10日 三种“未来收盘收益率”标签，
    # 后面训练阶段可以通过命令行参数选择到底训练哪一个目标。
    data = add_forward_return_targets(data, price_column="close")

    # 统一覆盖兼容列 `y`，不信任输入 CSV 中历史遗留的同名标签。
    # 这样所有入口都从当前复权口径重建标签，不会因旧文件里的 y 定义不同
    # 而得到无法比较的结果。其他 horizon 仍由 activate_target_horizon 显式切换。
    data["y"] = data[DEFAULT_TARGET_COLUMN]

    data = data.replace([np.inf, -np.inf], np.nan)
    return data


def time_based_train_test_split(
    data: pd.DataFrame,
    test_size: float = 0.2,
    date_column: str = "date",
    test_start_date: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按照日期做时间切分。

    这种切分方式比随机切分更符合时间序列预测场景，
    因为模型只能使用过去训练未来，不能打乱时间顺序。

    这个函数支持两种常见用法：

    1. 不传 `test_start_date`：
       按唯一日期的最后 `test_size` 比例切出测试集；
    2. 传入 `test_start_date`：
       把该日期及之后的数据全部视作 out-of-sample 测试集。

    第二种方式更接近真实研究流程，因为你可以显式指定：

    - 哪一天之前属于 in-sample 训练期
    - 哪一天开始属于未来的 OOS 检验期
    """

    unique_dates = np.array(sorted(pd.to_datetime(data[date_column]).dropna().unique()))
    if len(unique_dates) < 2:
        raise ValueError("At least two unique dates are required for time-based split.")

    if test_start_date is not None:
        split_date = pd.Timestamp(test_start_date)
        if split_date <= pd.Timestamp(unique_dates[0]) or split_date > pd.Timestamp(unique_dates[-1]):
            raise ValueError("test_start_date must fall inside the available date range.")
    else:
        # 这里按“唯一日期”而不是按“总行数”切分，
        # 是为了确保同一交易日不会一部分跑到训练集、一部分跑到测试集。
        # 对横截面股票因子研究来说，这一点很重要。
        split_index = int(len(unique_dates) * (1.0 - test_size))
        split_index = max(1, min(split_index, len(unique_dates) - 1))
        split_date = unique_dates[split_index]

    train_df = data[data[date_column] < split_date].copy()
    test_df = data[data[date_column] >= split_date].copy()
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
