from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is expected in the normal project env.
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factor_mining_workspace.heuristic_factor_search import (
    compute_composite_score,
    evaluate_candidate,
    standardize_candidate_cross_sectionally,
)
from factor_mining_workspace.auto_factor_mining import compute_rank_turnover_proxy
from factor_mining_workspace.mined_factor_model_ablation import (
    SafeFormulaEvaluator,
    build_feature_matrices,
    evaluate_prediction_frame,
    get_numeric_feature_columns,
    train_and_predict,
)
from factor_mining_workspace.single_factor_case_study import dataframe_to_markdown, sanitize_name
from factor_mining_workspace.single_factor_case_study import load_or_build_preprocessed_train_test
from src.model import build_model
from src.reporting import safe_corr
from src.runtime_config import (
    DEFAULT_FACTOR_DIAGNOSTICS_MODEL_DIR,
    DEFAULT_OOS_START_DATE,
    DEFAULT_PRIMARY_DATA_PATH,
    DEFAULT_PRIMARY_TARGET_HORIZON,
    DEFAULT_SAMPLE_START_DATE,
)


DEFAULT_CANDIDATE_PATH = (
    "factor_mining_workspace/auto_mining_outputs/"
    "warm_gp_10d_g2_p30_c90_s7/candidate_formulas.csv"
)
DEFAULT_OUTPUT_ROOT = "factor_mining_workspace/residual_alpha_outputs"
FORMULA_SIGNAL_CACHE_VERSION = "residual_formula_signal_cache_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine formula candidates against baseline residual returns.")
    parser.add_argument("--candidate-path", default=DEFAULT_CANDIDATE_PATH, help="自动挖因子阶段生成的 candidate_formulas.csv。")
    parser.add_argument("--data-path", default=DEFAULT_PRIMARY_DATA_PATH, help="原始日频数据路径。")
    parser.add_argument("--model-dir", default=DEFAULT_FACTOR_DIAGNOSTICS_MODEL_DIR, help="保留此参数以复用项目缓存接口。")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_ROOT, help="residual alpha 输出目录。")
    parser.add_argument("--cache-dir", default=".cache", help="特征/预处理缓存目录。")
    parser.add_argument("--sample-start-date", default=DEFAULT_SAMPLE_START_DATE, help="样本起始日期。")
    parser.add_argument("--oos-start-date", default=DEFAULT_OOS_START_DATE, help="OOS 起始日期。")
    parser.add_argument("--target-horizon", type=int, default=DEFAULT_PRIMARY_TARGET_HORIZON, help="目标收益周期。")
    parser.add_argument("--test-size", type=float, default=0.2, help="未指定 OOS 日期时的后段测试比例。")
    parser.add_argument("--n-groups", type=int, default=5, help="单因子分组数量。")
    parser.add_argument("--min-cross-section", type=int, default=30, help="每个日期最少股票数。")
    parser.add_argument("--residual-model", default="ridge", help="用于生成 residual 的 baseline 模型。")
    parser.add_argument("--model-folds", type=int, default=5, help="训练期 cross-fit residual 的时间折数。")
    parser.add_argument("--survivor-ratio", type=float, default=0.20, help="训练 residual 排序后进入候选池的比例。")
    parser.add_argument("--factor-zoo-size", type=int, default=5, help="最终 residual factor zoo 大小。")
    parser.add_argument("--max-signal-corr", type=float, default=0.85, help="训练期 residual factor zoo 内最大相关性。")
    parser.add_argument("--max-baseline-corr", type=float, default=0.40, help="候选信号和 baseline 预测的最大相关性。")
    parser.add_argument("--cost-bps", type=float, default=10.0, help="单边交易成本假设，单位 bps，用于 net spread proxy。")
    parser.add_argument("--top-retention-fraction", type=float, default=0.20, help="计算头部股票留存率时使用的头部比例。")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ridge"],
        help="模型层增量检验使用的模型。默认只跑 Ridge；如需更严格但更慢的对照，可传 ridge elastic_net。",
    )
    parser.add_argument("--random-seed", type=int, default=42, help="模型随机种子。")
    parser.add_argument("--run-name", default=None, help="输出目录名。")
    parser.add_argument("--disable-preprocessing-cache", action="store_true", help="关闭横截面预处理缓存。")
    parser.add_argument("--disable-formula-signal-cache", action="store_true", help="关闭 residual 候选公式信号缓存。")
    return parser.parse_args()


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def show_progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def load_candidate_formulas(candidate_path: Path) -> pd.DataFrame:
    candidate_df = pd.read_csv(candidate_path)
    required = {"candidate_id", "formula"}
    missing = required - set(candidate_df.columns)
    if missing:
        raise ValueError(f"candidate formula file is missing columns: {sorted(missing)}")
    candidate_df = candidate_df.dropna(subset=["candidate_id", "formula"]).copy()
    candidate_df["candidate_id"] = candidate_df["candidate_id"].astype(str)
    candidate_df["formula"] = candidate_df["formula"].astype(str)
    return candidate_df


def file_signature(path: Path) -> dict[str, object]:
    """生成文件签名，用于判断缓存是否仍然对应当前输入文件。

    只用文件路径不够安全，因为同一路径下的 CSV 可能已经被更新。
    所以这里同时记录文件大小和 mtime_ns；如果候选公式或原始数据改变，
    缓存键也会改变，上层会自动重新计算公式信号。
    """

    signature: dict[str, object] = {"path": str(path)}
    if path.exists():
        stat = path.stat()
        signature["size"] = int(stat.st_size)
        signature["mtime_ns"] = int(stat.st_mtime_ns)
    return signature


def dataframe_signature(data: pd.DataFrame) -> dict[str, object]:
    """提取影响公式信号的轻量 DataFrame 签名。

    公式信号由特征列、样本行、日期区间和股票池决定。
    这里不对整个 DataFrame 做 hash，避免为了缓存本身再做一次昂贵扫描。
    """

    dates = pd.to_datetime(data["date"], errors="coerce") if "date" in data.columns else pd.Series(dtype="datetime64[ns]")
    return {
        "rows": int(len(data)),
        "columns": list(map(str, data.columns)),
        "min_date": str(dates.min().date()) if len(dates) and pd.notna(dates.min()) else None,
        "max_date": str(dates.max().date()) if len(dates) and pd.notna(dates.max()) else None,
        "instrument_count": int(data["instrument_id"].nunique()) if "instrument_id" in data.columns else None,
    }


def candidate_formula_digest(candidate_df: pd.DataFrame) -> str:
    """对候选公式内容做稳定摘要。

    mtime 可以捕捉文件修改，但内容摘要更直接：
    如果用户复制出一个新文件、mtime 不同但内容相同，摘要仍然说明公式集合一致。
    """

    payload = candidate_df[[column for column in ["candidate_id", "formula"] if column in candidate_df.columns]].to_dict(
        orient="records"
    )
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def build_formula_signal_cache_key(
    *,
    candidate_path: Path,
    candidate_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    args: argparse.Namespace,
) -> str:
    """根据输入数据和候选公式生成 residual 公式信号缓存键。"""

    payload = {
        "version": FORMULA_SIGNAL_CACHE_VERSION,
        "candidate_file": file_signature(candidate_path),
        "candidate_digest": candidate_formula_digest(candidate_df),
        "data_file": file_signature(resolve_path(args.data_path)),
        "sample_start_date": args.sample_start_date,
        "oos_start_date": args.oos_start_date,
        "test_size": float(args.test_size),
        "target_horizon": int(args.target_horizon),
        "train_signature": dataframe_signature(train_df),
        "test_signature": dataframe_signature(test_df),
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:20]


def formula_signal_cache_dir(cache_root: Path, cache_key: str) -> Path:
    return cache_root / "residual_formula_signals" / cache_key


def load_formula_signal_cache(
    cache_root: Path,
    cache_key: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]] | None:
    """读取 residual 公式信号缓存。

    缓存包含：
    - 训练集标准化公式信号；
    - OOS 标准化公式信号；
    - 每个候选公式的计算状态；
    - 缓存 metadata。
    """

    cache_dir = formula_signal_cache_dir(cache_root, cache_key)
    train_path = cache_dir / "train_formula_signals.pkl"
    test_path = cache_dir / "test_formula_signals.pkl"
    metadata_path = cache_dir / "formula_signal_metadata.csv"
    cache_metadata_path = cache_dir / "cache_metadata.json"
    if not (train_path.exists() and test_path.exists() and metadata_path.exists() and cache_metadata_path.exists()):
        return None

    return (
        pd.read_pickle(train_path),
        pd.read_pickle(test_path),
        pd.read_csv(metadata_path),
        json.loads(cache_metadata_path.read_text(encoding="utf-8")),
    )


def save_formula_signal_cache(
    cache_root: Path,
    cache_key: str,
    *,
    train_signal_frame: pd.DataFrame,
    test_signal_frame: pd.DataFrame,
    signal_metadata_df: pd.DataFrame,
    cache_metadata: dict[str, object],
) -> Path:
    """保存 residual 公式信号缓存。"""

    cache_dir = formula_signal_cache_dir(cache_root, cache_key)
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_signal_frame.to_pickle(cache_dir / "train_formula_signals.pkl")
    test_signal_frame.to_pickle(cache_dir / "test_formula_signals.pkl")
    signal_metadata_df.to_csv(cache_dir / "formula_signal_metadata.csv", index=False)
    (cache_dir / "cache_metadata.json").write_text(
        json.dumps(cache_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return cache_dir


def build_or_load_formula_signal_cache(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    candidate_path: Path,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """构建或读取 residual 候选公式信号。

    residual alpha 实验里同一个公式会被多个阶段重复使用：

    1. 训练 residual 排序；
    2. factor zoo 相关性筛选；
    3. OOS residual 和换手诊断；
    4. 模型层增量测试。

    如果每个阶段都重新解析公式、计算横截面标准化，会产生大量重复 groupby。
    这层缓存把“公式 -> 标准化信号”单独保存，后续实验只需要读缓存。
    """

    cache_root = resolve_path(args.cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_key = build_formula_signal_cache_key(
        candidate_path=candidate_path,
        candidate_df=candidate_df,
        train_df=train_df,
        test_df=test_df,
        args=args,
    )

    if not args.disable_formula_signal_cache:
        cached = load_formula_signal_cache(cache_root=cache_root, cache_key=cache_key)
        if cached is not None:
            train_signal_frame, test_signal_frame, signal_metadata_df, cache_metadata = cached
            cache_metadata = dict(cache_metadata)
            cache_metadata["cache_status"] = "hit"
            cache_metadata["cache_key"] = cache_key
            print(f"[Info] Formula signal cache hit: {formula_signal_cache_dir(cache_root, cache_key)}", flush=True)
            return train_signal_frame, test_signal_frame, signal_metadata_df, cache_metadata

    print("[Info] Formula signal cache miss; computing residual candidate signals", flush=True)
    train_evaluator = SafeFormulaEvaluator(train_df)
    test_evaluator = SafeFormulaEvaluator(test_df)
    train_signal_columns: dict[str, pd.Series] = {}
    test_signal_columns: dict[str, pd.Series] = {}
    metadata_records: list[dict[str, object]] = []

    iterator = show_progress(
        candidate_df.itertuples(index=False),
        total=len(candidate_df),
        desc="Building formula signal cache",
        leave=False,
    )
    for row in iterator:
        candidate_id = str(row.candidate_id)
        formula = str(row.formula)
        try:
            raw_train_signal = train_evaluator.evaluate(formula)
            raw_test_signal = test_evaluator.evaluate(formula)
            train_signal_columns[candidate_id] = standardize_candidate_cross_sectionally(train_df, raw_train_signal)
            test_signal_columns[candidate_id] = standardize_candidate_cross_sectionally(test_df, raw_test_signal)
            metadata_records.append(
                {
                    "candidate_id": candidate_id,
                    "formula": formula,
                    "eval_status": "ok",
                    "error": "",
                }
            )
        except Exception as exc:
            metadata_records.append(
                {
                    "candidate_id": candidate_id,
                    "formula": formula,
                    "eval_status": "failed",
                    "error": str(exc),
                }
            )

    train_signal_frame = pd.DataFrame(train_signal_columns, index=train_df.index)
    test_signal_frame = pd.DataFrame(test_signal_columns, index=test_df.index)
    signal_metadata_df = pd.DataFrame(metadata_records)
    cache_metadata = {
        "cache_status": "disabled" if args.disable_formula_signal_cache else "miss_written",
        "cache_key": None if args.disable_formula_signal_cache else cache_key,
        "cache_dir": None if args.disable_formula_signal_cache else str(formula_signal_cache_dir(cache_root, cache_key)),
        "candidate_count": int(len(candidate_df)),
        "successful_signal_count": int(len(train_signal_columns)),
        "failed_signal_count": int(len(candidate_df) - len(train_signal_columns)),
        "version": FORMULA_SIGNAL_CACHE_VERSION,
    }

    if not args.disable_formula_signal_cache:
        save_formula_signal_cache(
            cache_root=cache_root,
            cache_key=cache_key,
            train_signal_frame=train_signal_frame,
            test_signal_frame=test_signal_frame,
            signal_metadata_df=signal_metadata_df,
            cache_metadata=cache_metadata,
        )

    return train_signal_frame, test_signal_frame, signal_metadata_df, cache_metadata


def chronological_date_folds(data: pd.DataFrame, n_folds: int) -> list[set[pd.Timestamp]]:
    """把训练日期切成按时间排列的验证块。

    这里不做随机 KFold。金融时间序列里随机切分会把未来市场状态混进训练。
    """

    unique_dates = pd.Series(pd.to_datetime(data["date"]).dropna().unique()).sort_values().to_list()
    if len(unique_dates) < n_folds + 1:
        raise ValueError("Not enough unique dates for cross-fit residual generation.")

    folds = np.array_split(np.asarray(unique_dates, dtype="datetime64[ns]"), n_folds)
    return [set(pd.to_datetime(fold)) for fold in folds if len(fold) > 0]


def build_crossfit_residuals(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    model_name: str,
    n_folds: int,
    random_seed: int,
) -> pd.Series:
    """用只看过去的 expanding-window 方式生成训练期 residual。

    如果直接用全训练集拟合后再算 residual，模型已经“看过”每一行标签，
    residual 会偏乐观。这里用 expanding window：

    - 第一个日期块没有更早训练样本，所以跳过；
    - 后续每个日期块只用它之前的日期训练；
    - 得到的 residual 更接近真实 out-of-sample 残差。
    """

    residual = pd.Series(np.nan, index=train_df.index, dtype=float)
    date_folds = chronological_date_folds(train_df, n_folds=n_folds)
    all_dates = pd.to_datetime(train_df["date"])

    for fold_number, valid_dates in enumerate(show_progress(date_folds, desc="Cross-fitting baseline residuals", leave=False)):
        if fold_number == 0:
            continue
        valid_mask = all_dates.isin(valid_dates)
        first_valid_date = min(valid_dates)
        train_mask = all_dates < first_valid_date
        if train_mask.sum() == 0 or valid_mask.sum() == 0:
            continue

        fold_train_df = train_df.loc[train_mask]
        fold_valid_df = train_df.loc[valid_mask]
        x_train, x_valid = build_feature_matrices(fold_train_df, fold_valid_df, feature_columns)
        y_train = pd.to_numeric(fold_train_df["y"], errors="coerce")
        valid_y_train = y_train.notna()
        x_train = x_train.loc[valid_y_train]
        y_train = y_train.loc[valid_y_train]

        model = build_model(model_name=model_name, random_state=random_seed)
        model.fit(x_train, y_train)
        prediction = pd.Series(model.predict(x_valid), index=fold_valid_df.index, dtype=float)
        residual.loc[fold_valid_df.index] = pd.to_numeric(fold_valid_df["y"], errors="coerce") - prediction

    return residual


def fit_final_baseline_and_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    model_name: str,
    random_seed: int,
) -> tuple[pd.Series, pd.Series]:
    """用完整训练集拟合 baseline，并返回训练集/OOS 的 baseline 预测。"""

    train_x, test_x = build_feature_matrices(train_df, test_df, feature_columns)
    y_train = pd.to_numeric(train_df["y"], errors="coerce")
    valid_train = y_train.notna()
    model = build_model(model_name=model_name, random_state=random_seed)
    model.fit(train_x.loc[valid_train], y_train.loc[valid_train])
    train_prediction = pd.Series(model.predict(train_x), index=train_df.index, dtype=float)
    test_prediction = pd.Series(model.predict(test_x), index=test_df.index, dtype=float)
    return train_prediction, test_prediction


def make_residual_target_frame(data: pd.DataFrame, residual: pd.Series) -> pd.DataFrame:
    residual_df = data.copy()
    residual_df["y"] = pd.to_numeric(residual, errors="coerce")
    return residual_df.dropna(subset=["y"]).copy()


def evaluate_candidate_on_residual(
    data: pd.DataFrame,
    candidate_name: str,
    candidate_series: pd.Series,
    n_groups: int,
    min_cross_section: int,
    target_horizon: int,
    include_spread_metrics: bool,
) -> dict[str, float]:
    metrics, _ = evaluate_candidate(
        data=data,
        candidate_name=candidate_name,
        candidate_series=candidate_series.loc[data.index],
        n_groups=n_groups,
        min_cross_section=min_cross_section,
        rebalance_step=target_horizon,
        include_spread_metrics=include_spread_metrics,
    )
    return metrics


def cost_adjusted_spread_proxy(
    gross_spread: float,
    top_retention: float,
    cost_bps: float,
) -> tuple[float, float, float]:
    """用 top bucket 留存率估计成本后的 long-short spread。

    这是交易成本 proxy，不是完整回测：

    - `top_turnover_proxy = 1 - top_retention` 估计头部股票每天换掉多少；
    - long-short 组合有多头和空头两边，因此成本乘以 2；
    - `cost_bps / 10000` 把 bps 转成收益率单位。

    这个近似足够回答一个早期研究问题：信号是否靠极高换手撑出毛收益。
    """

    if pd.isna(gross_spread) or pd.isna(top_retention):
        return float("nan"), float("nan"), float("nan")
    top_turnover_proxy = max(0.0, min(1.0, 1.0 - float(top_retention)))
    cost_penalty = 2.0 * top_turnover_proxy * float(cost_bps) / 10000.0
    net_spread = float(gross_spread) - cost_penalty
    return top_turnover_proxy, cost_penalty, net_spread


def orient_metrics_by_direction(metrics: dict[str, float], direction: float) -> dict[str, float]:
    """根据训练期决定的方向调整指标符号，避免重复评估同一个候选。

    若一个公式在训练 residual 上 IC 为负，我们实际使用的是 `-formula`。
    对相关性、IR、long-short 这类带方向的指标，符号会反转。
    对 positive ratio，方向反转后正 IC 天数比例近似变成 `1 - old_ratio`。
    """

    oriented = dict(metrics)
    if direction >= 0:
        return oriented

    signed_keys = {
        "pearson_ic_mean",
        "pearson_ic_ir",
        "spearman_ic_mean",
        "spearman_ic_ir",
        "long_short_spread",
        "group_monotonic_spearman",
    }
    ratio_keys = {"pearson_ic_positive_ratio", "spearman_ic_positive_ratio"}
    for key in signed_keys:
        if key in oriented and pd.notna(oriented[key]):
            oriented[key] = -float(oriented[key])
    for key in ratio_keys:
        if key in oriented and pd.notna(oriented[key]):
            oriented[key] = 1.0 - float(oriented[key])
    return oriented


def evaluate_residual_candidates(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    train_signal_frame: pd.DataFrame,
    signal_metadata_df: pd.DataFrame,
    train_residual: pd.Series,
    test_residual: pd.Series,
    baseline_train_prediction: pd.Series,
    baseline_test_prediction: pd.Series,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """用候选公式解释 baseline residual。

    训练期 residual 决定候选方向和排序。
    OOS residual 只作为验证字段写入报告，不参与训练期筛选。
    """

    train_residual_df = make_residual_target_frame(train_df, train_residual)
    signal_status = signal_metadata_df.set_index("candidate_id").to_dict(orient="index") if not signal_metadata_df.empty else {}
    records: list[dict[str, object]] = []

    iterator = show_progress(
        candidate_df.itertuples(index=False),
        total=len(candidate_df),
        desc="Evaluating residual candidates",
        leave=False,
    )
    for row in iterator:
        candidate_id = str(row.candidate_id)
        formula = str(row.formula)
        status_record = signal_status.get(candidate_id, {})
        if status_record.get("eval_status") != "ok" or candidate_id not in train_signal_frame.columns:
            records.append(
                {
                    "candidate_id": candidate_id,
                    "formula": formula,
                    "eval_status": "failed",
                    "error": status_record.get("error", "missing formula signal cache column"),
                }
            )
            continue
        train_signal = train_signal_frame[candidate_id]

        train_metrics = evaluate_candidate_on_residual(
            data=train_residual_df,
            candidate_name=candidate_id,
            candidate_series=train_signal,
            n_groups=args.n_groups,
            min_cross_section=args.min_cross_section,
            target_horizon=args.target_horizon,
            include_spread_metrics=False,
        )
        if not train_metrics:
            records.append({"candidate_id": candidate_id, "formula": formula, "eval_status": "no_train_metrics"})
            continue

        # 方向只允许由训练 residual 决定。若训练期 IC 为负，等价于使用 `-formula`。
        direction_source = float(train_metrics.get("pearson_ic_mean", float("nan")))
        direction = -1.0 if pd.notna(direction_source) and direction_source < 0 else 1.0
        directed_train_signal = train_signal * direction
        directed_formula = formula if direction > 0 else f"-({formula})"
        train_metrics = orient_metrics_by_direction(train_metrics, direction)

        train_baseline_corr = safe_corr(
            directed_train_signal.loc[train_residual_df.index],
            baseline_train_prediction.loc[train_residual_df.index],
            method="pearson",
        )
        train_score = compute_composite_score(train_metrics)
        residual_incremental_score = train_score * max(0.0, 1.0 - min(abs(train_baseline_corr), 1.0))

        records.append(
            {
                "candidate_id": candidate_id,
                "formula": formula,
                "directed_formula": directed_formula,
                "direction": direction,
                "eval_status": "ok",
                "family": getattr(row, "family", "unknown"),
                "fields": getattr(row, "fields", ""),
                "train_residual_score": train_score,
                "train_residual_incremental_score": residual_incremental_score,
                "train_signal_baseline_corr": train_baseline_corr,
                **{f"train_residual_{key}": value for key, value in train_metrics.items()},
            }
        )

    result_df = pd.DataFrame(records)
    if result_df.empty:
        return result_df
    ok_df = result_df[result_df["eval_status"] == "ok"].copy()
    failed_df = result_df[result_df["eval_status"] != "ok"].copy()
    if ok_df.empty:
        return result_df
    ok_df = ok_df.sort_values(
        ["train_residual_incremental_score", "train_residual_score"],
        ascending=False,
    ).reset_index(drop=True)
    return pd.concat([ok_df, failed_df], ignore_index=True)


def select_residual_factor_zoo(
    candidate_metrics_df: pd.DataFrame,
    train_df: pd.DataFrame,
    train_signal_frame: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """只用训练期 residual 指标选择 residual factor zoo。"""

    ok_df = candidate_metrics_df[candidate_metrics_df["eval_status"] == "ok"].copy()
    if ok_df.empty:
        return ok_df

    survivor_count = max(1, int(math.ceil(len(ok_df) * args.survivor_ratio)))
    survivor_df = ok_df.head(survivor_count).copy()
    selected_records: list[dict[str, object]] = []
    selected_signals: dict[str, pd.Series] = {}

    for row in survivor_df.itertuples(index=False):
        candidate_id = str(row.candidate_id)
        if abs(float(row.train_signal_baseline_corr)) > args.max_baseline_corr:
            continue

        if candidate_id not in train_signal_frame.columns:
            continue
        signal = train_signal_frame[candidate_id] * float(row.direction)

        max_selected_corr = 0.0
        for selected_signal in selected_signals.values():
            corr = safe_corr(signal, selected_signal, method="pearson")
            if pd.notna(corr):
                max_selected_corr = max(max_selected_corr, abs(float(corr)))
        if max_selected_corr > args.max_signal_corr:
            continue

        record = row._asdict()
        record["train_max_corr_to_selected"] = max_selected_corr
        selected_records.append(record)
        selected_signals[candidate_id] = signal
        if len(selected_records) >= args.factor_zoo_size:
            break

    if not selected_records:
        # 如果相关性约束过紧，保留训练 residual 分数最高的候选，保证报告可解释。
        fallback_df = survivor_df.head(args.factor_zoo_size).copy()
        fallback_df["train_max_corr_to_selected"] = np.nan
        return fallback_df
    return pd.DataFrame(selected_records)


def add_tradability_diagnostics(
    factor_zoo_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_signal_frame: pd.DataFrame,
    test_signal_frame: pd.DataFrame,
    test_residual: pd.Series,
    baseline_test_prediction: pd.Series,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """只对最终 residual factor zoo 计算 OOS、turnover 与成本诊断。

    不把 OOS 和 turnover 放进 90 个候选的全量扫描，原因很简单：

    - OOS 不应该参与候选选择，只能最终验证；
    turnover 需要按日期和股票反复 groupby，成本很高。
    工业上通常也是先用便宜指标缩小候选池，再对 shortlist 做更贵的交易可实现性检查。
    """

    if factor_zoo_df.empty:
        return factor_zoo_df.copy()

    test_residual_df = make_residual_target_frame(test_df, test_residual)
    records: list[dict[str, object]] = []

    for row in show_progress(
        factor_zoo_df.itertuples(index=False),
        total=len(factor_zoo_df),
        desc="Adding tradability diagnostics",
        leave=False,
    ):
        record = row._asdict()
        direction = float(record.get("direction", 1.0))
        candidate_id = str(record["candidate_id"])
        if candidate_id not in train_signal_frame.columns or candidate_id not in test_signal_frame.columns:
            record["tradability_status"] = "missing_formula_signal"
            records.append(record)
            continue

        train_signal = train_signal_frame[candidate_id] * direction
        test_signal = test_signal_frame[candidate_id] * direction
        train_turnover = compute_rank_turnover_proxy(
            data=train_df,
            candidate_series=train_signal,
            top_fraction=args.top_retention_fraction,
        )
        oos_turnover = compute_rank_turnover_proxy(
            data=test_df,
            candidate_series=test_signal,
            top_fraction=args.top_retention_fraction,
        )
        oos_metrics = evaluate_candidate_on_residual(
            data=test_residual_df,
            candidate_name=str(record["candidate_id"]),
            candidate_series=test_signal,
            n_groups=args.n_groups,
            min_cross_section=args.min_cross_section,
            target_horizon=args.target_horizon,
            include_spread_metrics=True,
        )
        oos_baseline_corr = safe_corr(
            test_signal.loc[test_residual_df.index],
            baseline_test_prediction.loc[test_residual_df.index],
            method="pearson",
        )
        train_top_turnover_proxy, train_cost_penalty, train_net_spread = cost_adjusted_spread_proxy(
            gross_spread=record.get("train_residual_long_short_spread", float("nan")),
            top_retention=train_turnover.get("top_retention", float("nan")),
            cost_bps=args.cost_bps,
        )
        oos_top_turnover_proxy, oos_cost_penalty, oos_net_spread = cost_adjusted_spread_proxy(
            gross_spread=oos_metrics.get("long_short_spread", float("nan")) if oos_metrics else float("nan"),
            top_retention=oos_turnover.get("top_retention", float("nan")),
            cost_bps=args.cost_bps,
        )

        record.update(
            {
                "oos_signal_baseline_corr": oos_baseline_corr,
                "train_rank_turnover": train_turnover.get("rank_turnover", float("nan")),
                "train_top_retention": train_turnover.get("top_retention", float("nan")),
                "train_top_turnover_proxy": train_top_turnover_proxy,
                "train_cost_penalty": train_cost_penalty,
                "train_residual_net_long_short_spread": train_net_spread,
                "oos_rank_turnover": oos_turnover.get("rank_turnover", float("nan")),
                "oos_top_retention": oos_turnover.get("top_retention", float("nan")),
                "oos_top_turnover_proxy": oos_top_turnover_proxy,
                "oos_cost_penalty": oos_cost_penalty,
                "oos_residual_net_long_short_spread": oos_net_spread,
                **{f"oos_residual_{key}": value for key, value in oos_metrics.items()},
            }
        )
        records.append(record)

    return pd.DataFrame(records)


def materialize_selected_residual_factors(
    data: pd.DataFrame,
    factor_zoo_df: pd.DataFrame,
    signal_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    residual_columns: list[str] = []
    residual_signal_columns: dict[str, pd.Series] = {}
    for row in show_progress(factor_zoo_df.itertuples(index=False), total=len(factor_zoo_df), desc="Materializing residual factors", leave=False):
        candidate_id = str(row.candidate_id)
        if candidate_id not in signal_frame.columns:
            continue
        direction = float(row.direction)
        column_name = f"residual_{sanitize_name(candidate_id)}"
        residual_signal_columns[column_name] = signal_frame[candidate_id] * direction
        residual_columns.append(column_name)
    if not residual_signal_columns:
        return data.copy(), residual_columns
    residual_signal_frame = pd.DataFrame(residual_signal_columns, index=data.index)
    enhanced_df = pd.concat([data.copy(), residual_signal_frame], axis=1).copy()
    return enhanced_df, residual_columns


def write_report(
    output_path: Path,
    dataset_summary: dict[str, object],
    candidate_count: int,
    residual_factor_zoo_df: pd.DataFrame,
    residual_model_metrics_df: pd.DataFrame,
    residual_model_delta_df: pd.DataFrame,
    timing_df: pd.DataFrame,
    baseline_feature_count: int,
    residual_feature_columns: list[str],
) -> None:
    zoo_columns = [
        column
        for column in [
            "candidate_id",
            "directed_formula",
            "family",
            "train_residual_pearson_ic_mean",
            "train_residual_spearman_ic_mean",
            "train_residual_long_short_spread",
            "train_residual_net_long_short_spread",
            "train_top_retention",
            "train_signal_baseline_corr",
            "oos_residual_pearson_ic_mean",
            "oos_residual_spearman_ic_mean",
            "oos_residual_long_short_spread",
            "oos_residual_net_long_short_spread",
            "oos_top_retention",
            "oos_rank_turnover",
            "oos_signal_baseline_corr",
            "train_max_corr_to_selected",
        ]
        if column in residual_factor_zoo_df.columns
    ]

    report_text = f"""# Residual Alpha Mining Report

## 1. Purpose

上一轮结果显示：自动挖出的 factor zoo 单因子 IC 很强，但模型层增量不足。
本实验把目标改成 baseline residual：

```text
residual = actual_return - baseline_prediction
```

也就是说，这一步不再奖励“已经被 baseline 技术指标解释过的收益”，而是寻找 baseline 没解释掉的剩余信号。

## 2. Dataset And Leakage Control

```json
{json.dumps(dataset_summary, ensure_ascii=False, indent=2)}
```

- 训练期 residual 使用 expanding-window cross-fit baseline prediction 计算。
- 第一个时间块没有过去样本训练 baseline，所以不参与 residual 训练评分。
- residual factor zoo 只按训练期 residual 指标和低相关约束选择。
- turnover/cost 诊断只对最终 shortlist 计算，默认交易成本假设为 `{dataset_summary.get("cost_bps", "N/A")}` bps。
- OOS residual 指标只用于最终验证和报告。

## 3. Candidate Pool

- Candidate formulas scanned: `{candidate_count}`
- Residual factor zoo size: `{len(residual_factor_zoo_df)}`

## 4. Selected Residual Factor Zoo

{dataframe_to_markdown(residual_factor_zoo_df[zoo_columns].copy())}

## 5. Model-Layer Incremental Test

| feature set | feature count |
| --- | ---: |
| canonical feature baseline | {baseline_feature_count} |
| canonical feature baseline + residual factor zoo | {baseline_feature_count + len(residual_feature_columns)} |

Residual factor columns:

```text
{chr(10).join(residual_feature_columns)}
```

## 6. OOS Model Metrics

{dataframe_to_markdown(residual_model_metrics_df)}

## 7. Incremental Delta

`delta = baseline_plus_residual_factors - canonical_feature_baseline`

{dataframe_to_markdown(residual_model_delta_df)}

## 8. Runtime

{dataframe_to_markdown(timing_df)}

## 9. How To Explain This In Interview

```text
After finding that mined factors were predictive but redundant with the canonical feature baseline,
I added a residual-alpha mining layer. The system first builds cross-fitted baseline residuals,
then searches formulaic candidates for signals that explain the residual and have low correlation
with baseline predictions. This turns factor mining from IC chasing into incremental signal discovery.
```

## 10. Interpretation Rule

- 如果 residual 因子单独能解释 residual，但模型层没有提升，说明它可能仍然和模型已用特征间接重叠。
- 如果 residual 因子提高 `Pearson IC mean`、`RankIC` 和 `long_short_spread`，它才是值得进入多窗口稳定性检验的增量候选。
- 如果 residual 因子的 OOS residual IC 为负，说明训练期 residual 结构没有稳定外推，应该降低这个 family 的权重。
- `*_net_long_short_spread` 只是基于 top retention 的成本近似，不等于正式组合回测。
"""
    output_path.write_text(report_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    total_start = time.perf_counter()

    candidate_path = resolve_path(args.candidate_path)
    output_root = resolve_path(args.output_dir)
    model_part = "_".join(args.models)
    run_name = args.run_name or f"residual_{sanitize_name(candidate_path.parent.name)}_{args.target_horizon}d_{model_part}"
    output_dir = output_root / sanitize_name(run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_df = load_candidate_formulas(candidate_path)
    print(f"[Info] Loaded candidate formulas: {candidate_path}", flush=True)
    print(f"[Info] Candidate count: {len(candidate_df)}", flush=True)

    train_df, test_df, target_column, dataset_summary = load_or_build_preprocessed_train_test(args)
    dataset_summary = dict(dataset_summary)
    dataset_summary["target_column"] = target_column
    dataset_summary["candidate_path"] = str(candidate_path)
    dataset_summary["residual_model"] = args.residual_model
    dataset_summary["model_folds"] = args.model_folds
    dataset_summary["incremental_models"] = list(args.models)
    dataset_summary["cost_bps"] = args.cost_bps
    dataset_summary["top_retention_fraction"] = args.top_retention_fraction

    baseline_feature_columns = get_numeric_feature_columns(train_df)

    formula_signal_start = time.perf_counter()
    train_signal_frame, test_signal_frame, signal_metadata_df, formula_signal_cache_metadata = (
        build_or_load_formula_signal_cache(
            train_df=train_df,
            test_df=test_df,
            candidate_df=candidate_df,
            candidate_path=candidate_path,
            args=args,
        )
    )
    formula_signal_seconds = time.perf_counter() - formula_signal_start
    dataset_summary["formula_signal_cache_status"] = formula_signal_cache_metadata.get("cache_status")
    dataset_summary["formula_signal_cache_key"] = formula_signal_cache_metadata.get("cache_key")
    dataset_summary["formula_signal_cache_dir"] = formula_signal_cache_metadata.get("cache_dir")
    dataset_summary["formula_signal_successful_count"] = formula_signal_cache_metadata.get("successful_signal_count")
    dataset_summary["formula_signal_failed_count"] = formula_signal_cache_metadata.get("failed_signal_count")

    residual_start = time.perf_counter()
    train_crossfit_residual = build_crossfit_residuals(
        train_df=train_df,
        feature_columns=baseline_feature_columns,
        model_name=args.residual_model,
        n_folds=args.model_folds,
        random_seed=args.random_seed,
    )
    baseline_train_prediction, baseline_test_prediction = fit_final_baseline_and_predict(
        train_df=train_df,
        test_df=test_df,
        feature_columns=baseline_feature_columns,
        model_name=args.residual_model,
        random_seed=args.random_seed,
    )
    oos_residual = pd.to_numeric(test_df["y"], errors="coerce") - baseline_test_prediction
    residual_seconds = time.perf_counter() - residual_start

    scan_start = time.perf_counter()
    residual_candidate_metrics_df = evaluate_residual_candidates(
        train_df=train_df,
        test_df=test_df,
        candidate_df=candidate_df,
        train_signal_frame=train_signal_frame,
        signal_metadata_df=signal_metadata_df,
        train_residual=train_crossfit_residual,
        test_residual=oos_residual,
        baseline_train_prediction=baseline_train_prediction,
        baseline_test_prediction=baseline_test_prediction,
        args=args,
    )
    residual_factor_zoo_df = select_residual_factor_zoo(
        candidate_metrics_df=residual_candidate_metrics_df,
        train_df=train_df,
        train_signal_frame=train_signal_frame,
        args=args,
    )
    residual_factor_zoo_df = add_tradability_diagnostics(
        factor_zoo_df=residual_factor_zoo_df,
        train_df=train_df,
        test_df=test_df,
        train_signal_frame=train_signal_frame,
        test_signal_frame=test_signal_frame,
        test_residual=oos_residual,
        baseline_test_prediction=baseline_test_prediction,
        args=args,
    )
    scan_seconds = time.perf_counter() - scan_start

    materialize_start = time.perf_counter()
    train_with_residual, residual_columns = materialize_selected_residual_factors(
        train_df,
        residual_factor_zoo_df,
        train_signal_frame,
    )
    test_with_residual, _ = materialize_selected_residual_factors(
        test_df,
        residual_factor_zoo_df,
        test_signal_frame,
    )
    materialize_seconds = time.perf_counter() - materialize_start

    baseline_start = time.perf_counter()
    baseline_predictions, baseline_timing = train_and_predict(
        train_df=train_df,
        test_df=test_df,
        feature_columns=baseline_feature_columns,
        model_names=args.models,
        random_seed=args.random_seed,
    )
    baseline_seconds = time.perf_counter() - baseline_start

    residual_model_start = time.perf_counter()
    residual_feature_columns = baseline_feature_columns + residual_columns
    residual_predictions, residual_timing = train_and_predict(
        train_df=train_with_residual,
        test_df=test_with_residual,
        feature_columns=residual_feature_columns,
        model_names=args.models,
        random_seed=args.random_seed,
    )
    residual_model_seconds = time.perf_counter() - residual_model_start

    baseline_metrics = evaluate_prediction_frame(baseline_predictions, "canonical_feature_baseline")
    residual_metrics = evaluate_prediction_frame(residual_predictions, "baseline_plus_residual_factor_zoo")
    residual_model_metrics_df = pd.DataFrame([baseline_metrics, residual_metrics])

    delta_record: dict[str, float | str] = {
        "comparison": "baseline_plus_residual_factor_zoo - canonical_feature_baseline"
    }
    for column in [col for col in residual_model_metrics_df.columns if col != "experiment"]:
        delta_record[column] = float(residual_model_metrics_df.loc[1, column] - residual_model_metrics_df.loc[0, column])
    residual_model_delta_df = pd.DataFrame([delta_record])

    timing_df = pd.concat(
        [
            pd.DataFrame(
                [
                    {"stage": "build_or_load_formula_signals", "runtime_seconds": formula_signal_seconds},
                    {"stage": "build_crossfit_residuals", "runtime_seconds": residual_seconds},
                    {"stage": "scan_residual_candidates", "runtime_seconds": scan_seconds},
                    {"stage": "materialize_residual_factors", "runtime_seconds": materialize_seconds},
                    {"stage": "train_canonical_feature_baseline_total", "runtime_seconds": baseline_seconds},
                    {"stage": "train_baseline_plus_residual_total", "runtime_seconds": residual_model_seconds},
                    {"stage": "total_script_runtime", "runtime_seconds": time.perf_counter() - total_start},
                ]
            ),
            baseline_timing.assign(stage=lambda frame: "baseline_model_" + frame["model"].astype(str))[
                ["stage", "runtime_seconds"]
            ],
            residual_timing.assign(stage=lambda frame: "residual_model_" + frame["model"].astype(str))[
                ["stage", "runtime_seconds"]
            ],
        ],
        ignore_index=True,
    )

    residual_candidate_metrics_df.to_csv(output_dir / "residual_candidate_metrics.csv", index=False)
    residual_factor_zoo_df.to_csv(output_dir / "residual_factor_zoo.csv", index=False)
    pd.DataFrame({"feature": residual_columns}).to_csv(output_dir / "residual_feature_columns.csv", index=False)
    baseline_predictions.to_csv(output_dir / "predictions_canonical_feature_baseline.csv", index=False)
    residual_predictions.to_csv(output_dir / "predictions_baseline_plus_residual.csv", index=False)
    residual_model_metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    residual_model_delta_df.to_csv(output_dir / "metric_delta.csv", index=False)
    timing_df.to_csv(output_dir / "runtime.csv", index=False)

    write_report(
        output_path=output_dir / "report.md",
        dataset_summary=dataset_summary,
        candidate_count=len(candidate_df),
        residual_factor_zoo_df=residual_factor_zoo_df,
        residual_model_metrics_df=residual_model_metrics_df,
        residual_model_delta_df=residual_model_delta_df,
        timing_df=timing_df,
        baseline_feature_count=len(baseline_feature_columns),
        residual_feature_columns=residual_columns,
    )

    print(f"[Done] Report written to: {output_dir / 'report.md'}", flush=True)
    print(residual_model_metrics_df.to_string(index=False), flush=True)
    print(residual_model_delta_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
