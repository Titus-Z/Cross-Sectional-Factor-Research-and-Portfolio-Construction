"""Strict mined-factor incremental experiment.

这个脚本专门回答一个更严格的问题：

    在当前 10d canonical pipeline 里加入 validation-selected 自挖因子，是否还能带来增量？

它和 `main_mined_factor_incremental_experiment.py` 的区别很关键：

1. 这里复用主线的 Top50 特征选择；
2. 默认同时比较 Ridge + Lasso + XGBoost；
3. 这里复用 walk-forward validation 得到的模型权重；
4. 先比较技术基线与 Alpha191 基线，再只改变是否加入 validation-selected 自挖因子。

所以这份实验比之前的 96 视角实验更适合回答：

    自挖因子有没有改善已经很强的原始 MyQuant 主线？

注意：这仍然是研究实验，不是实盘交易系统。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm 只影响显示。
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factor_mining_workspace.mined_factor_model_ablation import (  # noqa: E402
    add_mined_factor_columns,
    dataframe_to_markdown,
    load_factor_zoo,
)
from factor_mining_workspace.single_factor_case_study import sanitize_name  # noqa: E402
from main import (  # noqa: E402
    finalize_model_weights,
    resolve_alpha_factor_names,
    resolve_requested_models,
    summarize_weighted_feature_importance,
)
from main_rolling_oos_backtest import (  # noqa: E402
    OOSWindow,
    build_run_dir_name,
    build_windows,
    summarize_window_predictions,
    write_window_prediction_file,
)
from src.data_loader import PRICE_ADJUSTMENT_MODES, activate_target_horizon, load_daily_data  # noqa: E402
from src.alpha191 import CANONICAL_SCALE_INVARIANT_ALPHA_FACTORS  # noqa: E402
from src.feature_cache import build_feature_cache_key, load_feature_cache, save_feature_cache  # noqa: E402
from src.feature_selector import FeatureSelector  # noqa: E402
from src.long_short_backtest import LongShortBacktestConfig, run_long_short_backtest  # noqa: E402
from src.model import ModelEnsemble, build_model, normalize_feature_importance  # noqa: E402
from src.portfolio import load_market_snapshot_frame, load_prediction_frame  # noqa: E402
from src.preprocessing import DEFAULT_WINSORIZE_QUANTILE, apply_cross_sectional_preprocessing  # noqa: E402
from src.preprocessing_cache import (  # noqa: E402
    build_preprocessing_cache_key,
    load_preprocessing_cache,
    save_preprocessing_cache,
)
from src.progress import optional_progress  # noqa: E402
from src.project_paths import resolve_project_path  # noqa: E402
from src.provenance import dumps_strict_json, sha256_file  # noqa: E402
from src.reporting import calculate_prediction_metrics  # noqa: E402
from src.runtime_config import (  # noqa: E402
    DEFAULT_OOS_START_DATE,
    DEFAULT_PRIMARY_DATA_PATH,
    DEFAULT_PRIMARY_TARGET_HORIZON,
    DEFAULT_SAMPLE_START_DATE,
)
from src.time_series_pipeline import DEFAULT_HISTORY_WINDOW, strict_time_split_feature_engineering  # noqa: E402
from src.universe import get_symbol_sector_map  # noqa: E402
from src.validation import run_walk_forward_validation  # noqa: E402
from src.validation_cache import build_validation_cache_key, load_validation_cache, save_validation_cache  # noqa: E402


DEFAULT_OUTPUT_DIR = "outputs/strict_mined_factor_experiment_us300_10d_oos202601"
DEFAULT_OOS_START_DATE_STRICT = DEFAULT_OOS_START_DATE
DEFAULT_WARM_GP_ZOO_PATH = (
    "factor_mining_workspace/auto_mining_outputs/"
    "warm_gp_us300_10d_oos202601_v1/validation_selected_factor_zoo.csv"
)
DEFAULT_PPO_ZOO_PATH = (
    "factor_mining_workspace/deep_rl_mining_outputs/"
    "ppo_formula_us300_10d_v1/validation_selected_factor_zoo.csv"
)
STRICT_SELECTION_CONTRACT_VERSION = "validation_factor_zoo_v3_scale_invariant_fields_bound"
STRICT_SELECTION_SOURCE = "validation_reward_only"


@dataclass(frozen=True)
class StrictExperimentSpec:
    """一组严格增量实验。

    `feature_subset` 决定是否使用 Alpha191，`feature_group` 决定追加哪组自挖因子。
    其他模型训练流程全部保持一致，才能把差异解释为特征集合的增量。
    """

    experiment: str
    feature_group: str
    feature_subset: str
    comparison_baseline: str | None
    description: str


@dataclass(frozen=True)
class StrategySpec:
    """组合回测策略口径。"""

    strategy_name: str
    hold_days: int
    step_days: int


def configure_runtime_warning_display() -> None:
    """减少重复数值 warning，避免进度条被刷屏破坏。"""

    warnings.filterwarnings(
        "once",
        message="overflow encountered in square",
        category=RuntimeWarning,
        module=r"pandas\.core\.nanops",
    )
    warnings.filterwarnings(
        "once",
        message="overflow encountered in reduce",
        category=RuntimeWarning,
        module=r"numpy\.core\._methods",
    )


def progress_iter(iterable, *, total: int | None = None, desc: str = "", position: int = 0, leave: bool = True):
    """统一进度条入口；没有 tqdm 时退化成普通迭代。"""

    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, position=position, leave=leave)


def read_path(path_like: str | Path) -> Path:
    """把相对路径解析到项目根目录，避免中文目录和空格导致路径混乱。"""

    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def dataframe_hash(df: pd.DataFrame, columns: list[str]) -> str:
    """给 factor zoo 生成摘要，用于 validation cache 防串读。"""

    if df.empty:
        return "empty"
    payload = df[columns].fillna("").astype(str).to_dict(orient="records")
    text = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strict mined-factor incremental experiments.")
    parser.add_argument("--data-path", default=DEFAULT_PRIMARY_DATA_PATH, help="真实日频数据 CSV。")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="实验输出目录。")
    parser.add_argument("--cache-dir", default=".cache", help="缓存目录。")
    parser.add_argument("--sample-start-date", default=DEFAULT_SAMPLE_START_DATE, help="样本起始日期。")
    parser.add_argument("--oos-start-date", default=DEFAULT_OOS_START_DATE_STRICT, help="OOS 起始日期。")
    parser.add_argument("--target-horizon", type=int, default=DEFAULT_PRIMARY_TARGET_HORIZON, help="预测收益周期。")
    parser.add_argument(
        "--price-adjustment-mode",
        choices=list(PRICE_ADJUSTMENT_MODES),
        default="vendor_adjusted",
        help="价格口径；必须和 canonical、PPO 挖掘及 factor-zoo selector 一致。",
    )
    parser.add_argument(
        "--max-alpha",
        type=int,
        default=0,
        help="可选的 Alpha 数量上限；0 表示不截断 --alpha-factors 显式清单。",
    )
    parser.add_argument(
        "--alpha-factors",
        nargs="+",
        default=list(CANONICAL_SCALE_INVARIANT_ALPHA_FACTORS),
        help=(
            "严格 Alpha191 基线的明确因子清单；默认与公开 canonical "
            "的逐股价格尺度不变子集一致。"
        ),
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="未指定 OOS 日期时的后段测试比例。")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ridge", "lasso", "xgboost"],
        help="严格主线模型组；默认同时保留线性模型和一个非线性对照。",
    )
    parser.add_argument("--n-splits", type=int, default=5, help="walk-forward fold 数量。")
    parser.add_argument("--validation-score-metric", default="pearson_ic_mean", help="模型权重使用的验证指标。")
    parser.add_argument("--top-n", type=int, default=50, help="最终进入模型的 Top N 特征数。")
    parser.add_argument("--missing-threshold", type=float, default=0.5, help="缺失率过滤阈值。")
    parser.add_argument("--variance-threshold", type=float, default=0.001, help="低方差过滤阈值。")
    parser.add_argument("--correlation-threshold", type=float, default=0.95, help="高相关过滤阈值。")
    parser.add_argument("--feature-score-method", choices=["correlation", "mutual_info"], default="correlation")
    parser.add_argument("--random-state", type=int, default=42, help="随机种子。")
    parser.add_argument("--warm-gp-zoo-path", default=DEFAULT_WARM_GP_ZOO_PATH, help="Warm-GP factor zoo。")
    parser.add_argument("--ppo-zoo-path", default=DEFAULT_PPO_ZOO_PATH, help="PPO validation-selected factor zoo。")
    parser.add_argument(
        "--mined-groups",
        nargs="+",
        choices=["ppo", "warm_gp", "warm_gp_ppo"],
        default=["ppo"],
        help=(
            "要纳入严格消融的自挖因子组。默认只使用 validation-selected PPO，"
            "避免无意间读入已经用 OOS 排序的旧 factor zoo。"
        ),
    )
    parser.add_argument("--top-k-list", nargs="+", type=int, default=[10, 20], help="组合回测 Top-K 网格。")
    parser.add_argument("--cost-bps-list", nargs="+", type=float, default=[5.0], help="交易成本 bps 网格。")
    parser.add_argument("--neutral-mode", default="unconstrained", choices=["unconstrained", "sector_neutral"])
    parser.add_argument("--signal-delay-days", type=int, default=1, help="信号延迟天数，默认次日执行。")
    parser.add_argument(
        "--holding-clock",
        choices=["signal_horizon", "execution_horizon"],
        default="signal_horizon",
        help="组合持有期时钟；严格主线默认与信号日 y_10d 终点对齐。",
    )
    parser.add_argument(
        "--borrow-cost-bps",
        type=float,
        default=0.0,
        help="简化借券费敏感性：按空头名义金额和持有天数线性计提的年化 bps。",
    )
    parser.add_argument("--disable-preprocessing-cache", action="store_true", help="关闭横截面预处理缓存。")
    parser.add_argument("--disable-validation-cache", action="store_true", help="关闭 walk-forward validation 缓存。")
    parser.add_argument("--skip-portfolio", action="store_true", help="只跑模型层，不跑组合层。")
    return parser.parse_args()


def load_raw_data_for_strict_experiment(args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    """读取真实数据并激活目标列。"""

    data_path = resolve_project_path(args.data_path)
    raw_data = load_daily_data(data_path, price_adjustment_mode=args.price_adjustment_mode)
    raw_data["date"] = pd.to_datetime(raw_data["date"])

    if args.sample_start_date:
        raw_data = raw_data[raw_data["date"] >= pd.Timestamp(args.sample_start_date)].copy()
    if raw_data.empty:
        raise ValueError("No rows remain after applying sample_start_date.")

    if "sector" not in raw_data.columns or raw_data["sector"].isna().all():
        sector_map = get_symbol_sector_map(sorted(raw_data["instrument_id"].dropna().unique()))
        if sector_map:
            raw_data["sector"] = raw_data["instrument_id"].map(sector_map).fillna("Unknown")

    raw_data, target_column = activate_target_horizon(raw_data, target_horizon=args.target_horizon)
    return raw_data, target_column


def load_or_build_baseline_features(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """复用 canonical pipeline 的严格时间切分、特征工程和横截面预处理。

    返回的 `feature_columns` 是 canonical pipeline 允许进入模型筛选器的候选特征。
    这是严格实验最重要的边界：不能再用“所有数值列”替代它。
    """

    data_path = resolve_project_path(args.data_path)
    cache_root = resolve_project_path(args.cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    raw_data, target_column = load_raw_data_for_strict_experiment(args)
    alpha_factor_names = resolve_alpha_factor_names(args.alpha_factors, args.max_alpha)

    feature_cache_key = build_feature_cache_key(
        data_path=data_path,
        sample_start_date=args.sample_start_date,
        oos_start_date=args.oos_start_date,
        test_size=args.test_size,
        target_horizon=args.target_horizon,
        history_window=DEFAULT_HISTORY_WINDOW,
        alpha_factor_names=alpha_factor_names,
        price_adjustment_mode=args.price_adjustment_mode,
    )
    cached_features = load_feature_cache(cache_root=cache_root, cache_key=feature_cache_key)
    if cached_features is None:
        print("[Info] Feature cache miss; rebuilding strict baseline features.", flush=True)
        train_df, test_df, feature_columns, feature_metadata = strict_time_split_feature_engineering(
            raw_data=raw_data,
            test_size=args.test_size,
            history_window=DEFAULT_HISTORY_WINDOW,
            test_start_date=args.oos_start_date,
            target_horizon=args.target_horizon,
            alpha_factor_names=alpha_factor_names,
            show_progress=True,
        )
        save_feature_cache(
            cache_root=cache_root,
            cache_key=feature_cache_key,
            train_df=train_df,
            test_df=test_df,
            feature_columns=feature_columns,
            feature_metadata=feature_metadata,
            metadata={
                "target_horizon": args.target_horizon,
                "history_window": DEFAULT_HISTORY_WINDOW,
                "alpha_factor_names": "all" if alpha_factor_names is None else list(alpha_factor_names),
                "price_adjustment_mode": args.price_adjustment_mode,
            },
        )
    else:
        print("[Info] Feature cache hit; using strict baseline feature columns.", flush=True)
        train_df, test_df, feature_columns, feature_metadata = cached_features

    preprocessing_cache_key = build_preprocessing_cache_key(
        data_path=data_path,
        sample_start_date=args.sample_start_date,
        oos_start_date=args.oos_start_date,
        test_size=args.test_size,
        target_horizon=args.target_horizon,
        history_window=DEFAULT_HISTORY_WINDOW,
        feature_columns=feature_columns,
        apply_preprocessing=True,
        apply_neutralization=True,
        winsorize_quantile=DEFAULT_WINSORIZE_QUANTILE,
        price_adjustment_mode=args.price_adjustment_mode,
    )
    cached_preprocessing = None
    if not args.disable_preprocessing_cache:
        cached_preprocessing = load_preprocessing_cache(cache_root=cache_root, cache_key=preprocessing_cache_key)

    if cached_preprocessing is None:
        print("[Info] Preprocessing cache miss; applying cross-sectional preprocessing.", flush=True)
        train_df, preprocessing_summary = apply_cross_sectional_preprocessing(
            train_df,
            feature_columns=feature_columns,
            show_progress=True,
        )
        test_df, _ = apply_cross_sectional_preprocessing(
            test_df,
            feature_columns=feature_columns,
            show_progress=True,
        )
        preprocessing_summary = dict(preprocessing_summary)
        preprocessing_summary["cache_status"] = "disabled" if args.disable_preprocessing_cache else "miss_written"
        preprocessing_summary["cache_key"] = preprocessing_cache_key if not args.disable_preprocessing_cache else None
        if not args.disable_preprocessing_cache:
            save_preprocessing_cache(
                cache_root=cache_root,
                cache_key=preprocessing_cache_key,
                train_df=train_df,
                test_df=test_df,
                preprocessing_summary=preprocessing_summary,
                metadata={
                    "target_horizon": args.target_horizon,
                    "feature_count": len(feature_columns),
                    "winsorize_quantile": DEFAULT_WINSORIZE_QUANTILE,
                    "apply_neutralization": True,
                    "price_adjustment_mode": args.price_adjustment_mode,
                },
            )
    else:
        print("[Info] Preprocessing cache hit.", flush=True)
        train_df, test_df, preprocessing_summary = cached_preprocessing
        preprocessing_summary = dict(preprocessing_summary)
        preprocessing_summary["cache_status"] = "hit"
        preprocessing_summary["cache_key"] = preprocessing_cache_key

    dataset_summary = {
        "data_path": str(args.data_path),
        "min_date": str(pd.to_datetime(raw_data["date"]).min().date()),
        "max_date": str(pd.to_datetime(raw_data["date"]).max().date()),
        "sample_start_date": args.sample_start_date,
        "oos_start_date_used": args.oos_start_date,
        "target_horizon": int(args.target_horizon),
        "target_column": target_column,
        "price_adjustment_mode": args.price_adjustment_mode,
        "alpha_factor_count": None if alpha_factor_names is None else len(alpha_factor_names),
        "instrument_count": int(raw_data["instrument_id"].nunique()),
        "n_rows": int(len(train_df) + len(test_df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_min_date": str(pd.to_datetime(train_df["date"]).min().date()),
        "train_max_date": str(pd.to_datetime(train_df["date"]).max().date()),
        "test_min_date": str(pd.to_datetime(test_df["date"]).min().date()),
        "test_max_date": str(pd.to_datetime(test_df["date"]).max().date()),
        "n_splits": int(args.n_splits),
        "baseline_candidate_feature_count": int(len(feature_columns)),
        "preprocessing_cache_status": preprocessing_summary.get("cache_status"),
    }
    return train_df, test_df, feature_columns, feature_metadata, preprocessing_summary, dataset_summary


def select_strict_baseline_feature_columns(feature_metadata: dict[str, Any], subset_name: str) -> list[str]:
    """构造严格递进消融所需的原始特征集。

    `technical_only` 包含原始量价、技术指标、可用的 point-in-time 基本面和市场上下文，
    但显式排除 Alpha191。`technical_plus_alpha191` 在相同基线上追加 Alpha191。
    两组的唯一差异因此就是 Alpha191，不会把 context feature 误删后当成 Alpha 贡献。
    """

    technical_keys = [
        "raw_feature_columns",
        "fundamental_raw_columns",
        "base_feature_columns",
        "advanced_feature_columns",
        "context_feature_columns",
    ]
    technical_columns = [
        column
        for key in technical_keys
        for column in feature_metadata.get(key, [])
    ]
    alpha_columns = list(feature_metadata.get("alpha_feature_columns", []))

    if subset_name == "technical_only":
        selected = technical_columns
    elif subset_name == "technical_plus_alpha191":
        selected = technical_columns + alpha_columns
    else:
        raise ValueError(f"Unsupported strict feature subset: {subset_name}")

    selected = list(dict.fromkeys(selected))
    if not selected:
        raise ValueError(f"Strict feature subset {subset_name!r} resolved to no columns.")
    return selected


def _resolve_contract_artifact(relative_path: object, field_name: str) -> Path:
    """把选择契约中的公开相对路径解析为本地文件并拒绝外部绝对路径。"""

    value = str(relative_path or "").strip()
    if not value or value.startswith("external://"):
        raise ValueError(f"Strict selection contract has no reproducible {field_name}: {value!r}")
    artifact_path = read_path(value)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Strict selection provenance artifact is missing: {artifact_path}")
    return artifact_path


def load_named_factor_zoo(
    path: Path,
    source_name: str,
    *,
    allowed_formula_fields: set[str],
    expected_target_horizon: int,
    expected_sample_start_date: str,
    expected_oos_start_date: str,
    expected_data_sha256: str,
    expected_price_adjustment_mode: str,
) -> pd.DataFrame:
    """读取带 validation-only 契约的 factor zoo，并拒绝 OOS-screened 旧文件。

    仅靠 CSV 中一列 ``selection_source`` 无法证明选择过程没有读取 OOS。严格入口
    同时核对 companion summary、输入候选/config 哈希、数据哈希和时间边界。任何
    一项不一致都会终止实验，避免把旧输出或手工改名文件混入正式消融。
    """

    if not path.is_file():
        raise FileNotFoundError(f"Strict factor zoo does not exist: {path}")
    summary_path = path.with_name("validation_selected_factor_zoo_summary.json")
    if not summary_path.is_file():
        raise FileNotFoundError(
            "Strict factor zoo requires validation_selected_factor_zoo_summary.json: "
            f"{summary_path}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("selection_contract_version") != STRICT_SELECTION_CONTRACT_VERSION:
        raise ValueError(
            "Unsupported strict factor-zoo contract: "
            f"{summary.get('selection_contract_version')!r}"
        )
    if summary.get("selection_source") != STRICT_SELECTION_SOURCE:
        raise ValueError(f"Strict factor-zoo source is not validation-only: {summary.get('selection_source')!r}")
    expected_searchers = {
        "ppo": {"ppo_deep_rl_formula_mining"},
        "warm_gp": {"warm_gp", "warm_gp_formula_mining"},
    }
    if source_name not in expected_searchers:
        raise ValueError(f"Unsupported strict factor source: {source_name!r}")
    if summary.get("source_searcher") not in expected_searchers[source_name]:
        raise ValueError(
            f"Strict factor-zoo searcher does not match source {source_name!r}: "
            f"{summary.get('source_searcher')!r}"
        )
    if list(summary.get("oos_columns_used", [])):
        raise ValueError(f"Strict factor selection used OOS columns: {summary.get('oos_columns_used')}")
    if int(summary.get("target_horizon", -1)) != int(expected_target_horizon):
        raise ValueError(
            "Strict factor-zoo target horizon mismatch: "
            f"{summary.get('target_horizon')} != {expected_target_horizon}"
        )
    if pd.Timestamp(summary.get("sample_start_date")) != pd.Timestamp(expected_sample_start_date):
        raise ValueError(
            "Strict factor-zoo sample start mismatch: "
            f"{summary.get('sample_start_date')} != {expected_sample_start_date}"
        )
    if pd.Timestamp(summary.get("oos_start_date")) != pd.Timestamp(expected_oos_start_date):
        raise ValueError(
            "Strict factor-zoo OOS boundary mismatch: "
            f"{summary.get('oos_start_date')} != {expected_oos_start_date}"
        )
    if summary.get("price_adjustment_mode") != expected_price_adjustment_mode:
        raise ValueError(
            "Strict factor-zoo price-adjustment mode mismatch: "
            f"{summary.get('price_adjustment_mode')!r} != {expected_price_adjustment_mode!r}"
        )
    contract_formula_fields = set(summary.get("allowed_formula_fields", []))
    if contract_formula_fields != set(allowed_formula_fields):
        raise ValueError(
            "Strict factor-zoo canonical formula-field contract does not match "
            "the current baseline candidate list."
        )
    validation_end = pd.Timestamp(summary.get("validation_end"))
    if validation_end >= pd.Timestamp(expected_oos_start_date):
        raise ValueError(
            "Strict factor-zoo validation overlaps OOS: "
            f"{validation_end.date()} >= {pd.Timestamp(expected_oos_start_date).date()}"
        )

    data_fingerprint = summary.get("data_fingerprint") or {}
    if data_fingerprint.get("sha256") != expected_data_sha256:
        raise ValueError("Strict factor zoo was selected from a different input data file.")
    if summary.get("factor_zoo_sha256") != sha256_file(path):
        raise ValueError("Strict factor-zoo CSV hash does not match its selection contract.")

    candidate_path = _resolve_contract_artifact(
        summary.get("project_relative_candidate_file"),
        "candidate file",
    )
    if summary.get("candidate_file_sha256") != sha256_file(candidate_path):
        raise ValueError("Strict factor-zoo candidate file hash does not match its selection contract.")
    config_path = _resolve_contract_artifact(
        summary.get("project_relative_source_config_file"),
        "source search config file",
    )
    if summary.get("source_config_sha256") != sha256_file(config_path):
        raise ValueError("Strict factor-zoo source config hash does not match its selection contract.")

    factor_zoo = load_factor_zoo(
        path,
        allowed_formula_fields=allowed_formula_fields,
    ).copy()
    if "selection_source" not in factor_zoo.columns:
        raise ValueError(
            f"Strict factor zoo must contain selection_source: {path}. "
            "Generate a validation-selected zoo before running this experiment."
        )
    selection_sources = set(factor_zoo["selection_source"].dropna().astype(str).str.lower())
    if selection_sources != {STRICT_SELECTION_SOURCE}:
        raise ValueError(
            f"Strict factor zoo contains non-validation selection sources: {sorted(selection_sources)}"
        )
    if len(factor_zoo) != int(summary.get("selected_count", -1)):
        raise ValueError("Strict factor-zoo row count does not match its selection contract.")
    factor_zoo["factor_source"] = source_name
    factor_zoo["selection_contract_version"] = STRICT_SELECTION_CONTRACT_VERSION
    factor_zoo["selection_contract_sha256"] = sha256_file(summary_path)
    factor_zoo["selection_data_sha256"] = expected_data_sha256
    factor_zoo["selection_validation_start"] = summary.get("validation_start")
    factor_zoo["selection_validation_end"] = summary.get("validation_end")
    factor_zoo["candidate_id"] = factor_zoo["candidate_id"].astype(str)
    return factor_zoo


def build_factor_groups(warm_gp_zoo: pd.DataFrame, ppo_zoo: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """定义可选自挖因子组；空组用于两层原始基线。"""

    return {
        "baseline": pd.DataFrame(columns=["candidate_id", "formula", "factor_source"]),
        "ppo": ppo_zoo.copy(),
        "warm_gp": warm_gp_zoo.copy(),
        "warm_gp_ppo": pd.concat([warm_gp_zoo, ppo_zoo], ignore_index=True),
    }


def build_strict_specs(mined_groups: list[str]) -> list[StrictExperimentSpec]:
    """定义三层递进对照：技术基线 -> Alpha191 -> validation-selected 自挖因子。"""

    specs = [
        StrictExperimentSpec(
            experiment="strict_technical_baseline",
            feature_group="baseline",
            feature_subset="technical_only",
            comparison_baseline=None,
            description="原始量价 + 技术/Context 特征，不使用 Alpha191 或自挖因子",
        ),
        StrictExperimentSpec(
            experiment="strict_alpha191_baseline",
            feature_group="baseline",
            feature_subset="technical_plus_alpha191",
            comparison_baseline="strict_technical_baseline",
            description="技术基线 + Alpha191",
        ),
    ]
    descriptions = {
        "ppo": "技术基线 + Alpha191 + PPO validation-selected 自挖因子",
        "warm_gp": "技术基线 + Alpha191 + Warm-GP validation-selected 自挖因子",
        "warm_gp_ppo": "技术基线 + Alpha191 + PPO/Warm-GP validation-selected 自挖因子",
    }
    for group_name in dict.fromkeys(mined_groups):
        specs.append(
            StrictExperimentSpec(
                experiment=f"strict_{group_name}",
                feature_group=group_name,
                feature_subset="technical_plus_alpha191",
                comparison_baseline="strict_alpha191_baseline",
                description=descriptions[group_name],
            )
        )
    return specs


def materialize_feature_group(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    baseline_feature_columns: list[str],
    factor_zoo: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """在指定 canonical 候选特征后追加自挖因子列。"""

    if factor_zoo.empty:
        return train_df.copy(), test_df.copy(), list(baseline_feature_columns), []

    enhanced_train_df, mined_columns = add_mined_factor_columns(train_df, factor_zoo)
    enhanced_test_df, mined_columns_test = add_mined_factor_columns(test_df, factor_zoo)
    if mined_columns != mined_columns_test:
        raise ValueError("Train/test mined columns are inconsistent.")

    # Baseline candidates have already gone through the canonical daily
    # winsorization, z-score, and conditional exposure-neutralization pipeline.
    # A fair incremental ablation must apply the same treatment to newly
    # materialized formulas. ``add_mined_factor_columns`` standardizes formula
    # outputs for general-purpose scripts; this second pass is intentionally
    # limited to the new columns and additionally removes the same available
    # sector/real-size exposures used by the baseline.
    enhanced_train_df, _ = apply_cross_sectional_preprocessing(
        enhanced_train_df,
        feature_columns=mined_columns,
        winsorize_quantile=DEFAULT_WINSORIZE_QUANTILE,
        apply_neutralization=True,
        show_progress=False,
    )
    enhanced_test_df, _ = apply_cross_sectional_preprocessing(
        enhanced_test_df,
        feature_columns=mined_columns,
        winsorize_quantile=DEFAULT_WINSORIZE_QUANTILE,
        apply_neutralization=True,
        show_progress=False,
    )

    feature_columns = list(baseline_feature_columns) + list(mined_columns)
    return enhanced_train_df, enhanced_test_df, feature_columns, mined_columns


def selector_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Canonical pipeline 的特征筛选参数。"""

    return {
        "missing_threshold": args.missing_threshold,
        "variance_threshold": args.variance_threshold,
        "correlation_threshold": args.correlation_threshold,
        "top_n": args.top_n,
        "score_method": args.feature_score_method,
        "random_state": args.random_state,
    }


def train_strict_pipeline_group(
    *,
    spec: StrictExperimentSpec,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    mined_columns: list[str],
    factor_zoo: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
    model_names: list[str],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """按 canonical pipeline 训练一组模型并预测 OOS。

    流程和 `main.py` 保持一致：

    1. walk-forward validation，每个 fold 内部单独特征选择；
    2. 根据验证 IC 计算模型权重；
    3. 在完整训练期重新做 Top50 特征选择；
    4. 训练最终模型；
    5. 对 OOS 预测并计算指标。
    """

    cache_root = resolve_project_path(args.cache_dir)
    data_path = resolve_project_path(args.data_path)
    group_dir = output_dir / spec.experiment
    model_dir = group_dir / "models"
    group_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    selector_config = selector_config_from_args(args)
    factor_context = {
        "feature_group": spec.feature_group,
        "feature_subset": spec.feature_subset,
        "mined_column_count": len(mined_columns),
        "factor_zoo_hash": dataframe_hash(factor_zoo, ["candidate_id", "formula"])
        if not factor_zoo.empty
        else "empty",
    }

    validation_cache_key = build_validation_cache_key(
        data_path=data_path,
        sample_start_date=args.sample_start_date,
        oos_start_date=args.oos_start_date,
        test_size=args.test_size,
        target_horizon=args.target_horizon,
        history_window=DEFAULT_HISTORY_WINDOW,
        feature_columns=feature_columns,
        model_names=model_names,
        selector_config=selector_config,
        n_splits=args.n_splits,
        random_state=args.random_state,
        score_metric=args.validation_score_metric,
        apply_preprocessing=True,
        apply_neutralization=True,
        winsorize_quantile=DEFAULT_WINSORIZE_QUANTILE,
        extra_context=factor_context,
        price_adjustment_mode=args.price_adjustment_mode,
    )

    validation_start = time.perf_counter()
    cached_validation = None
    if not args.disable_validation_cache:
        cached_validation = load_validation_cache(cache_root=cache_root, cache_key=validation_cache_key)

    if cached_validation is None:
        print(f"[Info] {spec.experiment}: running strict walk-forward validation.", flush=True)
        fold_metrics_df, model_summary_df, model_weights = run_walk_forward_validation(
            train_df=train_df,
            feature_columns=feature_columns,
            model_names=model_names,
            selector_config=selector_config,
            random_state=args.random_state,
            n_splits=args.n_splits,
            purge_days=args.target_horizon,
            score_metric=args.validation_score_metric,
            show_progress=True,
        )
        if not args.disable_validation_cache:
            save_validation_cache(
                cache_root=cache_root,
                cache_key=validation_cache_key,
                fold_metrics_df=fold_metrics_df,
                model_summary_df=model_summary_df,
                model_weights=model_weights,
                metadata={
                    "experiment": spec.experiment,
                    "factor_context": factor_context,
                    "model_names": model_names,
                },
            )
    else:
        print(f"[Info] {spec.experiment}: validation cache hit.", flush=True)
        fold_metrics_df, model_summary_df, model_weights = cached_validation
    validation_seconds = time.perf_counter() - validation_start

    final_start = time.perf_counter()
    selector = FeatureSelector(**selector_config)
    selector.fit(
        train_df[feature_columns],
        train_df["y"],
        dates=train_df["date"],
    )

    x_train = selector.transform(train_df[feature_columns])
    y_train = train_df["y"].reset_index(drop=True)
    x_test = selector.transform(test_df[feature_columns])

    final_model_weights = finalize_model_weights(model_names, model_weights)
    ensemble = ModelEnsemble()
    weighted_importance_frames: list[pd.DataFrame] = []
    final_model_timing_rows: list[dict[str, Any]] = []

    for model_name in optional_progress(
        model_names,
        description=f"Final models: {spec.experiment}",
        enabled=True,
        total=len(model_names),
    ):
        model_start = time.perf_counter()
        model_wrapper = build_model(model_name=model_name, random_state=args.random_state)
        model_wrapper.fit(x_train, y_train)
        model_wrapper.save(model_dir / f"{model_name}_model.joblib")
        elapsed = time.perf_counter() - model_start

        weight = final_model_weights.get(model_name, 0.0)
        ensemble.add_model(model_name, model_wrapper, weight=weight)
        raw_importance_df = model_wrapper.get_feature_importance(selector.selected_features_, model_name=model_name)
        weighted_importance_frames.append(normalize_feature_importance(raw_importance_df, model_weight=weight))
        final_model_timing_rows.append(
            {
                "experiment": spec.experiment,
                "model": model_name,
                "elapsed_sec": elapsed,
                "ensemble_weight": weight,
            }
        )

    predictions = ensemble.predict(x_test)
    prediction_df = test_df[["date", "instrument_id", "y"]].copy()
    prediction_df["date"] = pd.to_datetime(prediction_df["date"]).dt.strftime("%Y-%m-%d")
    prediction_df["predicted_y"] = predictions
    prediction_df.to_csv(group_dir / "test_predictions_with_actual.csv", index=False)
    prediction_df[["date", "instrument_id", "predicted_y"]].to_csv(group_dir / "predictions.csv", index=False)

    selector.save(model_dir / "feature_selector.json")
    pd.DataFrame({"feature": selector.selected_features_}).to_csv(model_dir / "selected_features.csv", index=False)
    selector.get_top_features(top_k=50).to_csv(model_dir / "selected_feature_scores.csv", index=False)
    pd.DataFrame([{"model": model, "weight": weight} for model, weight in final_model_weights.items()]).to_csv(
        model_dir / "model_weights.csv",
        index=False,
    )
    fold_metrics_df.to_csv(group_dir / "walk_forward_fold_metrics.csv", index=False)
    model_summary_df.to_csv(group_dir / "walk_forward_model_summary.csv", index=False)
    pd.DataFrame(final_model_timing_rows).to_csv(group_dir / "final_model_timing.csv", index=False)

    importance_df = summarize_weighted_feature_importance(weighted_importance_frames)
    if not importance_df.empty:
        importance_df.to_csv(model_dir / "feature_importance.csv", index=False)

    metrics = calculate_prediction_metrics(prediction_df)
    final_seconds = time.perf_counter() - final_start
    metrics.update(
        {
            "experiment": spec.experiment,
            "feature_group": spec.feature_group,
            "feature_subset": spec.feature_subset,
            "comparison_baseline": spec.comparison_baseline,
            "description": spec.description,
            "baseline_feature_count": len(feature_columns) - len(mined_columns),
            "mined_feature_count": len(mined_columns),
            "candidate_feature_count": len(feature_columns),
            "selected_feature_count": len(selector.selected_features_),
            "selected_mined_feature_count": int(
                sum(1 for feature in selector.selected_features_ if feature in set(mined_columns))
            ),
            "validation_cache_key": validation_cache_key,
            "validation_seconds": validation_seconds,
            "final_fit_predict_seconds": final_seconds,
            "model_weights_json": dumps_strict_json(final_model_weights, indent=None),
        }
    )

    selected_mined_df = pd.DataFrame(
        {
            "selected_mined_feature": [feature for feature in selector.selected_features_ if feature in set(mined_columns)]
        }
    )
    selected_mined_df.to_csv(group_dir / "selected_mined_features.csv", index=False)
    pd.DataFrame([metrics]).to_csv(group_dir / "model_metrics.csv", index=False)
    return metrics, prediction_df, importance_df


def build_model_delta_table(
    model_metrics_df: pd.DataFrame,
    specs: list[StrictExperimentSpec],
) -> pd.DataFrame:
    """按实验层级计算模型增量。

    Alpha191 组和纯技术基线比较；自挖因子组和 Alpha191 基线比较。
    如果所有组都对同一个 baseline 求差，Alpha191 与自挖因子的贡献会被混在一起。
    """

    if model_metrics_df.empty:
        return pd.DataFrame()
    metric_lookup = model_metrics_df.set_index("experiment")
    metric_columns = [
        "pearson_corr",
        "spearman_corr",
        "rmse",
        "mae",
        "pearson_ic_mean",
        "spearman_ic_mean",
        "pearson_ic_median",
        "spearman_ic_median",
        "pearson_ic_positive_ratio",
        "spearman_ic_positive_ratio",
        "long_short_spread",
        "long_short_return",
        "selected_mined_feature_count",
    ]
    rows = []
    for spec in specs:
        if spec.comparison_baseline is None:
            continue
        if spec.experiment not in metric_lookup.index or spec.comparison_baseline not in metric_lookup.index:
            continue
        row = metric_lookup.loc[spec.experiment]
        baseline_row = metric_lookup.loc[spec.comparison_baseline]
        delta = {
            "experiment": spec.experiment,
            "feature_group": spec.feature_group,
            "feature_subset": spec.feature_subset,
            "baseline_experiment": spec.comparison_baseline,
        }
        for column in metric_columns:
            if column in row.index and column in baseline_row.index:
                row_value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
                baseline_value = pd.to_numeric(pd.Series([baseline_row[column]]), errors="coerce").iloc[0]
                if pd.notna(row_value) and pd.notna(baseline_value):
                    delta[f"delta_{column}"] = float(row_value - baseline_value)
        rows.append(delta)
    return pd.DataFrame(rows)


def build_model_subwindow_metrics(
    predictions_by_experiment: dict[str, pd.DataFrame],
    specs: list[StrictExperimentSpec],
    oos_start_date: str,
) -> pd.DataFrame:
    """在相同 OOS 内再按 3/6/12 个月窗口检查预测稳定性。

    这里不重新训练，也不用子窗口选因子或调参。子窗口只是对已固定模型的
    OOS 诊断，用来防止一个总体指标掩盖中间窗口失效。
    """

    rows: list[dict[str, Any]] = []
    for spec in specs:
        prediction_df = predictions_by_experiment.get(spec.experiment)
        if prediction_df is None or prediction_df.empty:
            continue
        frame = prediction_df.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame[frame["date"] >= pd.Timestamp(oos_start_date)].copy()
        if frame.empty:
            continue
        windows = build_windows(
            min_start=pd.Timestamp(frame["date"].min()),
            max_date=pd.Timestamp(frame["date"].max()),
            window_modes=["full", "3m", "6m", "12m"],
            include_partial_final_window=False,
        )
        for window in windows:
            window_df = frame[
                (frame["date"] >= window.start_date) & (frame["date"] <= window.end_date)
            ].copy()
            if window_df.empty:
                continue
            metrics = calculate_prediction_metrics(window_df)
            metrics.update(
                {
                    "experiment": spec.experiment,
                    "feature_group": spec.feature_group,
                    "feature_subset": spec.feature_subset,
                    "comparison_baseline": spec.comparison_baseline,
                    "window_id": window.window_id,
                    "window_mode": window.window_mode,
                    "window_start": str(window.start_date.date()),
                    "window_end": str(window.end_date.date()),
                    "calendar_months": window.calendar_months,
                    "window_rows": int(len(window_df)),
                    "window_dates": int(window_df["date"].nunique()),
                }
            )
            rows.append(metrics)
    return pd.DataFrame(rows)


def build_model_subwindow_delta_table(
    subwindow_metrics_df: pd.DataFrame,
    specs: list[StrictExperimentSpec],
) -> pd.DataFrame:
    """在每个 OOS 子窗口内使用对应的递进 baseline 求差。"""

    if subwindow_metrics_df.empty:
        return pd.DataFrame()
    key_columns = ["window_id", "window_mode", "window_start", "window_end"]
    metric_columns = [
        "pearson_corr",
        "spearman_corr",
        "rmse",
        "mae",
        "pearson_ic_mean",
        "spearman_ic_mean",
        "long_short_spread",
        "prediction_coverage_ratio",
    ]
    rows: list[dict[str, Any]] = []
    for spec in specs:
        if spec.comparison_baseline is None:
            continue
        comparison = subwindow_metrics_df[subwindow_metrics_df["experiment"] == spec.experiment]
        baseline = subwindow_metrics_df[
            subwindow_metrics_df["experiment"] == spec.comparison_baseline
        ]
        baseline_lookup = baseline.set_index(key_columns)
        for _, row in comparison.iterrows():
            key = tuple(row[column] for column in key_columns)
            if key not in baseline_lookup.index:
                continue
            base = baseline_lookup.loc[key]
            if isinstance(base, pd.DataFrame):
                base = base.iloc[0]
            delta_row: dict[str, Any] = {
                "experiment": spec.experiment,
                "baseline_experiment": spec.comparison_baseline,
                **{column: row[column] for column in key_columns},
            }
            for column in metric_columns:
                if column in row.index and column in base.index:
                    row_value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
                    base_value = pd.to_numeric(pd.Series([base[column]]), errors="coerce").iloc[0]
                    if pd.notna(row_value) and pd.notna(base_value):
                        delta_row[f"delta_{column}"] = float(row_value - base_value)
            rows.append(delta_row)
    return pd.DataFrame(rows)


def collect_strict_fold_metrics(output_dir: Path, specs: list[StrictExperimentSpec]) -> pd.DataFrame:
    """汇总每组严格实验的全部 fold，保留负 IC fold 和完整时间边界。"""

    frames: list[pd.DataFrame] = []
    for spec in specs:
        fold_path = output_dir / spec.experiment / "walk_forward_fold_metrics.csv"
        if not fold_path.exists():
            continue
        frame = pd.read_csv(fold_path)
        frame.insert(0, "comparison_baseline", spec.comparison_baseline)
        frame.insert(0, "feature_subset", spec.feature_subset)
        frame.insert(0, "feature_group", spec.feature_group)
        frame.insert(0, "experiment", spec.experiment)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_increment_verdict_table(
    model_metrics_df: pd.DataFrame,
    model_delta_df: pd.DataFrame,
    subwindow_delta_df: pd.DataFrame,
    specs: list[StrictExperimentSpec],
) -> pd.DataFrame:
    """用预先固定的规则生成增量验收结论，避免看完结果后改标准。

    初步通过需要：

    1. 整体 OOS 的 Pearson IC、Rank IC、long-short spread 至少两项改善；
    2. 至少两个非 full OOS 子窗口可评估，且过半窗口至少两项改善；
    3. 对自挖因子组，至少一个自挖因子真正进入最终 Top-N。

    这个 verdict 只是模型层继续研究门槛，不代表可交易 alpha。
    """

    if model_delta_df.empty:
        return pd.DataFrame()
    metric_lookup = model_metrics_df.set_index("experiment") if not model_metrics_df.empty else pd.DataFrame()
    core_delta_columns = [
        "delta_pearson_ic_mean",
        "delta_spearman_ic_mean",
        "delta_long_short_spread",
    ]
    rows: list[dict[str, Any]] = []
    for spec in specs:
        if spec.comparison_baseline is None:
            continue
        overall = model_delta_df[model_delta_df["experiment"] == spec.experiment]
        if overall.empty:
            continue
        overall_row = overall.iloc[0]
        overall_improvement_count = sum(
            pd.notna(overall_row.get(column)) and float(overall_row[column]) > 0
            for column in core_delta_columns
        )

        windows = (
            subwindow_delta_df[subwindow_delta_df["experiment"] == spec.experiment].copy()
            if not subwindow_delta_df.empty and "experiment" in subwindow_delta_df.columns
            else pd.DataFrame()
        )
        if "window_mode" in windows.columns:
            windows = windows[windows["window_mode"] != "full"].copy()
        positive_window_count = 0
        for _, window_row in windows.iterrows():
            positive_metric_count = sum(
                pd.notna(window_row.get(column)) and float(window_row[column]) > 0
                for column in core_delta_columns
            )
            positive_window_count += int(positive_metric_count >= 2)
        evaluated_window_count = int(len(windows))
        positive_window_ratio = (
            positive_window_count / evaluated_window_count if evaluated_window_count else float("nan")
        )

        selected_mined_count = 0
        if not metric_lookup.empty and spec.experiment in metric_lookup.index:
            selected_mined_count = int(
                pd.to_numeric(
                    pd.Series([metric_lookup.loc[spec.experiment].get("selected_mined_feature_count", 0)]),
                    errors="coerce",
                ).fillna(0).iloc[0]
            )
        selection_gate = spec.feature_group == "baseline" or selected_mined_count > 0
        window_gate = evaluated_window_count >= 2 and positive_window_ratio >= 0.5
        overall_gate = overall_improvement_count >= 2
        passed = bool(overall_gate and window_gate and selection_gate)

        rows.append(
            {
                "experiment": spec.experiment,
                "baseline_experiment": spec.comparison_baseline,
                "feature_group": spec.feature_group,
                "overall_core_improvement_count": int(overall_improvement_count),
                "evaluated_non_full_subwindows": evaluated_window_count,
                "positive_subwindows": int(positive_window_count),
                "positive_subwindow_ratio": positive_window_ratio,
                "selected_mined_feature_count": selected_mined_count,
                "overall_metric_gate": overall_gate,
                "subwindow_stability_gate": window_gate,
                "mined_feature_selected_gate": selection_gate,
                "preliminary_model_increment_gate": passed,
                "verdict": (
                    "passes_preliminary_model_increment_gate"
                    if passed
                    else "does_not_pass_preliminary_model_increment_gate"
                ),
            }
        )
    return pd.DataFrame(rows)


def strategy_specs() -> list[StrategySpec]:
    """固定组合策略口径。"""

    return [
        StrategySpec("hold10_step10", 10, 10),
        StrategySpec("hold10_step5", 10, 5),
        StrategySpec("hold20_step20", 20, 20),
        StrategySpec("hold20_step10", 20, 10),
    ]


def summarize_backtest_row(
    *,
    spec: StrictExperimentSpec,
    strategy: StrategySpec,
    top_k: int,
    cost_bps: float,
    neutral_mode: str,
    window: OOSWindow,
    prediction_df: pd.DataFrame,
    result_metrics: dict[str, Any],
) -> dict[str, Any]:
    """把一次回测结果压平成 CSV 一行。"""

    row: dict[str, Any] = {
        "experiment": spec.experiment,
        "feature_group": spec.feature_group,
        "feature_subset": spec.feature_subset,
        "comparison_baseline": spec.comparison_baseline,
        "strategy_name": strategy.strategy_name,
        "window_id": window.window_id,
        "window_mode": window.window_mode,
        "window_start": str(window.start_date.date()),
        "window_end": str(window.end_date.date()),
        "calendar_months": window.calendar_months,
        "top_k": int(top_k),
        "cost_bps": float(cost_bps),
        "neutral_mode": neutral_mode,
        "status": "ok",
        **summarize_window_predictions(prediction_df, window),
    }
    for field in [
        "hold_days",
        "holding_clock",
        "effective_holding_days",
        "step_days",
        "daily_count",
        "rebalance_count",
        "portfolio_total_return",
        "portfolio_annualized_return",
        "portfolio_annualized_vol",
        "portfolio_sharpe",
        "portfolio_max_drawdown",
        "portfolio_calmar",
        "hit_ratio",
        "average_gross_turnover",
        "average_net_turnover",
        "average_turnover_cost_bps",
        "total_turnover_cost",
        "benchmark_total_return",
        "relative_wealth_vs_equal_weight_long_only",
        "is_short_sample_warning",
        "max_active_sleeves",
    ]:
        row[field] = result_metrics.get(field)
    row["error"] = ""
    return row


def run_strict_portfolio_views(
    *,
    specs: list[StrictExperimentSpec],
    output_dir: Path,
    data_path: Path,
    oos_start_date: str,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """对所有递进对照组的预测文件运行同一套组合回测。"""

    market_snapshot_df = load_market_snapshot_frame(
        data_path,
        price_adjustment_mode=args.price_adjustment_mode,
    )
    portfolio_root = output_dir / "portfolio_runs"
    rows: list[dict[str, Any]] = []

    for spec in progress_iter(specs, total=len(specs), desc="Portfolio experiment groups"):
        prediction_path = output_dir / spec.experiment / "test_predictions_with_actual.csv"
        prediction_df = load_prediction_frame(prediction_path)
        prediction_df = prediction_df[prediction_df["date"] >= pd.Timestamp(oos_start_date)].copy()
        if prediction_df.empty:
            raise ValueError(f"No OOS predictions for {spec.experiment}.")

        windows = build_windows(
            min_start=pd.Timestamp(oos_start_date),
            max_date=pd.Timestamp(prediction_df["date"].max()),
            window_modes=["full", "3m", "6m", "12m"],
            include_partial_final_window=False,
        )
        window_prediction_dir = portfolio_root / spec.experiment / "_window_predictions"
        window_prediction_paths: dict[str, Path] = {}

        grid_items = [
            (window, strategy, top_k, cost_bps)
            for window in windows
            for strategy in strategy_specs()
            for top_k in args.top_k_list
            for cost_bps in args.cost_bps_list
        ]
        for window, strategy, top_k, cost_bps in progress_iter(
            grid_items,
            total=len(grid_items),
            desc=f"Backtests: {spec.experiment}",
            position=1,
            leave=False,
        ):
            if window.window_id not in window_prediction_paths:
                window_prediction_paths[window.window_id] = write_window_prediction_file(
                    window_prediction_dir,
                    prediction_df,
                    window,
                )

            run_dir_name = build_run_dir_name(
                base_run_name=spec.experiment,
                window=window,
                hold_days=strategy.hold_days,
                step_days=strategy.step_days,
                top_k=int(top_k),
                cost_bps=float(cost_bps),
                neutral_mode=args.neutral_mode,
                holding_clock=args.holding_clock,
            )
            window_market_df = market_snapshot_df[
                (market_snapshot_df["date"] >= window.start_date) & (market_snapshot_df["date"] <= window.end_date)
            ].copy()
            config = LongShortBacktestConfig(
                run_name=run_dir_name,
                predictions_path=window_prediction_paths[window.window_id],
                data_path=data_path,
                output_dir=portfolio_root / spec.experiment / window.window_id / run_dir_name,
                hold_days=int(strategy.hold_days),
                step_days=int(strategy.step_days),
                top_k=int(top_k),
                cost_bps=float(cost_bps),
                neutral_mode=args.neutral_mode,
                signal_delay_days=int(args.signal_delay_days),
                holding_clock=args.holding_clock,
                borrow_cost_bps=float(args.borrow_cost_bps),
                price_adjustment_mode=args.price_adjustment_mode,
            )
            try:
                result = run_long_short_backtest(config=config, market_snapshot_df=window_market_df)
                rows.append(
                    summarize_backtest_row(
                        spec=spec,
                        strategy=strategy,
                        top_k=int(top_k),
                        cost_bps=float(cost_bps),
                        neutral_mode=args.neutral_mode,
                        window=window,
                        prediction_df=prediction_df,
                        result_metrics=result["metrics"],
                    )
                )
            except Exception as exc:
                rows.append(
                    {
                        "experiment": spec.experiment,
                        "feature_group": spec.feature_group,
                        "feature_subset": spec.feature_subset,
                        "comparison_baseline": spec.comparison_baseline,
                        "strategy_name": strategy.strategy_name,
                        "window_id": window.window_id,
                        "window_mode": window.window_mode,
                        "window_start": str(window.start_date.date()),
                        "window_end": str(window.end_date.date()),
                        "calendar_months": window.calendar_months,
                        "top_k": int(top_k),
                        "cost_bps": float(cost_bps),
                        "neutral_mode": args.neutral_mode,
                        "status": "failed",
                        "error": str(exc),
                    }
                )
    return pd.DataFrame(rows)


def aggregate_portfolio_views(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    """按实验、策略、Top-K 和窗口聚合组合表现。"""

    if portfolio_df.empty:
        return pd.DataFrame()
    ok = portfolio_df[portfolio_df["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()

    group_columns = [
        "experiment",
        "feature_group",
        "feature_subset",
        "comparison_baseline",
        "strategy_name",
        "window_mode",
        "top_k",
        "cost_bps",
        "neutral_mode",
    ]
    for column in [
        "portfolio_total_return",
        "relative_wealth_vs_equal_weight_long_only",
        "portfolio_sharpe",
        "portfolio_max_drawdown",
        "portfolio_calmar",
        "hit_ratio",
        "average_gross_turnover",
        "average_turnover_cost_bps",
        "rebalance_count",
    ]:
        ok[column] = pd.to_numeric(ok[column], errors="coerce")

    rows = []
    for keys, frame in ok.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, keys))
        total_return = frame["portfolio_total_return"].dropna()
        relative_wealth = frame["relative_wealth_vs_equal_weight_long_only"].dropna()
        sharpe = frame["portfolio_sharpe"].dropna()
        row.update(
            {
                "window_count": int(frame["window_id"].nunique()),
                "ok_rows": int(len(frame)),
                "short_sample_warning_rows": int(frame["is_short_sample_warning"].fillna(False).sum()),
                "avg_total_return": float(total_return.mean()) if not total_return.empty else float("nan"),
                "min_total_return": float(total_return.min()) if not total_return.empty else float("nan"),
                "positive_total_return_windows": int((total_return > 0).sum()),
                "avg_relative_wealth_vs_equal_weight_long_only": (
                    float(relative_wealth.mean()) if not relative_wealth.empty else float("nan")
                ),
                "min_relative_wealth_vs_equal_weight_long_only": (
                    float(relative_wealth.min()) if not relative_wealth.empty else float("nan")
                ),
                "positive_relative_wealth_windows": int((relative_wealth > 0).sum()),
                "avg_sharpe": float(sharpe.mean()) if not sharpe.empty else float("nan"),
                "min_sharpe": float(sharpe.min()) if not sharpe.empty else float("nan"),
                "worst_max_drawdown": float(frame["portfolio_max_drawdown"].min()),
                "avg_calmar": float(frame["portfolio_calmar"].mean()),
                "avg_hit_ratio": float(frame["hit_ratio"].mean()),
                "avg_gross_turnover": float(frame["average_gross_turnover"].mean()),
                "avg_turnover_cost_bps": float(frame["average_turnover_cost_bps"].mean()),
                "avg_rebalance_count": float(frame["rebalance_count"].mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)


def build_portfolio_delta_table(
    view_df: pd.DataFrame,
    specs: list[StrictExperimentSpec],
) -> pd.DataFrame:
    """按三层实验结构计算组合增量。"""

    if view_df.empty:
        return pd.DataFrame()
    key_columns = ["strategy_name", "window_mode", "top_k", "cost_bps", "neutral_mode"]
    rows = []
    for spec in specs:
        if spec.comparison_baseline is None:
            continue
        comparison = view_df[view_df["experiment"] == spec.experiment].copy()
        baseline = view_df[view_df["experiment"] == spec.comparison_baseline].copy()
        if comparison.empty or baseline.empty:
            continue
        baseline_lookup = baseline.set_index(key_columns)
        for _, row in comparison.iterrows():
            key = tuple(row[column] for column in key_columns)
            if key not in baseline_lookup.index:
                continue
            base = baseline_lookup.loc[key]
            if isinstance(base, pd.DataFrame):
                base = base.iloc[0]
            delta_row = {
                "experiment": spec.experiment,
                "feature_group": spec.feature_group,
                "feature_subset": spec.feature_subset,
                "strategy_name": row["strategy_name"],
                "window_mode": row["window_mode"],
                "top_k": row["top_k"],
                "cost_bps": row["cost_bps"],
                "neutral_mode": row["neutral_mode"],
                "baseline_experiment": spec.comparison_baseline,
            }
            for column in [
                "avg_total_return",
                "min_total_return",
                "avg_relative_wealth_vs_equal_weight_long_only",
                "min_relative_wealth_vs_equal_weight_long_only",
                "avg_sharpe",
                "min_sharpe",
                "positive_total_return_windows",
                "positive_relative_wealth_windows",
            ]:
                delta_row[f"delta_{column}"] = float(row[column] - base[column])
            rows.append(delta_row)
    return pd.DataFrame(rows)


def write_report(
    *,
    output_dir: Path,
    dataset_summary: dict[str, Any],
    factor_zoo_summary_df: pd.DataFrame,
    model_metrics_df: pd.DataFrame,
    model_delta_df: pd.DataFrame,
    model_subwindow_metrics_df: pd.DataFrame,
    model_subwindow_delta_df: pd.DataFrame,
    increment_verdict_df: pd.DataFrame,
    fold_metrics_df: pd.DataFrame,
    view_df: pd.DataFrame,
    portfolio_delta_df: pd.DataFrame,
    runtime_df: pd.DataFrame,
) -> None:
    """写一份直接可读的严格实验报告。"""

    top_model = model_metrics_df.sort_values("pearson_ic_mean", ascending=False).copy()
    top_views = (
        view_df.sort_values(
            ["avg_relative_wealth_vs_equal_weight_long_only", "avg_total_return"],
            ascending=False,
        ).head(30).copy()
        if not view_df.empty
        else pd.DataFrame()
    )
    top_portfolio_delta = (
        portfolio_delta_df.sort_values(
            "delta_avg_relative_wealth_vs_equal_weight_long_only",
            ascending=False,
        ).head(30).copy()
        if not portfolio_delta_df.empty
        else pd.DataFrame()
    )

    report = f"""# Strict Mined Factor Incremental Experiment

## 1. Purpose

这个实验回答两个递进问题：

```text
1. Alpha191 相对纯技术基线是否有增量？
2. validation-selected 自挖因子相对“技术 + Alpha191”基线是否还有增量？
```

所有组共用相同时间边界、Top-N 特征选择、walk-forward 和模型协议。
默认同时使用 Ridge、Lasso 和 XGBoost，避免结论只依赖单一模型类型。

## 2. Dataset

```json
{dumps_strict_json(dataset_summary)}
```

## 3. Factor Groups

{dataframe_to_markdown(factor_zoo_summary_df)}

## 4. Model-Layer Metrics

{dataframe_to_markdown(top_model)}

## 5. Model-Layer Delta vs Progressive Baseline

{dataframe_to_markdown(model_delta_df)}

## 6. OOS Subwindow Metrics

{dataframe_to_markdown(model_subwindow_metrics_df)}

## 7. OOS Subwindow Delta

{dataframe_to_markdown(model_subwindow_delta_df)}

## 8. Pre-Registered Increment Verdict

{dataframe_to_markdown(increment_verdict_df)}

## 9. Walk-Forward Calibration Folds

For mined-factor groups, formulas were already chosen from pre-OOS validation data.
These folds calibrate the downstream model and must not be presented as nested,
independent mined-factor evidence. Only the untouched final OOS delta supports the
current incremental-factor verdict.

{dataframe_to_markdown(fold_metrics_df)}

## 10. Top Portfolio Views

{dataframe_to_markdown(top_views)}

## 11. Portfolio Delta vs Progressive Baseline

{dataframe_to_markdown(top_portfolio_delta)}

## 12. Runtime

{dataframe_to_markdown(runtime_df)}

## 13. Reading Rule

- 如果 `strict_ppo` / `strict_warm_gp` 的 IC 提升，但 selected mined feature count 为 0，说明提升不是新增因子直接贡献。
- 如果模型层提升但组合层不提升，说明信号不容易转成当前 Top-K 多空组合。
- 如果只有某一个 Top-K 或某一个窗口好看，不能宣称稳定 alpha。
- Alpha191 只和 `strict_technical_baseline` 比较；自挖因子只和 `strict_alpha191_baseline` 比较。
- 真正可以写进结论的自挖因子结果，必须在总 OOS、多个子窗口和组合参数上都有稳定增量。
"""
    (output_dir / "strict_experiment_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    configure_runtime_warning_display()
    args = parse_args()

    start_time = time.perf_counter()
    data_path = resolve_project_path(args.data_path)
    output_dir = read_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_names = resolve_requested_models(args.models)
    print(f"[Info] Strict models: {', '.join(model_names)}", flush=True)

    train_df, test_df, baseline_feature_columns, feature_metadata, preprocessing_summary, dataset_summary = (
        load_or_build_baseline_features(args)
    )
    dataset_summary.update(
        {
            "models": ",".join(model_names),
            "top_n": int(args.top_n),
            "validation_score_metric": args.validation_score_metric,
            "preprocessing_summary": preprocessing_summary,
            "feature_counts": feature_metadata.get("feature_counts", {}),
        }
    )

    requested_groups = list(dict.fromkeys(args.mined_groups))
    need_warm_gp = any(group in {"warm_gp", "warm_gp_ppo"} for group in requested_groups)
    need_ppo = any(group in {"ppo", "warm_gp_ppo"} for group in requested_groups)
    data_sha256 = sha256_file(data_path)
    empty_zoo = pd.DataFrame(columns=["candidate_id", "formula", "selection_source", "factor_source"])
    warm_gp_zoo = (
        load_named_factor_zoo(
            read_path(args.warm_gp_zoo_path),
            "warm_gp",
            allowed_formula_fields=set(baseline_feature_columns),
            expected_target_horizon=args.target_horizon,
            expected_sample_start_date=args.sample_start_date,
            expected_oos_start_date=args.oos_start_date,
            expected_data_sha256=data_sha256,
            expected_price_adjustment_mode=args.price_adjustment_mode,
        )
        if need_warm_gp
        else empty_zoo.copy()
    )
    ppo_zoo = (
        load_named_factor_zoo(
            read_path(args.ppo_zoo_path),
            "ppo",
            allowed_formula_fields=set(baseline_feature_columns),
            expected_target_horizon=args.target_horizon,
            expected_sample_start_date=args.sample_start_date,
            expected_oos_start_date=args.oos_start_date,
            expected_data_sha256=data_sha256,
            expected_price_adjustment_mode=args.price_adjustment_mode,
        )
        if need_ppo
        else empty_zoo.copy()
    )
    factor_groups = build_factor_groups(warm_gp_zoo, ppo_zoo)
    specs = build_strict_specs(requested_groups)

    experiment_contract = {
        "data_sha256": data_sha256,
        "target_horizon": int(args.target_horizon),
        "sample_start_date": args.sample_start_date,
        "oos_start_date": args.oos_start_date,
        "price_adjustment_mode": args.price_adjustment_mode,
        "max_alpha": int(args.max_alpha),
        "alpha_factors": list(args.alpha_factors),
        "models": model_names,
        "n_splits": int(args.n_splits),
        "top_n": int(args.top_n),
        "validation_score_metric": args.validation_score_metric,
        "requested_mined_groups": requested_groups,
        "factor_zoo_paths": {
            "ppo": str(args.ppo_zoo_path) if need_ppo else None,
            "warm_gp": str(args.warm_gp_zoo_path) if need_warm_gp else None,
        },
        "factor_zoo_contract": {
            "required_version": STRICT_SELECTION_CONTRACT_VERSION,
            "required_selection_source": STRICT_SELECTION_SOURCE,
            "requires_matching_data_sha256": True,
            "requires_validation_end_before_oos": True,
            "allows_oos_columns_in_selection": False,
        },
        "portfolio_protocol": {
            "enabled": not bool(args.skip_portfolio),
            "holding_clock": args.holding_clock,
            "signal_delay_days": int(args.signal_delay_days),
            "top_k_list": [int(value) for value in args.top_k_list],
            "cost_bps_list": [float(value) for value in args.cost_bps_list],
            "neutral_mode": args.neutral_mode,
            "borrow_cost_bps": float(args.borrow_cost_bps),
        },
        "progressive_experiments": [
            {
                "experiment": spec.experiment,
                "feature_group": spec.feature_group,
                "feature_subset": spec.feature_subset,
                "comparison_baseline": spec.comparison_baseline,
                "description": spec.description,
            }
            for spec in specs
        ],
        "increment_verdict_rule": {
            "overall_core_metrics": ["pearson_ic_mean", "spearman_ic_mean", "long_short_spread"],
            "minimum_overall_improvements": 2,
            "minimum_non_full_subwindows": 2,
            "minimum_positive_subwindow_ratio": 0.5,
            "mined_factor_must_enter_final_top_n": True,
            "claim_boundary": "preliminary model increment gate only; not tradable-alpha evidence",
        },
    }
    (output_dir / "strict_experiment_contract.json").write_text(
        dumps_strict_json(experiment_contract),
        encoding="utf-8",
    )

    factor_zoo_summary_df = pd.DataFrame(
        [
            {
                "feature_group": group_name,
                "factor_count": int(len(group_df)),
                "sources": ",".join(sorted(group_df["factor_source"].dropna().unique()))
                if "factor_source" in group_df.columns and not group_df.empty
                else "",
                "hash": dataframe_hash(group_df, ["candidate_id", "formula"]) if not group_df.empty else "empty",
                "selection_sources": ",".join(sorted(group_df["selection_source"].dropna().astype(str).unique()))
                if "selection_source" in group_df.columns and not group_df.empty
                else "",
                "selection_contract_versions": ",".join(
                    sorted(group_df["selection_contract_version"].dropna().astype(str).unique())
                )
                if "selection_contract_version" in group_df.columns and not group_df.empty
                else "",
                "selection_contract_hashes": ",".join(
                    sorted(group_df["selection_contract_sha256"].dropna().astype(str).unique())
                )
                if "selection_contract_sha256" in group_df.columns and not group_df.empty
                else "",
                "selection_validation_ranges": ",".join(
                    sorted(
                        (
                            group_df["selection_validation_start"].astype(str)
                            + ".."
                            + group_df["selection_validation_end"].astype(str)
                        ).dropna().unique()
                    )
                )
                if {
                    "selection_validation_start",
                    "selection_validation_end",
                }.issubset(group_df.columns)
                and not group_df.empty
                else "",
            }
            for group_name, group_df in factor_groups.items()
            if group_name == "baseline" or group_name in requested_groups
        ]
    )
    factor_zoo_summary_df.to_csv(output_dir / "factor_zoo_summary.csv", index=False)

    model_metric_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    predictions_by_experiment: dict[str, pd.DataFrame] = {}
    materialized_cache: dict[
        tuple[str, str],
        tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]],
    ] = {}

    for spec in progress_iter(specs, total=len(specs), desc="Strict experiment groups"):
        group_start = time.perf_counter()
        materialized_key = (spec.feature_subset, spec.feature_group)
        if materialized_key not in materialized_cache:
            materialize_start = time.perf_counter()
            active_baseline_columns = select_strict_baseline_feature_columns(
                feature_metadata,
                spec.feature_subset,
            )
            materialized_cache[materialized_key] = materialize_feature_group(
                train_df=train_df,
                test_df=test_df,
                baseline_feature_columns=active_baseline_columns,
                factor_zoo=factor_groups[spec.feature_group],
            )
            runtime_rows.append(
                {
                    "experiment": spec.experiment,
                    "stage": "materialize_feature_group",
                    "runtime_seconds": time.perf_counter() - materialize_start,
                }
            )

        group_train_df, group_test_df, feature_columns, mined_columns = materialized_cache[materialized_key]
        metrics, prediction_df, _ = train_strict_pipeline_group(
            spec=spec,
            train_df=group_train_df,
            test_df=group_test_df,
            feature_columns=feature_columns,
            mined_columns=mined_columns,
            factor_zoo=factor_groups[spec.feature_group],
            args=args,
            output_dir=output_dir,
            model_names=model_names,
        )
        model_metric_rows.append(metrics)
        predictions_by_experiment[spec.experiment] = prediction_df
        runtime_rows.append(
            {
                "experiment": spec.experiment,
                "stage": "experiment_total",
                "runtime_seconds": time.perf_counter() - group_start,
            }
        )

    model_metrics_df = pd.DataFrame(model_metric_rows)
    model_metrics_df.to_csv(output_dir / "strict_model_metrics.csv", index=False)
    model_delta_df = build_model_delta_table(model_metrics_df, specs)
    model_delta_df.to_csv(output_dir / "strict_model_metric_delta.csv", index=False)

    model_subwindow_metrics_df = build_model_subwindow_metrics(
        predictions_by_experiment=predictions_by_experiment,
        specs=specs,
        oos_start_date=args.oos_start_date,
    )
    model_subwindow_metrics_df.to_csv(output_dir / "strict_model_subwindow_metrics.csv", index=False)
    model_subwindow_delta_df = build_model_subwindow_delta_table(model_subwindow_metrics_df, specs)
    model_subwindow_delta_df.to_csv(output_dir / "strict_model_subwindow_delta.csv", index=False)

    increment_verdict_df = build_increment_verdict_table(
        model_metrics_df=model_metrics_df,
        model_delta_df=model_delta_df,
        subwindow_delta_df=model_subwindow_delta_df,
        specs=specs,
    )
    increment_verdict_df.to_csv(output_dir / "strict_increment_verdict.csv", index=False)

    fold_metrics_df = collect_strict_fold_metrics(output_dir, specs)
    fold_metrics_df.to_csv(output_dir / "strict_walk_forward_fold_metrics.csv", index=False)

    portfolio_df = pd.DataFrame()
    view_df = pd.DataFrame()
    portfolio_delta_df = pd.DataFrame()
    if not args.skip_portfolio:
        portfolio_start = time.perf_counter()
        portfolio_df = run_strict_portfolio_views(
            specs=specs,
            output_dir=output_dir,
            data_path=data_path,
            oos_start_date=args.oos_start_date,
            args=args,
        )
        portfolio_df.to_csv(output_dir / "strict_portfolio_metrics.csv", index=False)
        view_df = aggregate_portfolio_views(portfolio_df)
        view_df.to_csv(output_dir / "strict_portfolio_view_summary.csv", index=False)
        portfolio_delta_df = build_portfolio_delta_table(view_df, specs)
        portfolio_delta_df.to_csv(output_dir / "strict_portfolio_view_delta.csv", index=False)
        runtime_rows.append(
            {
                "experiment": "ALL",
                "stage": "portfolio_views_total",
                "runtime_seconds": time.perf_counter() - portfolio_start,
            }
        )

    runtime_rows.append(
        {
            "experiment": "ALL",
            "stage": "total_runtime",
            "runtime_seconds": time.perf_counter() - start_time,
        }
    )
    runtime_df = pd.DataFrame(runtime_rows)
    runtime_df.to_csv(output_dir / "strict_runtime.csv", index=False)

    write_report(
        output_dir=output_dir,
        dataset_summary=dataset_summary,
        factor_zoo_summary_df=factor_zoo_summary_df,
        model_metrics_df=model_metrics_df,
        model_delta_df=model_delta_df,
        model_subwindow_metrics_df=model_subwindow_metrics_df,
        model_subwindow_delta_df=model_subwindow_delta_df,
        increment_verdict_df=increment_verdict_df,
        fold_metrics_df=fold_metrics_df,
        view_df=view_df,
        portfolio_delta_df=portfolio_delta_df,
        runtime_df=runtime_df,
    )

    print("[Info] Strict mined-factor experiment finished.", flush=True)
    print(f"[Info] Output dir: {output_dir}", flush=True)
    print(f"[Info] Report: {output_dir / 'strict_experiment_report.md'}", flush=True)


if __name__ == "__main__":
    main()
