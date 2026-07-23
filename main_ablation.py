"""消融实验入口。

这个入口和 `main.py`、`main_experiments.py` 的定位不同：

- `main.py` 负责单次完整训练；
- `main_experiments.py` 负责对比不同目标周期和模型组合；
- `main_ablation.py` 专门回答“某个模块到底有没有贡献”。

这里的核心任务包括：

1. 固定一个相对稳定的 baseline；
2. 每次只改一个关键模块；
3. 比较改动前后 OOS 指标是否发生了可解释变化；
4. 用结果回答“这个模块值不值得保留”。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from main_experiments import (
    ALL_MODEL_SUITE,
    EXPERIMENT_STAGE_LABELS,
    configure_runtime_warning_display,
    finalize_model_weights,
    finish_stage_progress,
    prepare_raw_data,
    resolve_requested_models,
    write_timing_artifacts,
)
from src.alpha191 import list_supported_alpha_factors
from src.alpha_diagnostics import ALPHA_FAMILIES, classify_alpha_family
from src.data_loader import activate_target_horizon
from src.feature_cache import build_feature_cache_key, load_feature_cache, save_feature_cache
from src.feature_selector import FeatureSelector
from src.model import ModelEnsemble, build_model, normalize_feature_importance
from src.preprocessing import DEFAULT_WINSORIZE_QUANTILE, apply_cross_sectional_preprocessing
from src.preprocessing_cache import build_preprocessing_cache_key, load_preprocessing_cache, save_preprocessing_cache
from src.progress import create_progress_bar, format_duration
from src.project_paths import resolve_project_path
from src.reporting import calculate_prediction_metrics, write_training_report
from src.runtime_config import (
    DEFAULT_ABLATION_MODEL_ROOT,
    DEFAULT_ABLATION_OUTPUT_ROOT,
    DEFAULT_OOS_START_DATE,
    DEFAULT_PRIMARY_DATA_PATH,
    DEFAULT_PRIMARY_TARGET_HORIZON,
    DEFAULT_PRIMARY_UNIVERSE,
    DEFAULT_SAMPLE_START_DATE,
    LINEAR_MODEL_SUITE,
)
from src.time_series_pipeline import DEFAULT_HISTORY_WINDOW, strict_time_split_feature_engineering
from src.validation import run_walk_forward_validation
from src.validation_cache import build_validation_cache_key, load_validation_cache, save_validation_cache


ABLATION_PRESETS = {
    "baseline_all_models": {
        "description": "完整基线：全特征 + 横截面预处理 + 全模型",
        "model_names": ALL_MODEL_SUITE,
        "feature_subset": "all",
        "apply_preprocessing": True,
        "apply_neutralization": True,
    },
    "linear_models_only": {
        "description": "只保留线性模型，观察非线性模型是否真的贡献了增益",
        "model_names": LINEAR_MODEL_SUITE,
        "feature_subset": "all",
        "apply_preprocessing": True,
        "apply_neutralization": True,
    },
    "no_cross_sectional_preprocessing": {
        "description": "关闭横截面预处理，检验 winsorize / z-score / neutralization 整体贡献",
        "model_names": ALL_MODEL_SUITE,
        "feature_subset": "all",
        "apply_preprocessing": False,
        "apply_neutralization": False,
    },
    "no_neutralization": {
        "description": "保留 winsorize + z-score，但关闭行业 / 市值中性化",
        "model_names": ALL_MODEL_SUITE,
        "feature_subset": "all",
        "apply_preprocessing": True,
        "apply_neutralization": False,
    },
    "technical_only": {
        "description": "只保留原始量价 + 技术指标，不使用 Alpha191 因子",
        "model_names": ALL_MODEL_SUITE,
        "feature_subset": "technical_only",
        "apply_preprocessing": True,
        "apply_neutralization": True,
    },
    "alpha_only": {
        "description": "只保留原始量价 + Alpha191 因子，不使用技术指标",
        "model_names": ALL_MODEL_SUITE,
        "feature_subset": "alpha_only",
        "apply_preprocessing": True,
        "apply_neutralization": True,
    },
    "alpha_family_baseline_linear": {
        "description": "Alpha 家族消融基线：技术指标 + 线性模型，不加入 Alpha191",
        "model_names": LINEAR_MODEL_SUITE,
        "feature_subset": "technical_only",
        "apply_preprocessing": True,
        "apply_neutralization": True,
    },
    "alpha_family_all_linear": {
        "description": "Alpha 家族消融上界：技术指标 + 全部 Alpha191 + 线性模型",
        "model_names": LINEAR_MODEL_SUITE,
        "feature_subset": "technical_plus_all_alpha",
        "apply_preprocessing": True,
        "apply_neutralization": True,
    },
}


for alpha_family in ALPHA_FAMILIES:
    # 这些预设用于回答一个非常具体的问题：
    # “某一类 Alpha191 公式相对技术指标基线是否有增量贡献？”
    # 这里使用线性模型，是为了减少模型非线性带来的解释噪声，
    # 让差异更接近“特征家族本身”的贡献。
    ABLATION_PRESETS[f"alpha_family_{alpha_family}"] = {
        "description": f"Alpha 家族消融：技术指标 + {alpha_family} Alpha191 + 线性模型",
        "model_names": LINEAR_MODEL_SUITE,
        "feature_subset": f"technical_plus_alpha_family:{alpha_family}",
        "apply_preprocessing": True,
        "apply_neutralization": True,
    }


DEFAULT_ABLATION_NAMES = [
    "baseline_all_models",
    "linear_models_only",
    "no_cross_sectional_preprocessing",
    "no_neutralization",
    "technical_only",
    "alpha_only",
]


def parse_args() -> argparse.Namespace:
    """解析消融实验参数。"""

    parser = argparse.ArgumentParser(description="Run ablation studies on the current quant pipeline.")
    parser.add_argument("--data-path", type=str, default=DEFAULT_PRIMARY_DATA_PATH, help="输入数据路径。")
    parser.add_argument(
        "--output-root-dir",
        type=str,
        default=DEFAULT_ABLATION_OUTPUT_ROOT,
        help="消融实验输出根目录。",
    )
    parser.add_argument(
        "--model-root-dir",
        type=str,
        default=DEFAULT_ABLATION_MODEL_ROOT,
        help="消融实验模型与特征文件根目录。",
    )
    parser.add_argument(
        "--ablation-names",
        nargs="+",
        default=list(DEFAULT_ABLATION_NAMES),
        help="要运行的消融实验名字列表。",
    )
    parser.add_argument("--target-horizon", type=int, default=DEFAULT_PRIMARY_TARGET_HORIZON, choices=[1, 5, 10], help="当前消融实验统一使用的目标周期。")
    parser.add_argument("--sample-start-date", type=str, default=DEFAULT_SAMPLE_START_DATE, help="样本起始日期。")
    parser.add_argument("--oos-start-date", type=str, default=DEFAULT_OOS_START_DATE, help="OOS 起始日期。")
    parser.add_argument("--test-size", type=float, default=0.2, help="未指定 OOS 日期时的后段测试比例。")
    parser.add_argument("--top-n", type=int, default=50, help="最终保留特征数量。")
    parser.add_argument("--missing-threshold", type=float, default=0.5, help="缺失率过滤阈值。")
    parser.add_argument("--variance-threshold", type=float, default=0.001, help="低方差过滤阈值。")
    parser.add_argument("--correlation-threshold", type=float, default=0.95, help="高相关过滤阈值。")
    parser.add_argument(
        "--feature-score-method",
        type=str,
        choices=["correlation", "mutual_info"],
        default="correlation",
        help="特征打分方式。",
    )
    parser.add_argument("--n-splits", type=int, default=5, help="walk-forward fold 数量。")
    parser.add_argument(
        "--validation-score-metric",
        type=str,
        choices=["pearson_corr", "pearson_ic_mean", "spearman_ic_mean", "long_short_return"],
        default="pearson_ic_mean",
        help="模型集成权重依据的验证指标。",
    )
    parser.add_argument("--random-state", type=int, default=42, help="随机种子。")
    parser.add_argument("--fetch-yfinance", action="store_true", help="运行前是否重新下载真实数据。")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=".cache",
        help="缓存目录。当前用于保存横截面预处理后的 train/test 表。",
    )
    parser.add_argument(
        "--disable-preprocessing-cache",
        action="store_true",
        help="关闭横截面预处理缓存。",
    )
    parser.add_argument("--symbols", nargs="+", default=None, help="自定义下载股票列表。")
    parser.add_argument("--universe", type=str, default=DEFAULT_PRIMARY_UNIVERSE, help="默认下载股票池。")
    parser.add_argument("--start-date", type=str, default=DEFAULT_SAMPLE_START_DATE, help="下载开始日期。")
    parser.add_argument("--end-date", type=str, default=None, help="下载结束日期。")
    parser.add_argument("--auto-adjust", action="store_true", help="下载时是否自动复权。")
    return parser.parse_args()


def select_feature_subset(feature_metadata: dict, subset_name: str) -> list[str]:
    """根据消融实验配置挑选要保留的特征集合。

    这里故意把逻辑写得比较直白，因为消融实验最重要的是“可解释”：

    - `all`：使用完整特征集合；
    - `technical_only`：保留原始量价 + 技术指标，不使用 Alpha191；
    - `alpha_only`：保留原始量价 + Alpha191，不使用技术指标；
    - `technical_plus_alpha_family:<family>`：技术指标基线 + 某个 Alpha 家族；
    - `technical_plus_all_alpha`：技术指标基线 + 全部 Alpha191。
    """

    raw_feature_columns = list(feature_metadata.get("raw_feature_columns", []))
    fundamental_raw_columns = list(feature_metadata.get("fundamental_raw_columns", []))
    base_feature_columns = list(feature_metadata.get("base_feature_columns", []))
    advanced_feature_columns = list(feature_metadata.get("advanced_feature_columns", []))
    alpha_feature_columns = list(feature_metadata.get("alpha_feature_columns", []))

    technical_columns = raw_feature_columns + fundamental_raw_columns + base_feature_columns + advanced_feature_columns

    if subset_name == "all":
        selected = (
            raw_feature_columns
            + fundamental_raw_columns
            + base_feature_columns
            + advanced_feature_columns
            + alpha_feature_columns
        )
    elif subset_name == "technical_only":
        selected = technical_columns
    elif subset_name == "alpha_only":
        selected = raw_feature_columns + fundamental_raw_columns + alpha_feature_columns
    elif subset_name == "technical_plus_all_alpha":
        selected = technical_columns + alpha_feature_columns
    elif subset_name.startswith("technical_plus_alpha_family:"):
        requested_family = subset_name.split(":", 1)[1].strip()
        if requested_family not in ALPHA_FAMILIES:
            raise ValueError(f"Unsupported Alpha family: {requested_family}. Supported families: {ALPHA_FAMILIES}")
        family_alpha_columns = [
            column
            for column in alpha_feature_columns
            if classify_alpha_family(column) == requested_family
        ]
        selected = technical_columns + family_alpha_columns
    else:
        raise ValueError(f"Unsupported feature subset: {subset_name}")

    return list(dict.fromkeys(selected))


def resolve_alpha_factor_names_for_subset(subset_name: str) -> list[str] | None:
    """根据消融实验的特征集合，决定特征工程阶段应该生成哪些 Alpha。

    返回值语义：

    - `None`：使用 Alpha191 默认全量；
    - `[]`：明确不生成任何 Alpha；
    - `["alpha006", ...]`：只生成指定 Alpha。

    这个函数的价值很实际：`technical_only` 不应该花时间生成 176 个 Alpha，
    `technical_plus_alpha_family:momentum` 也不应该生成其他家族的 Alpha。
    """

    if subset_name == "technical_only":
        return []

    supported_alpha_names = list_supported_alpha_factors()

    if subset_name.startswith("technical_plus_alpha_family:"):
        requested_family = subset_name.split(":", 1)[1].strip()
        if requested_family not in ALPHA_FAMILIES:
            raise ValueError(f"Unsupported Alpha family: {requested_family}. Supported families: {ALPHA_FAMILIES}")
        return [
            alpha_name
            for alpha_name in supported_alpha_names
            if classify_alpha_family(alpha_name) == requested_family
        ]

    # `all`、`alpha_only`、`technical_plus_all_alpha` 都需要完整 Alpha191。
    # 返回 None 可以复用原有默认全量生成逻辑。
    return None


def summarize_weighted_feature_importance(importance_frames: list[pd.DataFrame]) -> pd.DataFrame:
    """把多个模型的重要性按最终模型权重融合。"""

    if not importance_frames:
        return pd.DataFrame(columns=["feature", "importance"])

    merged = pd.concat(importance_frames, ignore_index=True)
    return (
        merged.groupby("feature", as_index=False)["weighted_importance"]
        .sum()
        .rename(columns={"weighted_importance": "importance"})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def write_ablation_summary(summary_df: pd.DataFrame, output_root_dir: Path) -> None:
    """把全部消融实验摘要写成 CSV 和 Markdown。"""

    output_root_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_root_dir / "ablation_summary.csv", index=False)

    if summary_df.empty:
        markdown_text = "# Ablation Summary\n\n_No ablation experiments were executed._\n"
    else:
        markdown_text = (
            "# Ablation Summary\n\n"
            "这份摘要表用于比较不同模块开关或特征集合对 OOS 表现的影响。\n\n"
            + summary_df.to_markdown(index=False)
            + "\n"
        )

    (output_root_dir / "ablation_summary.md").write_text(markdown_text, encoding="utf-8")


def _extract_alpha_family_from_subset(subset_name: str) -> str:
    """从 feature subset 名称中提取 Alpha 家族标签，方便报告聚合。"""

    if subset_name == "technical_only":
        return "baseline_technical"
    if subset_name == "technical_plus_all_alpha":
        return "all_alpha"
    if subset_name.startswith("technical_plus_alpha_family:"):
        return subset_name.split(":", 1)[1].strip()
    if subset_name == "alpha_only":
        return "alpha_only"
    return "not_alpha_family_ablation"


def write_alpha_family_ablation_report(summary_df: pd.DataFrame, output_root_dir: Path) -> None:
    """输出 Alpha 家族消融报告。

    普通 `ablation_summary.csv` 适合机器读取，但面试表达需要更直接：

    1. 技术指标 baseline 是多少；
    2. 加入某个 Alpha 家族后 IC / long-short 改善了多少；
    3. 全部 Alpha 是否优于最好的单一家族；
    4. 哪些家族可能只是增加噪声或重复信号。
    """

    if summary_df.empty:
        return

    family_df = summary_df.copy()
    family_df["alpha_family"] = family_df["feature_subset_mode"].map(_extract_alpha_family_from_subset)
    family_df = family_df[
        (family_df["ablation_name"].str.startswith("alpha_family_"))
        | (family_df["alpha_family"].isin(["baseline_technical", "all_alpha"]))
    ].copy()
    if family_df.empty:
        return

    baseline_candidates = family_df[family_df["ablation_name"] == "alpha_family_baseline_linear"]
    if baseline_candidates.empty:
        baseline_candidates = family_df[family_df["alpha_family"] == "baseline_technical"]

    baseline_row = baseline_candidates.iloc[0] if not baseline_candidates.empty else None
    baseline_ic = float(baseline_row["pearson_ic_mean"]) if baseline_row is not None else float("nan")
    baseline_ls = float(baseline_row["long_short_return"]) if baseline_row is not None else float("nan")

    family_df["delta_pearson_ic_vs_baseline"] = family_df["pearson_ic_mean"] - baseline_ic
    family_df["delta_long_short_vs_baseline"] = family_df["long_short_return"] - baseline_ls
    family_df = family_df.sort_values(
        ["delta_pearson_ic_vs_baseline", "delta_long_short_vs_baseline"],
        ascending=[False, False],
    )

    practical_delta_threshold = 1e-4
    family_rows = family_df[
        family_df["alpha_family"].isin(ALPHA_FAMILIES)
    ].copy()
    useful_rows = family_rows[
        (family_rows["delta_pearson_ic_vs_baseline"] > practical_delta_threshold)
        | (family_rows["delta_long_short_vs_baseline"] > practical_delta_threshold)
    ]
    noisy_rows = family_rows[
        (family_rows["delta_pearson_ic_vs_baseline"] <= practical_delta_threshold)
        & (family_rows["delta_long_short_vs_baseline"] <= practical_delta_threshold)
    ]

    report_columns = [
        "ablation_name",
        "alpha_family",
        "selected_feature_count",
        "pearson_ic_mean",
        "spearman_ic_mean",
        "long_short_return",
        "delta_pearson_ic_vs_baseline",
        "delta_long_short_vs_baseline",
        "total_runtime_readable",
    ]
    available_columns = [column for column in report_columns if column in family_df.columns]

    best_family_text = "暂无可判断结果"
    if not family_rows.empty:
        best_row = family_rows.iloc[0]
        best_ic_delta = float(best_row["delta_pearson_ic_vs_baseline"])
        best_ls_delta = float(best_row["delta_long_short_vs_baseline"])
        if max(best_ic_delta, best_ls_delta) <= practical_delta_threshold:
            best_family_text = (
                "No Alpha family produced a practically meaningful positive increment "
                f"above the technical baseline under the current threshold ({practical_delta_threshold})."
            )
        else:
            best_family_text = (
                f"`{best_row['alpha_family']}` currently ranks first by delta Pearson IC "
                f"({best_ic_delta:.6f}) and delta long-short ({best_ls_delta:.6f})."
            )

    report_text = f"""# Alpha Family Ablation Report

## 1. What This Experiment Answers

这份报告只回答一个问题：

```text
在技术指标基线之上，哪一类 Alpha191 公式真的提供了增量贡献？
```

这比“我加入了 176 个 Alpha 因子”更有含金量，因为它把 Alpha191 拆成可解释的信号家族，并且用 OOS 指标判断是否值得保留。

## 2. Current Best Family

{best_family_text}

Practical positive evidence threshold:

```text
delta > {practical_delta_threshold}
```

## 3. Family Comparison

{family_df[available_columns].to_markdown(index=False)}

## 4. Families With Positive Incremental Evidence

{useful_rows[available_columns].to_markdown(index=False) if not useful_rows.empty else "_No family produced positive incremental evidence in this run._"}

## 5. Families That May Be Redundant Or Noisy

{noisy_rows[available_columns].to_markdown(index=False) if not noisy_rows.empty else "_No clearly redundant/noisy family in this run._"}

## 6. Resume Bullet Candidate

```text
Designed and ran Alpha191 family-level ablation tests, decomposing formulaic alphas into momentum, reversal, volatility, liquidity, volume-price, VWAP-deviation, range-position, and complex-mixed families, then measuring each family's incremental OOS IC and long-short contribution over a technical-indicator baseline.
```

## 7. How To Interpret

- `delta_pearson_ic_vs_baseline > 0`：加入该 Alpha 家族后，横截面预测相关性高于技术指标基线。
- `delta_long_short_vs_baseline > 0`：加入该 Alpha 家族后，按预测分组的多空收益高于技术指标基线。
- 如果单一家族优于 `all_alpha`，说明全部 Alpha 里可能有噪声，下一步应该筛 Alpha，而不是继续堆 Alpha。
- 如果 `all_alpha` 最好，说明多家族信号可能有互补性，下一步可以做 factor synergy 或 non-redundant alpha set。
"""

    (output_root_dir / "alpha_family_ablation_report.md").write_text(report_text, encoding="utf-8")


def apply_preprocessing_for_ablation(
    data: pd.DataFrame,
    feature_columns: list[str],
    apply_preprocessing: bool,
    apply_neutralization: bool,
) -> tuple[pd.DataFrame, dict]:
    """根据消融配置决定是否应用横截面预处理。"""

    if not apply_preprocessing:
        return data.copy(), {
            "enabled": False,
            "winsorize_quantile": None,
            "zscore_applied": False,
            "neutralization_applied": False,
            "size_neutralization_used": False,
            "sector_neutralization_used": False,
            "processed_date_count": int(pd.to_datetime(data["date"]).nunique()),
        }

    return apply_cross_sectional_preprocessing(
        data,
        feature_columns=feature_columns,
        apply_neutralization=apply_neutralization,
        show_progress=True,
    )


def run_single_ablation(
    ablation_name: str,
    ablation_config: dict,
    raw_data_base: pd.DataFrame,
    args: argparse.Namespace,
) -> dict:
    """执行单组消融实验。"""

    model_dir = resolve_project_path(args.model_root_dir) / ablation_name
    output_dir = resolve_project_path(args.output_root_dir) / ablation_name
    cache_root = resolve_project_path(args.cache_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    resolved_model_names = resolve_requested_models(list(ablation_config["model_names"]))
    total_stage_count = len(EXPERIMENT_STAGE_LABELS)
    experiment_start_time = time.perf_counter()
    stage_timing_records: list[dict] = []
    final_model_timing_records: list[dict] = []
    stage_progress = create_progress_bar(
        total=total_stage_count,
        description=f"{ablation_name}: stages",
        enabled=True,
    )

    stage_start = time.perf_counter()
    raw_data, target_column = activate_target_horizon(raw_data_base, target_horizon=args.target_horizon)
    stage_elapsed = time.perf_counter() - stage_start
    experiment_elapsed = time.perf_counter() - experiment_start_time
    stage_timing_records.append(
        {
            "stage_order": 1,
            "stage_key": "activate_target",
            "stage_label": "Activate target",
            "elapsed_sec": float(stage_elapsed),
            "elapsed_readable": format_duration(stage_elapsed),
            "details": target_column,
        }
    )
    finish_stage_progress(stage_progress, 1, total_stage_count, "activate target", stage_elapsed, experiment_elapsed)

    print(
        f"[Info] Running ablation: {ablation_name} | target={target_column} | "
        f"models={resolved_model_names} | feature_subset={ablation_config['feature_subset']}"
    )

    alpha_factor_names = resolve_alpha_factor_names_for_subset(ablation_config["feature_subset"])
    alpha_cache_context = None
    if alpha_factor_names == []:
        alpha_cache_context = {
            "entry": "main_ablation",
            "alpha_factor_mode": "none",
        }

    feature_cache_key = build_feature_cache_key(
        data_path=resolve_project_path(args.data_path),
        sample_start_date=args.sample_start_date,
        oos_start_date=args.oos_start_date,
        test_size=args.test_size,
        target_horizon=args.target_horizon,
        history_window=DEFAULT_HISTORY_WINDOW,
        alpha_factor_names=alpha_factor_names,
        extra_context=alpha_cache_context,
    )
    stage_start = time.perf_counter()
    cached_features = load_feature_cache(cache_root=cache_root, cache_key=feature_cache_key)
    if cached_features is not None:
        train_df, test_df, full_feature_columns, feature_metadata = cached_features
        feature_cache_status = "hit"
    else:
        train_df, test_df, full_feature_columns, feature_metadata = strict_time_split_feature_engineering(
            raw_data=raw_data,
            test_size=args.test_size,
            test_start_date=args.oos_start_date,
            history_window=DEFAULT_HISTORY_WINDOW,
            target_horizon=args.target_horizon,
            alpha_factor_names=alpha_factor_names,
            show_progress=True,
        )
        save_feature_cache(
            cache_root=cache_root,
            cache_key=feature_cache_key,
            train_df=train_df,
            test_df=test_df,
            feature_columns=full_feature_columns,
            feature_metadata=feature_metadata,
            metadata={
                "target_horizon": args.target_horizon,
                "history_window": DEFAULT_HISTORY_WINDOW,
                "alpha_factor_names": alpha_factor_names,
            },
        )
        feature_cache_status = "miss_written"
    active_feature_columns = select_feature_subset(feature_metadata, ablation_config["feature_subset"])
    feature_metadata["feature_counts"]["candidate_feature_count"] = len(active_feature_columns)
    stage_elapsed = time.perf_counter() - stage_start
    experiment_elapsed = time.perf_counter() - experiment_start_time
    stage_timing_records.append(
        {
            "stage_order": 2,
            "stage_key": "feature_engineering",
            "stage_label": "Time split + feature engineering",
            "elapsed_sec": float(stage_elapsed),
            "elapsed_readable": format_duration(stage_elapsed),
            "details": (
                f"{len(active_feature_columns)} active / {len(full_feature_columns)} total features | "
                f"feature cache {feature_cache_status}"
            ),
        }
    )
    finish_stage_progress(stage_progress, 2, total_stage_count, "feature engineering", stage_elapsed, experiment_elapsed)

    preprocessing_cache_enabled = (
        (not args.disable_preprocessing_cache)
        and ablation_config["apply_preprocessing"]
    )
    preprocessing_cache_key = build_preprocessing_cache_key(
        data_path=resolve_project_path(args.data_path),
        sample_start_date=args.sample_start_date,
        oos_start_date=args.oos_start_date,
        test_size=args.test_size,
        target_horizon=args.target_horizon,
        history_window=DEFAULT_HISTORY_WINDOW,
        feature_columns=active_feature_columns,
        apply_preprocessing=ablation_config["apply_preprocessing"],
        apply_neutralization=ablation_config["apply_neutralization"],
        winsorize_quantile=DEFAULT_WINSORIZE_QUANTILE,
        extra_context={
            "feature_subset": ablation_config["feature_subset"],
        },
    )

    cached_preprocessing = None
    if preprocessing_cache_enabled:
        stage_start = time.perf_counter()
        cached_preprocessing = load_preprocessing_cache(cache_root=cache_root, cache_key=preprocessing_cache_key)
        stage_elapsed = time.perf_counter() - stage_start
    else:
        stage_elapsed = 0.0

    if cached_preprocessing is not None:
        train_df, test_df, preprocessing_summary = cached_preprocessing
        preprocessing_summary = dict(preprocessing_summary)
        preprocessing_summary["cache_status"] = "hit"
        preprocessing_summary["cache_key"] = preprocessing_cache_key

        experiment_elapsed = time.perf_counter() - experiment_start_time
        stage_timing_records.append(
            {
                "stage_order": 3,
                "stage_key": "preprocess_train",
                "stage_label": "Cross-sectional preprocessing (train)",
                "elapsed_sec": float(stage_elapsed),
                "elapsed_readable": format_duration(stage_elapsed),
                "details": f"{len(train_df)} rows | cache hit",
            }
        )
        finish_stage_progress(stage_progress, 3, total_stage_count, "preprocess train (cache hit)", stage_elapsed, experiment_elapsed)

        stage_timing_records.append(
            {
                "stage_order": 4,
                "stage_key": "preprocess_test",
                "stage_label": "Cross-sectional preprocessing (test)",
                "elapsed_sec": 0.0,
                "elapsed_readable": format_duration(0.0),
                "details": f"{len(test_df)} rows | cache hit",
            }
        )
        finish_stage_progress(stage_progress, 4, total_stage_count, "preprocess test (cache hit)", 0.0, experiment_elapsed)
    else:
        stage_start = time.perf_counter()
        train_df, preprocessing_summary = apply_preprocessing_for_ablation(
            train_df,
            feature_columns=active_feature_columns,
            apply_preprocessing=ablation_config["apply_preprocessing"],
            apply_neutralization=ablation_config["apply_neutralization"],
        )
        stage_elapsed = time.perf_counter() - stage_start
        experiment_elapsed = time.perf_counter() - experiment_start_time
        stage_timing_records.append(
            {
                "stage_order": 3,
                "stage_key": "preprocess_train",
                "stage_label": "Cross-sectional preprocessing (train)",
                "elapsed_sec": float(stage_elapsed),
                "elapsed_readable": format_duration(stage_elapsed),
                "details": f"{len(train_df)} rows" + (" | cache miss" if preprocessing_cache_enabled else " | cache disabled"),
            }
        )
        finish_stage_progress(stage_progress, 3, total_stage_count, "preprocess train", stage_elapsed, experiment_elapsed)

        stage_start = time.perf_counter()
        test_df, _ = apply_preprocessing_for_ablation(
            test_df,
            feature_columns=active_feature_columns,
            apply_preprocessing=ablation_config["apply_preprocessing"],
            apply_neutralization=ablation_config["apply_neutralization"],
        )
        stage_elapsed = time.perf_counter() - stage_start
        experiment_elapsed = time.perf_counter() - experiment_start_time
        stage_timing_records.append(
            {
                "stage_order": 4,
                "stage_key": "preprocess_test",
                "stage_label": "Cross-sectional preprocessing (test)",
                "elapsed_sec": float(stage_elapsed),
                "elapsed_readable": format_duration(stage_elapsed),
                "details": f"{len(test_df)} rows" + (" | cache miss" if preprocessing_cache_enabled else " | cache disabled"),
            }
        )
        finish_stage_progress(stage_progress, 4, total_stage_count, "preprocess test", stage_elapsed, experiment_elapsed)

        preprocessing_summary = dict(preprocessing_summary)
        preprocessing_summary["cache_status"] = "disabled" if not preprocessing_cache_enabled else "miss_written"
        preprocessing_summary["cache_key"] = preprocessing_cache_key if preprocessing_cache_enabled else None

        if preprocessing_cache_enabled:
            save_preprocessing_cache(
                cache_root=cache_root,
                cache_key=preprocessing_cache_key,
                train_df=train_df,
                test_df=test_df,
                preprocessing_summary=preprocessing_summary,
                metadata={
                    "target_horizon": args.target_horizon,
                    "feature_subset": ablation_config["feature_subset"],
                    "feature_count": len(active_feature_columns),
                    "winsorize_quantile": DEFAULT_WINSORIZE_QUANTILE,
                    "apply_neutralization": ablation_config["apply_neutralization"],
                },
            )

    selector_config = {
        "missing_threshold": args.missing_threshold,
        "variance_threshold": args.variance_threshold,
        "correlation_threshold": args.correlation_threshold,
        "top_n": args.top_n,
        "score_method": args.feature_score_method,
        "random_state": args.random_state,
    }

    validation_cache_key = build_validation_cache_key(
        data_path=resolve_project_path(args.data_path),
        sample_start_date=args.sample_start_date,
        oos_start_date=args.oos_start_date,
        test_size=args.test_size,
        target_horizon=args.target_horizon,
        history_window=DEFAULT_HISTORY_WINDOW,
        feature_columns=active_feature_columns,
        model_names=resolved_model_names,
        selector_config=selector_config,
        n_splits=args.n_splits,
        random_state=args.random_state,
        score_metric=args.validation_score_metric,
        apply_preprocessing=ablation_config["apply_preprocessing"],
        apply_neutralization=ablation_config["apply_neutralization"],
        winsorize_quantile=DEFAULT_WINSORIZE_QUANTILE,
        extra_context={"feature_subset": ablation_config["feature_subset"]},
    )
    stage_start = time.perf_counter()
    cached_validation = load_validation_cache(cache_root=cache_root, cache_key=validation_cache_key)
    if cached_validation is not None:
        fold_metrics_df, model_summary_df, model_weights = cached_validation
        validation_detail = f"{args.n_splits} folds × {len(resolved_model_names)} models | cache hit"
    else:
        fold_metrics_df, model_summary_df, model_weights = run_walk_forward_validation(
            train_df=train_df,
            feature_columns=active_feature_columns,
            model_names=resolved_model_names,
            selector_config=selector_config,
            random_state=args.random_state,
            n_splits=args.n_splits,
            purge_days=args.target_horizon,
            score_metric=args.validation_score_metric,
            show_progress=True,
        )
        save_validation_cache(
            cache_root=cache_root,
            cache_key=validation_cache_key,
            fold_metrics_df=fold_metrics_df,
            model_summary_df=model_summary_df,
            model_weights=model_weights,
            metadata={
                "target_horizon": args.target_horizon,
                "model_names": resolved_model_names,
                "feature_subset": ablation_config["feature_subset"],
            },
        )
        validation_detail = f"{args.n_splits} folds × {len(resolved_model_names)} models | cache miss"
    stage_elapsed = time.perf_counter() - stage_start
    experiment_elapsed = time.perf_counter() - experiment_start_time
    stage_timing_records.append(
        {
            "stage_order": 5,
            "stage_key": "walk_forward",
            "stage_label": "Walk-forward validation",
            "elapsed_sec": float(stage_elapsed),
            "elapsed_readable": format_duration(stage_elapsed),
            "details": validation_detail,
        }
    )
    finish_stage_progress(stage_progress, 5, total_stage_count, "walk-forward", stage_elapsed, experiment_elapsed)

    stage_start = time.perf_counter()
    selector = FeatureSelector(**selector_config)
    selector.fit(
        train_df[active_feature_columns],
        train_df["y"],
        dates=train_df["date"],
    )
    X_train_full = selector.transform(train_df[active_feature_columns])
    y_train_full = train_df["y"].reset_index(drop=True)
    X_test = selector.transform(test_df[active_feature_columns])
    stage_elapsed = time.perf_counter() - stage_start
    experiment_elapsed = time.perf_counter() - experiment_start_time
    stage_timing_records.append(
        {
            "stage_order": 6,
            "stage_key": "final_feature_selection",
            "stage_label": "Final feature selection",
            "elapsed_sec": float(stage_elapsed),
            "elapsed_readable": format_duration(stage_elapsed),
            "details": f"{len(selector.selected_features_)} selected features",
        }
    )
    finish_stage_progress(stage_progress, 6, total_stage_count, "final feature selection", stage_elapsed, experiment_elapsed)

    final_model_weights = finalize_model_weights(resolved_model_names, model_weights)
    ensemble = ModelEnsemble()
    model_params: dict = {}
    weighted_importance_frames: list[pd.DataFrame] = []

    stage_start = time.perf_counter()
    final_model_progress = create_progress_bar(
        total=len(resolved_model_names),
        description=f"{ablation_name}: final model training",
        enabled=True,
    )
    for model_index, model_name in enumerate(resolved_model_names, start=1):
        model_start = time.perf_counter()
        model_wrapper = build_model(model_name=model_name, random_state=args.random_state)
        model_wrapper.fit(X_train_full, y_train_full)
        model_wrapper.save(model_dir / f"{model_name}_model.joblib")
        model_elapsed = time.perf_counter() - model_start

        current_weight = final_model_weights.get(model_name, 0.0)
        ensemble.add_model(model_name, model_wrapper, weight=current_weight)
        model_params[model_name] = model_wrapper.get_params()

        raw_importance_df = model_wrapper.get_feature_importance(selector.selected_features_, model_name=model_name)
        normalized_importance_df = normalize_feature_importance(raw_importance_df, model_weight=current_weight)
        weighted_importance_frames.append(normalized_importance_df)
        final_model_timing_records.append(
            {
                "model": model_name,
                "elapsed_sec": float(model_elapsed),
                "elapsed_readable": format_duration(model_elapsed),
                "ensemble_weight": float(current_weight),
            }
        )

        average_model_time = (time.perf_counter() - stage_start) / max(model_index, 1)
        estimated_remaining = average_model_time * max(len(resolved_model_names) - model_index, 0)
        final_model_progress.update(1)
        final_model_progress.set_postfix_str(
            f"{model_name} {format_duration(model_elapsed)} | est left {format_duration(estimated_remaining)}"
        )

    final_model_progress.close()
    stage_elapsed = time.perf_counter() - stage_start
    experiment_elapsed = time.perf_counter() - experiment_start_time
    stage_timing_records.append(
        {
            "stage_order": 7,
            "stage_key": "final_model_training",
            "stage_label": "Final model training",
            "elapsed_sec": float(stage_elapsed),
            "elapsed_readable": format_duration(stage_elapsed),
            "details": f"{len(resolved_model_names)} models",
        }
    )
    finish_stage_progress(stage_progress, 7, total_stage_count, "final model training", stage_elapsed, experiment_elapsed)

    stage_start = time.perf_counter()
    test_predictions = ensemble.predict(X_test)

    predictions_df = test_df[["date", "instrument_id"]].copy()
    predictions_df["date"] = pd.to_datetime(predictions_df["date"]).dt.strftime("%Y-%m-%d")
    predictions_df["predicted_y"] = test_predictions
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)

    predictions_with_actual_df = test_df[["date", "instrument_id", "y"]].copy()
    predictions_with_actual_df["date"] = pd.to_datetime(predictions_with_actual_df["date"]).dt.strftime("%Y-%m-%d")
    predictions_with_actual_df["predicted_y"] = test_predictions
    predictions_with_actual_df.to_csv(output_dir / "test_predictions_with_actual.csv", index=False)

    test_metrics = calculate_prediction_metrics(predictions_with_actual_df)
    fold_metrics_df.to_csv(output_dir / "walk_forward_fold_metrics.csv", index=False)
    model_summary_df.to_csv(output_dir / "walk_forward_model_summary.csv", index=False)
    pd.DataFrame(
        [{"model": model_name, "weight": weight} for model_name, weight in final_model_weights.items()]
    ).to_csv(model_dir / "model_weights.csv", index=False)
    pd.DataFrame({"feature": selector.selected_features_}).to_csv(model_dir / "selected_features.csv", index=False)
    selector.get_top_features(top_k=20).to_csv(model_dir / "selected_feature_scores.csv", index=False)
    selector.save(model_dir / "feature_selector.json")

    weighted_feature_importance_summary = summarize_weighted_feature_importance(weighted_importance_frames)
    if weighted_importance_frames:
        pd.concat(weighted_importance_frames, ignore_index=True).to_csv(
            model_dir / "feature_importance_by_model.csv",
            index=False,
        )
    if not weighted_feature_importance_summary.empty:
        weighted_feature_importance_summary.to_csv(model_dir / "feature_importance.csv", index=False)

    stage_elapsed = time.perf_counter() - stage_start
    total_experiment_elapsed = time.perf_counter() - experiment_start_time
    stage_timing_records.append(
        {
            "stage_order": 8,
            "stage_key": "prediction_and_report",
            "stage_label": "Prediction + report output",
            "elapsed_sec": float(stage_elapsed),
            "elapsed_readable": format_duration(stage_elapsed),
            "details": "predictions, reports, weights, importances",
        }
    )
    finish_stage_progress(stage_progress, 8, total_stage_count, "prediction + report", stage_elapsed, total_experiment_elapsed)
    stage_progress.close()

    stage_timing_df = pd.DataFrame(stage_timing_records)
    final_model_timing_df = pd.DataFrame(final_model_timing_records)
    write_timing_artifacts(output_dir=output_dir, stage_timing_df=stage_timing_df, final_model_timing_df=final_model_timing_df)

    dataset_summary = {
        "data_path": str(resolve_project_path(args.data_path)),
        "min_date": str(pd.to_datetime(raw_data["date"]).min().date()),
        "max_date": str(pd.to_datetime(raw_data["date"]).max().date()),
        "instrument_count": int(raw_data["instrument_id"].nunique()),
        "n_rows": int(len(train_df) + len(test_df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_min_date": str(pd.to_datetime(train_df["date"]).min().date()),
        "train_max_date": str(pd.to_datetime(train_df["date"]).max().date()),
        "test_min_date": str(pd.to_datetime(test_df["date"]).min().date()),
        "test_max_date": str(pd.to_datetime(test_df["date"]).max().date()),
        "n_splits": int(args.n_splits),
        "sample_start_date": args.sample_start_date,
        "oos_start_date_used": args.oos_start_date,
        "target_horizon": args.target_horizon,
        "target_column": target_column,
        "universe": args.universe if args.fetch_yfinance and not args.symbols else "custom_symbols_or_csv",
        "resolved_model_count": len(resolved_model_names),
        "experiment_name": ablation_name,
        "experiment_description": ablation_config["description"],
        "feature_subset_mode": ablation_config["feature_subset"],
        "preprocessing_mode": (
            "disabled"
            if not ablation_config["apply_preprocessing"]
            else ("winsorize+zscore+neutralize" if ablation_config["apply_neutralization"] else "winsorize+zscore")
        ),
    }

    selector_summary = {
        "missing_threshold": args.missing_threshold,
        "variance_threshold": args.variance_threshold,
        "correlation_threshold": args.correlation_threshold,
        "top_n": args.top_n,
        "score_method": args.feature_score_method,
        "stage_feature_counts": selector.stage_feature_counts_,
        "selected_feature_count": len(selector.selected_features_),
        "validation_score_metric": args.validation_score_metric,
    }

    write_training_report(
        output_path=output_dir / "training_report.md",
        dataset_summary=dataset_summary,
        feature_metadata=feature_metadata,
        preprocessing_summary=preprocessing_summary,
        selector_summary=selector_summary,
        test_metrics=test_metrics,
        model_params=model_params,
        top_score_features=selector.get_top_features(top_k=10),
        top_importance_features=weighted_feature_importance_summary.head(10),
        validation_summary_df=model_summary_df,
        model_weights=final_model_weights,
        stage_timing_df=stage_timing_df,
        final_model_timing_df=final_model_timing_df,
    )

    return {
        "ablation_name": ablation_name,
        "description": ablation_config["description"],
        "target_horizon": args.target_horizon,
        "models": ",".join(resolved_model_names),
        "feature_subset_mode": ablation_config["feature_subset"],
        "preprocessing_mode": dataset_summary["preprocessing_mode"],
        "selected_feature_count": len(selector.selected_features_),
        "total_runtime_sec": float(total_experiment_elapsed),
        "total_runtime_readable": format_duration(total_experiment_elapsed),
        **test_metrics,
    }


def main() -> None:
    """执行全部消融实验。"""

    configure_runtime_warning_display()
    args = parse_args()
    unknown_names = [name for name in args.ablation_names if name not in ABLATION_PRESETS]
    if unknown_names:
        raise ValueError(f"Unknown ablation names: {unknown_names}. Supported names: {list(ABLATION_PRESETS.keys())}")

    raw_data_base = prepare_raw_data(args)
    output_root_dir = resolve_project_path(args.output_root_dir)
    output_root_dir.mkdir(parents=True, exist_ok=True)

    summary_records: list[dict] = []
    timing_records: list[dict] = []
    overall_start = time.perf_counter()
    ablation_progress = create_progress_bar(
        total=len(args.ablation_names),
        description="Ablation presets",
        enabled=True,
    )

    for ablation_index, ablation_name in enumerate(args.ablation_names, start=1):
        start_time = time.perf_counter()
        summary_records.append(
            run_single_ablation(
                ablation_name=ablation_name,
                ablation_config=ABLATION_PRESETS[ablation_name],
                raw_data_base=raw_data_base,
                args=args,
            )
        )
        ablation_elapsed = time.perf_counter() - start_time
        total_elapsed = time.perf_counter() - overall_start
        average_ablation_time = total_elapsed / max(ablation_index, 1)
        estimated_remaining = average_ablation_time * max(len(args.ablation_names) - ablation_index, 0)
        timing_records.append(
            {
                "ablation_name": ablation_name,
                "elapsed_sec": float(ablation_elapsed),
                "elapsed_readable": format_duration(ablation_elapsed),
            }
        )
        ablation_progress.update(1)
        ablation_progress.set_postfix_str(
            f"last {ablation_name} {format_duration(ablation_elapsed)} | est left {format_duration(estimated_remaining)}"
        )
    ablation_progress.close()

    summary_df = pd.DataFrame(summary_records).sort_values(
        ["target_horizon", "pearson_ic_mean", "long_short_return"],
        ascending=[True, False, False],
    )
    write_ablation_summary(summary_df, output_root_dir)
    write_alpha_family_ablation_report(summary_df, output_root_dir)
    pd.DataFrame(timing_records).to_csv(output_root_dir / "ablation_timing_summary.csv", index=False)

    print("[Info] Ablation runner finished.")
    print(f"[Info] Summary saved to: {output_root_dir / 'ablation_summary.csv'}")
    if (output_root_dir / "alpha_family_ablation_report.md").exists():
        print(f"[Info] Alpha family report saved to: {output_root_dir / 'alpha_family_ablation_report.md'}")
    print(f"[Info] Timing summary saved to: {output_root_dir / 'ablation_timing_summary.csv'}")


if __name__ == "__main__":
    main()
