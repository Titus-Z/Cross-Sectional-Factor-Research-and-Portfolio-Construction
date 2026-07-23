from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factor_mining_workspace.single_factor_case_study import (
    build_spread_outputs,
    dataframe_to_markdown,
    load_or_build_preprocessed_train_test,
    sanitize_name,
)
from src.factor_diagnostics import summarize_factor_diagnostics
from src.runtime_config import (
    DEFAULT_FACTOR_DIAGNOSTICS_MODEL_DIR,
    DEFAULT_OOS_START_DATE,
    DEFAULT_PRIMARY_DATA_PATH,
    DEFAULT_PRIMARY_TARGET_HORIZON,
    DEFAULT_SAMPLE_START_DATE,
)


RAW_MARKET_COLUMNS = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "adjustment",
    "market_cap",
    "turnover",
    "log_return",
}

CURATED_SEED_CANDIDATES = [
    "return_std_20",
    "return_std_10",
    "return_std_5",
    "price_range",
    "xschannel_width_20",
    "boll_width_20",
    "close_to_ma_20",
    "close_to_ma_10",
    "momentum_20",
    "momentum_10",
    "vwap_gap",
    "volume_rank",
    "turnover_rank",
    "turnover_rate_rank",
    "shares_outstanding_proxy",
    "obv",
    "obv_ratio_20",
    "amt_ratio_20",
    "amt_range_20",
]

UNARY_TRANSFORMS = {
    "id": lambda series: series,
    "neg": lambda series: -series,
    "abs": lambda series: series.abs(),
    "tanh": lambda series: pd.Series(np.tanh(series), index=series.index, dtype=float),
    "signed_sq": lambda series: pd.Series(np.sign(series) * np.square(series), index=series.index, dtype=float),
}

POST_TRANSFORMS = {
    "id": lambda series: series,
    "tanh": lambda series: pd.Series(np.tanh(series), index=series.index, dtype=float),
}

BINARY_TEMPLATES = {
    "spread": lambda left, right, weight: left - right,
    "blend": lambda left, right, weight: weight * left + (1.0 - weight) * right,
    "interaction": lambda left, right, weight: left * np.tanh(right),
    "ratio": lambda left, right, weight: left / (1.0 + np.abs(right)),
    "confirm": lambda left, right, weight: left * np.sign(right),
}

WEIGHT_CHOICES = [0.25, 0.33, 0.5, 0.67, 0.75]
SEMANTIC_FAMILY_KEYWORDS = {
    "volatility": ["return_std_", "price_range"],
    "channel": ["xschannel_width_20", "boll_width_20"],
    "momentum": ["momentum_", "close_to_ma_"],
    "liquidity": ["turnover_rank", "volume_rank", "turnover_rate_rank"],
    "vwap": ["vwap_gap"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search new heuristic factors from existing non-Alpha seeds.")
    parser.add_argument("--data-path", default=DEFAULT_PRIMARY_DATA_PATH, help="原始数据路径。")
    parser.add_argument("--model-dir", default=DEFAULT_FACTOR_DIAGNOSTICS_MODEL_DIR, help="用于读取已选特征和重要度的模型目录。")
    parser.add_argument("--output-dir", default="factor_mining_workspace/heuristic_search_outputs", help="搜索结果输出根目录。")
    parser.add_argument("--cache-dir", default=".cache", help="缓存目录。")
    parser.add_argument("--sample-start-date", default=DEFAULT_SAMPLE_START_DATE, help="样本起始日期。")
    parser.add_argument("--oos-start-date", default=DEFAULT_OOS_START_DATE, help="OOS 起始日期。")
    parser.add_argument("--target-horizon", type=int, default=DEFAULT_PRIMARY_TARGET_HORIZON, help="目标周期。")
    parser.add_argument("--test-size", type=float, default=0.2, help="未指定 OOS 日期时的后段测试比例。")
    parser.add_argument("--n-groups", type=int, default=5, help="分组数量。")
    parser.add_argument("--min-cross-section", type=int, default=30, help="每个日期最少参与诊断的股票数。")
    parser.add_argument("--seed-top-k", type=int, default=14, help="参与搜索的基础种子特征数量。")
    parser.add_argument("--num-candidates", type=int, default=300, help="随机生成候选表达式数量。")
    parser.add_argument("--survivor-ratio", type=float, default=0.05, help="训练期保留比例，例如 0.05 表示保留前 5%。")
    parser.add_argument("--final-top-k", type=int, default=10, help="最终报告展示的 OOS 候选数量。")
    parser.add_argument("--max-per-family", type=int, default=2, help="最终 shortlist 中每个家族最多保留几个公式。")
    parser.add_argument("--random-seed", type=int, default=7, help="随机种子。")
    parser.add_argument(
        "--include-alpha-seeds",
        action="store_true",
        help="允许 canonical 价格尺度不变 Alpha 子集进入种子池；不会加载全部 Alpha191。",
    )
    parser.add_argument("--include-raw-market-seeds", action="store_true", help="允许 open/close/volume 等原始列进入种子池。默认关闭。")
    parser.add_argument("--disable-preprocessing-cache", action="store_true", help="关闭横截面预处理缓存。")
    return parser.parse_args()


def is_allowed_seed(feature_name: str, include_alpha_seeds: bool, include_raw_market_seeds: bool) -> bool:
    lowered = feature_name.lower()
    if lowered in {"date", "instrument_id", "sector", "y", "y_1d", "y_5d", "y_10d", "next_open"}:
        return False
    if not include_alpha_seeds and lowered.startswith("alpha"):
        return False
    if not include_raw_market_seeds and lowered in RAW_MARKET_COLUMNS:
        return False
    return True


def load_seed_feature_pool(
    train_df: pd.DataFrame,
    model_dir: Path,
    seed_top_k: int,
    include_alpha_seeds: bool,
    include_raw_market_seeds: bool,
) -> list[str]:
    ordered_features: list[str] = []
    seen: set[str] = set()

    def maybe_add(feature_name: str) -> None:
        if feature_name in seen:
            return
        if feature_name not in train_df.columns:
            return
        if not is_allowed_seed(feature_name, include_alpha_seeds, include_raw_market_seeds):
            return
        seen.add(feature_name)
        ordered_features.append(feature_name)

    selected_scores_path = model_dir / "selected_feature_scores.csv"
    importance_path = model_dir / "feature_importance.csv"
    selected_features_path = model_dir / "selected_features.csv"

    for feature_name in CURATED_SEED_CANDIDATES:
        maybe_add(feature_name)

    if selected_scores_path.exists():
        selector_df = pd.read_csv(selected_scores_path).sort_values("score", ascending=False)
        for feature_name in selector_df["feature"].dropna().astype(str):
            maybe_add(feature_name)

    if importance_path.exists():
        importance_df = pd.read_csv(importance_path).sort_values("importance", ascending=False)
        for feature_name in importance_df["feature"].dropna().astype(str):
            maybe_add(feature_name)

    if selected_features_path.exists():
        selected_df = pd.read_csv(selected_features_path)
        for feature_name in selected_df["feature"].dropna().astype(str):
            maybe_add(feature_name)

    return ordered_features[:seed_top_k]


def format_formula_unary(transform_name: str, feature_name: str) -> str:
    if transform_name == "id":
        return feature_name
    return f"{transform_name}({feature_name})"


def sample_candidate_spec(seed_features: list[str], rng: random.Random) -> dict[str, object]:
    if len(seed_features) < 2:
        raise ValueError("At least two seed features are required.")

    candidate_type = "unary" if rng.random() < 0.2 else "binary"
    if candidate_type == "unary":
        feature_name = rng.choice(seed_features)
        unary_name = rng.choice(list(UNARY_TRANSFORMS.keys()))
        post_name = rng.choice(list(POST_TRANSFORMS.keys()))
        formula = format_formula_unary(unary_name, feature_name)
        if post_name != "id":
            formula = f"{post_name}({formula})"
        return {
            "candidate_type": candidate_type,
            "feature_1": feature_name,
            "unary_1": unary_name,
            "post_transform": post_name,
            "formula": formula,
        }

    feature_1, feature_2 = rng.sample(seed_features, 2)
    unary_1 = rng.choice(list(UNARY_TRANSFORMS.keys()))
    unary_2 = rng.choice(list(UNARY_TRANSFORMS.keys()))
    template_name = rng.choice(list(BINARY_TEMPLATES.keys()))
    post_name = rng.choice(list(POST_TRANSFORMS.keys()))
    weight = rng.choice(WEIGHT_CHOICES)

    left_formula = format_formula_unary(unary_1, feature_1)
    right_formula = format_formula_unary(unary_2, feature_2)

    if template_name == "blend":
        formula = f"({weight:.2f}*{left_formula} + {(1.0 - weight):.2f}*{right_formula})"
    elif template_name == "spread":
        formula = f"({left_formula} - {right_formula})"
    elif template_name == "interaction":
        formula = f"({left_formula} * tanh({right_formula}))"
    elif template_name == "ratio":
        formula = f"({left_formula} / (1 + abs({right_formula})))"
    else:
        formula = f"({left_formula} * sign({right_formula}))"

    if post_name != "id":
        formula = f"{post_name}({formula})"

    return {
        "candidate_type": candidate_type,
        "feature_1": feature_1,
        "feature_2": feature_2,
        "unary_1": unary_1,
        "unary_2": unary_2,
        "binary_template": template_name,
        "weight": weight,
        "post_transform": post_name,
        "formula": formula,
    }


def build_candidate_series(data: pd.DataFrame, spec: dict[str, object]) -> pd.Series:
    left = UNARY_TRANSFORMS[str(spec["unary_1"])](pd.to_numeric(data[str(spec["feature_1"])], errors="coerce"))

    if spec["candidate_type"] == "unary":
        candidate = left
    else:
        right = UNARY_TRANSFORMS[str(spec["unary_2"])](pd.to_numeric(data[str(spec["feature_2"])], errors="coerce"))
        template = BINARY_TEMPLATES[str(spec["binary_template"])]
        candidate = template(left, right, float(spec["weight"]))

    candidate = pd.Series(candidate, index=data.index, dtype=float).replace([np.inf, -np.inf], np.nan)
    candidate = POST_TRANSFORMS[str(spec["post_transform"])](candidate)
    return pd.Series(candidate, index=data.index, dtype=float).replace([np.inf, -np.inf], np.nan)


def infer_formula_family(formula: str) -> str:
    matched_families: list[str] = []
    for family_name, keywords in SEMANTIC_FAMILY_KEYWORDS.items():
        if any(keyword in formula for keyword in keywords):
            matched_families.append(family_name)
    if not matched_families:
        return "other"
    return "+".join(sorted(dict.fromkeys(matched_families)))


def standardize_candidate_cross_sectionally(
    data: pd.DataFrame,
    candidate_series: pd.Series,
    winsorize_quantile: float = 0.01,
) -> pd.Series:
    standardized = pd.Series(np.nan, index=data.index, dtype=float)

    for _, index_values in data.groupby("date").groups.items():
        date_index = pd.Index(index_values)
        raw_series = pd.to_numeric(candidate_series.loc[date_index], errors="coerce")
        valid = raw_series.dropna()
        if valid.empty:
            continue

        lower = float(valid.quantile(winsorize_quantile))
        upper = float(valid.quantile(1.0 - winsorize_quantile))
        clipped = raw_series.clip(lower=lower, upper=upper)

        mean_value = float(clipped.mean())
        std_value = float(clipped.std(ddof=0))
        if math.isclose(std_value, 0.0, abs_tol=1e-12):
            continue
        standardized.loc[date_index] = (clipped - mean_value) / std_value

    return standardized.replace([np.inf, -np.inf], np.nan)


def evaluate_candidate(
    data: pd.DataFrame,
    candidate_name: str,
    candidate_series: pd.Series,
    n_groups: int,
    min_cross_section: int,
    rebalance_step: int | None = None,
    include_spread_metrics: bool = False,
) -> tuple[dict[str, float], dict[str, pd.DataFrame | dict[str, float]]]:
    prepared_series = standardize_candidate_cross_sectionally(data=data, candidate_series=candidate_series)
    working_df = data[["date", "instrument_id", "y"]].copy()
    working_df[candidate_name] = prepared_series

    summary_df, _, group_returns_df, average_group_returns_df = summarize_factor_diagnostics(
        data=working_df,
        factor_columns=[candidate_name],
        target_column="y",
        n_groups=n_groups,
        min_cross_section=min_cross_section,
        selector_scores=None,
        importance_scores=None,
        show_progress=False,
    )

    if summary_df.empty:
        return {}, {}

    summary_record = summary_df.iloc[0].to_dict()
    details: dict[str, pd.DataFrame | dict[str, float]] = {
        "group_returns_df": group_returns_df,
        "average_group_returns_df": average_group_returns_df,
    }

    if include_spread_metrics:
        spread_df, non_overlap_df, spread_metrics = build_spread_outputs(
            group_returns_df=group_returns_df,
            n_groups=n_groups,
            rebalance_step=rebalance_step or 1,
        )
        summary_record.update(spread_metrics)
        details["spread_df"] = spread_df
        details["non_overlap_df"] = non_overlap_df
        details["spread_metrics"] = spread_metrics

    return summary_record, details


def compute_composite_score(metrics: dict[str, float]) -> float:
    return float(
        metrics.get("pearson_ic_mean", float("nan")) * 1.0
        + metrics.get("spearman_ic_mean", float("nan")) * 0.5
        + metrics.get("long_short_spread", float("nan")) * 0.75
        + metrics.get("group_monotonic_spearman", float("nan")) * 0.02
        + max(metrics.get("pearson_ic_positive_ratio", float("nan")) - 0.5, 0.0) * 0.05
        + max(metrics.get("spearman_ic_positive_ratio", float("nan")) - 0.5, 0.0) * 0.03
    )


def lookup_metric(metrics: dict[str, float], key: str, prefixes: tuple[str, ...] = ("", "train_", "oos_")) -> float:
    for prefix in prefixes:
        candidate_key = f"{prefix}{key}"
        if candidate_key in metrics:
            return metrics[candidate_key]
    return float("nan")


def passes_train_filter(metrics: dict[str, float]) -> bool:
    return bool(
        metrics
        and lookup_metric(metrics, "coverage_ratio") >= 0.95
        and lookup_metric(metrics, "pearson_ic_mean") > 0.02
        and lookup_metric(metrics, "spearman_ic_mean") > 0.01
        and lookup_metric(metrics, "long_short_spread") > 0.003
        and lookup_metric(metrics, "group_monotonic_spearman") > 0.2
    )


def passes_oos_filter(metrics: dict[str, float]) -> bool:
    return bool(
        metrics
        and lookup_metric(metrics, "pearson_ic_mean") > 0.0
        and lookup_metric(metrics, "spearman_ic_mean") > 0.0
        and lookup_metric(metrics, "long_short_spread") > 0.0
        and lookup_metric(metrics, "group_monotonic_spearman") > 0.2
    )


def write_report(
    output_path: Path,
    settings: dict[str, object],
    seed_features: list[str],
    train_df: pd.DataFrame,
    oos_df: pd.DataFrame,
    final_df: pd.DataFrame,
) -> None:
    report_text = f"""# Heuristic Factor Search

## Search Settings

```json
{json.dumps(settings, ensure_ascii=False, indent=2)}
```

## Seed Features

{dataframe_to_markdown(pd.DataFrame({"seed_feature": seed_features}))}

## Train Survivors

{dataframe_to_markdown(train_df.copy())}

## OOS Survivors

{dataframe_to_markdown(oos_df.copy())}

## Final Shortlist

{dataframe_to_markdown(final_df.copy())}

## Notes

- 这里只是在训练期随机采样表达式，再用 OOS 做一次验收，不是证明这些候选已经可以实盘。
- 搜索空间被故意限制在现有非 Alpha191 技术/量价种子上，目的是降低数据窥探，不是追求“拼得越花越好”。
- 候选因子默认只做候选级别的横截面 winsorize 和 z-score；如果要正式进主流程，建议再补一轮单因子案例检查。
- 参考近年的 alpha mining 论文，这里额外强调 `family-aware` 选择，避免 shortlist 被同类公式占满。
- `survivor_ratio` 是搜索压力，不是真实统计显著性检验。
"""
    output_path.write_text(report_text, encoding="utf-8")


def select_diverse_shortlist(
    ranked_df: pd.DataFrame,
    top_k: int,
    max_per_family: int,
) -> pd.DataFrame:
    if ranked_df.empty:
        return ranked_df.copy()

    family_counts: dict[str, int] = {}
    selected_indices: list[int] = []

    for index, row in ranked_df.iterrows():
        family = str(row.get("family", "other"))
        current_count = family_counts.get(family, 0)
        if current_count >= max_per_family:
            continue
        selected_indices.append(index)
        family_counts[family] = current_count + 1
        if len(selected_indices) >= top_k:
            break

    if not selected_indices:
        return ranked_df.head(top_k).copy()
    return ranked_df.loc[selected_indices].copy()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.random_seed)

    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = PROJECT_ROOT / model_dir
    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    run_name = f"search_{args.target_horizon}d_{args.num_candidates}c_{args.random_seed}"
    output_dir = output_root / sanitize_name(run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df, target_column, dataset_summary = load_or_build_preprocessed_train_test(args)
    seed_features = load_seed_feature_pool(
        train_df=train_df,
        model_dir=model_dir,
        seed_top_k=args.seed_top_k,
        include_alpha_seeds=args.include_alpha_seeds,
        include_raw_market_seeds=args.include_raw_market_seeds,
    )
    if len(seed_features) < 2:
        raise ValueError("Seed feature pool is too small. Increase --seed-top-k or loosen filters.")

    generated_specs: list[dict[str, object]] = []
    seen_formulas: set[str] = set()

    while len(generated_specs) < args.num_candidates:
        spec = sample_candidate_spec(seed_features=seed_features, rng=rng)
        formula = str(spec["formula"])
        if formula in seen_formulas:
            continue
        seen_formulas.add(formula)
        spec = dict(spec)
        spec["candidate_id"] = f"hf_{len(generated_specs) + 1:04d}"
        generated_specs.append(spec)

    train_records: list[dict[str, object]] = []
    valid_specs: dict[str, dict[str, object]] = {}

    for spec in generated_specs:
        candidate_id = str(spec["candidate_id"])
        candidate_series = build_candidate_series(train_df, spec)
        metrics, _ = evaluate_candidate(
            data=train_df,
            candidate_name=candidate_id,
            candidate_series=candidate_series,
            n_groups=args.n_groups,
            min_cross_section=args.min_cross_section,
        )
        if not metrics:
            continue

        record = {
            **spec,
            **{f"train_{key}": value for key, value in metrics.items()},
        }
        record["family"] = infer_formula_family(str(spec["formula"]))
        record["train_score"] = compute_composite_score(metrics)
        train_records.append(record)
        valid_specs[candidate_id] = spec

    train_metrics_df = pd.DataFrame(train_records)
    if train_metrics_df.empty:
        raise ValueError("No valid heuristic candidates were produced.")

    train_metrics_df = train_metrics_df.sort_values("train_score", ascending=False).reset_index(drop=True)
    train_metrics_df.to_csv(output_dir / "candidate_metrics_train.csv", index=False)

    filtered_train_df = train_metrics_df[train_metrics_df.apply(lambda row: passes_train_filter(row.to_dict()), axis=1)].copy()
    survivor_count = max(1, int(math.ceil(len(train_metrics_df) * args.survivor_ratio)))
    survivor_train_df = filtered_train_df.head(survivor_count).copy()
    if survivor_train_df.empty:
        survivor_train_df = train_metrics_df.head(min(5, len(train_metrics_df))).copy()

    oos_records: list[dict[str, object]] = []
    for _, row in survivor_train_df.iterrows():
        candidate_id = str(row["candidate_id"])
        spec = valid_specs[candidate_id]
        candidate_series = build_candidate_series(test_df, spec)
        metrics, _ = evaluate_candidate(
            data=test_df,
            candidate_name=candidate_id,
            candidate_series=candidate_series,
            n_groups=args.n_groups,
            min_cross_section=args.min_cross_section,
            rebalance_step=args.target_horizon,
            include_spread_metrics=True,
        )
        if not metrics:
            continue

        oos_record = {
            **spec,
            **{column: row[column] for column in survivor_train_df.columns if column.startswith("train_") or column in {"train_score"}},
            **{f"oos_{key}": value for key, value in metrics.items()},
        }
        oos_record["family"] = infer_formula_family(str(spec["formula"]))
        oos_record["oos_score"] = compute_composite_score(metrics)
        oos_record["sign_consistent"] = bool(
            row.get("train_pearson_ic_mean", float("nan")) > 0 and metrics.get("pearson_ic_mean", float("nan")) > 0
        )
        oos_records.append(oos_record)

    oos_metrics_df = pd.DataFrame(oos_records)
    if oos_metrics_df.empty:
        raise ValueError("No train survivors produced valid OOS metrics.")

    oos_metrics_df = oos_metrics_df.sort_values(["oos_score", "oos_pearson_ic_mean"], ascending=False).reset_index(drop=True)
    oos_metrics_df.to_csv(output_dir / "candidate_metrics_oos.csv", index=False)

    final_df = oos_metrics_df[oos_metrics_df.apply(lambda row: passes_oos_filter({key[4:]: value for key, value in row.items() if key.startswith("oos_")}), axis=1)].copy()
    if final_df.empty:
        final_df = select_diverse_shortlist(oos_metrics_df, top_k=args.final_top_k, max_per_family=args.max_per_family)
    else:
        final_df = select_diverse_shortlist(final_df, top_k=args.final_top_k, max_per_family=args.max_per_family)

    display_columns = [
        "candidate_id",
        "formula",
        "family",
        "train_pearson_ic_mean",
        "train_long_short_spread",
        "train_group_monotonic_spearman",
        "oos_pearson_ic_mean",
        "oos_spearman_ic_mean",
        "oos_long_short_spread",
        "oos_group_monotonic_spearman",
        "oos_non_overlap_spread_mean",
        "oos_non_overlap_sharpe_horizon_adj",
        "oos_non_overlap_cumulative_return",
        "oos_score",
    ]
    final_df[display_columns].to_csv(output_dir / "final_shortlist.csv", index=False)

    settings = {
        "target_column": target_column,
        "dataset_summary": dataset_summary,
        "seed_feature_count": len(seed_features),
        "seed_features": seed_features,
        "num_candidates": args.num_candidates,
        "survivor_ratio": args.survivor_ratio,
        "final_top_k": args.final_top_k,
        "max_per_family": args.max_per_family,
        "random_seed": args.random_seed,
        "search_templates": list(BINARY_TEMPLATES.keys()),
        "unary_transforms": list(UNARY_TRANSFORMS.keys()),
        "post_transforms": list(POST_TRANSFORMS.keys()),
    }
    (output_dir / "search_settings.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")

    write_report(
        output_path=output_dir / "report.md",
        settings=settings,
        seed_features=seed_features,
        train_df=survivor_train_df[display_columns[:5]].copy(),
        oos_df=oos_metrics_df.head(args.final_top_k)[display_columns].copy(),
        final_df=final_df[display_columns].copy(),
    )

    print(f"[Info] Heuristic factor search finished: {output_dir}")
    print(f"[Info] Seed features used: {len(seed_features)}")
    print(f"[Info] Candidate formulas generated: {len(generated_specs)}")


if __name__ == "__main__":
    main()
