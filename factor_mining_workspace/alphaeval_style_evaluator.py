from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factor_mining_workspace.heuristic_factor_search import (
    CURATED_SEED_CANDIDATES,
    build_candidate_series,
    evaluate_candidate,
    infer_formula_family,
)
from factor_mining_workspace.single_factor_case_study import dataframe_to_markdown, load_or_build_preprocessed_train_test
from src.runtime_config import DEFAULT_OOS_START_DATE, DEFAULT_SAMPLE_START_DATE


DEFAULT_RUN_DIRS = [
    "factor_mining_workspace/heuristic_search_outputs/search_10d_300c_11",
    "factor_mining_workspace/heuristic_search_outputs/search_10d_200c_17",
    "factor_mining_workspace/heuristic_search_outputs/search_10d_200c_23",
    "factor_mining_workspace/heuristic_search_outputs/search_10d_120c_31",
]
OPERATOR_TOKENS = [
    "signed_sq",
    "tanh",
    "abs",
    "neg",
    "blend",
    "spread",
    "interaction",
    "ratio",
    "confirm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate formulaic alpha shortlists with AlphaEval-style dimensions.")
    parser.add_argument(
        "--run-dirs",
        nargs="*",
        default=DEFAULT_RUN_DIRS,
        help="要读取的搜索结果目录列表。",
    )
    parser.add_argument(
        "--output-dir",
        default="factor_mining_workspace/alphaeval_style_outputs",
        help="评估结果输出目录。",
    )
    parser.add_argument("--data-path", default="data/us_large_cap_300_daily.csv", help="原始数据路径。")
    parser.add_argument("--cache-dir", default=".cache", help="缓存目录。")
    parser.add_argument("--sample-start-date", default=DEFAULT_SAMPLE_START_DATE, help="样本起始日期。")
    parser.add_argument("--oos-start-date", default=DEFAULT_OOS_START_DATE, help="OOS 起始日期。")
    parser.add_argument("--target-horizon", type=int, default=10, help="目标周期。")
    parser.add_argument("--test-size", type=float, default=0.2, help="后段测试比例。")
    parser.add_argument("--n-groups", type=int, default=5, help="分组数量。")
    parser.add_argument("--min-cross-section", type=int, default=30, help="每个日期最少参与诊断的股票数。")
    parser.add_argument("--noise-scale", type=float, default=0.05, help="高斯扰动强度。")
    parser.add_argument("--dropout-rate", type=float, default=0.10, help="候选值缺失扰动比例。")
    parser.add_argument("--random-seed", type=int, default=42, help="扰动随机种子。")
    parser.add_argument("--disable-preprocessing-cache", action="store_true", help="关闭横截面预处理缓存。")
    return parser.parse_args()


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def find_shortlist_path(run_dir: Path) -> Path | None:
    preferred_files = [
        "strict_top1pct_oos_with_sharpe.csv",
        "strict_top1pct_oos_shortlist.csv",
        "final_shortlist.csv",
    ]
    for file_name in preferred_files:
        candidate_path = run_dir / file_name
        if candidate_path.exists():
            return candidate_path
    return None


def load_shortlist_candidates(run_dir: Path) -> pd.DataFrame:
    shortlist_path = find_shortlist_path(run_dir)
    if shortlist_path is None:
        return pd.DataFrame()

    shortlist_df = pd.read_csv(shortlist_path)
    if shortlist_df.empty or "candidate_id" not in shortlist_df.columns:
        return pd.DataFrame()

    metrics_candidates = []
    for file_name in ["candidate_metrics_oos.csv", "candidate_metrics_train.csv"]:
        candidate_path = run_dir / file_name
        if candidate_path.exists():
            metrics_candidates.append(pd.read_csv(candidate_path))
    if not metrics_candidates:
        return pd.DataFrame()

    metrics_df = metrics_candidates[0]
    for extra_df in metrics_candidates[1:]:
        for column in extra_df.columns:
            if column not in metrics_df.columns:
                metrics_df[column] = extra_df[column]

    merged_df = shortlist_df[["candidate_id"]].drop_duplicates().merge(metrics_df, on="candidate_id", how="left")
    merged_df["run_name"] = run_dir.name
    return merged_df.dropna(subset=["candidate_id", "formula"]).copy()


def perturb_candidate_series(
    series: pd.Series,
    rng: np.random.Generator,
    mode: str,
    noise_scale: float,
    dropout_rate: float,
) -> pd.Series:
    perturbed = pd.to_numeric(series, errors="coerce").copy()
    valid_mask = perturbed.notna()
    if not valid_mask.any():
        return perturbed

    if mode == "gaussian":
        std_value = float(perturbed[valid_mask].std(ddof=0))
        scale = noise_scale * (std_value if not math.isclose(std_value, 0.0, abs_tol=1e-12) else 1.0)
        noise = rng.normal(loc=0.0, scale=scale, size=int(valid_mask.sum()))
        perturbed.loc[valid_mask] = perturbed.loc[valid_mask].to_numpy(dtype=float) + noise
        return perturbed

    if mode == "dropout":
        draw = rng.random(int(valid_mask.sum()))
        valid_index = perturbed[valid_mask].index
        drop_index = valid_index[draw < dropout_rate]
        perturbed.loc[drop_index] = np.nan
        return perturbed

    raise ValueError(f"Unsupported perturbation mode: {mode}")


def split_oos_halves(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_dates = sorted(pd.to_datetime(data["date"]).dropna().unique())
    if len(unique_dates) < 2:
        return data.copy(), data.iloc[0:0].copy()
    midpoint = len(unique_dates) // 2
    first_dates = set(unique_dates[:midpoint])
    second_dates = set(unique_dates[midpoint:])
    first_half = data[pd.to_datetime(data["date"]).isin(first_dates)].copy()
    second_half = data[pd.to_datetime(data["date"]).isin(second_dates)].copy()
    return first_half, second_half


def safe_ratio(numerator: float, denominator: float) -> float:
    if pd.isna(numerator) or pd.isna(denominator):
        return float("nan")
    if math.isclose(float(denominator), 0.0, abs_tol=1e-12):
        return float("nan")
    return float(numerator / denominator)


def bounded_retention(candidate_value: float, baseline_value: float) -> float:
    ratio = safe_ratio(candidate_value, baseline_value)
    if pd.isna(ratio):
        return float("nan")
    return float(max(min(ratio, 2.0), -2.0))


def token_set(formula: str) -> set[str]:
    return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula))


def jaccard_similarity(left_tokens: set[str], right_tokens: set[str]) -> float:
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return float(len(left_tokens & right_tokens) / len(union))


def rank_to_unit(series: pd.Series, ascending: bool = True) -> pd.Series:
    valid = pd.to_numeric(series, errors="coerce")
    if valid.notna().sum() <= 1:
        return pd.Series(0.5, index=series.index, dtype=float)
    return valid.rank(method="average", ascending=ascending, pct=True)


def build_candidate_report_text(
    settings: dict[str, object],
    candidate_scores_df: pd.DataFrame,
    set_summary: dict[str, object],
) -> str:
    top_table = candidate_scores_df[
        [
            "run_name",
            "candidate_id",
            "formula",
            "family",
            "predictive_score",
            "stability_score",
            "robustness_score",
            "diversity_score",
            "interpretability_score",
            "alphaeval_style_score",
            "oos_pearson_ic_mean",
            "oos_non_overlap_sharpe_horizon_adj",
        ]
    ].copy()

    return f"""# AlphaEval-Style Evaluation

## Settings

```json
{json.dumps(settings, ensure_ascii=False, indent=2)}
```

## Candidate Scores

{dataframe_to_markdown(top_table)}

## Set Summary

```json
{json.dumps(set_summary, ensure_ascii=False, indent=2)}
```

## Interpretation

- `predictive_score` 对应 predictive power。
- `stability_score` 关注正 IC 占比和 OOS 前后半段一致性。
- `robustness_score` 来自高斯噪声和 dropout 扰动后的指标保持率。
- `diversity_score` 结合 family uniqueness 和公式 token 相似度。
- `interpretability_score` 用公式长度、操作符数量和原子特征数量做代理。
- 这是一版受 AlphaEval 抽象启发的本地实现，不是论文原作者官方实现。
"""


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.random_seed)

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = [resolve_path(run_dir) for run_dir in args.run_dirs]
    shortlist_frames = [load_shortlist_candidates(run_dir) for run_dir in run_dirs]
    shortlist_df = pd.concat([frame for frame in shortlist_frames if not frame.empty], ignore_index=True)
    if shortlist_df.empty:
        raise ValueError("No shortlist candidates were found in the provided run directories.")

    shortlist_df = shortlist_df.drop_duplicates(subset=["run_name", "candidate_id"]).reset_index(drop=True)

    train_df, test_df, target_column, dataset_summary = load_or_build_preprocessed_train_test(args)
    del train_df  # 当前评估只需要 OOS。

    first_half_df, second_half_df = split_oos_halves(test_df)
    family_counts = shortlist_df["formula"].map(infer_formula_family).value_counts().to_dict()
    formula_tokens = {row.candidate_id: token_set(str(row.formula)) for row in shortlist_df.itertuples(index=False)}

    candidate_records: list[dict[str, object]] = []

    for row in shortlist_df.itertuples(index=False):
        spec = row._asdict()
        candidate_id = str(spec["candidate_id"])
        formula = str(spec["formula"])
        family = infer_formula_family(formula)

        base_series = build_candidate_series(test_df, spec)
        baseline_metrics, _ = evaluate_candidate(
            data=test_df,
            candidate_name=candidate_id,
            candidate_series=base_series,
            n_groups=args.n_groups,
            min_cross_section=args.min_cross_section,
            rebalance_step=args.target_horizon,
            include_spread_metrics=True,
        )
        if not baseline_metrics:
            continue

        first_metrics, _ = evaluate_candidate(
            data=first_half_df,
            candidate_name=candidate_id,
            candidate_series=base_series.loc[first_half_df.index],
            n_groups=args.n_groups,
            min_cross_section=args.min_cross_section,
            rebalance_step=args.target_horizon,
            include_spread_metrics=True,
        )
        second_metrics, _ = evaluate_candidate(
            data=second_half_df,
            candidate_name=candidate_id,
            candidate_series=base_series.loc[second_half_df.index],
            n_groups=args.n_groups,
            min_cross_section=args.min_cross_section,
            rebalance_step=args.target_horizon,
            include_spread_metrics=True,
        )

        gaussian_series = perturb_candidate_series(
            series=base_series,
            rng=rng,
            mode="gaussian",
            noise_scale=args.noise_scale,
            dropout_rate=args.dropout_rate,
        )
        gaussian_metrics, _ = evaluate_candidate(
            data=test_df,
            candidate_name=f"{candidate_id}_gauss",
            candidate_series=gaussian_series,
            n_groups=args.n_groups,
            min_cross_section=args.min_cross_section,
            rebalance_step=args.target_horizon,
            include_spread_metrics=True,
        )

        dropout_series = perturb_candidate_series(
            series=base_series,
            rng=rng,
            mode="dropout",
            noise_scale=args.noise_scale,
            dropout_rate=args.dropout_rate,
        )
        dropout_metrics, _ = evaluate_candidate(
            data=test_df,
            candidate_name=f"{candidate_id}_drop",
            candidate_series=dropout_series,
            n_groups=args.n_groups,
            min_cross_section=args.min_cross_section,
            rebalance_step=args.target_horizon,
            include_spread_metrics=True,
        )

        tokens = formula_tokens[candidate_id]
        similarities = []
        for other_id, other_tokens in formula_tokens.items():
            if other_id == candidate_id:
                continue
            similarities.append(jaccard_similarity(tokens, other_tokens))
        max_similarity = float(max(similarities)) if similarities else 0.0

        seed_feature_hits = sum(1 for feature_name in CURATED_SEED_CANDIDATES if feature_name in formula)
        operator_count = sum(formula.count(operator_name + "(") for operator_name in OPERATOR_TOKENS)
        formula_length = len(formula)

        first_ic = first_metrics.get("pearson_ic_mean", float("nan")) if first_metrics else float("nan")
        second_ic = second_metrics.get("pearson_ic_mean", float("nan")) if second_metrics else float("nan")
        first_spread = first_metrics.get("long_short_spread", float("nan")) if first_metrics else float("nan")
        second_spread = second_metrics.get("long_short_spread", float("nan")) if second_metrics else float("nan")

        temporal_ic_consistency = 1.0 - min(
            abs(float(first_ic) - float(second_ic)) / max(abs(float(baseline_metrics["pearson_ic_mean"])), 1e-8),
            2.0,
        ) if not (pd.isna(first_ic) or pd.isna(second_ic)) else float("nan")
        temporal_spread_consistency = 1.0 - min(
            abs(float(first_spread) - float(second_spread)) / max(abs(float(baseline_metrics["long_short_spread"])), 1e-8),
            2.0,
        ) if not (pd.isna(first_spread) or pd.isna(second_spread)) else float("nan")

        gaussian_ic_retention = bounded_retention(
            gaussian_metrics.get("pearson_ic_mean", float("nan")) if gaussian_metrics else float("nan"),
            baseline_metrics["pearson_ic_mean"],
        )
        gaussian_spread_retention = bounded_retention(
            gaussian_metrics.get("long_short_spread", float("nan")) if gaussian_metrics else float("nan"),
            baseline_metrics["long_short_spread"],
        )
        dropout_ic_retention = bounded_retention(
            dropout_metrics.get("pearson_ic_mean", float("nan")) if dropout_metrics else float("nan"),
            baseline_metrics["pearson_ic_mean"],
        )
        dropout_spread_retention = bounded_retention(
            dropout_metrics.get("long_short_spread", float("nan")) if dropout_metrics else float("nan"),
            baseline_metrics["long_short_spread"],
        )

        candidate_records.append(
            {
                "run_name": str(spec["run_name"]),
                "candidate_id": candidate_id,
                "formula": formula,
                "family": family,
                "family_count": int(family_counts.get(family, 1)),
                "formula_length": formula_length,
                "operator_count": operator_count,
                "seed_feature_hits": seed_feature_hits,
                "max_formula_similarity": max_similarity,
                "oos_pearson_ic_mean": baseline_metrics["pearson_ic_mean"],
                "oos_spearman_ic_mean": baseline_metrics["spearman_ic_mean"],
                "oos_long_short_spread": baseline_metrics["long_short_spread"],
                "oos_group_monotonic_spearman": baseline_metrics["group_monotonic_spearman"],
                "oos_pearson_ic_positive_ratio": baseline_metrics["pearson_ic_positive_ratio"],
                "oos_spearman_ic_positive_ratio": baseline_metrics["spearman_ic_positive_ratio"],
                "oos_non_overlap_spread_mean": baseline_metrics["non_overlap_spread_mean"],
                "oos_non_overlap_sharpe_horizon_adj": baseline_metrics["non_overlap_sharpe_horizon_adj"],
                "oos_non_overlap_cumulative_return": baseline_metrics["non_overlap_cumulative_return"],
                "first_half_pearson_ic_mean": first_ic,
                "second_half_pearson_ic_mean": second_ic,
                "first_half_long_short_spread": first_spread,
                "second_half_long_short_spread": second_spread,
                "temporal_ic_consistency": temporal_ic_consistency,
                "temporal_spread_consistency": temporal_spread_consistency,
                "gaussian_ic_retention": gaussian_ic_retention,
                "gaussian_spread_retention": gaussian_spread_retention,
                "dropout_ic_retention": dropout_ic_retention,
                "dropout_spread_retention": dropout_spread_retention,
            }
        )

    candidate_df = pd.DataFrame(candidate_records)
    if candidate_df.empty:
        raise ValueError("No candidates produced valid AlphaEval-style metrics.")

    candidate_df["family_uniqueness"] = 1.0 / candidate_df["family_count"].clip(lower=1)
    candidate_df["formula_novelty"] = 1.0 - candidate_df["max_formula_similarity"].clip(lower=0.0, upper=1.0)

    candidate_df["predictive_score"] = (
        rank_to_unit(candidate_df["oos_pearson_ic_mean"])
        + rank_to_unit(candidate_df["oos_spearman_ic_mean"])
        + rank_to_unit(candidate_df["oos_long_short_spread"])
        + rank_to_unit(candidate_df["oos_non_overlap_sharpe_horizon_adj"])
    ) / 4.0

    candidate_df["stability_score"] = (
        rank_to_unit(candidate_df["oos_pearson_ic_positive_ratio"])
        + rank_to_unit(candidate_df["oos_spearman_ic_positive_ratio"])
        + rank_to_unit(candidate_df["temporal_ic_consistency"])
        + rank_to_unit(candidate_df["temporal_spread_consistency"])
    ) / 4.0

    candidate_df["robustness_score"] = (
        rank_to_unit(candidate_df["gaussian_ic_retention"])
        + rank_to_unit(candidate_df["gaussian_spread_retention"])
        + rank_to_unit(candidate_df["dropout_ic_retention"])
        + rank_to_unit(candidate_df["dropout_spread_retention"])
    ) / 4.0

    candidate_df["diversity_score"] = (
        rank_to_unit(candidate_df["family_uniqueness"])
        + rank_to_unit(candidate_df["formula_novelty"])
    ) / 2.0

    candidate_df["interpretability_score"] = (
        rank_to_unit(candidate_df["formula_length"], ascending=False)
        + rank_to_unit(candidate_df["operator_count"], ascending=False)
        + rank_to_unit(candidate_df["seed_feature_hits"], ascending=False)
    ) / 3.0

    candidate_df["alphaeval_style_score"] = (
        candidate_df["predictive_score"] * 0.35
        + candidate_df["stability_score"] * 0.20
        + candidate_df["robustness_score"] * 0.20
        + candidate_df["diversity_score"] * 0.15
        + candidate_df["interpretability_score"] * 0.10
    )

    candidate_df = candidate_df.sort_values(
        ["alphaeval_style_score", "oos_non_overlap_sharpe_horizon_adj", "oos_pearson_ic_mean"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    candidate_df.to_csv(output_dir / "candidate_alphaeval_style_scores.csv", index=False)

    family_distribution = candidate_df["family"].value_counts().to_dict()
    average_similarity = float(candidate_df["max_formula_similarity"].mean())
    set_summary = {
        "candidate_count": int(len(candidate_df)),
        "family_distribution": family_distribution,
        "average_max_formula_similarity": average_similarity,
        "top_candidate_id": str(candidate_df.iloc[0]["candidate_id"]),
        "top_formula": str(candidate_df.iloc[0]["formula"]),
        "top_alphaeval_style_score": float(candidate_df.iloc[0]["alphaeval_style_score"]),
        "target_column": target_column,
    }
    (output_dir / "set_summary.json").write_text(json.dumps(set_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    settings = {
        "run_dirs": [str(run_dir) for run_dir in run_dirs],
        "dataset_summary": dataset_summary,
        "target_column": target_column,
        "noise_scale": args.noise_scale,
        "dropout_rate": args.dropout_rate,
        "n_groups": args.n_groups,
        "min_cross_section": args.min_cross_section,
        "alphaeval_style_dimensions": [
            "predictive_power",
            "stability",
            "robustness",
            "diversity",
            "interpretability",
        ],
    }

    report_text = build_candidate_report_text(
        settings=settings,
        candidate_scores_df=candidate_df,
        set_summary=set_summary,
    )
    (output_dir / "report.md").write_text(report_text, encoding="utf-8")

    print(f"[Info] AlphaEval-style evaluation finished: {output_dir}")
    print(f"[Info] Evaluated candidates: {len(candidate_df)}")


if __name__ == "__main__":
    main()
