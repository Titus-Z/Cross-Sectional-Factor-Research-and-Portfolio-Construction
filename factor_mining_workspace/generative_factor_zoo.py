from __future__ import annotations

import argparse
import ast
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factor_mining_workspace.formula_language import is_forbidden_formula_field
from factor_mining_workspace.single_factor_case_study import (
    dataframe_to_markdown,
    load_or_build_preprocessed_train_test,
)
from src.runtime_config import (
    DEFAULT_OOS_START_DATE,
    DEFAULT_PRIMARY_DATA_PATH,
    DEFAULT_PRIMARY_TARGET_HORIZON,
    DEFAULT_SAMPLE_START_DATE,
)


DEFAULT_INPUT_ROOT = "factor_mining_workspace/generative_mining_outputs"
DEFAULT_OUTPUT_DIR = "factor_mining_workspace/generative_factor_zoo_outputs/latest"

ALLOWED_FUNCTION_NAMES = {"abs", "tanh", "signed_sq", "rank", "sign", "neg"}
ALLOWED_AST_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.USub,
    ast.UAdd,
)

FAMILY_KEYWORDS = {
    "price_range": ("price_range", "range_position", "amt_range"),
    "volatility": ("return_std", "volatility", "std_", "boll_width", "xschannel_width"),
    "vwap_deviation": ("vwap", "vwap_gap"),
    "volume_price": ("volume", "vma_", "obv", "turnover", "amt_", "liquidity"),
    "momentum": ("momentum", "close_to_ma", "macd", "dma", "log_return"),
    "oscillator": ("rsi", "kdj", "wr_", "uos", "rsv"),
    "channel": ("boll_", "xschannel", "mike_"),
    "alpha191": ("alpha",),
    "size_value": ("market_cap", "shares_outstanding", "pe", "pb", "ps", "roe", "roa", "earnings_yield"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate and deduplicate generated factor mining results.")
    parser.add_argument("--input-root", default=DEFAULT_INPUT_ROOT, help="生成式因子挖掘运行目录根路径。")
    parser.add_argument(
        "--run-dirs",
        nargs="*",
        default=None,
        help="只聚合这些运行目录；不传则扫描 input-root 下所有目录。",
    )
    parser.add_argument(
        "--exclude-run-substrings",
        nargs="*",
        default=["s43"],
        help="排除目录名包含这些字符串的运行；默认排除已确认泄露的 s43。",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="聚合报告输出目录。")
    parser.add_argument("--data-path", default=DEFAULT_PRIMARY_DATA_PATH, help="原始日频数据路径。")
    parser.add_argument("--cache-dir", default=".cache", help="缓存目录。")
    parser.add_argument("--sample-start-date", default=DEFAULT_SAMPLE_START_DATE, help="样本起始日期。")
    parser.add_argument("--oos-start-date", default=DEFAULT_OOS_START_DATE, help="OOS 起始日期。")
    parser.add_argument("--target-horizon", type=int, default=DEFAULT_PRIMARY_TARGET_HORIZON, help="目标收益周期。")
    parser.add_argument("--test-size", type=float, default=0.2, help="未指定 OOS 日期时的后段测试比例。")
    parser.add_argument("--disable-preprocessing-cache", action="store_true", help="关闭横截面预处理缓存。")
    parser.add_argument("--min-operator-count", type=int, default=1, help="最少操作符数量，默认只保留派生公式。")
    parser.add_argument("--oos-min-rankic", type=float, default=0.02, help="OOS RankIC 最低要求。")
    parser.add_argument("--oos-min-long-short", type=float, default=0.0, help="OOS 多空收益最低要求。")
    parser.add_argument(
        "--dedup-correlation-threshold",
        type=float,
        default=0.90,
        help="OOS 信号 Spearman 相关性超过该阈值时视为近重复。",
    )
    parser.add_argument("--family-cap", type=int, default=3, help="每个因子家族最多保留多少个候选。")
    parser.add_argument("--final-top-k", type=int, default=30, help="报告最多展示多少个最终因子。")
    parser.add_argument(
        "--require-existing-oos-pass",
        action="store_true",
        help="要求候选在原始 oos_metrics.csv 中已经 passes_oos_filter=True。",
    )
    return parser.parse_args()


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _to_bool(series: pd.Series) -> pd.Series:
    """把 CSV 中可能出现的 True/False/1/0 统一转为布尔值。"""

    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def discover_oos_metric_files(args: argparse.Namespace) -> list[Path]:
    input_root = resolve_path(args.input_root)
    if args.run_dirs:
        run_dirs = [resolve_path(run_dir) for run_dir in args.run_dirs]
    else:
        run_dirs = sorted(path for path in input_root.iterdir() if path.is_dir()) if input_root.exists() else []

    metric_files: list[Path] = []
    excluded_tokens = tuple(str(token) for token in args.exclude_run_substrings)
    for run_dir in run_dirs:
        if any(token and token in run_dir.name for token in excluded_tokens):
            continue
        metric_path = run_dir / "oos_metrics.csv"
        if metric_path.exists():
            metric_files.append(metric_path)
    return metric_files


def load_candidate_pool(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    metric_files = discover_oos_metric_files(args)
    frames: list[pd.DataFrame] = []
    for metric_file in metric_files:
        frame = pd.read_csv(metric_file)
        if frame.empty:
            continue
        frame["run_dir"] = metric_file.parent.name
        frame["source_file"] = str(metric_file)
        frames.append(frame)

    if not frames:
        return pd.DataFrame(), {"metric_files": [str(path) for path in metric_files]}

    pool = pd.concat(frames, ignore_index=True)
    pool["global_candidate_id"] = pool["run_dir"].astype(str) + "::" + pool["candidate_id"].astype(str)
    pool["operator_count"] = pd.to_numeric(pool.get("operator_count"), errors="coerce").fillna(0).astype(int)
    pool["oos_spearman_ic_mean"] = pd.to_numeric(pool.get("oos_spearman_ic_mean"), errors="coerce")
    pool["oos_pearson_ic_mean"] = pd.to_numeric(pool.get("oos_pearson_ic_mean"), errors="coerce")
    pool["oos_long_short_spread"] = pd.to_numeric(pool.get("oos_long_short_spread"), errors="coerce")
    pool["formula_complexity"] = pd.to_numeric(pool.get("formula_complexity"), errors="coerce").fillna(999)
    if "passes_oos_filter" in pool.columns:
        pool["passes_oos_filter_bool"] = _to_bool(pool["passes_oos_filter"])
    else:
        pool["passes_oos_filter_bool"] = False

    summary = {
        "metric_files": [str(path) for path in metric_files],
        "loaded_rows": int(len(pool)),
        "loaded_unique_formulas": int(pool["formula"].astype(str).nunique()) if "formula" in pool.columns else 0,
    }
    return pool, summary


def infer_canonical_family(formula: str, existing_family: str | None = None) -> str:
    """把原始 family 压到更适合报告和 family-cap 的家族标签。

    生成器内部已有 family，但它偏向旧的 `volatility/channel/liquidity`。
    这里额外根据公式文本识别 `price_range`、`vwap_deviation`、`oscillator`
    等更容易解释给面试官和老师听的类别。
    """

    lowered = str(formula).lower()
    matched = [
        family
        for family, keywords in FAMILY_KEYWORDS.items()
        if any(keyword in lowered for keyword in keywords)
    ]
    if not matched and existing_family:
        raw_family = str(existing_family).split("+")[0].strip()
        return raw_family or "unknown"
    if not matched:
        return "unknown"
    if len(matched) >= 3:
        return "complex_mixed"
    if len(matched) == 2:
        # price_range 经常与 volatility 共现；单独保留 price_range 标签能避免它
        # 在报告中全部被泛化成 volatility。
        if "price_range" in matched:
            return "price_range"
        return "+".join(matched)
    return matched[0]


def filter_candidate_pool(pool: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    if pool.empty:
        return pool, {}

    filtered = pool.copy()
    before = len(filtered)
    filtered = filtered[filtered["operator_count"] >= int(args.min_operator_count)].copy()
    after_operator = len(filtered)
    if args.require_existing_oos_pass:
        filtered = filtered[filtered["passes_oos_filter_bool"]].copy()
    after_pass = len(filtered)
    filtered = filtered[
        (filtered["oos_spearman_ic_mean"] > float(args.oos_min_rankic))
        & (filtered["oos_long_short_spread"] > float(args.oos_min_long_short))
    ].copy()
    after_threshold = len(filtered)

    filtered["canonical_family"] = [
        infer_canonical_family(formula, family)
        for formula, family in zip(filtered["formula"].astype(str), filtered.get("family", pd.Series(index=filtered.index, dtype=object)))
    ]
    filtered = filtered.sort_values(
        ["oos_spearman_ic_mean", "oos_long_short_spread", "oos_pearson_ic_mean", "formula_complexity"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    exact_dedup = filtered.drop_duplicates("formula", keep="first").copy()

    summary = {
        "before_filter_rows": int(before),
        "after_operator_filter_rows": int(after_operator),
        "after_existing_pass_filter_rows": int(after_pass),
        "after_rankic_longshort_filter_rows": int(after_threshold),
        "after_exact_formula_dedup_rows": int(len(exact_dedup)),
    }
    return exact_dedup, summary


def _as_series(value: Any, index: pd.Index) -> pd.Series:
    if isinstance(value, pd.Series):
        return pd.to_numeric(value.reindex(index), errors="coerce")
    return pd.Series(value, index=index, dtype=float)


def cross_sectional_rank(data: pd.DataFrame, series: pd.Series) -> pd.Series:
    ranked = pd.Series(np.nan, index=data.index, dtype=float)
    if "date" not in data.columns:
        return pd.to_numeric(series, errors="coerce").rank(pct=True)
    for _, row_index in data.groupby("date").groups.items():
        date_index = pd.Index(row_index)
        ranked.loc[date_index] = pd.to_numeric(series.loc[date_index], errors="coerce").rank(pct=True)
    return ranked


def validate_formula_ast(parsed: ast.AST, data_columns: set[str]) -> set[str]:
    """校验公式字符串只包含本项目支持的安全表达式。

    聚合脚本需要重新计算候选因子的 OOS 信号相关性。这里不能直接用
    不受限制的 eval，否则报告脚本本身会变成安全风险。校验规则很窄：
    只允许加减乘除、数值常量、字段名和少数公式函数。
    """

    used_fields: set[str] = set()
    for node in ast.walk(parsed):
        if not isinstance(node, ALLOWED_AST_NODES):
            raise ValueError(f"Unsupported expression node: {type(node).__name__}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_FUNCTION_NAMES:
                raise ValueError("Unsupported function call in generated formula.")
        if isinstance(node, ast.Name):
            name = node.id
            if name in ALLOWED_FUNCTION_NAMES:
                continue
            if name not in data_columns:
                raise ValueError(f"Unknown formula field: {name}")
            if is_forbidden_formula_field(name):
                raise ValueError(f"Forbidden formula field: {name}")
            used_fields.add(name)
    return used_fields


def evaluate_formula_string(formula: str, data: pd.DataFrame) -> pd.Series:
    parsed = ast.parse(str(formula), mode="eval")
    used_fields = validate_formula_ast(parsed, set(data.columns))
    index = data.index
    local_env: dict[str, Any] = {
        field: pd.to_numeric(data[field], errors="coerce")
        for field in used_fields
    }

    def fn_abs(value: Any) -> pd.Series:
        return _as_series(value, index).abs()

    def fn_tanh(value: Any) -> pd.Series:
        return pd.Series(np.tanh(_as_series(value, index)), index=index, dtype=float)

    def fn_signed_sq(value: Any) -> pd.Series:
        series = _as_series(value, index)
        return pd.Series(np.sign(series) * np.square(series), index=index, dtype=float)

    def fn_rank(value: Any) -> pd.Series:
        return cross_sectional_rank(data, _as_series(value, index))

    def fn_sign(value: Any) -> pd.Series:
        return pd.Series(np.sign(_as_series(value, index)), index=index, dtype=float)

    def fn_neg(value: Any) -> pd.Series:
        return -_as_series(value, index)

    local_env.update(
        {
            "abs": fn_abs,
            "tanh": fn_tanh,
            "signed_sq": fn_signed_sq,
            "rank": fn_rank,
            "sign": fn_sign,
            "neg": fn_neg,
        }
    )
    result = eval(compile(parsed, "<generated_formula>", "eval"), {"__builtins__": {}}, local_env)
    return _as_series(result, index).replace([np.inf, -np.inf], np.nan)


def build_signal_frame(data: pd.DataFrame, candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    signal_map: dict[str, pd.Series] = {}
    error_records: list[dict[str, object]] = []
    for row in candidates.itertuples(index=False):
        candidate_id = str(row.global_candidate_id)
        formula = str(row.formula)
        try:
            raw_signal = evaluate_formula_string(formula, data)
            # 用日期内 rank 计算相关性，目标是判断两个因子是否每天给出近似相同的横截面排序。
            signal_map[candidate_id] = cross_sectional_rank(data, raw_signal)
        except Exception as exc:
            error_records.append(
                {
                    "global_candidate_id": candidate_id,
                    "formula": formula,
                    "error": str(exc),
                }
            )
    signal_frame = pd.DataFrame(signal_map, index=data.index)
    error_df = pd.DataFrame(error_records)
    return signal_frame, error_df


def deduplicate_by_signal_correlation(
    candidates: pd.DataFrame,
    signal_frame: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if candidates.empty or signal_frame.empty:
        return candidates.copy(), pd.DataFrame(), pd.DataFrame()

    available_ids = set(signal_frame.columns)
    working = candidates[candidates["global_candidate_id"].isin(available_ids)].copy()
    corr_df = signal_frame[working["global_candidate_id"].tolist()].corr(method="spearman").fillna(0.0)

    selected_rows: list[pd.Series] = []
    removed_records: list[dict[str, object]] = []
    for _, row in working.iterrows():
        candidate_id = str(row["global_candidate_id"])
        duplicate_of: str | None = None
        duplicate_corr = float("nan")
        for selected_row in selected_rows:
            selected_id = str(selected_row["global_candidate_id"])
            corr_value = float(corr_df.loc[candidate_id, selected_id])
            if abs(corr_value) > float(threshold):
                duplicate_of = selected_id
                duplicate_corr = corr_value
                break
        if duplicate_of is not None:
            removed_records.append(
                {
                    "formula_removed": row["formula"],
                    "candidate_removed": candidate_id,
                    "formula_kept": working.loc[working["global_candidate_id"] == duplicate_of, "formula"].iloc[0],
                    "candidate_kept": duplicate_of,
                    "correlation": duplicate_corr,
                    "reason": f"abs_spearman_corr_gt_{threshold}",
                }
            )
            continue
        selected_rows.append(row)

    selected_df = pd.DataFrame(selected_rows).reset_index(drop=True) if selected_rows else pd.DataFrame(columns=working.columns)
    removed_df = pd.DataFrame(removed_records)
    return selected_df, removed_df, corr_df


def apply_family_cap(candidates: pd.DataFrame, family_cap: int) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    family_counts: Counter[str] = Counter()
    rows: list[pd.Series] = []
    for _, row in candidates.iterrows():
        family = str(row.get("canonical_family", "unknown"))
        if family_counts[family] >= int(family_cap):
            continue
        family_counts[family] += 1
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True) if rows else pd.DataFrame(columns=candidates.columns)


def table_or_empty(data: pd.DataFrame, columns: list[str], top_k: int = 20) -> str:
    if data.empty:
        return "_No rows._"
    existing = [column for column in columns if column in data.columns]
    return dataframe_to_markdown(data[existing].head(top_k))


def build_report(
    args: argparse.Namespace,
    dataset_summary: dict[str, object],
    load_summary: dict[str, Any],
    filter_summary: dict[str, Any],
    signal_error_df: pd.DataFrame,
    corr_dedup_df: pd.DataFrame,
    duplicate_df: pd.DataFrame,
    diverse_df: pd.DataFrame,
    runtime_seconds: float,
) -> str:
    family_distribution = (
        diverse_df["canonical_family"].value_counts().rename_axis("canonical_family").reset_index(name="count")
        if not diverse_df.empty and "canonical_family" in diverse_df.columns
        else pd.DataFrame()
    )
    final_columns = [
        "global_candidate_id",
        "run_dir",
        "formula",
        "canonical_family",
        "operator_count",
        "oos_pearson_ic_mean",
        "oos_spearman_ic_mean",
        "oos_long_short_spread",
        "formula_complexity",
    ]
    return f"""# Generated Factor Zoo Report

## 1. Purpose

This report aggregates multiple probabilistic-grammar factor generation runs.

The goal is not to count every formula that looks good. The goal is to keep
leakage-safe, derived, OOS-positive candidates and remove obvious duplicates.

## 2. Config

```json
{json.dumps(vars(args), ensure_ascii=False, indent=2)}
```

Runtime seconds: `{runtime_seconds:.2f}`

## 3. Dataset

```json
{json.dumps(dataset_summary, ensure_ascii=False, indent=2, default=str)}
```

## 4. Candidate Counts

```json
{json.dumps({**load_summary, **filter_summary}, ensure_ascii=False, indent=2)}
```

Additional counts:

- Signal evaluation errors: `{len(signal_error_df)}`
- After correlation deduplication: `{len(corr_dedup_df)}`
- After family cap selection: `{len(diverse_df)}`
- Removed near-duplicates: `{len(duplicate_df)}`

## 5. Final Survivor Table

{table_or_empty(diverse_df, final_columns, top_k=int(args.final_top_k))}

## 6. Family Distribution

{dataframe_to_markdown(family_distribution)}

## 7. Removed Near-Duplicates

{table_or_empty(duplicate_df, ["candidate_removed", "candidate_kept", "correlation", "reason", "formula_removed", "formula_kept"], top_k=30)}

## 8. Leakage Control

The aggregation layer does not relax the generator's leakage filters. During
signal re-evaluation it also rejects formula names matching forbidden labels,
future fields, prediction fields, adjustment fields, or metadata columns.

Forbidden examples include:

```text
date, instrument_id, sector, y, y_*, target_*, label_*, future_*,
next_*, next_open, predicted_y, adjustment
```

## 9. Interpretation Limit

The OOS window is short. These factors can be described as preliminary OOS
survivors, not as proven trading signals. The next strict gate is model-layer
incremental testing: baseline features versus baseline plus generated factors.
"""


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pool_df, load_summary = load_candidate_pool(args)
    filtered_df, filter_summary = filter_candidate_pool(pool_df, args)

    # 加载 OOS 特征表用于重新计算候选信号相关性。
    # 这里只使用 test_df，不参与训练期筛选，因此不会反向污染模型训练。
    _, test_df, _, dataset_summary = load_or_build_preprocessed_train_test(args)
    signal_frame, signal_error_df = build_signal_frame(test_df, filtered_df)
    corr_dedup_df, duplicate_df, corr_df = deduplicate_by_signal_correlation(
        candidates=filtered_df,
        signal_frame=signal_frame,
        threshold=float(args.dedup_correlation_threshold),
    )
    diverse_df = apply_family_cap(corr_dedup_df, family_cap=int(args.family_cap))

    pool_df.to_csv(output_dir / "candidate_pool_raw.csv", index=False)
    filtered_df.to_csv(output_dir / "candidate_pool_filtered.csv", index=False)
    signal_error_df.to_csv(output_dir / "signal_evaluation_errors.csv", index=False)
    corr_dedup_df.to_csv(output_dir / "survivors_after_correlation_dedup.csv", index=False)
    duplicate_df.to_csv(output_dir / "duplicates_removed.csv", index=False)
    diverse_df.to_csv(output_dir / "survivors_diverse.csv", index=False)
    corr_df.to_csv(output_dir / "signal_correlation_matrix.csv")

    runtime_seconds = time.perf_counter() - start_time
    report_text = build_report(
        args=args,
        dataset_summary=dataset_summary,
        load_summary=load_summary,
        filter_summary=filter_summary,
        signal_error_df=signal_error_df,
        corr_dedup_df=corr_dedup_df,
        duplicate_df=duplicate_df,
        diverse_df=diverse_df,
        runtime_seconds=runtime_seconds,
    )
    (output_dir / "generated_factor_zoo_report.md").write_text(report_text, encoding="utf-8")

    print(f"[Done] Generated factor zoo written to: {output_dir}", flush=True)
    print(
        json.dumps(
            {
                "filtered_candidates": int(len(filtered_df)),
                "after_correlation_dedup": int(len(corr_dedup_df)),
                "after_family_cap": int(len(diverse_df)),
                "signal_errors": int(len(signal_error_df)),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
