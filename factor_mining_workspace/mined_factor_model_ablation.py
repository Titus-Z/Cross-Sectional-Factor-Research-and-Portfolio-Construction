from __future__ import annotations

import argparse
import ast
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is in requirements, this is only a fallback.
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factor_mining_workspace.heuristic_factor_search import standardize_candidate_cross_sectionally
from factor_mining_workspace.formula_language import is_forbidden_formula_field
from factor_mining_workspace.single_factor_case_study import (
    dataframe_to_markdown,
    load_or_build_preprocessed_train_test,
    sanitize_name,
)
from src.model import build_model
from src.reporting import calculate_prediction_metrics
from src.runtime_config import (
    DEFAULT_FACTOR_DIAGNOSTICS_MODEL_DIR,
    DEFAULT_OOS_START_DATE,
    DEFAULT_PRIMARY_DATA_PATH,
    DEFAULT_PRIMARY_TARGET_HORIZON,
    DEFAULT_SAMPLE_START_DATE,
)


DEFAULT_FACTOR_ZOO_PATH = (
    "factor_mining_workspace/auto_mining_outputs/"
    "warm_gp_10d_g2_p30_c90_s7/factor_zoo.csv"
)
DEFAULT_OUTPUT_ROOT = "factor_mining_workspace/mined_factor_model_ablation_outputs"

IDENTIFIER_COLUMNS = {"instrument_id", "date", "sector"}
TARGET_COLUMNS = {"y", "y_1d", "y_5d", "y_10d"}
FUTURE_OR_LABEL_COLUMNS = {
    "next_open",
    # `adjustment` 容易让初学者误解成“复权信息可直接作为预测变量”。
    # 在这个增量实验里先排除它，让 baseline 更接近真实可解释的量价特征。
    "adjustment",
    # 财报日期元数据用于防泄露合并，不直接作为 alpha 或模型特征。
    # 真正进入模型的是 eps/pe/pb/roe/yoy/qoq 等数值基本面变量。
    "effective_date",
    "report_date",
    "filing_date",
    "accepted_date",
    "fiscal_period",
    "fiscal_year",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test whether mined factor zoo improves model OOS performance.")
    parser.add_argument("--factor-zoo-path", default=DEFAULT_FACTOR_ZOO_PATH, help="自动挖因子输出的 factor_zoo.csv。")
    parser.add_argument("--data-path", default=DEFAULT_PRIMARY_DATA_PATH, help="原始日频数据路径。")
    parser.add_argument("--model-dir", default=DEFAULT_FACTOR_DIAGNOSTICS_MODEL_DIR, help="保留这个参数是为了兼容缓存加载接口。")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_ROOT, help="增量实验输出目录。")
    parser.add_argument("--cache-dir", default=".cache", help="特征/预处理缓存目录。")
    parser.add_argument("--sample-start-date", default=DEFAULT_SAMPLE_START_DATE, help="样本起始日期。")
    parser.add_argument("--oos-start-date", default=DEFAULT_OOS_START_DATE, help="OOS 起始日期。")
    parser.add_argument("--target-horizon", type=int, default=DEFAULT_PRIMARY_TARGET_HORIZON, help="目标收益周期。")
    parser.add_argument("--test-size", type=float, default=0.2, help="没有显式 OOS 日期时的后段测试比例。")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ridge", "elastic_net"],
        help="用于增量检验的模型。默认只跑快而稳的线性模型。",
    )
    parser.add_argument("--random-seed", type=int, default=42, help="模型随机种子。")
    parser.add_argument(
        "--run-name",
        default=None,
        help="输出目录名；不传则根据 horizon、模型和 factor zoo 自动生成。",
    )
    parser.add_argument(
        "--disable-preprocessing-cache",
        action="store_true",
        help="关闭横截面预处理缓存。正式实验不建议关闭。",
    )
    return parser.parse_args()


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def show_progress(iterable, **kwargs):
    """在 tqdm 可用时显示进度条；不可用时退化成普通迭代。

    这个函数只是为了让脚本在极简环境也能运行，不把 tqdm 变成硬依赖。
    """

    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def cross_sectional_rank(data: pd.DataFrame, series: pd.Series) -> pd.Series:
    """按日期做横截面百分位排名。

    自动挖因子的公式语言里有 `rank(x)`。
    对量化横截面任务来说，这里的 rank 不是整列时间序列排名，
    而是“同一天不同股票之间谁更高”的横截面排名。
    """

    ranked = pd.Series(np.nan, index=data.index, dtype=float)
    for _, row_index in data.groupby("date").groups.items():
        date_index = pd.Index(row_index)
        ranked.loc[date_index] = pd.to_numeric(series.loc[date_index], errors="coerce").rank(pct=True)
    return ranked


def finite_series(values: Any, index: pd.Index) -> pd.Series:
    """把公式计算结果统一清洗成有限值 Series。"""

    if isinstance(values, pd.Series):
        result = pd.Series(values, index=index, dtype=float)
    else:
        result = pd.Series(float(values), index=index, dtype=float)
    return result.replace([np.inf, -np.inf], np.nan)


class SafeFormulaEvaluator:
    """只允许自动挖因子 DSL 所需的安全表达式。

    不能直接对公式字符串使用 `eval`，因为那会允许任意 Python 代码。
    这里用 `ast` 解析表达式，并且只放行加减乘除、少量函数和已有特征列。
    """

    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
        self.index = data.index

    def evaluate(self, formula: str) -> pd.Series:
        tree = ast.parse(str(formula), mode="eval")
        return finite_series(self._eval_node(tree.body), self.index)

    def _eval_node(self, node: ast.AST) -> pd.Series:
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise ValueError(f"Unsupported constant in formula: {node.value!r}")
            return finite_series(float(node.value), self.index)

        if isinstance(node, ast.Name):
            if node.id not in self.data.columns:
                raise KeyError(f"Formula references missing feature column: {node.id}")
            return finite_series(pd.to_numeric(self.data[node.id], errors="coerce"), self.index)

        if isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            if isinstance(node.op, ast.USub):
                return finite_series(-operand, self.index)
            if isinstance(node.op, ast.UAdd):
                return finite_series(operand, self.index)
            raise ValueError(f"Unsupported unary operator: {ast.dump(node.op)}")

        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            if isinstance(node.op, ast.Add):
                return finite_series(left + right, self.index)
            if isinstance(node.op, ast.Sub):
                return finite_series(left - right, self.index)
            if isinstance(node.op, ast.Mult):
                return finite_series(left * right, self.index)
            if isinstance(node.op, ast.Div):
                return finite_series(left / right.replace(0.0, np.nan), self.index)
            if isinstance(node.op, ast.Pow):
                return finite_series(np.power(left, right), self.index)
            raise ValueError(f"Unsupported binary operator: {ast.dump(node.op)}")

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple function calls are allowed in mined formulas.")
            if len(node.args) != 1 or node.keywords:
                raise ValueError("Formula functions must take exactly one positional argument.")

            function_name = node.func.id
            argument = self._eval_node(node.args[0])
            if function_name == "abs":
                return finite_series(argument.abs(), self.index)
            if function_name == "tanh":
                return finite_series(np.tanh(argument), self.index)
            if function_name == "sign":
                return finite_series(np.sign(argument), self.index)
            if function_name == "signed_sq":
                # signed_sq 保留原始方向，同时放大绝对值较大的信号：
                # 正数变成 x^2，负数变成 -x^2，常用于公式挖掘里的非线性变换。
                return finite_series(np.sign(argument) * np.square(argument), self.index)
            if function_name == "rank":
                return finite_series(cross_sectional_rank(self.data, argument), self.index)
            raise ValueError(f"Unsupported function in formula: {function_name}")

        raise ValueError(f"Unsupported formula AST node: {ast.dump(node)}")


def load_factor_zoo(
    path: Path,
    *,
    allowed_formula_fields: set[str] | None = None,
) -> pd.DataFrame:
    """Load a formula zoo and optionally enforce canonical feature fields.

    The materialized frame retains intermediate OHLC/VWAP columns for feature
    construction. Forbidden-label checks alone would let a hand-edited zoo refer
    to those scale-dependent intermediates, so strict callers pass the exact
    candidate list produced by the canonical feature generator.
    """

    factor_zoo = pd.read_csv(path)
    required_columns = {"candidate_id", "formula"}
    missing_columns = required_columns - set(factor_zoo.columns)
    if missing_columns:
        raise ValueError(f"factor_zoo.csv is missing required columns: {sorted(missing_columns)}")
    factor_zoo = factor_zoo.dropna(subset=["candidate_id", "formula"]).copy()
    if factor_zoo["candidate_id"].astype(str).duplicated().any():
        raise ValueError("factor_zoo.csv contains duplicate candidate_id values.")

    allowed_function_names = {"abs", "tanh", "sign", "signed_sq", "rank"}
    for row in factor_zoo[["candidate_id", "formula"]].itertuples(index=False):
        try:
            tree = ast.parse(str(row.formula), mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"Invalid formula syntax for candidate {row.candidate_id}: {exc}") from exc
        field_names = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and node.id not in allowed_function_names
        }
        forbidden_fields = sorted(field for field in field_names if is_forbidden_formula_field(field))
        if forbidden_fields:
            raise ValueError(
                f"Candidate {row.candidate_id} references forbidden fields: {forbidden_fields}"
            )
        if allowed_formula_fields is not None:
            out_of_contract_fields = sorted(field_names - set(allowed_formula_fields))
            if out_of_contract_fields:
                raise ValueError(
                    f"Candidate {row.candidate_id} references non-canonical feature fields: "
                    f"{out_of_contract_fields}"
                )
    return factor_zoo


def add_mined_factor_columns(data: pd.DataFrame, factor_zoo: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """把 factor zoo 的公式重新计算为模型可用特征列。

    注意：这里输入的 data 已经是严格时间切分后的 train 或 OOS。
    因此每个公式只在各自数据段内部计算，不会用 OOS 信息去构造训练特征。
    """

    enhanced_df = data.copy()
    evaluator = SafeFormulaEvaluator(enhanced_df)
    mined_columns: list[str] = []

    iterator = show_progress(
        factor_zoo.itertuples(index=False),
        total=len(factor_zoo),
        desc="Materializing mined factors",
        leave=False,
    )
    for row in iterator:
        candidate_id = str(row.candidate_id)
        formula = str(row.formula)
        column_name = f"mined_{sanitize_name(candidate_id)}"

        raw_series = evaluator.evaluate(formula)
        # 新公式可能把多个已标准化特征重新组合到一起。
        # 为了让它和其他横截面因子尺度一致，这里再做一次按日 winsorize + z-score。
        enhanced_df[column_name] = standardize_candidate_cross_sectionally(
            data=enhanced_df,
            candidate_series=raw_series,
        )
        mined_columns.append(column_name)

    return enhanced_df, mined_columns


def get_numeric_feature_columns(data: pd.DataFrame, mined_columns: list[str] | None = None) -> list[str]:
    """选择模型特征列，并显式排除标签、日期、标识符和未来价格列。"""

    mined_set = set(mined_columns or [])
    excluded_columns = IDENTIFIER_COLUMNS | TARGET_COLUMNS | FUTURE_OR_LABEL_COLUMNS
    feature_columns: list[str] = []
    for column in data.columns:
        if column in excluded_columns:
            continue
        if column.startswith("predicted_"):
            continue
        if column.startswith("mined_") and column not in mined_set:
            continue
        if pd.api.types.is_numeric_dtype(data[column]):
            feature_columns.append(column)
    return feature_columns


def build_feature_matrices(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """用训练集统计量填充缺失值，避免全数据 imputation 泄漏。

    Ridge / ElasticNet 不能直接处理 NaN，所以这里需要填充。
    关键点是：median 只能从训练集估计，然后应用到 OOS。
    """

    train_x = train_df[feature_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    test_x = test_df[feature_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    train_medians = train_x.median(axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return train_x.fillna(train_medians), test_x.fillna(train_medians)


def train_and_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    model_names: list[str],
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """训练多个模型，并用简单平均得到 ensemble 预测。

    这里不用 OOS 指标给模型加权，因为这个脚本的目标是做增量验证。
    如果用 OOS 表现决定权重，就会把测试集变成调参集。
    """

    train_x, test_x = build_feature_matrices(train_df, test_df, feature_columns)
    y_train = pd.to_numeric(train_df["y"], errors="coerce")
    valid_train_mask = y_train.notna()
    train_x = train_x.loc[valid_train_mask]
    y_train = y_train.loc[valid_train_mask]

    prediction_table = test_df[["date", "instrument_id", "y"]].copy()
    model_rows: list[dict[str, object]] = []
    model_predictions: list[np.ndarray] = []

    for model_name in show_progress(model_names, desc="Training models", leave=False):
        start_time = time.perf_counter()
        model = build_model(model_name=model_name, random_state=random_seed)
        model.fit(train_x, y_train)
        prediction = np.asarray(model.predict(test_x), dtype=float)
        elapsed_seconds = time.perf_counter() - start_time

        prediction_table[f"predicted_{model_name}"] = prediction
        model_predictions.append(prediction)
        model_rows.append(
            {
                "model": model_name,
                "feature_count": len(feature_columns),
                "train_rows": int(len(train_x)),
                "oos_rows": int(len(test_x)),
                "runtime_seconds": elapsed_seconds,
            }
        )

    if not model_predictions:
        raise ValueError("No model predictions were produced.")

    prediction_table["predicted_y"] = np.mean(np.column_stack(model_predictions), axis=1)
    return prediction_table, pd.DataFrame(model_rows)


def evaluate_prediction_frame(prediction_df: pd.DataFrame, experiment_name: str) -> dict[str, float | str]:
    metrics = calculate_prediction_metrics(prediction_df[["date", "instrument_id", "y", "predicted_y"]].copy())
    metrics["experiment"] = experiment_name
    return metrics


def write_report(
    output_path: Path,
    dataset_summary: dict[str, object],
    factor_zoo: pd.DataFrame,
    metrics_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    timing_df: pd.DataFrame,
    baseline_feature_count: int,
    mined_feature_columns: list[str],
) -> None:
    factor_view_columns = [
        column
        for column in [
            "candidate_id",
            "formula",
            "family",
            "oos_pearson_ic_mean",
            "oos_long_short_spread",
            "alphaeval_style_score",
            "max_signal_corr_abs",
            "rank_turnover",
            "top_retention",
        ]
        if column in factor_zoo.columns
    ]
    factor_view = factor_zoo[factor_view_columns].copy()

    report_text = f"""# Mined Factor Model Ablation Report

## 1. Purpose

这个实验回答一个更严格的问题：

```text
自动挖出来的 factor zoo，加入模型后是否能提升 OOS 预测效果？
```

单因子 IC 高，只能说明这个信号单独排序时有信息。
进入多变量模型后仍然提升，才说明它对当前 baseline 有增量贡献。

## 2. Dataset

```json
{json.dumps(dataset_summary, ensure_ascii=False, indent=2)}
```

## 3. Feature Sets

| feature set | feature count |
| --- | ---: |
| canonical feature baseline | {baseline_feature_count} |
| canonical feature baseline + mined factor zoo | {baseline_feature_count + len(mined_feature_columns)} |

Mined factor columns:

```text
{chr(10).join(mined_feature_columns)}
```

## 4. Factor Zoo Used

{dataframe_to_markdown(factor_view)}

## 5. OOS Model Metrics

{dataframe_to_markdown(metrics_df)}

## 6. Incremental Delta

`delta = baseline_plus_mined - baseline`

{dataframe_to_markdown(delta_df)}

## 7. Runtime

{dataframe_to_markdown(timing_df)}

## 8. Interpretation Rule

- 如果 `pearson_ic_mean`、`spearman_ic_mean`、`long_short_spread` 中至少两项改善，才记录为值得进一步检验的模型增量。
- 如果只改善单个指标，要谨慎，因为可能只是噪声或某种排序偏差。
- 如果模型层没有提升，但单因子层很强，通常意味着这些 mined factors 和 baseline 中已有技术指标高度重叠。
- 这里的 canonical baseline 是数据管线产生的原有数值特征集，通常包含技术指标和已启用的 Alpha191，不应简写成“仅技术指标”。

## 9. Interview Story

可以这样讲：

```text
我没有停留在“自动挖出几个 IC 高的公式”这一层，而是把 factor zoo 接回模型训练层，
做了 canonical feature baseline 和 baseline+mined factors 的 OOS 增量消融。
这个闭环可以检查新因子是否真的提供模型增量，而不是只在单因子报告里看起来好。
```
"""
    output_path.write_text(report_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    total_start = time.perf_counter()

    factor_zoo_path = resolve_path(args.factor_zoo_path)
    output_root = resolve_path(args.output_dir)
    run_name = args.run_name
    if not run_name:
        model_part = "_".join(args.models)
        run_name = f"zoo_{sanitize_name(factor_zoo_path.parent.name)}_{args.target_horizon}d_{model_part}"
    output_dir = output_root / sanitize_name(run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df, target_column, dataset_summary = load_or_build_preprocessed_train_test(args)
    dataset_summary = dict(dataset_summary)
    dataset_summary["target_column"] = target_column
    dataset_summary["factor_zoo_path"] = str(factor_zoo_path)
    dataset_summary["models"] = list(args.models)

    baseline_feature_columns = list(dataset_summary.get("candidate_feature_columns", []))
    if not baseline_feature_columns:
        raise ValueError("Preprocessed dataset did not preserve its canonical candidate-feature list.")
    factor_zoo = load_factor_zoo(
        factor_zoo_path,
        allowed_formula_fields=set(baseline_feature_columns),
    )
    print(f"[Info] Loaded factor zoo: {factor_zoo_path}", flush=True)
    print(f"[Info] Factor zoo size: {len(factor_zoo)}", flush=True)

    materialize_start = time.perf_counter()
    train_with_mined, mined_columns = add_mined_factor_columns(train_df, factor_zoo)
    test_with_mined, _ = add_mined_factor_columns(test_df, factor_zoo)
    materialize_seconds = time.perf_counter() - materialize_start

    mined_feature_columns = baseline_feature_columns + mined_columns

    baseline_start = time.perf_counter()
    baseline_predictions, baseline_timing = train_and_predict(
        train_df=train_df,
        test_df=test_df,
        feature_columns=baseline_feature_columns,
        model_names=args.models,
        random_seed=args.random_seed,
    )
    baseline_seconds = time.perf_counter() - baseline_start

    mined_start = time.perf_counter()
    mined_predictions, mined_timing = train_and_predict(
        train_df=train_with_mined,
        test_df=test_with_mined,
        feature_columns=mined_feature_columns,
        model_names=args.models,
        random_seed=args.random_seed,
    )
    mined_seconds = time.perf_counter() - mined_start

    baseline_metrics = evaluate_prediction_frame(baseline_predictions, "canonical_feature_baseline")
    mined_metrics = evaluate_prediction_frame(mined_predictions, "baseline_plus_mined_factor_zoo")
    metrics_df = pd.DataFrame([baseline_metrics, mined_metrics])

    metric_columns = [column for column in metrics_df.columns if column != "experiment"]
    delta_values: dict[str, float | str] = {"comparison": "baseline_plus_mined - canonical_feature_baseline"}
    for column in metric_columns:
        delta_values[column] = float(metrics_df.loc[1, column] - metrics_df.loc[0, column])
    delta_df = pd.DataFrame([delta_values])

    timing_df = pd.concat(
        [
            pd.DataFrame(
                [
                    {"stage": "materialize_mined_factors", "runtime_seconds": materialize_seconds},
                    {"stage": "train_canonical_feature_baseline_total", "runtime_seconds": baseline_seconds},
                    {"stage": "train_baseline_plus_mined_total", "runtime_seconds": mined_seconds},
                    {"stage": "total_script_runtime", "runtime_seconds": time.perf_counter() - total_start},
                ]
            ),
            baseline_timing.assign(stage=lambda frame: "baseline_model_" + frame["model"].astype(str))[
                ["stage", "runtime_seconds"]
            ],
            mined_timing.assign(stage=lambda frame: "mined_model_" + frame["model"].astype(str))[
                ["stage", "runtime_seconds"]
            ],
        ],
        ignore_index=True,
    )

    baseline_predictions.to_csv(output_dir / "predictions_canonical_feature_baseline.csv", index=False)
    mined_predictions.to_csv(output_dir / "predictions_baseline_plus_mined.csv", index=False)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    delta_df.to_csv(output_dir / "metric_delta.csv", index=False)
    timing_df.to_csv(output_dir / "runtime.csv", index=False)
    factor_zoo.to_csv(output_dir / "factor_zoo_used.csv", index=False)
    pd.DataFrame({"feature": mined_columns}).to_csv(output_dir / "mined_feature_columns.csv", index=False)

    write_report(
        output_path=output_dir / "report.md",
        dataset_summary=dataset_summary,
        factor_zoo=factor_zoo,
        metrics_df=metrics_df,
        delta_df=delta_df,
        timing_df=timing_df,
        baseline_feature_count=len(baseline_feature_columns),
        mined_feature_columns=mined_columns,
    )

    print(f"[Done] Report written to: {output_dir / 'report.md'}", flush=True)
    print(metrics_df.to_string(index=False), flush=True)
    print(delta_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
