"""Build a PDF summary for factor mining runs.

这个脚本只做汇总，不重新挖因子、不重新训练模型。

它读取已经落盘的结果：

- warm-GP / contextual bandit / probabilistic grammar / PPO 的候选因子；
- 这些因子的 OOS 单因子表现；
- mined factor zoo 接入模型后的消融结果；
- 传统线性模型和非线性模型的重要性排序。

输出 PDF 的目的很明确：邮件里发给自己，可以在手机上直接看。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pdf_report import PdfSection, write_pdf_report
from src.project_paths import resolve_project_path
from src.reporting import calculate_prediction_metrics


DEFAULT_FACTOR_RUN_DIRS = [
    "factor_mining_workspace/auto_mining_outputs_oos202506/warm_gp_10d_g5_p80_c500_s7",
    "factor_mining_workspace/rl_mining_outputs/rl_bandit_10d_oos202506_e80_s31",
    "factor_mining_workspace/generative_mining_outputs/generative_grammar_10d_oos202506_n240_s62_derived_safe",
    "factor_mining_workspace/deep_rl_mining_outputs/ppo_formula_us300_10d_oos202506_v1",
]

DEFAULT_EXPERIMENT_OUTPUT_DIRS = [
    "outputs/experiments_us300_oos202506/10d_linear_models",
    "outputs/experiments_us300_oos202506/10d_all_models",
]

DEFAULT_EXPERIMENT_MODEL_DIRS = [
    "models/experiments_us300_oos202506/10d_linear_models",
    "models/experiments_us300_oos202506/10d_all_models",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize factor mining and model feature results into PDF.")
    parser.add_argument("--output-dir", default="outputs/reports/factor_mining_oos202506", help="报告输出目录。")
    parser.add_argument("--run-label", default="OOS 2025-06 factor mining summary", help="报告标题标签。")
    parser.add_argument("--factor-run-dirs", nargs="+", default=DEFAULT_FACTOR_RUN_DIRS, help="因子挖掘输出目录。")
    parser.add_argument(
        "--experiment-output-dirs",
        nargs="+",
        default=DEFAULT_EXPERIMENT_OUTPUT_DIRS,
        help="传统模型预测输出目录。",
    )
    parser.add_argument(
        "--experiment-model-dirs",
        nargs="+",
        default=DEFAULT_EXPERIMENT_MODEL_DIRS,
        help="传统模型特征重要性输出目录。",
    )
    parser.add_argument(
        "--ablation-run-dirs",
        nargs="+",
        default=[],
        help="mined factor model ablation 输出目录。",
    )
    return parser.parse_args()


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def first_existing(base_dir: Path, file_names: list[str]) -> Path | None:
    for file_name in file_names:
        path = base_dir / file_name
        if path.exists():
            return path
    return None


def infer_method_name(run_dir: Path) -> str:
    text = str(run_dir)
    if "auto_mining_outputs" in text or "warm_gp" in text:
        return "warm_gp"
    if "rl_mining_outputs" in text or "rl_bandit" in text:
        return "contextual_bandit"
    if "generative_mining_outputs" in text or "generative" in text:
        return "probabilistic_grammar"
    if "deep_rl_mining_outputs" in text or "ppo" in text:
        return "ppo_deep_rl"
    return run_dir.name


def select_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        column
        for column in [
            "candidate_id",
            "formula",
            "family",
            "oos_pearson_ic_mean",
            "oos_spearman_ic_mean",
            "oos_long_short_spread",
            "oos_score",
            "alphaeval_style_score",
            "rank_turnover",
            "top_retention",
            "max_signal_corr_abs",
        ]
        if column in df.columns
    ]
    return df[columns].copy() if columns else df.head(0).copy()


def sort_factor_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    for column in ["alphaeval_style_score", "oos_score", "oos_pearson_ic_mean"]:
        if column in df.columns:
            return df.sort_values(column, ascending=False)
    return df


def summarize_factor_run(run_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    method = infer_method_name(run_dir)
    candidate_path = first_existing(run_dir, ["candidate_formulas.csv", "oos_metrics.csv"])
    selected_path = first_existing(
        run_dir,
        [
            "validation_selected_factor_zoo.csv",
            "selected_factor_zoo.csv",
            "factor_zoo.csv",
            "final_shortlist.csv",
            "oos_metrics.csv",
        ],
    )

    candidate_df = read_csv_if_exists(candidate_path) if candidate_path else pd.DataFrame()
    selected_df = read_csv_if_exists(selected_path) if selected_path else pd.DataFrame()
    display_df = sort_factor_table(select_display_columns(selected_df if not selected_df.empty else candidate_df)).head(12)

    overview = {
        "method": method,
        "run_dir": str(run_dir),
        "exists": bool(run_dir.exists()),
        "candidate_count": int(len(candidate_df)),
        "selected_count": int(len(selected_df)),
        "candidate_file": str(candidate_path) if candidate_path else "",
        "selected_file": str(selected_path) if selected_path else "",
    }
    return overview, display_df


def summarize_experiment_output(output_dir: Path) -> dict[str, Any]:
    prediction_path = output_dir / "test_predictions_with_actual.csv"
    row: dict[str, Any] = {
        "experiment": output_dir.name,
        "output_dir": str(output_dir),
        "prediction_rows": 0,
        "pearson_ic_mean": None,
        "spearman_ic_mean": None,
        "long_short_return": None,
    }
    prediction_df = read_csv_if_exists(prediction_path)
    if prediction_df.empty:
        return row
    metrics = calculate_prediction_metrics(prediction_df[["date", "instrument_id", "y", "predicted_y"]])
    row.update(
        {
            "prediction_rows": int(len(prediction_df)),
            "pearson_ic_mean": metrics.get("pearson_ic_mean"),
            "spearman_ic_mean": metrics.get("spearman_ic_mean"),
            "long_short_return": metrics.get("long_short_return"),
            "rmse": metrics.get("rmse"),
            "mae": metrics.get("mae"),
        }
    )
    return row


def summarize_model_features(model_dir: Path) -> pd.DataFrame:
    importance_df = read_csv_if_exists(model_dir / "feature_importance.csv")
    if importance_df.empty:
        return pd.DataFrame()
    columns = [column for column in ["feature", "importance"] if column in importance_df.columns]
    top = importance_df[columns].head(15).copy()
    top.insert(0, "model_run", model_dir.name)
    return top


def summarize_ablation_run(ablation_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_df = read_csv_if_exists(ablation_dir / "metrics.csv")
    delta_df = read_csv_if_exists(ablation_dir / "metric_delta.csv")
    if not metrics_df.empty:
        metrics_df.insert(0, "ablation_run", ablation_dir.name)
    if not delta_df.empty:
        delta_df.insert(0, "ablation_run", ablation_dir.name)
    return metrics_df, delta_df


def write_markdown_report(
    output_path: Path,
    *,
    overview_df: pd.DataFrame,
    experiment_metrics_df: pd.DataFrame,
    feature_tables: list[pd.DataFrame],
    ablation_metrics_df: pd.DataFrame,
    ablation_delta_df: pd.DataFrame,
) -> None:
    feature_importance_df = pd.concat(feature_tables, ignore_index=True) if feature_tables else pd.DataFrame()
    text = f"""# Factor Mining Summary Report

## 1. Factor Mining Runs

{overview_df.to_markdown(index=False) if not overview_df.empty else "_No factor run overview._"}

## 2. Traditional Model OOS Metrics

{experiment_metrics_df.to_markdown(index=False) if not experiment_metrics_df.empty else "_No experiment metrics._"}

## 3. Traditional Linear / Nonlinear Top Features

{feature_importance_df.to_markdown(index=False) if not feature_importance_df.empty else "_No feature importance table._"}

## 4. Mined Factor Model Ablation Metrics

{ablation_metrics_df.to_markdown(index=False) if not ablation_metrics_df.empty else "_No ablation metrics._"}

## 5. Mined Factor Model Ablation Delta

{ablation_delta_df.to_markdown(index=False) if not ablation_delta_df.empty else "_No ablation delta._"}

## 6. Interpretation Rule

- 单因子 IC 只证明排序信号本身有信息。
- 模型消融改善才说明新因子对 baseline 有增量贡献。
- 如果 OOS 边界改变，所有因子选择和表现都必须重算。
"""
    output_path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    factor_overview_rows: list[dict[str, Any]] = []
    factor_sections: list[PdfSection] = []
    for run_dir_like in args.factor_run_dirs:
        run_dir = resolve_project_path(run_dir_like)
        overview, top_factors = summarize_factor_run(run_dir)
        factor_overview_rows.append(overview)
        factor_sections.append(
            PdfSection(
                f"{overview['method']} top mined factors",
                body=f"Run dir: {overview['run_dir']}",
                table=top_factors,
                max_table_rows=12,
            )
        )

    experiment_metrics_df = pd.DataFrame(
        [summarize_experiment_output(resolve_project_path(path_like)) for path_like in args.experiment_output_dirs]
    )
    feature_tables = [
        summarize_model_features(resolve_project_path(path_like)) for path_like in args.experiment_model_dirs
    ]
    feature_importance_df = (
        pd.concat([table for table in feature_tables if not table.empty], ignore_index=True)
        if feature_tables
        else pd.DataFrame()
    )

    ablation_metrics: list[pd.DataFrame] = []
    ablation_delta: list[pd.DataFrame] = []
    for path_like in args.ablation_run_dirs:
        metrics_df, delta_df = summarize_ablation_run(resolve_project_path(path_like))
        if not metrics_df.empty:
            ablation_metrics.append(metrics_df)
        if not delta_df.empty:
            ablation_delta.append(delta_df)
    ablation_metrics_df = pd.concat(ablation_metrics, ignore_index=True) if ablation_metrics else pd.DataFrame()
    ablation_delta_df = pd.concat(ablation_delta, ignore_index=True) if ablation_delta else pd.DataFrame()

    overview_df = pd.DataFrame(factor_overview_rows)
    write_markdown_report(
        output_dir / "factor_mining_summary_report.md",
        overview_df=overview_df,
        experiment_metrics_df=experiment_metrics_df,
        feature_tables=[feature_importance_df],
        ablation_metrics_df=ablation_metrics_df,
        ablation_delta_df=ablation_delta_df,
    )

    sections = [
        PdfSection(
            "Scope",
            body=(
                f"Run label: {args.run_label}\n"
                "This PDF reads saved outputs only. It does not rerun mining or model training.\n"
                "The key rule is strict: mined factors are useful for the resume only if they improve model-level OOS ablation."
            ),
        ),
        PdfSection("Factor Mining Run Overview", table=overview_df, max_table_rows=12),
        *factor_sections,
        PdfSection("Traditional Model OOS Metrics", table=experiment_metrics_df, max_table_rows=8),
        PdfSection("Linear / Nonlinear Top Model Features", table=feature_importance_df, max_table_rows=28),
        PdfSection("Mined Factor Model Ablation Metrics", table=ablation_metrics_df, max_table_rows=20),
        PdfSection("Mined Factor Model Ablation Delta", table=ablation_delta_df, max_table_rows=12),
        PdfSection(
            "Interpretation",
            body=(
                "Standalone factor IC means the factor can sort stocks by itself. "
                "Model ablation asks whether adding these mined factors improves the existing technical baseline. "
                "If the OOS start date is moved to 2025-06-01, older factor mining performance cannot be reused."
            ),
        ),
    ]
    pdf_path = write_pdf_report(
        output_dir / "factor_mining_summary_report.pdf",
        title="MyQuant Factor Mining Summary",
        subtitle=args.run_label,
        sections=sections,
    )

    print(f"[FactorMiningReport] markdown={output_dir / 'factor_mining_summary_report.md'}")
    print(f"[FactorMiningReport] pdf={pdf_path}")
    print(f"[FactorMiningReport] overview={json.dumps(factor_overview_rows, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
