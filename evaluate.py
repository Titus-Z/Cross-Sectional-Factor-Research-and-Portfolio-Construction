"""评估脚本。

这个脚本负责读取预测结果与真实标签，然后输出一组常见的回归 / 量化评估指标，
并把图表保存到指定目录中。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loader import (
    PRICE_ADJUSTMENT_MODES,
    SUPPORTED_TARGET_HORIZONS,
    activate_target_horizon,
    load_daily_data,
)
from src.project_paths import resolve_project_path
from src.provenance import dumps_strict_json
from src.reporting import calculate_prediction_metrics, compute_daily_ic, compute_group_long_short
from src.runtime_config import DEFAULT_PRIMARY_DATA_PATH, DEFAULT_PRIMARY_TARGET_HORIZON


def parse_args() -> argparse.Namespace:
    """解析评估脚本的命令行参数。"""

    parser = argparse.ArgumentParser(description="Evaluate prediction results.")
    parser.add_argument(
        "--predictions-path",
        type=str,
        default="outputs/public_us300_release_v1/predictions.csv",
        help="模型预测结果文件路径。",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=DEFAULT_PRIMARY_DATA_PATH,
        help="原始完整数据路径，用于读取真实标签 y。",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/public_us300_release_v1",
        help="模型目录。若其中存在 feature_importance.csv，则会一并绘制特征重要性图。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluation",
        help="评估指标和图表的输出目录。",
    )
    parser.add_argument(
        "--n-groups",
        type=int,
        default=10,
        help="分组收益分析中的分组数量，默认 10 组。",
    )
    parser.add_argument(
        "--target-horizon",
        type=int,
        choices=list(SUPPORTED_TARGET_HORIZONS),
        default=DEFAULT_PRIMARY_TARGET_HORIZON,
        help="评估时使用哪个未来收益周期的真实标签，必须与训练时保持一致。",
    )
    parser.add_argument(
        "--price-adjustment-mode",
        choices=list(PRICE_ADJUSTMENT_MODES),
        default="vendor_adjusted",
        help="真实标签使用的价格复权口径，必须与训练时保持一致。",
    )
    return parser.parse_args()


def plot_scatter(merged_df: pd.DataFrame, output_path: Path) -> None:
    """绘制预测值 vs 实际值散点图。"""

    plt.figure(figsize=(8, 6))
    plt.scatter(
        merged_df["predicted_y"],
        merged_df["y"],
        alpha=0.35,
        s=18,
        edgecolors="none",
    )
    plt.xlabel("Predicted y")
    plt.ylabel("Actual y")
    plt.title("Predicted vs Actual")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_group_returns(group_returns: pd.DataFrame, output_path: Path) -> None:
    """绘制分组收益柱状图。"""

    plt.figure(figsize=(9, 5))
    plt.bar(group_returns["group"].astype(str), group_returns["average_actual_return"])
    plt.xlabel("Group")
    plt.ylabel("Average Actual Return")
    plt.title("Grouped Return Analysis")
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_feature_importance(feature_importance_path: Path, output_path: Path, top_n: int = 20) -> None:
    """绘制特征重要性条形图。

    如果模型目录中没有保存特征重要性文件，这个函数会被跳过。
    """

    importance_df = pd.read_csv(feature_importance_path)
    importance_df = importance_df.sort_values("importance", ascending=False).head(top_n)

    plt.figure(figsize=(10, 7))
    plt.barh(importance_df["feature"][::-1], importance_df["importance"][::-1])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_metrics(metrics: dict[str, Any], output_path: Path) -> None:
    """保存评估指标到 JSON 文件。"""

    output_path.write_text(dumps_strict_json(metrics), encoding="utf-8")


def main() -> None:
    """执行完整评估流程。"""

    args = parse_args()
    predictions_path = resolve_project_path(args.predictions_path)
    data_path = resolve_project_path(args.data_path)
    model_dir = resolve_project_path(args.model_dir)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_df = pd.read_csv(predictions_path)
    required_prediction_columns = {"date", "instrument_id", "predicted_y"}
    missing_prediction_columns = required_prediction_columns - set(predictions_df.columns)
    if missing_prediction_columns:
        raise ValueError(
            "Prediction file is missing required columns: "
            f"{sorted(missing_prediction_columns)}"
        )
    actual_df = load_daily_data(
        data_path,
        price_adjustment_mode=args.price_adjustment_mode,
    )
    actual_df, target_column = activate_target_horizon(actual_df, target_horizon=args.target_horizon)

    predictions_df["date"] = pd.to_datetime(predictions_df["date"])
    predictions_df["instrument_id"] = predictions_df["instrument_id"].astype(str)
    predictions_df["predicted_y"] = pd.to_numeric(
        predictions_df["predicted_y"], errors="coerce"
    )
    actual_df["date"] = pd.to_datetime(actual_df["date"])
    actual_df["instrument_id"] = actual_df["instrument_id"].astype(str)
    duplicate_prediction_keys = predictions_df.duplicated(
        subset=["date", "instrument_id"], keep=False
    )
    if duplicate_prediction_keys.any():
        raise ValueError(
            "Prediction file contains duplicate date/instrument_id keys; "
            "evaluation would double-count observations."
        )

    # 评估阶段只需要真实标签和主键列，因此这里先裁剪字段，避免无关列干扰。
    actual_df = actual_df[["date", "instrument_id", "y"]].rename(
        columns={"y": "reconstructed_y"}
    )
    if actual_df.duplicated(subset=["date", "instrument_id"], keep=False).any():
        raise ValueError(
            "Market data contains duplicate date/instrument_id keys; "
            "ground-truth labels are not uniquely defined."
        )

    # Standard MyQuant prediction files already contain the label used during
    # training. Keep it under a separate audit name. Merging two columns both
    # named `y` would create y_x/y_y and can also hide a target-horizon mismatch.
    if "y" in predictions_df.columns:
        predictions_df = predictions_df.rename(columns={"y": "saved_prediction_y"})
        predictions_df["saved_prediction_y"] = pd.to_numeric(
            predictions_df["saved_prediction_y"], errors="coerce"
        )

    merged_df = predictions_df.merge(
        actual_df,
        on=["date", "instrument_id"],
        how="left",
        validate="one_to_one",
    )

    # 如果真实标签缺失，说明预测结果和原始数据主键对不上。
    # 这里显式报错，方便定位数据问题。
    if merged_df["reconstructed_y"].isna().all():
        raise ValueError("All ground-truth labels are missing after merge. Please check date/instrument_id keys.")
    missing_ground_truth_rows = int(merged_df["reconstructed_y"].isna().sum())
    if missing_ground_truth_rows:
        raise ValueError(
            f"{missing_ground_truth_rows} prediction rows have no reconstructed ground-truth label. "
            "Refusing to evaluate a silently reduced sample."
        )

    label_audit = {
        "saved_label_available": bool("saved_prediction_y" in merged_df.columns),
        "saved_label_compared_rows": 0,
        "saved_label_mismatch_rows": 0,
        "saved_label_max_abs_error": float("nan"),
    }
    if "saved_prediction_y" in merged_df.columns:
        comparable = merged_df.dropna(subset=["saved_prediction_y", "reconstructed_y"]).copy()
        label_audit["saved_label_compared_rows"] = int(len(comparable))
        if not comparable.empty:
            absolute_error = (
                comparable["saved_prediction_y"] - comparable["reconstructed_y"]
            ).abs()
            mismatch_mask = ~np.isclose(
                comparable["saved_prediction_y"],
                comparable["reconstructed_y"],
                rtol=1e-7,
                atol=1e-10,
                equal_nan=False,
            )
            label_audit["saved_label_mismatch_rows"] = int(mismatch_mask.sum())
            label_audit["saved_label_max_abs_error"] = float(absolute_error.max())
            if mismatch_mask.any():
                raise ValueError(
                    "Saved prediction labels do not match labels reconstructed from the requested "
                    "data, target horizon, and price-adjustment mode. Refusing to report mixed-contract metrics."
                )

    merged_df = merged_df.rename(columns={"reconstructed_y": "y"})
    merged_df = merged_df.dropna(subset=["predicted_y", "y"]).reset_index(drop=True)
    if merged_df.empty:
        raise ValueError("No valid prediction/label rows remain after the evaluation merge.")

    spearman_ic_series = compute_daily_ic(merged_df, method="spearman")
    pearson_ic_series = compute_daily_ic(merged_df, method="pearson")

    group_returns_df, long_short_return = compute_group_long_short(
        merged_df,
        n_groups=args.n_groups,
    )

    metrics = {
        **calculate_prediction_metrics(merged_df[["date", "instrument_id", "y", "predicted_y"]]),
        # 兼容旧报告继续保留 long_short_return；新字段明确说明它是每天分组
        # Top-Bottom 前瞻收益差的跨日均值，不是可复利的组合累计收益。
        "long_short_return": long_short_return,
        "mean_daily_group_long_short_spread": long_short_return,
        "long_short_return_definition": (
            "mean across dates of top predicted quantile y minus bottom predicted quantile y"
        ),
        "evaluation_rows": int(len(merged_df)),
        "target_horizon": int(args.target_horizon),
        "price_adjustment_mode": args.price_adjustment_mode,
        "label_contract_audit": label_audit,
    }

    save_metrics(metrics, output_dir / "metrics.json")
    pearson_ic_series.to_csv(output_dir / "daily_pearson_ic.csv", header=True)
    spearman_ic_series.to_csv(output_dir / "daily_spearman_ic.csv", header=True)
    group_returns_df.to_csv(output_dir / "group_returns.csv", index=False)

    plot_scatter(merged_df, output_dir / "scatter_pred_vs_actual.png")
    plot_group_returns(group_returns_df, output_dir / "group_returns.png")

    feature_importance_path = model_dir / "feature_importance.csv"
    if feature_importance_path.exists():
        plot_feature_importance(
            feature_importance_path=feature_importance_path,
            output_path=output_dir / "feature_importance.png",
        )

    print("[Info] Evaluation finished.")
    print(f"[Info] Active target column: {target_column}")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value}")
    print(f"[Info] Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
