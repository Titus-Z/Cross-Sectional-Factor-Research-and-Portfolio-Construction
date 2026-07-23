from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factor_mining_workspace.formula_language import is_forbidden_formula_field
from factor_mining_workspace.heuristic_factor_search import standardize_candidate_cross_sectionally
from factor_mining_workspace.mined_factor_model_ablation import SafeFormulaEvaluator, load_factor_zoo
from factor_mining_workspace.single_factor_case_study import load_or_build_preprocessed_train_test
from src.data_loader import PRICE_ADJUSTMENT_MODES
from src.runtime_config import (
    DEFAULT_OOS_START_DATE,
    DEFAULT_PRIMARY_DATA_PATH,
    DEFAULT_PRIMARY_TARGET_HORIZON,
    DEFAULT_SAMPLE_START_DATE,
)
from src.provenance import (
    build_data_fingerprint,
    dumps_strict_json,
    project_relative_path,
    sha256_file,
)
from src.time_series_pipeline import purge_training_label_overlap


DEFAULT_PPO_RUN_DIR = "factor_mining_workspace/deep_rl_mining_outputs/ppo_formula_us300_10d_v1"
SELECTION_CONTRACT_VERSION = "validation_factor_zoo_v3_scale_invariant_fields_bound"
VALIDATION_SELECTION_SOURCE = "validation_reward_only"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a PPO factor zoo using validation metrics only, without OOS ranking."
    )
    parser.add_argument("--ppo-run-dir", default=DEFAULT_PPO_RUN_DIR, help="PPO Deep RL 输出目录。")
    parser.add_argument("--candidate-file", default=None, help="可选候选文件；默认读取 ppo-run-dir/candidate_formulas.csv。")
    parser.add_argument("--output-file", default=None, help="输出文件；默认写入 ppo-run-dir/validation_selected_factor_zoo.csv。")
    parser.add_argument("--data-path", default=DEFAULT_PRIMARY_DATA_PATH, help="原始日频数据路径。")
    parser.add_argument("--cache-dir", default=".cache", help="特征和预处理缓存目录。")
    parser.add_argument("--sample-start-date", default=DEFAULT_SAMPLE_START_DATE, help="样本起始日期。")
    parser.add_argument("--oos-start-date", default=DEFAULT_OOS_START_DATE, help="OOS 起始日期。")
    parser.add_argument("--target-horizon", type=int, default=DEFAULT_PRIMARY_TARGET_HORIZON, help="目标收益周期。")
    parser.add_argument(
        "--price-adjustment-mode",
        choices=list(PRICE_ADJUSTMENT_MODES),
        default="vendor_adjusted",
        help="价格口径；必须与 PPO source config 完全一致。",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="没有显式 OOS 日期时的后段测试比例。")
    parser.add_argument("--validation-fraction", type=float, default=0.25, help="训练期尾部多少日期作为 validation。")
    parser.add_argument("--top-k", type=int, default=10, help="最多选择多少个因子。")
    parser.add_argument("--max-corr", type=float, default=0.80, help="候选之间最大允许 validation signal 相关性。")
    parser.add_argument("--max-scan", type=int, default=200, help="最多扫描 validation_reward 排名前多少的候选。")
    parser.add_argument("--disable-preprocessing-cache", action="store_true", help="关闭预处理缓存。")
    return parser.parse_args()


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def split_train_validation_by_time(
    train_df: pd.DataFrame,
    validation_fraction: float,
    purge_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    dates = sorted(pd.to_datetime(train_df["date"]).dropna().unique())
    if len(dates) < 30:
        raise ValueError("Not enough train dates for validation factor zoo selection.")
    validation_fraction = min(max(float(validation_fraction), 0.05), 0.50)
    split_index = int(math.floor(len(dates) * (1.0 - validation_fraction)))
    split_index = min(max(split_index, 1), len(dates) - 1)
    fit_dates = set(dates[:split_index])
    validation_dates = set(dates[split_index:])
    date_series = pd.to_datetime(train_df["date"])
    fit_df = train_df[date_series.isin(fit_dates)].copy()
    validation_df = train_df[date_series.isin(validation_dates)].copy()
    fit_df, purge_summary = purge_training_label_overlap(fit_df, target_horizon=purge_days)
    return fit_df, validation_df, purge_summary


def parse_fields(raw_fields: object) -> list[str]:
    if raw_fields is None or pd.isna(raw_fields):
        return []
    return [item.strip() for item in str(raw_fields).split(",") if item.strip()]


def has_forbidden_fields(raw_fields: object) -> bool:
    return any(is_forbidden_formula_field(field) for field in parse_fields(raw_fields))


def max_abs_corr(signal: pd.Series, selected_signals: list[pd.Series]) -> float:
    if not selected_signals:
        return 0.0
    candidate = pd.to_numeric(signal, errors="coerce")
    values: list[float] = []
    for selected_signal in selected_signals:
        aligned = pd.concat(
            [candidate.rename("candidate"), pd.to_numeric(selected_signal, errors="coerce").rename("selected")],
            axis=1,
        ).dropna()
        if len(aligned) < 50:
            continue
        corr_value = aligned["candidate"].corr(aligned["selected"], method="pearson")
        if pd.notna(corr_value):
            values.append(abs(float(corr_value)))
    return max(values) if values else 0.0


def build_validation_selected_factor_zoo(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, object]]:
    ppo_run_dir = resolve_path(args.ppo_run_dir)
    candidate_file = resolve_path(args.candidate_file) if args.candidate_file else ppo_run_dir / "candidate_formulas.csv"
    ppo_config_file = ppo_run_dir / "config.json"
    if not candidate_file.is_file():
        raise FileNotFoundError(f"PPO candidate file does not exist: {candidate_file}")
    if not ppo_config_file.is_file():
        raise FileNotFoundError(
            f"PPO config is required for validation-only provenance: {ppo_config_file}"
        )

    ppo_config = json.loads(ppo_config_file.read_text(encoding="utf-8"))
    if ppo_config.get("searcher") != "ppo_deep_rl_formula_mining":
        raise ValueError(f"Unexpected PPO searcher in config: {ppo_config.get('searcher')!r}")
    if int(ppo_config.get("target_horizon", -1)) != int(args.target_horizon):
        raise ValueError(
            "PPO candidate target horizon does not match selector target horizon: "
            f"{ppo_config.get('target_horizon')} != {args.target_horizon}"
        )
    if pd.Timestamp(ppo_config.get("sample_start_date")) != pd.Timestamp(args.sample_start_date):
        raise ValueError(
            "PPO candidate sample start does not match selector sample start: "
            f"{ppo_config.get('sample_start_date')} != {args.sample_start_date}"
        )
    if pd.Timestamp(ppo_config.get("oos_start_date")) != pd.Timestamp(args.oos_start_date):
        raise ValueError(
            "PPO candidate OOS boundary does not match selector boundary: "
            f"{ppo_config.get('oos_start_date')} != {args.oos_start_date}"
        )
    if ppo_config.get("price_adjustment_mode") != args.price_adjustment_mode:
        raise ValueError(
            "PPO candidate price-adjustment mode does not match selector mode: "
            f"{ppo_config.get('price_adjustment_mode')!r} != {args.price_adjustment_mode!r}"
        )

    data_path = resolve_path(args.data_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Strict factor selection requires a file-backed dataset: {data_path}")
    current_data_fingerprint = build_data_fingerprint(data_path, PROJECT_ROOT)
    source_data_fingerprint = ppo_config.get("data_fingerprint", {}) or {}
    if source_data_fingerprint.get("sha256") != current_data_fingerprint.get("sha256"):
        raise ValueError(
            "PPO config does not prove that candidates were mined from the current dataset. "
            "Rerun PPO mining with the current provenance schema."
        )

    candidates = pd.read_csv(candidate_file)

    required_columns = {
        "candidate_id",
        "formula",
        "validation_status",
        "validation_reward",
        "operator_count",
        "fields",
    }
    missing = required_columns - set(candidates.columns)
    if missing:
        raise ValueError(f"Candidate file is missing required columns: {sorted(missing)}")
    oos_columns = sorted(column for column in candidates.columns if str(column).lower().startswith("oos_"))
    if oos_columns:
        raise ValueError(
            "Validation selector refuses candidate tables containing OOS metrics. "
            f"Use candidate_formulas.csv, not oos_metrics.csv. Found: {oos_columns}"
        )

    filtered = candidates.copy()
    filtered["operator_count_numeric"] = pd.to_numeric(filtered["operator_count"], errors="coerce").fillna(0)
    filtered["validation_reward_numeric"] = pd.to_numeric(filtered["validation_reward"], errors="coerce")
    filtered = filtered[filtered["validation_status"].astype(str).str.lower() == "ok"].copy()
    filtered = filtered[filtered["operator_count_numeric"] > 0].copy()
    filtered = filtered[~filtered["fields"].apply(has_forbidden_fields)].copy()
    filtered = filtered.dropna(subset=["validation_reward_numeric", "formula", "candidate_id"]).copy()
    filtered = filtered.sort_values("validation_reward_numeric", ascending=False).head(int(args.max_scan)).copy()

    train_df, _, target_column, dataset_summary = load_or_build_preprocessed_train_test(args)
    canonical_formula_fields = sorted(set(dataset_summary.get("candidate_feature_columns", [])))
    if not canonical_formula_fields:
        raise ValueError("Preprocessed dataset did not preserve its canonical candidate-feature list.")

    # Validate the whole source table, not only the final selected rows.  A v3
    # mining run must have searched inside the same canonical feature language;
    # silently discarding raw-price candidates here would leave the provenance
    # claim stronger than the source artifact actually supports.
    load_factor_zoo(
        candidate_file,
        allowed_formula_fields=set(canonical_formula_fields),
    )
    _, validation_df, internal_purge_summary = split_train_validation_by_time(
        train_df,
        args.validation_fraction,
        purge_days=int(args.target_horizon),
    )
    evaluator = SafeFormulaEvaluator(validation_df)

    selected_rows: list[pd.Series] = []
    selected_signals: list[pd.Series] = []
    scan_records: list[dict[str, object]] = []

    iterator = progress(filtered.itertuples(index=False), total=len(filtered), desc="Selecting validation PPO zoo")
    for row in iterator:
        record = row._asdict()
        formula = str(record["formula"])
        candidate_id = str(record["candidate_id"])
        status = "pending"
        corr_to_selected = float("nan")

        try:
            raw_signal = evaluator.evaluate(formula)
            signal = standardize_candidate_cross_sectionally(validation_df, raw_signal)
            if signal.notna().sum() <= 0:
                status = "empty_signal"
            else:
                corr_to_selected = max_abs_corr(signal, selected_signals)
                if corr_to_selected <= float(args.max_corr):
                    selected = pd.Series(record).copy()
                    selected["validation_selection_rank"] = len(selected_rows) + 1
                    selected["validation_max_corr_to_selected"] = corr_to_selected
                    selected["selection_source"] = VALIDATION_SELECTION_SOURCE
                    selected_rows.append(selected)
                    selected_signals.append(signal)
                    status = "selected"
                else:
                    status = "high_corr_rejected"
        except Exception as exc:  # formula parser or missing field should not kill the whole selector
            status = f"failed: {exc}"

        scan_records.append(
            {
                "candidate_id": candidate_id,
                "formula": formula,
                "validation_reward": record.get("validation_reward"),
                "status": status,
                "validation_max_corr_to_selected": corr_to_selected,
            }
        )
        if len(selected_rows) >= int(args.top_k):
            break

    if not selected_rows:
        raise ValueError("No validation-selected PPO factors were selected.")

    selected_df = pd.DataFrame(selected_rows).drop(columns=["operator_count_numeric", "validation_reward_numeric"], errors="ignore")
    selected_df = selected_df.reset_index(drop=True)
    validation_end = pd.to_datetime(validation_df["date"]).max()
    if validation_end >= pd.Timestamp(args.oos_start_date):
        raise ValueError(
            "Validation factor selection overlaps OOS: "
            f"validation_end={validation_end.date()}, oos_start={pd.Timestamp(args.oos_start_date).date()}"
        )

    summary = {
        "selection_contract_version": SELECTION_CONTRACT_VERSION,
        "selection_source": VALIDATION_SELECTION_SOURCE,
        "selection_method": "validation_reward_rank_then_validation_signal_correlation_pruning",
        "ranking_columns_used": ["validation_reward"],
        "diversity_columns_used": ["validation_signal_pearson_correlation"],
        "oos_columns_used": [],
        "project_relative_candidate_file": project_relative_path(candidate_file, PROJECT_ROOT),
        "candidate_file_sha256": sha256_file(candidate_file),
        "source_searcher": ppo_config.get("searcher"),
        "project_relative_source_config_file": project_relative_path(ppo_config_file, PROJECT_ROOT),
        "source_config_sha256": sha256_file(ppo_config_file),
        "data_fingerprint": current_data_fingerprint,
        "candidate_file": project_relative_path(candidate_file, PROJECT_ROOT),
        "target_horizon": int(args.target_horizon),
        "sample_start_date": args.sample_start_date,
        "oos_start_date": args.oos_start_date,
        "target_column": target_column,
        "price_adjustment_mode": args.price_adjustment_mode,
        "allowed_formula_fields": canonical_formula_fields,
        "raw_candidate_count": int(len(candidates)),
        "filtered_candidate_count": int(len(filtered)),
        "selected_count": int(len(selected_df)),
        "max_corr": float(args.max_corr),
        "max_scan": int(args.max_scan),
        "validation_fraction": float(args.validation_fraction),
        "validation_start": str(pd.to_datetime(validation_df["date"]).min().date()),
        "validation_end": str(validation_end.date()),
        "validation_rows": int(len(validation_df)),
        "internal_validation_purge": internal_purge_summary,
        "dataset_summary": dataset_summary,
    }
    scan_df = pd.DataFrame(scan_records)
    return selected_df, {**summary, "scan_records": scan_df}


def main() -> None:
    start = time.perf_counter()
    args = parse_args()
    ppo_run_dir = resolve_path(args.ppo_run_dir)
    output_file = resolve_path(args.output_file) if args.output_file else ppo_run_dir / "validation_selected_factor_zoo.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    selected_df, summary = build_validation_selected_factor_zoo(args)
    selected_df.to_csv(output_file, index=False)
    scan_records = summary.pop("scan_records")
    scan_records.to_csv(output_file.with_name("validation_selection_scan.csv"), index=False)
    summary["runtime_seconds"] = time.perf_counter() - start
    summary["factor_zoo_file"] = project_relative_path(output_file, PROJECT_ROOT)
    summary["factor_zoo_sha256"] = sha256_file(output_file)
    output_file.with_name("validation_selected_factor_zoo_summary.json").write_text(
        dumps_strict_json(summary),
        encoding="utf-8",
    )

    print(f"[Done] validation-selected factor zoo written to: {output_file}", flush=True)
    print(selected_df[["candidate_id", "validation_reward", "validation_max_corr_to_selected", "formula"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
