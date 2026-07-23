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

from factor_mining_workspace.auto_alpha_reward import REWARD_MODES, compute_reward, finite_float


DEFAULT_FACTOR_RUN_DIRS = [
    "factor_mining_workspace/auto_mining_outputs_oos202506/warm_gp_10d_g5_p80_c500_s7",
    "factor_mining_workspace/rl_mining_outputs/rl_bandit_10d_oos202506_e80_s31",
    "factor_mining_workspace/generative_mining_outputs/generative_grammar_10d_oos202506_n240_s62_derived_safe",
    "factor_mining_workspace/deep_rl_mining_outputs/ppo_formula_us300_10d_oos202506_v1",
]


DISPLAY_COLUMNS = [
    "method",
    "candidate_id",
    "formula",
    "family",
    "selection_metric_prefix",
    "selection_predictive_ic_reward",
    "selection_incremental_proxy_reward",
    "oos_predictive_ic_reward",
    "oos_incremental_proxy_reward",
    "oos_pearson_ic_mean",
    "oos_spearman_ic_mean",
    "oos_long_short_spread",
    "oos_non_overlap_sharpe_horizon_adj",
    "formula_complexity",
    "operator_count",
    "financial_logic_score",
    "passes_oos_filter",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a unified Auto Alpha benchmark across factor miners.")
    parser.add_argument("--factor-run-dirs", nargs="+", default=DEFAULT_FACTOR_RUN_DIRS, help="各类因子挖掘输出目录。")
    parser.add_argument(
        "--output-dir",
        default="factor_mining_workspace/auto_alpha_benchmark_outputs/us300_10d_oos202506_v1",
        help="benchmark 输出目录。",
    )
    parser.add_argument("--top-k", type=int, default=10, help="每个方法、每套 reward 选取多少个候选做 OOS 审计汇总。")
    parser.add_argument(
        "--run-label",
        default="us300 + y_10d + OOS 2025-06 Auto Alpha Benchmark",
        help="报告标题。",
    )
    return parser.parse_args()


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[Warning] Failed to read {path}: {exc}", flush=True)
        return pd.DataFrame()


def first_existing(run_dir: Path, file_names: list[str]) -> Path | None:
    for file_name in file_names:
        path = run_dir / file_name
        if path.exists():
            return path
    return None


def infer_method_name(run_dir: Path) -> str:
    text = str(run_dir).lower()
    if "ppo_reward_predictive_ic" in text:
        return "ppo_deep_rl_predictive_ic"
    if "ppo_reward_incremental_proxy" in text:
        return "ppo_deep_rl_incremental_proxy"
    if "deep_rl_mining_outputs" in text or "ppo" in text:
        return "ppo_deep_rl"
    if "auto_mining_outputs" in text or "warm_gp" in text:
        return "warm_gp"
    if "rl_mining_outputs" in text or "rl_bandit" in text:
        return "contextual_bandit"
    if "generative_mining_outputs" in text or "grammar" in text:
        return "probabilistic_grammar"
    if "llm" in text or "alphaagent" in text:
        return "llm_proposal"
    return run_dir.name


def infer_selection_prefix(df: pd.DataFrame) -> str:
    """选择严格排序用的 in-sample 指标前缀。

    benchmark 排名不能优先用 OOS。PPO 有 `validation_*`，其他历史挖掘器多用
    `train_*`。如果两者都没有，才退回空前缀，并在输出中保留 prefix 方便审计。
    """

    if "validation_pearson_ic_mean" in df.columns:
        return "validation_"
    if "train_pearson_ic_mean" in df.columns:
        return "train_"
    return ""


def scalar_from_row(row: pd.Series, *columns: str, default: float = 0.0) -> float:
    for column in columns:
        if column in row.index:
            value = finite_float(row.get(column), default)
            if value != default or pd.notna(row.get(column)):
                return value
    return float(default)


def score_candidate_row(row: pd.Series, reward_mode: str, prefix: str) -> float:
    return compute_reward(
        row.to_dict(),
        reward_mode=reward_mode,
        prefix=prefix,
        financial_logic_score=scalar_from_row(row, "financial_logic_score", default=0.0),
        complexity=scalar_from_row(row, "formula_complexity", "complexity", default=1.0),
        max_signal_corr_abs=scalar_from_row(
            row,
            f"{prefix}max_signal_corr_abs",
            "validation_max_signal_corr_abs",
            "max_signal_corr_abs",
            "selected_max_corr_to_previous",
            default=0.0,
        ),
    )


def load_factor_run(run_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    method = infer_method_name(run_dir)
    metrics_path = first_existing(
        run_dir,
        [
            "oos_metrics.csv",
            "alphaeval_scores.csv",
            "candidate_metrics_oos.csv",
            "survivors_diverse.csv",
            "candidate_pool_filtered.csv",
        ],
    )
    overview = {
        "method": method,
        "run_dir": str(run_dir),
        "exists": run_dir.exists(),
        "metrics_file": str(metrics_path) if metrics_path else "",
        "candidate_count": 0,
        "status": "missing_metrics",
    }
    if metrics_path is None:
        return overview, pd.DataFrame()

    df = read_csv_if_exists(metrics_path)
    if df.empty:
        overview["status"] = "empty"
        return overview, pd.DataFrame()

    selection_prefix = infer_selection_prefix(df)
    df = df.copy()
    df.insert(0, "method", method)
    df.insert(1, "run_dir", str(run_dir))
    df.insert(2, "source_file", str(metrics_path))
    df["selection_metric_prefix"] = selection_prefix

    for reward_mode in REWARD_MODES:
        df[f"selection_{reward_mode}_reward"] = df.apply(
            lambda row: score_candidate_row(row, reward_mode=reward_mode, prefix=selection_prefix),
            axis=1,
        )
        df[f"oos_{reward_mode}_reward"] = df.apply(
            lambda row: score_candidate_row(row, reward_mode=reward_mode, prefix="oos_"),
            axis=1,
        )

    if "formula" in df.columns:
        overview["duplicate_formula_ratio"] = float(1.0 - df["formula"].astype(str).nunique() / max(len(df), 1))
    else:
        overview["duplicate_formula_ratio"] = float("nan")
    overview["candidate_count"] = int(len(df))
    overview["status"] = "ok"
    return overview, df


def summarize_method_reward(candidate_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if candidate_df.empty:
        return pd.DataFrame()

    for method, method_df in candidate_df.groupby("method"):
        for reward_mode in REWARD_MODES:
            selection_column = f"selection_{reward_mode}_reward"
            oos_reward_column = f"oos_{reward_mode}_reward"
            ranked = method_df.sort_values(selection_column, ascending=False).head(int(top_k)).copy()
            if ranked.empty:
                continue
            top = ranked.iloc[0]
            rows.append(
                {
                    "method": method,
                    "reward_mode": reward_mode,
                    "selected_top_k": int(len(ranked)),
                    "selection_score_mean": float(ranked[selection_column].mean()),
                    "oos_reward_mean": float(ranked[oos_reward_column].mean()),
                    "oos_pearson_ic_mean": float(pd.to_numeric(ranked.get("oos_pearson_ic_mean"), errors="coerce").mean()),
                    "oos_spearman_ic_mean": float(pd.to_numeric(ranked.get("oos_spearman_ic_mean"), errors="coerce").mean()),
                    "oos_long_short_spread_mean": float(pd.to_numeric(ranked.get("oos_long_short_spread"), errors="coerce").mean()),
                    "oos_non_overlap_sharpe_mean": float(
                        pd.to_numeric(ranked.get("oos_non_overlap_sharpe_horizon_adj"), errors="coerce").mean()
                    ),
                    "oos_pass_ratio": float(pd.to_numeric(ranked.get("passes_oos_filter"), errors="coerce").fillna(0).mean()),
                    "top_candidate_id": top.get("candidate_id", ""),
                    "top_formula": top.get("formula", ""),
                    "top_family": top.get("family", ""),
                    "top_selection_reward": finite_float(top.get(selection_column), 0.0),
                    "top_oos_reward": finite_float(top.get(oos_reward_column), 0.0),
                }
            )
    return pd.DataFrame(rows)


def summarize_overview(overview_df: pd.DataFrame, candidate_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, overview in overview_df.iterrows():
        method = str(overview.get("method", ""))
        subset = candidate_df[candidate_df["method"] == method] if not candidate_df.empty else pd.DataFrame()
        row = overview.to_dict()
        if not subset.empty:
            row.update(
                {
                    "oos_pass_count": int(pd.to_numeric(subset.get("passes_oos_filter"), errors="coerce").fillna(0).sum()),
                    "best_oos_predictive_ic_reward": float(subset["oos_predictive_ic_reward"].max()),
                    "best_oos_incremental_proxy_reward": float(subset["oos_incremental_proxy_reward"].max()),
                    "best_oos_pearson_ic_mean": float(pd.to_numeric(subset.get("oos_pearson_ic_mean"), errors="coerce").max()),
                    "best_oos_spearman_ic_mean": float(pd.to_numeric(subset.get("oos_spearman_ic_mean"), errors="coerce").max()),
                    "best_oos_long_short_spread": float(pd.to_numeric(subset.get("oos_long_short_spread"), errors="coerce").max()),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(
    output_path: Path,
    *,
    run_label: str,
    overview_df: pd.DataFrame,
    reward_summary_df: pd.DataFrame,
    top_predictive_df: pd.DataFrame,
    top_incremental_df: pd.DataFrame,
    top_k: int,
) -> None:
    text = f"""# Auto Alpha Benchmark Report

Run label: `{run_label}`

## 1. What This Benchmark Does

This report unifies existing factor-mining outputs under one comparable table.
It does not rerun factor generation and it does not retrain the main model.

The purpose is to compare search methods and reward definitions:

- `predictive_ic`: rewards standalone single-factor predictive power.
- `incremental_proxy`: rewards ranking, long-short spread, lower redundancy, lower complexity and portfolio-like stability.

Strict reading rule:

- Selection scores use `validation_*` if available, otherwise `train_*`.
- OOS columns are audit metrics, not selection inputs.
- A high single-factor score is not yet proof of model-layer incremental alpha.

## 2. Miner Overview

{overview_df.to_markdown(index=False) if not overview_df.empty else "_No miner overview._"}

## 3. Method x Reward Summary

Each row selects top `{top_k}` candidates within one method by one in-sample reward, then reports their OOS audit average.

{reward_summary_df.to_markdown(index=False) if not reward_summary_df.empty else "_No reward summary._"}

## 4. Top Candidates By Predictive Reward

{top_predictive_df.to_markdown(index=False) if not top_predictive_df.empty else "_No predictive candidates._"}

## 5. Top Candidates By Incremental Proxy Reward

{top_incremental_df.to_markdown(index=False) if not top_incremental_df.empty else "_No incremental proxy candidates._"}

## 6. Interpretation

If `predictive_ic` selects a factor but `incremental_proxy` does not, that factor is likely a strong standalone sorter but may be redundant, complex, or weak as a portfolio-oriented signal.

If both reward modes select similar candidates, the signal is more robust, but still needs model ablation:

```text
baseline
vs
baseline + selected mined factors
```

The next strict step is to train the same model twice under the two reward-selected factor zoo files, then compare OOS IC, RankIC, long-short and portfolio results.
"""
    output_path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    overview_rows: list[dict[str, Any]] = []
    candidate_frames: list[pd.DataFrame] = []
    for run_dir_raw in args.factor_run_dirs:
        overview, run_df = load_factor_run(resolve_path(run_dir_raw))
        overview_rows.append(overview)
        if not run_df.empty:
            candidate_frames.append(run_df)

    raw_overview_df = pd.DataFrame(overview_rows)
    candidate_df = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()
    overview_df = summarize_overview(raw_overview_df, candidate_df)
    reward_summary_df = summarize_method_reward(candidate_df, top_k=int(args.top_k))

    output_columns = [column for column in DISPLAY_COLUMNS if column in candidate_df.columns]
    top_predictive_df = (
        candidate_df.sort_values("selection_predictive_ic_reward", ascending=False)[output_columns].head(30).copy()
        if not candidate_df.empty
        else pd.DataFrame()
    )
    top_incremental_df = (
        candidate_df.sort_values("selection_incremental_proxy_reward", ascending=False)[output_columns].head(30).copy()
        if not candidate_df.empty
        else pd.DataFrame()
    )

    config = {
        "run_label": args.run_label,
        "factor_run_dirs": args.factor_run_dirs,
        "top_k": args.top_k,
        "reward_modes": list(REWARD_MODES),
    }
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    candidate_df.to_csv(output_dir / "candidate_universe.csv", index=False)
    overview_df.to_csv(output_dir / "miner_overview.csv", index=False)
    reward_summary_df.to_csv(output_dir / "reward_comparison.csv", index=False)
    top_predictive_df.to_csv(output_dir / "top_candidates_predictive_ic.csv", index=False)
    top_incremental_df.to_csv(output_dir / "top_candidates_incremental_proxy.csv", index=False)

    write_report(
        output_dir / "report.md",
        run_label=str(args.run_label),
        overview_df=overview_df,
        reward_summary_df=reward_summary_df,
        top_predictive_df=top_predictive_df,
        top_incremental_df=top_incremental_df,
        top_k=int(args.top_k),
    )

    print(f"[Done] Auto Alpha benchmark written to: {output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
