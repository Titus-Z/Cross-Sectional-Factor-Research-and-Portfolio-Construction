from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factor_mining_workspace.formula_language import (
    BINARY_OPERATORS,
    DEFAULT_BLEND_WEIGHTS,
    UNARY_OPERATORS,
    FormulaNode,
    formula_similarity,
    node_from_legacy_spec,
)
from factor_mining_workspace.heuristic_factor_search import (
    CURATED_SEED_CANDIDATES,
    compute_composite_score,
    evaluate_candidate,
    load_seed_feature_pool,
    passes_oos_filter,
    passes_train_filter,
    standardize_candidate_cross_sectionally,
)
from factor_mining_workspace.single_factor_case_study import (
    dataframe_to_markdown,
    load_or_build_preprocessed_train_test,
    sanitize_name,
)
from src.runtime_config import (
    DEFAULT_FACTOR_DIAGNOSTICS_MODEL_DIR,
    DEFAULT_OOS_START_DATE,
    DEFAULT_PRIMARY_DATA_PATH,
    DEFAULT_PRIMARY_TARGET_HORIZON,
    DEFAULT_SAMPLE_START_DATE,
)
from src.progress import format_duration


DEFAULT_OUTPUT_ROOT = "factor_mining_workspace/auto_mining_outputs"
DEFAULT_HISTORY_RUN_DIR = "factor_mining_workspace/heuristic_search_outputs"
DEFAULT_DYNAMIC_WINDOW = 60
DEFAULT_FACTOR_ZOO_SIZE = 8
DEFAULT_MAX_FACTOR_CORR = 0.85
DEFAULT_TOP_RETENTION_FRACTION = 0.20

MUTATION_UNARY_OPERATORS = ("neg", "abs", "tanh", "signed_sq", "rank")
SEARCH_BINARY_OPERATORS = tuple(BINARY_OPERATORS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local formulaic alpha auto-mining.")
    subparsers = parser.add_subparsers(dest="command")

    search_parser = subparsers.add_parser("search", help="Run warm-start formula search and evaluation.")
    search_parser.add_argument("--searcher", choices=["warm_gp"], default="warm_gp", help="搜索器类型。")
    search_parser.add_argument("--data-path", default=DEFAULT_PRIMARY_DATA_PATH, help="原始数据路径。")
    search_parser.add_argument("--model-dir", default=DEFAULT_FACTOR_DIAGNOSTICS_MODEL_DIR, help="模型与特征选择产物目录。")
    search_parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_ROOT, help="自动挖因子输出根目录。")
    search_parser.add_argument("--history-run-dir", default=DEFAULT_HISTORY_RUN_DIR, help="旧 heuristic 搜索产物目录。")
    search_parser.add_argument("--cache-dir", default=".cache", help="缓存目录。")
    search_parser.add_argument("--sample-start-date", default=DEFAULT_SAMPLE_START_DATE, help="样本起始日期。")
    search_parser.add_argument("--oos-start-date", default=DEFAULT_OOS_START_DATE, help="OOS 起始日期。")
    search_parser.add_argument("--target-horizon", type=int, default=DEFAULT_PRIMARY_TARGET_HORIZON, help="目标周期。")
    search_parser.add_argument("--test-size", type=float, default=0.2, help="未指定 OOS 日期时的后段测试比例。")
    search_parser.add_argument("--n-groups", type=int, default=5, help="分组数量。")
    search_parser.add_argument("--min-cross-section", type=int, default=30, help="每个日期最少参与诊断的股票数。")
    search_parser.add_argument("--seed-top-k", type=int, default=16, help="从当前模型上下文取多少个 seed 特征。")
    search_parser.add_argument("--population-size", type=int, default=80, help="每代候选数量。")
    search_parser.add_argument("--generations", type=int, default=5, help="warm-GP 迭代代数。")
    search_parser.add_argument("--num-candidates", type=int, default=500, help="最多注册多少个候选公式。")
    search_parser.add_argument("--survivor-ratio", type=float, default=0.05, help="训练期保留比例。")
    search_parser.add_argument("--final-top-k", type=int, default=10, help="报告展示的最终候选数量。")
    search_parser.add_argument("--factor-zoo-size", type=int, default=DEFAULT_FACTOR_ZOO_SIZE, help="Factor zoo 最大候选数量。")
    search_parser.add_argument("--max-per-family", type=int, default=2, help="每个 family 最多进入多少个候选。")
    search_parser.add_argument("--max-factor-corr", type=float, default=DEFAULT_MAX_FACTOR_CORR, help="Factor zoo 内候选信号允许的最大绝对相关性。")
    search_parser.add_argument("--top-retention-fraction", type=float, default=DEFAULT_TOP_RETENTION_FRACTION, help="计算 top retention 时使用的头部比例。")
    search_parser.add_argument("--max-depth", type=int, default=4, help="公式 AST 最大深度。")
    search_parser.add_argument("--max-complexity", type=int, default=9, help="公式 AST 最大复杂度。")
    search_parser.add_argument("--max-fields", type=int, default=4, help="单个公式最多使用多少个原子字段。")
    search_parser.add_argument("--dynamic-window", type=int, default=DEFAULT_DYNAMIC_WINDOW, help="动态组合使用的滞后窗口长度。")
    search_parser.add_argument("--noise-scale", type=float, default=0.05, help="AlphaEval-style 高斯扰动强度。")
    search_parser.add_argument("--dropout-rate", type=float, default=0.10, help="AlphaEval-style 缺失扰动比例。")
    search_parser.add_argument("--random-seed", type=int, default=7, help="随机种子。")
    search_parser.add_argument(
        "--include-alpha-seeds",
        action="store_true",
        help="允许 canonical 价格尺度不变 Alpha 子集进入 seed pool；不会加载全部 Alpha191。",
    )
    search_parser.add_argument("--include-raw-market-seeds", action="store_true", help="允许原始量价列进入 seed pool。")
    search_parser.add_argument("--disable-preprocessing-cache", action="store_true", help="关闭横截面预处理缓存。")

    return parser.parse_args()


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def bounded_retention(candidate_value: float, baseline_value: float) -> float:
    if pd.isna(candidate_value) or pd.isna(baseline_value):
        return float("nan")
    if math.isclose(float(baseline_value), 0.0, abs_tol=1e-12):
        return float("nan")
    ratio = float(candidate_value) / float(baseline_value)
    return float(max(min(ratio, 2.0), -2.0))


def rank_to_unit(series: pd.Series, ascending: bool = True) -> pd.Series:
    valid = pd.to_numeric(series, errors="coerce")
    if valid.notna().sum() <= 1:
        return pd.Series(0.5, index=series.index, dtype=float)
    return valid.rank(method="average", ascending=ascending, pct=True).fillna(0.5)


def split_oos_halves(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_dates = sorted(pd.to_datetime(data["date"]).dropna().unique())
    if len(unique_dates) < 2:
        return data.copy(), data.iloc[0:0].copy()
    midpoint = len(unique_dates) // 2
    first_dates = set(unique_dates[:midpoint])
    second_dates = set(unique_dates[midpoint:])
    date_series = pd.to_datetime(data["date"])
    return data[date_series.isin(first_dates)].copy(), data[date_series.isin(second_dates)].copy()


def compute_signal_correlation_metadata(candidate_series_by_id: dict[str, pd.Series]) -> dict[str, dict[str, object]]:
    """计算候选因子之间的 OOS 信号相关性。

    公式相似度只能说明两个表达式长得像不像，不能说明它们在真实股票池上是否产生同样排序。
    所以这里额外用 OOS 标准化信号计算 Pearson 相关性，并在 factor zoo 选择时惩罚高相关候选。
    """

    if not candidate_series_by_id:
        return {}

    signal_frame = pd.DataFrame(candidate_series_by_id).replace([np.inf, -np.inf], np.nan)
    if signal_frame.shape[1] <= 1:
        only_id = next(iter(candidate_series_by_id))
        return {
            only_id: {
                "max_signal_corr_abs": 0.0,
                "signal_uniqueness": 1.0,
                "signal_corr_json": "{}",
            }
        }

    correlation_df = signal_frame.corr(method="pearson").fillna(0.0).clip(lower=-1.0, upper=1.0)
    metadata: dict[str, dict[str, object]] = {}
    for candidate_id in correlation_df.columns:
        peer_corr = correlation_df.loc[candidate_id].drop(labels=[candidate_id], errors="ignore")
        corr_map = {str(peer_id): float(value) for peer_id, value in peer_corr.items()}
        max_abs_corr = float(peer_corr.abs().max()) if not peer_corr.empty else 0.0
        metadata[str(candidate_id)] = {
            "max_signal_corr_abs": max_abs_corr,
            "signal_uniqueness": 1.0 - max_abs_corr,
            "signal_corr_json": json.dumps(corr_map, ensure_ascii=False, sort_keys=True),
        }
    return metadata


def compute_rank_turnover_proxy(
    data: pd.DataFrame,
    candidate_series: pd.Series,
    top_fraction: float,
) -> dict[str, float]:
    """用横截面排名变化估计因子换手压力。

    这不是完整组合换手，因为还没有仓位优化、调仓约束和交易成本。
    但它能快速回答一个关键问题：这个因子每天给出的股票排序是否剧烈变化。
    """

    top_fraction = float(min(max(top_fraction, 0.01), 0.50))
    working_df = pd.DataFrame(
        {
            "date": pd.to_datetime(data["date"]),
            "instrument_id": data["instrument_id"].astype(str),
            "signal": pd.to_numeric(candidate_series, errors="coerce"),
        },
        index=data.index,
    ).replace([np.inf, -np.inf], np.nan)
    working_df = working_df.dropna(subset=["date", "instrument_id", "signal"]).copy()
    if working_df.empty:
        return {"rank_turnover": float("nan"), "top_retention": float("nan")}

    working_df["rank"] = working_df.groupby("date")["signal"].rank(method="average", pct=True)
    sorted_df = working_df.sort_values(["instrument_id", "date"])
    rank_turnover = sorted_df.groupby("instrument_id")["rank"].diff().abs().mean()

    top_sets: list[set[str]] = []
    for _, date_df in working_df.groupby("date", sort=True):
        if date_df.empty:
            continue
        top_k = max(1, int(math.ceil(len(date_df) * top_fraction)))
        top_instruments = set(date_df.nlargest(top_k, "rank")["instrument_id"].astype(str))
        top_sets.append(top_instruments)

    retention_values: list[float] = []
    for previous_set, current_set in zip(top_sets, top_sets[1:]):
        denominator = max(len(previous_set), 1)
        retention_values.append(len(previous_set & current_set) / denominator)

    top_retention = float(np.mean(retention_values)) if retention_values else float("nan")
    return {
        "rank_turnover": float(rank_turnover) if pd.notna(rank_turnover) else float("nan"),
        "top_retention": top_retention,
    }


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
        std_value = float(perturbed.loc[valid_mask].std(ddof=0))
        scale = noise_scale * (std_value if not math.isclose(std_value, 0.0, abs_tol=1e-12) else 1.0)
        perturbed.loc[valid_mask] = perturbed.loc[valid_mask].to_numpy(dtype=float) + rng.normal(
            loc=0.0,
            scale=scale,
            size=int(valid_mask.sum()),
        )
        return perturbed

    if mode == "dropout":
        valid_index = perturbed.loc[valid_mask].index
        draw = rng.random(int(valid_mask.sum()))
        perturbed.loc[valid_index[draw < dropout_rate]] = np.nan
        return perturbed

    raise ValueError(f"Unsupported perturbation mode: {mode}")


def node_metadata(node: FormulaNode) -> dict[str, object]:
    return {
        "formula": node.to_formula(),
        "family": node.infer_family(),
        "hypothesis": node.hypothesis(),
        "fields": ",".join(sorted(node.fields)),
        "field_count": len(node.fields),
        "operators": ",".join(node.operators),
        "operator_count": len(node.operators),
        "formula_depth": node.depth,
        "formula_complexity": node.complexity,
        "financial_logic_score": node.financial_logic_score(),
    }


def load_historical_warm_nodes(history_run_dir: Path, available_columns: set[str], args: argparse.Namespace) -> list[FormulaNode]:
    if not history_run_dir.exists():
        return []

    candidate_frames: list[pd.DataFrame] = []
    preferred_names = [
        "strict_top1pct_oos_with_sharpe.csv",
        "strict_top1pct_oos_shortlist.csv",
        "final_shortlist.csv",
        "candidate_metrics_oos.csv",
    ]
    for run_dir in sorted(path for path in history_run_dir.iterdir() if path.is_dir()):
        for file_name in preferred_names:
            candidate_path = run_dir / file_name
            if not candidate_path.exists():
                continue
            try:
                frame = pd.read_csv(candidate_path)
            except Exception:
                continue
            if frame.empty:
                continue
            frame["history_run_name"] = run_dir.name
            candidate_frames.append(frame)
            break

    if not candidate_frames:
        return []

    history_df = pd.concat(candidate_frames, ignore_index=True)
    score_columns = [
        "oos_non_overlap_sharpe_horizon_adj",
        "oos_score",
        "alphaeval_style_score",
        "oos_pearson_ic_mean",
    ]
    existing_score_columns = [column for column in score_columns if column in history_df.columns]
    if existing_score_columns:
        history_df = history_df.sort_values(existing_score_columns, ascending=False)

    nodes: list[FormulaNode] = []
    seen_formulas: set[str] = set()
    for record in history_df.to_dict("records"):
        node = node_from_legacy_spec(record)
        if node is None:
            continue
        if not node.is_legal(available_columns, args.max_depth, args.max_complexity, args.max_fields):
            continue
        formula = node.to_formula()
        if formula in seen_formulas:
            continue
        seen_formulas.add(formula)
        nodes.append(node)
        if len(nodes) >= max(args.population_size, 20):
            break
    return nodes


def build_seed_nodes(train_df: pd.DataFrame, model_dir: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, list[FormulaNode]]:
    seed_features = load_seed_feature_pool(
        train_df=train_df,
        model_dir=model_dir,
        seed_top_k=args.seed_top_k,
        include_alpha_seeds=args.include_alpha_seeds,
        include_raw_market_seeds=args.include_raw_market_seeds,
    )

    available_columns = set(train_df.columns)
    seed_nodes: list[FormulaNode] = []
    seed_records: list[dict[str, object]] = []
    seen_formulas: set[str] = set()

    def add_seed(node: FormulaNode, source: str) -> None:
        formula = node.to_formula()
        if formula in seen_formulas:
            return
        if not node.is_legal(available_columns, args.max_depth, args.max_complexity, args.max_fields):
            return
        seen_formulas.add(formula)
        seed_nodes.append(node)
        seed_records.append({"seed_source": source, **node_metadata(node)})

    for feature_name in seed_features:
        add_seed(FormulaNode.column(feature_name), "current_model_seed")

    for feature_name in CURATED_SEED_CANDIDATES:
        if feature_name in available_columns:
            add_seed(FormulaNode.column(feature_name), "curated_formula_library")

    history_nodes = load_historical_warm_nodes(resolve_path(args.history_run_dir), available_columns, args)
    for node in history_nodes:
        add_seed(node, "historical_shortlist")

    return pd.DataFrame(seed_records), seed_nodes


def sample_mutation(node: FormulaNode, seed_nodes: list[FormulaNode], rng: random.Random) -> FormulaNode:
    action = rng.random()
    if action < 0.40:
        operator = rng.choice(MUTATION_UNARY_OPERATORS)
        return FormulaNode.unary(operator, node)

    partner = rng.choice(seed_nodes)
    operator = rng.choice(SEARCH_BINARY_OPERATORS)
    weight = rng.choice(DEFAULT_BLEND_WEIGHTS)
    if action < 0.70:
        return FormulaNode.binary(operator, node, partner, weight=weight)
    return FormulaNode.binary(operator, partner, node, weight=weight)


def sample_crossover(left: FormulaNode, right: FormulaNode, rng: random.Random) -> FormulaNode:
    operator = rng.choice(SEARCH_BINARY_OPERATORS)
    weight = rng.choice(DEFAULT_BLEND_WEIGHTS)
    return FormulaNode.binary(operator, left, right, weight=weight)


def build_initial_population(seed_nodes: list[FormulaNode], available_columns: set[str], args: argparse.Namespace, rng: random.Random) -> list[FormulaNode]:
    if not seed_nodes:
        raise ValueError("No seed nodes were found. Check model outputs or loosen seed filters.")

    population: list[FormulaNode] = []
    seen_formulas: set[str] = set()

    def maybe_add(node: FormulaNode) -> None:
        formula = node.to_formula()
        if formula in seen_formulas:
            return
        if not node.is_legal(available_columns, args.max_depth, args.max_complexity, args.max_fields):
            return
        seen_formulas.add(formula)
        population.append(node)

    for node in seed_nodes:
        maybe_add(node)
        if len(population) >= args.population_size:
            return population

    attempts = 0
    while len(population) < args.population_size and attempts < args.population_size * 30:
        attempts += 1
        maybe_add(sample_mutation(rng.choice(seed_nodes), seed_nodes, rng))

    return population


def select_next_population(train_records: dict[str, dict[str, object]], nodes_by_id: dict[str, FormulaNode], population_size: int) -> list[FormulaNode]:
    if not train_records:
        return []
    train_df = pd.DataFrame(train_records.values()).sort_values("train_score", ascending=False)
    filtered = train_df[train_df.apply(lambda row: passes_train_filter(row.to_dict()), axis=1)].copy()
    if filtered.empty:
        filtered = train_df
    selected_ids = filtered.head(population_size)["candidate_id"].astype(str).tolist()
    return [nodes_by_id[candidate_id] for candidate_id in selected_ids if candidate_id in nodes_by_id]


def evaluate_train_candidate(
    node: FormulaNode,
    candidate_id: str,
    generation: int,
    source: str,
    train_df: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, object] | None:
    try:
        candidate_series = node.evaluate(train_df)
        metrics, _ = evaluate_candidate(
            data=train_df,
            candidate_name=candidate_id,
            candidate_series=candidate_series,
            n_groups=args.n_groups,
            min_cross_section=args.min_cross_section,
        )
    except Exception:
        return None
    if not metrics:
        return None
    record = {
        "candidate_id": candidate_id,
        "generation": generation,
        "search_source": source,
        **node_metadata(node),
        **{f"train_{key}": value for key, value in metrics.items()},
    }
    record["train_score"] = compute_composite_score(metrics)
    record["passes_train_filter"] = passes_train_filter(record)
    return record


def run_warm_gp_search(
    train_df: pd.DataFrame,
    seed_nodes: list[FormulaNode],
    args: argparse.Namespace,
    rng: random.Random,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, FormulaNode]]:
    available_columns = set(train_df.columns)
    population = build_initial_population(seed_nodes, available_columns, args, rng)

    nodes_by_formula: dict[str, FormulaNode] = {}
    nodes_by_id: dict[str, FormulaNode] = {}
    candidate_records: dict[str, dict[str, object]] = {}
    train_records: dict[str, dict[str, object]] = {}
    formula_to_id: dict[str, str] = {}

    def register_node(node: FormulaNode, generation: int, source: str) -> str | None:
        if not node.is_legal(available_columns, args.max_depth, args.max_complexity, args.max_fields):
            return None
        formula = node.to_formula()
        if formula in formula_to_id:
            return formula_to_id[formula]
        if len(candidate_records) >= args.num_candidates:
            return None
        candidate_id = f"am_{len(candidate_records) + 1:04d}"
        formula_to_id[formula] = candidate_id
        nodes_by_formula[formula] = node
        nodes_by_id[candidate_id] = node
        candidate_records[candidate_id] = {
            "candidate_id": candidate_id,
            "first_generation": generation,
            "search_source": source,
            "train_eval_status": "pending",
            **node_metadata(node),
        }
        return candidate_id

    def evaluate_population(current_population: list[FormulaNode], generation: int, source: str) -> None:
        for node in current_population:
            candidate_id = register_node(node, generation=generation, source=source)
            if candidate_id is None or candidate_id in train_records:
                continue
            train_record = evaluate_train_candidate(
                node=node,
                candidate_id=candidate_id,
                generation=generation,
                source=source,
                train_df=train_df,
                args=args,
            )
            if train_record is None:
                candidate_records[candidate_id]["train_eval_status"] = "failed"
                continue
            train_records[candidate_id] = train_record
            candidate_records[candidate_id]["train_eval_status"] = "ok"
            candidate_records[candidate_id]["train_score"] = train_record["train_score"]

    for generation in range(args.generations + 1):
        print(f"[Info] Warm-GP generation {generation}: evaluating {len(population)} formulas", flush=True)
        evaluate_population(population, generation=generation, source="warm_population" if generation == 0 else "offspring")
        print(
            f"[Info] Warm-GP generation {generation}: registered={len(candidate_records)} train_valid={len(train_records)}",
            flush=True,
        )
        if generation >= args.generations or len(candidate_records) >= args.num_candidates:
            break

        ranked_population = select_next_population(train_records, nodes_by_id, args.population_size)
        if not ranked_population:
            ranked_population = population

        offspring: list[FormulaNode] = []
        seen_offspring: set[str] = set()
        attempts = 0
        while len(offspring) < args.population_size and attempts < args.population_size * 80:
            attempts += 1
            if rng.random() < 0.70 or len(ranked_population) < 2:
                parent = rng.choice(ranked_population)
                child = sample_mutation(parent, seed_nodes, rng)
            else:
                left, right = rng.sample(ranked_population, 2)
                child = sample_crossover(left, right, rng)
            formula = child.to_formula()
            if formula in formula_to_id or formula in seen_offspring:
                continue
            if not child.is_legal(available_columns, args.max_depth, args.max_complexity, args.max_fields):
                continue
            seen_offspring.add(formula)
            offspring.append(child)

        if not offspring:
            break
        population = offspring

    candidate_df = pd.DataFrame(candidate_records.values()).sort_values("candidate_id").reset_index(drop=True)
    train_df_out = pd.DataFrame(train_records.values())
    if not train_df_out.empty:
        train_df_out = train_df_out.sort_values("train_score", ascending=False).reset_index(drop=True)
    return candidate_df, train_df_out, nodes_by_id


def evaluate_oos_survivors(
    train_metrics_df: pd.DataFrame,
    nodes_by_id: dict[str, FormulaNode],
    test_df: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    if train_metrics_df.empty:
        return pd.DataFrame()

    filtered_train_df = train_metrics_df[train_metrics_df.apply(lambda row: passes_train_filter(row.to_dict()), axis=1)].copy()
    # 大规模生成式搜索时，单纯按比例送入 OOS 会让不同实验之间难比较。
    # 如果调用方提供 train_top_k，就固定取训练期前 K 个；旧 warm-GP 脚本没有该参数，
    # 所以这里用 getattr 保持向后兼容。
    if getattr(args, "train_top_k", None) is not None:
        survivor_count = max(1, min(int(args.train_top_k), len(train_metrics_df)))
    else:
        survivor_count = max(1, int(math.ceil(len(train_metrics_df) * args.survivor_ratio)))
    survivor_train_df = filtered_train_df.head(survivor_count).copy()
    if survivor_train_df.empty:
        survivor_train_df = train_metrics_df.head(min(5, len(train_metrics_df))).copy()

    oos_records: list[dict[str, object]] = []
    for row in survivor_train_df.itertuples(index=False):
        record = row._asdict()
        candidate_id = str(record["candidate_id"])
        node = nodes_by_id[candidate_id]
        try:
            candidate_series = node.evaluate(test_df)
            metrics, _ = evaluate_candidate(
                data=test_df,
                candidate_name=candidate_id,
                candidate_series=candidate_series,
                n_groups=args.n_groups,
                min_cross_section=args.min_cross_section,
                rebalance_step=args.target_horizon,
                include_spread_metrics=True,
            )
        except Exception:
            continue
        if not metrics:
            continue
        oos_record = {
            **{key: value for key, value in record.items() if key.startswith("train_") or key in {"candidate_id", "generation", "search_source"}},
            **node_metadata(node),
            **{f"oos_{key}": value for key, value in metrics.items()},
        }
        oos_record["oos_score"] = compute_composite_score(metrics)
        oos_record["sign_consistent"] = bool(
            record.get("train_pearson_ic_mean", float("nan")) > 0 and metrics.get("pearson_ic_mean", float("nan")) > 0
        )
        oos_record["passes_oos_filter"] = passes_oos_filter(metrics)
        oos_records.append(oos_record)

    oos_df = pd.DataFrame(oos_records)
    if not oos_df.empty:
        oos_df = oos_df.sort_values(["oos_score", "oos_pearson_ic_mean"], ascending=False).reset_index(drop=True)
    return oos_df


def compute_alphaeval_scores(
    oos_metrics_df: pd.DataFrame,
    nodes_by_id: dict[str, FormulaNode],
    test_df: pd.DataFrame,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if oos_metrics_df.empty:
        return pd.DataFrame()

    first_half_df, second_half_df = split_oos_halves(test_df)
    candidate_nodes = {str(row.candidate_id): nodes_by_id[str(row.candidate_id)] for row in oos_metrics_df.itertuples(index=False)}
    family_counts = oos_metrics_df["family"].value_counts().to_dict()
    raw_series_by_id: dict[str, pd.Series] = {}
    standardized_signal_by_id: dict[str, pd.Series] = {}

    for candidate_id, node in candidate_nodes.items():
        try:
            raw_series = node.evaluate(test_df)
            raw_series_by_id[candidate_id] = raw_series
            standardized_signal_by_id[candidate_id] = standardize_candidate_cross_sectionally(test_df, raw_series)
        except Exception:
            continue

    signal_correlation_metadata = compute_signal_correlation_metadata(standardized_signal_by_id)
    turnover_metadata = {
        candidate_id: compute_rank_turnover_proxy(
            data=test_df,
            candidate_series=standardized_series,
            top_fraction=args.top_retention_fraction,
        )
        for candidate_id, standardized_series in standardized_signal_by_id.items()
    }

    records: list[dict[str, object]] = []
    for row in oos_metrics_df.itertuples(index=False):
        base_record = row._asdict()
        candidate_id = str(base_record["candidate_id"])
        node = candidate_nodes[candidate_id]
        base_series = raw_series_by_id.get(candidate_id)
        if base_series is None:
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

        gaussian_metrics, _ = evaluate_candidate(
            data=test_df,
            candidate_name=f"{candidate_id}_gauss",
            candidate_series=perturb_candidate_series(base_series, rng, "gaussian", args.noise_scale, args.dropout_rate),
            n_groups=args.n_groups,
            min_cross_section=args.min_cross_section,
            rebalance_step=args.target_horizon,
            include_spread_metrics=True,
        )
        dropout_metrics, _ = evaluate_candidate(
            data=test_df,
            candidate_name=f"{candidate_id}_drop",
            candidate_series=perturb_candidate_series(base_series, rng, "dropout", args.noise_scale, args.dropout_rate),
            n_groups=args.n_groups,
            min_cross_section=args.min_cross_section,
            rebalance_step=args.target_horizon,
            include_spread_metrics=True,
        )

        similarities = [
            formula_similarity(node, other_node)
            for other_id, other_node in candidate_nodes.items()
            if other_id != candidate_id
        ]
        max_similarity = float(max(similarities)) if similarities else 0.0

        baseline_ic = float(base_record.get("oos_pearson_ic_mean", float("nan")))
        baseline_spread = float(base_record.get("oos_long_short_spread", float("nan")))
        first_ic = first_metrics.get("pearson_ic_mean", float("nan")) if first_metrics else float("nan")
        second_ic = second_metrics.get("pearson_ic_mean", float("nan")) if second_metrics else float("nan")
        first_spread = first_metrics.get("long_short_spread", float("nan")) if first_metrics else float("nan")
        second_spread = second_metrics.get("long_short_spread", float("nan")) if second_metrics else float("nan")

        temporal_ic_consistency = float("nan")
        if not (pd.isna(first_ic) or pd.isna(second_ic) or pd.isna(baseline_ic)):
            temporal_ic_consistency = 1.0 - min(abs(float(first_ic) - float(second_ic)) / max(abs(baseline_ic), 1e-8), 2.0)

        temporal_spread_consistency = float("nan")
        if not (pd.isna(first_spread) or pd.isna(second_spread) or pd.isna(baseline_spread)):
            temporal_spread_consistency = 1.0 - min(
                abs(float(first_spread) - float(second_spread)) / max(abs(baseline_spread), 1e-8),
                2.0,
            )

        records.append(
            {
                **base_record,
                "family_count": int(family_counts.get(str(base_record.get("family", "other")), 1)),
                "max_formula_similarity": max_similarity,
                "formula_novelty": 1.0 - max_similarity,
                **signal_correlation_metadata.get(
                    candidate_id,
                    {"max_signal_corr_abs": float("nan"), "signal_uniqueness": float("nan"), "signal_corr_json": "{}"},
                ),
                **turnover_metadata.get(candidate_id, {"rank_turnover": float("nan"), "top_retention": float("nan")}),
                "temporal_ic_consistency": temporal_ic_consistency,
                "temporal_spread_consistency": temporal_spread_consistency,
                "gaussian_ic_retention": bounded_retention(
                    gaussian_metrics.get("pearson_ic_mean", float("nan")) if gaussian_metrics else float("nan"),
                    baseline_ic,
                ),
                "gaussian_spread_retention": bounded_retention(
                    gaussian_metrics.get("long_short_spread", float("nan")) if gaussian_metrics else float("nan"),
                    baseline_spread,
                ),
                "dropout_ic_retention": bounded_retention(
                    dropout_metrics.get("pearson_ic_mean", float("nan")) if dropout_metrics else float("nan"),
                    baseline_ic,
                ),
                "dropout_spread_retention": bounded_retention(
                    dropout_metrics.get("long_short_spread", float("nan")) if dropout_metrics else float("nan"),
                    baseline_spread,
                ),
            }
        )

    score_df = pd.DataFrame(records)
    if score_df.empty:
        return score_df

    score_df["family_uniqueness"] = 1.0 / pd.to_numeric(score_df["family_count"], errors="coerce").clip(lower=1)
    score_df["predictive_score"] = (
        rank_to_unit(score_df["oos_pearson_ic_mean"])
        + rank_to_unit(score_df["oos_spearman_ic_mean"])
        + rank_to_unit(score_df["oos_long_short_spread"])
        + rank_to_unit(score_df["oos_non_overlap_sharpe_horizon_adj"])
    ) / 4.0
    score_df["stability_score"] = (
        rank_to_unit(score_df["oos_pearson_ic_positive_ratio"])
        + rank_to_unit(score_df["oos_spearman_ic_positive_ratio"])
        + rank_to_unit(score_df["temporal_ic_consistency"])
        + rank_to_unit(score_df["temporal_spread_consistency"])
    ) / 4.0
    score_df["robustness_score"] = (
        rank_to_unit(score_df["gaussian_ic_retention"])
        + rank_to_unit(score_df["gaussian_spread_retention"])
        + rank_to_unit(score_df["dropout_ic_retention"])
        + rank_to_unit(score_df["dropout_spread_retention"])
    ) / 4.0
    score_df["diversity_score"] = (
        rank_to_unit(score_df["family_uniqueness"])
        + rank_to_unit(score_df["formula_novelty"])
        + rank_to_unit(score_df["signal_uniqueness"])
    ) / 3.0
    score_df["tradability_score"] = (
        rank_to_unit(score_df["rank_turnover"], ascending=False)
        + rank_to_unit(score_df["top_retention"])
    ) / 2.0
    score_df["interpretability_score"] = (
        rank_to_unit(score_df["formula_complexity"], ascending=False)
        + rank_to_unit(score_df["formula_depth"], ascending=False)
        + rank_to_unit(score_df["field_count"], ascending=False)
    ) / 3.0
    score_df["alphaeval_style_score"] = (
        score_df["predictive_score"] * 0.28
        + score_df["stability_score"] * 0.18
        + score_df["robustness_score"] * 0.14
        + score_df["diversity_score"] * 0.16
        + score_df["tradability_score"] * 0.12
        + score_df["interpretability_score"] * 0.06
        + score_df["financial_logic_score"] * 0.06
    )
    return score_df.sort_values(
        ["alphaeval_style_score", "oos_non_overlap_sharpe_horizon_adj", "oos_pearson_ic_mean"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def build_factor_zoo(alphaeval_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if alphaeval_df.empty:
        return alphaeval_df.copy()

    def signal_corr_to_selected(row: pd.Series, selected_candidate_ids: list[str]) -> float:
        if not selected_candidate_ids:
            return 0.0
        raw_json = row.get("signal_corr_json", "{}")
        try:
            correlation_map = json.loads(raw_json) if isinstance(raw_json, str) else {}
        except json.JSONDecodeError:
            correlation_map = {}
        selected_corr = [abs(float(correlation_map.get(candidate_id, 0.0))) for candidate_id in selected_candidate_ids]
        return float(max(selected_corr)) if selected_corr else 0.0

    eligible = alphaeval_df[alphaeval_df["passes_oos_filter"] == True].copy()  # noqa: E712
    if eligible.empty:
        eligible = alphaeval_df.copy()
    eligible = eligible.sort_values(
        ["alphaeval_style_score", "oos_non_overlap_sharpe_horizon_adj"],
        ascending=[False, False],
    )

    selected_indices: list[int] = []
    selected_candidate_ids: list[str] = []
    family_counts: dict[str, int] = {}
    for index, row in eligible.iterrows():
        family = str(row.get("family", "other"))
        if family_counts.get(family, 0) >= args.max_per_family:
            continue
        if signal_corr_to_selected(row, selected_candidate_ids) > args.max_factor_corr:
            continue
        selected_indices.append(index)
        selected_candidate_ids.append(str(row.get("candidate_id")))
        family_counts[family] = family_counts.get(family, 0) + 1
        if len(selected_indices) >= args.factor_zoo_size:
            break

    if not selected_indices:
        return eligible.head(args.factor_zoo_size).copy()
    return eligible.loc[selected_indices].copy().reset_index(drop=True)


def build_dynamic_combination(
    factor_zoo_df: pd.DataFrame,
    nodes_by_id: dict[str, FormulaNode],
    test_df: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    if factor_zoo_df.empty:
        return pd.DataFrame()

    spread_series_by_id: dict[str, pd.Series] = {}
    for row in factor_zoo_df.itertuples(index=False):
        candidate_id = str(row.candidate_id)
        node = nodes_by_id[candidate_id]
        try:
            metrics, details = evaluate_candidate(
                data=test_df,
                candidate_name=candidate_id,
                candidate_series=node.evaluate(test_df),
                n_groups=args.n_groups,
                min_cross_section=args.min_cross_section,
                rebalance_step=args.target_horizon,
                include_spread_metrics=True,
            )
        except Exception:
            continue
        if not metrics or "spread_df" not in details:
            continue
        spread_df = details["spread_df"]
        if not isinstance(spread_df, pd.DataFrame) or spread_df.empty:
            continue
        spread_series_by_id[candidate_id] = (
            spread_df.assign(date=lambda frame: pd.to_datetime(frame["date"]))
            .set_index("date")["long_short_spread"]
            .sort_index()
        )

    if not spread_series_by_id:
        return pd.DataFrame()

    spread_matrix = pd.DataFrame(spread_series_by_id).sort_index()
    candidate_ids = list(spread_matrix.columns)
    records: list[dict[str, object]] = []
    lag = max(args.target_horizon, 1)
    window = max(args.dynamic_window, 1)

    for row_position, current_date in enumerate(spread_matrix.index):
        history_end = max(row_position - lag, 0)
        history_start = max(history_end - window, 0)
        history = spread_matrix.iloc[history_start:history_end]

        if history.empty:
            raw_scores = pd.Series(1.0, index=candidate_ids, dtype=float)
            score_source = "equal_no_lagged_history"
        else:
            means = history.mean(axis=0)
            stds = history.std(axis=0, ddof=0).replace(0.0, np.nan)
            raw_scores = (means / stds).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
            if math.isclose(float(raw_scores.sum()), 0.0, abs_tol=1e-12):
                raw_scores = pd.Series(1.0, index=candidate_ids, dtype=float)
                score_source = "equal_nonpositive_lagged_scores"
            else:
                score_source = "lagged_ir"

        weights = raw_scores / raw_scores.sum()
        current_spreads = spread_matrix.loc[current_date]
        valid_mask = current_spreads.notna()
        if valid_mask.any():
            active_weights = weights.loc[valid_mask]
            active_weights = active_weights / active_weights.sum()
            combined_spread = float((current_spreads.loc[valid_mask] * active_weights).sum())
        else:
            combined_spread = float("nan")

        record: dict[str, object] = {
            "date": current_date,
            "combined_long_short_spread": combined_spread,
            "score_source": score_source,
            "lagged_history_rows": int(history.shape[0]),
        }
        for candidate_id in candidate_ids:
            record[f"weight_{candidate_id}"] = float(weights[candidate_id])
            record[f"lagged_score_{candidate_id}"] = float(raw_scores[candidate_id])
            record[f"spread_{candidate_id}"] = float(current_spreads[candidate_id]) if pd.notna(current_spreads[candidate_id]) else float("nan")
        records.append(record)

    dynamic_df = pd.DataFrame(records)
    if not dynamic_df.empty:
        valid_spread = pd.to_numeric(dynamic_df["combined_long_short_spread"], errors="coerce").fillna(0.0)
        dynamic_df["combined_cumulative_return"] = (1.0 + valid_spread).cumprod() - 1.0
    return dynamic_df


def summarize_dynamic_combination(dynamic_df: pd.DataFrame, target_horizon: int) -> dict[str, object]:
    if dynamic_df.empty:
        return {
            "dynamic_days": 0,
            "combined_spread_mean": float("nan"),
            "combined_spread_std": float("nan"),
            "combined_sharpe_horizon_adj": float("nan"),
            "combined_cumulative_return": float("nan"),
        }
    spread = pd.to_numeric(dynamic_df["combined_long_short_spread"], errors="coerce").dropna()
    mean_value = float(spread.mean()) if not spread.empty else float("nan")
    std_value = float(spread.std(ddof=0)) if not spread.empty else float("nan")
    sharpe = float("nan")
    if not spread.empty and not math.isclose(std_value, 0.0, abs_tol=1e-12):
        sharpe = float(mean_value / std_value * math.sqrt(252 / max(target_horizon, 1)))
    cumulative = float("nan")
    if "combined_cumulative_return" in dynamic_df.columns and not dynamic_df.empty:
        cumulative = float(dynamic_df["combined_cumulative_return"].iloc[-1])
    return {
        "dynamic_days": int(len(dynamic_df)),
        "combined_spread_mean": mean_value,
        "combined_spread_std": std_value,
        "combined_sharpe_horizon_adj": sharpe,
        "combined_cumulative_return": cumulative,
    }


def table_or_empty(df: pd.DataFrame, columns: list[str], top_k: int | None = None) -> str:
    if df.empty:
        return "_No data available._"
    existing_columns = [column for column in columns if column in df.columns]
    display_df = df[existing_columns].copy()
    if top_k is not None:
        display_df = display_df.head(top_k)
    return dataframe_to_markdown(display_df)


def write_report(
    output_path: Path,
    settings: dict[str, object],
    seed_library_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    train_df: pd.DataFrame,
    oos_df: pd.DataFrame,
    alphaeval_df: pd.DataFrame,
    factor_zoo_df: pd.DataFrame,
    dynamic_summary: dict[str, object],
) -> None:
    report_text = f"""# Auto Factor Mining Report

## Settings

```json
{json.dumps(settings, ensure_ascii=False, indent=2)}
```

## Seed Library

{table_or_empty(seed_library_df, ["seed_source", "formula", "family", "fields", "formula_complexity", "financial_logic_score"], top_k=30)}

## Candidate Generation

- Registered candidate formulas: `{len(candidate_df)}`
- Train-valid candidate formulas: `{int((candidate_df.get("train_eval_status", pd.Series(dtype=str)) == "ok").sum()) if not candidate_df.empty else 0}`
- Searcher: `warm_gp`

## Train Survivors

{table_or_empty(train_df, ["candidate_id", "generation", "formula", "family", "train_pearson_ic_mean", "train_spearman_ic_mean", "train_long_short_spread", "train_group_monotonic_spearman", "train_score", "passes_train_filter"], top_k=15)}

## OOS Survivors

{table_or_empty(oos_df, ["candidate_id", "formula", "family", "oos_pearson_ic_mean", "oos_spearman_ic_mean", "oos_long_short_spread", "oos_non_overlap_sharpe_horizon_adj", "oos_score", "passes_oos_filter"], top_k=15)}

## AlphaEval-Style Scores

{table_or_empty(alphaeval_df, ["candidate_id", "formula", "family", "predictive_score", "stability_score", "robustness_score", "diversity_score", "tradability_score", "interpretability_score", "financial_logic_score", "max_signal_corr_abs", "rank_turnover", "top_retention", "alphaeval_style_score", "in_factor_zoo"], top_k=15)}

## Factor Zoo

{table_or_empty(factor_zoo_df, ["candidate_id", "formula", "family", "alphaeval_style_score", "oos_pearson_ic_mean", "oos_non_overlap_sharpe_horizon_adj", "max_signal_corr_abs", "rank_turnover", "top_retention", "hypothesis"], top_k=None)}

## Dynamic Combination

```json
{json.dumps(dynamic_summary, ensure_ascii=False, indent=2)}
```

## Paper Mapping

- `101 Formulaic Alphas`: implemented as a local seed library using current technical, Alpha191-style, and historical shortlist factors.
- `Warm Start Genetic Programming`: implemented as warm population search with mutation, crossover, depth, complexity, field-count, and duplicate controls.
- `QuantFactor / Alpha2`: implemented only as legality, validity, diversity, and reward-shaping constraints; no RL policy is trained in this MVP.
- `AlphaForge`: implemented as factor mining followed by factor-zoo dynamic combination.
- `AlphaAgent / AlphaLogics`: implemented as offline hypothesis, AST novelty, and complexity metadata; no LLM API calls are made.
- `AlphaEval / AlphaBench`: implemented as executed, multi-dimensional evaluation. Zero-shot LLM evaluation is not used.
- Portfolio realism extension: factor-zoo selection now penalizes high signal correlation and high rank turnover before lagged dynamic combination.

## Leakage Guard

- Train metrics choose survivors before OOS evaluation.
- OOS metrics are used only to validate and rank candidates for research reporting.
- Factor-zoo correlation filtering uses OOS candidate signals, but only for redundancy control among already validated research candidates.
- Dynamic combination uses only lagged spread history ending at least `target_horizon` trading rows before the current OOS date.
- This report is a research artifact, not a live trading proof.
"""
    output_path.write_text(report_text, encoding="utf-8")


def run_search(args: argparse.Namespace) -> None:
    workflow_start = time.perf_counter()
    rng = random.Random(args.random_seed)
    np_rng = np.random.default_rng(args.random_seed)

    model_dir = resolve_path(args.model_dir)
    output_root = resolve_path(args.output_dir)
    run_name = sanitize_name(
        f"{args.searcher}_{args.target_horizon}d_g{args.generations}_p{args.population_size}_c{args.num_candidates}_s{args.random_seed}"
    )
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[Info] Loading and preprocessing train/OOS data", flush=True)
    train_df, test_df, target_column, dataset_summary = load_or_build_preprocessed_train_test(args)
    print("[Info] Building seed library", flush=True)
    seed_library_df, seed_nodes = build_seed_nodes(train_df=train_df, model_dir=model_dir, args=args)
    if seed_library_df.empty:
        raise ValueError("Seed library is empty.")

    print("[Info] Running warm-start formula search", flush=True)
    candidate_df, train_metrics_df, nodes_by_id = run_warm_gp_search(train_df=train_df, seed_nodes=seed_nodes, args=args, rng=rng)
    print("[Info] Evaluating train survivors on OOS", flush=True)
    oos_metrics_df = evaluate_oos_survivors(train_metrics_df=train_metrics_df, nodes_by_id=nodes_by_id, test_df=test_df, args=args)
    print("[Info] Computing AlphaEval-style scores", flush=True)
    alphaeval_df = compute_alphaeval_scores(oos_metrics_df=oos_metrics_df, nodes_by_id=nodes_by_id, test_df=test_df, args=args, rng=np_rng)
    print("[Info] Building factor zoo and lagged dynamic combination", flush=True)
    factor_zoo_df = build_factor_zoo(alphaeval_df, args)
    if not alphaeval_df.empty:
        alphaeval_df["in_factor_zoo"] = alphaeval_df["candidate_id"].isin(set(factor_zoo_df["candidate_id"]))
    dynamic_df = build_dynamic_combination(factor_zoo_df=factor_zoo_df, nodes_by_id=nodes_by_id, test_df=test_df, args=args)
    dynamic_summary = summarize_dynamic_combination(dynamic_df, target_horizon=args.target_horizon)
    runtime_seconds = time.perf_counter() - workflow_start

    settings = {
        "command": "search",
        "searcher": args.searcher,
        "runtime_seconds": runtime_seconds,
        "runtime_readable": format_duration(runtime_seconds),
        "data_path": str(resolve_path(args.data_path)),
        "model_dir": str(model_dir),
        "target_column": target_column,
        "dataset_summary": dataset_summary,
        "population_size": args.population_size,
        "generations": args.generations,
        "num_candidates": args.num_candidates,
        "survivor_ratio": args.survivor_ratio,
        "final_top_k": args.final_top_k,
        "factor_zoo_size": args.factor_zoo_size,
        "max_per_family": args.max_per_family,
        "max_factor_corr": args.max_factor_corr,
        "top_retention_fraction": args.top_retention_fraction,
        "max_depth": args.max_depth,
        "max_complexity": args.max_complexity,
        "max_fields": args.max_fields,
        "dynamic_window": args.dynamic_window,
        "noise_scale": args.noise_scale,
        "dropout_rate": args.dropout_rate,
        "random_seed": args.random_seed,
        "llm_usage": "offline_placeholder",
        "alphaeval_dimensions": [
            "predictive_power",
            "stability",
            "robustness",
            "diversity",
            "tradability",
            "interpretability",
            "financial_logic",
            "signal_redundancy",
        ],
        "alphaeval_score_weights": {
            "predictive_score": 0.28,
            "stability_score": 0.18,
            "robustness_score": 0.14,
            "diversity_score": 0.16,
            "tradability_score": 0.12,
            "interpretability_score": 0.06,
            "financial_logic_score": 0.06,
        },
    }

    (output_dir / "config.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    seed_library_df.to_csv(output_dir / "seed_library.csv", index=False)
    candidate_df.to_csv(output_dir / "candidate_formulas.csv", index=False)
    train_metrics_df.to_csv(output_dir / "train_metrics.csv", index=False)
    oos_metrics_df.to_csv(output_dir / "oos_metrics.csv", index=False)
    alphaeval_df.to_csv(output_dir / "alphaeval_scores.csv", index=False)
    factor_zoo_df.to_csv(output_dir / "factor_zoo.csv", index=False)
    dynamic_df.to_csv(output_dir / "dynamic_combination.csv", index=False)
    write_report(
        output_path=output_dir / "report.md",
        settings=settings,
        seed_library_df=seed_library_df,
        candidate_df=candidate_df,
        train_df=train_metrics_df,
        oos_df=oos_metrics_df,
        alphaeval_df=alphaeval_df,
        factor_zoo_df=factor_zoo_df,
        dynamic_summary=dynamic_summary,
    )

    print(f"[Info] Auto factor mining finished: {output_dir}")
    print(f"[Info] Total runtime: {format_duration(runtime_seconds)}")
    print(f"[Info] Seed formulas: {len(seed_library_df)}")
    print(f"[Info] Candidate formulas registered: {len(candidate_df)}")
    print(f"[Info] Train-valid candidates: {len(train_metrics_df)}")
    print(f"[Info] OOS survivors evaluated: {len(oos_metrics_df)}")
    print(f"[Info] Factor zoo size: {len(factor_zoo_df)}")


def main() -> None:
    args = parse_args()
    if args.command is None:
        raise SystemExit("Choose a command, e.g. `search`. Run with --help for usage.")
    if args.command == "search":
        run_search(args)
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
