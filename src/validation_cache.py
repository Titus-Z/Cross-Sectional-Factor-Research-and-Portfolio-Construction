"""Walk-forward 验证缓存。

和特征工程相比，walk-forward 的单次耗时通常没那么夸张，
但在下面这些场景里仍然会形成明显重复：

- 同一份数据反复重跑主实验
- 同一组实验改报告、不改训练逻辑再跑
- 长时间多轮调参时重复验证完全相同的模型集合

所以这里把 walk-forward 的三个关键产物缓存下来：

- `fold_metrics_df`
- `model_summary_df`
- `model_weights`

这样当输入条件完全一致时，可以直接跳过整段验证计算。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.cache_fingerprint import build_source_fingerprint
from src.provenance import dumps_strict_json


# v9 also invalidates validation based on loader-filled OHLCV observations.
VALIDATION_CACHE_VERSION = "validation_cache_v9_missing_market_data_preserved_no_window_fill_post_split_returns_instrument_purged_adjustment_aware_no_synthetic_market_cap"
VALIDATION_CACHE_SOURCE_FILES = (
    "src/alpha191.py",
    "src/data_loader.py",
    "src/feature_generator.py",
    "src/feature_selector.py",
    "src/model.py",
    "src/model_params.py",
    "src/preprocessing.py",
    "src/reporting.py",
    "src/time_series_pipeline.py",
    "src/utils.py",
    "src/validation.py",
)


def _serialize_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _file_signature(data_path: Path) -> dict[str, Any]:
    signature: dict[str, Any] = {"path": str(data_path)}
    if data_path.exists():
        stat = data_path.stat()
        signature["size"] = int(stat.st_size)
        signature["mtime_ns"] = int(stat.st_mtime_ns)
    return signature


def build_validation_cache_key(
    *,
    data_path: Path,
    sample_start_date: str | None,
    oos_start_date: str | None,
    test_size: float,
    target_horizon: int,
    history_window: int,
    feature_columns: list[str],
    model_names: list[str],
    selector_config: dict[str, Any],
    n_splits: int,
    random_state: int,
    score_metric: str,
    apply_preprocessing: bool,
    apply_neutralization: bool,
    winsorize_quantile: float,
    price_adjustment_mode: str = "vendor_adjusted",
    extra_context: dict[str, Any] | None = None,
) -> str:
    """根据验证结果真正依赖的条件生成稳定缓存键。"""

    payload = {
        "version": VALIDATION_CACHE_VERSION,
        "source_fingerprint": build_source_fingerprint(VALIDATION_CACHE_SOURCE_FILES),
        "file_signature": _file_signature(data_path),
        "sample_start_date": sample_start_date,
        "oos_start_date": oos_start_date,
        "test_size": float(test_size),
        "target_horizon": int(target_horizon),
        "history_window": int(history_window),
        "feature_columns": list(feature_columns),
        "model_names": list(model_names),
        "selector_config": selector_config,
        "n_splits": int(n_splits),
        "random_state": int(random_state),
        "score_metric": score_metric,
        "apply_preprocessing": bool(apply_preprocessing),
        "apply_neutralization": bool(apply_neutralization),
        "winsorize_quantile": float(winsorize_quantile),
        "price_adjustment_mode": price_adjustment_mode,
        "extra_context": extra_context or {},
    }
    digest = hashlib.sha1(_serialize_payload(payload).encode("utf-8")).hexdigest()
    return digest[:20]


def _resolve_cache_dir(cache_root: Path, cache_key: str) -> Path:
    return cache_root / "validation" / cache_key


def load_validation_cache(
    cache_root: Path,
    cache_key: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]] | None:
    """读取 walk-forward 验证缓存。"""

    cache_dir = _resolve_cache_dir(cache_root, cache_key)
    fold_path = cache_dir / "fold_metrics.csv"
    model_path = cache_dir / "model_summary.csv"
    weight_path = cache_dir / "model_weights.json"

    if not (fold_path.exists() and model_path.exists() and weight_path.exists()):
        return None

    fold_metrics_df = pd.read_csv(fold_path)
    model_summary_df = pd.read_csv(model_path)
    model_weights = json.loads(weight_path.read_text(encoding="utf-8"))
    return fold_metrics_df, model_summary_df, model_weights


def save_validation_cache(
    cache_root: Path,
    cache_key: str,
    *,
    fold_metrics_df: pd.DataFrame,
    model_summary_df: pd.DataFrame,
    model_weights: dict[str, float],
    metadata: dict[str, Any] | None = None,
) -> Path:
    """保存 walk-forward 验证缓存。"""

    cache_dir = _resolve_cache_dir(cache_root, cache_key)
    cache_dir.mkdir(parents=True, exist_ok=True)

    fold_metrics_df.to_csv(cache_dir / "fold_metrics.csv", index=False)
    model_summary_df.to_csv(cache_dir / "model_summary.csv", index=False)
    (cache_dir / "model_weights.json").write_text(
        dumps_strict_json(model_weights),
        encoding="utf-8",
    )

    if metadata is not None:
        (cache_dir / "cache_metadata.json").write_text(
            dumps_strict_json(metadata),
            encoding="utf-8",
        )

    return cache_dir
