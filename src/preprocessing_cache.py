"""横截面预处理缓存。

当前项目里，最明显的重复计算之一是：

1. 已经生成好了 train/test 特征矩阵；
2. 但在不同实验之间，又重复做同样的横截面预处理。

这在下面几种场景里尤其浪费时间：

- `5d_all_models` 和 `5d_linear_models`
- `10d_all_models` 和 `10d_linear_models`
- 消融实验里多个只改模型、不改预处理配置的实验

这个模块先实现一层“预处理结果缓存”：

- 输入条件一样，就直接复用之前的预处理后 train/test 表；
- 不再重复做 winsorize / z-score / neutralization。

注意：

- 这里缓存的是“预处理之后的 DataFrame”，不是模型本身；
- 当前使用 `pickle`，因为它不依赖额外 parquet 引擎；
- 后面如果要进一步优化，可以再把缓存格式换成 parquet。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.cache_fingerprint import build_source_fingerprint
from src.provenance import dumps_strict_json


# v9 follows feature_cache_v9, including preserved missing OHLCV observations.
PREPROCESSING_CACHE_VERSION = "preprocessing_cache_v9_missing_market_data_preserved_no_window_fill_post_split_returns_instrument_purged_adjustment_aware_no_synthetic_market_cap"
PREPROCESSING_CACHE_SOURCE_FILES = (
    "src/alpha191.py",
    "src/data_loader.py",
    "src/feature_generator.py",
    "src/preprocessing.py",
    "src/time_series_pipeline.py",
    "src/utils.py",
)


def _serialize_payload(payload: dict[str, Any]) -> str:
    """把缓存元信息稳定序列化，用于生成哈希键。"""

    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def build_preprocessing_cache_key(
    *,
    data_path: Path,
    sample_start_date: str | None,
    oos_start_date: str | None,
    test_size: float,
    target_horizon: int,
    history_window: int,
    feature_columns: list[str],
    apply_preprocessing: bool,
    apply_neutralization: bool,
    winsorize_quantile: float,
    price_adjustment_mode: str = "vendor_adjusted",
    extra_context: dict[str, Any] | None = None,
) -> str:
    """根据当前实验配置生成稳定缓存键。

    核心原则：

    - 只要影响预处理结果的条件发生变化，缓存键就必须变化；
    - 不影响预处理结果的条件，例如模型名字，不应该进入缓存键。

    因此这里故意包含：

    - 数据文件路径和文件状态
    - 时间切分参数
    - 目标周期
    - history window
    - 特征列集合
    - 预处理配置
    """

    file_signature: dict[str, Any] = {"path": str(data_path)}
    if data_path.exists():
        stat = data_path.stat()
        file_signature["size"] = int(stat.st_size)
        file_signature["mtime_ns"] = int(stat.st_mtime_ns)

    payload = {
        "version": PREPROCESSING_CACHE_VERSION,
        "source_fingerprint": build_source_fingerprint(PREPROCESSING_CACHE_SOURCE_FILES),
        "file_signature": file_signature,
        "sample_start_date": sample_start_date,
        "oos_start_date": oos_start_date,
        "test_size": float(test_size),
        "target_horizon": int(target_horizon),
        "history_window": int(history_window),
        "feature_columns": list(feature_columns),
        "apply_preprocessing": bool(apply_preprocessing),
        "apply_neutralization": bool(apply_neutralization),
        "winsorize_quantile": float(winsorize_quantile),
        "price_adjustment_mode": price_adjustment_mode,
        "extra_context": extra_context or {},
    }

    digest = hashlib.sha1(_serialize_payload(payload).encode("utf-8")).hexdigest()
    return digest[:20]


def _resolve_cache_dir(cache_root: Path, cache_key: str) -> Path:
    """返回某个缓存键对应的目录。"""

    return cache_root / "preprocessing" / cache_key


def load_preprocessing_cache(
    cache_root: Path,
    cache_key: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]] | None:
    """读取预处理缓存。

    如果缓存不完整，直接返回 `None`，上层照常重新计算。
    """

    cache_dir = _resolve_cache_dir(cache_root, cache_key)
    train_path = cache_dir / "train_preprocessed.pkl"
    test_path = cache_dir / "test_preprocessed.pkl"
    summary_path = cache_dir / "preprocessing_summary.json"

    if not (train_path.exists() and test_path.exists() and summary_path.exists()):
        return None

    train_df = pd.read_pickle(train_path)
    test_df = pd.read_pickle(test_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return train_df, test_df, summary


def save_preprocessing_cache(
    cache_root: Path,
    cache_key: str,
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    preprocessing_summary: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> Path:
    """保存预处理缓存。"""

    cache_dir = _resolve_cache_dir(cache_root, cache_key)
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_pickle(cache_dir / "train_preprocessed.pkl")
    test_df.to_pickle(cache_dir / "test_preprocessed.pkl")
    (cache_dir / "preprocessing_summary.json").write_text(
        dumps_strict_json(preprocessing_summary),
        encoding="utf-8",
    )

    if metadata is not None:
        (cache_dir / "cache_metadata.json").write_text(
            dumps_strict_json(metadata),
            encoding="utf-8",
        )

    return cache_dir
