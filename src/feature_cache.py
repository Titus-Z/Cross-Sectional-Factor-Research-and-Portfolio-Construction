"""特征工程缓存。

当前项目最昂贵的重复工作包括：

1. 时间切分；
2. 训练集特征工程；
3. 测试集上下文拼接；
4. 测试集特征工程；
5. 候选特征对齐。

这些步骤在下面几种场景里会被反复重复：

- `10d_all_models` 和 `10d_linear_models`
- 多组消融实验
- 因子诊断重新构造 OOS 特征
- 主入口和实验入口反复跑同一份数据

所以这里把“严格时间切分后的 train/test 特征矩阵”单独缓存起来。

注意：

- 缓存的是特征矩阵，不是模型；
- 这里仍然使用 `pickle`，因为它不依赖额外 parquet 引擎；
- 这一层缓存可以跨不同模型、不同消融配置复用；
- 但不能跨不同目标 horizon 直接复用，因为 `train_df/test_df` 里包含激活后的 `y` 列。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.cache_fingerprint import build_source_fingerprint
from src.provenance import dumps_strict_json


# v9 禁止 loader 填充缺失 OHLCV。旧缓存可能含伪造的
# 价格/成交路径、切分前收益特征或 synthetic market-cap，禁止复用。
FEATURE_CACHE_VERSION = "feature_cache_v9_missing_market_data_preserved_no_window_fill_post_split_returns_instrument_purged_adjustment_aware_no_synthetic_market_cap"
FEATURE_CACHE_SOURCE_FILES = (
    "src/alpha191.py",
    "src/data_loader.py",
    "src/feature_generator.py",
    "src/time_series_pipeline.py",
    "src/utils.py",
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


def build_feature_cache_key(
    *,
    data_path: Path,
    sample_start_date: str | None,
    oos_start_date: str | None,
    test_size: float,
    target_horizon: int,
    history_window: int,
    alpha_factor_names: list[str] | None = None,
    price_adjustment_mode: str = "vendor_adjusted",
    extra_context: dict[str, Any] | None = None,
) -> str:
    """根据影响特征工程结果的条件生成稳定缓存键。"""

    # `alpha_factor_names=None` 在当前项目里表示“使用默认的全部 Alpha191”；
    # `alpha_factor_names=[]` 表示“明确不生成 Alpha191”。
    # 这两种语义完全不同，缓存键必须区分，否则轻量技术指标实验可能误读全量 Alpha 缓存，
    # 或全量 Alpha 实验误读无 Alpha 缓存。
    if alpha_factor_names is None:
        alpha_cache_scope: list[str] | str = "__ALL_ALPHA191__"
    else:
        alpha_cache_scope = list(alpha_factor_names)

    payload = {
        "version": FEATURE_CACHE_VERSION,
        "source_fingerprint": build_source_fingerprint(FEATURE_CACHE_SOURCE_FILES),
        "file_signature": _file_signature(data_path),
        "sample_start_date": sample_start_date,
        "oos_start_date": oos_start_date,
        "test_size": float(test_size),
        "target_horizon": int(target_horizon),
        "history_window": int(history_window),
        "alpha_factor_names": alpha_cache_scope,
        "price_adjustment_mode": price_adjustment_mode,
        "extra_context": extra_context or {},
    }
    digest = hashlib.sha1(_serialize_payload(payload).encode("utf-8")).hexdigest()
    return digest[:20]


def _resolve_cache_dir(cache_root: Path, cache_key: str) -> Path:
    return cache_root / "features" / cache_key


def load_feature_cache(
    cache_root: Path,
    cache_key: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]] | None:
    """读取特征缓存。

    如果任一关键文件缺失，直接返回 `None`，上层重新计算。
    """

    cache_dir = _resolve_cache_dir(cache_root, cache_key)
    train_path = cache_dir / "train_features.pkl"
    test_path = cache_dir / "test_features.pkl"
    columns_path = cache_dir / "feature_columns.json"
    metadata_path = cache_dir / "feature_metadata.json"

    if not (train_path.exists() and test_path.exists() and columns_path.exists() and metadata_path.exists()):
        return None

    train_df = pd.read_pickle(train_path)
    test_df = pd.read_pickle(test_path)
    feature_columns = json.loads(columns_path.read_text(encoding="utf-8"))
    feature_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return train_df, test_df, feature_columns, feature_metadata


def save_feature_cache(
    cache_root: Path,
    cache_key: str,
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    feature_metadata: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> Path:
    """保存特征缓存。"""

    cache_dir = _resolve_cache_dir(cache_root, cache_key)
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_pickle(cache_dir / "train_features.pkl")
    test_df.to_pickle(cache_dir / "test_features.pkl")
    (cache_dir / "feature_columns.json").write_text(
        dumps_strict_json(list(feature_columns)),
        encoding="utf-8",
    )
    (cache_dir / "feature_metadata.json").write_text(
        dumps_strict_json(feature_metadata),
        encoding="utf-8",
    )

    if metadata is not None:
        (cache_dir / "cache_metadata.json").write_text(
            dumps_strict_json(metadata),
            encoding="utf-8",
        )

    return cache_dir
