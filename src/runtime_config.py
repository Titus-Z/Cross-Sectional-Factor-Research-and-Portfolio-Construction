"""项目运行默认配置。

这个模块只做一件事：把“当前主线默认口径”集中到一个地方，
避免 `main.py`、`main_experiments.py`、`main_ablation.py`、
`main_factor_diagnostics.py` 各自写一套默认值，最后互相漂移。

当前公开主线口径是：

- canonical 股票池：`us_large_cap_300`（修复后结果等待正式重跑）
- 数据文件：`data/us_large_cap_300_daily.csv`
- 样本起点：`2022-01-01`
- OOS 起点：`2025-06-01`
- 主目标：`10d`
- 主模型集合：去掉 `random_forest` 的核心模型组

`us_active_3000` 的下载和训练能力仍然保留，但在正式结果同步前，
不能作为公开默认实验。这样可以避免代码默认值暗示尚未公开验证的结果。

说明：

- `random_forest` 没有被删除，仍然保留在代码层面支持；
- 但它不再属于默认主实验模型集合，而是降级为显式对照组。
"""

from __future__ import annotations


DEFAULT_SAMPLE_START_DATE = "2022-01-01"
DEFAULT_OOS_START_DATE = "2025-06-01"
DEFAULT_PRIMARY_TARGET_HORIZON = 10

DEFAULT_PRIMARY_UNIVERSE = "us_large_cap_300"
DEFAULT_PRIMARY_DATA_PATH = "data/us_large_cap_300_daily.csv"

DEFAULT_EXPERIMENT_MODEL_ROOT = "models/experiments_us300"
DEFAULT_EXPERIMENT_OUTPUT_ROOT = "outputs/experiments_us300"

DEFAULT_ABLATION_MODEL_ROOT = "models/ablation_us300"
DEFAULT_ABLATION_OUTPUT_ROOT = "outputs/ablation_us300"

DEFAULT_FACTOR_DIAGNOSTICS_MODEL_DIR = "models/public_us300_release_v1"
DEFAULT_FACTOR_DIAGNOSTICS_OUTPUT_DIR = "outputs/factor_diagnostics/public_us300_release_v1"

CORE_MODEL_SUITE = [
    "xgboost",
    "extra_trees",
    "ridge",
    "lasso",
    "elastic_net",
]

LINEAR_MODEL_SUITE = [
    "ridge",
    "lasso",
    "elastic_net",
]

RANDOM_FOREST_CONTROL_SUITE = [
    "random_forest",
]

DEFAULT_EXPERIMENT_PRESETS = [
    "10d_all_models",
    "10d_linear_models",
]
