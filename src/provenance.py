"""可复现实验的 provenance（来源与环境）记录工具。

量化结果如果没有对应代码版本、输入数据指纹和运行环境，就很难复现。
这个模块把这些信息收集成 JSON，供训练入口在每次完整运行结束时保存。

实现刻意只读取本地状态：不会上传数据，也不会把环境变量或 API key 写入
manifest。命令行参数会被记录，因此以后若新增敏感 CLI 参数，也必须先在
``sanitize_arguments`` 中加入脱敏规则。
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import date, datetime, timezone
from importlib import metadata
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping


DEPENDENCY_DISTRIBUTIONS = [
    "pandas",
    "numpy",
    "scikit-learn",
    "lightgbm",
    "xgboost",
    "matplotlib",
    "seaborn",
    "joblib",
    "yfinance",
    "tqdm",
    "requests",
    "pyarrow",
    "torch",
]

SENSITIVE_ARGUMENT_TOKENS = ("api_key", "apikey", "token", "secret", "password")


def utc_now_iso() -> str:
    """返回带时区的 UTC 时间，避免不同电脑的本地时区造成歧义。"""

    return datetime.now(timezone.utc).isoformat()


def project_relative_path(path: Path, project_root: Path) -> str:
    """优先记录仓库相对路径，避免公开报告暴露本机用户名和目录。"""

    resolved_path = path.resolve()
    resolved_root = project_root.resolve()
    try:
        return str(resolved_path.relative_to(resolved_root))
    except ValueError:
        # 外部数据可以位于仓库之外。公开 manifest 只保留文件名；完整绝对路径
        # 没有复现价值，还会泄露个人机器目录。
        return f"external://{resolved_path.name}"


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    """流式计算文件 SHA256，避免把大型 CSV 一次性读入内存。"""

    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        while True:
            chunk = file_obj.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_data_fingerprint(data_path: Path, project_root: Path) -> dict[str, Any]:
    """记录输入文件身份，保证“同名文件”不会被误认为同一份数据。"""

    stat = data_path.stat()
    return {
        "path": project_relative_path(data_path, project_root),
        "size_bytes": int(stat.st_size),
        "modified_time_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "sha256": sha256_file(data_path),
    }


def _run_git(project_root: Path, *args: str) -> str | None:
    """安全读取 Git 状态；Git 不可用时返回空值，不让训练因此失败。"""

    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def collect_git_state(project_root: Path) -> dict[str, Any]:
    """记录 commit 与 dirty 状态，避免把未提交代码的结果当成可复现实验。"""

    commit = _run_git(project_root, "rev-parse", "HEAD")
    branch = _run_git(project_root, "rev-parse", "--abbrev-ref", "HEAD")
    # data/models/outputs 已由 .gitignore 排除，因此这里可以安全地把未跟踪源码
    # 也计入 dirty 状态。否则新建但尚未 git add 的 Python 文件会被错误记录为 clean。
    status = _run_git(project_root, "status", "--porcelain", "--untracked-files=normal")
    return {
        "commit": commit,
        "branch": branch,
        "dirty_tracked_worktree": None if status is None else bool(status),
    }


def collect_dependency_versions() -> dict[str, str | None]:
    """记录研究核心依赖的真实版本，未安装的可选包写成 null。"""

    versions: dict[str, str | None] = {}
    for distribution_name in DEPENDENCY_DISTRIBUTIONS:
        try:
            versions[distribution_name] = metadata.version(distribution_name)
        except metadata.PackageNotFoundError:
            versions[distribution_name] = None
    return versions


def sanitize_arguments(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """对参数字典脱敏，防止未来误把凭据写入公开输出。"""

    safe_arguments: dict[str, Any] = {}
    for key, value in arguments.items():
        normalized_key = str(key).lower()
        if any(token in normalized_key for token in SENSITIVE_ARGUMENT_TOKENS):
            safe_arguments[str(key)] = "<redacted>" if value else None
        else:
            safe_arguments[str(key)] = value
    return safe_arguments


def sanitize_command(command: list[str]) -> list[str]:
    """脱敏命令行中的 ``--key value`` 与 ``--key=value`` 两种凭据写法。"""

    safe_command: list[str] = []
    redact_next = False
    for token in command:
        if redact_next:
            safe_command.append("<redacted>")
            redact_next = False
            continue

        normalized_token = token.lower()
        is_sensitive_flag = token.startswith("--") and any(
            sensitive in normalized_token for sensitive in SENSITIVE_ARGUMENT_TOKENS
        )
        if is_sensitive_flag and "=" in token:
            safe_command.append(f"{token.split('=', maxsplit=1)[0]}=<redacted>")
        else:
            safe_command.append(token)
            redact_next = is_sensitive_flag
    return safe_command


def collect_environment(project_root: Path) -> dict[str, Any]:
    """收集不含密钥的运行环境和 Git 信息。"""

    return {
        "python_version": platform.python_version(),
        "python_executable": Path(sys.executable).name,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "dependencies": collect_dependency_versions(),
        "git": collect_git_state(project_root),
    }


def make_json_compatible(value: Any) -> Any:
    """Recursively convert research outputs into strict JSON-compatible values.

    Python's JSON encoder accepts ``NaN`` and infinity by default, even though
    they are not valid JSON numbers.  Quant metrics frequently contain such
    values when a sample is too short or a denominator is zero.  Public evidence
    must encode them as ``null`` so browsers and non-Python parsers can read the
    files without implementation-specific behavior.
    """

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        numeric_value = float(value)
        return numeric_value if math.isfinite(numeric_value) else None
    if isinstance(value, Mapping):
        return {
            str(key): make_json_compatible(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [make_json_compatible(item) for item in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)

    # NumPy/Pandas scalar objects are not all registered as numbers.Real. Their
    # ``item`` method converts them to a Python scalar without importing those
    # libraries into this small provenance module. Recurse once after conversion
    # so np.bool_, np.datetime64 and nullable scalar outputs receive the same
    # finite-number and date handling as native Python values.
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            scalar_value = item_method()
        except (TypeError, ValueError):
            scalar_value = value
        if scalar_value is not value:
            return make_json_compatible(scalar_value)
    return str(value)


def dumps_strict_json(value: Any, *, indent: int | None = 2) -> str:
    """Serialize one value as standards-compliant, human-readable JSON."""

    return json.dumps(
        make_json_compatible(value),
        ensure_ascii=False,
        indent=indent,
        allow_nan=False,
    )


def write_run_manifest(output_path: Path, manifest: Mapping[str, Any]) -> None:
    """以稳定、可读的 JSON 格式保存 manifest。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        dumps_strict_json(dict(manifest)),
        encoding="utf-8",
    )
