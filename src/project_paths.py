"""项目路径辅助模块。

这个模块的作用很简单，但非常重要：

- 无论用户从哪个工作目录执行 `python main.py`
- 只要传入的是相对路径
- 都会自动把它解析到 MyQuant 项目根目录之下

这样就不会出现“结果文件被写到项目同级目录”这种常见问题。
"""

from __future__ import annotations

from pathlib import Path


# `src/project_paths.py` 位于项目根目录下的 `src/` 目录里，
# 因此向上一级就是整个 MyQuant 项目根目录。
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_project_path(path_like: str | Path) -> Path:
    """把路径统一解析到项目根目录下。

    规则如下：

    - 如果用户传入的是绝对路径，则原样返回；
    - 如果用户传入的是相对路径，则自动拼接到项目根目录下。
    """

    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
