"""轻量级进度条工具。

这个模块的目标很简单：

- 如果环境里装了 `tqdm`，就显示友好的进度条；
- 如果没有装，也不要影响主流程运行。

之所以单独拆一个文件，而不是每个模块都各写一遍 `try/except import tqdm`，
是为了让主流程代码更干净，也方便后续统一调整进度条样式。
"""

from __future__ import annotations

from typing import Iterable

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


class _DummyProgressBar:
    """当环境里没有 `tqdm` 时使用的空进度条。

    这样上层代码依然可以安全地调用：

    - `update(...)`
    - `set_postfix_str(...)`
    - `close()`

    而不用到处写额外判断。
    """

    def update(self, n: int = 1) -> None:
        return None

    def set_postfix_str(self, text: str) -> None:
        return None

    def close(self) -> None:
        return None


def optional_progress(
    items: Iterable,
    description: str,
    enabled: bool = True,
    total: int | None = None,
    leave: bool = True,
):
    """给任意可迭代对象包上一层可选进度条。"""

    if not enabled or tqdm is None:
        return items
    return tqdm(items, desc=description, total=total, leave=leave)


def create_progress_bar(
    total: int,
    description: str,
    enabled: bool = True,
    leave: bool = True,
):
    """创建一个可手动 `update()` 的进度条。"""

    if not enabled or tqdm is None:
        return _DummyProgressBar()
    return tqdm(total=total, desc=description, leave=leave)


def format_duration(seconds: float | int | None) -> str:
    """把秒数格式化成更适合终端阅读的持续时间文本。

    进度条虽然会自动显示 ETA，但默认格式在嵌套进度条比较多时不够直观。
    这里补一个统一格式函数，方便我们在：

    - postfix 文本里显示当前阶段已耗时；
    - 训练报告里显示每个阶段用时；
    - 最终实验摘要里显示总耗时。
    """

    if seconds is None:
        return "N/A"

    total_seconds = max(float(seconds), 0.0)
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    secs = total_seconds % 60

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:04.1f}"
    return f"{minutes:02d}:{secs:04.1f}"
