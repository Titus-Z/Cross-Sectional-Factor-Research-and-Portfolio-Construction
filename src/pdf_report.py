"""Minimal PDF report helpers.

这个模块只解决一个具体问题：把实验结果压缩成手机上能直接看的 PDF。

这里没有引入 reportlab / weasyprint 之类的新依赖，而是复用项目已经有的
`matplotlib`。优点是环境更稳定；缺点是排版不会像正式论文那样精细。
对当前目标来说，PDF 的职责是“能读、能转发、和 CSV 结果一致”。
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from matplotlib import font_manager
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties


@dataclass(frozen=True)
class PdfSection:
    """PDF 中的一个文本段落或表格段落。"""

    title: str
    body: str = ""
    table: pd.DataFrame | None = None
    max_table_rows: int = 18


def find_readable_font() -> FontProperties:
    """尽量选择能显示中文的本机字体。

    macOS 上通常有 PingFang 或 Heiti。找不到时退回 matplotlib 默认字体，
    这样脚本仍可运行，只是中文可能显示为方框。
    """

    candidate_paths = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    for font_path in candidate_paths:
        if Path(font_path).exists():
            font_manager.fontManager.addfont(font_path)
            return FontProperties(fname=font_path)
    return FontProperties()


def wrap_text(text: str, width: int = 88) -> list[str]:
    """把长文本切成适合 PDF 页面宽度的行。

    对中文来说，`textwrap` 不能完美按词切分，但按字符宽度切行足够用于报告。
    """

    lines: list[str] = []
    for raw_line in str(text).splitlines():
        if not raw_line.strip():
            lines.append("")
            continue
        wrapped = textwrap.wrap(
            raw_line,
            width=width,
            break_long_words=True,
            replace_whitespace=False,
            drop_whitespace=False,
        )
        lines.extend(wrapped or [""])
    return lines


def format_table_for_pdf(table: pd.DataFrame, max_rows: int = 18) -> str:
    """把小表格转成等宽文本，避免复杂表格排版。"""

    if table is None or table.empty:
        return "No table data."

    preview = table.head(max_rows).copy()
    for column in preview.columns:
        if pd.api.types.is_float_dtype(preview[column]):
            preview[column] = preview[column].map(lambda value: f"{value:.6g}" if pd.notna(value) else "")
    try:
        return preview.to_string(index=False, max_colwidth=34)
    except TypeError:
        return preview.to_string(index=False)


def write_pdf_report(
    output_path: str | Path,
    *,
    title: str,
    sections: Iterable[PdfSection],
    subtitle: str | None = None,
) -> Path:
    """把若干 section 写成 PDF。

    这个函数故意保持简单：

    - 每页固定大小；
    - 文本自动换行；
    - 表格以等宽文本展示；
    - 行数超过一页时自动分页。
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    font_prop = find_readable_font()

    page_width, page_height = 8.27, 11.69
    left_margin = 0.55
    top_y = 11.15
    line_height = 0.18
    title_line_height = 0.25

    def new_page(pdf: PdfPages, page_title: str) -> tuple[plt.Figure, plt.Axes, float]:
        fig = plt.figure(figsize=(page_width, page_height))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        ax.text(
            left_margin,
            top_y,
            page_title,
            fontproperties=font_prop,
            fontsize=14,
            weight="bold",
            va="top",
        )
        y = top_y - title_line_height
        if subtitle:
            ax.text(left_margin, y, subtitle, fontproperties=font_prop, fontsize=8.5, va="top")
            y -= line_height * 1.5
        return fig, ax, y

    with PdfPages(output_path) as pdf:
        fig, ax, y = new_page(pdf, title)

        for section in sections:
            rendered_lines = [f"## {section.title}"]
            if section.body:
                rendered_lines.extend(wrap_text(section.body))
            if section.table is not None:
                rendered_lines.append("")
                rendered_lines.extend(wrap_text(format_table_for_pdf(section.table, section.max_table_rows), width=112))
            rendered_lines.append("")

            for line in rendered_lines:
                if y < 0.55:
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
                    fig, ax, y = new_page(pdf, title)

                font_size = 10.0
                weight = "normal"
                if line.startswith("## "):
                    line = line[3:]
                    font_size = 11.5
                    weight = "bold"

                ax.text(
                    left_margin,
                    y,
                    line,
                    fontproperties=font_prop,
                    fontsize=font_size,
                    weight=weight,
                    va="top",
                )
                y -= line_height if font_size <= 10.0 else title_line_height

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    return output_path
