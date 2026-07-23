from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


UNARY_OPERATORS = ("id", "neg", "abs", "tanh", "signed_sq", "rank")
BINARY_OPERATORS = ("spread", "blend", "interaction", "ratio", "confirm")
DEFAULT_BLEND_WEIGHTS = (0.25, 0.33, 0.5, 0.67, 0.75)

# 公式挖掘只能使用预测时已经可见的输入变量。
# 下面这些字段要么是标签，要么是未来价格，要么是非数值元信息；
# 如果允许它们进入自动搜索，候选因子会在统计上“看起来很好”，
# 但本质上是在偷看答案，不能用于真实项目或面试展示。
FORBIDDEN_FIELD_NAMES = {
    "date",
    "instrument_id",
    "symbol",
    "ticker",
    "sector",
    "industry",
    "next_open",
    "future_return",
    "y",
    "target",
    "label",
    "predicted_y",
    "adjustment",
    "effective_date",
    "report_date",
    "filing_date",
    "accepted_date",
    "fiscal_period",
    "fiscal_year",
}
FORBIDDEN_FIELD_PREFIXES = (
    "y_",
    "target_",
    "label_",
    "future_",
    "next_",
)

FAMILY_KEYWORDS = {
    "volatility": ("return_std_", "volatility_", "price_range"),
    "channel": ("xschannel_width", "boll_width", "channel"),
    "momentum": ("momentum_", "close_to_ma", "macd", "dma"),
    "liquidity": ("turnover", "volume", "amt_", "obv"),
    "vwap": ("vwap",),
    "alpha191": ("alpha",),
    "size": ("market_cap", "shares_outstanding"),
}

FAMILY_HYPOTHESES = {
    "volatility": "Volatility expansion and cross-sectional repricing.",
    "channel": "Price-channel width and range expansion.",
    "momentum": "Recent trend or mean-reversion structure.",
    "liquidity": "Trading activity and liquidity pressure.",
    "vwap": "Deviation from volume-weighted execution level.",
    "alpha191": "Industrial formula-library seed from Alpha191-style factors.",
    "size": "Size or float proxy exposure; treat carefully after neutralization.",
    "other": "Unclassified formulaic signal; requires manual review.",
}


def _finite_series(series: pd.Series, index: pd.Index) -> pd.Series:
    finite = pd.Series(series, index=index, dtype=float)
    return finite.replace([np.inf, -np.inf], np.nan)


def _cross_sectional_rank(data: pd.DataFrame, series: pd.Series) -> pd.Series:
    if "date" not in data.columns:
        return pd.Series(series, index=data.index, dtype=float).rank(pct=True)
    ranked = pd.Series(np.nan, index=data.index, dtype=float)
    for _, row_index in data.groupby("date").groups.items():
        date_index = pd.Index(row_index)
        ranked.loc[date_index] = pd.to_numeric(series.loc[date_index], errors="coerce").rank(pct=True)
    return ranked


def is_forbidden_formula_field(field_name: str) -> bool:
    """判断字段是否禁止进入自动挖因子。

    这个函数故意写得保守。金融时间序列里，最危险的错误通常不是模型复杂度，
    而是把未来标签、未来价格、预测结果、元信息列混进特征搜索空间。
    """

    lowered = str(field_name).strip().lower()
    if lowered in FORBIDDEN_FIELD_NAMES:
        return True
    return any(lowered == prefix or lowered.startswith(prefix) for prefix in FORBIDDEN_FIELD_PREFIXES)


@dataclass(frozen=True)
class FormulaNode:
    kind: str
    name: str
    children: tuple["FormulaNode", ...] = ()
    weight: float | None = None

    @staticmethod
    def column(name: str) -> "FormulaNode":
        return FormulaNode(kind="column", name=str(name))

    @staticmethod
    def unary(operator: str, child: "FormulaNode") -> "FormulaNode":
        if operator not in UNARY_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {operator}")
        if operator == "id":
            return child
        return FormulaNode(kind="unary", name=operator, children=(child,))

    @staticmethod
    def binary(operator: str, left: "FormulaNode", right: "FormulaNode", weight: float | None = None) -> "FormulaNode":
        if operator not in BINARY_OPERATORS:
            raise ValueError(f"Unsupported binary operator: {operator}")
        if operator == "blend" and weight is None:
            weight = 0.5
        return FormulaNode(kind="binary", name=operator, children=(left, right), weight=weight)

    def to_formula(self) -> str:
        if self.kind == "column":
            return self.name
        if self.kind == "unary":
            child_formula = self.children[0].to_formula()
            return f"{self.name}({child_formula})"
        if self.kind == "binary":
            left_formula = self.children[0].to_formula()
            right_formula = self.children[1].to_formula()
            if self.name == "blend":
                weight = 0.5 if self.weight is None else float(self.weight)
                return f"({weight:.2f}*{left_formula} + {(1.0 - weight):.2f}*{right_formula})"
            if self.name == "spread":
                return f"({left_formula} - {right_formula})"
            if self.name == "interaction":
                return f"({left_formula} * tanh({right_formula}))"
            if self.name == "ratio":
                return f"({left_formula} / (1 + abs({right_formula})))"
            if self.name == "confirm":
                return f"({left_formula} * sign({right_formula}))"
        raise ValueError(f"Unsupported formula node: {self}")

    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        if self.kind == "column":
            if self.name not in data.columns:
                raise KeyError(f"Formula column is missing from data: {self.name}")
            return _finite_series(pd.to_numeric(data[self.name], errors="coerce"), data.index)

        if self.kind == "unary":
            child = self.children[0].evaluate(data)
            if self.name == "neg":
                return _finite_series(-child, data.index)
            if self.name == "abs":
                return _finite_series(child.abs(), data.index)
            if self.name == "tanh":
                return _finite_series(np.tanh(child), data.index)
            if self.name == "signed_sq":
                return _finite_series(np.sign(child) * np.square(child), data.index)
            if self.name == "rank":
                return _finite_series(_cross_sectional_rank(data, child), data.index)
            raise ValueError(f"Unsupported unary operator: {self.name}")

        if self.kind == "binary":
            left = self.children[0].evaluate(data)
            right = self.children[1].evaluate(data)
            if self.name == "spread":
                return _finite_series(left - right, data.index)
            if self.name == "blend":
                weight = 0.5 if self.weight is None else float(self.weight)
                return _finite_series(weight * left + (1.0 - weight) * right, data.index)
            if self.name == "interaction":
                return _finite_series(left * np.tanh(right), data.index)
            if self.name == "ratio":
                return _finite_series(left / (1.0 + np.abs(right)), data.index)
            if self.name == "confirm":
                return _finite_series(left * np.sign(right), data.index)
            raise ValueError(f"Unsupported binary operator: {self.name}")

        raise ValueError(f"Unsupported formula node kind: {self.kind}")

    @property
    def fields(self) -> frozenset[str]:
        if self.kind == "column":
            return frozenset({self.name})
        field_set: set[str] = set()
        for child in self.children:
            field_set.update(child.fields)
        return frozenset(field_set)

    @property
    def operators(self) -> tuple[str, ...]:
        if self.kind == "column":
            return ()
        operators = [self.name]
        for child in self.children:
            operators.extend(child.operators)
        return tuple(operators)

    @property
    def complexity(self) -> int:
        return 1 + sum(child.complexity for child in self.children)

    @property
    def depth(self) -> int:
        if not self.children:
            return 1
        return 1 + max(child.depth for child in self.children)

    def token_set(self) -> set[str]:
        tokens = {self.kind, self.name}
        tokens.update(self.fields)
        tokens.update(self.operators)
        return tokens

    def infer_family(self) -> str:
        matched: list[str] = []
        lowered_fields = [field.lower() for field in self.fields]
        for family_name, keywords in FAMILY_KEYWORDS.items():
            if any(any(keyword in field_name for keyword in keywords) for field_name in lowered_fields):
                matched.append(family_name)
        if not matched:
            return "other"
        return "+".join(sorted(dict.fromkeys(matched)))

    def hypothesis(self) -> str:
        families = self.infer_family().split("+")
        descriptions = [FAMILY_HYPOTHESES.get(family, FAMILY_HYPOTHESES["other"]) for family in families]
        return " ".join(dict.fromkeys(descriptions))

    def financial_logic_score(self) -> float:
        family = self.infer_family()
        base = 0.35 if family == "other" else 0.75
        if self.complexity <= 7:
            base += 0.10
        if len(self.fields) <= 3:
            base += 0.10
        if len(set(self.operators)) <= 3:
            base += 0.05
        if "size" in family:
            base -= 0.15
        return float(max(0.0, min(base, 1.0)))

    def is_legal(self, available_columns: Iterable[str], max_depth: int, max_complexity: int, max_fields: int) -> bool:
        available_set = set(available_columns)
        return bool(
            self.fields.issubset(available_set)
            and not any(is_forbidden_formula_field(field) for field in self.fields)
            and self.depth <= max_depth
            and self.complexity <= max_complexity
            and len(self.fields) <= max_fields
        )


def formula_similarity(left: FormulaNode, right: FormulaNode) -> float:
    left_tokens = left.token_set()
    right_tokens = right.token_set()
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return float(len(left_tokens & right_tokens) / len(union))


def apply_unary(operator: str, node: FormulaNode) -> FormulaNode:
    return FormulaNode.unary(operator, node)


def node_from_legacy_spec(spec: dict[str, object]) -> FormulaNode | None:
    candidate_type = str(spec.get("candidate_type", "")).strip()
    feature_1 = spec.get("feature_1")
    if not feature_1 or str(feature_1) == "nan":
        return None

    left = apply_unary(str(spec.get("unary_1", "id")), FormulaNode.column(str(feature_1)))
    if candidate_type == "unary":
        node = left
    else:
        feature_2 = spec.get("feature_2")
        if not feature_2 or str(feature_2) == "nan":
            return None
        right = apply_unary(str(spec.get("unary_2", "id")), FormulaNode.column(str(feature_2)))
        operator = str(spec.get("binary_template", "blend"))
        raw_weight = spec.get("weight", 0.5)
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError):
            weight = 0.5
        node = FormulaNode.binary(operator, left, right, weight=weight)

    post_transform = str(spec.get("post_transform", "id"))
    if post_transform != "id":
        node = apply_unary(post_transform, node)
    return node
