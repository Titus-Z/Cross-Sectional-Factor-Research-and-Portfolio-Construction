"""Helpers for passing controlled model parameter overrides through CLIs."""

from __future__ import annotations

import itertools
import json
from typing import Any


PARAM_NAMES_BY_MODEL = {
    "ridge": {"alpha"},
    "lasso": {"alpha", "max_iter", "tol", "selection"},
    "elastic_net": {"alpha", "l1_ratio", "max_iter", "tol", "selection"},
    "random_forest": {"n_estimators", "max_depth", "min_samples_leaf", "max_features", "n_jobs"},
    "extra_trees": {"n_estimators", "max_depth", "min_samples_leaf", "max_features", "n_jobs"},
    "lightgbm": {"n_estimators", "learning_rate", "max_depth", "num_leaves", "subsample", "colsample_bytree"},
    "xgboost": {"n_estimators", "learning_rate", "max_depth", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"},
}


def parse_model_params_json(raw_value: str | None) -> dict[str, dict[str, Any]]:
    """Parse a JSON object mapping model names to parameter dictionaries."""

    if not raw_value:
        return {}

    try:
        payload = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--model-params-json must be valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("--model-params-json must be a JSON object.")

    parsed: dict[str, dict[str, Any]] = {}
    for model_name, params in payload.items():
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("--model-params-json keys must be non-empty model names.")
        if params is None:
            parsed[model_name.strip().lower()] = {}
            continue
        if not isinstance(params, dict):
            raise ValueError(f"Parameters for model '{model_name}' must be a JSON object.")
        parsed[model_name.strip().lower()] = dict(params)

    return parsed


def _parse_scalar(raw_value: str) -> Any:
    value = raw_value.strip()
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        if "." not in value and "e" not in value.lower():
            return int(value)
        return float(value)
    except ValueError:
        return value


def _normalized_model_names(model_names: list[str]) -> list[str]:
    return [model_name.strip().lower() for model_name in model_names if model_name.strip()]


def parse_hyperparameter_grid(
    raw_value: str | None,
    model_names: list[str],
    *,
    max_combinations_per_model: int = 12,
) -> dict[str, list[dict[str, Any]]]:
    """Parse a compact grid string into per-model parameter candidates.

    The UI syntax is intentionally small:

    - `alpha=0.1,1,10` applies to configured models that support `alpha`.
    - `elastic_net_l1_ratio=0.1,0.5` targets one model explicitly.
    - `ridge_alpha=2.0;lasso_alpha=0.001` can mix model-specific keys.
    """

    if not raw_value or not raw_value.strip():
        return {}

    configured_models = _normalized_model_names(model_names)
    if not configured_models:
        return {}

    values_by_model: dict[str, dict[str, list[Any]]] = {model_name: {} for model_name in configured_models}
    supported_model_names = sorted(PARAM_NAMES_BY_MODEL, key=len, reverse=True)
    for token in raw_value.replace("\n", ";").split(";"):
        if "=" not in token:
            continue
        raw_key, raw_values = token.split("=", 1)
        key = raw_key.strip().lower().replace("-", "_")
        values = [_parse_scalar(value) for value in raw_values.split(",") if value.strip()]
        if not key or not values:
            continue

        explicit_model = ""
        param_name = key
        for model_name in supported_model_names:
            prefix = f"{model_name}_"
            if key.startswith(prefix):
                explicit_model = model_name
                param_name = key.removeprefix(prefix)
                break

        target_models = [explicit_model] if explicit_model else configured_models
        for model_name in target_models:
            if model_name not in values_by_model:
                continue
            if param_name not in PARAM_NAMES_BY_MODEL.get(model_name, set()):
                continue
            values_by_model[model_name][param_name] = values

    grid_by_model: dict[str, list[dict[str, Any]]] = {}
    for model_name, param_values in values_by_model.items():
        if not param_values:
            continue
        param_names = sorted(param_values)
        combinations = [
            dict(zip(param_names, values))
            for values in itertools.product(*(param_values[param_name] for param_name in param_names))
        ]
        grid_by_model[model_name] = combinations[: max(1, int(max_combinations_per_model))]

    return grid_by_model


def model_params_for_model(
    model_name: str,
    model_params_by_name: dict[str, dict[str, Any]] | None,
) -> dict[str, Any] | None:
    """Return a copy of model-specific or wildcard parameter overrides."""

    if not model_params_by_name:
        return None
    normalized_name = model_name.strip().lower()
    params = model_params_by_name.get(normalized_name) or model_params_by_name.get("*")
    if not params:
        return None
    return dict(params)


def model_param_candidates_for_model(
    model_name: str,
    model_params_by_name: dict[str, dict[str, Any]] | None,
    hyperparameter_grid_by_name: dict[str, list[dict[str, Any]]] | None,
) -> list[dict[str, Any] | None]:
    """Build final candidate parameter dictionaries for validation."""

    base_params = model_params_for_model(model_name, model_params_by_name) or {}
    grid_candidates = (hyperparameter_grid_by_name or {}).get(model_name.strip().lower()) or []
    if not grid_candidates:
        return [base_params or None]
    return [{**base_params, **candidate} for candidate in grid_candidates]


def selected_model_params_from_summary(summary_df: Any, model_name: str) -> dict[str, Any] | None:
    """Recover selected model params from validation summary rows."""

    if summary_df is None or getattr(summary_df, "empty", True):
        return None
    matches = summary_df[summary_df["model"] == model_name]
    if matches.empty or "model_params_json" not in matches.columns:
        return None
    raw_value = str(matches.iloc[0]["model_params_json"] or "").strip()
    if not raw_value or raw_value == "{}":
        return None
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
