"""模型封装模块。

这个文件的目标是把不同模型统一成相似的接口，方便主流程做这些事情：

1. 按模型名字创建模型；
2. 调用统一的 `fit / predict / save`；
3. 统一提取特征重要性；
4. 把多个模型放进集成器中做加权平均。

当前支持的模型分成两类：

- 树模型：
  - `lightgbm`
  - `xgboost`
  - `random_forest`
  - `extra_trees`
- 线性 baseline：
  - `ridge`
  - `lasso`
  - `elastic_net`

说明：

- 这里没有默认加入 `LogisticRegression`，因为当前项目的目标变量 `y` 是连续收益率，
  本质上是一个回归问题，而不是分类问题。
- 如果你后续把任务改成“涨跌分类”，再把逻辑回归加进来会更自然。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None


def calculate_regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """计算回归任务常用指标。

    这个函数主要给 walk-forward 验证和训练阶段的快速反馈使用。
    更完整的量化指标会在 `src/reporting.py` 里计算。
    """

    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)

    rmse = float(np.sqrt(np.mean((y_true_array - y_pred_array) ** 2)))
    mae = float(np.mean(np.abs(y_true_array - y_pred_array)))

    if len(y_true_array) >= 2 and np.std(y_true_array) > 0 and np.std(y_pred_array) > 0:
        pearson_corr = float(pd.Series(y_true_array).corr(pd.Series(y_pred_array), method="pearson"))
    else:
        pearson_corr = float("nan")

    return {
        "rmse": rmse,
        "mae": mae,
        "pearson_corr": pearson_corr,
    }


class BaseRegressorWrapper:
    """所有回归模型封装的共同父类。

    这个类把不同模型的公共逻辑放到一起，子类只需要关心两件事：

    1. 默认参数是什么；
    2. 如何从底层模型中提取“特征重要性”。
    """

    model_name: str = "base"

    def __init__(self, random_state: int = 42, params: Optional[Dict] = None) -> None:
        default_params = self.get_default_params(random_state=random_state)
        if params:
            default_params.update(params)

        self.params = default_params
        self.model = self.build_model()

    @staticmethod
    def is_available() -> bool:
        """判断当前模型是否可用。

        对 scikit-learn 自带模型来说，默认总是可用。
        对 LightGBM / XGBoost 这种可选依赖，子类会重写这个方法。
        """

        return True

    def get_default_params(self, random_state: int) -> Dict:
        """返回默认参数。

        子类必须实现这个方法。
        """

        raise NotImplementedError

    def build_model(self):
        """构造底层模型对象。

        子类必须实现这个方法。
        """

        raise NotImplementedError

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """训练模型，并在有验证集时返回快速回归指标。"""

        self.model.fit(X_train, y_train)

        if X_valid is not None and y_valid is not None:
            valid_pred = self.predict(X_valid)
            return calculate_regression_metrics(y_valid, valid_pred)
        return {}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """生成预测值。"""

        return self.model.predict(X)

    def get_params(self) -> Dict:
        """返回模型参数，便于保存和写报告。"""

        return dict(self.params)

    def save(self, output_path: str | Path) -> None:
        """保存训练好的模型。"""

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, output_path)

    def _extract_raw_importance(self) -> np.ndarray:
        """提取原始重要性数组。

        子类需要根据各自模型的结构实现这个方法：

        - 树模型通常用 `feature_importances_`
        - 线性模型通常用 `abs(coef_)`
        """

        raise NotImplementedError

    def get_feature_importance(self, feature_names: list[str], model_name: str | None = None) -> pd.DataFrame:
        """返回标准化前的特征重要性表。

        这里返回的 `importance` 还不是跨模型可直接比较的值，
        因为树模型和线性模型的重要性尺度并不一致。
        后续在主流程里会再做“模型内归一化 + 按模型权重加权”。
        """

        raw_importance = np.asarray(self._extract_raw_importance(), dtype=float)
        if len(raw_importance) != len(feature_names):
            raise ValueError("Feature importance length does not match feature_names length.")

        return (
            pd.DataFrame(
                {
                    "feature": feature_names,
                    "importance": raw_importance,
                    "model": model_name or self.model_name,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )


class LightGBMModel(BaseRegressorWrapper):
    """LightGBM 回归模型封装。"""

    model_name = "lightgbm"

    def __init__(self, random_state: int = 42, params: Optional[Dict] = None) -> None:
        if lgb is None:
            raise ImportError("lightgbm is not installed.")
        super().__init__(random_state=random_state, params=params)

    @staticmethod
    def is_available() -> bool:
        return lgb is not None

    def get_default_params(self, random_state: int) -> Dict:
        return {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": random_state,
        }

    def build_model(self):
        return lgb.LGBMRegressor(**self.params)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """LightGBM 支持显式验证集，因此这里把它传给底层模型。"""

        fit_kwargs = {}
        if X_valid is not None and y_valid is not None:
            fit_kwargs["eval_set"] = [(X_valid, y_valid)]
            fit_kwargs["eval_metric"] = "l2"

        self.model.fit(X_train, y_train, **fit_kwargs)

        if X_valid is not None and y_valid is not None:
            valid_pred = self.predict(X_valid)
            return calculate_regression_metrics(y_valid, valid_pred)
        return {}

    def _extract_raw_importance(self) -> np.ndarray:
        return np.asarray(self.model.feature_importances_, dtype=float)


class XGBoostModel(BaseRegressorWrapper):
    """XGBoost 回归模型封装。"""

    model_name = "xgboost"

    def __init__(self, random_state: int = 42, params: Optional[Dict] = None) -> None:
        if xgb is None:
            raise ImportError("xgboost is not installed.")
        super().__init__(random_state=random_state, params=params)

    @staticmethod
    def is_available() -> bool:
        return xgb is not None

    def get_default_params(self, random_state: int) -> Dict:
        return {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:squarederror",
            "random_state": random_state,
        }

    def build_model(self):
        return xgb.XGBRegressor(**self.params)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """XGBoost 也支持显式验证集。"""

        fit_kwargs = {}
        if X_valid is not None and y_valid is not None:
            fit_kwargs["eval_set"] = [(X_valid, y_valid)]
            fit_kwargs["verbose"] = False

        self.model.fit(X_train, y_train, **fit_kwargs)

        if X_valid is not None and y_valid is not None:
            valid_pred = self.predict(X_valid)
            return calculate_regression_metrics(y_valid, valid_pred)
        return {}

    def _extract_raw_importance(self) -> np.ndarray:
        return np.asarray(self.model.feature_importances_, dtype=float)


class RandomForestModel(BaseRegressorWrapper):
    """随机森林回归模型。

    这个模型通常比提升树更慢一些，也未必更强，
    但它是一个很常见、很直观的树模型 baseline。
    """

    model_name = "random_forest"

    def get_default_params(self, random_state: int) -> Dict:
        return {
            "n_estimators": 300,
            "max_depth": 8,
            "min_samples_leaf": 5,
            "n_jobs": -1,
            "random_state": random_state,
        }

    def build_model(self):
        return RandomForestRegressor(**self.params)

    def _extract_raw_importance(self) -> np.ndarray:
        return np.asarray(self.model.feature_importances_, dtype=float)


class ExtraTreesModel(BaseRegressorWrapper):
    """Extra Trees 回归模型。

    这个模型和随机森林很接近，但随机性更强一些。
    对噪声较大的表格特征任务来说，它常常是一个值得尝试的树模型补充项。
    """

    model_name = "extra_trees"

    def get_default_params(self, random_state: int) -> Dict:
        return {
            "n_estimators": 400,
            "max_depth": 10,
            "min_samples_leaf": 3,
            "n_jobs": -1,
            "random_state": random_state,
        }

    def build_model(self):
        return ExtraTreesRegressor(**self.params)

    def _extract_raw_importance(self) -> np.ndarray:
        return np.asarray(self.model.feature_importances_, dtype=float)


class RidgeModel(BaseRegressorWrapper):
    """Ridge 线性回归 baseline。

    线性模型的主要价值包括：

    - 结构简单，容易理解；
    - 是判断“复杂树模型是否真的有额外收益”的好 baseline；
    - 系数可解释性通常更强。
    """

    model_name = "ridge"

    def get_default_params(self, random_state: int) -> Dict:
        return {
            "alpha": 1.0,
        }

    def build_model(self):
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", Ridge(**self.params)),
            ]
        )

    def _extract_raw_importance(self) -> np.ndarray:
        coefficients = np.asarray(self.model.named_steps["model"].coef_, dtype=float)
        return np.abs(coefficients)


class LassoModel(BaseRegressorWrapper):
    """Lasso 线性回归 baseline。"""

    model_name = "lasso"

    def get_default_params(self, random_state: int) -> Dict:
        return {
            "alpha": 0.0005,
            "max_iter": 5000,
            "random_state": random_state,
        }

    def build_model(self):
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", Lasso(**self.params)),
            ]
        )

    def _extract_raw_importance(self) -> np.ndarray:
        coefficients = np.asarray(self.model.named_steps["model"].coef_, dtype=float)
        return np.abs(coefficients)


class ElasticNetModel(BaseRegressorWrapper):
    """ElasticNet 线性模型。

    你可以把它理解成 Ridge 和 Lasso 的折中：

    - 它既保留了部分稀疏选择能力；
    - 又比纯 Lasso 对共线特征更稳一些。

    对很多相关因子并存的量化特征场景，它通常是一个很自然的线性增强版 baseline。
    """

    model_name = "elastic_net"

    def get_default_params(self, random_state: int) -> Dict:
        return {
            "alpha": 0.0008,
            "l1_ratio": 0.5,
            "max_iter": 5000,
            "random_state": random_state,
        }

    def build_model(self):
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", ElasticNet(**self.params)),
            ]
        )

    def _extract_raw_importance(self) -> np.ndarray:
        coefficients = np.asarray(self.model.named_steps["model"].coef_, dtype=float)
        return np.abs(coefficients)


MODEL_REGISTRY = {
    "lightgbm": LightGBMModel,
    "xgboost": XGBoostModel,
    "random_forest": RandomForestModel,
    "extra_trees": ExtraTreesModel,
    "ridge": RidgeModel,
    "lasso": LassoModel,
    "elastic_net": ElasticNetModel,
}


def build_model(model_name: str, random_state: int = 42, params: Optional[Dict] = None):
    """根据模型名称创建一个模型实例。"""

    normalized_name = model_name.strip().lower()
    if normalized_name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model name: {model_name}")

    model_class = MODEL_REGISTRY[normalized_name]
    if not model_class.is_available():
        raise ImportError(f"Model '{model_name}' is not available in the current environment.")
    return model_class(random_state=random_state, params=params)


def list_available_models() -> list[str]:
    """列出当前环境中真正可用的模型名字。"""

    available_models = []
    for model_name, model_class in MODEL_REGISTRY.items():
        if model_class.is_available():
            available_models.append(model_name)
    return available_models


def list_supported_models() -> list[str]:
    """列出代码层面支持的全部模型名字。"""

    return list(MODEL_REGISTRY.keys())


def normalize_feature_importance(
    importance_df: pd.DataFrame,
    model_weight: float,
) -> pd.DataFrame:
    """对单个模型的重要性做归一化，再乘以该模型的集成权重。

    这样做的目的有两个：

    1. 不同模型的重要性数值量纲不一样，先归一化后更容易融合；
    2. 如果某个模型在 walk-forward 验证里表现更好，它的特征重要性也应该占更大权重。
    """

    normalized_df = importance_df.copy()
    absolute_importance = normalized_df["importance"].abs()
    total_importance = float(absolute_importance.sum())

    if total_importance <= 0:
        normalized_df["normalized_importance"] = 0.0
    else:
        normalized_df["normalized_importance"] = absolute_importance / total_importance

    normalized_df["model_weight"] = float(model_weight)
    normalized_df["weighted_importance"] = normalized_df["normalized_importance"] * float(model_weight)
    return normalized_df


class ModelEnsemble:
    """加权平均集成模型。

    和之前的简单平均相比，这个版本多了一层“模型权重”：

    - 每个模型都可以带一个权重；
    - 预测时按权重做加权平均；
    - 如果所有权重都没设置好，就退回等权平均。
    """

    def __init__(self) -> None:
        self.models: list[tuple[str, object, float]] = []

    def add_model(self, model_name: str, model_wrapper, weight: float = 1.0) -> None:
        """向集成器中添加一个已训练模型及其权重。"""

        self.models.append((model_name, model_wrapper, float(weight)))

    def get_weights(self) -> pd.DataFrame:
        """返回当前集成器中的模型权重表。"""

        if not self.models:
            return pd.DataFrame(columns=["model", "weight"])

        return pd.DataFrame(
            [{"model": model_name, "weight": weight} for model_name, _, weight in self.models]
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """对多个模型预测结果做加权平均。"""

        if not self.models:
            raise ValueError("No models have been added to the ensemble.")

        predictions = []
        weights = []

        for _, model_wrapper, weight in self.models:
            predictions.append(model_wrapper.predict(X))
            candidate_weight = float(weight)
            weights.append(candidate_weight if np.isfinite(candidate_weight) and candidate_weight > 0.0 else 0.0)

        stacked_predictions = np.column_stack(predictions)
        weight_array = np.asarray(weights, dtype=float)

        if weight_array.sum() <= 0:
            weight_array = np.ones_like(weight_array, dtype=float)

        normalized_weights = weight_array / weight_array.sum()
        return np.average(stacked_predictions, axis=1, weights=normalized_weights)
