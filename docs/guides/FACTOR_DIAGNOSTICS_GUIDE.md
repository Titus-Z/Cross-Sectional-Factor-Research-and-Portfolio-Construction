# Factor Diagnostics Guide

这份文档只回答 3 个问题：

1. 因子诊断是什么
2. 这个项目里会输出什么
3. 你现在最应该怎么看这些结果

---

## 1. 因子诊断是什么

训练报告回答的是：

- 整个模型有没有用

因子诊断回答的是：

- 单个特征到底有没有用
- 它的作用是稳定的，还是偶然的
- 它更像排序信号，还是只是噪声

所以因子诊断用于拆开检查：

- `技术指标`
- `Alpha191`
- `原始量价特征`

到底谁在提供真实信号。

---

## 2. 这个脚本怎么做

入口：

- [`main_factor_diagnostics.py`](../../main_factor_diagnostics.py)

它会：

1. 读取真实数据
2. 激活指定目标周期，例如 `10d`
3. 按和训练实验相同的时间切分重建 OOS 特征
4. 复用横截面预处理逻辑
5. 读取已经训练好的实验目录里的特征列表
6. 在 OOS 上对每个特征分别计算：
   - 日度 Pearson IC
   - 日度 Spearman IC
   - IC mean / IC std / ICIR
   - 分组收益
   - Top-Bottom spread
   - 分组单调性

---

## 3. 默认推荐口径

当前默认口径是：

- 数据：`us_large_cap_300`
- 目标：`10d`
- 特征来源：`10d_linear_models` 的已选特征

原因很简单：

- 这是你当前最强、也最稳定的实验版本
- 用已经选中的特征做诊断，比一开始就诊断全部 200+ 候选列更聚焦

---

## 4. 主要输出文件

默认输出目录类似：

- `outputs/factor_diagnostics/10d_linear_models_us300/`

里面最重要的是：

- `factor_ic_summary.csv`
  - 每个特征的 OOS 诊断摘要
- `factor_daily_ic.csv`
  - 每个特征每天的 IC 明细
- `factor_group_returns.csv`
  - 每个特征、每个日期、每个分组的平均未来收益
- `factor_average_group_returns.csv`
  - 每个特征按分组聚合后的平均收益
- `factor_report.md`
  - 适合直接阅读的 Markdown 报告
- `stage_timing.csv`
  - 因子诊断本身的耗时拆解

---

## 5. 你现在最该看什么

先看 `factor_ic_summary.csv` 里的这些列：

- `pearson_ic_mean`
- `spearman_ic_mean`
- `pearson_ic_ir`
- `long_short_spread`
- `group_monotonic_spearman`

优先关注满足下面条件的特征：

- `pearson_ic_mean` 为正
- `spearman_ic_mean` 也为正
- `long_short_spread` 为正
- `group_monotonic_spearman` 接近 `1`

这类特征更像“稳定排序信号”。

如果一个特征：

- `IC` 还行
- 但 `long_short_spread` 很弱

那它可能只在中间排序上有一点信息，但不适合直接拿来做极端分组。

如果一个特征：

- `selector_score` 高
- `model_importance` 也高
- OOS 单因子诊断仍然靠前

那它就是当前最值得重点理解的核心特征。

---

## 6. 建议的第一轮阅读顺序

1. 先读 `factor_report.md`
2. 再看 `factor_ic_summary.csv` 前 20 行
3. 再看最靠前 3 到 5 个特征的分组收益
4. 最后再回头对照训练时的：
   - `selected_feature_scores.csv`
   - `feature_importance.csv`

这样你能回答一个很重要的问题：

- 模型选中的特征，是否在 OOS 单因子层面也站得住
