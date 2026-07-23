# FMP 基本面数据接入说明

这个文件只回答三个问题：

1. 现在项目里哪些基本面字段会被识别？
2. FMP 数据怎么先下载下来？
3. 下载后的季度数据如何安全地 merge 到日频价格表？

## 1. 当前会被模型识别的基本面字段

当前项目在 [`src/feature_generator.py`](../../src/feature_generator.py) 里会自动识别这些列：

- `eps`
- `pe`
- `pb`
- `ps`
- `roe`
- `roa`
- `yoy`
- `qoq`

只要这些列真实存在于日频训练 CSV 中，特征工程就会自动把它们纳入候选特征，并额外构造横截面排名特征。

## 2. FMP 下载入口

新增入口：

- [`main_fmp_fundamentals.py`](../../main_fmp_fundamentals.py)

核心模块：

- [`src/fmp_fundamentals.py`](../../src/fmp_fundamentals.py)

运行前需要准备：

- FMP API key
- 环境变量 `FMP_API_KEY`

示例：

```bash
export FMP_API_KEY="your_api_key_here"
python3 main_fmp_fundamentals.py \
  --daily-data-path data/us_large_cap_300_daily.csv \
  --fundamentals-output-path data/fmp/fundamentals_quarterly.csv \
  --merged-output-path data/us_large_cap_300_with_fundamentals.csv
```

## 免费计划的现实限制

你的 FMP 免费计划当前有两个关键限制：

- 每日 API 调用上限约 `250`
- 季度 `ratios` / `key-metrics` 接口会返回 `402`
- `income-statement` 和 `balance-sheet-statement` 可以正常使用
- `limit` 不能大于 `5`

因此当前实现已经调整成：

- 只用免费可用的两张财报表
- 自己推导 `roe / roa / yoy / qoq`
- 在 merge 后再计算 `pe / pb / ps`

但这也意味着：

- 300 只股票一次性完整下载仍然不现实
- 因为 `300 * 2 = 600` 次请求，超过你的免费额度

更实际的做法是分批下载，例如每天 100 只左右。

当前实测还有一个限制：

- 免费计划下，即使没有超过每日请求数，部分 symbol 的 `income-statement` 仍会返回 `402 Payment Required`
- 因此 `us100` 测试里实际只有 `35` 只股票拿到了可用季度基本面
- `skipped_symbols.csv` 只用于记录失败 symbol 和状态码；代码已经把 URL 里的 `apikey` 替换成 `<redacted>`，避免密钥落盘

## 3. merge 为什么必须按披露日期

最关键的原则只有一句：

- 某个交易日只能看到“当天之前已经公开披露”的财报信息

所以 merge 时不能按：

- 财报所属季度末日期

而应该优先按：

- `acceptedDate`
- 其次 `filingDate / fillingDate`
- 最后才退回 `report_date + 90 个日历日`

当前实现按这个顺序生成 `effective_date`，再对每只股票做 backward `merge_asof`。如果收入表和资产负债表的可用日期不同，合并行使用其中最晚的日期，避免较晚披露字段提前出现。90 天只是披露时间戳缺失时的保守近似，正式研究仍应优先使用真实 `acceptedDate` / `filingDate`。

FMP 历史接口可能返回后续重述后的旧财季数值。按披露日期合并只能控制“什么时候开始使用一行数据”，无法恢复每个历史时点当时看到的原始版本。因此这里属于 point-in-time-style 受控实验；需要严格 as-reported 数据时必须换用保留 filing vintage 的数据源。

这一步的目标是避免未来信息泄露，字段数量并非验收标准。

## 4. 当前字段定义

当前版本里：

- `yoy`：定义为 `revenue_yoy`
- `qoq`：定义为 `revenue_qoq`
- `roe`：定义为 `net_income_ttm / avg_equity_4q`
- `roa`：定义为 `net_income_ttm / avg_assets_4q`
- `pe`：merge 到日频后按 `close / eps`
- `pb`：merge 到日频后按 `market_cap / total_equity`
- `ps`：merge 到日频后按 `market_cap / revenue_ttm`

原因很直接：

- 先用收入增长率做一个稳定、容易解释的版本
- 后面如果你更想要 `eps_yoy / eps_qoq`，再扩展也不难

## 5. 当前输出

脚本会生成两份文件：

- 季度基本面面板：`data/fmp/fundamentals_quarterly.csv`
- 合并后的日频数据：`data/us_large_cap_300_with_fundamentals.csv`

建议你先检查：

1. 基本面面板里每只股票是否有足够季度记录
2. 合并后的日频数据里是否真的出现了 `eps / pe / pb / ps / roe / roa / yoy / qoq`
3. 这些列是否在早期日期大量为空

如果前期空值很多，这是正常的，因为基本面只能从首次披露日期开始向后可见。

## 6. Context 接入验证

当前已经跑过一组低覆盖率 context 接入验证：

- context CSV：`data/us_large_cap_100_fmp_macro_context.csv`
- 模型输出：`outputs/context_integration_us100/10d_linear_context_check/training_report.md`
- 汇总报告：`outputs/context_integration_us100/context_integration_report.md`

该验证证明：

- FMP 基本面字段可以进入候选特征
- 宏观代理变量可以进入候选特征
- 经过新的 date-level 特征保留逻辑后，宏观变量不会再被横截面预处理抹掉
- 最终模型输入里包含 `27` 个宏观特征和 `23` 个基本面或基本面派生特征

但这不代表基本面已经在主线 `us300` 上被充分验证。
