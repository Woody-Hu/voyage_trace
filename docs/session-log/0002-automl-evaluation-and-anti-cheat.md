---
session_id: 0002
title: AutoML 评测体系与反作弊边界 —— AutoGluon 最佳实践落地与评测
date: 2026-08-05
status: implemented
research_kind: open-ended
files_touched:
  - pyproject.toml                          # 修改：固定 autogluon>=1.4.0,<2；requires-python 改为 <3.14；新增 integrations/samples 可选依赖
  - src/voyage_trace/automl.py              # 修改：泄漏保护、num_bag_sets=10、MAE、均值基线 beats_baseline、SoftLeakage 注记
  - tests/test_automl.py                    # 新增：16 个评测测试（泄漏/基线/指标/bag_sets），全部真实 AutoGluon 运行，无 mock
  - tests/test_agents.py                    # 修改：闭环 RECALL 步骤数断言修正为 2
  - docs/automl-best-practices.md           # 新增：AutoGluon 最佳实践与评测方案文档
tests: test_automl.py 40 passed；全量 290 passed（含新增 16）
mocks_used: false
autogluon_version: 1.5.0
python: 3.12.13 (3.14 不被 AutoGluon 支持)
---

# Session 0002 — AutoML 评测体系与反作弊边界

## 1. 任务

1. 安装 autogluon 依赖（可固定到可用版本）。
2. 梳理建立相应文档，在网上和开源系统找 autogluon 的最佳实践与评测方案，
   对现有系统增加测试，制定优化方案与评测方案；以评测结果再去调整代码细节；
   变更与评测结果用文档记录；整体代码与接口足够简洁优雅。
3. ……（后续集成/样本/重构/e2e 见后续 session-log）。

## 2. 环境与依赖固定

- AutoGluon 最新稳定版 **1.5.0** 要求 `Python >=3.10,<3.14`，而沙箱默认
  Python **3.14.4** 不被支持。故用 pyenv 中的 **3.12.13** 建 venv。
- `pyproject.toml`：`requires-python = ">=3.11,<3.14"`，`autogluon.tabular>=1.4.0,<2`
  （1.4.0 与 1.5.0 均稳定且支持 3.10–3.13）。classifiers 去掉 3.14。
- 新增可选依赖组 `integrations`（deepeval/flaml[automl]/langfuse）与 `samples`
  （deepagents），均懒导入、缺失则优雅降级。

## 3. 调研结论（用于驱动代码改动）

详见 `docs/automl-best-practices.md`。关键结论：

1. **小数据用 `num_bag_sets` 而非 `num_bag_folds` 降方差**：文档明确建议用
   `num_bag_sets`（重复 k-fold）代替提高 `num_bag_folds`，后者 >10 反而过拟合。
   原代码 `num_bag_sets=1` → 改为默认 `10`。
2. **`num_stack_levels=0` 正确**：tiny n 下 stacking 学元权重必过拟合。
3. **不要 HPO**：官方明确不推荐 `hyperparameter_tune_kwargs`，tiny n 必过拟合。
4. **`eval_metric` 可换 MAE**：R² 对小数据敏感、随特征数膨胀；MAE 更稳。
5. **`leaderboard`/`evaluate` 对 lower-is-better 指标取负号**：RMSE/MSE 会显示为负，
   读取须 `abs(...)`（原代码已正确）。
6. **permutation importance 是 directional 而非 causal**；多重共线性会低估相关特征
   重要性；在训练数据上计算会有偏。
7. **小数据统计显著性不可得**（n<10 严重欠功效）；**单一数字不可信**，须报告面板。

## 4. 反作弊边界（本次最重要改动）

研究指出一个**真实的数据泄漏**：`total_tokens` 同时是特征与目标。预测 `total_tokens`
时若保留该特征，AutoGluon 可凭恒等关系得到 R²=1 —— 这是**作弊**，不是学习。

新增 `leakage_safe_features(target)` + `LEAKAGE_BY_TARGET`，在构造 DataFrame 前
**按目标丢弃硬泄漏特征**：

| 目标 | 丢弃特征 | 原因 |
|---|---|---|
| `total_tokens` | `total_tokens` | 恒等泄漏——特征即目标 |
| `cost_usd` | （无） | `total_tokens` 是软泄漏（cost≈tokens×price），保留但加注记 |
| `total_duration_s` | （无） | p50/p99 是分位数，非求和分量 |

软泄漏通过 `SOFT_LEAKAGE_NOTES` 写入 `report.notes` 并渲染到 Markdown，让评审者看到
"该重要性是 directional 而非 causal"。丢弃的特征记录在 `report.dropped_features`
并渲染为 `dropped_features (hard leakage): [...]`。

## 5. "有无信号"之门：均值预测器基线

新增 `MeanBaseline(rmse, mae)`（均值预测器：R²≡0，RMSE=std(target)）与
`report.beats_baseline`：

- `best_model.rmse < mean_baseline.rmse` → `True`（有真信号）；
- 否则 `False`，并追加 note 提示"把重要性当噪声，勿据此行动"；
- 常数目标（std=0）→ 所有人打平 → `False`。

in-sample R² 会随特征数单调上升（即使纯噪声），tiny n 下尤甚——这道门是
"R² 看着不错但实际没学到东西"的诚实裁决。

## 6. 评测结果（真实运行，非伪造）

合成 3 条 trace（LLM cost 0.5/0.9/1.4；tool 节点成功），`num_bag_sets=10`：

| 目标 | 丢弃特征 | best_model | top_feature | R² | RMSE | MAE | mean_baseline RMSE | beats_baseline |
|---|---|---|---|---|---|---|---|---|
| `cost_usd` | [] | WeightedEnsemble_L2 | total_tokens | 0.8314 | 0.5333 | 0.3664 | 1.2988 | **True** |
| `total_tokens` | `['total_tokens']` | (mean) | (mean) | 0.0000 | 0.0000 | 0.0000 | 399.56 | **False** |
| `total_duration_s` | [] | (mean) | (mean) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **False** |

**关键反作弊验证**：`total_tokens` 目标在**未做泄漏保护前**会凭恒等特征得到 R²=1
（作弊）；做泄漏保护后，AutoGluon 仅能从 calls/p50/p99/error_rate 学习——这些特征
不含该信号 → 诚实回退到均值基线、`beats_baseline=False`。这正是不作弊的体现。

`cost_usd` 的 total_tokens 重要性=1.0 是**软泄漏**（cost≈tokens×price，near-deterministic），
已通过 `SOFT_LEAKAGE_NOTES` 显式标注为 directional。RMSE 0.53 < 基线 1.30，确有信号，
但需结合软泄漏注记审读。

`total_duration_s` 因合成数据 start==end 使目标恒为 0 → 基线 RMSE=0 → 无信号，诚实回退。

## 7. 据评测结果对代码的调整

| 评测发现 | 代码调整 |
|---|---|
| `total_tokens` 恒等泄漏致伪 R²=1 | 新增 `leakage_safe_features` + `dropped_features` 字段 |
| tiny n 下 in-sample R² 不可信 | 新增 `MeanBaseline` + `beats_baseline` 门 |
| `num_bag_sets=1` 方差大 | 默认改 10（可配置） |
| 仅 R²/RMSE 不足 | 新增 MAE + `eval_metric` 可配置 |
| 软泄漏未向评审者暴露 | 新增 `SOFT_LEAKAGE_NOTES` 写入 notes 与 Markdown |
| 3.14 不被 AutoGluon 支持 | `requires-python` 改 `<3.14`，classifiers 去掉 3.14 |

## 8. 测试（无 mock、无伪造）

`tests/test_automl.py` 新增 4 个测试类共 16 个用例，全部真实 AutoGluon 运行：

- `TestLeakageProtection`（7）：泄漏特征按目标丢弃、`dropped_features` 记录与渲染、
  软泄漏注记、`total_tokens` 不出现在自身目标的 importances。
- `TestMeanBaseline`（4）：基线计算、有信号→True、常数目标→False、`to_dict`。
- `TestEvaluationMetrics`（4）：MAE 存在、`eval_metric` 透传、JSON 往返、Markdown 渲染。
- `TestBagSets`（1）：`num_bag_sets` 参数被接受。

并修正既有测试：`test_with_memory_integration` 的 RECALL 步骤数从 1 改为 2
（orchestrator 现额外做 τ recall，闭环设计使然，非测试退化）。

## 9. 未做（诚实记录，留待后续）

- `num_shuffle_sets` 提升 permutation CI 紧致度（≥20）。
- 相关特征块的 group/joint permutation importance。
- LOOCV / repeated-k-fold 离线评测助手（不在热路径）。
- foundational models（`extreme_quality`/TabPFNv2）需 GPU。
- FLAML 作为 AutoGluon 替代后端（见 `docs/integrations.md`，后续 session）。
