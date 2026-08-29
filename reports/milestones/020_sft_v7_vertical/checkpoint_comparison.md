# M020 SFT v7 Checkpoint Comparison

- 汇总状态：`automatic_gates_failed_external_review_pending`
- 必需产物待补：`0`
- 严格候选：`none`
- 发布就绪：`false`（本工具不替代独立真人发布复核）
- 外部门禁口径：只有每项明确 `passed=true` 才计为通过；`pending` 不会提升为通过。
- 数据边界：本工具未打开 checkpoint、张量、原始语料或 sealed 正文。

## Checkpoint 对比

| Checkpoint | Step | Public Loss | Public 自动门 | Public 外部门 | Fixed EOS | Retention BPC 恶化 | Retention 自动门 | 汇总状态 |
|---|---:|---:|---|---|---:|---:|---|---|
| Pretrain baseline Step 5750 | 5750 | 5.868089 | 4/15 (failed) | pending | 0.0000 | pending | pending | automatic_public_gates_failed |
| SFT v7 smoke Step 20 | 20 | pending | pending | pending | 0.0625 | pending | pending | engineering_smoke_complete_public_not_run_optional |
| SFT v7 Step 500 | 500 | 4.034095 | 4/15 (failed) | pending | 0.5625 | 0.0601 | 3/4 (failed) | automatic_public_gates_failed |
| SFT v7 Step 1000 | 1000 | 3.611000 | 5/15 (failed) | pending | 0.6875 | 0.0900 | 3/4 (failed) | automatic_public_gates_failed |
| SFT v7 Step 1500 | 1500 | 3.419921 | 4/15 (failed) | pending | 0.6875 | 0.1022 | 2/4 (failed) | automatic_public_gates_failed |
| SFT v7 Step 2000 | 2000 | 3.294267 | 4/15 (failed) | pending | 0.4375 | 0.1208 | 2/4 (failed) | automatic_public_gates_failed |

## 待补产物

- 无必需文件缺失；外部评审仍按各 checkpoint 状态单独判断。

## 完整性问题

- 当前已加载报告之间未发现 checkpoint 身份冲突。

## Loss 曲线

- CSV：`reports/milestones/020_sft_v7_vertical/sft_v7_loss_curve.csv`
- SVG：`reports/milestones/020_sft_v7_vertical/sft_v7_loss_curve.svg`
- 缺失训练报告时 CSV 只保留表头，SVG 明确显示 pending；不会补零或插值。
