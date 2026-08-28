# M010 v4 SFT 500步正式小跑

## 结论

本阶段从 M007 的 v4 最佳预训练 checkpoint 启动，对 2999 条 AI 审核后的 training-ready SFT 候选执行 500 步监督微调。当前 Codex 工具环境无法使用 MPS，因此本次记录为 CPU 可复现实验；用户本机终端可用 MPS 时，可用同一脚本加 `--device mps` 复跑。

500 步后，模型已经明显学会 SFT 的“短答 + 停止”格式，但事实准确性仍不稳定。最佳验证 Loss 出现在 step 400，因此本阶段不能简单选最后一步作为最佳模型。

## 训练配置

| 项目 | 值 |
|---|---:|
| 初始化 checkpoint | `runs/pretrain_v4_m4_continue6000/best.pt` |
| 数据 | `data/cloud_v4/sft_v4_ai_training_ready_tensors.pt` |
| 训练/验证/测试记录数 | 2399 / 300 / 300 |
| 模型参数量 | 8,105,025 |
| Context | 512 |
| Micro Batch | 2 |
| 学习率 | 5e-5 |
| Weight Decay | 0.05 |
| 梯度裁剪 | 1.0 |
| 训练步数 | 500 |
| 评估间隔 | 100 |
| 固定样本间隔 | 100 |
| 测试集是否参与 | 0条，未使用 |

## Loss

| Step | Train Loss | Val Loss |
|---:|---:|---:|
| 0 | 6.8907 | 6.7152 |
| 100 | 2.7259 | 2.8780 |
| 200 | 2.5167 | 2.9589 |
| 300 | 2.3654 | 2.4184 |
| 400 | 1.4536 | 1.3442 |
| 500 | 1.8302 | 1.5823 |

最佳验证 Loss：`1.3442 @ step 400`。最终验证 Loss：`1.5823 @ step 500`。

## 固定10题样本观察

| # | split | 类型 | 问题摘要 | 期望 | Step 500输出 | EOS |
|---:|---|---|---|---|---|---|
| 1 | train | relationship_reason_timeline | 第351章证据中药尘和雷尊者谁出现 | 当前证据片段明确出现了药尘。 | 当前证据片段明确出现了萧战。 | 是 |
| 2 | train | relationship_reason_timeline | 第54章萧炎为什么买七份材料 | 因为此次急需钱财。 | 第86章的标题是《破！》第6章的标题是《斗破苍穹》。 | 是 |
| 3 | train | ambiguity_unknown_clarification | 第4章能否得知纳兰桀具体年龄 | 不能；片段提到了纳兰桀，但没有直接给出其当前的具体年龄。 | 能确认；片段同时提到了海波东和萧鼎。 | 是 |
| 4 | train | context_understanding | 第64章证据片段明确提到谁 | 当前证据片段明确提到了药老。 | 当前证据片段明确提到了萧炎。 | 是 |
| 5 | train | context_understanding | 片段中提到的功法/斗技 | 能确认其中提到了天火三玄变。 | 能确认其中提到了其中提到了天火三玄变。 | 是 |
| 6 | val | ambiguity_unknown_clarification | 用户只说介绍相关人物时如何追问 | 请先说明作品名称、具体人物和故事阶段…… | 请先说明你指的是哪部作品，以及想了解萧炎的哪个故事阶段…… | 否 |
| 7 | val | direct_fact | 小说第三百章标题 | 第300章的标题是《收场》。 | 第93续章的标题是《雷尊者》。 | 是 |
| 8 | val | ambiguity_unknown_clarification | 第1046章能否得知叶重身高 | 不能；片段提到了叶重，但没有直接给出其身高。 | 能确出现了连续字符“前证据片段中出现了连续字符“丹塔”。 | 是 |
| 9 | val | relationship_reason_timeline | 第836章韩枫和紫研是否都被提到 | 是，证据片段同时提到了韩枫和紫研。 | 正确。 | 是 |
| 10 | val | ambiguity_unknown_clarification | 用户只说介绍萧炎时如何追问 | 请先说明你指的是哪部作品，以及想了解萧炎的哪个故事阶段…… | 请先说明你指的是哪个故事阶段，我再核对与“第一千五百八十五章的原因是什么 | 否 |

## 判断

- SFT链路通过：loss大幅下降，checkpoint、日志、报告、曲线和固定样本都已生成。
- 格式能力开始出现：10个样本中8个触发EOS停止，回答长度明显短于预训练续写。
- 内容能力未达标：人物、章节标题、局部证据判断仍经常答错。
- 模型选择应优先考虑 step 400 的最佳验证 Loss，而不是 step 500 最后模型。

## 核心材料

- `sft_v4_step500_report.json`：完整机器可读报告。
- `sft_v4_step500_report_loss.csv`：Loss数据表。
- `sft_v4_step500_report_loss_curve.png` / `.svg`：Loss曲线。
- `novel_vs_general_samples.md`：小说问题与非小说问题的固定样本对比。
- `runs/sft_v4_step500/best.pt`：本阶段最佳验证checkpoint，本机保留，不提交GitHub。
- `runs/sft_v4_step500/latest.pt`：本阶段最后checkpoint，本机保留，不提交GitHub。
- `runs/sft_v4_step500/logs/`：data、sft、validation、checkpoint、orchestrator等独立JSONL日志。

## 下一步

继续正式SFT时建议从同一预训练 checkpoint 重新启动，先把本阶段的 `max_steps` 扩到 2000～3000，并保持同一批固定10题、同一采样参数和同一验证集。若验证 Loss 继续下降且样本事实正确率改善，再进入保留测试集的一次性评估。
