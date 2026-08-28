# M007：v4 低学习率续训至 6000 步与全局评测 Harness

## 为什么继续训练

M006 在 Step 2600 得到最低验证 Loss 5.0447，但固定小说续写仍不够流畅。本阶段不扩大模型，先从这个验证集选出的权重继续训练，判断当前 810 万参数结构是否还能从同一语料中学习。

## 参数

| 项目 | 值 |
|---|---:|
| 参数量 | 8,105,025 |
| Embedding / Blocks / Heads | 256 / 8 / 8 |
| FFN / Context | 1,024 / 512 Token |
| 初始化权重 | M006 Step 2600 best |
| 优化器 | 新建 AdamW，不继承旧动量 |
| 学习率 | 5e-5 cosine 降至 1e-5 |
| Weight Decay / Betas | 0.1 / 0.9, 0.95 |
| Micro Batch / 梯度累积 | 2 / 4 |
| 每个优化步训练 Token | 4,096 |
| 续训范围 | Step 2601–6000 |
| 设备 / 精度 | Apple M4 MPS / float32 |

第二阶段耗时 2,111.8 秒（约 35 分 12 秒），从零阶段加续训累计 4,057.5 秒（约 67 分 38 秒）。累计处理 24,576,000 个训练 Token，约为 3,416,400 个训练 Token 的 7.19 遍。

## Loss 结果

| 指标 | M006 | M007 |
|---|---:|---:|
| 所选 Step | 2,600 | 6,000 |
| 最佳 Validation Loss | 5.0447 | **4.7759** |
| 独立 Test Loss | 4.9958 | **4.7476** |

验证 Loss 相对 M006 降低 0.2687，即 5.33%。Test 只在选模完成后用固定随机种子的 20 个 Batch 评估，没有参与模型选择。

## Harness 如何控制全局指标

Harness 不是新的训练 Loss，也不直接修改梯度。它位于“训练之后、选模之时”，采用三层决策：

1. Validation Loss 先筛选候选 checkpoint。
2. 固定 5 个小说提示词检查长度、中文比例、4-gram 重复、最长单字连写和训练集逐字复现；任一硬门槛失败就标记为 `REVIEW`，不能自动晋级。
3. 通过机械门槛的样本仍需人工判断语法、人物一致性、情节连贯性和整体可读性。

Step 2600 因出现连续 10 个“试”而被自动否决；Step 5000 因连续 15 个“丝”而被否决；独立重新加载的 Step 6000 发布候选也出现连续 12 个“试”，因此状态是 `REVIEW`，没有因 Loss 最低而自动晋级。

此外，Step 6000 的人工抽样仍有明显语法错误、角色关系混乱和情节跳跃。因此准确结论是：**低学习率续训改善了 Loss，但综合质量没有通过 Harness，尚不能证明小说生成已成熟。**

## 复现与调试

- 实际配置：`effective_config.json`
- 完整 Loss：`pretrain_v4_loss.csv`
- 曲线：`pretrain_v4_loss_curve.png` / `pretrain_v4_loss_curve.svg`
- 所选模型独立评估：`selected_model_evaluation.json`
- 40 组固定小说样本：`story_harness_samples.md`
- 自动门槛汇总：`story_harness_summary.csv` / `story_harness_report.json`
- 人工复核表：`manual_review_template.csv`
- 文件与 checkpoint 校验：`SHA256SUMS.md`

训练日志位于 `runs/pretrain_v4_m4_continue6000/logs/`，Harness 日志位于 `reports/story_harness_v4/logs/`。日志按数据、训练、验证、checkpoint、设备和编排模块独立轮转；模块等级在配置中单独控制，默认不记录密码、Token、密钥或完整授权头。

## 下一步

先填写人工复核表，不因 Loss 下降而盲目追加训练。若 Step 6000 的人工评分优于较早候选，就冻结它作为 v4 SFT 初始化点；SFT 仍需使用独立的问答/指令评测 Harness，不能沿用小说续写指标代替回答质量。
