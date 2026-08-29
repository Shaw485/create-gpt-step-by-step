# Create GPT Step by Step

这是一个从字符级 Tokenizer、Bigram、Self-Attention 开始，逐步手写 Decoder-only GPT，并完成 BPE 预训练与监督微调（SFT）的教学项目。

项目使用已获授权的《斗破苍穹》作为固定实验语料。目标是理解 GPT 的完整训练链路，并保留可复现的 Loss、样本、评估、Checkpoint 和视频材料；它不是通用聊天模型。

## 当前状态

- 已完成字符级 Tokenizer、训练批次、Bigram 和 Transformer 基础结构。
- 已完成字符级模型 10,000 步预训练及多轮 SFT 对照。
- 已归档手写 BPE、810 万参数 v4 预训练和 M013 等早期 SFT 对照；这些是历史实验，不再代表当前正式模型口径。
- 当前正式结构为 14,880,745 参数、BPE 词表 7,465、Context 512、10 层、Embedding 320、8 头；M019 已确认纯预训练 Step 5750 是当前 SFT 基座。
- M020 已构建并验证 10,000 条小说垂直 SFT v7，切分为 8,000 train / 800 val / 600 public / 600 sealed；训练与 public 张量物理隔离，sealed 保持冻结。
- Step 5750 在新 public 上的基线 Loss 为 5.86808865，EOS 7.1667%、机械重复 85.5%；冻结 16 题为 0/16 EOS、16/16 达到长度上限，证明纯预训练基座尚不会稳定聊天。
- M020 的 20 步安全试跑和 2,000 步正式 SFT 已完成；Validation Loss 从 5.897454 降至 3.356181，public/sealed 训练消耗均为 0。
- Public Loss 从纯预训练基线的 5.868089 降至 Step 2000 的 3.294267，EOS 从 7.17% 升至 82.33%，但机械重复仍为 42.17%；固定 16 题在 Step 1500 也只有 1 题 AI 辅助基本通过。
- 汇总状态为 `automatic_gates_failed_external_review_pending`，严格候选为空。Step 1000 只作为保守诊断候选，Step 1500 只作为行为较优研究候选，Step 2000 已排除；当前没有发布模型，也没有启封 sealed test。

完整路线见 [ROADMAP.md](ROADMAP.md)，当前任务见 [TODO.md](TODO.md)，实验材料入口见 [VIDEO_MATERIALS.md](VIDEO_MATERIALS.md)。

## 目标模型

| 参数 | 正式方案 |
|---|---:|
| 参数总量 | 14,880,745 |
| BPE 词表 | 7,465（含 6 个特殊 Token） |
| Transformer 层数 | 10 |
| Embedding | 320 |
| 注意力头 | 8 |
| 上下文 | 512 Token |
| FFN | 1,280（Embedding 的 4 倍） |
| 权重共享 | Token Embedding 与输出层共享 |

当前模型容量实测见 [M015](reports/milestones/015_scaling_stage_a/README.md)，正式预训练见 [M016](reports/milestones/016_formal_pretrain_14m/README.md)，纯预训练验收见 [M019](reports/milestones/019_pretrain_capability_audit/README.md)，当前小说垂直 SFT 见 [M020](reports/milestones/020_sft_v7_vertical/README.md)。810 万参数 v4 与 M013 等历史路线仍保留在各自里程碑中用于教学对照。云端扩展备选见 [CLOUD_TRAINING_PLAN.md](CLOUD_TRAINING_PLAN.md)。
模型用途、评测和发布边界见 [MODEL_CARD.md](MODEL_CARD.md)。

M020 的同口径节点对比见 [checkpoint_comparison.md](reports/milestones/020_sft_v7_vertical/checkpoint_comparison.md)，Loss 曲线见 [sft_v7_loss_curve.svg](reports/milestones/020_sft_v7_vertical/sft_v7_loss_curve.svg)，固定 16 题 AI 辅助复核见 [fixed_samples_milestone_review.md](reports/milestones/020_sft_v7_vertical/fixed_samples_milestone_review.md)。

## 本地验证

```bash
git clone https://github.com/Shaw485/create-gpt-step-by-step.git
cd create-gpt-step-by-step
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

测试不会启动正式训练，主要验证 Tokenizer、数据隔离、Loss、因果注意力、模型参数和 Checkpoint 行为。

## 数据和模型发布边界

- 原始小说、清洗后全文和可还原全文的 Token 张量不进入 Git 历史。
- 训练数据与公开教学材料分离，正式云端产物放在被忽略的 `data/cloud_v4/`。
- 模型允许开源；选定的 best 权重应通过 GitHub Release 或 Git LFS 发布，不把大量中间 Checkpoint 提交进源码历史。
- 模型卡必须注明数据来源、预期用途、能力边界、评测结果和已知风险。
- 代码和模型的最终许可证仍需仓库所有者在正式发布前确定。

## 日志与调试

训练采用分模块轮转 JSONL 日志。数据、预训练、验证、Checkpoint、GPU、SFT 和编排日志能够分别调节级别；默认不记录密码、访问令牌、密钥和完整授权头。首阶段日志位于 `runs/pretrain_v4_m4/logs/`，续训日志位于 `runs/pretrain_v4_m4_continue6000/logs/`；配置分别见 `configs/local_m4_8m.json` 和 `configs/local_m4_8m_continue_6000.json`。M008 SFT数据修复日志位于被Git忽略的`data/sft/v4_teacher_repair/logs/`；M013构建、训练、评估和checkpoint/数据兼容性日志位于`reports/milestones/013_v5_1_no_math_sft/logs/`（被Git忽略）。审核工具使用`review_logs/`分别记录UI、数据保存和验证事件，不记录审核人或审核正文。

M020 继续按构建、验证、编码、训练、生成、公开评估、checkpoint 和编排分模块记录轮转 JSONL；每个模块可独立设为 `OFF/INFO/DEBUG`。默认日志不写小说正文、完整提示、Token 或敏感凭证，汇总产物集中在 `reports/milestones/020_sft_v7_vertical/`。

M008本地真人审核页面可用以下命令启动，只监听`127.0.0.1:8765`：

```bash
.venv/bin/python review_sft_v4.py
```

## 重要说明

一本小说能够训练领域续写与有限问答实验模型，但不能凭反复训练获得天气、编程、数学等书外知识。SFT 能改变回答格式和行为，不能替代缺失的预训练知识。
