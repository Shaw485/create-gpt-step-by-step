# Create GPT Step by Step

这是一个从字符级 Tokenizer、Bigram、Self-Attention 开始，逐步手写 Decoder-only GPT，并完成 BPE 预训练与监督微调（SFT）的教学项目。

项目使用已获授权的《斗破苍穹》作为固定实验语料。目标是理解 GPT 的完整训练链路，并保留可复现的 Loss、样本、评估、Checkpoint 和视频材料；它不是通用聊天模型。

## 当前状态

- 已完成字符级 Tokenizer、训练批次、Bigram 和 Transformer 基础结构。
- 已完成字符级模型 10,000 步预训练及多轮 SFT 对照。
- 已完成手写 BPE、812 万参数模型 10,000 步预训练及 BPE SFT。
- 已完成正式 v4 语料审核、按章节 train/validation/test 切分、每章 EOS 和 BPE 张量生成。
- 已在 Apple M4/MPS 上把 810 万参数模型训练到累计 6,000 步；最佳验证 Loss 为 4.7759，独立 Test Loss 为 4.7476。
- 已建立小说续写评测 Harness：验证 Loss 负责筛选，重复退化、中文比例、训练集复现和固定提示词负责自动否决，语义连贯性仍由人工复核；Step 6000 发布候选因连续重复而被标记为 `REVIEW`。
- 已生成并审计3000条v4教师SFT候选，但当前状态仍为`needs_review`：600条val/test尚未真人批准，训练集的证据与语义风险也需处理。验收通过后才编码正式SFT张量并开始20步安全试跑。

完整路线见 [ROADMAP.md](ROADMAP.md)，当前任务见 [TODO.md](TODO.md)，实验材料入口见 [VIDEO_MATERIALS.md](VIDEO_MATERIALS.md)。

## 目标模型

| 参数 | 正式方案 |
|---|---:|
| 参数总量 | 8,105,025 |
| BPE 词表 | 6,465 |
| Transformer 层数 | 8 |
| Embedding | 256 |
| 注意力头 | 8 |
| 上下文 | 512 Token |
| Micro Batch / 梯度累积 | 2 / 4 |
| 权重共享 | Token Embedding 与输出层共享 |

从零训练阶段见 [M006 里程碑](reports/milestones/006_v4_local_pretrain/README.md)，低学习率续训与全局评测见 [M007 里程碑](reports/milestones/007_v4_continue6000/README.md)。云端扩展备选见 [CLOUD_TRAINING_PLAN.md](CLOUD_TRAINING_PLAN.md)。
模型用途、评测和发布边界见 [MODEL_CARD.md](MODEL_CARD.md)。

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

训练采用分模块轮转 JSONL 日志。数据、预训练、验证、Checkpoint、GPU、SFT 和编排日志能够分别调节级别；默认不记录密码、访问令牌、密钥和完整授权头。首阶段日志位于 `runs/pretrain_v4_m4/logs/`，续训日志位于 `runs/pretrain_v4_m4_continue6000/logs/`；配置分别见 `configs/local_m4_8m.json` 和 `configs/local_m4_8m_continue_6000.json`。M008 SFT数据修复日志位于被Git忽略的`data/sft/v4_teacher_repair/logs/`，数据、构建和验证可分别通过环境变量调节日志级别。

## 重要说明

一本小说能够训练领域续写与有限问答实验模型，但不能凭反复训练获得天气、编程、数学等书外知识。SFT 能改变回答格式和行为，不能替代缺失的预训练知识。
