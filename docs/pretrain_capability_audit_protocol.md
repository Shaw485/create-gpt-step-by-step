# 单本小说预训练能力审计协议 v1

- 冻结日期：2026-08-29
- 适用模型：1488 万参数《斗破苍穹》基础语言模型
- 目的：在查看新审计结果前冻结评测口径，避免根据结果移动标准

## 审计问题

本审计只回答三个问题：

1. 与训练早期相比，模型是否学到了小说语言分布？
2. 高频与低频人物在验证原文前缀后的候选排序是否出现可测改善？
3. Step 5750 到 Step 6000 是否已经进入当前配置下的实用平台期？

本审计不测试聊天、通用百科、数学、实时信息、指令遵循或领域外拒答。这些不是单本小说预训练的能力门。

## 冻结输入

- 模型配置：`configs/formal_pretrain_14m_bpe3000.json`
- Tokenizer：`data/scaling_a/bpe_3000/tokenizer.json`
- 早期对照：`runs/formal_pretrain_14m_bpe3000_6000/checkpoints/step_00250.pt`
- 正式基座：`runs/formal_pretrain_14m_bpe3000_6000/checkpoints/step_05750.pt`
- 末步对照：`runs/formal_pretrain_14m_bpe3000_6000/checkpoints/step_06000.pt`
- 探针来源：正式 `train` 与 `val` 章节；不得读取 SFT 数据
- `test`：继续封存，不参与本轮构建、运行或选择

Step 250 和 Step 6000 只用于诊断对照，不自动改变 Step 5750 的正式基座身份。

## 指标与样本量

### 1. Validation 下一 Token 诊断

- 在 `val_tokens.pt` 上选取 60 个等距、可复算的 512 Token 窗口。
- 共评估 30,720 个目标 Token。
- 报告 Token Loss、Token Perplexity 与 Top-1 下一 Token 准确率。
- 同时引用训练历史中的 Train/Validation Loss 与 BPC，计算训练—验证差距。

`val` 已经参与过 early stopping 和 checkpoint 选择，因此这里只称 validation diagnostic，不称独立测试。

### 2. Validation 原文续写

- 从验证章节生成 16 条固定原文前缀。
- 三个 checkpoint 使用相同 prompt、seed、temperature 与 top-k。
- 每条最多展示 120 字，记录 EOS、空输出、4-gram 重复和最长单字连写。
- 自动指标只识别机械退化；流畅、连贯、承接和人物一致性仍需人工审阅。

### 3. Validation 人物 Cloze 候选排序

- 12 条验证原文探针：训练语料高频人物 6 条、低频人物 6 条。
- 正确人物是验证原文前缀后的精确下一个实体；答案不能出现在 prompt 中。
- 干扰项必须同为人物，并尽量匹配训练频次和长度。
- 同时报告总对数概率、每 Token、每字符和固定中性前缀先验校正后的排名。
- 随机四选一基线：Top-1 为 25%；期望 MRR 为 0.5208。

该指标只能称为“验证原文前缀候选排序”，不能单独证明模型掌握了可聊天调用的人物知识。

## 预先冻结的解释规则

### 形成有效语言基座

同时满足以下条件，才可以说预训练形成了可用于后训练的小说语言基座：

1. Step 5750 相比 Step 250 的 Validation BPC 至少改善 20%。
2. Validation Top-1 下一 Token 准确率相对早期对照提高。
3. 16 条固定续写没有空输出，机械退化率不高于 25%。
4. AI 辅助人工复核的平均流畅度与局部连贯度均不低于 2/5；该复核必须标注为 AI，不冒充独立真人验收。

### 当前配置下的实用平台期

只有同时满足以下条件，才称为“当前配置下进入实用平台期”：

1. Step 6000 相比 Step 5750 的 Validation BPC 改善小于 0.01。
2. Top-1 改善小于 0.5 个百分点。
3. 先验校正 Cloze MRR 改善不超过 0.05。
4. 固定续写的机械退化和人工语义结果没有一致改善。

“实用平台期”不等于理论极限，也不等于语料所能达到的绝对上限。它只表示继续沿用当前学习率、重复曝光和模型配置，边际收益已经很小。

### 成品小说生成器

只有独立真人复核的流畅度、连贯度、提示承接和人物一致性均达到 4/5，才可以称为成熟小说生成器。本轮自动审计不能给出这一结论。

## 测试架构

- 单元测试：窗口选择、精确交叉熵、Cloze 多 Token 排名、特殊 Token 屏蔽。
- 数据契约测试：语料/Token manifest/checkpoint/probe SHA-256 端到端绑定。
- 安全边界测试：默认禁止 test，拒绝 SFT checkpoint 与 SFT 派生探针。
- 集成测试：CPU 小样本完整运行并生成 JSON、Markdown 与分模块日志。
- 正式运行：通过上述测试后，才在 MPS 上评估三个 checkpoint。

## 日志与复现

评估日志按 data、checkpoint、validation、generation、cloze 和 orchestrator 分开，包含 UTC 时间与 run_id。每个模块可独立设置级别或关闭，日志轮转并默认脱敏。正式报告必须记录所有输入、输出和探针文件的 SHA-256。
