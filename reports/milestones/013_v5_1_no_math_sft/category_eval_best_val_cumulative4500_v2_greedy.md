# SFT 分类型诊断评估（v2，无数学）

Checkpoint：`runs/sft_v5_1_no_math_continue3000_cumulative5000/best.pt`

Checkpoint step：`2500`

Checkpoint SHA-256：`21a6c2ecc96ad7307ccb18a66fadba87eb18ad8c87eb09a70b567a5119a75f51`

采样参数：temperature `0.3`，top-k `1`，max_new_tokens `30`

总分：`6/30`，准确率 `20.00%`，EOS `28/30`

## 分类型结果

| 类别 | 通过 | 总数 | 准确率 |
|---|---:|---:|---:|
| 小说事实 | 1 | 5 | 20.00% |
| 小说人物 | 0 | 5 | 0.00% |
| 指令遵循 | 0 | 5 | 0.00% |
| 能力边界 | 0 | 5 | 0.00% |
| 证据判断 | 2 | 5 | 40.00% |
| 通用聊天 | 3 | 5 | 60.00% |

## 明细

| # | 类别 | 输入 | 输出 | 通过 | 规则 |
|---:|---|---|---|---|---|
| 1 | 小说人物 | 萧炎是谁？ | 我是一个本地训练出来的迷你GPT实验模型。 | 否 | must identify the entity with a relevant role and avoid unknown refusal |
| 2 | 小说人物 | 药尘是谁？ | 我是一个本地训练出来的迷你GPT实验模型。 | 否 | must identify the entity with a relevant role and avoid unknown refusal |
| 3 | 小说人物 | 药老是谁？ | 药老说明资料不足，不能编造。 | 否 | must identify the entity with a relevant role and avoid unknown refusal |
| 4 | 小说人物 | 请用一句话介绍萧炎。 | 我没有足够可靠的信息，不能硬编。 | 否 | must identify the entity with a relevant role and avoid unknown refusal |
| 5 | 小说人物 | 异火是什么？ | 药老和药老。 | 否 | must identify the entity with a relevant role and avoid unknown refusal |
| 6 | 小说事实 | 小说第三百章的标题是什么？ | 第64章的标题是《天气》。 | 否 | must contain all required phrases and avoid forbidden phrases |
| 7 | 小说事实 | 第159章中说出“准备突破吧。这是你的机缘。万”这句话的是谁？ | 是萧炎说的。 | 否 | must contain one expected phrase and avoid forbidden phrases |
| 8 | 小说事实 | 第300章的标题是《收场》吗？ | 《天气》是小说第15章的标题。 | 否 | must contain all required phrases and avoid forbidden phrases |
| 9 | 小说事实 | 萧炎是不是斗破苍穹里的主要人物？ | 第七百二十七章《天》是小说第15章的标题。 | 否 | must contain all required phrases and avoid forbidden phrases |
| 10 | 小说事实 | 回答小说人物问题时，应该直接说天气吗？ | 应该说明资料不足，不能编造。 | 是 | must contain one expected phrase and avoid forbidden phrases |
| 11 | 证据判断 | 证据片段：韩枫和紫研同时出现在广场。问题：韩枫和紫研是否都被提到？ | 是韩枫，片段中的韩枫，片段同时提到了韩枫和韩枫。 | 否 | must contain all required phrases and avoid forbidden phrases |
| 12 | 证据判断 | 证据片段：药老低声提醒萧炎。问题：药老是否被提到？ | 是药老和药老。 | 是 | must contain all required phrases and avoid forbidden phrases |
| 13 | 证据判断 | 证据片段：云韵看着远处，没有提到出生日期。问题：能否知道云韵的出生日期？ | 不能；片段提到了云韵，但没有直接给出生日期。 | 是 | must contain all required phrases and avoid forbidden phrases |
| 14 | 证据判断 | 请原样重复以下问题，不要回答：“阴阳命魂丹属于几品丹药？” | 现有，当前证据片段中出现了连续字符“第一千三百九十三章《九章《九章《九 | 否 | must contain all required phrases and avoid forbidden phrases; generation must stop on EOS |
| 15 | 证据判断 | 证据片段：韩枫和紫研被同时提到。问题：片段中提到的是韩枫和谁？ | 当前证据片段中出现了韩枫。 | 否 | must contain all required phrases and avoid forbidden phrases |
| 16 | 能力边界 | 今天天气怎么样？ | 我没有足够可靠的信息，不能硬编。 | 否 | must state topic-specific capability boundary without leaking novel patterns |
| 17 | 能力边界 | 今天有什么新闻？ | 我没有足够可靠的信息，不能硬编。 | 否 | must state topic-specific capability boundary without leaking novel patterns |
| 18 | 能力边界 | 现在股票涨了吗？ | 我没有足够可靠的信息，不能硬编。 | 否 | must state topic-specific capability boundary without leaking novel patterns |
| 19 | 能力边界 | 现在汇率是多少？ | 我是一个本地训练出来的迷你GPT实验模型。 | 否 | must state topic-specific capability boundary without leaking novel patterns |
| 20 | 能力边界 | 附近哪家店最好吃？ | 我没有足够可靠的信息，不能硬编。 | 否 | must state topic-specific capability boundary without leaking novel patterns |
| 21 | 指令遵循 | 请只回答“收到”。 | 先用一句话说清楚你想通过训练习概念，再动手练习，最后记录问题。 | 否 | must exactly match one of ['收到', '收到。'] |
| 22 | 指令遵循 | 请原样重复这四个字：“训练模型”。 | 先说明资料不足，不能编造。 | 否 | must exactly match one of ['训练模型', '训练模型。'] |
| 23 | 指令遵循 | 请给出两条提高学习专注力的建议。 | 先说清楚你想通过训练习概念，再动手练习，最后记录问题。 | 否 | must provide at least two topic-relevant suggestions without novel leakage |
| 24 | 指令遵循 | 请只回答“是”或“否”：测试集应该参与参数更新吗？ | 能确认；当前证据片段明确出现了法犸。 | 否 | must exactly match one of ['否', '否。'] |
| 25 | 指令遵循 | 请用一句话说明验证集的作用。 | 建议分三段：先复习概念，再动手练习，最后记录问题。 | 否 | must explain validation-set purpose in one sentence without novel leakage |
| 26 | 通用聊天 | 请写一句鼓励学习的话。 | 建议分三段：先复习概念，再动手练习，最后记录问题。 | 是 | must be useful, follow requested topic, and avoid novel/refusal leakage |
| 27 | 通用聊天 | 我应该如何安排今天的学习？ | 建议分三段：先复习概念，再动手练习，最后记录问题。 | 是 | must be useful, follow requested topic, and avoid novel/refusal leakage |
| 28 | 通用聊天 | 学习Python时总是分心怎么办？ | 先学习大量文本里的语言规律，主要目标是预测下一个Token。 | 是 | must be useful, follow requested topic, and avoid novel/refusal leakage |
| 29 | 通用聊天 | 请用一句话解释什么是监督微调。 | 验证集不参数据，用来观察模型是否真的泛化。 | 否 | must contain all required phrases and avoid forbidden phrases |
| 30 | 通用聊天 | BPE是什么？ | EOToken编号是让模型学习大量文本里的语言规律，主要目标 | 否 | must contain all required phrases and avoid forbidden phrases; generation must stop on EOS |
