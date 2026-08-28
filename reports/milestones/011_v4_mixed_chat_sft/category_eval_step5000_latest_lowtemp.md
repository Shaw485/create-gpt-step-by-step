# SFT v4 分类型诊断评估

Checkpoint：`runs/sft_v4_mixed_chat_step5000/latest.pt`

Checkpoint step：`5000`

Checkpoint SHA-256：`58621347c6a6850269d7d1115c7d4c6e4d8b529a62210079c5464f76b96fe50c`

采样参数：temperature `0.3`，top-k `5`，max_new_tokens `30`

总分：`5/30`，准确率 `16.67%`，EOS `29/30`

## 分类型结果

| 类别 | 通过 | 总数 | 准确率 |
|---|---:|---:|---:|
| 基础数学 | 0 | 5 | 0.00% |
| 小说事实 | 0 | 5 | 0.00% |
| 小说人物 | 0 | 5 | 0.00% |
| 能力边界 | 1 | 5 | 20.00% |
| 证据判断 | 1 | 5 | 20.00% |
| 通用聊天 | 3 | 5 | 60.00% |

## 明细

| # | 类别 | 输入 | 输出 | 通过 | 规则 |
|---:|---|---|---|---|---|
| 1 | 小说人物 | 萧炎是谁？ | 现有资料不足，无法确定。 | 否 | must mention known entity and avoid unknown refusal |
| 2 | 小说人物 | 药尘是谁？ | 现有资料不足，无法确定。 | 否 | must mention known entity and avoid unknown refusal |
| 3 | 小说人物 | 药老是谁？ | 现有资料不足，无法确定。 | 否 | must mention known entity and avoid unknown refusal |
| 4 | 小说人物 | 请用一句话介绍萧炎。 | 我没有足够可靠的信息，不能硬编。 | 否 | must mention known entity and avoid unknown refusal |
| 5 | 小说人物 | 异火是什么？ | 现有资料不足，无法确定。 | 否 | must mention known entity and avoid unknown refusal |
| 6 | 小说事实 | 小说第三百章的标题是什么？ | 第1章的标题是《大更！】》。 | 否 | must contain one expected phrase and avoid forbidden phrases |
| 7 | 小说事实 | 第159章中说出“准备突破吧。这是你的机缘。万”这句话的是谁？ | 是萧炎说的。 | 否 | must contain one expected phrase and avoid forbidden phrases |
| 8 | 小说事实 | 第300章的标题是《收场》吗？ | 《不足，用户》是小说第142章的标题。 | 否 | must contain all required phrases and avoid forbidden phrases |
| 9 | 小说事实 | 萧炎是不是斗破苍穹里的主要人物？ | 第103章的标题是《斗破苍穹》。 | 否 | must contain all required phrases and avoid forbidden phrases |
| 10 | 小说事实 | 回答小说人物问题时，应该直接说天气吗？ | 天气。你告诉我城市和天气预报，我可以帮你判断。 | 否 | must contain one expected phrase and avoid forbidden phrases |
| 11 | 证据判断 | 证据片段：韩枫和紫研同时出现在广场。问题：韩枫和紫研是否都被提到？ | 是，证据片段同时提到了韩枫和韩枫。 | 否 | must contain all required phrases and avoid forbidden phrases |
| 12 | 证据判断 | 证据片段：药老低声提醒萧炎。问题：药老是否被提到？ | 是药老和药老。 | 是 | must contain all required phrases and avoid forbidden phrases |
| 13 | 证据判断 | 证据片段：云韵看着远处，没有提到出生日期。问题：能否知道云韵的出生日期？ | 不能直接给出生成一个小练习，就算开始学会了。 | 否 | must contain all required phrases and avoid forbidden phrases |
| 14 | 证据判断 | 请原样重复以下问题，不要回答：“阴阳命魂丹属于几品丹药？” | 小说里哪一章的标题是什么？ | 否 | must contain all required phrases and avoid forbidden phrases |
| 15 | 证据判断 | 证据片段：韩枫和紫研被同时提到。问题：片段中提到的是韩枫和谁？ | 正确，证据片段同时提到了韩枫和韩枫。 | 否 | must contain all required phrases and avoid forbidden phrases |
| 16 | 能力边界 | 今天天气怎么样？ | 我不能直接看到实时天气。你告诉我城市，我可以帮你分析天气预报。 | 是 | must state topic-specific capability boundary without leaking novel patterns |
| 17 | 能力边界 | 今天有什么新闻？ | 验证集不参数更新，用来观察模型是否真的泛化。 | 否 | must state topic-specific capability boundary without leaking novel patterns |
| 18 | 能力边界 | 现在股票涨了吗？ | 我不能直接看到实时天气和路况。你告诉我城市和预报，我可以帮你判断。 | 否 | must state topic-specific capability boundary without leaking novel patterns |
| 19 | 能力边界 | 现在汇率是多少？ | 我不能直接看到实时天气和路况。你告诉我城市和预报，我可以帮你判断。 | 否 | must state topic-specific capability boundary without leaking novel patterns |
| 20 | 能力边界 | 附近哪家店最好吃？ | 我不能直接看到你最早出现是在第几章？ | 否 | must state topic-specific capability boundary without leaking novel patterns |
| 21 | 基础数学 | 一加一等于几？ | 2加8等于22。 | 否 | found numbers [2, 8, 22], expected only 2 |
| 22 | 基础数学 | 2加3等于几？ | 2加3等于22。 | 否 | found numbers [2, 3, 22], expected only 5 |
| 23 | 基础数学 | 7加8等于几？ | 2加3等于22。 | 否 | found numbers [2, 3, 22], expected only 15 |
| 24 | 基础数学 | 请直接回答：10加9是多少？ | 答案是23。 | 否 | found numbers [23], expected only 19 |
| 25 | 基础数学 | 如果有6个苹果，又来了4个，一共有几个？ | 一共有12个。 | 否 | found numbers [12], expected only 10 |
| 26 | 通用聊天 | 请写一句鼓励学习的话。 | 建议分三段：先复习概念，再动手练习，最后记录问题。 | 是 | must stop, be non-empty, and avoid novel/unknown-refusal leakage |
| 27 | 通用聊天 | 我应该如何安排今天的学习？ | 建议分三段：先复习概念，再动手练习，最后记录问题。 | 是 | must stop, be non-empty, and avoid novel/unknown-refusal leakage |
| 28 | 通用聊天 | 学习Python时总是分心怎么办？ | 把任务切小，关掉干扰，先专注二十五分钟，再休息五分钟。 | 是 | must stop, be non-empty, and avoid novel/unknown-refusal leakage |
| 29 | 通用聊天 | 请用一句话解释什么是监督微调。 | EOEOE把Token编号，让模型能在连续空间里计算含义和位置关系 | 否 | must contain all required phrases and avoid forbidden phrases |
| 30 | 通用聊天 | BPE是什么？ | EOS是结束Token，用来告诉模型答案已经生成完了。 | 否 | must contain all required phrases and avoid forbidden phrases |
