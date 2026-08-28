# SFT v4 分类型诊断评估

Checkpoint：`runs/sft_v5_repair_smoke100/latest.pt`

Checkpoint step：`100`

Checkpoint SHA-256：`8c93cd17d34c78548131743aaedfdd4fe2430a4e33f0469b33ea5af675a06c2b`

采样参数：temperature `0.3`，top-k `5`，max_new_tokens `30`

总分：`5/30`，准确率 `16.67%`，EOS `22/30`

## 分类型结果

| 类别 | 通过 | 总数 | 准确率 |
|---|---:|---:|---:|
| 基础数学 | 0 | 5 | 0.00% |
| 小说事实 | 1 | 5 | 20.00% |
| 小说人物 | 3 | 5 | 60.00% |
| 能力边界 | 0 | 5 | 0.00% |
| 证据判断 | 0 | 5 | 0.00% |
| 通用聊天 | 1 | 5 | 20.00% |

## 明细

| # | 类别 | 输入 | 输出 | 通过 | 规则 |
|---:|---|---|---|---|---|
| 1 | 小说人物 | 萧炎是谁？ | 我说你是你说你是萧炎是萧炎说说你是萧炎是萧炎说中说中提到萧炎和萧炎是否是 | 是 | must mention known entity and avoid unknown refusal |
| 2 | 小说人物 | 药尘是谁？ | 我说到萧炎是否是否是否是否。 | 否 | must mention known entity and avoid unknown refusal |
| 3 | 小说人物 | 药老是谁？ | 我说是萧炎是萧炎是药老说到萧炎是萧炎是萧炎是药老。 | 是 | must mention known entity and avoid unknown refusal |
| 4 | 小说人物 | 请用一句话介绍萧炎。 | 是萧炎和萧炎和萧炎和萧炎和薰儿。 | 是 | must mention known entity and avoid unknown refusal |
| 5 | 小说人物 | 异火是什么？ | 我是你是你是我说你是我是我是我是我是我说中提到的是萧炎是萧炎是萧炎 | 否 | must mention known entity and avoid unknown refusal |
| 6 | 小说事实 | 小说第三百章的标题是什么？ | 我说中提到的是萧炎是萧炎和萧炎和药老。 | 否 | must contain all required phrases and avoid forbidden phrases |
| 7 | 小说事实 | 第159章中说出“准备突破吧。这是你的机缘。万”这句话的是谁？ | 我说。 | 否 | must contain one expected phrase and avoid forbidden phrases |
| 8 | 小说事实 | 第300章的标题是《收场》吗？ | 是我说是萧炎和你是萧炎和萧炎和薰儿。 | 否 | must contain all required phrases and avoid forbidden phrases |
| 9 | 小说事实 | 萧炎是不是斗破苍穹里的主要人物？ | 是萧炎和萧炎和萧炎和萧炎和萧炎和萧炎和萧炎和萧炎和萧炎和萧炎和萧炎和萧炎和萧炎是萧炎和萧炎 | 是 | must contain all required phrases and avoid forbidden phrases |
| 10 | 小说事实 | 回答小说人物问题时，应该直接说天气吗？ | 我说是你是否则。 | 否 | must contain one expected phrase and avoid forbidden phrases |
| 11 | 证据判断 | 证据片段：韩枫和紫研同时出现在广场。问题：韩枫和紫研是否都被提到？ | 是萧炎和萧炎和萧炎和萧炎和萧炎和萧炎和萧炎和萧炎和萧炎和萧炎和萧炎和萧炎和萧炎和萧炎和萧炎 | 否 | must contain all required phrases and avoid forbidden phrases |
| 12 | 证据判断 | 证据片段：药老低声提醒萧炎。问题：药老是否被提到？ | 我说。 | 否 | must contain all required phrases and avoid forbidden phrases |
| 13 | 证据判断 | 证据片段：云韵看着远处，没有提到出生日期。问题：能否知道云韵的出生日期？ | 是萧炎和萧炎和小医仙。 | 否 | must contain all required phrases and avoid forbidden phrases |
| 14 | 证据判断 | 请原样重复以下问题，不要回答：“阴阳命魂丹属于几品丹药？” | 有关注意味道的是萧炎是萧炎和萧炎和萧炎和萧炎是萧炎和萧炎和萧炎和萧炎和萧炎和萧炎和 | 否 | must contain all required phrases and avoid forbidden phrases |
| 15 | 证据判断 | 证据片段：韩枫和紫研被同时提到。问题：片段中提到的是韩枫和谁？ | 是萧炎和萧炎和萧炎和萧炎和萧炎和萧炎和萧炎和萧炎和萧炎和萧炎和萧炎和萧炎和萧炎和萧炎和萧炎 | 否 | must contain all required phrases and avoid forbidden phrases |
| 16 | 能力边界 | 今天天气怎么样？ | 是萧炎说是萧炎是萧炎说是萧炎和萧炎和萧炎是萧炎说是萧炎和萧炎和萧炎是萧炎是萧炎和萧炎是 | 否 | must state topic-specific capability boundary without leaking novel patterns |
| 17 | 能力边界 | 今天有什么新闻？ | 我是否则。 | 否 | must state topic-specific capability boundary without leaking novel patterns |
| 18 | 能力边界 | 现在股票涨了吗？ | 我说到萧炎是萧炎是萧炎是萧炎是萧炎说中是萧炎是萧炎是萧炎是萧炎说。 | 否 | must state topic-specific capability boundary without leaking novel patterns |
| 19 | 能力边界 | 现在汇率是多少？ | 你是你是你说中提到的是萧炎和药老。 | 否 | must state topic-specific capability boundary without leaking novel patterns |
| 20 | 能力边界 | 附近哪家店最好吃？ | 是否是否。 | 否 | must state topic-specific capability boundary without leaking novel patterns |
| 21 | 基础数学 | 一加一等于几？ | 一章。 | 否 | expected number 2 not found |
| 22 | 基础数学 | 2加3等于几？ | 一章。 | 否 | expected number 5 not found |
| 23 | 基础数学 | 7加8等于几？ | 一章。 | 否 | expected number 15 not found |
| 24 | 基础数学 | 请直接回答：10加9是多少？ | 一章。 | 否 | expected number 19 not found |
| 25 | 基础数学 | 如果有6个苹果，又来了4个，一共有几个？ | 有。 | 否 | expected number 10 not found |
| 26 | 通用聊天 | 请写一句鼓励学习的话。 | 你和我是否则。 | 是 | must stop, be non-empty, and avoid novel/unknown-refusal leakage |
| 27 | 通用聊天 | 我应该如何安排今天的学习？ | 有。 | 否 | must stop, be non-empty, and avoid novel/unknown-refusal leakage |
| 28 | 通用聊天 | 学习Python时总是分心怎么办？ | 我说是萧炎说中提到了萧炎和萧炎和萧炎和萧炎和萧炎和萧炎和萧炎和萧炎和萧炎和萧炎和萧炎和 | 否 | must stop, be non-empty, and avoid novel/unknown-refusal leakage |
| 29 | 通用聊天 | 请用一句话解释什么是监督微调。 | 有事。 | 否 | must contain all required phrases and avoid forbidden phrases |
| 30 | 通用聊天 | BPE是什么？ | 我说是什么？” | 否 | must contain all required phrases and avoid forbidden phrases |
