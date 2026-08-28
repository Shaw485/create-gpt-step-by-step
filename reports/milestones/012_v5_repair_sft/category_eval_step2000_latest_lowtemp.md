# SFT v4 分类型诊断评估

Checkpoint：`runs/sft_v5_repair_step2000/latest.pt`

Checkpoint step：`2000`

Checkpoint SHA-256：`53ecfb7a362f9a9ffc1ef18361d4eb8e81eeba8a2551e9153d95711dfaec08bb`

采样参数：temperature `0.3`，top-k `5`，max_new_tokens `30`

总分：`6/30`，准确率 `20.00%`，EOS `24/30`

## 分类型结果

| 类别 | 通过 | 总数 | 准确率 |
|---|---:|---:|---:|
| 基础数学 | 0 | 5 | 0.00% |
| 小说事实 | 1 | 5 | 20.00% |
| 小说人物 | 2 | 5 | 40.00% |
| 能力边界 | 1 | 5 | 20.00% |
| 证据判断 | 1 | 5 | 20.00% |
| 通用聊天 | 1 | 5 | 20.00% |

## 明细

| # | 类别 | 输入 | 输出 | 通过 | 规则 |
|---:|---|---|---|---|---|
| 1 | 小说人物 | 萧炎是谁？ | 第300。 | 否 | must mention known entity and avoid unknown refusal |
| 2 | 小说人物 | 药尘是谁？ | 30。 | 否 | must mention known entity and avoid unknown refusal |
| 3 | 小说人物 | 药老是谁？ | 药老是小说中被提到的人物。 | 是 | must mention known entity and avoid unknown refusal |
| 4 | 小说人物 | 请用一句话介绍萧炎。 | 答案是230。 | 否 | must mention known entity and avoid unknown refusal |
| 5 | 小说人物 | 异火是什么？ | 异火是小说中被提到的重要人物。 | 是 | must mention known entity and avoid unknown refusal |
| 6 | 小说事实 | 小说第三百章的标题是什么？ | 第3000章的标题是《收场》。 | 否 | must contain all required phrases and avoid forbidden phrases |
| 7 | 小说事实 | 第159章中说出“准备突破吧。这是你的机缘。万”这句话的是谁？ | 是药老说的。 | 是 | must contain one expected phrase and avoid forbidden phrases |
| 8 | 小说事实 | 第300章的标题是《收场》吗？ | 是，第30000章的标题是《收场》。 | 否 | must contain all required phrases and avoid forbidden phrases |
| 9 | 小说事实 | 萧炎是不是斗破苍穹里的主要人物？ | 第390章的标题是《收场》。 | 否 | must contain all required phrases and avoid forbidden phrases |
| 10 | 小说事实 | 回答小说人物问题时，应该直接说天气吗？ | 我不能直接看到实时天气和天气和天气。你告诉我城市和天气预报，我可以帮你判 | 否 | must contain one expected phrase and avoid forbidden phrases |
| 11 | 证据判断 | 证据片段：韩枫和紫研同时出现在广场。问题：韩枫和紫研是否都被提到？ | 是，证据片段同时提到了韩枫和韩枫。 | 否 | must contain all required phrases and avoid forbidden phrases |
| 12 | 证据判断 | 证据片段：药老低声提醒萧炎。问题：药老是否被提到？ | 是药老和萧炎。 | 是 | must contain all required phrases and avoid forbidden phrases |
| 13 | 证据判断 | 证据片段：云韵看着远处，没有提到出生日期。问题：能否知道云韵的出生日期？ | 能确定其首次出现章节。 | 否 | must contain all required phrases and avoid forbidden phrases |
| 14 | 证据判断 | 请原样重复以下问题，不要回答：“阴阳命魂丹属于几品丹药？” | 九十九十九章《斗破苍穹》第九百九百九十九章《斗破苍穹》的第一千一 | 否 | must contain all required phrases and avoid forbidden phrases |
| 15 | 证据判断 | 证据片段：韩枫和紫研被同时提到。问题：片段中提到的是韩枫和谁？ | 片段中提到的是韩枫和韩枫。 | 否 | must contain all required phrases and avoid forbidden phrases |
| 16 | 能力边界 | 今天天气怎么样？ | 我不能直接看到实时天气和天气和天气。你告诉我城市和天气和天气和天气和 | 是 | must state topic-specific capability boundary without leaking novel patterns |
| 17 | 能力边界 | 今天有什么新闻？ | 我不能直接看到实时天气和天气和天气和天气和天气和天气和天气和天气 | 否 | must state topic-specific capability boundary without leaking novel patterns |
| 18 | 能力边界 | 现在股票涨了吗？ | 我不能直接看到实时天气和路况。你告诉我城市和预报，我可以帮你判断。 | 否 | must state topic-specific capability boundary without leaking novel patterns |
| 19 | 能力边界 | 现在汇率是多少？ | 第300。 | 否 | must state topic-specific capability boundary without leaking novel patterns |
| 20 | 能力边界 | 附近哪家店最好吃？ | 我不能直接看到实时天气和天气和天气和天气和天气和天气和天气和天气 | 否 | must state topic-specific capability boundary without leaking novel patterns |
| 21 | 基础数学 | 一加一等于几？ | 1加3加5等于2加1。 | 否 | found numbers [1, 3, 5, 2, 1], expected only 2 |
| 22 | 基础数学 | 2加3等于几？ | 1加2加1等于2。 | 否 | found numbers [1, 2, 1, 2], expected only 5 |
| 23 | 基础数学 | 7加8等于几？ | 1加2加2等于2等于2。 | 否 | found numbers [1, 2, 2, 2, 2], expected only 15 |
| 24 | 基础数学 | 请直接回答：10加9是多少？ | 答案是22。 | 否 | found numbers [22], expected only 19 |
| 25 | 基础数学 | 如果有6个苹果，又来了4个，一共有几个？ | 一共有323个。 | 否 | found numbers [323], expected only 10 |
| 26 | 通用聊天 | 请写一句鼓励学习的话。 | 请先看用户是否明确问小说。 | 是 | must stop, be non-empty, and avoid novel/unknown-refusal leakage |
| 27 | 通用聊天 | 我应该如何安排今天的学习？ | 先看用户是小说章节号。 | 否 | must stop, be non-empty, and avoid novel/unknown-refusal leakage |
| 28 | 通用聊天 | 学习Python时总是分心怎么办？ | 先看用户是否明确问小说章节，否则不要强行回答章节号。 | 否 | must stop, be non-empty, and avoid novel/unknown-refusal leakage |
| 29 | 通用聊天 | 请用一句话解释什么是监督微调。 | GGTokenenTokenn。 | 否 | must contain all required phrases and avoid forbidden phrases |
| 30 | 通用聊天 | BPE是什么？ | TokeennenTokenToenTokeeenTokee | 否 | must contain all required phrases and avoid forbidden phrases |
