# SFT 分类型诊断评估（v2，无数学）

Checkpoint：`runs/sft_v5_1_no_math_step2000/latest.pt`

Checkpoint step：`2000`

Checkpoint SHA-256：`c05b806f3db9420b1418a8a64b963c54aaea1bf70a3c061e3554dc9e5bf36ba5`

采样参数：temperature `0.3`，top-k `1`，max_new_tokens `30`

总分：`4/30`，准确率 `13.33%`，EOS `29/30`

## 分类型结果

| 类别 | 通过 | 总数 | 准确率 |
|---|---:|---:|---:|
| 小说事实 | 2 | 5 | 40.00% |
| 小说人物 | 0 | 5 | 0.00% |
| 指令遵循 | 0 | 5 | 0.00% |
| 能力边界 | 0 | 5 | 0.00% |
| 证据判断 | 1 | 5 | 20.00% |
| 通用聊天 | 1 | 5 | 20.00% |

## 明细

| # | 类别 | 输入 | 输出 | 通过 | 规则 |
|---:|---|---|---|---|---|
| 1 | 小说人物 | 萧炎是谁？ | 是一个本地训练习，最后用自己的话解释，用来告诉模型。 | 否 | must identify the entity with a relevant role and avoid unknown refusal |
| 2 | 小说人物 | 药尘是谁？ | 现有资料不足，无法确定其首次出现章节。 | 否 | must identify the entity with a relevant role and avoid unknown refusal |
| 3 | 小说人物 | 药老是谁？ | 是药老和药老。 | 否 | must identify the entity with a relevant role and avoid unknown refusal |
| 4 | 小说人物 | 请用一句话介绍萧炎。 | 我没有足够可靠的信息，不能硬编。 | 否 | must identify the entity with a relevant role and avoid unknown refusal |
| 5 | 小说人物 | 异火是什么？ | 是一个本。 | 否 | must identify the entity with a relevant role and avoid unknown refusal |
| 6 | 小说事实 | 小说第三百章的标题是什么？ | 第14444444章的标题是《龙》。 | 否 | must contain all required phrases and avoid forbidden phrases |
| 7 | 小说事实 | 第159章中说出“准备突破吧。这是你的机缘。万”这句话的是谁？ | 是药老说的。 | 是 | must contain one expected phrase and avoid forbidden phrases |
| 8 | 小说事实 | 第300章的标题是《收场》吗？ | 第1444章的标题是《毒宗》。 | 否 | must contain all required phrases and avoid forbidden phrases |
| 9 | 小说事实 | 萧炎是不是斗破苍穹里的主要人物？ | 是小说的。 | 否 | must contain all required phrases and avoid forbidden phrases |
| 10 | 小说事实 | 回答小说人物问题时，应该直接说天气吗？ | 先说明资料不足，不能编造。 | 是 | must contain one expected phrase and avoid forbidden phrases |
| 11 | 证据判断 | 证据片段：韩枫和紫研同时出现在广场。问题：韩枫和紫研是否都被提到？ | 是韩枫。 | 否 | must contain all required phrases and avoid forbidden phrases |
| 12 | 证据判断 | 证据片段：药老低声提醒萧炎。问题：药老是否被提到？ | 是药老和药老。 | 是 | must contain all required phrases and avoid forbidden phrases |
| 13 | 证据判断 | 证据片段：云韵看着远处，没有提到出生日期。问题：能否知道云韵的出生日期？ | 能确认；当前证据片段同时提到了萧炎和欣蓝。 | 否 | must contain all required phrases and avoid forbidden phrases |
| 14 | 证据判断 | 请原样重复以下问题，不要回答：“阴阳命魂丹属于几品丹药？” | 现有资料不足，无法确定其首次出现章节。 | 否 | must contain all required phrases and avoid forbidden phrases |
| 15 | 证据判断 | 证据片段：韩枫和紫研被同时提到。问题：片段中提到的是韩枫和谁？ | 是韩枫。 | 否 | must contain all required phrases and avoid forbidden phrases |
| 16 | 能力边界 | 今天天气怎么样？ | 我不能直接给出。 | 否 | must state topic-specific capability boundary without leaking novel patterns |
| 17 | 能力边界 | 今天有什么新闻？ | 现有资料不足，无法确定其首次出现章节。 | 否 | must state topic-specific capability boundary without leaking novel patterns |
| 18 | 能力边界 | 现在股票涨了吗？ | 我是一个本。 | 否 | must state topic-specific capability boundary without leaking novel patterns |
| 19 | 能力边界 | 现在汇率是多少？ | 我是一个本。 | 否 | must state topic-specific capability boundary without leaking novel patterns |
| 20 | 能力边界 | 附近哪家店最好吃？ | 我是一个本。 | 否 | must state topic-specific capability boundary without leaking novel patterns |
| 21 | 指令遵循 | 请只回答“收到”。 | 我没有足够可靠的信息，不能硬编。 | 否 | must exactly match one of ['收到', '收到。'] |
| 22 | 指令遵循 | 请原样重复这四个字：“训练模型”。 | 第14444章的标题是《毒宗》。 | 否 | must exactly match one of ['训练模型', '训练模型。'] |
| 23 | 指令遵循 | 请给出两条提高学习专注力的建议。 | 先复习，最后记录问题。 | 否 | must provide at least two topic-relevant suggestions without novel leakage |
| 24 | 指令遵循 | 请只回答“是”或“否”：测试集应该参与参数更新吗？ | 是否定。 | 否 | must exactly match one of ['否', '否。'] |
| 25 | 指令遵循 | 请用一句话说明验证集的作用。 | 模型是否认；片段同时提到了萧炎和目标。 | 否 | must explain validation-set purpose in one sentence without novel leakage |
| 26 | 通用聊天 | 请写一句鼓励学习的话。 | 我没有足够可靠的信息，不能硬编。 | 否 | must be useful, follow requested topic, and avoid novel/refusal leakage |
| 27 | 通用聊天 | 我应该如何安排今天的学习？ | 先学习，最后五分钟练习，最后五分钟练习，最后五分钟练习，最后五分钟练习，最后五分钟练习，最后五 | 否 | must be useful, follow requested topic, and avoid novel/refusal leakage; generation must stop on EOS |
| 28 | 通用聊天 | 学习Python时总是分心怎么办？ | 先复习，十五分钟练习，最后五分钟练习，最后五分钟练习，最后五分钟记录问题。 | 是 | must be useful, follow requested topic, and avoid novel/refusal leakage |
| 29 | 通用聊天 | 请用一句话解释什么是监督微调。 | 模型是否认其中提到了模型。 | 否 | must contain all required phrases and avoid forbidden phrases |
| 30 | 通用聊天 | BPE是什么？ | 我是一个本地训练习的标题是否认；片段同时提到了萧炎和雅妃。 | 否 | must contain all required phrases and avoid forbidden phrases |
