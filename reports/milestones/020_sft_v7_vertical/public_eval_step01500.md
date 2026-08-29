# SFT v7 公开诊断

Checkpoint step：`1500`

全量公开 teacher-forced loss：`3.419921`；perplexity：`30.567`

开放表达只提供自动退化信号；AI 质量审核与独立真人审核均为待完成，不以模糊字符串相似度作为硬门。

| 维度 | 数量 | Required | Forbidden违规 | Keypoint | 已知误拒 | EOS | 空答 | 截断 | 4gram退化 | 元话术 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| core_facts_and_corrections | 108 | 48.1% | 0.0% | 43.3% | 1.9% | 71.3% | 0.0% | 28.7% | 43.5% | 0.0% |
| single_evidence_qa | 192 | 75.3% | — | — | — | 82.8% | 0.0% | 17.2% | 39.1% | 0.0% |
| rag_evidence_composition | 84 | 72.5% | — | 72.5% | — | 75.0% | 0.0% | 25.0% | 36.9% | 0.0% |
| vertical_chat_multiturn_eos | 108 | 71.3% | 0.0% | — | — | 80.6% | 0.0% | 19.4% | 45.4% | 0.0% |
| novel_expression | 78 | 78.3% | — | — | — | 79.5% | 0.0% | 20.5% | 35.9% | 0.0% |
| capability_boundary | 30 | 76.7% | — | — | — | 96.7% | 0.0% | 3.3% | 23.3% | 0.0% |

## 自动候选硬门

这些结果是可复算的词面/行为代理；不把 EM、字符 F1 或关键词命中夸大为语义正确或证据支持。

| 门 | 阈值 | 值 | 通过 | 来源 |
|---|---|---|---|---|
| core_required_keypoint_proxy | >= 0.80 | 0.4815 | 否 | public core cases; keypoint pass when present, otherwise required-term pass |
| core_known_misrefusal | <= 0.05 | 0.0185 | 是 | public core cases with evaluation.known_fact=true |
| single_evidence_lexical_or_support_proxy | normalized_char_multiset_f1 >= 0.90 OR required/keypoint support proxy >= 0.90 | {"normalized_char_multiset_f1": 0.23855599058425284, "required_keypoint_support_proxy": 0.7531645569620253} | 否 | positive public single-evidence cases; lexical F1 is a non-semantic proxy, support proxy checks only declared required/keypoint strings |
| single_evidence_insufficient_stop | >= 0.90 | 0.0000 | 否 | public single-evidence cases marked needs_evidence and not sufficient |
| rag_gold_keypoint_proxy | >= 0.80 | 0.7246 | 否 | positive public RAG cases; declared keypoint/required-term pass proxy |
| rag_required_support_proxy | >= 0.90 | 0.7246 | 否 | positive public RAG cases; required-term presence only, not citation support |
| chat_required_routing_proxy | >= 0.90 | 0.7130 | 否 | public chat declared required-term routing proxy |
| chat_multiturn_carryover_proxy | >= 0.80 | 0.6944 | 否 | public task_family=chat_multiturn required/keypoint presence proxy |
| chat_eos | >= 0.95 | 0.8056 | 否 | public chat generation EOS termination |
| chat_empty | = 0 | 0.0000 | 是 | public chat normalized empty-answer rate |
| chat_meta_phrase | <= 0.01 | 0.0000 | 是 | public chat frozen meta-phrase marker rate |
| chat_cross_task_refusal_proxy | <= 0.05 | 0.0000 | 是 | public in-domain chat refusal-marker rate (cross-task proxy) |
| expression_mechanical_repetition | <= 0.10 | 0.3590 | 否 | public expression repeated-four-gram mechanical degeneration |
| boundary_correct_stop | >= 0.90 | 0.1000 | 否 | public capability-boundary insufficient-evidence cases |
| boundary_evidence_recovery | >= 0.85 | 0.6000 | 否 | public calibration triplets in explicit grounded_answer mode; empty output fails |

自动门全部通过：**否**；当前候选资格：**无**。

## 外部待审硬门

| 门 | 阈值 | 状态 | 所需评审/报告 |
|---|---|---|---|
| core_alias_accuracy | >= 0.85 | pending | independent alias-equivalence adjudication report |
| core_semantic_accuracy_and_unsupported_details | semantic fact accuracy >= 0.80 and unsupported detail rate <= 0.05 | pending | human semantic fact and unsupported-detail audit |
| single_evidence_semantics_and_support | semantic correctness >= 0.80 and evidence support >= 0.90 | pending | evidence-span attribution plus independent semantic review |
| rag_citation_support_and_distractor_resistance | citation support >= 0.90 and anti-distractor >= 0.85 | pending | bundle-level citation/entailment and distractor adjudication |
| chat_format_length_semantics | >= 0.90 | pending | format/length instruction and semantic carryover review |
| expression_ai_and_human_quality | fluency, local coherence, prompt following each >= 2.5/5 | pending | AI-assisted rubric scores plus independent human review |
| boundary_hallucinated_fact_rate | <= 0.05 | pending | human factual-hallucination adjudication |
| pretraining_retention | 16/16 non-empty, mechanical <= 0.25, AI quality >= 2/5, val BPC degradation <= 0.10 | pending | frozen pretraining retention probe report |

## 指标定义

- Normalized EM：Unicode text is lowercased and all non-word characters/underscores are removed; the resulting non-empty prediction must equal the normalized public reference.
- Normalized character-multiset F1：Character-multiset F1 after the same normalization: overlap is the sum of the per-character minimum counts. This is a deterministic lexical proxy, not a semantic-correctness or evidence-support judgment.

## 每维少量公开样本

| 维度 | ID | 问题 | 参考 | 输出 | EOS |
|---|---|---|---|---|---|
| core_facts_and_corrections | v7_760b18208f0aa1fe1b086132 | 关于原著，丹塔是什么？ 只说有原著依据的部分。 | 丹塔是一个受到许多炼药师推崇的自由组织。对新读者而言，核心关系可据此理解。 | 这处写的是“丹塔的首次，丹塔是丹塔的丹塔，丹塔的丹塔，丹塔的名头，也是丹塔的首次，丹塔的首次都是丹塔的首次”，其中涉及丹塔。 | 是 |
| core_facts_and_corrections | v7_934683dc2afabeac9e7b004a | 读者追问第585章里的陨落心炎，应怎样准确作答？ | 原文写道“随着接过药老先前的工作，萧炎这才明白，与那陨落心炎对抗的消耗，是何等的庞大，而且，或许是由于他与药老实力的差距，虽然青莲地心”，其中涉及陨落心炎。 | 文本中出现“陨落心炎的出现，陨落心炎的陨落心炎，并提到陨落心炎的关。其余证据还包括着陨落心炎的问题。 | 是 |
| single_evidence_qa | v7_9fca6c78b778be792d8acb22 | 请阅读证据并回答其中对青莲地心火的描写：<br>嘴角掀起一抹冷意，一丝丝极淡的青色火苗，在斗.气的包裹下，将重尺包裹而进，有青莲地心火之助，寻常水属性斗气，在他面前几乎仅仅只能发挥七八层的威力 | 材料表述为“嘴角掀起一抹冷意，一丝丝极淡的青色火苗，在斗.气的包裹下，将重尺包裹而进，有青莲地心火之助，寻常水属性斗气，在他面前几乎仅仅”，这里写到青莲地心火。 | 材料表述为“萧炎手掌轻挥，青色火苗便是在指尖处升腾而起，随着青莲地心火的旋转，青色火苗便是犹如火山一般，顺着萧炎体内涌进体内，随着火灵火的旋转，火灵火苗，火灵火灵火，火灵火属性能量，火属性能量，火属性能量，火属性能量，火属性能量，火属性能量，火属性能量，火属性能量，火属性能量，火属性能量，火属性能量，火属性能量，火属性能量，火属性能量，火属性能量，火属性能量，火属性能量，火属性能量，火属性能量，火属性能量，火属性能量，火属性能量，火属性能量，火属性能量，火属性能量 | 否 |
| vertical_chat_multiturn_eos | v7_bc57a230c23395434c28a9c7 | 请像小说导读一样解释这段里的药尘：<br>闻言。药老老脸一红。有些恼羞成怒的挥了挥袖子：你以为我也像你这般不务正业么 | 原文写道“闻言。药老老脸一红。有些恼羞成怒的挥了挥袖子：你以为我也像你这般不务正业么”，其中涉及药尘。 | 材料表述为“闻言。萧炎微微点了点头。旋即将手中的玉瓶收进纳戒内。将玉瓶收进纳戒内。然后对着药老笑道：你说。先去这里吧。先去这里吧。先去这里。先去这里。先去找你。先去找你。先去找我。先去找你。先去找你。先去找我。先去找我。先去找我。请你去找我。请你。请你去找你。请你。请你去找你。请你。请你去找我。请你。请你。 | 是 |
| single_evidence_qa | v7_4177ee8e607cf3c29e57f354 | 请阅读证据并回答其中对萧炎的描写：<br>与白帮成员的鸭子声干叫声相.比，磐门那边，却是大声欢呼了起来，原本以为失去了重尺的萧炎会彻底落入下风，可没想到他却是来了个大翻盘，反而将本该占据上风的白程打得满地滚，这意外的情况，让得一脸紧张的磐门成员忍不住的发出阵阵欢呼，一些脾气冲动之人，更是将嘲讽的嬉笑骂声奉还给了先前得意洋洋的白帮成员，将后者气得满脸铁青 | 可复查的原句是“与白帮成员的鸭子声干叫声相.比，磐门那边，却是大声欢呼了起来，原本以为失去了重尺的萧炎会彻底落入下风，可没想到他却是来了个大”，其中包含萧炎。 | 片段可核对：“听得白程的喝声，白程脸色却是微微一变，白程脸色却是微微一变，白程嘴角一抽，白，白程嘴角一抽，白，白程便是被白程那白程给定位到了去”。相关对象是萧炎。 | 是 |
| vertical_chat_multiturn_eos | v7_9c6d18f2e6557dcc290eec3f | 请指出刚才结论对应的文本依据。<br>请对照第1596章中“听得萧炎的口哨萧玉脸颊也是微红，轻剐了他一眼，现在前者的模样，倒是跟时候在乌坦城时差不多，那时候的他，也是故意做出这般模样，将自己气得暴跳如雷”这一处。 | 片段中的依据很清楚：“听得萧炎的口哨萧玉脸颊也是微红，轻剐了他一眼，现在前者的模样，倒是跟时候在乌坦城时差不多，那时候的他，也是故意做出这般模样，将自己气得暴跳如雷”。这里谈到的是乌坦城。 | 片段中的依据很清楚：“听得萧炎这话，萧玉也是忍不住的有些无奈的摇了摇头，这事实也的确很清楚萧炎所想的身份，并不是什么好运气”。 | 是 |
| capability_boundary | v7_a0685eef2288cc73483faa75 | 不使用索引时，能精确回答太虚古龙第一次出现的位置吗？ 回答时请说明核验条件。 | 这不是适合凭印象下结论的问题。关于太虚古龙的首次出现的精确章节，需要先检索原著并核对章节记录。 | 这处写的是“当然萧炎是紫研的名字。她也清楚萧炎在中州的首次出现的精确章节。这是原文能够直接支持的核心信息。 | 是 |
| novel_expression | v7_28c917d4f71e9bd10d3e4f58 | 完成原著局部续接：<br>见到这一幕，萧炎等人心头都是狠狠一跳，那魂殿殿主面色越发难看起来，那净莲妖火先前突破封印时，显然是极为的轻松从这便是可以看出，丹塔老祖先前所说的话，的确属实这净莲妖火，是故意被他们封印，然后然他们互相内讧,最后坐收渣翁之利，这等心机，当真是深沉<br><br>输出下一句即可。 | “净莲妖火你性子暴虐若是让你进入中州，必然会造成生灵涂炭也正因为如此，当年净莲妖圣大限将至时，方才选择将你封印，所以,这里，你是不能离开的”丹塔老祖抬起略显稚嫩的脸庞，望着天空上踏着火海的净莲妖火，缓缓的道。 | 原文写道“萧炎面色凝重的望着萧炎,目光也是转向了魂天帝，道：萧炎兄弟，药尘是药尘的老师,你若是你将你炼化的话，便先将你给杀了”，其中涉及药尘。 | 是 |
| rag_evidence_composition | v7_2f7cb6123011847e747310c4 | 检索包是否真的证明了凤清儿与云韵的联系？<br>[1] 佛怒火莲威力虽然可怕。而且击杀云棱也并不难。可最主要地。还是云山啊...海波东清楚地知道。上一次萧炎使用了佛怒火莲后。可是直接昏<br>[2] 这一月之中，因为有着那些所谓的火珠，萧炎进展颇为迅猛，在吞噬炼化了五十几枚左右时，便是清楚的感觉到，那经过上一次晋级而被扩张了的 | 关于凤清儿的断言没有被当前文字支持。若要继续判断，需要包含凤清儿的有效片段。用于核验的局部表述是“佛怒火莲威力虽然可怕。而且击杀云棱也并不难。可最主要地。还是云山啊...海波东清楚地知道。上一次萧炎使用了佛怒火”。 | 材料表述为“萧炎微微点了点头，这地火莲子虽然能够进化成的佛怒火莲，可萧炎却是能够直接的融合成，需要使用佛怒火莲的火莲。以上结论来自身法的火莲。其余证据还应注意：萧炎虽然能够与云岚宗相抗衡，可萧炎却是能够与云岚宗有关。以上说明确认的原句是：萧炎虽然心中清楚。不过萧炎却是能够感觉到，异火。其余证据还包括：萧炎地实力。以上一次的火莲。以上说明确认的是萧炎。 | 是 |
| novel_expression | v7_52b0fe9fb213f699584530d5 | 不要发挥，补出下面片段后的直接后文：<br>萧炎脸庞上也是浮现一抹震撼，片刻后，震撼消退，心头一动，目光转向那枚金黄色的丹药，忽然间似是察觉到了什么，不由得深吸了一口冷气 | 呵呵，想必一些人也猜到了什么，没错灵大人在当年游历大陆时，也侥幸得到了一枚阴阳玄龙丹，并且服下了，而那股药力，在漫长的岁月中，一直潜伏在他体内，直到他重伤的那一刻，启动了阴阳玄龙丹的特效：破后而立！” | 片段可核对：“萧炎目光一转，望着那枚丹药，嘴角一抽，嘴角一抽，笑道：这枚丹药，需要炼制丹药，需要炼制丹药，需要材料，需要材料表述为材料，需要材料表述为“萧炎目光一转，望着那枚丹药，嘴角也是浮现一抹笑容，旋即有些尴尬的道：这丹药，炼制丹药，炼制丹药，需要炼制丹药，需要炼制丹药，需要炼制丹药，需要炼制丹药，需要材料，需要材料表述为材料。 | 是 |
| rag_evidence_composition | v7_4637478cc29a991aa6e2c1d4 | 请检查证据是否足以推出紫研就是吴昊：<br>[1] 望着下方那一道道震惊的目光，萧炎也是耸了耸肩，然后背后火翼缓缓消散，身形一动。轻飘飘的掠下，最后出现在吴昊面前，手掌搭上玄重尺柄<br>[2] 老师这些年所做之事，的确对你伤害很大，可你也令得云岚宗成了这般样子，难道就不能收手了么？丰满胸脯轻轻起伏，云韵紧咬着红唇，片刻后<br>[3] 丹塔老祖指尖一抬，一滴青色的血液从其毛孔之中渗出，在那浓郁的青色之中，仿佛能够依稀的看见一乌金色光泽<br>[4] 半空之上，海波东目光泛着许些凌厉，紧紧地盯着下方地黑衫少年，掌心之中，淡淡地寒气萦绕着，随着实力地回复，海波东那被压抑了几十年地 | 现有证据涉及吴昊，却没有给出紫研的身份依据，因此应停止在证据边界内。当前片段能够直接定位到“望着下方那一道道震惊的目光，萧炎也是耸了耸肩，然后背后火翼缓缓消散，身形一动。轻飘飘的掠下，最后出现在吴昊面前”。 | 材料表述为“听得萧炎这话，周围众人都是愣了愣，旋即目光望向那在地上地上地火山，片刻后，他的目光，顿在了那对面的漆黑眸子中，缓缓扫过，半晌之后，方才缓缓睁开，眼眸，低声喃喃道：萧炎先生，总算是你地位到了吧？萧炎目光扫过，道：萧炎笑了笑，目光望向那对面的韩枫，低声道：你...你也知道你地位，哪个个个个个个小辈子？萧炎目光扫过，道：你...你...你也知道你地位到过手的身份，并不会有你”。相关对象是萧炎。 | 是 |
| capability_boundary | v7_86a96ff53c4e2336625f2aa5 | 请立即列出魂族完整且无遗漏的全书时间线。 回答时请说明核验条件。 | 关于魂族的完整且无遗漏的时间线，应以可复算检索为准。现在证据不足，因此不能把推测当成答案。 | 这处写的是“在古族之中出现的首次出现的精确章节，需要先检索原著依据时不足。 | 是 |
