# SFT v7 公开诊断

Checkpoint step：`2000`

全量公开 teacher-forced loss：`3.294267`；perplexity：`26.958`

开放表达只提供自动退化信号；AI 质量审核与独立真人审核均为待完成，不以模糊字符串相似度作为硬门。

| 维度 | 数量 | Required | Forbidden违规 | Keypoint | 已知误拒 | EOS | 空答 | 截断 | 4gram退化 | 元话术 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| core_facts_and_corrections | 108 | 49.1% | 0.0% | 47.8% | 3.7% | 81.5% | 0.0% | 18.5% | 47.2% | 0.0% |
| single_evidence_qa | 192 | 75.3% | — | — | — | 81.2% | 0.0% | 18.8% | 40.1% | 0.0% |
| rag_evidence_composition | 84 | 78.3% | — | 78.3% | — | 73.8% | 0.0% | 26.2% | 59.5% | 0.0% |
| vertical_chat_multiturn_eos | 108 | 80.6% | 0.0% | — | — | 89.8% | 0.0% | 10.2% | 37.0% | 0.0% |
| novel_expression | 78 | 71.7% | — | — | — | 79.5% | 0.0% | 20.5% | 38.5% | 0.0% |
| capability_boundary | 30 | 90.0% | — | — | — | 96.7% | 0.0% | 3.3% | 16.7% | 0.0% |

## 自动候选硬门

这些结果是可复算的词面/行为代理；不把 EM、字符 F1 或关键词命中夸大为语义正确或证据支持。

| 门 | 阈值 | 值 | 通过 | 来源 |
|---|---|---|---|---|
| core_required_keypoint_proxy | >= 0.80 | 0.4907 | 否 | public core cases; keypoint pass when present, otherwise required-term pass |
| core_known_misrefusal | <= 0.05 | 0.0370 | 是 | public core cases with evaluation.known_fact=true |
| single_evidence_lexical_or_support_proxy | normalized_char_multiset_f1 >= 0.90 OR required/keypoint support proxy >= 0.90 | {"normalized_char_multiset_f1": 0.24107446113731154, "required_keypoint_support_proxy": 0.7531645569620253} | 否 | positive public single-evidence cases; lexical F1 is a non-semantic proxy, support proxy checks only declared required/keypoint strings |
| single_evidence_insufficient_stop | >= 0.90 | 0.0000 | 否 | public single-evidence cases marked needs_evidence and not sufficient |
| rag_gold_keypoint_proxy | >= 0.80 | 0.7826 | 否 | positive public RAG cases; declared keypoint/required-term pass proxy |
| rag_required_support_proxy | >= 0.90 | 0.7826 | 否 | positive public RAG cases; required-term presence only, not citation support |
| chat_required_routing_proxy | >= 0.90 | 0.8056 | 否 | public chat declared required-term routing proxy |
| chat_multiturn_carryover_proxy | >= 0.80 | 0.7778 | 否 | public task_family=chat_multiturn required/keypoint presence proxy |
| chat_eos | >= 0.95 | 0.8981 | 否 | public chat generation EOS termination |
| chat_empty | = 0 | 0.0000 | 是 | public chat normalized empty-answer rate |
| chat_meta_phrase | <= 0.01 | 0.0000 | 是 | public chat frozen meta-phrase marker rate |
| chat_cross_task_refusal_proxy | <= 0.05 | 0.0000 | 是 | public in-domain chat refusal-marker rate (cross-task proxy) |
| expression_mechanical_repetition | <= 0.10 | 0.3846 | 否 | public expression repeated-four-gram mechanical degeneration |
| boundary_correct_stop | >= 0.90 | 0.0667 | 否 | public capability-boundary insufficient-evidence cases |
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
| core_facts_and_corrections | v7_760b18208f0aa1fe1b086132 | 关于原著，丹塔是什么？ 只说有原著依据的部分。 | 丹塔是一个受到许多炼药师推崇的自由组织。对新读者而言，核心关系可据此理解。 | 丹塔是丹塔的首次出现的精确章节，需要检索原著依据。 | 是 |
| core_facts_and_corrections | v7_934683dc2afabeac9e7b004a | 读者追问第585章里的陨落心炎，应怎样准确作答？ | 原文写道“随着接过药老先前的工作，萧炎这才明白，与那陨落心炎对抗的消耗，是何等的庞大，而且，或许是由于他与药老实力的差距，虽然青莲地心”，其中涉及陨落心炎。 | 文本中出现“随着陨落心炎的出现，陨落心炎也是逐渐的变得极为缓慢，陨落心炎的融合，需要补充可复，需要补充相关材料，需要补充相关材料。 | 是 |
| single_evidence_qa | v7_9fca6c78b778be792d8acb22 | 请阅读证据并回答其中对青莲地心火的描写：<br>嘴角掀起一抹冷意，一丝丝极淡的青色火苗，在斗.气的包裹下，将重尺包裹而进，有青莲地心火之助，寻常水属性斗气，在他面前几乎仅仅只能发挥七八层的威力 | 材料表述为“嘴角掀起一抹冷意，一丝丝极淡的青色火苗，在斗.气的包裹下，将重尺包裹而进，有青莲地心火之助，寻常水属性斗气，在他面前几乎仅仅”，这里写到青莲地心火。 | 材料表述为“萧炎手掌轻握，青色火翼微微振动，犹如火山般的火山，顺着尺子中暴射而出，将玄重尺包裹在尺身之上，顿时，火山尖，火山暴射而出，将玄重尺的尺身子尽数包裹，直冲向萧炎。 | 是 |
| vertical_chat_multiturn_eos | v7_bc57a230c23395434c28a9c7 | 请像小说导读一样解释这段里的药尘：<br>闻言。药老老脸一红。有些恼羞成怒的挥了挥袖子：你以为我也像你这般不务正业么 | 原文写道“闻言。药老老脸一红。有些恼羞成怒的挥了挥袖子：你以为我也像你这般不务正业么”，其中涉及药尘。 | 文本中出现“闻言。萧炎也是忍不住的有些疑惑。目光盯着药老。沉吟道：我知道你是什么时候。便先去找你。先去找你。先去找你。先去找你。先去找你。先去找你。先去找你。先去找你。先去找你。先去找你。先去找你。先去找你。请你去找你。先去找你。请你去找你。先去找你。 | 是 |
| single_evidence_qa | v7_4177ee8e607cf3c29e57f354 | 请阅读证据并回答其中对萧炎的描写：<br>与白帮成员的鸭子声干叫声相.比，磐门那边，却是大声欢呼了起来，原本以为失去了重尺的萧炎会彻底落入下风，可没想到他却是来了个大翻盘，反而将本该占据上风的白程打得满地滚，这意外的情况，让得一脸紧张的磐门成员忍不住的发出阵阵欢呼，一些脾气冲动之人，更是将嘲讽的嬉笑骂声奉还给了先前得意洋洋的白帮成员，将后者气得满脸铁青 | 可复查的原句是“与白帮成员的鸭子声干叫声相.比，磐门那边，却是大声欢呼了起来，原本以为失去了重尺的萧炎会彻底落入下风，可没想到他却是来了个大”，其中包含萧炎。 | 文本中出现“白程在白程手中重尺上下，那白程却是犹如一只白程般白程一般，白程脸色微变，白程脸色却是变得有些阴沉，白程那白程，白程虽然他对白程，可却是能够知道，这白程是白程”，其中涉及内院。 | 是 |
| vertical_chat_multiturn_eos | v7_9c6d18f2e6557dcc290eec3f | 请指出刚才结论对应的文本依据。<br>请对照第1596章中“听得萧炎的口哨萧玉脸颊也是微红，轻剐了他一眼，现在前者的模样，倒是跟时候在乌坦城时差不多，那时候的他，也是故意做出这般模样，将自己气得暴跳如雷”这一处。 | 片段中的依据很清楚：“听得萧炎的口哨萧玉脸颊也是微红，轻剐了他一眼，现在前者的模样，倒是跟时候在乌坦城时差不多，那时候的他，也是故意做出这般模样，将自己气得暴跳如雷”。这里谈到的是乌坦城。 | 片段中的依据很清楚：“听得萧炎这话，萧玉顿时有些无奈的摇了摇头，这事实不足以和萧家有关的萧家有关。 | 是 |
| capability_boundary | v7_a0685eef2288cc73483faa75 | 不使用索引时，能精确回答太虚古龙第一次出现的位置吗？ 回答时请说明核验条件。 | 这不是适合凭印象下结论的问题。关于太虚古龙的首次出现的精确章节，需要先检索原著并核对章节记录。 | 这处写的是“在远古古龙的神秘人身上。它能够直接对太虚古龙的关系；能够直接支持的核心信息。 | 是 |
| novel_expression | v7_28c917d4f71e9bd10d3e4f58 | 完成原著局部续接：<br>见到这一幕，萧炎等人心头都是狠狠一跳，那魂殿殿主面色越发难看起来，那净莲妖火先前突破封印时，显然是极为的轻松从这便是可以看出，丹塔老祖先前所说的话，的确属实这净莲妖火，是故意被他们封印，然后然他们互相内讧,最后坐收渣翁之利，这等心机，当真是深沉<br><br>输出下一句即可。 | “净莲妖火你性子暴虐若是让你进入中州，必然会造成生灵涂炭也正因为如此，当年净莲妖圣大限将至时，方才选择将你封印，所以,这里，你是不能离开的”丹塔老祖抬起略显稚嫩的脸庞，望着天空上踏着火海的净莲妖火，缓缓的道。 | 原文写道“见到这一幕，萧炎面色也是微微一变，面色也是变得凝重了起来，没想到这净莲妖火竟然还敢如此轻易的”，其中涉及药尘。 | 是 |
| rag_evidence_composition | v7_2f7cb6123011847e747310c4 | 检索包是否真的证明了凤清儿与云韵的联系？<br>[1] 佛怒火莲威力虽然可怕。而且击杀云棱也并不难。可最主要地。还是云山啊...海波东清楚地知道。上一次萧炎使用了佛怒火莲后。可是直接昏<br>[2] 这一月之中，因为有着那些所谓的火珠，萧炎进展颇为迅猛，在吞噬炼化了五十几枚左右时，便是清楚的感觉到，那经过上一次晋级而被扩张了的 | 关于凤清儿的断言没有被当前文字支持。若要继续判断，需要包含凤清儿的有效片段。用于核验的局部表述是“佛怒火莲威力虽然可怕。而且击杀云棱也并不难。可最主要地。还是云山啊...海波东清楚地知道。上一次萧炎使用了佛怒火”。 | 材料表述为“萧炎地确认了云山。这次的原文是加玛帝国。以上结论来自由云岚宗的局部表述为：萧炎地实力。以上结论来自已经定位到云岚宗。以上结论来自已经定位到。以上说明只的说明。 | 是 |
| novel_expression | v7_52b0fe9fb213f699584530d5 | 不要发挥，补出下面片段后的直接后文：<br>萧炎脸庞上也是浮现一抹震撼，片刻后，震撼消退，心头一动，目光转向那枚金黄色的丹药，忽然间似是察觉到了什么，不由得深吸了一口冷气 | 呵呵，想必一些人也猜到了什么，没错灵大人在当年游历大陆时，也侥幸得到了一枚阴阳玄龙丹，并且服下了，而那股药力，在漫长的岁月中，一直潜伏在他体内，直到他重伤的那一刻，启动了阴阳玄龙丹的特效：破后而立！” | 文本中出现“萧炎目光望着萧炎那枚丹药，眉头也是微微皱了起来，道：这枚丹药，炼制丹药，需要炼制丹药，需要炼制丹药，需要炼制丹药，需要炼制丹药，需要炼制丹药，需要炼制丹药，需要炼制丹药，需要炼制材料，需要炼制材料。 | 是 |
| rag_evidence_composition | v7_4637478cc29a991aa6e2c1d4 | 请检查证据是否足以推出紫研就是吴昊：<br>[1] 望着下方那一道道震惊的目光，萧炎也是耸了耸肩，然后背后火翼缓缓消散，身形一动。轻飘飘的掠下，最后出现在吴昊面前，手掌搭上玄重尺柄<br>[2] 老师这些年所做之事，的确对你伤害很大，可你也令得云岚宗成了这般样子，难道就不能收手了么？丰满胸脯轻轻起伏，云韵紧咬着红唇，片刻后<br>[3] 丹塔老祖指尖一抬，一滴青色的血液从其毛孔之中渗出，在那浓郁的青色之中，仿佛能够依稀的看见一乌金色光泽<br>[4] 半空之上，海波东目光泛着许些凌厉，紧紧地盯着下方地黑衫少年，掌心之中，淡淡地寒气萦绕着，随着实力地回复，海波东那被压抑了几十年地 | 现有证据涉及吴昊，却没有给出紫研的身份依据，因此应停止在证据边界内。当前片段能够直接定位到“望着下方那一道道震惊的目光，萧炎也是耸了耸肩，然后背后火翼缓缓消散，身形一动。轻飘飘的掠下，最后出现在吴昊面前”。 | 材料表述为“望着那在地上地上，萧炎脸色顿时变得凝重了起来，目光缓缓扫过，片刻后，他发现，前者那黑尺，赫然便是当年在迦南学院地域中地域”，其中涉及萧炎。 | 是 |
| capability_boundary | v7_86a96ff53c4e2336625f2aa5 | 请立即列出魂族完整且无遗漏的全书时间线。 回答时请说明核验条件。 | 关于魂族的完整且无遗漏的时间线，应以可复算检索为准。现在证据不足，因此不能把推测当成答案。 | 这处写的是中州的核验片段。 | 是 |
