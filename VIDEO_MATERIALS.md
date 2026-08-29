# 项目视频素材索引

## 用途

本文件维护“手搓 GPT”项目从预训练、SFT 到最终模型的可复现素材。项目视频应优先引用这里列出的里程碑文件，不直接使用可能被后续运行覆盖的临时输出。

## 留档规则

每次模型结构、数据、训练目标或生成策略发生实质变化后，都建立一个新的里程碑编号，并保留以下材料：

1. 模型阶段、参数配置、训练数据和目标函数。
2. 独立命名的 checkpoint，不覆盖上一阶段模型。
3. Loss、Top-1、Top-5、重复率等指标及计算口径。
4. 同一组固定 prompt 的完整输入与输出。
5. Loss 曲线、CSV 数据表和机器可读 JSON 报告。
6. 本阶段做了什么、改善了什么、仍有什么问题。
7. 关键文件 SHA-256，确认视频制作时材料没有变化。

所有 SFT 文件使用 `sft_` 前缀，不能覆盖 `gpt_stage5_pretrain_step10000_best.pt`。后续比较必须使用同一组固定 prompt、相同最大生成长度和明确记录的采样参数。

注意：`checkpoints/` 当前被 `.gitignore` 排除，模型 checkpoint 只保存在本机，不会随 GitHub 提交上传；文档、指标和图表可以进入版本管理。项目收尾和视频制作前，需要再把归档 checkpoint 复制到用户指定的外部备份位置，并用本里程碑 SHA-256 复核。

## 里程碑索引

| 编号 | 阶段 | 状态 | 核心材料 |
|---|---|---|---|
| M001 | 字符级领域预训练，Stage 5 Step 10000 | 已归档 | [里程碑说明](reports/milestones/001_pretrain_stage5_step10000/README.md) |
| M002 | 字符级模型 SFT 数据集 v1 | 已完成 | [数据设计与质量说明](reports/milestones/002_sft_dataset_v1/README.md) |
| M002a | 50条蒸馏式 SFT 试验数据 | 已完成 | [试验数据说明](reports/milestones/002a_sft_pilot50_v1/README.md) |
| M002b | SFT前固定测试题基线 | 已完成 | [完整基线结果](reports/milestones/002b_pre_sft_baseline/README.md) |
| M003a | SFT 20步安全试跑 | 已完成 | [试跑结果](reports/milestones/003a_sft_smoke20/README.md) |
| M003b | SFT数据扩充至100条 | 已完成 | [扩充结果](reports/milestones/003b_sft_data100/README.md) |
| M003c | SFT数据扩充至500条 | 已完成 | [扩充与校验结果](reports/milestones/003c_sft_data500/README.md) |
| M003d | 高质量SFT数据扩充至1000条 | 已完成 | [构造与质量报告](reports/milestones/003d_sft_hq1000/README.md) |
| M003e | 1000条数据的20步安全试跑 | 已完成 | [试跑结果](reports/milestones/003e_sft_hq1000_smoke20/README.md) |
| M003f | 1000条数据的800步SFT与正式评估 | 已完成 | [训练、曲线与前后对比](reports/milestones/003f_sft_hq1000_step800/README.md) |
| M003g | 平衡版SFT v3数据重构 | 已完成 | [任务比例、质量边界与训练协议](reports/milestones/003g_sft_balanced_v3/README.md) |
| M003h | 平衡版SFT v3的20步安全试跑 | 已完成 | [试跑结果](reports/milestones/003h_sft_balanced_v3_smoke20/README.md) |
| M003i | 平衡版SFT v3的800步训练与公平评估 | 已完成 | [Loss、六类指标与固定10题](reports/milestones/003i_sft_balanced_v3_step800/README.md) |
| M003 | 字符级模型 SFT v1 | 已完成 | M003a至M003i完整记录，最终选用平衡版第400步实验模型 |
| M004 | 手写BPE与10,000步从零预训练 | 已完成 | [词表、压缩率、预训练与样本演化](reports/milestones/004_bpe_pretrain/README.md) |
| M005 | BPE模型800步SFT与字符版公平对比 | 已完成 | [Loss、六类指标与固定10题](reports/milestones/005_bpe_sft/README.md) |
| M006 | v4正式语料与本机810万参数预训练 | 已完成 | [语料审核、Loss、固定10题与完整归档](reports/milestones/006_v4_local_pretrain/README.md) |
| M007 | v4低学习率续训至6000步与全局评测Harness | 已完成 | [续训参数、曲线、固定小说样本与质量门槛](reports/milestones/007_v4_continue6000/README.md) |
| M008 | 3000条教师SFT候选的保守修复与审计 | AI审核已合并，training-ready 2999条 | [结构指标、风险分层与复核协议](reports/milestones/008_sft_v4_teacher_repair/README.md) |
| M009 | v4 SFT 20步安全试跑 | 已完成 | [SFT张量、Loss与样本边界](reports/milestones/009_v4_sft_smoke20/README.md) |
| M010 | v4 SFT 500步正式小跑 | 已完成 | [Loss曲线、固定10题与质量边界](reports/milestones/010_v4_sft_step500/README.md) |
| M011 | v4 混合聊天SFT 2000步实验 | 已完成 | [混合数据、Loss曲线和小说/非小说样本](reports/milestones/011_v4_mixed_chat_sft/README.md) |
| M012 | v5定向补强SFT数据与100步安全试训 | 已完成 | [补强配额、试训Loss和分类型诊断](reports/milestones/012_v5_repair_sft/README.md) |
| M013 | v5.1去数学、污染修复与严格零提示重合SFT | 已完成，行为门未通过 | [数据审计、严格评估与累计5000步结果](reports/milestones/013_v5_1_no_math_sft/README.md) |
| M014 | v5.2已知实体路由修复与可恢复训练Harness | 核心实体修复通过，综合聊天未达标 | [数据、训练、隐藏评估与上限路线](reports/milestones/014_v5_2_entity_routing/README.md) |
| M015 | Stage A分词器与4M/8M/14M容量实测 | 已完成，14M质量最高、8M效率甜点 | [BPC表格、曲线、样本与决策](reports/milestones/015_scaling_stage_a/README.md) |
| M016 | 1488万参数、BPE 3000正式预训练 | 已完成，选择Step 5750进入SFT | [24点BPC曲线、候选对照与固定样本](reports/milestones/016_formal_pretrain_14m/README.md) |
| M017 | 1488万模型的SFT数据准入审查 | 已完成，v5.2.2需重构后才能正式训练 | [全量风险、Token规模、维度缺口与v6配额](reports/milestones/017_sft_data_readiness_audit/README.md) |
| M018 | 一万条SFT v6、独立审计与2000步正式训练 | 训练完成，行为门未通过 | [数据门槛、Loss曲线、公开诊断与失败样本](reports/milestones/018_sft_v6_10000/README.md) |
| M019 | 单本小说纯预训练能力审计 | Step 5750形成语言基座；当前配置进入实用平台期 | [三Checkpoint对照、固定续写、Cloze与AI复核](reports/milestones/019_pretrain_capability_audit/README.md) |
| M020 | 小说垂直SFT v7 | 实验完成；自动门失败、外部复核pending、无发布候选 | [数据、曲线、节点对比、保持评估与不发布结论](reports/milestones/020_sft_v7_vertical/README.md) |
| M021 | SFT v7.1语义密度与遗忘控制Canary | 有限问答容量已证明；0.25回放为最优诊断节点，但保持门差1条、无候选 | [审计、Canary、三档回放、语义复核与A0计划](reports/milestones/021_sft_v7_1_canary/README.md) |

## M020 已冻结视频素材

| 素材 | 已完成事实 | 文件 |
|---|---|---|
| v7数据与编码 | 10,000条，8000/800/600/600；清单SHA `422c35fa...`，Train+Val/Public张量SHA `81a06541...` / `bf287053...` | [M020说明](reports/milestones/020_sft_v7_vertical/README.md)、[编码报告](reports/milestones/020_sft_v7_vertical/tensor_report.json) |
| Step 5750 public基线 | Loss 5.86808865；EOS 7.1667%；空回答1.1667%；截断92.8333%；机械重复85.5% | [公开评估](reports/milestones/020_sft_v7_vertical/public_eval_step05750.md) |
| Step 5750固定16题 | 0/16 EOS，16/16达到128 Token上限；保留完整输入与输出 | [完整16题](reports/milestones/020_sft_v7_vertical/fixed_samples_step05750.md) |
| 20步smoke | 两阶段切换通过；Validation Loss 5.897454→5.556195；public/sealed训练消耗0 | [训练报告](reports/milestones/020_sft_v7_vertical/smoke_train_report.json)、[固定16题](reports/milestones/020_sft_v7_vertical/fixed_samples_smoke20.md) |
| 2000步正式训练 | Validation Loss从5.897454降至3.356181；每250步曲线已归档 | [正式报告](reports/milestones/020_sft_v7_vertical/formal_train_report.json)、[SVG曲线](reports/milestones/020_sft_v7_vertical/sft_v7_loss_curve.svg)、[CSV](reports/milestones/020_sft_v7_vertical/sft_v7_loss_curve.csv) |
| Public节点对比 | Step500/1000/1500/2000 Loss为4.034095/3.611000/3.419921/3.294267；所有自动行为门失败 | [汇总表](reports/milestones/020_sft_v7_vertical/checkpoint_comparison.md) |
| 固定16题AI辅助复核 | 基本通过0/0/1/0；严重重复13/11/8/15，Step2000明显反弹 | [逐题复核](reports/milestones/020_sft_v7_vertical/fixed_samples_milestone_review.md) |
| 预训练保持 | BPC退化6.007%/8.995%/10.217%/12.084%，四节点均只有13/16续写非空 | [Step500](reports/milestones/020_sft_v7_vertical/pretrain_retention_step00500.md)、[Step1000](reports/milestones/020_sft_v7_vertical/pretrain_retention_step01000.md)、[Step1500](reports/milestones/020_sft_v7_vertical/pretrain_retention_step01500.md)、[Step2000](reports/milestones/020_sft_v7_vertical/pretrain_retention_step02000.md) |
| 最终决定 | 严格候选`[]`，必需产物缺失0、完整性错误0，`release_ready=false` | [机器可读汇总](reports/milestones/020_sft_v7_vertical/checkpoint_comparison.json)、[SHA索引](reports/milestones/020_sft_v7_vertical/SHA256SUMS.md) |

视频表述必须区分三件事：纯预训练Step5750只是最低可用小说语言基座，20步smoke只证明训练链路安全可运行，2000步正式SFT虽改善Loss、EOS与截断，却没有通过行为和保持门。Step1000是保守诊断候选，Step1500是行为较优研究候选，Step2000已排除；它们都不是发布候选，独立真人复核仍未完成。

## M021 已冻结视频素材

| 素材 | 可展示事实 | 文件 |
|---|---|---|
| v7语义密度审计 | 固定开头49.95%，裸核心问0/18，核心相对密度0.6188，证据高复制66.01% | [审计报告](reports/milestones/021_sft_v7_1_canary/semantic_density_audit.md) |
| Canary数据与张量 | 8个fact ID、64条Train、16条Dev/Selection；assistant-only与EOS通过 | [数据报告](reports/milestones/021_sft_v7_1_canary/canary_data_report.md)、[张量报告](reports/milestones/021_sft_v7_1_canary/canary_tensor_report.md) |
| 高LR容量证明 | Step375严格问答100%/100%，但BPC退化43.64%、续写只有9/16非空 | [问答](reports/milestones/021_sft_v7_1_canary/canary_best375_generation_eval.md)、[保持](reports/milestones/021_sft_v7_1_canary/pretrain_retention_best375.md) |
| 三档联合回放 | 0.50/0.25/0.10的问答、BPC、非空、机械退化完整对照 | [机器可读表](reports/milestones/021_sft_v7_1_canary/canary_tradeoff.csv)、[里程碑结论](reports/milestones/021_sft_v7_1_canary/README.md) |
| 0.25当前最优诊断 | 严格87.50%/68.75%，命题95.31%/87.50%，BPC退化2.33%，15/16非空 | [问答](reports/milestones/021_sft_v7_1_canary/canary_replay_w025_step00400_generation_eval.md)、[保持](reports/milestones/021_sft_v7_1_canary/pretrain_retention_replay_w025_step00400.md) |
| 指标审核 | 13条严格失败中8条语义等价、5条真错误；exact不能冒充语义 | [逐条语义复核](reports/milestones/021_sft_v7_1_canary/canary_replay_w025_semantic_review.md) |
| 下一训练计划 | A0 3920条，结构化命题评分，回放0.20/0.25/0.30，RAG并行 | [M021训练计划](docs/m021_sft_v7_1_training_plan.md) |

推荐视频核心表述：

> 64条高质量问答已经证明小模型不是“只能续写”：高学习率能把未见改写也答对，却会迅速忘掉小说语言。加入预训练Train Token回放后，0.25权重把BPC退化压到2.33%，命题问答达到95.31%/87.50%，但仍有一条小说提示立刻EOS。真正的训练目标不是某一个Loss最低，而是问答语义、关系方向、EOS、重复和预训练保持在同一个checkpoint同时过门。

## 视频中的核心对比表

| 指标 | M001 预训练后 | M003 字符级 SFT 后 | M005 BPE+SFT 后 |
|---|---:|---:|---:|
| 验证 Loss | 3.0863 | 3.4099（仅回答区，不可直接同比） | 预训练4.8036/BPE Token；SFT 3.7711（回答区） |
| Top-1 | 42.71% | 不适用 | 待测 |
| Top-5 | 62.86% | 不适用 | 待测 |
| 相邻字符重复率 | 1.30% | 未作为SFT主指标 | 待测 |
| 问答相关性 | 基本没有 | 精确38/100 | 精确36/100；包含答案42/100 |
| EOS停止率 | 未实现 | 100/100 | 100/100 |
| 固定10题输出 | 已保存 | 已保存四模型完整对比 | 已保存字符/BPE四状态完整对比 |

## 当前路线决定

Stage 5 已经形成小说语言能力，但训练目标仍是小说续写。继续在同一字符级小说语料上增加预训练步数，只会带来逐渐变小的续写收益，不能直接解决问答目标不匹配。

字符级SFT与BPE路线均已完成。BPE将654万字符压缩为406.8万Token，在同一Block Size下扩大约1.61倍有效字符上下文，但1000条SFT上的严格正确率未超过字符版。下一阶段不应盲目增加Step，而应增加高质量独立事实、自然对话与多样化回答，并单独改进清洗语料中的站点/作者噪声。

M006已经按章节组重新冻结612万字符正式语料，并在Apple M4上完成810万参数、3000步从零预训练。M007再从M006的Step 2600验证最优权重以低学习率续训到累计6000步，最佳验证Loss降至4.7759/BPE Token，独立Test Loss为4.7476。但发布候选在固定小说提示词中出现连续12个“试”，被自动门槛标记为`REVIEW`；人工抽样也存在语法、人物关系和情节跳跃。这正是Harness阻止“只看Loss宣布成功”的案例。下一阶段进入v4 SFT时继续保留Loss、自动门槛和人工语义复核三层协议。

M008直接修复而非重新生成3000条教师数据，保留了生成成本并完成结构、切分、证据和分词审计。第一次机器门槛通过后，内容抽样仍发现截断复制、伪一句话和不自然澄清，因此进行了第二轮修复；证据定位提高到2999/3000，最长序列降到149 Token。继续修复原因题语病和2条修为事实错误后，600条评估记录仍未真人批准。视频中应保留这一“机器门槛通过不等于高质量训练集”的阶段，人工验收后再进入20步SFT安全试跑。

进一步核验发现冻结语料同时保留正式分隔符标题和少量旧版内嵌标题，旧定位器误把两者都当成章节边界。统一采用正式v4标题后，完整数据累计重绑定262条旧章节号。随后又发现字面索引会把人物“云山”命中在“高耸入云山峰”中，说明索引一致也不能自动证明人物首次登场。最终把1067条涉及首次、先后、最多、主角和全书缺失的任务改成当前证据可直接回答的问题，全局断言扫描从172条降为0条。这个过程适合作为视频中的数据治理案例：局部证据存在、甚至字面索引一致，都不能证明真正的语义断言；发现指标口径错误后必须回到任务定义重做。

为完成最后的审核环节，项目新增仅在本机运行的审核页面，按“406条任务改写、194条低风险”排序展示问题、答案和证据，并将决定独立保存。经用户授权，Codex AI随后完成600/600审核，521条直接通过、79条修改后通过；修改中实际发现了错误修为、主客体颠倒和病句，证明审核不是批量盖章。候选哈希变化会使旧决定失效，日志不记录审核人和正文。随后审核决定被合并进AI冻结副本：79条问答修改生效，19条重复问题做轻量去重，1条训练样本因证据章节漂移被排除，形成2999条training-ready候选。视频应明确展示治理边界：这是如实署名的AI审核，不是独立真人签字。

M009把2999条training-ready候选编码成v4 BPE聊天张量，并从M007最佳预训练checkpoint执行20步SFT安全试跑。Loss只监督回答和EOS，不监督用户问题；验证Loss从7.5522降到5.0587，说明链路能学习。两条监控样本仍输出“第一个不说的”和““嘭！””，这应作为视频里的重要边界：20步只证明训练流程可行，不证明模型已经会回答。

M010把同一SFT链路推进到500步，并固定10个监控问题每100步生成一次。验证Loss从6.7152降到1.5823，最佳点在step 400的1.3442；这说明SFT目标已经被模型快速吸收。固定样本同时显示格式能力和内容能力分离：模型开始短答、停止，但仍会答错人物和章节标题。视频中可以用这一阶段展示“Loss下降 ≠ 问答达标”，下一步应继续扩大正式SFT步数并使用保留测试集做最终一次性评估。

M011回应了M010里“怎么全是数字、一点都不聊天”的问题：在2999条小说SFT之外加入3000条通用聊天、数学、项目解释、实时能力边界、学习计划、指令改写、诚实未知和领域切换样本，合计5999条重新编码训练。2000步后验证Loss降到1.6600，模型开始能对未知实体说资料不足，也会对天气问题给出能力边界，而不是一味回答章节号。继续训练到5000步后，100个验证batch复核显示step5000验证Loss低于step3500，但样本仍暴露过度拒答和基础推理失败：天气和学习计划更像聊天，一加一仍错，萧炎/药尘/异火被答成资料不足。视频中可以把这一阶段作为“数据分布改变模型行为，但不是魔法”的案例：SFT先学回答格式，再逐步逼近可靠内容；总Loss下降也可能掩盖某些能力退化。

随后建立M011分类型公开诊断评估，固定30题拆成小说人物、小说事实、证据判断、能力边界、基础数学和通用聊天。规则收紧后，step5000 latest低温结果只有5/30，EOS为29/30；小说人物、小说事实和基础数学均为0/5，通用聊天为3/5。这个素材很适合放进视频：模型看起来“能说话”以后，真正的验收才开始；评估规则本身也要审核，否则宽松规则会把错误输出误判成通过。

M012把M011失败项直接转成数据工程动作：在原5999条混合SFT上新增2000条定向补强样本，覆盖已知小说人物、事实锚点、证据实体匹配、基础数学、实时能力边界和项目概念解释。100步安全试训验证Loss从6.3811降到5.1405，但30题诊断仍为5/30，样本还会重复萧炎和串题。这段素材适合说明“补数据不是一贴灵药”：数据准备通过只是下一轮正式训练的起点，效果要等2000/5000步后再验收。

M012继续从同一M007预训练checkpoint正式SFT到2000步，验证Loss降至1.6750，但严格30题诊断只有6/30。更有价值的是评估器本身被继续审核：旧规则会把“第3000章的标题是《收场》”误判为第300章标题正确，收紧后分数从8/30回落到6/30。视频中可以把这一段作为“评估标准也会说谎”的案例：模型会钻宽松指标的空子，真正的验收要看精确规则和样本明细。

M013进一步发现，旧公开30题有21题与M012数据精确重合，其中15题位于train；这说明“隐藏test没有参与训练”不等于“公开诊断没有泄漏”。因此本阶段从训练目标中彻底移除数学及相关元提示污染，把第六类评估改为指令遵循，并冻结“完整提示及其子串包装重合为0、修复语义组不跨split”的v2口径；同主题的不同问法仍有意保留，所以不是完全盲测。最终数据缩减为4676条（3629/520/527），累计5000步latest由step2000的4/30提高到7/30，小说事实和证据判断各2/5、通用聊天3/5，但EOS从29/30降到25/30，人物、能力边界和指令遵循仍为0/5。这个阶段适合展示两个重要结论：数据更干净、Loss更低仍不等于行为达标；失败评估同样有工程价值，因为它阻止团队继续盲目堆step或发布一个不可靠模型。

M015把“这本小说到底适合多大模型”从经验判断改成实测。先用同一4M主体比较1000/2000/3000 merges，并用可跨分词器比较的BPC选出3000；再让420万、836万和1488万模型各看同样409.6万训练Token。验证BPC依次为5.4969、5.3820、5.3239，说明14M在当前范围质量最高，但相对8M只改善约1.08%，耗时却从22.5分钟增到38.2分钟。视频应同时展示三联曲线和仍不流畅的1000步样本，强调“选型胜出不等于模型已经训练好”；整个选型阶段test保持封存。

M016把选出的1488万参数、BPE 3000模型从随机权重正式训练6000步。验证BPC从Step 250的5.9149降到Step 5750的3.7612，训练耗时约91.9分钟，共暴露2457.6万Token；Step 6000虽取得更低原始BPC 3.7532，但改善不足0.01，独立固定样本又因过早EOS导致平均长度85.4而被Harness标记为`REVIEW`。项目因此选择Step 5750进入SFT，并保留Step 6000作备选。这段素材适合展示“最低Loss也不自动等于最佳发布候选”：稳定阈值、机械退化门和人工文本复核必须共同决策；test仍未启封。

M017在正式SFT前重新审查4516条v5.2.2数据。虽然文件结构、BPE 3000编码和精确去重都通过，但全量复算发现2340条pending、995条元证据包装、1299条风格前缀模板、1600条高重复答案、2条确定错章和17条残缺标点；多轮、长上下文和长回答均为0。更关键的是，一条曾被旧AI决定批准的“第167章下一章=第162章”被重新确认错误。视频应保留这一反转：training-ready文件名、机器门禁甚至批量AI审核都不是最终质量证明，训练前必须重新验证原始结论。正式SFT目标因此改为8000至12000条高质量数据，并新建真正封存的未见事实测试。

M018按新目标构造10000条SFT v6并完成独立审计、BPE多轮编码、2步冒烟和2000步正式训练。结构指标全部通过，训练集337900个监督Token无放回覆盖一轮，完整验证Loss从6.1353降至2.5940；然而公开生成完全匹配仍为0，模型会把自然聊天的“可以先……”模板和小说抽取答案串在一起。项目因此选择公开指标较好的Step2000作为当前轮次产物，但不把它标记为发布候选。这一阶段适合在视频中直接展示“10K、零泄漏、Loss下降仍不等于高质量”：数据规模与结构合格之后，还要审查回答分布、模板多样性和真实生成。

M019回到第一性原理，先暂停宽泛SFT，单独审计纯预训练模型到底学到了什么。56条探针只来自正式train/validation，test继续封存；同一Harness比较Step 250、5750和6000。Step 5750的Validation BPC比Step 250改善36.41%，固定窗口Top-1由7.81%升到24.17%，16条续写没有机械退化，但Codex AI流畅度和局部连贯度只有2.3125/5与2.0625/5，说明它只是最低可用语言基座。Step 6000的BPC只再改善0.00808，重复与AI评分反而变差，因此按冻结规则保留Step 5750。这段素材适合解释“预训练、SFT和检索各自负责什么”：预训练学小说语言分布，SFT学有限交互任务，精确全书事实交给检索；不能要求单本小说底座变成通用ChatGPT，也不能把实用平台期说成理论极限。

M020把M019的能力边界落实为新数据和新验收：不再把数学、通用百科或项目问答塞进后训练，而是用10,000条六维数据训练小说核心问答、给定证据理解、RAG组合、垂直多轮、小说表达和需要证据时的边界行为。纯预训练Step5750在新public上只有7.17% EOS、92.83%截断和85.5%机械重复；正式SFT到Step2000后变为82.33%/17.67%/42.17%，但固定16题AI辅助基本通过在四个正式节点只有0/0/1/0，严重重复还在Step2000反弹到15/16。这个完整素材最适合解释“优化指标之间会冲突”：Step1000仍在10% BPC保持线内却有空续写和行为门失败，Step1500行为样本相对最好却越过BPC线，Step2000 Loss最低却不是最好模型；最终严格候选为空、外部复核pending、本轮不发布也不启封sealed test。
