# M008：3000条教师SFT数据的保守修复与审计

## 阶段结论

本阶段直接复用教师模型生成的3000条原始JSONL，避免重新生成整批回答；原文件只读，SHA-256为`9b7f475ae0af689c96a4001fe5fda4c5bf8b0578ee3e8401680904183250fe63`。修复后的文件仍是`candidate`，不是正式训练集。

自动结构检查只剩一个失败门槛：独立真人审核。经用户明确授权，Codex已作为AI审核员完成600条验证/测试记录的逐条决定，其中521条直接通过、79条修改后通过、0条拒绝。决定如实标记为`Codex AI reviewer`，因此不能冒充独立真人审核；本阶段仍不能写成“已完成人类验收”。这些AI决定已被合并成新的冻结副本，并生成2999条training-ready候选供下一步安全试跑。

## 自动修复完成了什么

- 将原始9类任务映射并重平衡为冻结的7类配额。
- 按知识单元、声明章节和实际证据章节组成连通组，再整体分配到2400/300/300，主题、章节和组泄漏均为0。
- 将证据重新绑定到冻结v4语料的精确行、字符区间和SHA-256；章节标题类问题直接绑定正式章节标题，保守的章节内模糊重绑仍必须进入复核。
- 使用v4 BPE逐条编码，3000条均无词表错误，最长序列149 Token，小于Context 512。
- 修复可由证据机械确定的少数残缺人物关系、重复章节焦点回答和章节标题填充；复制任务不再截断证据，一句话任务只保留一个完整句子，无法由单条证据证明的“唯一/最多”任务被降级为证据片段理解题。
- 对所有不确定内容保留原因码，不自动伪造`approved`、审核人或审核时间。

## 关键指标

| 指标 | 结果 |
|---|---:|
| 总记录 | 3000 |
| train / val / test | 2400 / 300 / 300 |
| 主题数 | 1760 |
| 每事实最多问题数 | 2 |
| 可定位到冻结语料的证据 | 2999 / 3000（99.97%） |
| 证据缺失 | 1 |
| 最常见答案占比 | 1.83% |
| 最大BPE序列长度 | 149 |
| 跨集合主题/章节/组泄漏 | 0 / 0 / 0 |
| val/test未批准 | 600 |
| 全部复核队列 | 2287 |
| AI审核合并后修改记录 | 79 |
| 冻结阶段去重问句 | 19 |
| training-ready记录 | 2999 |

训练候选的互斥风险分层为：713条无自动风险标记，9条优先复核证据，1678条优先复核语义或任务改写。600条val/test无论自动标记如何，都属于决定发布门槛的P0真人审核对象。“无自动风险标记”只表示机器未发现已知问题，不表示事实已经过人工认证。

600条评估记录又经过一轮AI辅助预审。修复60条原因题语病并纠正2条“尚未达到二星”却答成“已经二星”的事实错误后，又重新核对章节边界。冻结v4语料同时含正式分隔符标题和少量缩进的旧版内嵌标题；旧定位器错误地把两者都当成正式边界。修正为只采用分隔符确认的正式标题后，全数据共262条旧章节号完成同步重绑定，并保留原编号与原标题供追溯。

原有“首次同框”任务只能由单条证据证明两个人名在局部同时出现，不能证明全书首次。相关记录因此统一改成局部证据理解、核验或诚实未知任务。继续抽查发现，字面索引也可能把人物“云山”误命中为“高耸入云山峰”，所以不能把“索引一致”直接当作语义事实。

为避免把这些不可靠结论写进SFT，最终将首次出现、出场先后、首次同现、戏份最多、章节主角、全书未说明等任务全部限制为当前证据片段可直接回答的问题；异火排名和斗技等级则从同一证据中机械抽取正确值。共1067条候选采用这种局部证据改写，其中评估集406条进入改写复核、194条进入低风险确认。全书断言核验器重新扫描后选中0条，说明候选中不再存在需要字面全书索引裁决的活动断言；这不等于1067条已经人工批准。

构建器另对1067条局部改写逐条做失败即停止的二次检查：答案涉及的人名、连续字符、异火排名或斗技等级必须能在同一条已验证证据中重新找到。本次1067/1067通过；该检查验证的是局部证据一致性，不替代真人对题目自然度和训练价值的判断。

## 本地产物与发布边界

完整候选、原始快照、证据文本、复核CSV和日志位于Git忽略目录`data/sft/v4_teacher_repair/`，不会随代码提交。该目录包含：

- `sft_v4_teacher_candidates.jsonl`：3000条候选。
- `sft_v4_teacher_ai_reviewed_candidates.jsonl`：合并600条Codex AI审核决定后的3000条冻结副本。
- `sft_v4_teacher_ai_training_ready.jsonl`：排除1条证据章节漂移样本后的2999条SFT候选。
- `sft_v4_teacher_ai_review_sidecar.json`：AI审核修改、冻结去重和训练证据处理明细。
- `sft_v4_teacher_ai_review_freeze_report.json`：AI审核合并与training-ready生成报告。
- `sft_v4_teacher_repair_queue.jsonl`：2287条待复核记录。
- `manual_review_val_test.csv`：600条发布门槛审核表。
- `review_priority_summary.json`：互斥优先级统计。
- `evaluation_ai_pre_review_summary.json`：600条评估记录的AI预审分层。
- `ai_review_summary.json`：用户授权的Codex AI审核统计，不冒充独立真人结论。
- `global_claim_verification.json`：全局断言扫描结果；当前活动断言为0条。
- `sft_v4_teacher_audit.json`与`independent_validation.json`：构建器和独立验证器报告。
- `logs/`与`validation_logs/`：数据、构建和验证分模块轮转日志。
- `freeze_logs/`：AI审核合并、training-ready输出和冻结验证日志。

仓库只归档不含长篇原文的统计报告。原始小说、教师数据、修复后的完整问答和证据原文不提交。

## 日志与独立调试

`SFT_V4_DATA_LOG_LEVEL`、`SFT_V4_BUILD_LOG_LEVEL`和`SFT_V4_VALIDATION_LOG_LEVEL`可分别控制数据、构建和验证日志级别；`SFT_V4_CONSOLE_LOG=0`可关闭终端日志。每个日志文件按10MB轮转并保留5份。日志记录run ID、文件哈希和计数，不记录密码、令牌、私钥或授权头。

## 本地真人审核工具

`review_sft_v4.py`在`127.0.0.1`提供浏览器审核页，默认按406条任务改写、194条低风险记录的顺序展示问题、答案、原文证据和风险标记。审核人可以选择“通过”“修改后通过”或“拒绝”；后两种必须填写说明。每次决定立即原子写入独立文件，不修改教师原始数据或候选JSONL。

```bash
.venv/bin/python review_sft_v4.py
```

启动后访问`http://127.0.0.1:8765`。决定保存在Git忽略目录`data/sft/v4_teacher_repair/human_review_decisions.jsonl`。每条决定记录实际审核主体、时间和对应候选SHA-256；候选发生变化时旧决定会拒绝加载，防止把过期审核错误套到新数据。当前文件含600条`Codex AI reviewer`决定：521条直接通过、79条修改后通过。它证明AI审核已完成，但不自动满足独立真人治理门槛。

审核日志位于`data/sft/v4_teacher_repair/review_logs/`，分为UI、数据保存和验证三类。分别使用`SFT_REVIEW_UI_LOG_LEVEL`、`SFT_REVIEW_DATA_LOG_LEVEL`和`SFT_REVIEW_VALIDATION_LOG_LEVEL`调节；默认按10MB轮转并保留5份。日志只记录记录ID、决定类型和数量，不写审核人、备注、问题或答案正文。

合并AI审核并生成training-ready候选：

```bash
.venv/bin/python finalize_sft_v4_ai_review.py
```

该脚本会校验每条审核决定的候选SHA-256，拒绝过期决定；79条修改后通过只更新问答，不改原始教师文件。若修改后问题文本重复，会加轻量任务角度前缀并写入sidecar。训练集里8条模糊重绑证据已由正式v4语料再次验证；1条证据章节漂移记录保留在完整冻结副本中，但从training-ready文件排除。

复现自动修复：

```bash
.venv/bin/python repair_teacher_sft_v4.py \
  --source /Users/bytedance/Downloads/sft_novel_3000_teacher_raw_v1.jsonl
```

独立验证：

```bash
.venv/bin/python validate_sft_v4.py \
  --dataset data/sft/v4_teacher_repair/sft_v4_teacher_candidates.jsonl \
  --corpus data/cloud_v4/corpus.txt \
  --report data/sft/v4_teacher_repair/independent_validation.json \
  --log-dir data/sft/v4_teacher_repair/validation_logs
```

全局断言索引核验：

```bash
.venv/bin/python verify_sft_global_claims.py
```

## 下一步验收顺序

1. 由数据所有者决定是否接受已完成的600条Codex AI审核；若发布协议坚持独立真人签字，则再对79条修改记录和风险抽样做人审。
2. 使用2999条training-ready候选重新编码SFT张量，保留SFT前基线并做20步安全试跑。
3. 安全试跑通过后再进入正式SFT；训练报告必须继续同时记录Loss、固定题输出和人工语义抽样。
