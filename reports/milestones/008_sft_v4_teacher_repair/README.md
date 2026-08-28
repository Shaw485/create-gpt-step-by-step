# M008：3000条教师SFT数据的保守修复与审计

## 阶段结论

本阶段直接复用教师模型生成的3000条原始JSONL，避免重新生成整批回答；原文件只读，SHA-256为`9b7f475ae0af689c96a4001fe5fda4c5bf8b0578ee3e8401680904183250fe63`。修复后的文件仍是`candidate`，不是正式训练集。

自动结构检查只剩一个失败门槛：600条验证/测试记录尚未完成真人审核。因此本阶段状态是`needs_review`，不能开始正式SFT，也不能把自动修复写成“3000条高质量数据已经验收”。

## 自动修复完成了什么

- 将原始9类任务映射并重平衡为冻结的7类配额。
- 按知识单元、声明章节和实际证据章节组成连通组，再整体分配到2400/300/300，主题、章节和组泄漏均为0。
- 将证据重新绑定到冻结v4语料的精确行、字符区间和SHA-256；保守的章节内模糊重绑必须进入复核。
- 使用v4 BPE逐条编码，3000条均无词表错误，最长序列204 Token，小于Context 512。
- 修复可由证据机械确定的少数残缺人物关系、重复章节焦点回答和章节标题填充。
- 对所有不确定内容保留原因码，不自动伪造`approved`、审核人或审核时间。

## 关键指标

| 指标 | 结果 |
|---|---:|
| 总记录 | 3000 |
| train / val / test | 2400 / 300 / 300 |
| 主题数 | 1760 |
| 每事实最多问题数 | 2 |
| 可定位到冻结语料的证据 | 2908 / 3000（96.93%） |
| 证据缺失 | 92 |
| 最常见答案占比 | 1.20% |
| 最大BPE序列长度 | 204 |
| 跨集合主题/章节/组泄漏 | 0 / 0 / 0 |
| val/test未批准 | 600 |
| 全部复核队列 | 2310 |

训练候选的互斥风险分层为：690条无自动风险标记，679条优先复核证据，1031条优先复核语义或任务改写。600条val/test无论自动标记如何，都属于决定发布门槛的P0真人审核对象。“无自动风险标记”只表示机器未发现已知问题，不表示事实已经过人工认证。

## 本地产物与发布边界

完整候选、原始快照、证据文本、复核CSV和日志位于Git忽略目录`data/sft/v4_teacher_repair/`，不会随代码提交。该目录包含：

- `sft_v4_teacher_candidates.jsonl`：3000条候选。
- `sft_v4_teacher_repair_queue.jsonl`：2310条待复核记录。
- `manual_review_val_test.csv`：600条发布门槛审核表。
- `review_priority_summary.json`：互斥优先级统计。
- `sft_v4_teacher_audit.json`与`independent_validation.json`：构建器和独立验证器报告。
- `logs/`与`validation_logs/`：数据、构建和验证分模块轮转日志。

仓库只归档不含长篇原文的统计报告。原始小说、教师数据、修复后的完整问答和证据原文不提交。

## 日志与独立调试

`SFT_V4_DATA_LOG_LEVEL`、`SFT_V4_BUILD_LOG_LEVEL`和`SFT_V4_VALIDATION_LOG_LEVEL`可分别控制数据、构建和验证日志级别；`SFT_V4_CONSOLE_LOG=0`可关闭终端日志。每个日志文件按10MB轮转并保留5份。日志记录run ID、文件哈希和计数，不记录密码、令牌、私钥或授权头。

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

## 下一步验收顺序

1. 先审核600条val/test，并填写真实的决定、审核人和审核时间。
2. 再处理训练集679条证据风险；证据无法确认的记录删除或改成诚实未知任务。
3. 审核1031条语义/任务改写风险，尤其是全局、唯一、首次、出现次数等断言。
4. 重新运行双重验证；只有所有门槛通过后才冻结正式SFT张量并做20步安全试跑。
