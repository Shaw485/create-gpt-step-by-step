# SFT v4 数据准备指南

SFT v4 把“候选数据”和“可发布训练数据”分开。自动导入或自动构造的记录默认都是
`pending`，不会被描述为已经人工审核的高质量数据。只有所有质量门槛通过后，才允许
导出云训练使用的文件。

## 当前质量合同

- 总计 3000 条，`train/val/test = 2400/300/300`。
- 七类配额依次为 750、600、450、450、300、300、150。
- 至少 1200 个独立 `topic_id`，同一个 `fact_id` 最多两个问题。
- 完全相同答案占比必须小于 2%。
- 至少 70% 的记录必须同时具有可复核的语料哈希、章节、行和字符跨度、证据哈希。
- 同一 topic、章节或 group 不得跨越数据切分。
- 验证集和测试集必须标记为 `approved`，并填写 reviewer 与 reviewed_at。

完整、机器可读的合同位于 `data/sft/v4/sft_v4_schema.json`。

## 构建第一批候选

```bash
python build_sft_v4.py
```

输出包括：

- `data/sft/v4/sft_v4_candidates.jsonl`：通过去重及“一事实最多两问”的候选。
- `data/sft/v4/sft_v4_rejections.jsonl`：被拒记录及明确原因。
- `data/sft/v4/sft_v4_review_queue.jsonl`：评测集优先的人工复核队列。
- `data/sft/v4/sft_v4_audit.json`：配额、缺口、泄漏和全部质量门结果。
- `data/sft/v4/sft_v4_validation.json`：独立复验结果，不覆盖构建审计。
- `data/sft/v4/sft_v4_schema.json`：稳定 schema 和发布质量合同。

导入新的 JSONL 候选时，每一行至少需要 `question`、`answer`、
`task_family`、`topic_id`；`fact_id` 省略时等于 `topic_id`。只有同时提供能在语料中
精确匹配的 `evidence` 和一基 `source_line` 时，流水线才会生成章节、跨度和哈希，
否则会如实标为缺少证据。

```bash
python build_sft_v4.py --import-jsonl data/sft/v4/my_candidates.jsonl
```

## 验证、人工复核与发布

人工审核时直接修改候选记录的 `review`：通过的记录填写 `status=approved`、
reviewer 和 ISO 8601 reviewed_at；无法确认的记录保持 pending 或标为 rejected。

普通审计不会因为数据尚未齐全而假装成功：

```bash
python validate_sft_v4.py
```

发布命令会在任一门槛不满足时以失败状态退出：

```bash
python validate_sft_v4.py --export-release
```

全部通过时，原子导出：

- `data/cloud_v4/sft_train.jsonl`
- `data/cloud_v4/sft_val.jsonl`
- `data/cloud_v4/sft_test.jsonl`
- `data/cloud_v4/sft_manifest.json`
- `data/cloud_v4/sft_manifest.json.sha256`

云目录是运行产物，应由 `.gitignore` 保护，不上传原始训练数据。

## 分模块诊断日志

日志位于 `logs/sft_v4_{data,build,validation}.log`，每个文件按 10 MB 轮转，保留
5 份。默认只记录 INFO 及以上，不记录令牌、密码或私钥。每次运行带有可搜索的
`run_id`。

三个模块可独立调整；取值为 `DEBUG/INFO/WARNING/ERROR/OFF`：

```bash
SFT_V4_DATA_LOG_LEVEL=WARNING \
SFT_V4_BUILD_LOG_LEVEL=INFO \
SFT_V4_VALIDATION_LOG_LEVEL=DEBUG \
python build_sft_v4.py
```

设置 `SFT_V4_CONSOLE_LOG=0` 可关闭终端副本而保留文件日志。按一次运行筛选时使用：

```bash
rg 'run_id=要查找的编号' logs/sft_v4_*.log
```

构建失败优先查看 validation 日志；导入数量异常查看 data；去重、拒绝和输出路径问题
查看 build。导出诊断材料时复制相应轮转日志即可，日志本身不包含语料正文。

## 测试

```bash
.venv/bin/python -m unittest tests.test_sft_v4_pipeline -v
```

测试覆盖完整发布成功、真实 v3 缺口、确定性输出、证据篡改、评测集未审核、失败日志
以及云 manifest 和校验 sidecar。
