# M004：手写 BPE 与从零预训练

## 做了什么

没有调用现成 Tokenizer 库，而是手写字符起步的 BPE：先保留原来的 4478 个字符 Token，再从均匀覆盖全书的 749952 字符样本中学习 2000 条相邻 Token 合并规则。随后用固定合并顺序编码完整清洗语料，并逐字解码复核。

## 数据结果

| 指标 | 字符级 | BPE |
|---|---:|---:|
| 语料字符数 | 6,544,278 | 6,544,278 |
| Token 数 | 6,544,278 | 4,067,842 |
| 词表大小 | 4,478 | 6,478 |
| 每 Token 平均字符 | 1.000 | 1.609 |
| Token 数减少 | — | 37.84% |
| 全文无损还原 | 是 | 是 |

相同 `block_size=256` 下，字符级约覆盖 256 字符，BPE 平均约覆盖 412 字符。BPE 不跨换行合并，未知字符会给出明确错误。

## 预训练结果

- 架构：Embedding 256、8 Heads、6 Blocks、Block 256、Batch 4。
- 参数量：8,123,214。
- 设备：CPU。
- 步数：10,000。
- 第0步训练/验证 Loss：8.8407 / 8.8201。
- 第10,000步训练/验证 Loss：4.2365 / 4.8036。
- 最佳步数：10,000。
- 耗时：2925.7 秒，约48.8分钟。

BPE 的单 Token Loss 不能直接与字符级 Loss 比，因为词表和 Token 粒度不同。按全语料平均压缩率粗略换算，第10,000步约为 `4.8036 / 1.6088 = 2.9858 nats/字符`；字符版正式评测约为3.0863 nats/字符。这个换算只用于同一语料的近似效率比较。

## 样本结论

第0步是随机乱码；第500步出现中文小说短语；第3,000步以后出现较完整的叙述和对话；第10,000步局部文法明显改善。但“今天天气怎么样”等问题仍被续写成小说，说明预训练学到的是下一个 Token 预测，不是指令服从。

## 关键文件

- `data/bpe/tokenizer_v1.json`：词表和2000条合并规则。
- `data/bpe/bpe_v1_metrics.json`：压缩率、还原检查和哈希。
- `data/bpe/doupo_bpe_v1_tensors.pt`：完整 BPE 训练/验证张量。
- `checkpoints/bpe_pretrain_step10000_best.pt`：最佳预训练模型。
- `bpe_pretrain_report.json`：21个评估点和每500步固定10题。

## 调试与日志

- `logs/bpe_learning.log`：合并规则学习。
- `logs/bpe_encoding.log`：完整语料编码进度。
- `logs/bpe_validation.log`：无损还原与压缩率。
- `logs/bpe_pretrain_data.log`、`step.log`、`validation.log`、`generation.log`、`checkpoint.log`：预训练各模块。

每类日志可用对应的 `BPE_LOG_*_LEVEL` 或 `BPE_PRETRAIN_*_LOG_LEVEL` 单独调节，`BPE_LOG_CONSOLE=0`、`BPE_PRETRAIN_CONSOLE_LOG=0` 可关闭控制台输出。日志按1 MB轮转，保留3份，不记录密码、Token 或授权头。
