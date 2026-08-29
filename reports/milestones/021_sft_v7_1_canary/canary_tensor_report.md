# M021 SFT v7.1 Canary 编码报告

状态：**prepared**

| 项目 | Train | Holdout eval |
|---|---:|---:|
| 记录数 | 64 | 16 |
| 监督 Token | 720 | 180 |

- 事实数：8
- 序列长度：min=14，mean=23.56，max=36
- 仅 assistant 参与 Loss：True
- EOS 已追加且参与监督：True
- Tensor SHA-256：`8511bea2aa449f9dc29cc268239951786d512d9895f29a8730ccb7179f26914e`
- Manifest SHA-256：`68908fdabe4f8ae470f6bcd4ec6d11b59304829119835af901df7bf9888ef50d`

## 日志与独立调试

data、encoding、validation、artifact、orchestrator 各自写入轮转 JSONL。可使用 `--data-log-level DEBUG` 等参数，或设置 `GPT_CANARY_LOG_LEVEL_DATA=DEBUG` 单独打开某一类；传入 `OFF` 可关闭。日志仅包含数量、长度、SHA、状态和错误码，不包含问题、答案或 Token ID。定位完成后恢复 INFO；默认单文件 1 MiB，保留 3 份备份。
