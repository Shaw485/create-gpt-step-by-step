# M006：v4 正式语料与本机 810 万参数预训练

## 本阶段完成了什么

这次不沿用旧的整本连续切分，而是重新冻结正式 v4 语料：审核 113 组章节版本，修复 29 行可核验的异常标题，删除 1 段 SHA-256 锁定的跨章旧稿和 14 个非正文块。原始 stage3 文件没有被修改。

初次解析发现 26 个缺失章节编号，标题修复后恢复 17 个，最终确认 9 个编号在源文件中没有独立标题：94、295、385、396、740、741、742、743、1010。之前预计的“20 个”无法从当前源文件与解析规则复现，因此没有为了凑数而伪造章节。

## 正式数据

| 集合 | 章节段 | 字符 | BPE Token | EOS |
|---|---:|---:|---:|---:|
| Train | 1,599 | 5,508,660 | 3,416,400 | 1,599 |
| Validation | 92 | 314,610 | 195,068 | 92 |
| Test | 84 | 297,005 | 184,808 | 84 |

切分以完整章节版本组为边界，同一章节不会跨集合。BPE 的 2,000 条合并规则只从训练文本统计；为了让验证集与测试集可以无损编码，基础字符表额外保留 60 个只在验证/测试出现的字符，但这些字符没有参与合并频次。最终词表为 6,465，三个集合全部逐字还原通过，每章末尾恰好插入一个 `<EOS>`。

## 模型和训练配置

| 参数 | 值 |
|---|---:|
| 参数量 | 8,105,025 |
| Embedding | 256 |
| Transformer Blocks | 8 |
| Attention Heads | 8 |
| FFN | 1,024 |
| Context | 512 Token |
| Micro Batch | 2 |
| 梯度累积 | 4 |
| 每个优化步 Token | 4,096 |
| 优化器 | AdamW |
| 学习率 | 3e-4，经 warmup + cosine 降至 3e-5 |
| 设备 | Apple M4 / MPS / float32 |
| 正式步数 | 3,000 |

训练前先完成 3 步 MPS 冒烟、checkpoint 重新加载和断点续训验证。正式训练由随机权重开始，耗时 2,201.6 秒（约 36 分 42 秒）。

## 训练结果

| 指标 | 结果 |
|---|---:|
| Step 100 验证 Loss | 7.2220 |
| Step 500 验证 Loss | 6.2111 |
| Step 1,000 验证 Loss | 5.6599 |
| 最佳验证 Loss | **5.0447（Step 2,600）** |
| Step 3,000 训练 Loss | 4.9333 |
| Step 3,000 验证 Loss | 5.0767 |
| 验证集选出的最佳模型 Test Loss | **4.9958** |

模型选择只使用验证 Loss；Test 集没有参与选择。最终发布候选是 Step 2,600 的 `best.pt`，不是 Step 3,000 的 `latest.pt`。

## 固定样本结论

每 500 步对同一组 10 个问题最多生成 30 个字，完整结果见 `fixed_prompt_samples.md`。训练后输出已从随机字符变为带人物、叙述和对话结构的小说片段，但“今天天气怎么样”等问题仍被当作小说开头继续写，并不会真正回答。

这是符合预期的预训练结果：模型只学习“预测下一个 Token”，没有学习“听从问题并给出答案”。这组输出将作为 v4 SFT 前基线，后续必须保持同一组问题、采样参数和最大字符数再测，才能展示 SFT 带来的行为变化。

## 文件与复现材料

- `pretrain_v4_report.json`：30 个训练/验证评估点。
- `selected_model_evaluation.json`：验证集选出的最佳模型及一次性测试结果。
- `fixed_prompt_samples.md`、`sample_history.json`：固定 10 题的完整演化。
- `pretrain_v4_loss.csv`：可用于视频和二次分析的表格。
- `pretrain_v4_loss_curve.png`、`pretrain_v4_loss_curve.svg`：Loss 曲线。
- `corpus_manifest.json`、`chapter_pair_review.json`、`missing_chapters_audit.json`：语料冻结依据。
- `token_manifest.json`：BPE、EOS 和张量校验信息。
- `effective_config.json`：本次实际训练配置。
- `SHA256SUMS.md`：报告、图表、Tokenizer 和两个 checkpoint 的校验值。

Checkpoint 位于 `runs/pretrain_v4_m4/`，受 `.gitignore` 保护，不随代码默认上传。`best.pt` 和 `latest.pt` 都有独立 SHA-256 sidecar，并在保存后完成重新加载验证。

## 日志与独立调试

训练日志位于 `runs/pretrain_v4_m4/logs/`，按 `data`、`pretrain`、`validation`、`checkpoint`、`gpu`、`orchestrator` 等模块分别写入 JSONL。每个模块可在 `configs/local_m4_8m.json` 的 `logging.module_levels` 中独立设为 `DEBUG`、`INFO` 或 `OFF`；日志按 10 MB 轮转并保留 5 份。

日志记录时间、run id、step、Loss、学习率、梯度范数和检查点哈希，不记录密码、访问令牌或授权头。定位数据问题只看 `*.data.jsonl`，Loss/梯度问题看 `*.pretrain.jsonl`，保存与恢复问题看 `*.checkpoint.jsonl`，避免所有信息挤在一个文件中。

## 下一步

冻结 Step 2,600 最佳预训练模型作为 SFT 初始化点；先把 v4 的高质量指令数据准备、审计和切分完成，再做 SFT 前基线复测、20 步安全试跑和正式 SFT。不能继续使用旧 Tokenizer 的 SFT checkpoint，因为 v4 的词表和参数 ID 已经改变。
