# M009：v4 SFT 20步安全试跑

## 阶段结论

本阶段使用M008生成的2999条training-ready SFT候选，重新编码为v4 BPE聊天格式张量，并从M007的`runs/pretrain_v4_m4_continue6000/best.pt`继续进行20步监督微调安全试跑。目标不是得到可用聊天模型，而是验证数据格式、回答区mask、梯度、Loss下降、checkpoint保存和报告留档都能闭环。

试跑完成。由于当前Codex工具环境中`torch.backends.mps.is_available()`为`False`，本次实际使用CPU；这不代表用户终端里的正式本机训练速度。模型结构保持M007配置：8105025参数、Embedding 256、8层、8头、Context 512、BPE词表6465。

## 数据编码

输入数据为`data/sft/v4_teacher_repair/sft_v4_teacher_ai_training_ready.jsonl`，共2999条，其中train/val/test为2399/300/300。编码后写入`data/cloud_v4/sft_v4_ai_training_ready_tensors.pt`，不会提交到Git。

聊天序列格式：

```text
<BOS><USER>问题<ASSISTANT>答案<EOS>
```

Loss只监督`答案<EOS>`部分，`<BOS><USER>问题<ASSISTANT>`全部用`-100` mask掉，所以模型训练目标是“看到问题后生成回答”，不是复读问题。

关键数据指标：

| 指标 | 结果 |
|---|---:|
| SFT记录数 | 2999 |
| train / val / test | 2399 / 300 / 300 |
| 任务类型数 | 7 |
| 词表大小 | 6465 |
| 最短/最长序列 | 12 / 149 Token |
| 平均序列长度 | 60.25 Token |
| 监督Token数 | 59426 |
| 被mask Token数 | 118262 |

## 20步试跑结果

| 指标 | 结果 |
|---|---:|
| 设备 | CPU |
| 步数 | 20 |
| Micro batch | 1 |
| 学习率 | 5e-5 |
| 初始训练Loss | 7.5634 |
| 初始验证Loss | 7.5522 |
| 最终训练Loss | 5.3162 |
| 最终验证Loss | 5.0587 |
| 最佳验证Loss | 5.0587 |
| 测试集消耗 | 0 |

Loss下降说明SFT数据、mask和反向传播是连通的；但20步样本仍很差，不能作为最终效果展示。当前样本只适合说明“开始从续写目标转向回答目标”，不适合宣称问答能力达标。

## 固定样本

训练样本提示：

```text
原问题是“雷尊者与药尘相比，谁更早出现在故事里”。现只做局部证据核验：不比较全书登场先后，只看第351章的当前证据：药尘和雷尊者中明确出现了谁？
```

20步后输出：

```text
第一个不说的
```

验证样本提示：

```text
参考原问题“第五百五十八章中，紫研为什么没有反对接受免费的本源锻体？”，如果用户只说“请介绍相关人物”，但没有说明作品、人物和故事阶段，应该先怎么回应？
```

20步后输出：

```text
“嘭！”
```

## 日志与复现

SFT试跑日志位于`runs/sft_v4_smoke20/logs/`，按模块分为`data`、`sft`、`validation`、`checkpoint`和`orchestrator`。日志是JSONL格式，包含run_id、时间、级别、模块和非敏感上下文；checkpoint采用原子写入、SHA-256 sidecar和重新加载校验。

复现数据编码：

```bash
BPE_SFT_DATASET=data/sft/v4_teacher_repair/sft_v4_teacher_ai_training_ready.jsonl \
BPE_SFT_TOKENIZER=data/cloud_v4/tokenizer.json \
BPE_SFT_OUTPUT=data/cloud_v4/sft_v4_ai_training_ready_tensors.pt \
BPE_SFT_REPORT=reports/milestones/009_v4_sft_smoke20/sft_v4_data_report.json \
BPE_SFT_EXPECTED_SPLITS='{"train": 2399, "val": 300, "test": 300}' \
BPE_SFT_MAX_SEQUENCE_LENGTH=512 \
.venv/bin/python prepare_bpe_sft.py
```

复现20步试跑：

```bash
.venv/bin/python train_sft_v4.py \
  --device cpu \
  --micro-batch-size 1 \
  --eval-batches 2 \
  --max-steps 20 \
  --eval-interval 5
```

正式训练时应在可用MPS或CUDA环境中运行，并提高micro batch、评估batch和训练步数。本阶段仍保留M008治理边界：Codex AI审核不等于独立真人签字。
