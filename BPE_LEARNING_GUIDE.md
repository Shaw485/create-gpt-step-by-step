# BPE 逐行学习笔记

## 核心原理

BPE 的目标是把频繁相邻的 Token 合并成一个新 Token。例如字符序列 `萧 炎` 经常一起出现，就新增 Token `萧炎`。编码时必须严格按学习到的合并先后顺序执行，解码时只需把每个 Token 保存的文字拼回去。

## 最小核心代码与逐行解释

```python
tokenizer = learn_bpe(
    sequences=sample_sequences,
    base_tokens=base_tokens,
    num_merges=2000,
    min_frequency=3,
)
token_ids = tokenizer.encode(text)
restored_text = tokenizer.decode(token_ids)
assert restored_text == text
```

1. `tokenizer = learn_bpe(`：调用我们手写的合并规则学习器。
2. `sequences=sample_sequences,`：提供均匀覆盖整本语料的代表性片段；这里只用于统计哪些组合常见。
3. `base_tokens=base_tokens,`：用原来的4478个单字符作为初始词表，保证语料中的每个字符都可表示。
4. `num_merges=2000,`：最多学习2000次合并，每次为最常见的合法相邻组合新增一个 Token。
5. `min_frequency=3,`：只合并至少出现3次的组合，避免为偶然组合浪费词表。
6. `)`：完成学习，得到基础字符、合并 Token 和有先后顺序的规则。
7. `token_ids = tokenizer.encode(text)`：在完整文本上按规则优先级反复合并，得到整数 Token 序列。
8. `restored_text = tokenizer.decode(token_ids)`：把每个整数对应的文字片段顺序拼接回来。
9. `assert restored_text == text`：全文必须逐字符一致；不一致就立即停止，不允许进入训练。

## 模型训练的关键四行

```python
_, loss = model(inputs, targets)
optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
```

1. `_, loss = model(inputs, targets)`：前向传播，模型预测每个位置的下一个 BPE Token，并计算交叉熵 Loss。
2. `optimizer.zero_grad(set_to_none=True)`：清除上一步残留梯度，避免错误累加。
3. `loss.backward()`：反向传播，从 Loss 沿计算图求出每个参数应如何改变。
4. `optimizer.step()`：AdamW 根据梯度真正更新 Embedding、Q/K/V、前馈层、归一化和输出层等全部参数。

## SFT 的关键区别

SFT 序列为 `<BOS><USER>问题<ASSISTANT>回答<EOS>`。问题部分的标签设为 `-100`，交叉熵会忽略它们；只有回答与 `<EOS>` 参与 Loss。这样监督信号表达的是“看到问题后，应生成什么回答以及何时停止”，而不是让模型背诵问题。

## 完整汇总代码

- `bpe_tokenizer.py`：BPE 学习、编码、解码和保存。
- `prepare_bpe_data.py`：完整语料编码、切分和无损校验。
- `train_bpe_pretrain.py`：10,000步从零预训练。
- `prepare_bpe_sft.py`：1000条SFT数据的BPE编码。
- `initialize_bpe_sft.py`：复制预训练权重并扩展特殊Token。
- `train_sft_stage1.py`：20步检查与800步正式SFT。
- `evaluate_bpe_sft.py`：同题、同种子、同字符长度的公平评测。
- `plot_bpe_results.py`：Loss曲线和CSV。
