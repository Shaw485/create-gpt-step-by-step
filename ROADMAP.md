# Create GPT Step by Step

## 项目目标

使用 Python 和 PyTorch，从零实现一个可以训练和生成文本的迷你 GPT。

项目首先聚焦 GPT 的核心原理，不追求 ChatGPT 级别的参数规模。完成基础模型后，再逐步加入 BPE tokenizer、指令微调和聊天能力。

## 学习原则

- 每次只实现一个核心概念
- 先理解原理，再编写代码
- 不直接复制完整 GPT 实现
- 每个阶段都必须有可验证的结果
- 每完成一个阶段，都在 `RECORD.md` 中记录
- 所有代码和笔记都提交到 GitHub

## 技术栈

- Python 3
- PyTorch
- Git 和 GitHub
- Markdown
- 小型文本训练数据集

## Roadmap

### 1.0 项目初始化

任务：

- 创建本地项目目录
- 创建公开 GitHub 仓库
- 创建 `ROADMAP.md`
- 创建 `RECORD.md`
- 检查 Python 和 PyTorch 环境

完成标准：

- GitHub 仓库可以公开访问
- 两份 Markdown 文档已经上传
- Python 和 PyTorch 可以正常运行

### 1.1 字符级 Tokenizer

任务：

- 读取训练文本
- 找出全部字符
- 创建字符到整数的映射
- 实现 `encode`
- 实现 `decode`

完成标准：

```python
decode(encode(text)) == text
```

### 1.2 构造训练数据

任务：

- 把文本转换为 token 序列
- 划分训练集和验证集
- 理解 context window
- 构造输入 `x` 和目标 `y`
- 实现 batch 采样

完成标准：

- 能打印一个 batch
- 能解释输入和目标为什么错开一个 token

### 2.0 Bigram 基线模型

任务：

- 创建 token embedding
- 根据当前 token 预测下一个 token
- 计算交叉熵损失
- 完成第一次训练
- 生成第一段文本

完成标准：

- 训练损失明显下降
- 模型可以连续生成 token

### 3.0 Self-Attention

任务：

- 理解 Query、Key 和 Value
- 手动实现单头注意力
- 添加 causal mask
- 实现缩放点积注意力
- 实现多头注意力

完成标准：

- 张量形状全部正确
- 每个位置不能看到未来 token
- 能解释注意力权重的含义

### 4.0 Transformer Block

任务：

- 实现 LayerNorm
- 实现前馈神经网络
- 添加残差连接
- 组合多头注意力和前馈网络
- 堆叠多个 Transformer Block

完成标准：

- 一个 batch 可以正常前向传播
- 梯度可以正常反向传播

### 5.0 GPT 模型

任务：

- 添加 token embedding
- 添加 position embedding
- 堆叠 Transformer Block
- 添加最终 LayerNorm
- 添加语言模型输出层
- 实现 loss 计算

完成标准：

- 输入 token 后可以输出 logits
- 提供目标 token 时可以输出 loss
- 可以统计模型参数量

### 6.0 训练 GPT

任务：

- 编写训练循环
- 使用 AdamW 优化器
- 定期计算训练集和验证集损失
- 保存模型 checkpoint
- 支持重新加载模型

完成标准：

- 训练损失和验证损失整体下降
- 保存后的模型可以重新加载
- 加载后可以继续训练或生成文本

### 7.0 文本生成

任务：

- 实现自回归生成
- 理解 temperature
- 实现 top-k sampling
- 对比不同生成参数

完成标准：

- 模型可以根据提示词续写文本
- 固定随机种子时结果可以复现

### 8.0 测试与解释

任务：

- 测试 tokenizer
- 测试 causal mask
- 测试模型输出形状
- 测试模型保存和加载
- 绘制训练损失变化
- 为核心代码添加原理说明

完成标准：

- 核心测试全部通过
- 能从输入到输出完整解释一次 GPT 前向传播

### 9.0 BPE Tokenizer

任务：

- 理解字符级 tokenizer 的局限
- 学习 byte pair encoding
- 手动实现一个简单 BPE tokenizer
- 比较字符级和 BPE 的序列长度

完成标准：

- BPE tokenizer 可以编码和解码文本
- 能解释 GPT 为什么通常不使用纯字符 tokenizer

### 10.0 聊天模型扩展

任务：

- 准备指令问答数据
- 设计聊天消息格式
- 进行监督微调
- 添加命令行聊天界面

完成标准：

- 模型可以接受用户问题
- 模型能够生成回答格式的文本

## 最终成果

完成项目后，应当能够：

1. 解释 GPT 的整体架构
2. 独立实现 causal self-attention
3. 独立实现 decoder-only Transformer
4. 训练并保存一个小型语言模型
5. 使用模型进行文本生成
6. 解释基础 GPT 与聊天模型之间的区别
