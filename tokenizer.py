import torch
from pathlib import Path

text = Path("data/input.txt").read_text(encoding="utf-8")

chars = sorted(set(text))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(input_text):
    token_ids = []
    for ch in input_text:
        token_ids.append(stoi[ch])
    return token_ids


def decode(token_ids):
    decoded_chars = []
    for token_id in token_ids:
        decoded_chars.append(itos[token_id])
    return "".join(decoded_chars)


sample_char = "猫"
sample_id = stoi[sample_char]

test_text = "小猫看雨。"
encoded = encode(test_text)
decoded = decode(encoded)

print("字符总数：", len(text))
print("词表大小：", vocab_size)
print("字符表：", chars)
print("前 50 个字符：", repr(text[:50]))
print("字符转数字：", sample_char, "->", sample_id)
print("数字转字符：", sample_id, "->", itos[sample_id])
print("测试文本：", test_text)
print("编码结果：", encoded)
print("解码结果：", decoded)
print("还原一致：", decoded == test_text)
print("全文还原一致：", decode(encode(text)) == text)
encoded_text = encode(text)
data = torch.tensor(encoded_text, dtype=torch.long)

print("数据类型：", data.dtype)
print("张量形状：", data.shape)
print("Token 总数：", data.numel())
print("前 20 个 Token：", data[:20])
split_index = int(0.9 * len(data))
train_data = data[:split_index]
val_data = data[split_index:]

print("训练集大小：", len(train_data))
print("验证集大小：", len(val_data))
print("总数检查：", len(train_data) + len(val_data))
block_size = 8

x = train_data[:block_size]
y = train_data[1:block_size + 1]

print("输入 Token：", x)
print("目标 Token：", y)
print("输入文字：", decode(x.tolist()))
print("目标文字：", decode(y.tolist()))

for position in range(block_size):
    context = x[:position + 1]
    target = y[position]
    context_text = decode(context.tolist())
    target_text = decode([target.item()])
    print("上下文：", repr(context_text), "-> 目标：", repr(target_text))

torch.manual_seed(42)

max_start = len(train_data) - block_size
start_index = torch.randint(0, max_start, (1,)).item()

sample_x = train_data[start_index:start_index + block_size]
sample_y = train_data[start_index + 1:start_index + block_size + 1]

print("随机起点：", start_index)
print("随机输入：", decode(sample_x.tolist()))
print("随机目标：", decode(sample_y.tolist()))

batch_size = 4

start_indices = torch.randint(0, max_start, (batch_size,))

batch_x = torch.stack(
    [train_data[i:i + block_size] for i in start_indices]
)

batch_y = torch.stack(
    [train_data[i + 1:i + block_size + 1] for i in start_indices]
)

print("批次起点：", start_indices)
print("Batch 输入形状：", batch_x.shape)
print("Batch 目标形状：", batch_y.shape)
print("第一个 Batch 输入：", decode(batch_x[0].tolist()))
print("第一个 Batch 目标：", decode(batch_y[0].tolist()))

def get_batch(split):
    if split == "train":
        data_source = train_data
    else:
        data_source = val_data

    max_start_index = len(data_source) - block_size
    batch_indices = torch.randint(0, max_start_index, (batch_size,))

    inputs = torch.stack(
        [data_source[i:i + block_size] for i in batch_indices]
    )

    targets = torch.stack(
        [data_source[i + 1:i + block_size + 1] for i in batch_indices]
    )

    return inputs, targets


train_x, train_y = get_batch("train")
val_x, val_y = get_batch("val")

print("训练 Batch：", train_x.shape, train_y.shape)
print("验证 Batch：", val_x.shape, val_y.shape)
