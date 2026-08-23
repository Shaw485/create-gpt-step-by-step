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
bigram_table = torch.nn.Embedding(vocab_size, vocab_size)
logits = bigram_table(train_x)

print("Bigram 参数形状：", bigram_table.weight.shape)
print("Bigram 参数数量：", bigram_table.weight.numel())
print("Logits 形状：", logits.shape)
print("第一个位置的前 10 个分数：", logits[0, 0, :10])
batch_count, sequence_length, class_count = logits.shape

logits_flat = logits.reshape(
    batch_count * sequence_length,
    class_count,
)

targets_flat = train_y.reshape(
    batch_count * sequence_length
)

loss = torch.nn.functional.cross_entropy(
    logits_flat,
    targets_flat,
)

print("整理后的 Logits：", logits_flat.shape)
print("整理后的 Targets：", targets_flat.shape)
print("初始 Loss：", loss.item())
optimizer = torch.optim.AdamW(
    bigram_table.parameters(),
    lr=0.01,
)

optimizer.zero_grad()
loss.backward()
optimizer.step()

updated_logits = bigram_table(train_x)

updated_logits_flat = updated_logits.reshape(
    batch_count * sequence_length,
    class_count,
)

updated_loss = torch.nn.functional.cross_entropy(
    updated_logits_flat,
    targets_flat,
)

print("参数更新前 Loss：", loss.item())
print("参数更新后 Loss：", updated_loss.item())
training_steps = 1000

for step in range(training_steps):
    batch_inputs, batch_targets = get_batch("train")
    batch_logits = bigram_table(batch_inputs)

    current_batch, current_length, current_classes = batch_logits.shape

    batch_logits_flat = batch_logits.reshape(
        current_batch * current_length,
        current_classes,
    )

    batch_targets_flat = batch_targets.reshape(
        current_batch * current_length
    )

    batch_loss = torch.nn.functional.cross_entropy(
        batch_logits_flat,
        batch_targets_flat,
    )

    optimizer.zero_grad()
    batch_loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"Step {step:4d} | Loss {batch_loss.item():.4f}")
evaluation_steps = 100


def estimate_loss(split):
    losses = []

    with torch.no_grad():
        for _ in range(evaluation_steps):
            eval_inputs, eval_targets = get_batch(split)
            eval_logits = bigram_table(eval_inputs)

            eval_logits_flat = eval_logits.reshape(-1, vocab_size)
            eval_targets_flat = eval_targets.reshape(-1)

            eval_loss = torch.nn.functional.cross_entropy(
                eval_logits_flat,
                eval_targets_flat,
            )

            losses.append(eval_loss.item())

    return sum(losses) / len(losses)


average_train_loss = estimate_loss("train")
average_val_loss = estimate_loss("val")

print("平均训练 Loss：", average_train_loss)
print("平均验证 Loss：", average_val_loss)
generation_steps = 100
generated_ids = [stoi["小"]]

with torch.no_grad():
    for _ in range(generation_steps):
        current_id = generated_ids[-1]
        current_token = torch.tensor([[current_id]], dtype=torch.long)
        current_logits = bigram_table(current_token)
        next_token_logits = current_logits[0, -1]
        probabilities = torch.softmax(next_token_logits, dim=-1)
        next_id = torch.multinomial(probabilities, num_samples=1).item()
        generated_ids.append(next_id)

generated_text = decode(generated_ids)

print("生成结果：")
print(generated_text)
n_embd = 32

attention_inputs, _ = get_batch("train")

token_embedding_table = torch.nn.Embedding(vocab_size, n_embd)
position_embedding_table = torch.nn.Embedding(block_size, n_embd)

token_embeddings = token_embedding_table(attention_inputs)

position_ids = torch.arange(block_size)
position_embeddings = position_embedding_table(position_ids)

attention_x = token_embeddings + position_embeddings

print("Attention 输入形状：", attention_inputs.shape)
print("Token Embedding 形状：", token_embeddings.shape)
print("位置编号：", position_ids)
print("Position Embedding 形状：", position_embeddings.shape)
print("合并后形状：", attention_x.shape)
print("第一个 Token 的前 5 个特征：", attention_x[0, 0, :5])

head_size = 16

key_layer = torch.nn.Linear(n_embd, head_size, bias=False)
query_layer = torch.nn.Linear(n_embd, head_size, bias=False)

keys = key_layer(attention_x)
queries = query_layer(attention_x)

attention_scores = queries @ keys.transpose(-2, -1)
attention_scores = attention_scores * (head_size ** -0.5)

causal_mask = torch.tril(torch.ones(block_size, block_size))

masked_scores = attention_scores.masked_fill(
    causal_mask == 0,
    float("-inf"),
)

attention_weights = torch.softmax(masked_scores, dim=-1)

print("Keys 形状：", keys.shape)
print("Queries 形状：", queries.shape)
print("Attention Scores 形状：", attention_scores.shape)
print("因果遮罩：")
print(causal_mask)
print("第一个 Batch 的注意力权重：")
print(attention_weights[0])
print("每行概率之和：", attention_weights[0].sum(dim=-1))
head_size = 16

key_layer = torch.nn.Linear(n_embd, head_size, bias=False)
query_layer = torch.nn.Linear(n_embd, head_size, bias=False)

keys = key_layer(attention_x)
queries = query_layer(attention_x)

attention_scores = queries @ keys.transpose(-2, -1)
attention_scores = attention_scores * (head_size ** -0.5)

causal_mask = torch.tril(torch.ones(block_size, block_size))

masked_scores = attention_scores.masked_fill(
    causal_mask == 0,
    float("-inf"),
)

attention_weights = torch.softmax(masked_scores, dim=-1)

print("Keys 形状：", keys.shape)
print("Queries 形状：", queries.shape)
print("Attention Scores 形状：", attention_scores.shape)
print("因果遮罩：")
print(causal_mask)
print("第一个 Batch 的注意力权重：")
print(attention_weights[0])
print("每行概率之和：", attention_weights[0].sum(dim=-1))
value_layer = torch.nn.Linear(n_embd, head_size, bias=False)

values = value_layer(attention_x)

attention_output = attention_weights @ values

first_position_matches = torch.allclose(
    attention_output[0, 0],
    values[0, 0],
)

print("Values 形状：", values.shape)
print("Attention 输出形状：", attention_output.shape)
print("第一个位置只能读取自己：", first_position_matches)
print("第一个位置的输出：")
print(attention_output[0, 0])
print("最后一个位置的注意力权重：")
print(attention_weights[0, -1])
class AttentionHead(torch.nn.Module):
    def __init__(self, input_size, output_size, context_size):
        super().__init__()
        self.output_size = output_size
        self.key = torch.nn.Linear(input_size, output_size, bias=False)
        self.query = torch.nn.Linear(input_size, output_size, bias=False)
        self.value = torch.nn.Linear(input_size, output_size, bias=False)
        mask = torch.tril(torch.ones(context_size, context_size))
        self.register_buffer("causal_mask", mask)

    def forward(self, input_vectors):
        sequence_length_now = input_vectors.shape[1]
        keys_now = self.key(input_vectors)
        queries_now = self.query(input_vectors)
        scores_now = queries_now @ keys_now.transpose(-2, -1)
        scores_now = scores_now * (self.output_size ** -0.5)
        mask_now = self.causal_mask[:sequence_length_now, :sequence_length_now]
        scores_now = scores_now.masked_fill(mask_now == 0, float("-inf"))
        weights_now = torch.softmax(scores_now, dim=-1)
        values_now = self.value(input_vectors)
        output_now = weights_now @ values_now
        return output_now


num_heads = 4
size_per_head = n_embd // num_heads

attention_heads = torch.nn.ModuleList(
    [AttentionHead(n_embd, size_per_head, block_size) for _ in range(num_heads)]
)

head_outputs = [head(attention_x) for head in attention_heads]
multi_head_output = torch.cat(head_outputs, dim=-1)

multi_head_parameter_count = sum(
    parameter.numel() for parameter in attention_heads.parameters()
)

print("每个 Head 输出形状：", [output.shape for output in head_outputs])
print("Multi-Head 输出形状：", multi_head_output.shape)
print("Multi-Head 参数数量：", multi_head_parameter_count)
attention_projection_layer = torch.nn.Linear(n_embd, n_embd)

projected_attention_output = attention_projection_layer(multi_head_output)

projection_parameter_count = sum(
    parameter.numel()
    for parameter in attention_projection_layer.parameters()
)

same_shape_for_residual = projected_attention_output.shape == attention_x.shape

print("投影前形状：", multi_head_output.shape)
print("投影后形状：", projected_attention_output.shape)
print("输出投影参数数量：", projection_parameter_count)
print("可以进行残差连接：", same_shape_for_residual)
layer_norm_1 = torch.nn.LayerNorm(n_embd)

normalized_attention_input = layer_norm_1(attention_x)

normalized_head_outputs = [
    head(normalized_attention_input)
    for head in attention_heads
]

normalized_multi_head_output = torch.cat(
    normalized_head_outputs,
    dim=-1,
)

normalized_projected_output = attention_projection_layer(
    normalized_multi_head_output
)

attention_residual_output = attention_x + normalized_projected_output

layer_norm_parameter_count = sum(
    parameter.numel()
    for parameter in layer_norm_1.parameters()
)

normalized_mean = normalized_attention_input[0, 0].mean().item()
normalized_std = normalized_attention_input[0, 0].std(
    unbiased=False
).item()

print("归一化输入形状：", normalized_attention_input.shape)
print("归一化多头输出形状：", normalized_multi_head_output.shape)
print("残差连接输出形状：", attention_residual_output.shape)
print("LayerNorm 参数数量：", layer_norm_parameter_count)
print("归一化后平均值：", normalized_mean)
print("归一化后标准差：", normalized_std)
layer_norm_2 = torch.nn.LayerNorm(n_embd)

normalized_feed_forward_input = layer_norm_2(attention_residual_output)

feed_forward_hidden_size = 4 * n_embd

feed_forward_up = torch.nn.Linear(
    n_embd,
    feed_forward_hidden_size,
)

feed_forward_activation = torch.nn.GELU()

feed_forward_down = torch.nn.Linear(
    feed_forward_hidden_size,
    n_embd,
)

expanded_features = feed_forward_up(
    normalized_feed_forward_input
)

activated_features = feed_forward_activation(expanded_features)

feed_forward_output = feed_forward_down(activated_features)

transformer_block_output = attention_residual_output + feed_forward_output

feed_forward_parameter_count = (
    feed_forward_up.weight.numel()
    + feed_forward_up.bias.numel()
    + feed_forward_down.weight.numel()
    + feed_forward_down.bias.numel()
)

print("Feed Forward 输入形状：", normalized_feed_forward_input.shape)
print("扩大后形状：", expanded_features.shape)
print("GELU 后形状：", activated_features.shape)
print("缩小后形状：", feed_forward_output.shape)
print("Transformer Block 输出形状：", transformer_block_output.shape)
print("Feed Forward 参数数量：", feed_forward_parameter_count)
