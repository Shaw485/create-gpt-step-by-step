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
