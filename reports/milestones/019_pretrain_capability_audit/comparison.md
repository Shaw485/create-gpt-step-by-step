# 单本小说预训练能力自动汇总

> 本表不包含主观人工评分。固定窗口 Token Loss/PPL/Top-1 与训练历史 BPC 是不同口径。
> Token bits/token = 固定窗口 Token Loss ÷ ln(2)；BPC 只引用完整 Validation 训练历史，绝不由前者冒充。

| Step | Checkpoint SHA | Token Loss | bits/token | Validation BPC | Token PPL | Next-token Top-1 | Empty | EOS | Degeneration | Unique | 4-gram repeat |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 250 | `20fa8bb335d8…` | 6.998717 | 10.097015 | 5.914876 | 1095.228 | 7.806% | 0.0% | 0.0% | 12.5% | 0.397 | 0.013 |
| 5750 | `bfe4fec5e604…` | 4.438907 | 6.403989 | 3.761229 | 84.682 | 24.170% | 0.0% | 0.0% | 0.0% | 0.636 | 0.021 |
| 6000 | `bfc05c0c045e…` | 4.430017 | 6.391163 | 3.753153 | 83.933 | 24.336% | 0.0% | 0.0% | 6.2% | 0.626 | 0.028 |

## Cloze 四类排名

### Step 250

| Metric | Overall Top-1/MRR | High Top-1/MRR | Low Top-1/MRR |
|---|---:|---:|---:|
| total_log_probability | 25.0% / 0.514 | 50.0% / 0.667 | 0.0% / 0.361 |
| mean_token_log_probability | 33.3% / 0.576 | 50.0% / 0.667 | 16.7% / 0.486 |
| per_character_log_probability | 25.0% / 0.514 | 50.0% / 0.667 | 0.0% / 0.361 |
| context_lift | 58.3% / 0.701 | 66.7% / 0.764 | 50.0% / 0.639 |

### Step 5750

| Metric | Overall Top-1/MRR | High Top-1/MRR | Low Top-1/MRR |
|---|---:|---:|---:|
| total_log_probability | 66.7% / 0.806 | 83.3% / 0.889 | 50.0% / 0.722 |
| mean_token_log_probability | 66.7% / 0.806 | 83.3% / 0.889 | 50.0% / 0.722 |
| per_character_log_probability | 66.7% / 0.806 | 83.3% / 0.889 | 50.0% / 0.722 |
| context_lift | 66.7% / 0.819 | 100.0% / 1.000 | 33.3% / 0.639 |

### Step 6000

| Metric | Overall Top-1/MRR | High Top-1/MRR | Low Top-1/MRR |
|---|---:|---:|---:|
| total_log_probability | 66.7% / 0.819 | 83.3% / 0.917 | 50.0% / 0.722 |
| mean_token_log_probability | 66.7% / 0.833 | 83.3% / 0.917 | 50.0% / 0.750 |
| per_character_log_probability | 66.7% / 0.819 | 83.3% / 0.917 | 50.0% / 0.722 |
| context_lift | 66.7% / 0.799 | 100.0% / 1.000 | 33.3% / 0.597 |

## 冻结门槛

- `language_base`：**manual_review_required**
- `practical_plateau`：**manual_review_required**
- `mature_novel_generator`：**not_assessed_requires_independent_human_review**

自动汇总不会给流畅度、连贯度、承接、人物一致性或成熟生成器状态打分。
