# Replay 0.25 Step 400 命题级语义复核

- 来源：`canary_replay_w025_step00400_generation_eval.json`
- 审核范围：13 条规范化逐字不一致样本
- 审核性质：查看过问题、参考答案和生成答案的非盲 Codex AI 辅助交叉复核；不是独立真人签字
- Public 正文读取：0
- Sealed 正文读取：0

## 结论

13 条严格失败中，8 条是命题等价，5 条是真正错误，0 条暂定模糊。把命题等价计为正确后：

| Split | 严格完整匹配 | 命题正确 | 命题门槛 | 命题子门 |
|---|---:|---:|---:|---|
| Train | 56/64，87.50% | 61/64，95.31% | ≥95% | 通过 |
| Dev/Selection | 11/16，68.75% | 14/16，87.50% | ≥75% | 通过 |
| 合计 | 67/80，83.75% | 75/80，93.75% | 描述项 | — |

逐 fact 的 Train 最低值为 7/8，Dev/Selection 最低值为 1/2，也通过 Canary 的逐 fact 命题门。真正错误集中在三类：遗漏被问实体、师徒/父子串题、焚决与异火炼药串题。

## 逐条复核

| ID | Split | Fact | 结论 | 理由 |
|---|---|---|---|---|
| `canary-train-498a27c9d7ae7e548274` | Train | 萧炎身份 | 命题等价 | “萧战是萧炎的父亲”与“萧炎是萧战的儿子”互为方向正确的逆关系。 |
| `canary-train-368e10ddd48f3e54f74c` | Train | 萧炎身份 | 命题等价 | 父子关系及方向正确，能够唯一确定人物。 |
| `canary-train-ed7956ed23bc94d75fc3` | Train | 萧炎身份 | 命题等价 | 给出父子关系，已经反驳“没有亲属关系”。 |
| `canary-train-d23a3de1795fbcfae905` | Train | 药尘身份 | 真错误 | 问原名却只回答老师身份，遗漏“药尘”。 |
| `canary-train-f86bf1e53ee6218114f9` | Train | 药老/药尘别名 | 命题等价 | “药尘就是药老”明确表达同一人物。 |
| `canary-train-1cfd945f6afbc8c1e06e` | Train | 药老/药尘别名 | 命题等价 | “就是”表达同一人物，附加老师身份不改变结论。 |
| `canary-train-857ab2bf5fb49eb5ca4e` | Train | 药老/药尘别名 | 真错误 | 问曾用名却没有回答“药尘”。 |
| `canary-train-9a0df5dbd678666ec730` | Train | 焚决 | 真错误 | 串到异火炼药作用，遗漏“焚决”。 |
| `canary-holdout_eval-066362b97b82c5ec4814` | Dev/Selection | 萧炎身份 | 命题等价 | 父亲关系与儿子关系方向正确。 |
| `canary-holdout_eval-edd93740fbd1331097f1` | Dev/Selection | 萧炎身份 | 命题等价 | 正确完成父子方向纠错。 |
| `canary-holdout_eval-c123afb06dd3c274470a` | Dev/Selection | 药尘身份 | 命题等价 | 明确药尘和药老是同一人物，可推出常用称呼。 |
| `canary-holdout_eval-ab7c99b308ab0c51f65f` | Dev/Selection | 药老/萧炎师徒 | 真错误 | 问老师却串到萧炎与萧战的父子关系。 |
| `canary-holdout_eval-484c6c43b82e876519e8` | Dev/Selection | 焚决 | 真错误 | 问功法名称却串到异火炼药作用。 |

## 边界

这项复核证明规范化逐字匹配会误伤正确同义表述，但不会追溯把原严格 exact gate 改成通过。报告中的 `holdout_eval` 是代码兼容字段，实际是已用于 teacher-loss 与 checkpoint 选择的 Dev/Selection，不是盲测。该命题复核也不能替代 public、独立真人审核或 sealed 最终验收。

同一 checkpoint 的小说保持仍只有 15/16 条续写非空，因此即使命题问答子门通过，整体候选仍失败。
