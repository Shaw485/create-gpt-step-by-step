# M021 SFT v7 语义密度只读审计

- 状态：`needs_revision`
- 风险级别：`high`
- 决策门统计范围：`train`；`val` 与总体统计只用于描述，不参与 Stop/Go。
- 审计记录：8800
- 监督 Token：577646
- 数据边界：只读取 `train` 与 `val`；未读取 Public/Sealed 正文。

## 六种固定开头

| 开头ID | 固定前缀 | Train | Val | 合计 |
|---|---|---:|---:|---:|
| `original_text_says` | `原文写道“` | 666 | 68 | 734 |
| `material_states` | `材料表述为“` | 666 | 67 | 733 |
| `passage_verifiable` | `片段可核对：` | 666 | 66 | 732 |
| `this_place_says` | `这处写的是“` | 666 | 66 | 732 |
| `text_contains` | `文本中出现“` | 666 | 66 | 732 |
| `reviewable_original_sentence` | `可复查的原句是“` | 666 | 66 | 732 |

总体命中：4395 / 8800（49.94%）；Train：3996 / 8000（49.95%，参与门控）；Val：399 / 800（49.88%，仅描述）。

## 各维度监督密度（总体，保留旧口径）

| 维度 | 记录 | 记录占比 | 监督Token | Token占比 | 相对密度 |
|---|---:|---:|---:|---:|---:|
| `parameter_core_fact_and_correction` | 1584 | 18.00% | 64319 | 11.13% | 0.619 |
| `single_passage_grounded_qa` | 2816 | 32.00% | 149779 | 25.93% | 0.810 |
| `multi_passage_rag_evidence_composition` | 1232 | 14.00% | 149957 | 25.96% | 1.854 |
| `vertical_chat_multiturn_eos` | 1584 | 18.00% | 132652 | 22.96% | 1.276 |
| `novel_summary_rewrite_short_continuation` | 1144 | 13.00% | 64121 | 11.10% | 0.854 |
| `capability_boundary_clarification_evidence_request` | 440 | 5.00% | 16818 | 2.91% | 0.582 |

核心维度相对密度分范围：

| 范围 | 记录（分子/分母） | 记录占比 | 监督Token（分子/分母） | Token占比 | 相对密度 | 用于门控 |
|---|---:|---:|---:|---:|---:|---|
| Train | 1440 / 8000 | 18.00% | 58462 / 524886 | 11.14% | 0.619 | 是 |
| Val | 144 / 800 | 18.00% | 5857 / 52760 | 11.10% | 0.617 | 否 |
| 总体 | 1584 / 8800 | 18.00% | 64319 / 577646 | 11.13% | 0.619 | 否 |

## 核心问题、证据复制与循环初筛

- 裸核心问题总体：命中 0 条，覆盖 0 / 18 个事实（0.00%，仅描述）。
- 裸核心问题 Train：命中 0 条，覆盖 0 / 18 个事实（0.00%，参与门控）；Val 覆盖 0 / 18（0.00%，仅描述）。
- 证据复制总体：5050 / 7651（66.00%，仅描述）；Train：4592 / 6956（66.01%，参与门控）；Val：458 / 695（65.90%，仅描述）。
- 自指初筛：0；两节点循环：0；重复子句：0。
- 自指/循环决策只使用 Train：自指 0，循环 0；总体结果仅是关键词启发式初筛，命中项必须再做语义审核。

## 风险结论（仅由 Train 决定）

- `P1` `fixed_opening_template_concentration`：范围 `train`，观测值 `0.4995`，门槛 `<=0.20`。
- `P1` `bare_core_question_coverage_too_low`：范围 `train`，观测值 `0.0`，门槛 `>=0.50`。
- `P1` `core_supervision_density_too_low`：范围 `train`，观测值 `0.61877989`，门槛 `>=0.75`。
- `P1` `evidence_copy_concentration`：范围 `train`，观测值 `0.66014951`，门槛 `<=0.50`。

当前结论：在继续增加训练步数前，先重构 SFT v7.1 的语义密度与裸问题监督。

## 可复算信息

- `train`：`data/sft/v7/train.jsonl`，记录 8000，SHA-256 `0f894ef6ab5413f25a86fabbe530c6607a8bd730301a5529eef13e1579c6bb81`。
- `val`：`data/sft/v7/val.jsonl`，记录 800，SHA-256 `257efe75da51df4f5de239c08e5d9d72d03dfbb6068c50b3e369b397023dd58c`。
- 实现：`audit_sft_v7_semantic_density.py`，算法版本 `semantic-density-train-gates/2.0`，SHA-256 `c4049d8aff90a7dea0303a1edce9b4cf20681687fb14b58be1c250cc2375b232`，Git `dbb436d7e459033abe03b87e76fd2864206e9159` （dirty）。
