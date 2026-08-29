# M020：小说垂直 SFT v7

## 当前状态

`EXPERIMENT_COMPLETE / NOT_RELEASED`：10,000 条垂直 SFT、Step 5750 冻结基线、20 步安全试跑、2,000 步正式训练、四个里程碑 public 评估、固定 16 题 AI 辅助复核和预训练保持评估均已完成。最终汇总状态为 `automatic_gates_failed_external_review_pending`，严格候选为空，因此本阶段没有合格发布候选。

## 本阶段目标

从纯预训练 Step 5750 重新开始，把单本小说语言基座调整为《斗破苍穹》垂直助手：

1. 核心人物、别名、关系和设定能自然直接回答。
2. 给定小说证据后能抽取、概括、比较和解释。
3. 对首次出现、全书计数、精确章节和长尾多跳关系使用检索，不靠参数猜。
4. 学会领域多轮、指令遵循、EOS 和自然停止。
5. 不加入数学、通用百科或项目学习问答。

完整冻结规则见 [小说垂直 SFT v7 协议](../../../docs/sft_v7_vertical_protocol.md)。

## 固定初始化点

| 项目 | 冻结值 |
|---|---|
| Checkpoint | `runs/formal_pretrain_14m_bpe3000_6000/checkpoints/step_05750.pt` |
| Checkpoint SHA-256 | `bfe4fec5e6045d4c06d22393e7c2079fdc03897be71829c9d9dcbaf0fcaf5c1e` |
| 参数量 | 14,880,745 |
| BPE 词表 | 7,465（含 6 个特殊 Token） |
| Context | 512 Token |
| Tokenizer SHA-256 | `e70cf3dc0ed185a6b22ab7dc08b6a850eeb59864ba161dd156c644e003862822` |

M018 SFT Step 2000 只作为失败对照，不允许继续训练；它已经受到固定模板与任务串扰影响。

## 为什么不能直接重训 v6

独立审计只读取 v6 的 train、val 和 public 前 9,400 行，未读取末 600 行正文。报告状态为 `needs_revision`，SHA-256 为 `18852e6a6608c2c1ed920a2cea895949d488482c11b887f5cc1a92cb0cb92046`。结论：

- 训练集中至少 25.5% 明显偏离小说垂直目标。
- 1,500 条自然对话含高度固定的学习建议结构；确定存在 300 条“先先”拼接错误。
- 3,000 条所谓实体事实主要是 6 套局部专名抽取模板，不是身份、关系、事件或因果问答。
- 萧炎约占训练实体元数据 46.8%，长尾实体覆盖失衡。
- 旧编码工件曾物理载入并保存 sealed 记录；“未参与梯度”不等于“从未读取”。M020 对这一表述作出纠正。

因此 v6 判定为 `needs_revision`，只保留失败实验与工程经验，不迁移原模板或 SFT 权重。

## v7 冻结配额

| 维度 | 总量 | Train | Val | Public | Sealed |
|---|---:|---:|---:|---:|---:|
| 核心事实与纠错 | 1800 | 1440 | 144 | 108 | 108 |
| 单段证据问答 | 3200 | 2560 | 256 | 192 | 192 |
| 多片段 RAG 证据组合 | 1400 | 1120 | 112 | 84 | 84 |
| 垂直聊天、多轮与 EOS | 1800 | 1440 | 144 | 108 | 108 |
| 小说摘要、改写与短续写 | 1300 | 1040 | 104 | 78 | 78 |
| 边界、澄清与索取证据 | 500 | 400 | 40 | 30 | 30 |

## 物理隔离设计

- `train.jsonl` 与 `val.jsonl` 编码为独立的 `train_val_tensors.pt`。
- `public_diagnostic.jsonl` 编码为独立的 `public_diagnostic_tensors.pt`。
- `sealed_test.jsonl` 不进入常规编码器、训练器或公开评估器的 CLI、payload、日志与缓存。
- 训练器会递归拒绝任何包含 public、test 或 sealed 键的输入工件。
- 新 sealed 在构建期独立验证后冻结；正式一次性使用前只暴露 SHA、schema 和计数。

含小说证据的 JSONL 与张量不进入 Git；代码、聚合清单、验证报告、曲线和不泄漏密封题面的样本材料可以进入里程碑。

## 数据构建与独立验证结果

正式发布清单 SHA-256 为 `422c35fa130a3e6fc3f656019515fc7c1115616aa396712ea501751aeda1b9e9`，dataset identity 为 `47fb0f0af2aaa61f3239883f22966f3b0828a8a3942c78436b07ef5f5118d133`。四个 split 为 `8000/800/600/600`，六维实际数量与冻结配额完全一致。清单中的语料、Tokenizer 与 split 路径均为稳定仓库相对路径；从旧清单迁移时只改了两个路径元数据字段，没有读取或重建密封正文。

- 已审核核心事实 18 项，直接核心问答 900 条，覆盖 50 个核心实体或概念；核心已知事实误拒为 0。
- 真实多轮 1,200 条，2–4 片段 RAG 1,400 条，单证据与 RAG 负例比例均为 17.5%，三联校准 130 组。
- 33–96 BPE Token 回答占 79.09%，97–160 Token 回答占 10.09%；最大固定 12 字短语占比 2.00%，通用完整答案最多重复 4 次。
- 构建期独立验证覆盖全部 10,000 条，P0=0、P1=0、风险记录=0；精确/规范化问题重复、semantic/evidence/chapter 跨 split 泄漏均为 0，不可编码和超过 512 Token 均为 0。
- 构建期密封验证报告 SHA-256 为 `13f5504f467d38a25dc54a478011f42e12ce2418017cc849d5811802446723f4`。密封正文只在新发布构建时显式读取一次，之后已冻结；常规验证报告明确 `sealed_body_accessed=false`。

第一次独立联调曾发现验证器把重复章节编号误当成同一来源章节、把三个已审别名误判为证据缺失，并把固定 public 验收 ID 误当三联 ID；同时发现边界样本 semantic group 含正文实体名。前者按真实 heading line + chapter SHA、已审别名和独立 ID 语义修正验证器，后者改为稳定哈希并重建全量数据；修正后重新运行完整双重验证，不绕过任何冻结阈值。

## BPE 编码结果

| 工件 | 记录 | 监督 Token | SHA-256 |
|---|---:|---:|---|
| Train | 8,000 | 524,886 | 与 Val 共同封装于训练工件 |
| Val | 800 | 52,760 | 与 Train 共同封装于训练工件 |
| Public | 600 | 39,338 | `bf28705327c416b7a8de37a913ca85b0a28ac5faa4839b7b412ce69817c90166` |
| Train + Val 张量 | 8,800 | 577,646 | `81a0654150e8834cf4bd0ef30cbdb7975eac93fe5f1acf8208c715b0badfef47` |

最长序列为 393 Token。编码器逐条验证 assistant-only mask、每个 assistant turn 的 EOS、BPE 7,465 词表、Step 5750 来源和数据发布清单；训练 payload 不含 public、sealed 或 test 字段，public payload 与训练 payload 物理分离。

## 工程诊断

M020 的构建、验证、编码、训练、生成、公开评估、checkpoint 与编排日志按模块分离，支持单独 `OFF/INFO/DEBUG`。日志为带 UTC 时间和 run id 的轮转 JSONL；默认不记录小说正文、完整提示、Token、密码、密钥或授权信息。

## Step 5750 冻结基线

公开基线使用 600 条 `public_diagnostic`、低温 greedy（`temperature=0.3`、`top_k=1`）和最多 160 个新 Token。它评估的是纯预训练 Step 5750，还没有经过 M020 SFT：

| 指标 | 结果 |
|---|---:|
| Public teacher-forced Loss | 5.86808865 |
| EOS 停止率 | 7.1667% |
| 空回答率 | 1.1667% |
| 达到长度上限率 | 92.8333% |
| 机械重复率 | 85.5000% |

这组结果符合预期边界：Step 5750 是小说语言续写基座，不是已经会问答和自然停止的聊天模型。冻结 16 题完整对照中，0/16 生成 EOS、16/16 达到 128 Token 上限；完整输出见 [固定 16 题基线](fixed_samples_step05750.md)，公开分任务明细见 [Step 5750 public 评估](public_eval_step05750.md)。自动门未通过，8 项独立语义、事实、表达与预训练保持门仍为 `pending`，因此它不是发布候选。

## 20 步安全试跑

Smoke 从冻结纯预训练 Step 5750 全新初始化，使用 MPS、Batch 2、学习率 `2e-5`。前 10 步执行核心事实/垂直聊天/边界三池路由，后 10 步切换到六维完整混合；两个阶段均完成采样、前向、反向、验证和 checkpoint 保存，证明正式两阶段训练链路可运行。

| 指标 | 结果 |
|---|---:|
| Step 0 Validation Loss | 5.897454 |
| Step 10 Validation Loss | 5.695114 |
| Step 20 / 最佳 Validation Loss | 5.556195 |
| Public 记录参与训练 | 0 |
| Sealed 记录参与训练 | 0 |
| 状态 | `training_complete_public_evaluation_pending` |

Smoke 只验证安全性和工程闭环，不用于选择最终模型，也不从它续训正式候选。600 条完整 `public_diagnostic` 只要求冻结基线和正式训练的每 500 步节点运行；Smoke 不重复运行这组耗时评估，汇总中记为 `not_run_optional`，不算缺失。报告见 [smoke_train_report.json](smoke_train_report.json)，固定输出见 [fixed_samples_smoke20.md](fixed_samples_smoke20.md)。

## 2,000 步正式训练结果

正式训练再次从纯预训练 Step 5750 全新初始化，没有接续 smoke。设备为 MPS、Batch 4、学习率 `2e-5`；前 400 步使用核心事实/垂直聊天/边界路由，之后切换六维完整混合。2,000 个优化步耗时 1,978.1 秒，训练阶段 public 与 sealed 消耗均为 0。

| Step | Train Loss | Validation Loss |
|---:|---:|---:|
| 0 | 5.856427 | 5.897454 |
| 250 | 4.574566 | 4.443956 |
| 500 | 4.175485 | 4.080846 |
| 750 | 3.877102 | 3.835044 |
| 1,000 | 3.661784 | 3.671394 |
| 1,250 | 3.513502 | 3.563222 |
| 1,500 | 3.391344 | 3.481479 |
| 1,750 | 3.305156 | 3.412457 |
| 2,000 | 3.218450 | 3.356181 |

训练和验证 Loss 都持续下降，说明模型确实在拟合 SFT 目标；这只是学习信号，不是发布证明。完整报告见 [formal_train_report.json](formal_train_report.json)，曲线数据与图见 [sft_v7_loss_curve.csv](sft_v7_loss_curve.csv) 和 [sft_v7_loss_curve.svg](sft_v7_loss_curve.svg)。

## Public 行为对比

五个节点使用同一 600 条 public、相同低温 greedy 参数和相同评分器。SFT 明显提高 EOS 并降低截断，但机械重复没有随 Loss 单调改善；所有正式节点仍未通过自动行为门。

| 节点 | Public Loss | EOS | 截断 | 机械重复 | 自动行为门 |
|---|---:|---:|---:|---:|---|
| 纯预训练 Step 5750 | 5.868089 | 7.17% | 92.83% | 85.50% | 失败 |
| SFT Step 500 | 4.034095 | 73.33% | 26.67% | 41.67% | 失败 |
| SFT Step 1,000 | 3.611000 | 76.17% | 23.83% | 41.83% | 失败 |
| SFT Step 1,500 | 3.419921 | 79.50% | 20.50% | 39.50% | 失败 |
| SFT Step 2,000 | 3.294267 | 82.33% | 17.67% | 42.17% | 失败 |

逐节点分任务指标见 [Step 500](public_eval_step00500.md)、[Step 1000](public_eval_step01000.md)、[Step 1500](public_eval_step01500.md) 和 [Step 2000](public_eval_step02000.md)。汇总见 [checkpoint_comparison.md](checkpoint_comparison.md)。

## 固定 16 题 AI 辅助复核

同一 16 题、相同生成参数下，AI 辅助复核只把 Step 1500 的 1 题判为“维度基本通过”；这不是独立真人审核，也不能外推为整体正确率。

| 节点 | 基本通过 | 严重重复 | EOS | 截断 |
|---|---:|---:|---:|---:|
| 纯预训练 Step 5750 | 0/16 | 16/16 | 0/16 | 16/16 |
| SFT Step 500 | 0/16 | 13/16 | 9/16 | 7/16 |
| SFT Step 1,000 | 0/16 | 11/16 | 11/16 | 5/16 |
| SFT Step 1,500 | 1/16 | 8/16 | 11/16 | 5/16 |
| SFT Step 2,000 | 0/16 | 15/16 | 7/16 | 9/16 |

完整逐题复核见 [fixed_samples_milestone_review.md](fixed_samples_milestone_review.md)，原始完整输出分别保存在 `fixed_samples_step*.md/json`，没有用 30 字预览代替完整证据。

## 预训练能力保持

保持门要求固定窗口 BPC 相对纯预训练基线退化不超过 10%、16/16 续写非空且机械退化不超过 25%。四个正式节点都只有 13/16 非空；Step 1500 与 2000 还超过 BPC 退化阈值，因此没有任何节点通过完整保持门。

| Step | BPC 相对退化 | 16 题非空 | BPC 阈值 | 完整保持门 |
|---:|---:|---:|---|---|
| 500 | 6.007% | 13/16 | 通过 | 失败 |
| 1,000 | 8.995% | 13/16 | 通过 | 失败 |
| 1,500 | 10.217% | 13/16 | 失败 | 失败 |
| 2,000 | 12.084% | 13/16 | 失败 | 失败 |

报告见 [Step 500](pretrain_retention_step00500.md)、[Step 1000](pretrain_retention_step01000.md)、[Step 1500](pretrain_retention_step01500.md) 和 [Step 2000](pretrain_retention_step02000.md)。这些评估的 sealed 正文读取均为 0。

## 候选解释与最终决定

- **Step 1000：保守诊断候选，不是发布候选。** 它仍在 10% BPC 退化线内，但只有 13/16 预训练续写非空，固定 16 题基本通过为 0，public 自动行为门失败。
- **Step 1500：行为较优研究候选，不是发布候选。** 它在 public 上取得最低机械重复率，固定 16 题唯一出现 1 个基本通过；但 BPC 退化 10.217% 已越门，public 自动行为门仍失败。
- **Step 2000：排除。** 虽然 public Loss 最低、EOS 最高、截断最低，但固定 16 题严重重复反弹到 15/16，BPC 退化扩大到 12.084%。

最终汇总的严格候选列表为 `[]`，必需产物缺失 0、完整性错误 0，但 `release_ready=false`。独立事实、语义、表达和真人审核仍为 `pending`；由于自动门已经失败，本阶段不启封 sealed test、不发布权重，也不把任何节点称为合格模型。完整汇总与哈希入口见 [checkpoint_comparison.md](checkpoint_comparison.md) 和 [SHA256SUMS.md](SHA256SUMS.md)。

## 对外措辞边界

本阶段可以说“完成了一次可复现的小说垂直 SFT 实验，显著改善了 EOS 与截断，但没有产生通过全部门禁的发布候选”。不能说它已经是合格垂直助手、通用 GPT、掌握开放世界知识、精确记住整本小说全部事实，或仅凭 Loss 证明达到成熟聊天质量。
