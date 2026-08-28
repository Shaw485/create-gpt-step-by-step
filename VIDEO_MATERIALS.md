# 项目视频素材索引

## 用途

本文件维护“手搓 GPT”项目从预训练、SFT 到最终模型的可复现素材。项目视频应优先引用这里列出的里程碑文件，不直接使用可能被后续运行覆盖的临时输出。

## 留档规则

每次模型结构、数据、训练目标或生成策略发生实质变化后，都建立一个新的里程碑编号，并保留以下材料：

1. 模型阶段、参数配置、训练数据和目标函数。
2. 独立命名的 checkpoint，不覆盖上一阶段模型。
3. Loss、Top-1、Top-5、重复率等指标及计算口径。
4. 同一组固定 prompt 的完整输入与输出。
5. Loss 曲线、CSV 数据表和机器可读 JSON 报告。
6. 本阶段做了什么、改善了什么、仍有什么问题。
7. 关键文件 SHA-256，确认视频制作时材料没有变化。

所有 SFT 文件使用 `sft_` 前缀，不能覆盖 `gpt_stage5_pretrain_step10000_best.pt`。后续比较必须使用同一组固定 prompt、相同最大生成长度和明确记录的采样参数。

注意：`checkpoints/` 当前被 `.gitignore` 排除，模型 checkpoint 只保存在本机，不会随 GitHub 提交上传；文档、指标和图表可以进入版本管理。项目收尾和视频制作前，需要再把归档 checkpoint 复制到用户指定的外部备份位置，并用本里程碑 SHA-256 复核。

## 里程碑索引

| 编号 | 阶段 | 状态 | 核心材料 |
|---|---|---|---|
| M001 | 字符级领域预训练，Stage 5 Step 10000 | 已归档 | [里程碑说明](reports/milestones/001_pretrain_stage5_step10000/README.md) |
| M002 | 字符级模型 SFT 数据集 v1 | 已完成 | [数据设计与质量说明](reports/milestones/002_sft_dataset_v1/README.md) |
| M002a | 50条蒸馏式 SFT 试验数据 | 已完成 | [试验数据说明](reports/milestones/002a_sft_pilot50_v1/README.md) |
| M002b | SFT前固定测试题基线 | 已完成 | [完整基线结果](reports/milestones/002b_pre_sft_baseline/README.md) |
| M003a | SFT 20步安全试跑 | 已完成 | [试跑结果](reports/milestones/003a_sft_smoke20/README.md) |
| M003b | SFT数据扩充至100条 | 已完成 | [扩充结果](reports/milestones/003b_sft_data100/README.md) |
| M003c | SFT数据扩充至500条 | 已完成 | [扩充与校验结果](reports/milestones/003c_sft_data500/README.md) |
| M003d | 高质量SFT数据扩充至1000条 | 已完成 | [构造与质量报告](reports/milestones/003d_sft_hq1000/README.md) |
| M003e | 1000条数据的20步安全试跑 | 已完成 | [试跑结果](reports/milestones/003e_sft_hq1000_smoke20/README.md) |
| M003f | 1000条数据的800步SFT与正式评估 | 已完成 | [训练、曲线与前后对比](reports/milestones/003f_sft_hq1000_step800/README.md) |
| M003g | 平衡版SFT v3数据重构 | 已完成 | [任务比例、质量边界与训练协议](reports/milestones/003g_sft_balanced_v3/README.md) |
| M003h | 平衡版SFT v3的20步安全试跑 | 已完成 | [试跑结果](reports/milestones/003h_sft_balanced_v3_smoke20/README.md) |
| M003i | 平衡版SFT v3的800步训练与公平评估 | 已完成 | [Loss、六类指标与固定10题](reports/milestones/003i_sft_balanced_v3_step800/README.md) |
| M003 | 字符级模型 SFT v1 | 已完成 | M003a至M003i完整记录，最终选用平衡版第400步实验模型 |
| M004 | 手写BPE与10,000步从零预训练 | 已完成 | [词表、压缩率、预训练与样本演化](reports/milestones/004_bpe_pretrain/README.md) |
| M005 | BPE模型800步SFT与字符版公平对比 | 已完成 | [Loss、六类指标与固定10题](reports/milestones/005_bpe_sft/README.md) |
| M006 | v4正式语料与本机810万参数预训练 | 已完成 | [语料审核、Loss、固定10题与完整归档](reports/milestones/006_v4_local_pretrain/README.md) |

## 视频中的核心对比表

| 指标 | M001 预训练后 | M003 字符级 SFT 后 | M005 BPE+SFT 后 |
|---|---:|---:|---:|
| 验证 Loss | 3.0863 | 3.4099（仅回答区，不可直接同比） | 预训练4.8036/BPE Token；SFT 3.7711（回答区） |
| Top-1 | 42.71% | 不适用 | 待测 |
| Top-5 | 62.86% | 不适用 | 待测 |
| 相邻字符重复率 | 1.30% | 未作为SFT主指标 | 待测 |
| 问答相关性 | 基本没有 | 精确38/100 | 精确36/100；包含答案42/100 |
| EOS停止率 | 未实现 | 100/100 | 100/100 |
| 固定10题输出 | 已保存 | 已保存四模型完整对比 | 已保存字符/BPE四状态完整对比 |

## 当前路线决定

Stage 5 已经形成小说语言能力，但训练目标仍是小说续写。继续在同一字符级小说语料上增加预训练步数，只会带来逐渐变小的续写收益，不能直接解决问答目标不匹配。

字符级SFT与BPE路线均已完成。BPE将654万字符压缩为406.8万Token，在同一Block Size下扩大约1.61倍有效字符上下文，但1000条SFT上的严格正确率未超过字符版。下一阶段不应盲目增加Step，而应增加高质量独立事实、自然对话与多样化回答，并单独改进清洗语料中的站点/作者噪声。

M006已经按章节组重新冻结612万字符正式语料，并在Apple M4上完成810万参数、3000步从零预训练。最佳验证Loss为5.0447/BPE Token，所选Step 2600模型Test Loss为4.9958；固定10题仍是小说续写而非回答。下一阶段进入v4 SFT，必须从M006最佳checkpoint开始并保留同一评测协议。
