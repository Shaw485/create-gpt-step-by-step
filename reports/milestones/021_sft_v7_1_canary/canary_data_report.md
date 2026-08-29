# M021 SFT v7.1 Canary 数据报告

状态：**pass**

## 数量

| 项目 | 数量 |
|---|---:|
| 已审核事实 | 8 |
| 训练问法 | 64 |
| 未见改写评估 | 16 |
| 总记录 | 80 |

## 事实覆盖

| fact_id | 实体 | required terms | Train | Holdout |
|---|---|---|---:|---:|
| `xiaoyan_identity` | 萧炎 | 萧炎、萧战 | 8 | 2 |
| `xiaozhan_identity` | 萧战 | 萧战、萧炎、族长 | 8 | 2 |
| `yaochen_identity` | 药尘 | 药尘、药老、老师 | 8 | 2 |
| `yaolao_yaochen_alias` | 药老 | 药老、药尘、同一人物 | 8 | 2 |
| `yaolao_teacher` | 药老 | 药老、萧炎、老师 | 8 | 2 |
| `fanjue_identity` | 焚决 | 焚决、功法、异火 | 8 | 2 |
| `yihuo_role` | 异火 | 异火、炼药 | 8 | 2 |
| `yunlanzong_identity` | 云岚宗 | 云岚宗、加玛帝国 | 8 | 2 |

## 质量门

| 质量门 | 结果 |
|---|---|
| `exactly_eight_reviewed_facts` | PASS |
| `exactly_64_training_records` | PASS |
| `exactly_16_unseen_holdout_paraphrases` | PASS |
| `all_required_terms_present` | PASS |
| `zero_banned_meta_prefixes` | PASS |
| `zero_exact_or_normalized_question_duplicates` | PASS |
| `train_and_eval_roles_unambiguous` | PASS |
| `eos_deferred_to_encoding` | PASS |
| `public_and_sealed_bodies_untouched` | PASS |
| `frozen_lineage_files_verified` | PASS |

## 数据角色

`train.jsonl` 的64条记录是优化目标；`holdout_eval.jsonl` 的16条记录只用于未见改写评估，`use_for_training=false`。JSONL中不写入字面量 `<EOS>`，编码器在每个assistant答案末尾追加EOS。

问题与答案来自已审核的 `configs/sft_v7_1_canary_facts.json`，由 Codex AI 辅助构造并逐项与 `KNOWN_CORE_FACTS` 交叉核对；这不是独立真人签字。父 v7 `manifest.json` 只提供冻结版本和数据集身份元数据。构建还会对 Step 5750 基座、BPE tokenizer 与 token manifest 做路径和 SHA-256 闭环校验，但不会加载权重，也不会读取v7 train、public、sealed JSONL正文或正式预训练语料正文。

## 日志和独立调试

日志位于 `logs/sft_v7_1_canary_build/`，data、validation、orchestrator分别写入轮转JSONL。可使用 `--data-log-level`、`--validation-log-level`、`--orchestrator-log-level` 独立调整，也可设置 `GPT_CANARY_LOG_LEVEL_DATA` 等环境变量。日志只包含数量、状态、SHA和错误码，不包含问题、答案或消息正文；敏感字段由公共日志组件自动脱敏。默认单文件1 MiB、保留3份轮转备份。

调试顺序：data失败先核对配置和父manifest；validation失败查看质量门错误码；orchestrator失败查看最终状态和remediation。生产运行保持INFO，只有定位单个模块时临时启用DEBUG。

## 完整性

- Dataset identity: `b2012953980c823d018494d2a5212e79b51370c632dceb1e559b552f152f92b9`
- Manifest SHA-256: `68908fdabe4f8ae470f6bcd4ec6d11b59304829119835af901df7bf9888ef50d`
- Train SHA-256: `e5f0f90b26f9dbacb68017bbaa4243a41ccdacb4ac60484677964061fd4d008a`
- Holdout SHA-256: `fe8a72efcd8e3f179d61ca8e4b2de2b3c775dbd9841d25692fe6743f3548c64d`
