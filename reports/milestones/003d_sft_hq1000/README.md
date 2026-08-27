# M003d：高质量 SFT 1000 条数据集

## 结果

构造 `sft_hq1000_v2`，共1000条，切分为800条训练、100条验证和100条测试。原有100条证据事实问答以及 `test_001` 至 `test_005` 五道固定基准题全部保留。

新增900条围绕150个可在授权原文中定位的概念生成。新增问题不复制长篇小说上下文，只使用经过人工定义的概念名称和类别，避免把原始文本里的OCR错字带进问题。

## 数据组成

| 数据类型 | 数量 | 用途 |
|---|---:|---|
| 原有证据事实问答 | 100 | 学习直接回答领域问题 |
| 干净概念分类问答 | 900 | 学习多种问法、判断、纠错和EOS停止 |
| 合计 | 1000 | 字符级SFT v1正式数据 |

150个新增概念覆盖：

| 类别 | 概念数 |
|---|---:|
| 人物 | 46 |
| 斗技或功法 | 20 |
| 异火 | 15 |
| 丹药 | 15 |
| 势力 | 26 |
| 地点 | 18 |
| 物品或体质 | 10 |

每个概念产生5至7种问法，包括直接分类、完整类别判断、肯定判断、错误类别纠正、多项选择、一句话解释和分类体系问答。同一概念的全部变体只进入一个集合，防止训练集中的一种问法泄漏为验证集中的另一种问法。

## 为什么没有沿用最初的900条原文上下文题

初版自动方案的标签可以机械验证，但抽样发现原始小说中仍有OCR错字、半角标点和截断句。该方案已淘汰，构建程序保存在 `rejected_context_builder.py`，没有进入最终训练张量。这段失败记录可用于视频解释“数量正确不等于数据质量高”。

## 数据质量

- 1000个问题全部唯一。
- 150个新增概念分别使用150个不同原文行作为存在性证据。
- 新增概念没有跨训练、验证、测试集合泄漏。
- 最长序列51个字符级Token，小于模型256 Token上下文。
- 原100条历史数据在第378行存在一处训练/验证来源重叠；为保持历史实验可比性，本阶段保留并披露，没有新增来源行泄漏。
- 48项自动化测试全部通过。

需要明确：这是一套标签干净、可复现的“概念分类+基础事实问答”数据，不等于1000条开放式知识问答。若要继续提升解释能力，下一阶段应增加人工或教师模型审核的原因、关系和长答案数据。

## 关键文件

- 完整数据：`data/sft/sft_hq1000_v2.jsonl`
- 新增900条：`data/sft/sft_hq1000_expansion900_v2.jsonl`
- 训练张量：`data/sft/sft_hq1000_v2_tensors.pt`
- 构建程序：`build_sft_hq1000.py`
- 自动化测试：`tests/test_build_sft_hq1000.py`
- 固定质量样本：`sft_hq1000_quality_sample24.json`
- 机器报告：`sft_hq1000_report.json`
- 质量审查：`VALIDATION.md`

## 复现命令

```bash
cd /Users/bytedance/Documents/ChatGPT/game/create-gpt-step-by-step
source .venv/bin/activate
python build_sft_hq1000.py
python -m unittest discover -s tests -v
```

## 日志与独立调试

日志按生成、验证、输出分开，每个文件最大1 MB并保留3个轮转备份，不记录密码、Token或授权秘密。

| 模块 | 日志 | 级别环境变量 |
|---|---|---|
| 概念与问法生成 | `logs/sft_hq_generation.log` | `SFT_HQ_GENERATION_LOG_LEVEL` |
| 数据验证 | `logs/sft_hq_validation.log` | `SFT_HQ_VALIDATION_LOG_LEVEL` |
| 文件输出 | `logs/sft_hq_output.log` | `SFT_HQ_OUTPUT_LOG_LEVEL` |

级别可使用 `DEBUG`、`INFO`、`WARNING`、`ERROR` 或 `OFF`。`SFT_CONSOLE_LOG=0` 只关闭终端输出，不影响文件日志。例如只观察验证模块：

```bash
SFT_HQ_GENERATION_LOG_LEVEL=OFF \
SFT_HQ_OUTPUT_LOG_LEVEL=OFF \
SFT_HQ_VALIDATION_LOG_LEVEL=DEBUG \
python build_sft_hq1000.py
```

## 下一步

从 `checkpoints/archive/sft_stage1_init_pre_sft.pt` 重新开始训练。建议先跑20步安全测试，再根据验证Loss运行约3至5个epoch；800条训练数据、Batch Size 4时，每个epoch约200步。
