# M003c：SFT 数据扩充至 500 条

## 本阶段做了什么

在完全保留 M003b 100 条数据的前提下，新增 400 条由授权小说原文确定性生成的补全指令，形成 `sft_pilot500_v1`。

最终切分如下：

| 集合 | 原有 | 新增 | 合计 |
|---|---:|---:|---:|
| 训练集 | 80 | 320 | 400 |
| 验证集 | 15 | 60 | 75 |
| 测试集 | 5 | 20 | 25 |
| 总计 | 100 | 400 | 500 |

原来的 `test_001` 至 `test_005` 保持不变，可继续用于预训练、50条 SFT、100条 SFT和500条 SFT之间的公平比较。

## 新增数据是什么

原100条是带原文证据的短事实问答。新增400条是原文补全指令：从一条20至46字符的干净原文中，在自然逗号处拆分，问题提供前半句，回答监督后半句和 `<EOS>`。

四种提示各100条：

1. `请补全这句原文：...`
2. `请续写这句原文：...`
3. `原文接下来是什么：...`
4. `请把后半句补上：...`

这种数据适合训练模型识别用户指令、进入助手回答区并主动输出 `<EOS>`。它不是400条人工审核的事实问答，不能据此宣称模型增加了400个可稳定问答的知识点。

## 数据质量结果

- 500条记录和500个问题均唯一。
- 新增400条分别来自400个不同原文行。
- 每条新增答案均可在记录的 `source_line` 中精确还原。
- 新增记录没有训练、验证、测试来源行交叉。
- 原100条历史数据在原文第378行存在一处训练/验证共同来源；为保持旧实验可比性，本阶段没有修改它，报告中已单独披露。
- 词表仍为4478个基础字符加5个特殊 Token，共4483。
- 全项目42项自动化测试通过。

详细机器可读结果见 `sft_data500_report.json`，方法学复核见 `VALIDATION.md`。

## 复现

```bash
cd /Users/bytedance/Documents/ChatGPT/game/create-gpt-step-by-step
source .venv/bin/activate
python build_sft_pilot500.py
python -m unittest discover -s tests -v
```

## 日志与独立排查

日志按功能拆分并轮转，每个文件最大1 MB，保留3个备份；不记录 Token、密码或私密授权信息。

| 功能 | 日志文件 | 独立开关 |
|---|---|---|
| 原文候选生成 | `logs/sft_generation.log` | `SFT_GENERATION_LOG_LEVEL` |
| 500条数据校验 | `logs/sft_validation500.log` | `SFT_VALIDATION500_LOG_LEVEL` |
| 张量与报告输出 | `logs/sft_output500.log` | `SFT_OUTPUT500_LOG_LEVEL` |

级别可设为 `DEBUG`、`INFO`、`WARNING`、`ERROR` 或 `OFF`。设置 `SFT_CONSOLE_LOG=0` 可以关闭终端日志，但文件日志仍按各模块级别工作。例如只排查验证：

```bash
SFT_GENERATION_LOG_LEVEL=OFF \
SFT_OUTPUT500_LOG_LEVEL=OFF \
SFT_VALIDATION500_LOG_LEVEL=DEBUG \
python build_sft_pilot500.py
```

验证日志保留了两次被及时拦截的开发错误：一次是分组排序实现错误，另一次是发现旧100条中第378行的历史来源重叠。两次失败都发生在最终数据写出之前。

## 下一步

从同一个 `sft_stage1_init_pre_sft.pt` 初始化模型，用500条张量进行一轮短 SFT。不能从20步试跑 checkpoint 接着训练，否则无法把效果变化主要归因于数据量。
