# 云端无人值守训练运行手册

这套运行时用于正式 A10 训练的安全底座。它不会把现有训练脚本静默切到 CPU，也不会绕过尚未完成的数据复核。

## 固定配置和哈希

- 正式配置：`configs/cloud_a10.json`
- 配置哈希：`configs/cloud_a10.json.sha256`
- 配置格式版本：`cloud-training-config/v1`
- 本次配置版本：`1.0.0`

每次修改配置后都必须人工检查变更，然后重新生成 SHA-256。预检发现配置和哈希不一致时会停止，旧 checkpoint 也不能被不同配置误恢复。

## 数据放行门

正式训练只读取 `data/cloud_v4`，该目录是私有运行产物，不应提交到公开仓库。预检要求以下三份 manifest 均为 `status=ready`：

1. `corpus_manifest.json`：认证 train/val/test 文本。
2. `token_manifest.json`：认证训练集学习得到的 BPE tokenizer 和三份 Token 张量。
3. `sft_manifest.json`：认证 SFT train/val/test JSONL。

每份 manifest 自己必须带 `.sha256` sidecar；其中列出的每个文件还会重新计算 SHA-256。任何文件缺失、为空、未完成复核或复核后发生变化，预检都会停止并给出修复动作。

## 云实例预检

在项目根目录运行：

```bash
python cloud_preflight.py --config configs/cloud_a10.json
```

这条命令依次验证：

1. 配置版本和配置 SHA-256。
2. 三类发布数据的状态和文件 SHA-256。
3. CUDA 可用且没有回退到 CPU。
4. 可见 GPU 名称匹配 NVIDIA A10、计算能力不低于 8.6、显存不低于 22 GiB。
5. 启动前已有显存占用不超过 20%，预计正式峰值不超过总显存 80%。
6. 在目标卡上真实完成一次 BF16 前向、Loss、反向传播和优化器更新，并检查输出、Loss 和梯度全部有限。

失败时会返回稳定错误码、原因和下一项操作，并在本次 run 目录写入 `FAILED.json`。预检通过会写入 `preflight_report.json`；预检通过不等于整个训练完成，因此不会提前写 `DONE.json`。

## 日志查看和独立调试

日志位于 `runs/cloud/<run-id>/logs/`，每行是一个 JSON 对象。每类日志独立保存、独立调级别：

| 模块 | 文件后缀 | 单独诊断内容 |
|---|---|---|
| preflight | `.preflight.jsonl` | 配置、数据和 GPU 放行门 |
| data | `.data.jsonl` | 数据加载、split、hash 和 batch |
| pretrain | `.pretrain.jsonl` | 预训练 step、Loss、吞吐和 ETA |
| validation | `.validation.jsonl` | 固定验证窗口、早停和最佳指标 |
| checkpoint | `.checkpoint.jsonl` | latest/best/emergency 保存和恢复 |
| gpu | `.gpu.jsonl` | 显存、BF16、设备状态和峰值 |
| sft | `.sft.jsonl` | SFT step、assistant mask 和指标 |
| orchestrator | `.orchestrator.jsonl` | 阶段切换、墙钟和终态 |

在 `configs/cloud_a10.json` 的 `logging.module_levels` 中，可把单个模块设为 `DEBUG`、`INFO`、`WARNING`、`ERROR`、`CRITICAL` 或 `OFF`。正式运行默认全部为 `INFO`，不启用冗长 DEBUG。日志按每个模块 10 MiB 轮转，保留 5 份，避免整夜训练填满磁盘。

只看错误可运行：

```bash
rg '"level": "(ERROR|CRITICAL)"' runs/cloud/<run-id>/logs
```

只导出 checkpoint 事件可运行：

```bash
rg '"module": "cloud.checkpoint"' runs/cloud/<run-id>/logs > checkpoint-events.jsonl
```

结构化上下文会自动遮盖 password、token、authorization、secret、private key 等字段；代码也不应把原始秘密拼进普通自然语言消息。若需要分享日志，仍应先人工检查导出文件。

## Checkpoint 和恢复

`training_runtime.py` 的 checkpoint 包含：

- 模型和优化器 state dict；
- 当前 step、最佳指标、完整评估历史和 early-stopping 状态；
- CPU RNG、所有 CUDA RNG、Python RNG 和采样 generator；
- 可选 AMP scaler；
- 本次配置 SHA-256。

保存过程采用同目录临时文件、flush、fsync 和原子替换。保存后生成 checkpoint `.sha256`，立即重新加载并校验结构和 step。恢复时必须再次验证文件 SHA-256 和配置 SHA-256。

训练入口应在 `optimizer.step()` 前分别调用 Loss 和梯度有限性检查。SIGTERM 或 SIGINT 会通过 emergency hook 原子保存紧急 checkpoint 后退出。墙钟上限为 12 小时，并预留 15 分钟保存和归档；验证集连续不改善时按配置早停。

完整成功后由流水线写 `DONE.json`；任何未处理异常写 `FAILED.json`。判断任务是否完成只能看这两个终态文件和归档 manifest，不能只看进程是否消失。

## 本地验证

Mac 不需要 CUDA。以下测试使用 mock A10 覆盖成功、无 CUDA、错误卡型、不支持 BF16、显存超限、非有限数值和数据哈希失败：

```bash
python -m unittest -v tests.test_training_runtime tests.test_cloud_preflight
```

正式租卡后还必须在真实 A10 上重新运行预检和 100-step smoke，并完成“保存 → 中断 → 恢复 → 再训练 20 步”的验收，mock 测试不能替代真实 CUDA 验证。
