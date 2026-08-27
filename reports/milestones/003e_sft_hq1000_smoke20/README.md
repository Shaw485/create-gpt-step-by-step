# M003e：1000条数据的SFT 20步安全试跑

## 目的

从统一的 `sft_stage1_init_pre_sft.pt` 重新开始，确认新的1000条数据能够完成加载、Loss计算、反向传播、梯度裁剪、参数更新、验证和checkpoint保存。该模型只用于安全检查，不作为最终模型。

## 配置

- 训练/验证/测试：800/100/100；测试集没有参与。
- Step：20；Batch Size：4；约0.1个epoch。
- 学习率：`1e-4`；Weight Decay：`0.01`；梯度裁剪：`1.0`。
- CPU训练；随机种子42；每5步完整评估训练集和验证集。
- 参数数量：7,099,779。

## 结果

| Step | 训练 Loss | 验证 Loss |
|---:|---:|---:|
| 0 | 5.1175 | 5.2546 |
| 5 | 3.8226 | 4.0087 |
| 10 | 2.9052 | 3.1619 |
| 15 | 2.2461 | 2.5476 |
| 20 | 1.7259 | 2.0471 |

所有检查点均有限且可重新加载，两个监控回答都主动生成 `<EOS>`。训练闭环通过，因此进入800步正式实验。

## 文件

- 报告：`sft_hq1000_smoke20_report.json`
- 最终安全检查点：`checkpoints/sft_hq1000_smoke20.pt`
- 最佳安全检查点：`checkpoints/sft_hq1000_smoke20_best.pt`

## 日志

数据、训练步骤、验证和检查点分别写入 `logs/sft_train_data.log`、`logs/sft_train_step.log`、`logs/sft_train_validation.log` 和 `logs/sft_train_checkpoint.log`。各模块使用 `SFT_DATA_LOG_LEVEL`、`SFT_TRAIN_LOG_LEVEL`、`SFT_VALIDATION_LOG_LEVEL` 和 `SFT_CHECKPOINT_LOG_LEVEL` 独立调节；日志轮转上限1 MB，保留3个备份。
