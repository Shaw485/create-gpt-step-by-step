# M011 SHA-256 校验清单

本清单用于后续复盘和视频制作时确认材料未被意外覆盖。模型 checkpoint 和训练张量保存在本机忽略目录中，不随 GitHub 提交上传；公开仓库只保留脚本、测试、文档、指标和图表。

| SHA-256 | 文件 |
|---|---|
| `e9dc68772b334db31c4c9c8834fb6b714bebe1a3e96e78b33c729793d9832542` | `.gitignore` |
| `ba88f4c8ab59923197924c787887ae17d8e7bdfb604beac8d0121e84f79f1e20` | `build_sft_v4_mixed_chat.py` |
| `f4ca0963e52a29c5e0c8b1dcecd84c666a8676abb5b9397e1394c4a50db37fb8` | `tests/test_build_sft_v4_mixed_chat.py` |
| `ac69c1b077f2c8d0fcb8adbf834bbe44c43f0b42b7c6c12b94c0d928c9e204ed` | `reports/milestones/011_v4_mixed_chat_sft/README.md` |
| `35922d2f5e8d834933f1a856eed4b45e06a6427f326c45e9d282fa01360ac5f5` | `reports/milestones/011_v4_mixed_chat_sft/data_report.json` |
| `d447fdca00c8e3da6c3e8481e744aa98b62935ed332409ebde1f5bab12f8f126` | `reports/milestones/011_v4_mixed_chat_sft/tensor_report.json` |
| `08235d220175f4946590805ae6fafb373f2a34c71125873bbddea4c778e06534` | `reports/milestones/011_v4_mixed_chat_sft/sft_v4_mixed_chat_smoke100_report.json` |
| `2abd1b105901a47d4961729be78a4f518ba8ef7054039942954012964cc1f51c` | `reports/milestones/011_v4_mixed_chat_sft/sft_v4_mixed_chat_smoke100_report_loss.csv` |
| `137bb9134a067331f3f6d80e1293df501dbe472f6578b6ef1eadb113c0e4cb3b` | `reports/milestones/011_v4_mixed_chat_sft/sft_v4_mixed_chat_smoke100_report_loss_curve.png` |
| `fd9cd70d89e6b8378ef3d91dec484e3574b7665fb73b44d6f39c4e071320d9a4` | `reports/milestones/011_v4_mixed_chat_sft/sft_v4_mixed_chat_smoke100_report_loss_curve.svg` |
| `8b4130d7da6c5d16426e92a3801eca07fe7ac735ae1a869ffd7fc2bbfc5ee5ef` | `reports/milestones/011_v4_mixed_chat_sft/novel_vs_general_smoke100_samples.json` |
| `0f5fe73b756155fcb70214bcd1dcb6154ce31b25191a92af36d80e0681d0179b` | `reports/milestones/011_v4_mixed_chat_sft/sft_v4_mixed_chat_step500_report.json` |
| `339a8c7d345641a59f5c412e06b98d379ca03c3c47f920568cadfaaf3d53cd1c` | `reports/milestones/011_v4_mixed_chat_sft/sft_v4_mixed_chat_step500_report_loss.csv` |
| `5e96b1f38c2b356da02ddcd267e19f1d71984219efe5f54444a6bef35ab6e1e7` | `reports/milestones/011_v4_mixed_chat_sft/sft_v4_mixed_chat_step500_report_loss_curve.png` |
| `3f353cf94f0b0312f507bf7d5c1d5f553a69bc892593764000507542f89963de` | `reports/milestones/011_v4_mixed_chat_sft/sft_v4_mixed_chat_step500_report_loss_curve.svg` |
| `13dac11726e2ed9c61ace839d7fd2c211cf4213cd3455a38e16c09407c5925ee` | `reports/milestones/011_v4_mixed_chat_sft/novel_vs_general_step500_samples.json` |
| `c8ce97a25620672e2ef810a17a15da6c3ae84776365b8e1783881f8cb29d15e1` | `reports/milestones/011_v4_mixed_chat_sft/novel_vs_general_step500_samples.md` |
| `9ce9e7baae03fe58a4ccde03fdbcb0517321d94a0444cd4cfd9276508047f967` | `reports/milestones/011_v4_mixed_chat_sft/sft_v4_mixed_chat_step2000_report.json` |
| `9c215addf1f3f10a40fd2003932e2552bcd587c9baa27d1a0c0e0b609e7a58df` | `reports/milestones/011_v4_mixed_chat_sft/sft_v4_mixed_chat_step2000_report_loss.csv` |
| `889f721e6f948b31fdb0818aa04c007d17bf4ff7f5d4140c0ac290cc0ddbc7af` | `reports/milestones/011_v4_mixed_chat_sft/sft_v4_mixed_chat_step2000_report_loss_curve.png` |
| `aacb669fa790a275559112204640de4d74e0e7113cd125f3a03b0dad6376ae57` | `reports/milestones/011_v4_mixed_chat_sft/sft_v4_mixed_chat_step2000_report_loss_curve.svg` |
| `50736d5807e3539c9c74afaf37a73f6079b02ab90d087c15e266449319017d62` | `reports/milestones/011_v4_mixed_chat_sft/novel_vs_general_step2000_samples.json` |
| `c7c679b361668ff11324fe56f0df4b07f8ca57b9a183df2966b2334b650ad375` | `reports/milestones/011_v4_mixed_chat_sft/novel_vs_general_step2000_samples.md` |

本机忽略文件的关键哈希：

| SHA-256 | 文件 |
|---|---|
| `50e38ec19541ecd35ea6da64c25ce32ccd655acd74013eb9a380071c07133a95` | `data/sft/v4_mixed_chat/sft_v4_mixed_chat_training_ready.jsonl` |
| `73e46a1a3031e6e3d422d10922520c4759bc18a324f8c2fd35524c689f622573` | `data/cloud_v4/sft_v4_mixed_chat_tensors.pt` |
| `5d6397b3bb97b8a14369117e2f1e1e9f9addf515da1117e2f0f35dfe8ac8af44` | `runs/pretrain_v4_m4_continue6000/best.pt` |
| `bb2d77f543173f7994000c2c3fafefb80d4334fc080c7d9ad7a009ea9f39c1e7` | `runs/sft_v4_mixed_chat_step2000/latest.pt` |
