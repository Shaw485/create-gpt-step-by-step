# M012 SHA-256 校验清单

本清单用于后续复盘和视频制作时确认材料未被意外覆盖。训练JSONL、张量和checkpoint保存在本机忽略目录中；公开仓库只保留脚本、测试、文档、指标和图表。

| SHA-256 | 文件 |
|---|---|
| `1a8c03b1cf216dfcfdba956c984082b99a4f505a8a785f29792a98edd3442071` | `.gitignore` |
| `7de85cb9d22980b92e5cf64f51b9a349bb61ba4e36c74c4c86704cbf40fc2960` | `build_sft_v5_repair.py` |
| `2d4ae81481f10dc2357a1d6eb53533ca075cfd457a5353642e07d9a5d0458232` | `sample_sft_v4_custom.py` |
| `22a884ca7d343d9f3156128758b1700863f9fa3ccf234bc86c2ae46e910ddddb` | `tests/test_build_sft_v5_repair.py` |
| `45f8f9b1cb1092883c9a3f72a64e9199bdee5fc9658f52ec5d220bf547191d4f` | `reports/milestones/012_v5_repair_sft/README.md` |
| `b128606b69c259ba59ac01cd95f40fd851aeb86d4243f94d35d39f0d97347715` | `reports/milestones/012_v5_repair_sft/data_report.json` |
| `869ac9e831d97c5511c30eab127af8ef02d6165a2796417418870d55142a9a44` | `reports/milestones/012_v5_repair_sft/tensor_report.json` |
| `8bb3e68105ba9517cc26faac7ae56a6bba0d10a0dc4d1ada3b30352eec6ebd90` | `reports/milestones/012_v5_repair_sft/sft_v5_repair_smoke100_report.json` |
| `f3b06907e524f777a411c52b808e01dcef72c21f26bab2e6323a2c884fb9b77a` | `reports/milestones/012_v5_repair_sft/sft_v5_repair_smoke100_loss_curve.png` |
| `35607f4064d30e82e2c84abb96e0b662e2d53e00cdf080af52d1e049a15fce6c` | `reports/milestones/012_v5_repair_sft/sft_v5_repair_smoke100_loss_curve.svg` |
| `1fc82bf0e3d110e79e4b94603b4c3ac217cad37982d280d3f517921a5b9164b4` | `reports/milestones/012_v5_repair_sft/category_eval_smoke100_latest_lowtemp.json` |
| `97d4420dbb264bff2903089e7fcced36afdb8f2ce836c1bbd9d53894a22ba80f` | `reports/milestones/012_v5_repair_sft/category_eval_smoke100_latest_lowtemp.md` |
| `c388b51f2bfd0def1700279f0032b240671c4e8e8cb36021387acddfbc48ba7b` | `reports/milestones/012_v5_repair_sft/novel_vs_general_smoke100_samples.json` |
| `f80f96e37d852150d67a1d16c6e347ee80e42f5e483f398e457450e653ff3656` | `reports/milestones/012_v5_repair_sft/novel_vs_general_smoke100_samples.md` |

本机忽略文件的关键哈希：

| SHA-256 | 文件 |
|---|---|
| `5baa071aa180e6afa473badb2258661daca2cae74cce45c21ae0e5275a8f1c40` | `data/sft/v5_repair/sft_v5_repair_training_ready.jsonl` |
| `9ba003074f133cfd3c1d343211e2d095875fcae44e3edc9b0ab9c84eb050dd0b` | `data/cloud_v4/sft_v5_repair_tensors.pt` |
| `5d6397b3bb97b8a14369117e2f1e1e9f9addf515da1117e2f0f35dfe8ac8af44` | `runs/pretrain_v4_m4_continue6000/best.pt` |
| `438e27477f653e82ad28d13002f746fef1c152dd1257e94de5210dd0bf0c0d72` | `runs/sft_v5_repair_smoke100/best.pt` |
| `8c93cd17d34c78548131743aaedfdd4fe2430a4e33f0469b33ea5af675a06c2b` | `runs/sft_v5_repair_smoke100/latest.pt` |
