# M019 SHA-256

| 文件 | SHA-256 |
|---|---|
| `README.md` | `bfd4c5eebab306b22a2f1195b6b181adece2c09abd2f7e37ef5a0b8d1a978893` |
| `comparison.json` | `94551b1e72a1937c67c0d36d7b82ee21119340190e791249c9755e03fbe6df89` |
| `comparison.csv` | `42a91c19bd028f9425272065aa8851c7017fc182206e1c78ff3e628067183c07` |
| `comparison.md` | `c57ecda0eb0485b088e5953e08a1308c8e0dd5ed2ef1eebbd30273d520060dfd` |
| `final_verdict.json` | `e10be5c56f90cc790d3af742e613e1f630037da438cd8895d30d5423d4a13113` |
| `ai_manual_review.csv` | `df85dc8821fd55d61fc0b75878aa126517aeb07c671b3f359f62b8634bfef9d8` |
| `ai_manual_review.md` | `125b648c5e93fea47660a0e4a910ddf007b3edba80687edec98c697f640ce3d2` |
| `step_00250/audit.json` | `722056ecd754b0bcf90b9fdbce9f84b4bdd20d3c159a139f15c55d4e07c9704b` |
| `step_00250/audit.md` | `9dbe37003676c377de96e07567a508c2fbd480541d8aca3bfaa289f95142a41e` |
| `step_05750/audit.json` | `274d6f8b2c32ed5e54987ff1ee8586b2e38fa1837b71fc25cf84ca2cfc02ab62` |
| `step_05750/audit.md` | `a30f89cbaef46344d3743cd5c28d3e5dffe12bc92ddb5cf204131b59adb0c547` |
| `step_06000/audit.json` | `2ac053faafc3c94de7fdf44989f31dbcde3859e978b4d098f09644bcd0b87c5d` |
| `step_06000/audit.md` | `c5ecf5f760b5c4b5956621b506bef8a963404d90d814ea517b2ce87038d4ef81` |
| `data/eval/pretrain_capability_probes.json` | `f95d594eaa8d08ef340a704c7b9103627ab949e88c1796fe3db0527cfc54f36e` |
| `data/eval/pretrain_capability_prompts.txt` | `032fd93f12ffba3df3d4d9ad4bc1898a6a86ec416a812cd85a0c0f1928a253a7` |
| `docs/pretrain_capability_audit_protocol.md` | `6544867cd76ab093c70c417aceea3e43cd71521a975127372d8e68ce6d3fa92c` |
| `build_pretrain_capability_probes.py` | `38aa435c77335be6f4b3e72e2708c1e26e15a75faf4bb9c8461847c630d67f0c` |
| `evaluate_pretrain_capabilities.py` | `b74a20448ff32c85bd52d3594689924d0b7ecb939e1cc62a5e337de8d9ccd34c` |
| `summarize_pretrain_capability_audit.py` | `fc7273099803a33335650421876020095dd99407903ea4da90ffbf2e413b25e7` |
| `runs/formal_pretrain_14m_bpe3000_6000/checkpoints/step_00250.pt` | `20fa8bb335d8e05716e4773152bec0c70f15a1291ce7752b2fa54215fd171358` |
| `runs/formal_pretrain_14m_bpe3000_6000/checkpoints/step_05750.pt` | `bfe4fec5e6045d4c06d22393e7c2079fdc03897be71829c9d9dcbaf0fcaf5c1e` |
| `runs/formal_pretrain_14m_bpe3000_6000/checkpoints/step_06000.pt` | `bfc05c0c045e473316d0f44a3e766570ac7325ab42f8d10570106bc2c9daaab5` |

Checkpoint 文件仍由 `.gitignore` 排除，只保存在本机；上表记录的是本轮实际加载文件的摘要。日志不纳入固定摘要，因为它们是可轮转诊断材料，正式输入与结果摘要已经写入各审计 JSON。
