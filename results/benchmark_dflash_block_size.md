# DFlash Block Size (Draft Token Count) Sensitivity Analysis

本报告分析了投机解码中不同的 **Block Size（即 `--speculative-num-draft-tokens`，代表每次草稿模型生成的 token 数量）** 对 DFlash 推理吞吐和接受率的影响。

## 测试配置
- **模型 (Target Model)**: Qwen3.5-9B
- **草稿模型 (Draft Model)**: Qwen3.5-9B-DFlash
- **硬件环境**: 4× NVIDIA H200 (16GB VRAM limit per card), TP=4, NVLink
- **评测数据集**: GSM8K (128 prompts, concurrency=1)
- **基线吞吐量 (Baseline, No Speculation)**: 50.41 tok/s

## 评测数据对比

| Block Size | 吞吐量 (tok/s) | 延迟 (s) | 输出 Tokens 数 | 平均接受长度 (Accept Length) | 验证次数 | 相对加速比 (vs Baseline) |
|:----------:|:-------------:|:-------:|:-------------:|:----------------------------:|:--------:|:-----------------------:|
| 4 | 135.03 | 405.6 | 54768 | 3.535 | 15760 | 2.68x |
| 8 | 202.30 | 277.8 | 56192 | 5.521 | 10685 | 4.01x |
| 16 | 243.85 | 229.7 | 56018 | 6.918 | 8754 | 4.84x |
| 32 | 227.46 | 234.8 | 53413 | 6.243 | 8914 | 4.51x |

## 结果与发现
1. **加速比趋势**：
   - 随着 Block Size 从 4 增加到 16，平均接受长度增加，吞吐量和加速比显著上升。在 Block Size = 16 时，DFlash 达到了最高的加速比。
   - 当 Block Size 增加到 32 时，吞吐量可能会由于验证开销的增加或草稿接受效率的饱和而略有回落或增长放缓。
2. **分析**：
   - 块大小（Block Size）本质上是**并行生成长度**与**验证开销**之间的博弈。
   - 较小的 Block Size（如 4）限制了单次验证所能接受的最大 Token 数，导致虽然验证速度快，但未能充分利用 Target 模型的并行吞吐能力。
   - 较大的 Block Size（如 32）如果接受率不够高，多余生成的 Token 将被丢弃，且验证较长的 Block 会增加 Target 模型的自注意力计算延迟，从而降低整体效率。
