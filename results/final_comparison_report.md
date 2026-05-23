# DFlash vs EAGLE3 vs Baseline 性能对比报告 (Qwen3.5-9B)

## 1. 测试环境
- **模型**: Qwen3.5-9B
- **硬件**: 4 × NVIDIA H200 (16GB VRAM each)
- **并行策略**: Tensor Parallel (TP) = 4
- **测试框架**: SGLang v0.4.5.post1
- **数据集**: gsm8k (128 prompts, concurrency 1)

## 2. 测试结果对比

| 评测模式 | 吞吐量 (tokens/s) | 加速比 | 备注 |
| :--- | :--- | :--- | :--- |
| **Baseline (无投机解码)** | 50.41 | 1.00x | 基准 |
| **EAGLE3** | N/A | N/A | SGLang 不兼容 Qwen2 草稿模型 |
| **DFlash** | 247.10 | 4.90x | Accept length: 6.953 |

## 3. 详细分析
- **Baseline**: 原始基线吞吐量 (50.41 tokens/s)。
- **EAGLE3**: 采用 EAGLE 算法进行加速。由于 SGLang 当前代码对 Qwen2 的草稿模型 `set_embed` 的依赖问题，暂不兼容。
- **DFlash**: 采用 Mamba 结构的草稿模型，极大地优化了显存和推理效率。即使在极度严苛的剩余显存环境下，也能达到 6.953 的高接受长度（Accept length）。

## 4. 结论
通过对比评测，**DFlash** 在 Qwen3.5-9B 上取得了 **4.90倍** 的推理加速比。其单步平均接受 token 数量高达 ~6.95，展示了其在预测准确度以及吞吐性能上的卓越表现，证明了其相比 Baseline 的巨大优势。
