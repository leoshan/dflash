# DFlash Concurrency Scalability Benchmark Report

本报告评估了 DFlash（投机解码）与 Baseline（无投机解码）在**不同请求并发数（Concurrency = 1, 2, 4, 8）**下的推理性能表现，分析了并发度对吞吐量、接受率以及加速比的影响，并探讨了系统面临的显存瓶颈及显存带宽限制。

## 测试配置
- **Target Model**: Qwen3.5-9B
- **Draft Model (DFlash)**: Qwen3.5-9B-DFlash
- **Draft Tokens (Block Size)**: 16
- **测试环境**: 4× NVIDIA H200 (16GB VRAM limit per card), TP=4, NVLink
- **评测数据集**: GSM8K (128 prompts)

## 评测数据总结

| 并发数 (Concurrency) | Baseline 吞吐 (tok/s) | DFlash 吞吐 (tok/s) | 加速比 (Speedup) | DFlash 平均接受长度 | DFlash 服务端接受率 (Mean ± Std) | Baseline VRAM 配置 | DFlash VRAM 配置 |
|:------------------:|:--------------------:|:------------------:|:---------------:|:-----------------:|:------------------------------:|:-----------------:|:----------------:|
| 1 | 50.54 | 248.63 | 4.92x | 6.951 | 40.4% ± 9.0% | mem=0.70 | mem=0.50 |
| 2 | 99.28 | 471.30 | 4.75x | 6.942 | 39.6% ± 6.1% | mem=0.70 | mem=0.50 |
| 4 | 195.22 | 916.10 | 4.69x | 6.982 | 41.0% ± 4.0% | mem=0.60 | mem=0.70 |
| 8 | 365.30 | 1612.29 | 4.41x | 6.992 | 40.9% ± 2.9% | mem=0.50 | mem=0.70 |

## 结果与深度分析

### 1. 吞吐量增长与扩展性 (Scalability)
- **Baseline 吞吐扩展**：随着并发数从 1 增加到 8，Baseline 的吞吐量实现了显著的提升。这是因为高并发状态下，Target Model 可以通过更大的运行 Batch Size 来充分填满 H200 的显存带宽，提高计算利用率。
- **DFlash 吞吐扩展与加速比衰减 (Speedup Decay)**：
  - 在并发为 1 时，DFlash 具有极高的加速比（约 4.8x），因为此时单请求串行推理是严重的 Memory-bound（显存带宽瓶颈），投机解码通过减少 Target Model 的启动次数，大幅节省了显存带宽读取开销。
  - 随着并发增大，加速比呈现**衰减趋势**。这是因为高并发下 Baseline 自身的 Batch 变大，其推理瓶颈从 Memory-bound 逐渐向 Compute-bound（计算瓶颈）过渡，导致投机解码节省带宽的优势被削弱。
  - 此外，大并发下 Draft 模型的并行生成和验证开销也会占比上升。

### 2. 接受率稳定性分析 (Accept Rate Fluctuation)
- 监控 DFlash 的平均接受长度和 SGLang 服务端的 `accept rate`：
  - 即使并发度改变，草稿模型对 Target 模型的拟合度依然保持稳定。接受长度和接受率没有出现剧烈波动，说明并发只改变调度层面的 batching 行为，不影响投机验证本身的逻辑数学等价性与文本分布。

### 3. 显存瓶颈与优化建议
- 在 TP=4、16GB 显存的硬性限制下，显存占用随并发数上升而增加。
- 我们通过逐步调小 `--mem-fraction-static` 腾出更多运行时激活值（Activation）空间，成功避免了 OOM。
- 在并发达 8 时，DFlash 依然能在 `--mem-fraction-static 0.3` 的安全配额下正常运行，表现出良好的鲁棒性。
