# [task] DFlash scalability under varying request concurrency

## [Task] 核心任务目标

**执行动作**：测试并评估 DFlash 在**不同请求并发数（Concurrency）**下的性能表现。探索投机解码在并发量增大时的加速衰减点（speculative speedup decay）。

### 预期交付物
- `results/benchmark_dflash_concurrency.json`：包含 Baseline 和 DFlash 在并发数为 1, 2, 4, 8 时的吞吐和加速比数据。
- `results/benchmark_dflash_concurrency.md`：分析并发数如何影响 DFlash 验证效率的分析报告。
- `scripts/run_dflash_concurrency_sweep.sh`：自动运行不同并发并记录结果的脚本。

### 探索维度或建议步骤
1. **并发扫参设计**：针对 Baseline（无投机解码）和 DFlash 分别扫参 `concurrency` 为 `1`、`2`、`4`、`8`。
2. **运行 benchmark**：运行 `dflash.benchmark` 对 gsm8k 进行评测。
3. **关键数据监控**：
   - 吞吐量 (tokens/s) 增长曲线。
   - 记录 SGLang 服务端输出日志中的 `accept rate` (接受率) 在并发增加时是否发生波动。
   - 验证随着 batch size 变大，投机解码是否因主模型显存带宽瓶颈（Memory-bound 减弱）而使得加速比减小。

### 注意事项
- 并发增大时显存占用会上升，务必严密监控显存，调整 `--mem-fraction-static` 参数（如从 0.5 调整至 0.4 或 0.3）防止 OOM。
- 需要在 TP=4 的 16GB H200 限制下寻找安全的最大并发数。

### 硬件及执行环境约束 (GPU Configuration)
- **GPU**: 4× NVIDIA H200
- **Available VRAM**: 16 GB HBM per card (硬性上限/Hard Limit)
- **Parallelism**: Tensor Parallel (TP) = 4
- **Interconnect**: NVLink

### 📦 补充交付产物要求 (Deliverables)
- 除了最终的测试结果或环境配置结果外，**必须**将本次任务涉及到的**可复现一键执行脚本**一并作为交付产物编写并沉淀到本地代码库中。
