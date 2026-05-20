# [task] DFlash behavior under long contexts and sliding window configuration

## [Task] 核心任务目标

**执行动作**：探索 DFlash 在**长文本（Long Context）**场景下的性能，并测试草稿模型滑动窗口机制（`--draft-sliding-window-size`）对显存和接受效率的优化效果。

### 预期交付物
- `results/benchmark_dflash_long_context.json`：长文本评测下的显存、吞吐和接受率对比数据。
- `results/benchmark_dflash_long_context.md`：分析滑动窗口大小对生成质量和推理效率影响的技术报告。
- `scripts/run_dflash_long_context.sh`：长文本环境下的测试运行脚本。

### 探索维度或建议步骤
1. **构造长文本测试集**：选择或合成 4k, 8k, 16k 长度的 Prompt（例如利用长文档问答或重复拼接数据）。
2. **对比实验设计**：
   - 配置 1: 无滑动窗口（`draft-sliding-window-size` 为 None）。
   - 配置 2: 滑动窗口大小为 2048。
   - 配置 3: 滑动窗口大小为 4096。
3. **监控指标**：
   - 监控随着上下文增长，KV cache 显存的增长速率。
   - 观察滑动窗口对 DFlash 草稿模型接受长度（Accept length）是否有负面影响。
   - 测量 Time-to-First-Token (TTFT) 和 Inter-Token Latency (ITL)。

### 注意事项
- 在 16GB 的 VRAM 硬性上限下，长文本非常容易 OOM，必须谨慎配置 `--mem-fraction-static` 以及最大 Token 数。
- 观察 Sliding Window 机制在超长 prompt 中是否能有效防范显存崩溃。

### 硬件及执行环境约束 (GPU Configuration)
- **GPU**: 4× NVIDIA H200
- **Available VRAM**: 16 GB HBM per card (硬性上限/Hard Limit)
- **Parallelism**: Tensor Parallel (TP) = 4
- **Interconnect**: NVLink

### 📦 补充交付产物要求 (Deliverables)
- 除了最终的测试结果或环境配置结果外，**必须**将本次任务涉及到的**可复现一键执行脚本**一并作为交付产物编写并沉淀到本地代码库中。
