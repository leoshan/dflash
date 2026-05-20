# [task] DFlash block size (draft token count) sensitivity analysis

## [Task] 核心任务目标

**执行动作**：在 Qwen3.5-9B 上对 DFlash 的投机解码性能进行 **Block Size（即 `--speculative-num-draft-tokens`）的敏感度分析**。测试不同的 Block Size 设置对吞吐量、平均接受长度和系统开销的影响。

### 预期交付物
- `results/benchmark_dflash_block_size.json`：包含不同 Block Size（4, 8, 16, 32）下的测试吞吐量、平均接受长度等指标的 JSON 数据。
- `results/benchmark_dflash_block_size.md`：详细的 Block Size 敏感度分析报告与折线图/数据表格。
- `scripts/run_dflash_block_size_sweep.sh`：一键自动扫参并收集结果的 Bash 脚本。

### 探索维度或建议步骤
1. **编写扫参脚本**：编写 `scripts/run_dflash_block_size_sweep.sh`，在 SGLang 服务启动时依次将 `--speculative-num-draft-tokens` 设置为 `4`、`8`、`16`、`32`。
2. **控制变量**：保持其他参数一致（`--tp-size 4`、`--mem-fraction-static 0.5`、`--max-running-requests 1`、`--port 30000`）。
3. **运行评测**：使用 `dflash.benchmark` 评测 gsm8k 数据集（128 prompts，concurrency 1）。
4. **性能分析**：
   - 提取吞吐量 (tokens/s) 与平均接受长度 (Accept length)。
   - 分析随着 Block Size 变大，平均接受长度是否饱和，以及验证开销（Target Model 的 Forward Pass 延迟）是否增加导致吞吐下降。

### 注意事项
- 服务重启前需彻底清理显存，避免 OOM。
- 保证测试期间无其他显卡负载干扰。
- 注意观察 `--speculative-num-draft-tokens` 大于 16 时是否会导致显存超限（16GB limit）。

### 硬件及执行环境约束 (GPU Configuration)
- **GPU**: 4× NVIDIA H200
- **Available VRAM**: 16 GB HBM per card (硬性上限/Hard Limit)
- **Parallelism**: Tensor Parallel (TP) = 4
- **Interconnect**: NVLink

### 📦 补充交付产物要求 (Deliverables)
- 除了最终的测试结果或环境配置结果外，**必须**将本次任务涉及到的**可复现一键执行脚本**一并作为交付产物编写并沉淀到本地代码库中。
