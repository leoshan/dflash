# [task] DFlash SGLang Spec V2 overlap scheduling performance evaluation

## [Task] 核心任务目标

**执行动作**：测试并评估 SGLang Speculative V2 中的**重叠调度（Overlap Scheduling）**机制对 DFlash 性能的提升。通过并行化 Target 验证与 Draft 序列生成，最大化推理管线的并行度。

### 预期交付物
- `results/benchmark_dflash_spec_v2.json`：开启与关闭 Overlap 调度时的性能吞吐与延迟数据。
- `results/benchmark_dflash_spec_v2.md`：分析 Overlap 调度增益与系统瓶颈的对比报告。
- `scripts/run_dflash_spec_v2_comparison.sh`：一键对比重叠调度的测试脚本。

### 探索维度或建议步骤
1. **环境变量配置对比**：
   - **No Overlap (基准)**：仅启用 DFlash 投机解码，不设置 Overlap 变量。
   - **Overlap Enabled**：配置以下环境变量：
     ```bash
     export SGLANG_ENABLE_SPEC_V2=1
     export SGLANG_ENABLE_DFLASH_SPEC_V2=1
     export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
     ```
2. **运行 benchmark**：在两种配置下启动 SGLang 并运行 gsm8k 评测（128 prompts，concurrency 1）。
3. **深度分析**：
   - 对比吞吐量和首 token 延迟 (TTFT)。
   - 检查 SGLang 服务的运行日志，验证其稳定性，确认是否有死锁或算子 crash 现象。

### 注意事项
- Overlap 功能目前在 SGLang 中属于高度实验性（experimental）特性，需特别注意稳定性。
- 确保测试环境的 NCCL/CUDA 相关库与 SGLang v0.4.5.post1 的重叠调度需求匹配。

### 硬件及执行环境约束 (GPU Configuration)
- **GPU**: 4× NVIDIA H200
- **Available VRAM**: 16 GB HBM per card (硬性上限/Hard Limit)
- **Parallelism**: Tensor Parallel (TP) = 4
- **Interconnect**: NVLink

### 📦 补充交付产物要求 (Deliverables)
- 除了最终的测试结果或环境配置结果外，**必须**将本次任务涉及到的**可复现一键执行脚本**一并作为交付产物编写并沉淀到本地代码库中。
