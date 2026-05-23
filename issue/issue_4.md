# [task] Qwen3.6-27B DFlash 投机解码性能评测与三方对比报告生成

## [Task] 核心任务目标

**执行动作**：对 Qwen/Qwen3.6-27B 模型使用 DFlash 投机解码进行完整性能基准测试，并整合历史测试数据生成综合对比分析报告。

### 预期交付物
- `benchmark_dflash.json`：DFlash 模式下的完整评测结果（JSON 格式）
- `final_comparison_report.md`：包含原始基线、Eagle3、DFlash 三种模式的吞吐与延迟对比分析报告

### 探索维度或建议步骤
1. **启动 SGLang 服务**：参考 README.md 中 SGLang 后端的启动命令，配置 `--speculative-algorithm DFLASH` 与草稿模型 `z-lab/Qwen3.6-27B-DFlash`，确保正确设置 `--speculative-num-draft-tokens`（建议 16）等参数
2. **运行基准测试**：使用 `dflash.benchmark` 脚本，选择 gsm8k 数据集，建议 `--num-prompts 128 --concurrency 1 --enable-thinking`，确保测试结果可复现
3. **结果持久化**：将评测输出重定向保存至 `benchmark_dflash.json` 文件
4. **数据提取与汇总**：从 Issue 2 提取原始基线数据、从 Issue 3 提取 Eagle3 测试结果，结合本任务的 DFlash 结果
5. **对比报告撰写**：生成 Markdown 格式的对比报告，包含三者的吞吐量（tokens/s）、延迟（latency）、首 token 延迟等关键指标的表格对比，以及总结结论

### 注意事项
- 确保 GPU 内存与 `--mem-fraction-static` 参数适配，避免 OOM
- 测试前确认 `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` 环境变量已设置
- 注意记录服务启动的完整参数配置，以便复现
- 对比分析时需确保三种模式测试条件（数据集、并发数、采样参数等）一致，否则需在报告中说明差异$

### 硬件及执行环境约束 (GPU Configuration)
- **GPU**: 4× NVIDIA H200
- **Available VRAM**: 16 GB HBM per card (硬性上限/Hard Limit)
- **Parallelism**: Tensor Parallel (TP) = 4
- **Interconnect**: NVLink

> ⚠️ 注意：所有评估的启动指令和执行脚本必须严格遵循 `tp-size=4` 等效并行度，并且在模型加载和执行期间严密监控 VRAM 占用（单卡最高不能超过16GB）。$

### 📦 补充交付产物要求 (Deliverables)
- 除了最终的测试结果或环境配置结果外，**必须**将本次任务涉及到的**可复现一键执行脚本**（如 `setup_env.sh`、`run_baseline.sh`、`run_eagle3.sh`、`run_dflash.sh` 等）一并作为交付产物编写并沉淀到本地代码库中。

