# [task] Qwen3.6-27B基座模型SGLang基准性能测试

## [Task] 核心任务目标

**执行动作**：在已配置的SGLang环境中，对原始基座模型 Qwen/Qwen3.6-27B（不启用任何投机解码/Draft模型）执行基准性能测试。启动SGLang服务后，运行 dflash.benchmark 测试脚本，记录吞吐量（Tokens per second）和请求延迟数据。

### 预期交付物
- `benchmark_original.json`：包含测试结果的JSON文件，记录吞吐量和延迟数据
- `benchmark_original.md`：对应的Markdown格式测试日志

### 探索维度或建议步骤
1. 启动SGLang服务，加载 Qwen/Qwen3.6-27B 模型（注意：不指定 `--speculative-algorithm` 和 `--speculative-draft-model-path` 参数，确保无投机解码）
2. 参考 README.md 中的 SGLang benchmark 命令格式，运行 dflash.benchmark：
   - `--backend sglang`
   - `--model Qwen/Qwen3.6-27B`
   - `--dataset gsm8k`
   - `--num-prompts 128`
   - `--concurrency 1`
   - 根据模型特性决定是否添加 `--enable-thinking` 参数
3. 捕获并解析测试输出，提取关键性能指标：
   - Tokens per second（吞吐量）
   - 请求延迟（首token延迟TTFT、端到端延迟E2E Latency等）
4. 将测试结果结构化保存为 benchmark_original.json
5. 生成对应的 benchmark_original.md 测试日志，包含测试环境、配置参数、结果摘要

### 注意事项
- 确保SGLang服务在测试前已正确启动并处于健康状态
- 测试前确认GPU显存充足，可能需要调整 `--mem-fraction-static` 参数
- 注意 Qwen3.6 系列模型的 special token 处理和 thinking mode 配置
- 保存完整的终端输出日志以便后续分析
- 与后续使用 DFlash 投机解码的性能数据进行对比时，需确保测试条件一致（数据集、并发数、prompt数量等）$

### 硬件及执行环境约束 (GPU Configuration)
- **GPU**: 4× NVIDIA H200
- **Available VRAM**: 16 GB HBM per card (硬性上限/Hard Limit)
- **Parallelism**: Tensor Parallel (TP) = 4
- **Interconnect**: NVLink

> ⚠️ 注意：所有评估的启动指令和执行脚本必须严格遵循 `tp-size=4` 等效并行度，并且在模型加载和执行期间严密监控 VRAM 占用（单卡最高不能超过16GB）。$

### 📦 补充交付产物要求 (Deliverables)
- 除了最终的测试结果或环境配置结果外，**必须**将本次任务涉及到的**可复现一键执行脚本**（如 `setup_env.sh`、`run_baseline.sh`、`run_eagle3.sh`、`run_dflash.sh` 等）一并作为交付产物编写并沉淀到本地代码库中。

