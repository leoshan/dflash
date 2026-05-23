# [task] SGLang EAGLE3 投机解码性能测试 - Qwen3.6-27B

## [Task] 核心任务目标

**执行动作**：在已配置的 SGLang 环境中，使用 EAGLE3 算法对 Qwen/Qwen3.6-27B 模型进行投机解码性能测试。具体包括：配置 SGLang 启动参数（speculative algorithm 设为 EAGLE，挂载对应的 eagle3 草稿模型），运行 dflash.benchmark 测试脚本（gsm8k 数据集，128 prompts，并发 1），并将性能测试结果保存为 benchmark_eagle3.json 文件。

### 预期交付物
- `benchmark_eagle3.json`：包含吞吐量、延迟、生成成功率等关键性能指标的 JSON 格式测试报告
- 单次执行的简要日志记录（启动参数、测试进度、关键输出）

### 探索维度或建议步骤

1. **确认 EAGLE3 草稿模型**：查找适用于 Qwen3.6-27B 的 EAGLE3 草稿模型（可能位于 HuggingFace 或本地模型库），确认模型路径和兼容性
2. **配置 SGLang 启动参数**：参考 DFlash 的 SGLang 配置，调整以下参数：
   - `--speculative-algorithm EAGLE`
   - `--speculative-draft-model-path <eagle3_draft_model_path>`
   - `--speculative-num-draft-tokens` 合理值（建议 8-16）
   - 其他必要参数：`--tp-size`, `--attention-backend`, `--mem-fraction-static` 等
3. **启动 SGLang 服务**：确保服务正常启动并监听指定端口（如 30000）
4. **运行 benchmark 测试**：
   ```bash
   python -m dflash.benchmark --backend sglang \
       --base-url http://127.0.0.1:30000 --model Qwen/Qwen3.6-27B \
       --dataset gsm8k --num-prompts 128 --concurrency 1 --enable-thinking
   ```
5. **收集与保存结果**：将测试输出解析为 JSON 格式，保存至 `benchmark_eagle3.json`

### 注意事项

- EAGLE3 与 DFlash 是不同的投机解码算法，需确认 SGLang 版本是否支持 EAGLE speculative algorithm
- 注意草稿模型与目标模型的版本兼容性（Qwen3.6 vs Qwen3.5）
- 测试环境需保证充足的 GPU 显存（草稿模型 + 目标模型同时加载）
- 记录完整的启动命令和测试命令，便于复现
- 若 EAGLE3 模型不可用，需在日志中明确说明并记录替代方案或错误信息
- 可参考 `/home/shanchuang/dflash/` 目录下的配置和脚本$

### 硬件及执行环境约束 (GPU Configuration)
- **GPU**: 4× NVIDIA H200
- **Available VRAM**: 16 GB HBM per card (硬性上限/Hard Limit)
- **Parallelism**: Tensor Parallel (TP) = 4
- **Interconnect**: NVLink

> ⚠️ 注意：所有评估的启动指令和执行脚本必须严格遵循 `tp-size=4` 等效并行度，并且在模型加载和执行期间严密监控 VRAM 占用（单卡最高不能超过16GB）。$

### 📦 补充交付产物要求 (Deliverables)
- 除了最终的测试结果或环境配置结果外，**必须**将本次任务涉及到的**可复现一键执行脚本**（如 `setup_env.sh`、`run_baseline.sh`、`run_eagle3.sh`、`run_dflash.sh` 等）一并作为交付产物编写并沉淀到本地代码库中。

