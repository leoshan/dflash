# [task] 自研 DFlash vs. 第三方模型对比评测

## [Task] 核心任务目标

**执行动作**：在 GSM8K 等 benchmark 上，对比自研模型 `qwen3.5-9b-dflash-custom` 与官方第三方模型 `z-lab/Qwen3.5-9B-DFlash` 在实际服务（SGLang）中的端到端推理性能和接受率指标。

### 预期交付物
- `scripts/compare_dflash_models.py`：对比两款草稿模型（自研 vs 官方第三方）性能与接受率的自动化测试脚本。
- `results/custom_vs_thirdparty_dflash_report.md`：详细的对比测试报告，分析其加速比、吞吐量、平均接受长度及接受率的分布和直方图。

### 探索维度或建议步骤
1. **评估对比脚本编写**：
   - 编写 `scripts/compare_dflash_models.py`，能够分别拉起加载了第三方模型 `z-lab/Qwen3.5-9B-DFlash` 和自研模型 `checkpoints/qwen3.5-9b-dflash-custom` 的 SGLang 服务。
   - 对两款模型运行相同的评估测试集（如 GSM8K，128个 prompts，concurrency=1）。
   - 从 SGLang 服务的输出和 `dflash.benchmark` 日志中提取 `accept rate`、`accept_length`（即 $\tau$）以及端到端 `throughput (tok/s)`。
2. **比较指标提取**：
   - 对比两者在相同 Block Size（例如 16）下的平均接受长度 $\tau$。
   - 记录两者在服务端的接受率均值与方差。
   - 导出吞吐量（Tokens/s）并计算加速比。
3. **数据报告与分析**：
   - 整理两者在同等测试环境下的数据。
   - 编写对比报告 `results/custom_vs_thirdparty_dflash_report.md`。

### 注意事项
- 测试时，确保 SGLang 的静态显存占比 `--mem-fraction-static` 参数已合理设定（推荐设为 `0.5`），以防 OOM。
- 启动指令应能完全清理上一次测试残留的服务进程和绑定的端口（如 `30000` 端口）。

### 硬件及执行环境约束 (GPU Configuration)
- **GPU**: 4× NVIDIA H200 (16 GB per card)
- **Parallelism**: TP=4, NVLink
- **Framework**: SGLang v0.4.5.post1
