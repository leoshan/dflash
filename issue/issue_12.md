# [task] DFlash自建草稿模型无损正确性校验

## [Task] 核心任务目标

**执行动作**：编写并运行无损正确性验证脚本，在 Greedy Decoding (Temperature = 0) 模式下，对比【目标大模型独自分回归生成】与【目标大模型 + 自建 DFlash 草稿模型联合投机解码生成】的 Token IDs 序列，确保两者在任何 Prompt 下均能输出 100% 相同的结果，保证投机解码的数学无损性。

### 预期交付物
- `scripts/verify_correctness.py`：投机解码正确性与数学无损性校验脚本
- `logs/verify_correctness_report.md`：自建草稿模型正确性校验报告

### 探索维度或建议步骤
1. **多卡混合推理部署**：
   - 目标大模型 `Qwen3.5-9B` 采用 `device_map="auto"` 分布式均匀切分至 4 张 GPU 上，控制每张卡显存占用。
   - 自建草稿模型 `qwen3.5-9b-dflash-custom` 加载并部署至主 GPU（通常为 `cuda:0`），与目标大模型在同一环境中混合计算。
2. **多卡设备兼容与数据流对齐**：
   - 确保 `extract_context_feature` 从各卡提取的指定隐藏状态能正确 move 至相同 GPU 进行拼接和前传。
   - 确保 `lm_head` 计算和投机 `sample()` 计算在不同 GPU 设备间传输时不会抛出 `device mismatch` 运行时异常。
3. **分步测试与差异打印**：
   - 针对不同类别的测试 Prompts 进行生成。
   - 逐个比对生成的 Token 序列，如有任何 Token ID 不一致，能精确指示发生 Mismatch 的位置、生成 Token 的 Decode 文本及概率分布差异，便于定位对齐问题。

### 注意事项
- 本验证为 Greedy Decoding，设置 `temperature = 0.0`。
- 启动验证前需确保 `checkpoints/qwen3.5-9b-dflash-custom/` 目录下已保存训练好的模型权重和配置文件。
- 验证运行命令：
  ```bash
  python scripts/verify_correctness.py
  ```

### 硬件及执行环境约束 (GPU Configuration)
- **GPU**: 4× NVIDIA H200 (16 GB per card)
- **Parallelism**: `device_map="auto"` 对基座模型进行层级拆分，以实现多卡承载 9B 推理
