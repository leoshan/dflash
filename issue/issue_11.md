# [task] DFlash草稿模型构建与分片训练 (FSDP)

## [Task] 核心任务目标

**执行动作**：构建 DFlash 自建草稿模型（命名隔离为 `qwen3.5-9b-dflash-custom`），编写训练脚本，加载上一步录入的特征张量数据，配置单卡或多卡 Fully Sharded Data Parallel (FSDP) 训练逻辑；并使用 **前置 Token 权重衰减 (Early Loss Decay)** 和 **渐进式 Block Curriculum 学习**，将最终的权重保存至本地磁盘。

### 预期交付物
- `scripts/train_dflash.py`：草稿模型预训练脚本
- `checkpoints/qwen3.5-9b-dflash-custom/`：已训练完成保存的自定义草稿模型目录，包含 `model.safetensors` 或 `pytorch_model.bin`、`config.json` 以及分词配置文件

### 探索维度或建议步骤
1. **网络加载与参数隔离**：通过 `z-lab/Qwen3.5-9B-DFlash` 的配置模板加载自定义的 `DFlashDraftModel`，保存路径和配置标识设为 `qwen3.5-9b-dflash-custom`，防止命名冲突。
2. **轻量化特征训练与权重拷贝**：为了节省显存，训练期间完全不载入 9B 基座模型。在初始化时，仅从目标模型首块权重 `.safetensors` 中安全加载映射层 `embed_tokens.weight` 和预测层 `lm_head.weight`，以 frozen 状态放置于 GPU，避免多余显存占用。
3. **分片策略封装 (FSDP / ZeRO-3)**：脚本需检测运行状态，如果使用多卡 `torchrun` 启动，使用 `FullyShardedDataParallel` 对模型层及优化器状态进行分片，并开启梯度检查点 (Gradient Checkpointing)，将单卡训练总显存控制在 8.5 GB 以内。
4. **两项自研对齐优化算法实现**：
   - **Early Loss Decay**：损失计算时，对同一个 block（16个 tokens）内的不同位置 $d \in [0, 15]$ 分配衰减权重 $\gamma^d$，前置 token 赋予更高权重，降低后期被拒绝 token 的权重。
   - **Block Curriculum**：设置渐进式训练难度，课程学习的预测长度从 2 逐渐增长到最终的 16，提升收敛效率和精度。

### 注意事项
- 启动前确保 `data/features/` 路径下已存在提取好的特征文件。
- 如果在单卡调试，可直接通过 `python scripts/train_dflash.py` 运行；若在多卡执行，使用 `torchrun --nproc_per_node=4 scripts/train_dflash.py` 进行分布式分片训练。

### 硬件及执行环境约束 (GPU Configuration)
- **GPU**: 4× NVIDIA H200
- **Available VRAM**: 16 GB HBM per card (硬性上限/Hard Limit)
- **Parallelism**: FSDP (ZeRO-3) / Single GPU
- **Interconnect**: NVLink
