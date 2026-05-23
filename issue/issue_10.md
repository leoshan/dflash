# [task] DFlash训练实战 | Issue 2: 目标模型离线特征抽取 (TP=4)

## [Task] 核心任务目标

**执行动作**：编写特征抓取脚本，使用基座模型的前向传播，遍历已分词打包好的训练集序列，抽取指定 speculator 关联层（第 1, 8, 15, 22, 29 层）的 Hidden States，并将特征拼接后序列化保存至本地磁盘。

### 预期交付物
- `scripts/extract_features.py`：离线特征抓取及打包存储脚本
- `data/features/features_chunk_*.pt`：包含输入 Token IDs 以及拼接好 Hidden States 特征的张量序列化文件

### 探索维度或建议步骤
1. **多卡模型分片加载**：由于单卡 16GB 显存限制，需以模型并行/分布式多卡分片方式加载 `Qwen3.5-9B` 目标模型。可利用 `transformers` 的 `device_map="auto"` 将基座模型的各层平均放置于可用的 4 张 GPU 上，使单卡显存开销限制在 4.5 GB 左右。
2. **前向抓取层设置**：在前向传播中启用 `output_hidden_states=True`，并按照配置抽取对应的第 `[1, 8, 15, 22, 29]` 层的隐藏状态。
3. **特征拼接对齐**：按照 DFlash 的结构要求，将这 5 层对应相同 Token 的 Hidden States 沿最后一个特征维度（dim=-1）进行 `torch.cat` 拼接，生成 shape 为 `[seq_len, 5 * hidden_size]` (即 `[2048, 20480]`) 的单个序列特征。
4. **分块序列化保存**：为了避免单个张量文件体积过大对内存和磁盘 I/O 的瞬时冲击，支持分块（如每 10 个 sequence 打包成一个 chunk）写入 `data/features/features_chunk_*.pt` 目录中。

### 注意事项
- 执行前必须确保 `data/qwen3.5_tokenized.jsonl` 已成功生成。
- 提取特征时，对模型与中间特征使用 `torch.no_grad()` 及 CPU 内存写盘以防 GPU 发生 OOM。
- 默认的模型加载路径需设为 `/home/shanchuang/nvext/models/Qwen3.5-9B`。

### 硬件及执行环境约束 (GPU Configuration)
- **GPU**: 4× NVIDIA H200
- **Available VRAM**: 16 GB HBM per card (硬性上限/Hard Limit)
- **Parallelism**: Tensor Parallel (TP) = 4 或 device_map="auto" 层级切分
- **Interconnect**: NVLink
