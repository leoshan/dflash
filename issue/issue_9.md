# [task] DFlash训练实战 | Issue 1: 训练数据集收集与分词预处理

## [Task] 核心任务目标

**执行动作**：编写预处理脚本下载并对齐投机解码所需的高质量对话数据集，并使用 `Qwen3.5-9B` 对应的 Tokenizer 进行分词预处理，最终保存为统一的 2048 长度块。

### 预期交付物
- `scripts/prepare_dataset.py`：预处理及打包脚本
- `data/qwen3.5_tokenized.jsonl`：本地预处理后的分词数据文件，包含 2048 长度的 Token IDs

### 探索维度或建议步骤
1. **数据集收集**：支持在线下载 `ShareGPT` / `UltraChat` 高质量对话数据集，或使用本地 JSON/JSONL 语料，且包含合成数据回退，确保离线及异常情况的稳定性。
2. **对话模板应用**：使用 `Qwen3.5-9B` 对应的分词模板格式化多轮对话文本。
3. **分词与打包**：使用基座模型的分词器对语料进行分词，并将所有 Token IDs 拼接成连续流，然后按 2048 的 Block Size 切分为均匀分片，溢出部分进行截断或保留。
4. **输出保存**：将结果序列化保存为 JSONL 格式（每行包含 `"input_ids"` 键），便于下一步特征提取的并发读取。

### 注意事项
- 单个 Block Size 固定为 2048 以适配后续特征提取与草稿模型预训练。
- 确保分词器加载路径为 `/home/shanchuang/nvext/models/Qwen3.5-9B`。
- 如果在离线环境下运行，应提供自动合成数据回退生成能力，确保整个端到端测试链路的可执行性。

### 硬件及执行环境约束 (GPU Configuration)
- **GPU**: 4× NVIDIA H200
- **Available VRAM**: 16 GB HBM per card (硬性上限/Hard Limit)
- **Parallelism**: Tensor Parallel (TP) = 4
- **Interconnect**: NVLink

### 📦 补充交付产物要求 (Deliverables)
- 本次任务的交付产物除了最终的数据文件外，还包含 `scripts/prepare_dataset.py` 一键预处理脚本。
