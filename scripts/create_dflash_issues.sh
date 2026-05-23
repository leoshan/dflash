#!/bin/bash
# scripts/create_dflash_issues.sh
# 使用 GitHub CLI 自动化批量创建 DFlash 训练与评测架构的 Issues

set -e

# 确保 gh 已经登录
if ! gh auth status >/dev/null 2>&1; then
    echo "Error: GitHub CLI (gh) 未登录，请先执行 'gh auth login' 登录。"
    exit 1
fi

echo "开始创建 DFlash 自研模型训练与评测相关 Issues..."

# Issue 1
gh issue create \
  --title "DFlash训练实战 | Issue 1: 训练数据集收集与分词预处理" \
  --body "### 目标
下载并处理投机解码对齐所需的高质量数据集（如 ShareGPT/UltraChat），完成 Qwen3.5-9B 分词预处理。

### 任务细分
1. 编写 \`scripts/prepare_dataset.py\` 脚本下载数据集。
2. 对语料进行截断和拼接，处理成统一的 2048 长度块。
3. 利用 Qwen3.5-9B 的分词器将数据 Tokenize，转存为便于特征提取的格式。

### 交付件
- \`scripts/prepare_dataset.py\` 预处理代码
- 本地预处理后的 Token 数据文件"

# Issue 2
gh issue create \
  --title "DFlash训练实战 | Issue 2: 目标模型离线特征抽取 (TP=4)" \
  --body "### 目标
由于单卡 16GB 显存限制，采用离线解耦抽取特征。部署 Qwen3.5-9B (TP=4) 前向传播抓取隐藏状态。

### 任务细分
1. 编写 \`scripts/extract_features.py\`，利用 vLLM 加载 Qwen3.5-9B (TP=4) 搭建 Serving 服务。
2. 输入预处理好的 Token，通过前向传播抓取采样层（如 1, 8, 16, 24, 31 层）的 Hidden States。
3. 将特征张量与其对应的 Token IDs 配对，序列化保存到本地 NVMe 硬盘中。

### 交付件
- \`scripts/extract_features.py\` 抽取代码
- 保存到 SSD 的离线特征数据"

# Issue 3
gh issue create \
  --title "DFlash训练实战 | Issue 3: DFlash草稿模型构建与分片训练 (FSDP)" \
  --body "### 目标
不加载目标模型以节省显存。利用离线特征训练 1.5B 的自建草稿模型，控制显存低于 16GB。

### 任务细分
1. 构建自研草稿模型结构，将其与第三方命名隔离，配置为 \`qwen3.5-9b-dflash-custom\`。
2. 编写 \`scripts/train_dflash.py\`，配置 FSDP / ZeRO-3 显存优化及梯度检查点。
3. 载入离线特征，设置 Block Size = 16，开始训练草稿模型，保存权重到 \`./checkpoints/qwen3.5-9b-dflash-custom/\`。

### 交付件
- \`scripts/train_dflash.py\` 训练代码
- 成功训练并保存的草稿模型目录 \`./checkpoints/qwen3.5-9b-dflash-custom/\`"

# Issue 4
gh issue create \
  --title "DFlash训练实战 | Issue 4: 自建草稿模型无损正确性校验" \
  --body "### 目标
验证投机解码框架下的生成文本与目标模型原始自回归生成的一致性，确保 100% 无损。

### 任务细分
1. 编写 \`scripts/verify_correctness.py\` 脚本。
2. 在 Greedy Decoding (temp=0) 模式下，对比【目标模型+自建草稿模型】与【独立目标模型】生成的 Token ID。
3. 确保所有生成的测试样本完全一致。

### 交付件
- \`scripts/verify_correctness.py\` 校验代码与结果报告"

# Issue 5
gh issue create \
  --title "DFlash训练实战 | Issue 5: 自研 DFlash vs 第三方模型对比评测" \
  --body "### 目标
在 GSM8K 等 benchmark 上，对比自研模型 \`qwen3.5-9b-dflash-custom\` 与官方第三方模型的加速性能与接受率。

### 任务细分
1. 对比两者在相同 Block Size (如 16) 下的平均接受长度 \$\tau\$ 及直方图。
2. 评测两者带来的端到端加速比及吞吐量。
3. 将对比报告整理为 \`results/custom_vs_thirdparty_dflash_report.md\`。

### 交付件
- 最终评测报告 \`results/custom_vs_thirdparty_dflash_report.md\`"

echo "Issues 创建完成！"
