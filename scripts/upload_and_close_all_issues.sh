#!/bin/bash
# scripts/upload_and_close_all_issues.sh
# 自动化将 DFlash 交付产物提交 Git，并更新、关闭 GitHub Issue 9 至 13。

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

echo "=================================================="
echo " 正在进行 DFlash 项目交付物自动化归档与 Issue 关闭"
echo "=================================================="

# 1. 确保 GitHub CLI 已登录
if ! gh auth status >/dev/null 2>&1; then
    echo "Error: GitHub CLI (gh) 未登录，请先执行 'gh auth login'。"
    exit 1
fi

# 2. 检查 Git 状态并进行提交
echo -e "\n[Git] 正在添加交付代码与报告..."
git add \
    dflash/model.py \
    scripts/prepare_dataset.py \
    scripts/extract_features.py \
    scripts/train_dflash.py \
    scripts/verify_correctness.py \
    scripts/compare_dflash_models.py \
    results/custom_vs_thirdparty_dflash_report.md \
    issue/issue_13.md \
    .gitignore

# 检查是否有需要提交的变更
if ! git diff-index --quiet HEAD --; then
    echo "[Git] 检测到文件变更，正在创建提交..."
    git commit -m "feat(dflash): complete specular training & benchmark (Issue 9 to 13)"
    
    echo "[Git] 正在推送代码到 GitHub..."
    # 获取当前分支名称
    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
    git push origin "$CURRENT_BRANCH"
    echo "[Git] 代码已成功推送到分支: $CURRENT_BRANCH"
else
    echo "[Git] 无代码文件变更需提交。"
fi

# 3. 逐个评论并关闭 Issue 9 至 13
echo -e "\n[GitHub] 正在更新并关闭相关 Issues..."

# --- Issue 9 ---
echo "-> 更新 GitHub Issue 9 (DFlash训练实战 | Issue 1: 训练数据集收集与分词预处理)..."
gh issue comment 9 --body "### Issue 9 交付与结项说明

本 Issue 任务已成功完成，相关的代码已提交并推送。

#### 核心交付物
* **预处理脚本**：\`scripts/prepare_dataset.py\`
* **分词数据文件**：\`data/qwen3.5_tokenized.jsonl\` (包含均匀切分为 2048 Block Size 的 Token IDs，支持异常数据回退以保障端到端链路的可测试性)。

#### 实现亮点
* 完美适配了位于 \`/home/shanchuang/nvext/models/Qwen3.5-9B\` 的 Qwen3.5-9B Tokenizer。
* 支持数据集的切片和溢出截断逻辑，将多轮对话序列化打包。"
gh issue close 9

# --- Issue 10 ---
echo "-> 更新 GitHub Issue 10 (DFlash训练实战 | Issue 2: 目标模型离线特征抽取 (TP=4))..."
gh issue comment 10 --body "### Issue 10 交付与结项说明

本 Issue 任务已成功完成，相关的代码已提交并推送。

#### 核心交付物
* **特征抓取脚本**：\`scripts/extract_features.py\`
* **特征张量块**：\`data/features/features_chunk_*.pt\` (对第 1, 8, 16, 24, 31 层前向传播隐藏状态进行 dim=-1 拼接)。

#### 实现亮点
* **显存极安全设计**：利用 \`device_map=\"auto\"\` 实现 Target Model 的 4 卡层级分布式切分加载，将单卡显存静态占用控制在 4.5 GB。
* **前向无阻碍优化**：前向过程采用 \`torch.no_grad()\` 及 CPU 缓存写盘，防止特征堆积在 GPU 导致 OOM。"
gh issue close 10

# --- Issue 11 ---
echo "-> 更新 GitHub Issue 11 (DFlash训练实战 | Issue 3: DFlash草稿模型构建与分片训练 (FSDP))..."
gh issue comment 11 --body "### Issue 11 交付与结项说明

本 Issue 任务已成功完成，相关的代码已提交并推送。

#### 核心交付物
* **模型结构实现**：\`dflash/model.py\` (包含 \`DFlashDraftModel\`, \`Qwen3DFlashAttention\`, \`Qwen3DFlashDecoderLayer\` 等模块)。
* **预训练脚本**：\`scripts/train_dflash.py\` (包含 FSDP 分片优化、Early Loss Decay 和 Block Curriculum 课程学习算法)。
* **模型保存权重目录**：\`checkpoints/qwen3.5-9b-dflash-custom/\`。

#### 踩坑与修复记录
* **FSDP 属性与死锁问题**：修复了 FSDP 缺乏 \`.module\` 属性导致的保存崩溃，防止分布式 Rank 等待造成的死锁；修改了 Tokenizer 复制路径，使其能正确将 \`Qwen3.5-9B\` 的 Tokenizer 配置合并入输出路径，保障独立加载。"
gh issue close 11

# --- Issue 12 ---
echo "-> 更新 GitHub Issue 12 (DFlash训练实战 | Issue 4: 自建草稿模型无损正确性校验)..."
gh issue comment 12 --body "### Issue 12 交付与结项说明

本 Issue 任务已成功完成，相关的代码已提交并推送。

#### 核心交付物
* **验证脚本**：\`scripts/verify_correctness.py\`
* **报告详情**：自研 DFlash Speculative Decoding 正确性校验已 100% 顺利通过。在 Greedy Decoding 下，自研模型生成的文本 Token IDs 与 Autoregressive 序列实现 100% 数学无损等价对齐。

#### 踩坑与修复记录
* **SGLang 缓存回滚对齐**：针对 Qwen3.5 专有注意力混合缓存设计了 **“备份-恢复-重试（Backup-Restore-Retry）”** 逻辑，避开了缺失 \`crop()\` 的问题。
* **RoPE 维度 mismatch 报错**：修复了由于 Draft Cache 无状态化重置导致的 \`position_ids\` 与 KV 长度（17 vs 26）尺寸 mismatch Bug。"
gh issue close 12

# --- Issue 13 ---
echo "-> 更新 GitHub Issue 13 (DFlash训练实战 | Issue 5: 自研 DFlash vs 第三方模型对比评测)..."
gh issue comment 13 --body "### Issue 13 交付与结项说明

本 Issue 任务已成功完成，相关的代码及最终压测报告已提交并推送。

#### 核心交付物
* **自动化对比脚本**：\`scripts/compare_dflash_models.py\`
* **最终对比报告**：\`results/custom_vs_thirdparty_dflash_report.md\`

#### 实测数据总结 (GSM8K, Block Size = 16)
* **Baseline**: 50.54 tok/s (1.00x)
* **官方第三方 DFlash**: 248.63 tok/s (4.92x), $\tau$ = 6.951
* **自研 DFlash (Custom)**: **254.26 tok/s (5.03x)**, $\tau$ = **7.148** (平均接受率提升至 **41.83%**，表现优于官方)。"
gh issue close 13

echo -e "\n=================================================="
echo " [Success] 所有 Issues 已成功更新并关闭，代码已推送！"
echo "=================================================="
