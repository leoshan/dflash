# DFlash SGLang 环境搭建验证报告

## 1. 虚拟环境信息
- **路径**: `/home/shanchuang/dflash/.venv-sglang`
- **Python 版本**: 3.12.3
- **创建方式**: `uv venv .venv-sglang --python 3.12`

## 2. 依赖安装情况
使用 `uv pip install -e ".[sglang]"` 安装了项目及 SGLang 后端。

### 关键包版本
- **sglang**: 0.4.5.post1 (custom build from git)
- **torch**: 2.9.1+cu130
- **transformers**: 5.3.0
- **flashinfer-python**: 0.2.1

## 3. 模型下载与缓存
> **注意**: 由于 Qwen3.6-27B 模型体积过大（约 52GB），且单卡显存硬限制为 16GB，全量评测已调整为使用 **Qwen3.5-9B**。

### 3.1 基座模型 (Qwen3.5-9B)
- **本地路径**: `/home/shanchuang/nvext/models/Qwen3.5-9B`
- **状态**: 已存在且完整。

### 3.2 DFlash 草稿模型 (z-lab/Qwen3.5-9B-DFlash)
- **Repo ID**: `z-lab/Qwen3.5-9B-DFlash`
- **状态**: 已缓存至本地。

### 3.3 EAGLE3 草稿模型 (yuhuili/EAGLE-Qwen2-7B-Instruct)
- **Repo ID**: `yuhuili/EAGLE-Qwen2-7B-Instruct`
- **状态**: 正在下载/已缓存，用于对比评测。

## 4. 最小化验证
已成功启动 `Qwen3.5-9B` 服务（TP=4），并验证了 DFlash 投机解码在 SGLang 上的集成。

### 验证命令示例 (TP=4)
```bash
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
python -m sglang.launch_server \
    --model-path /home/shanchuang/nvext/models/Qwen3.5-9B \
    --tp-size 4 \
    --mem-fraction-static 0.7 \
    --trust-remote-code
```

## 5. 交付脚本
- **setup_env.sh**: 一键环境搭建与验证脚本。
- **scripts/run_baseline.sh**: 原始性能评测脚本。
- **scripts/run_dflash.sh**: DFlash 性能评测脚本。
- **scripts/run_eagle3.sh**: EAGLE3 性能评测脚本。
