#!/bin/bash
set -e

PROJECT_ROOT="/home/shanchuang/dflash"
VENV_PATH="$PROJECT_ROOT/.venv-sglang"
PYTHON_BIN="$VENV_PATH/bin/python"

echo "=== DFlash SGLang 环境搭建脚本 ==="

# 1. 创建虚拟环境
if [ ! -d "$VENV_PATH" ]; then
    echo "正在创建虚拟 environment: $VENV_PATH ..."
    uv venv "$VENV_PATH" --python 3.12
else
    echo "虚拟环境已存在。"
fi

# 2. 安装依赖
echo "正在安装 SGLang 及项目依赖..."
uv pip install -e "$PROJECT_ROOT.[sglang]" --python "$PYTHON_BIN"

# 3. 验证关键包版本
echo "=== 依赖版本验证 ==="
"$PYTHON_BIN" -c "import sglang; print(f'SGLang version: {sglang.__version__}')"
"$PYTHON_BIN" -c "import torch; print(f'Torch version: {torch.__version__}')"
"$PYTHON_BIN" -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

# 4. 下载模型 (如果设置了 HF_TOKEN)
echo "=== 模型下载 (可选) ==="
if [ -n "$HF_TOKEN" ]; then
    echo "检测到 HF_TOKEN，尝试下载模型..."
    "$PYTHON_BIN" -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.6-27B')"
    "$PYTHON_BIN" -c "from huggingface_hub import snapshot_download; snapshot_download('z-lab/Qwen3.6-27B-DFlash')"
else
    echo "未检测到 HF_TOKEN。请确保模型已存在于本地或手动执行下载。"
    echo "基座模型路径建议: /home/shanchuang/nvext/models/Qwen3.6-27B"
fi

echo "=== 环境搭建完成 ==="
