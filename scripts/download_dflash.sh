#!/bin/bash
set -e

PROJECT_ROOT="/home/shanchuang/dflash"
PYTHON_BIN="$PROJECT_ROOT/.venv-sglang/bin/python"

if [ ! -f "$PYTHON_BIN" ]; then
    echo "错误: 找不到虚拟环境中的 Python 解释器。请先运行 setup_env.sh。"
    exit 1
fi

echo "正在通过虚拟环境下载 z-lab/Qwen3.6-27B-DFlash ..."
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/download_dflash.py"
