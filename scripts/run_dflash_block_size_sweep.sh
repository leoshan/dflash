#!/bin/bash
# DFlash Block Size Sensitivity Sweep Wrapper

PROJECT_ROOT="/home/shanchuang/dflash"
PYTHON_BIN="$PROJECT_ROOT/.venv-sglang/bin/python"

echo "开始运行 DFlash Block Size (draft token count) 敏感度扫参测试..."
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/run_dflash_block_size_sweep.py"
echo "扫参测试完成！"
