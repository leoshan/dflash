#!/bin/bash
# DFlash concurrency sweep run script

PROJECT_ROOT="/home/shanchuang/dflash"
export PATH="$PROJECT_ROOT/.venv-sglang/bin:$PATH"
PYTHON_BIN="$PROJECT_ROOT/.venv-sglang/bin/python"

echo "开始运行 DFlash Concurrency Sweep 评测..."
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/run_dflash_concurrency_sweep.py"
echo "DFlash Concurrency Sweep 评测已完成！"
