#!/bin/bash
# EAGLE3 测试脚本 (Qwen3.5-9B on SGLang)

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_ENABLE_SPEC_V2=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJECT_ROOT="/home/shanchuang/dflash"
export PATH="$PROJECT_ROOT/.venv-sglang/bin:$PATH"
PYTHON_BIN="$PROJECT_ROOT/.venv-sglang/bin/python"
MODEL_PATH="/home/shanchuang/nvext/models/Qwen3.5-9B"
DRAFT_MODEL="yuhuili/EAGLE-Qwen2-7B-Instruct"

# 1. 启动 SGLang 服务 (EAGLE)
echo "正在启动 SGLang 服务 (EAGLE3, 9B)..."
"$PYTHON_BIN" -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --tp-size 4 \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path "$DRAFT_MODEL" \
    --mem-fraction-static 0.7 \
    --mamba-scheduler-strategy extra_buffer \
    --disable-cuda-graph \
    --port 30000 \
    --trust-remote-code > sglang_eagle3.log 2>&1 &

SGL_PID=$!

# 2. 等待服务就绪
echo "等待服务就绪 (PID: $SGL_PID)..."
max_retries=60
count=0
while true; do
    if curl -s http://127.0.0.1:30000/v1/models > /dev/null; then
        echo "SGLang (EAGLE3) 服务已就绪！"
        break
    fi
    if ! kill -0 $SGL_PID 2>/dev/null; then
        echo "错误: SGLang 服务启动失败。请检查 sglang_eagle3.log。"
        exit 1
    fi
    count=$((count + 1))
    if [ $count -ge $max_retries ]; then
        echo "错误: 服务启动超时。"
        kill $SGL_PID
        exit 1
    fi
    sleep 5
done

# 3. 运行 Benchmark
echo "开始运行 Benchmark (gsm8k)..."
"$PYTHON_BIN" -m dflash.benchmark \
    --backend sglang \
    --base-url http://127.0.0.1:30000 \
    --model "$MODEL_PATH" \
    --dataset gsm8k \
    --num-prompts 128 \
    --concurrency 1 > benchmark_eagle3_raw.log 2>&1

# 4. 提取结果并保存
echo "测试完成，正在整理结果..."
cat benchmark_eagle3_raw.log

# 5. 清理服务
kill $SGL_PID
echo "服务已停止。"
