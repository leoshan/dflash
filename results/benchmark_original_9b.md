# Baseline Benchmark Result (Qwen3.5-9B)

- **Model**: Qwen3.5-9B
- **Framework**: SGLang (Baseline)
- **Quantization**: None (BF16)
- **GPU**: 4× NVIDIA H200 (TP=4)
- **Memory Config**: mem-fraction-static=0.7, disable-cuda-graph=True
- **Dataset**: GSM8K
- **Num Prompts**: 32
- **Concurrency**: 4

## Performance Metrics
- **Total Latency**: 73.0s
- **Total Output Tokens**: 13,247
- **Throughput**: 181.37 tokens/s
- **Avg. Tokens per Prompt**: ~414

## Notes
- CUDA graphs were disabled to fit within the 16GB VRAM limit (actual available space ~11GB).
- Baseline performance is established for comparison with DFlash.
