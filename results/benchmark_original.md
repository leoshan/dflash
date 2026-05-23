# Baseline Benchmark Result (Qwen3.5-9B)

- **Model**: Qwen3.5-9B
- **Framework**: SGLang (Baseline)
- **Quantization**: None (BF16)
- **GPU**: 4× NVIDIA H200 (TP=4)
- **Memory Config**: mem-fraction-static=0.7, disable-cuda-graph=True
- **Dataset**: GSM8K
- **Num Prompts**: 128
- **Concurrency**: 1

## Performance Metrics
- **Total Latency**: 1053.4s
- **Total Output Tokens**: 53,107
- **Throughput**: 50.41 tokens/s
- **Avg. Tokens per Prompt**: ~414.9

## Notes
- CUDA graphs were disabled to fit within the 16GB VRAM limit.
- Model scale adjusted to 9B as 27B exceeds the 16GB hard limit across 4 GPUs with TP=4.
- Baseline performance is established for comparison with DFlash and EAGLE3.
