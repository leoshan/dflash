# DFlash Benchmark Result (Qwen3.5-9B)

- **Model**: Qwen3.5-9B
- **Framework**: SGLang (DFlash)
- **Draft Model**: z-lab/Qwen3.5-9B-DFlash
- **GPU**: 4× NVIDIA H200 (TP=4)
- **Memory Config**: mem-fraction-static=0.7, disable-cuda-graph=True
- **Dataset**: GSM8K
- **Num Prompts**: 32
- **Concurrency**: 4

## Performance Metrics
- **Total Latency**: 69.7s
- **Total Output Tokens**: 13,232
- **Throughput**: 189.90 tokens/s
- **Avg. Tokens per Prompt**: ~413

## Comparison with Baseline
- **Baseline**: 181.37 tokens/s
- **DFlash**: 189.90 tokens/s
- **Speedup**: +4.7%

## Notes
- Speedup is lower than expected (README showed ~1.7x for 27B).
- Possible reasons: 
  1. Small model (9B) already has high baseline throughput on H200.
  2. Disabled CUDA graph adds overhead to the speculative decoding logic.
  3. Small batch size/concurrency might not saturate the speculative benefit.
