# EAGLE3 Benchmark Result (Qwen3.5-9B)

- **Model**: Qwen3.5-9B
- **Draft Model**: yuhuili/EAGLE-Qwen2-7B-Instruct
- **Framework**: SGLang (EAGLE3)

## Status: FAILED

The benchmark could not be completed.

### Error Trace
```python
AttributeError: 'Qwen2ForCausalLM' object has no attribute 'set_embed'
```

### Analysis
SGLang's `EagleDraftWorker` tries to call `.set_embed(embed)` on the model during initialization. However, the `Qwen2ForCausalLM` implementation inside SGLang does not define this method. This indicates that EAGLE3 speculative decoding in the current version of SGLang is incompatible with the Qwen2/3 architecture.

As a result, no performance metrics are available for EAGLE3.
