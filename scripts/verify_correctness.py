#!/usr/bin/env python3
# scripts/verify_correctness.py
# 验证 DFlash 自建草稿模型在投机解码模式下的生成结果与目标模型自回归生成的一致性 (无损正确性校验)

import argparse
import os
import torch
import sys
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from dflash.model import DFlashDraftModel

def parse_args():
    parser = argparse.ArgumentParser(description="DFlash Speculative Decoding Correctness Verification")
    parser.add_argument(
        "--target_model_path",
        type=str,
        default="/home/shanchuang/nvext/models/Qwen3.5-9B",
        help="Target model checkpoint path"
    )
    parser.add_argument(
        "--draft_model_path",
        type=str,
        default="checkpoints/qwen3.5-9b-dflash-custom",
        help="Draft model checkpoint path"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
        help="Number of tokens to generate for verification"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Load tokenizer
    print(f"Loading tokenizer from {args.target_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path, trust_remote_code=True)
    
    # 2. Load target model using device_map="auto"
    print(f"Loading target model from {args.target_model_path}...")
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model_path,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True
    )
    target_model.eval()
    
    # 3. Load draft model
    print(f"Loading draft model from {args.draft_model_path}...")
    draft_model = DFlashDraftModel.from_pretrained(args.draft_model_path)
    # Put draft model on the first device of target model (typically cuda:0)
    first_device = next(target_model.parameters()).device
    draft_model.to(first_device)
    draft_model.eval()
    
    # Define test prompts
    test_prompts = [
        "What is the capital of France? Response:",
        "Solve the equation: 2x + 5 = 15. x =",
        "Explain the concept of speculative decoding in one sentence. Explanation:",
        "Translate to Chinese: 'Artificial intelligence is changing the world.' Translation:",
        "The quick brown fox jumps over the lazy dog. Rewrite this sentence in a formal style:"
    ]
    
    print("\n" + "="*50)
    print("Starting Correctness Verification (Greedy, Temp=0)")
    print("="*50)
    
    all_passed = True
    
    for idx, prompt in enumerate(test_prompts):
        print(f"\n[Test Prompt {idx + 1}/{len(test_prompts)}]: '{prompt}'")
        
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(first_device)
        
        # A. Run standard Autoregressive (AR) generation
        print(" -> Running Autoregressive (AR) generation...")
        with torch.no_grad():
            ar_output_ids = target_model.generate(
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # B. Run Speculative Decoding (SD) generation
        print(" -> Running Speculative Decoding (SD) generation...")
        with torch.no_grad():
            sd_output = draft_model.spec_generate(
                target=target_model,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=0.0
            )
            
        # If output was wrapped in SimpleNamespace (if return_stats was True, but here it's False by default)
        sd_output_ids = sd_output.output_ids if hasattr(sd_output, "output_ids") else sd_output
        
        # Get generated tokens (excluding prompt)
        prompt_len = input_ids.shape[1]
        ar_generated = ar_output_ids[0, prompt_len:].cpu().tolist()
        sd_generated = sd_output_ids[0, prompt_len:prompt_len + len(ar_generated)].cpu().tolist()
        
        # Verify alignment
        is_match = (ar_generated == sd_generated)
        
        ar_text = tokenizer.decode(ar_generated, skip_special_tokens=True)
        sd_text = tokenizer.decode(sd_generated, skip_special_tokens=True)
        
        print(f" AR Output: {ar_generated} -> '{ar_text}'")
        print(f" SD Output: {sd_generated} -> '{sd_text}'")
        
        if is_match:
            print(" Result: [PASSED] (Token IDs match 100%!)")
        else:
            print(" Result: [FAILED] (Token IDs mismatch!)")
            # Show diff
            min_len = min(len(ar_generated), len(sd_generated))
            for t_idx in range(min_len):
                if ar_generated[t_idx] != sd_generated[t_idx]:
                    print(f"   Mismatch at token index {t_idx}: AR={ar_generated[t_idx]} ('{tokenizer.decode([ar_generated[t_idx]])}'), SD={sd_generated[t_idx]} ('{tokenizer.decode([sd_generated[t_idx]])}')")
                    break
            all_passed = False
            
    print("\n" + "="*50)
    if all_passed:
        print("VERIFICATION SUCCESSFUL: Speculative decoding results are 100% mathematically equivalent to Autoregressive generation.")
        sys.exit(0)
    else:
        print("VERIFICATION FAILED: Found differences in generated outputs.")
        sys.exit(1)

if __name__ == "__main__":
    main()
