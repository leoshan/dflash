#!/usr/bin/env python3
# scripts/extract_features.py
# 部署 Qwen3.5-9B 基座模型，离线抓取指定层（1, 8, 15, 22, 29 层）的 Hidden States 并序列化保存

import argparse
import os
import json
import torch
import sys
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM

def parse_args():
    parser = argparse.ArgumentParser(description="DFlash Offline Feature Extraction for Qwen3.5-9B")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/shanchuang/nvext/models/Qwen3.5-9B",
        help="Target model checkpoint path"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="data/qwen3.5_tokenized.jsonl",
        help="Input tokenized dataset JSONL file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/features",
        help="Output directory to save extracted feature .pt files"
    )
    parser.add_argument(
        "--target_layers",
        type=int,
        nargs="+",
        default=[1, 8, 15, 22, 29],
        help="Target model layers to extract hidden states from"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for forward propagation (default: 2, safe for VRAM)"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=10,
        help="Number of sequences to pack into each saved .pt file chunk"
    )
    return parser.parse_args()

def load_target_model(model_path):
    print(f"Loading Qwen3.5-9B target model from {model_path}...")
    print("Distributing layers across GPUs using device_map='auto' for VRAM safety...")
    
    try:
        # device_map="auto" will automatically and evenly distribute the 9B model
        # across available H200 cards (TP-like pipelining), keeping VRAM per card around 4.5 GB.
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            dtype=torch.bfloat16,
            trust_remote_code=True
        )
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading target model: {e}")
        sys.exit(1)

def extract_features_batch(model, input_ids_batch, target_layers):
    """
    Run forward pass and extract hidden states from specified layers.
    Concatenates the selected hidden states along the feature dimension (dim=-1).
    """
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids_batch,
            output_hidden_states=True,
            use_cache=False
        )
        
        # outputs.hidden_states is a tuple of length (num_layers + 1)
        # index 0 is embedding output, index i is output of layer i
        # The offset is +1 relative to layer ID if we match transformers conventions
        hidden_states = outputs.hidden_states
        
        extracted_batch = []
        for i in range(input_ids_batch.shape[0]):
            selected_states = []
            for layer_id in target_layers:
                # Get the state for this sequence in the batch
                # Layer outputs are [batch_size, seq_len, hidden_size]
                # We move tensors to CPU/bfloat16 immediately to save GPU memory
                layer_state = hidden_states[layer_id + 1][i].cpu().to(torch.bfloat16)
                selected_states.append(layer_state)
            
            # Concatenate along dim=-1 (hidden dimension)
            # Shape: [seq_len, len(target_layers) * hidden_size]
            concatenated = torch.cat(selected_states, dim=-1)
            extracted_batch.append(concatenated)
            
        # Stack into [batch_size, seq_len, concatenated_hidden]
        return torch.stack(extracted_batch)

def main():
    args = parse_args()
    
    # 1. Read input tokenized dataset
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Tokenized dataset file not found at {input_path}")
        print("Please run scripts/prepare_dataset.py first.")
        sys.exit(1)
        
    print(f"Reading tokenized sequences from {input_path}...")
    sequences = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                sequences.append(json.loads(line)["input_ids"])
                
    total_seqs = len(sequences)
    print(f"Loaded {total_seqs} sequences of length 2048.")
    
    # 2. Load model
    model = load_target_model(args.model_path)
    print(f"Model successfully loaded. Devices in use: {model.hf_device_map}")
    
    # 3. Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 4. Batch processing and extracting
    batch_size = args.batch_size
    chunk_size = args.chunk_size
    
    current_chunk_ids = []
    current_chunk_features = []
    chunk_idx = 0
    
    print(f"Starting feature extraction in batches of {batch_size}...")
    for idx in tqdm(range(0, total_seqs, batch_size), desc="Extracting features"):
        batch_seqs = sequences[idx : idx + batch_size]
        
        # Convert to tensor and send to the model's first device
        # Note: device_map="auto" expects inputs on the device of the first layer
        first_device = model.device
        input_ids_batch = torch.tensor(batch_seqs, dtype=torch.long, device=first_device)
        
        # Extract features
        features_batch = extract_features_batch(model, input_ids_batch, args.target_layers)
        
        # Add to current chunk (as CPU tensors)
        for i in range(len(batch_seqs)):
            current_chunk_ids.append(batch_seqs[i])
            current_chunk_features.append(features_batch[i])
            
        # Save chunk if size reached, or if at the end of dataset
        if len(current_chunk_features) >= chunk_size or idx + batch_size >= total_seqs:
            chunk_file = output_dir / f"features_chunk_{chunk_idx}.pt"
            
            # Stack features in chunk: [chunk_size, seq_len, concatenated_hidden]
            chunk_features_tensor = torch.stack(current_chunk_features)
            chunk_ids_tensor = torch.tensor(current_chunk_ids, dtype=torch.long)
            
            payload = {
                "input_ids": chunk_ids_tensor,
                "target_hidden": chunk_features_tensor,
                "target_layers": args.target_layers
            }
            
            print(f"\nSaving chunk {chunk_idx} to {chunk_file} (Contains {len(current_chunk_ids)} sequences)...")
            torch.save(payload, chunk_file)
            
            # Reset chunk buffers
            current_chunk_ids = []
            current_chunk_features = []
            chunk_idx += 1
            
    print(f"\nFeature extraction complete! Saved {chunk_idx} chunks to {args.output_dir}.")

if __name__ == "__main__":
    main()
