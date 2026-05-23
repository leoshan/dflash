#!/usr/bin/env python3
# scripts/train_dflash.py
# 训练 DFlash 草稿模型 (Speculator)，支持单卡及多卡 FSDP 训练，并应用前置衰减与渐进式 Curriculum 学习

import argparse
import os
import glob
import functools
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sys
from pathlib import Path
from tqdm import tqdm
from safetensors.torch import load_file
from transformers import Qwen3Config
from dflash.model import DFlashDraftModel

def parse_args():
    parser = argparse.ArgumentParser(description="DFlash Speculator Training Script")
    parser.add_argument(
        "--target_model_path",
        type=str,
        default="/home/shanchuang/nvext/models/Qwen3.5-9B",
        help="Target model checkpoint path (to extract embed_tokens and lm_head)"
    )
    parser.add_argument(
        "--features_dir",
        type=str,
        default="data/features",
        help="Directory containing extracted target hidden features (.pt files)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/qwen3.5-9b-dflash-custom",
        help="Output directory to save trained speculator weights"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size per GPU (default: 2)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.85,
        help="Early loss decay parameter (gamma)"
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="Gradient clipping threshold"
    )
    parser.add_argument(
        "--max_block_size",
        type=int,
        default=16,
        help="Max block size for speculative decoding (default: 16)"
    )
    return parser.parse_args()

class FeatureDataset(Dataset):
    def __init__(self, features_dir):
        self.files = sorted(glob.glob(os.path.join(features_dir, "features_chunk_*.pt")))
        if len(self.files) == 0:
            raise FileNotFoundError(f"No feature files found in {features_dir}. Please run scripts/extract_features.py first.")
            
        print(f"Dataset: found {len(self.files)} chunk files.")
        
        self.input_ids_list = []
        self.target_hidden_list = []
        
        for file in self.files:
            data = torch.load(file, map_location="cpu")
            self.input_ids_list.append(data["input_ids"])
            self.target_hidden_list.append(data["target_hidden"])
            
        self.input_ids = torch.cat(self.input_ids_list, dim=0)
        self.target_hidden = torch.cat(self.target_hidden_list, dim=0)
        
        assert self.input_ids.shape[0] == self.target_hidden.shape[0]
        print(f"Dataset: loaded {self.input_ids.shape[0]} total training sequences.")

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "target_hidden": self.target_hidden[idx]
        }

def setup_distributed():
    if "RANK" in os.environ:
        # Running under torchrun
        torch.distributed.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return int(os.environ["RANK"]), local_rank, int(os.environ["WORLD_SIZE"])
    return 0, 0, 1

def main():
    args = parse_args()
    rank, local_rank, world_size = setup_distributed()
    is_distributed = world_size > 1
    device = torch.device(f"cuda:{local_rank}")
    
    if rank == 0:
        print("=== Starting DFlash Speculator Training ===")
        print(f"Distributed Setup: World Size = {world_size}")
        
    # 1. Load speculator configuration
    # We use z-lab/Qwen3.5-9B-DFlash config as template to ensure identical architecture
    if rank == 0:
        print("Loading config template from z-lab/Qwen3.5-9B-DFlash...")
    config = Qwen3Config.from_pretrained("z-lab/Qwen3.5-9B-DFlash")
    
    # 2. Instantiate custom draft model
    raw_model = DFlashDraftModel(config)
    raw_model.to(device, dtype=torch.bfloat16)
    
    # Enable gradient checkpointing for VRAM safety
    raw_model.gradient_checkpointing_enable()
    
    # 3. Load embed_tokens & lm_head from Qwen3.5-9B first chunk safetensor
    # These parameters are frozen and only used for computing embeddings and predictions
    if rank == 0:
        print("Loading target model embedding and lm_head weights...")
    sf_path = os.path.join(args.target_model_path, "model.safetensors-00001-of-00004.safetensors")
    if not os.path.exists(sf_path):
        print(f"Error: safetensors file not found at {sf_path}")
        sys.exit(1)
        
    weights = load_file(sf_path)
    embed_tokens_weight = weights["model.language_model.embed_tokens.weight"]
    lm_head_weight = weights["lm_head.weight"]
    
    embed_tokens = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.hidden_size)
    embed_tokens.weight.data.copy_(embed_tokens_weight)
    embed_tokens.to(device, dtype=torch.bfloat16)
    embed_tokens.requires_grad_(False)
    
    lm_head = nn.Linear(in_features=config.hidden_size, out_features=config.vocab_size, bias=False)
    lm_head.weight.data.copy_(lm_head_weight)
    lm_head.to(device, dtype=torch.bfloat16)
    lm_head.requires_grad_(False)
    
    del weights # Free memory
    
    # 4. Wrap with FSDP if distributed
    if is_distributed:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
        
        if rank == 0:
            print("Wrapping model with PyTorch FSDP...")
            
        my_auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=1e6
        )
        model = FSDP(
            raw_model,
            auto_wrap_policy=my_auto_wrap_policy,
            cpu_offload=CPUOffload(offload_params=False),
            device_id=torch.cuda.current_device()
        )
    else:
        model = raw_model
        
    # 5. Prepare DataLoader
    try:
        dataset = FeatureDataset(args.features_dir)
    except Exception as e:
        if rank == 0:
            print(f"Error loading dataset: {e}")
        sys.exit(1)
        
    sampler = None
    if is_distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None)
    )
    
    # 6. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Compute total steps for curriculum learning
    total_steps = len(dataloader) * args.epochs
    step_count = 0
    
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    
    if rank == 0:
        print(f"Starting training for {args.epochs} epochs...")
        
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
            
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=(rank != 0))
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            target_hidden = batch["target_hidden"].to(device)
            batch_size, seq_len = input_ids.shape
            
            # Step-based Curriculum Block Size: starts at 2, increases to max_block_size (16)
            current_block_size = min(
                args.max_block_size,
                int(2 + (step_count / max(1, total_steps)) * (args.max_block_size - 2))
            )
            
            # Embed target tokens
            noise_embedding = embed_tokens(input_ids)
            
            # Both noise_embedding and target_hidden are seq_len (2048) in length during training.
            # DFlash applies position embeddings to both context (target_hidden) and query (noise_embedding)
            # concatenated together, so we need position_ids of length ctx_len + seq_len.
            ctx_len = target_hidden.shape[1]
            position_ids = torch.arange(ctx_len + seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            
            # Construct causal attention mask to prevent draft tokens attending to future tokens
            causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=torch.bfloat16), diagonal=1)
            # Concatenate two causal masks along the key sequence dimension (ctx_len + seq_len)
            attention_mask = torch.cat([causal_mask, causal_mask], dim=-1)
            # Expand to 4D: [batch_size, 1, seq_len, ctx_len + seq_len]
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, -1, -1)
            
            # Forward pass
            outputs = model(
                position_ids=position_ids,
                noise_embedding=noise_embedding,
                target_hidden=target_hidden,
                use_cache=False,
                attention_mask=attention_mask
            )
            
            # Project outputs to vocabulary
            logits = lm_head(outputs)
            
            # Compute loss on shifted logits (predict next token)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            shift_len = shift_logits.shape[1]
            
            # Loss weights logic:
            # 1. Early Loss Decay: decay factor based on position inside block (t % max_block_size)
            decay_weights = torch.tensor(
                [args.gamma ** (t % args.max_block_size) for t in range(shift_len)],
                dtype=torch.float,
                device=device
            )
            # 2. Curriculum mask: block positions >= current_block_size are masked out
            curriculum_mask = torch.tensor(
                [1.0 if (t % args.max_block_size) < current_block_size else 0.0 for t in range(shift_len)],
                dtype=torch.float,
                device=device
            )
            
            # Combined weights
            step_weights = decay_weights * curriculum_mask
            
            # Compute cross entropy per token
            loss_per_token = loss_fn(shift_logits.view(-1, config.vocab_size), shift_labels.view(-1))
            loss_per_token = loss_per_token.view(batch_size, shift_len)
            
            # Apply weights
            weighted_loss = (loss_per_token * step_weights.unsqueeze(0)).sum() / (step_weights.sum() * batch_size + 1e-8)
            
            # Backward and optimizer step
            weighted_loss.backward()
            
            if args.grad_clip > 0:
                if is_distributed:
                    model.clip_grad_norm_(args.grad_clip)
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    
            optimizer.step()
            
            epoch_loss += weighted_loss.item()
            step_count += 1
            
            if rank == 0:
                progress_bar.set_postfix({
                    "loss": f"{weighted_loss.item():.4f}",
                    "curriculum_b": current_block_size
                })
                
        if rank == 0:
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch} Complete. Average Loss: {avg_loss:.4f}")
            
    # 7. Save custom draft model
    if is_distributed:
        from torch.distributed.fsdp import StateDictType, FullStateDictConfig
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        
        # Configure FSDP to gather full state dict on rank 0 and offload to CPU
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            state_dict = model.state_dict()
    else:
        state_dict = model.state_dict()
        
    if rank == 0:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Saving speculator config and model weights to {output_path}...")
        
        # Save weights using the raw model with the gathered state dict
        raw_model.save_pretrained(output_path, state_dict=state_dict)
        
        # Copy tokenizer files from target model path to ensure it's a complete Hugging Face model folder
        try:
            import shutil
            for file in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"]:
                src = os.path.join(args.target_model_path, file)
                if os.path.exists(src):
                    shutil.copy(src, output_path)
            print("Successfully copied tokenizer config files to checkpoints directory.")
        except Exception as e:
            print(f"Warning: could not copy tokenizer files: {e}")
            
        print("Training successfully complete!")
        
    if is_distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()
