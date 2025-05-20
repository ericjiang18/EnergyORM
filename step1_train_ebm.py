import os
import argparse
import json
import random
import argparse
import copy
from collections import defaultdict
from functools import partial # For collate_fn

# --- FIX: Disable Tokenizer Parallelism ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# -----------------------------------------

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

from ebm_model import TransEBM
from dataset import TrainValChunkDS, collate_fn as original_collate_fn
from utils import (
    get_device_and_amp_helpers,
    setup_tokenizer,
    bradley_terry_loss,
    evaluate,
    load_q2cands_from_jsonl
)


script_directory = os.path.dirname(os.path.abspath(__file__))

default_data_path = os.path.join(
    script_directory,
    "demo_dataset",
    "results_gsm8k_llama_train_n4_temp0.7_p0.9_train_corrected.jsonl" 
)


def main(args):
    DEV, use_amp, autocaster, scaler = get_device_and_amp_helpers(args.device, args.fp16)
    tok, PAD_ID, CLS_ID = setup_tokenizer(args.tok, args.max_length)

    # Prepare collate function with PAD_ID
    collate_with_pad = partial(original_collate_fn, pad_id=PAD_ID)

    print("\n--- Loading Training Data ---")
    # Load Llama GSM training data
    train_val_q2cands = load_q2cands_from_jsonl(args.train_llama_gsm_data, "Llama GSM training/validation")
    
    if not train_val_q2cands:
        print(f"Error: Could not load training data from {args.train_llama_gsm_data}. Exiting.")
        exit(1)
    print(f"Total unique questions loaded for training/validation: {len(train_val_q2cands)}")

    print("\n--- Initializing Datasets ---")
    train_ds, val_ds = None, None
    train_dl, val_dl = None, None

    try:
        train_ds = TrainValChunkDS(tok, args.max_length, CLS_ID, PAD_ID,
                                   q2cands_data=copy.deepcopy(train_val_q2cands),
                                   split="train", holdout=args.val_holdout,
                                   dataset_name_log_prefix="llama_gsm_")
        val_ds = TrainValChunkDS(tok, args.max_length, CLS_ID, PAD_ID,
                                 q2cands_data=copy.deepcopy(train_val_q2cands),
                                 split="val", holdout=args.val_holdout,
                                 dataset_name_log_prefix="llama_gsm_")
    except Exception as data_err:
        print(f"Fatal Error: Could not initialize datasets. {data_err}")
        import traceback
        traceback.print_exc()
        exit(1)

    pin_mem = DEV.type == 'cuda'
    num_workers = args.num_workers

    if train_ds and len(train_ds) > 0:
        train_dl = DataLoader(train_ds, batch_size=args.bsz, shuffle=True,
                              collate_fn=collate_with_pad, pin_memory=pin_mem, num_workers=num_workers)
    else:
        print("Error: Training dataset is empty or failed to load. Cannot proceed.")
        exit(1)

    if val_ds and len(val_ds) > 0:
        val_dl = DataLoader(val_ds, batch_size=args.bsz, shuffle=False,
                            collate_fn=collate_with_pad, pin_memory=pin_mem, num_workers=num_workers)
    else:
        print("Warning: Validation dataset is empty or failed to load. Validation will be skipped.")
    
    print("--- Data Loading and DataLoader Creation Complete ---")
    
    if val_ds and hasattr(val_ds, 'groups') and len(val_ds.groups) > 0:
        val_oracle_hits = sum(1 for grp_dict in val_ds.groups if "meta" in grp_dict and grp_dict["meta"].get("has_correct", False))
        val_oracle_acc  = 100 * val_oracle_hits / len(val_ds.groups)
        print(f"Maximum achievable accuracy on val set (oracle): {val_oracle_acc:.2f}% ({val_oracle_hits}/{len(val_ds.groups)} groups have a correct answer)")
    else:
        print("Skipping oracle accuracy calculation as validation set is empty or invalid.")


    print("\n--- Model Setup ---")
    model = TransEBM(
        vocab_size=len(tok),
        d_model=args.embed_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout
    ).to(DEV)

    if hasattr(model, 'resize_token_embeddings') and len(tok) != model.emb.num_embeddings:
         print(f"Resizing token embeddings for model. Tokenizer vocab size: {len(tok)}, Model embedding size: {model.emb.num_embeddings}.")
         model.resize_token_embeddings(len(tok))

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    num_train_batches = len(train_dl)
    if num_train_batches == 0:
        print("Error: Training DataLoader has zero batches. Cannot setup scheduler or train.")
        exit(1)
    total_steps = args.epochs * num_train_batches
    num_warmup_steps = int(args.warmup_ratio * total_steps)
    sched = get_cosine_schedule_with_warmup(
        opt,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps
    )
    print("--- Model Setup Complete ---")

    print(f"\n--- Starting Training (using Bradley-Terry Loss) ---")
    print(f"Config: d_model={args.embed_dim}, n_layers={args.n_layers}, n_heads={args.n_heads}, dropout={args.dropout}, max_len={args.max_length}")
    print(f"Tokenizer: {args.tok}, Effective CLS_ID: {CLS_ID}, PAD_ID: {PAD_ID}")
    print(f"Device: {DEV}, FP16 (AMP): {use_amp}, Train Groups: {len(train_ds)}, Val Groups: {len(val_ds) if val_ds else 0}")
    print(f"Epochs: {args.epochs}, Batch Size: {args.bsz}, LR: {args.lr}, Weight Decay: {args.weight_decay}, Val Holdout: {args.val_holdout}")
    print(f"Validate every {args.validate_every} epochs.")


    best_val_acc, best_state = 0.0, None

    for ep in range(1, args.epochs + 1):
        model.train()
        total_ep_loss = 0.0
        batches_processed = 0
        pbar = tqdm(train_dl, desc=f"Epoch {ep}/{args.epochs} Training", unit="batch")
        for idsL, maskL, labL in pbar:
            opt.zero_grad(set_to_none=True)
            batch_losses = []
            for ids_b, mask_b, lab_b in zip(idsL, maskL, labL): # Renamed to avoid conflict
                if ids_b.numel() == 0: continue
                ids_b, mask_b, lab_b = ids_b.to(DEV), mask_b.to(DEV), lab_b.to(DEV)
                with autocaster():
                    e = model(ids_b, mask_b)
                    loss_group = bradley_terry_loss(e, lab_b)
                if loss_group is not None and torch.isfinite(loss_group):
                    batch_losses.append(loss_group)
                elif loss_group is not None:
                     print(f"Warning: Encountered non-finite loss ({loss_group.item()}) in epoch {ep}. Skipping gradient update for this group.")
            
            if not batch_losses: continue
            loss = torch.stack(batch_losses).mean()

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            
            sched.step()
            total_ep_loss += loss.item()
            batches_processed += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{sched.get_last_lr()[0]:.2e}")

        avg_ep_loss = total_ep_loss / batches_processed if batches_processed > 0 else 0.0
        print(f"Epoch {ep} finished. Average Training Loss: {avg_ep_loss:.4f}")

        if ep % args.validate_every == 0:
            current_val_acc = 0.0
            if val_dl:
                naive_val_acc, current_val_acc = evaluate(model, val_dl, DEV, autocaster, eval_type="Validation")
                print(f"Epoch {ep} Validation → EBM Acc: {current_val_acc:.2f}% | Naive Acc: {naive_val_acc:.2f}%")
                if current_val_acc > best_val_acc:
                    best_val_acc = current_val_acc
                    best_state = copy.deepcopy(model.state_dict())
                    print(f"✨ New best validation accuracy: {best_val_acc:.2f}%. State saved.")
                else:
                    print(f"(Current best val acc: {best_val_acc:.2f}%)")
            else:
                print("Skipping validation as validation data loader is not available.")
    
    print("\n--- Training Finished ---")

    if best_state is not None:
        model_save_path = f"{args.save_prefix}_model.pt"
        tokenizer_save_path = f"{args.save_prefix}_tokenizer"
        print(f"\nSaving best model state dictionary (Val Acc: {best_val_acc:.2f}%) to {model_save_path}")
        torch.save(best_state, model_save_path)
        
        print(f"Saving tokenizer configuration (from {args.tok}) to {tokenizer_save_path}/")
        try:
            os.makedirs(tokenizer_save_path, exist_ok=True)
            tok.save_pretrained(tokenizer_save_path)
            print(f"✓ Model state and tokenizer saved successfully with prefix '{args.save_prefix}'.")
        except Exception as e:
            print(f"Error saving tokenizer: {e}")
    else:
        print("\nNo best model state was saved (validation accuracy did not improve or validation was skipped/failed).")

    print("\n--- Training Script Complete ---")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train TransEBM with Llama GSM data.")

    # Data Args
    p.add_argument("--train_llama_gsm_data", default=default_data_path,help="Path to Llama GSM training data (.jsonl). Default is relative to the script location.")    
    p.add_argument("--val_holdout", type=float, default=0.2, help="Fraction of training data to hold out for validation (e.g., 0.2 for 20%)")

    # Model Args
    p.add_argument("--tok", default="gpt2", help="HuggingFace tokenizer name or local path")
    p.add_argument("--embed_dim", type=int, default=768, help="Model embedding dimension")
    p.add_argument("--n_layers", type=int, default=2, help="Number of Transformer encoder layers")
    p.add_argument("--n_heads", type=int, default=4, help="Number of attention heads")
    p.add_argument("--dropout", type=float, default=0.2, help="Dropout rate in Transformer layers")
    p.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length for tokenizer") # Reduced default from 4096

    # Training Args
    p.add_argument("--epochs", type=int, default=20, help="Number of training epochs") # Reduced default
    p.add_argument("--bsz", type=int, default=1, help="Batch size (number of groups per batch)")
    p.add_argument("--lr", type=float, default=5e-5, help="Learning rate") # Adjusted default
    p.add_argument("--fp16", action="store_true", help="Use Automatic Mixed Precision (AMP) - CUDA only")
    p.add_argument("--device", default="auto", help="Device ('auto', 'cuda', 'cpu', 'mps')")
    p.add_argument("--validate_every", type=int, default=1, help="Validate on validation set every N epochs")
    p.add_argument("--weight_decay",type=float, default=0.01, help="AdamW weight decay")
    p.add_argument("--warmup_ratio",type=float, default=0.1, help="Warmup steps ratio") # Adjusted default
    p.add_argument("--num_workers", type=int, default=2, help="Number of dataloader workers")


    # Output Args
    p.add_argument("--save_prefix", default="ebm_llama_gsm", help="Prefix for saving model and tokenizer")

    args = p.parse_args()
    main(args)