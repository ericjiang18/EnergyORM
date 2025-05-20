import os
import argparse
import json
from collections import defaultdict
from functools import partial # For collate_fn

# --- FIX: Disable Tokenizer Parallelism ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# -----------------------------------------

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ebm_model import TransEBM
from dataset import TestChunkDS, collate_fn as original_collate_fn
from utils import (
    get_device_and_amp_helpers,
    setup_tokenizer,
    evaluate,
    load_q2cands_from_jsonl
)

def main(args):
    DEV, use_amp, autocaster, scaler = get_device_and_amp_helpers(args.device, args.fp16)
    
    # Load tokenizer from the path where it was saved during training
    tok, PAD_ID, CLS_ID = setup_tokenizer(args.tokenizer_path, args.max_length)

    # Prepare collate function with PAD_ID
    collate_with_pad = partial(original_collate_fn, pad_id=PAD_ID)

    print("\n--- Loading Test Data ---")
    test_q2cands = load_q2cands_from_jsonl(args.test_llama_gsm_data, "Llama GSM test")
    
    if not test_q2cands:
        print(f"Error: Could not load test data from {args.test_llama_gsm_data}. Exiting.")
        exit(1)
    print(f"Total unique questions loaded for testing: {len(test_q2cands)}")

    print("\n--- Initializing Test Dataset ---")
    test_ds = None
    test_dl = None

    try:
        # For TestChunkDS, we pass the path directly if q2cands are not pre-loaded,
        # or pass q2cands_data if already loaded. Here, we use path.
        test_ds = TestChunkDS(tok, args.max_length, CLS_ID, PAD_ID,
                              path=args.test_llama_gsm_data, # TestChunkDS will load from this path
                              dataset_name="llama_gsm_test")
    except Exception as data_err:
        print(f"Fatal Error: Could not initialize test dataset. {data_err}")
        import traceback
        traceback.print_exc()
        exit(1)

    pin_mem = DEV.type == 'cuda'
    num_workers = args.num_workers

    if test_ds and len(test_ds) > 0:
        test_dl = DataLoader(test_ds, batch_size=args.bsz, shuffle=False,
                             collate_fn=collate_with_pad, pin_memory=pin_mem, num_workers=num_workers)
    else:
        print("Error: Test dataset is empty or failed to load. Cannot proceed with testing.")
        exit(1)
    
    print("--- Test Data Loading and DataLoader Creation Complete ---")

    print("\n--- Model Setup for Testing ---")
    # Instantiate model with parameters used during training
    # vocab_size comes from the loaded tokenizer
    model = TransEBM(
        vocab_size=len(tok),
        d_model=args.embed_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout # Dropout is typically active during eval if part of model, but here it's for construction
    ).to(DEV)

    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}. Exiting.")
        exit(1)
        
    print(f"Loading model state from {args.model_path}")
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=DEV))
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        print("Ensure that the model parameters (embed_dim, n_layers, n_heads, dropout) match the saved model.")
        exit(1)
    
    # If tokenizer vocab size changed after initial model save (e.g. new special tokens),
    # ensure model's embedding layer matches. This should ideally be handled by saving
    # the final vocab size with the model config, or ensuring tokenizer is identical.
    # The current resize_token_embeddings is for training. For testing, we assume vocab is compatible.
    if len(tok) != model.emb.num_embeddings:
        print(f"Warning: Tokenizer vocab size ({len(tok)}) differs from model embedding size ({model.emb.num_embeddings}).")
        print("This might lead to issues if the model was not trained with this tokenizer vocabulary size.")
        # Optionally, attempt to resize if necessary and safe:
        # model.resize_token_embeddings(len(tok)) 
        # However, this is risky if the original embeddings are critical and not correctly transferred/initialized.
        # It's best practice to use the exact tokenizer config the model was trained with.


    print("--- Model Setup Complete ---")

    print(f"\n--- Starting Testing ---")
    print(f"Test Data: {args.test_llama_gsm_data}")
    print(f"Model: {args.model_path}, Tokenizer: {args.tokenizer_path}")
    print(f"Config: d_model={args.embed_dim}, n_layers={args.n_layers}, n_heads={args.n_heads}, max_len={args.max_length}")
    print(f"Device: {DEV}, FP16 (AMP): {use_amp}, Test Groups: {len(test_ds)}")

    naive_test_acc, ebm_test_acc = evaluate(model, test_dl, DEV, autocaster, eval_type="Test Set 'Llama GSM'")
    print(f"\n--- Test Results for Llama GSM ---")
    print(f"EBM Accuracy: {ebm_test_acc:.2f}%")
    print(f"Naive Accuracy: {naive_test_acc:.2f}%")

    print("\n--- Testing Script Complete ---")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Test TransEBM with Llama GSM data.")

    # Data Args
    p.add_argument("--test_llama_gsm_data", default="demo_dataset/results_gsm8k_llama_test_n4_temp0.7_p0.9_test_corrected.jsonl", help="Path to Llama GSM test data (.jsonl)")
    
    # Model and Tokenizer Paths
    p.add_argument("--model_path", default="ebm_llama_gsm_model.pt", help="Path to the saved model.pt file")
    p.add_argument("--tokenizer_path", default="gpt2", help="Path to the saved tokenizer directory (output by train.py's --save_prefix_tokenizer)")

    # Model Args (must match the trained model's architecture)
    p.add_argument("--embed_dim", type=int, default=768, help="Model embedding dimension (must match trained model)")
    p.add_argument("--n_layers", type=int, default=2, help="Number of Transformer encoder layers (must match trained model)")
    p.add_argument("--n_heads", type=int, default=4, help="Number of attention heads (must match trained model)")
    p.add_argument("--dropout", type=float, default=0.2, help="Dropout rate (must match trained model for architecture consistency)")
    p.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length for tokenizer (must match training)")

    # Testing Args
    p.add_argument("--bsz", type=int, default=1, help="Batch size (number of groups per batch)")
    p.add_argument("--fp16", action="store_true", help="Use Automatic Mixed Precision (AMP) - CUDA only")
    p.add_argument("--device", default="auto", help="Device ('auto', 'cuda', 'cpu', 'mps')")
    p.add_argument("--num_workers", type=int, default=2, help="Number of dataloader workers")

    args = p.parse_args()
    main(args)