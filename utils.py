# utils.py
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from tqdm import tqdm
from contextlib import nullcontext
from collections import defaultdict
import json


def get_device_and_amp_helpers(device_arg="auto", fp16_arg=False):
    if device_arg == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        elif torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")
    else:
        dev = torch.device(device_arg)

    use_amp = fp16_arg and dev.type == "cuda"
    autocaster = torch.cuda.amp.autocast if use_amp else nullcontext
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    print(f"Using device: {dev}")
    print(f"Using FP16 (AMP): {use_amp}")
    return dev, use_amp, autocaster, scaler

def setup_tokenizer(tokenizer_name_or_path="gpt2", max_length=4096):
    print(f"Loading tokenizer from: {tokenizer_name_or_path}")
    try:
        tok = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    except Exception as e:
        print(f"Error loading tokenizer from '{tokenizer_name_or_path}': {e}")
        exit(1)

    print(f"Setting tokenizer max length to: {max_length}")
    tok.model_max_length = max_length

    pad_id = None
    if tok.pad_token_id is None:
        if tok.eos_token_id is not None:
            print("Tokenizer missing pad token; using EOS token as pad token.")
            tok.pad_token = tok.eos_token
            pad_id = tok.eos_token_id
        elif tok.unk_token_id is not None:
            print("Tokenizer missing pad and EOS token; using UNK token as pad token.")
            tok.pad_token = tok.unk_token
            pad_id = tok.unk_token_id
        else:
            print("Tokenizer missing pad, EOS, and UNK token; adding new pad token '[PAD]'.")
            tok.add_special_tokens({'pad_token': '[PAD]'})
            pad_id = tok.convert_tokens_to_ids('[PAD]')
    else:
        pad_id = tok.pad_token_id

    cls_id = None
    if hasattr(tok, "bos_token_id") and tok.bos_token_id is not None:
        cls_id = tok.bos_token_id
        print(f"Using BOS token ID ({cls_id}) as CLS_ID (effective).")
    elif hasattr(tok, "cls_token_id") and tok.cls_token_id is not None:
        cls_id = tok.cls_token_id
        print(f"Using CLS token ID ({cls_id}) as CLS_ID (effective).")
    elif tok.eos_token_id is not None:
        cls_id = tok.eos_token_id
        print(f"Warning: BOS/CLS token not found. Using EOS token ID ({cls_id}) as CLS_ID (effective).")
    else:
        print("Error: Tokenizer has no BOS, CLS, or EOS token. Cannot determine CLS_ID for sequence representation.")
        exit(1)

    if pad_id is None:
        print("Error: Failed to determine PAD ID.")
        exit(1)

    print(f"Tokenizer loaded. Vocab size: {len(tok)}. Pad ID: {pad_id}, CLS ID (effective): {cls_id}, Max Length: {tok.model_max_length}")
    return tok, pad_id, cls_id

def bradley_terry_loss(e: torch.Tensor, l: torch.Tensor):
    pos_indices = torch.where(l == 1)[0]
    neg_indices = torch.where(l == 0)[0]
    if len(pos_indices) == 0 or len(neg_indices) == 0:
        return None 
    pos_scores = e[pos_indices]
    neg_scores = e[neg_indices]
    # pairwise differences: (pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0))
    # E_good - E_bad. We want E_good to be smaller.
    # So, we want E_bad - E_good to be positive.
    # The loss is log(1 + exp(-(E_bad - E_good))) = log(1 + exp(E_good - E_bad))
    # which is softplus(E_good - E_bad)
    energy_diffs = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0) 
    loss_matrix = F.softplus(energy_diffs) 
    return loss_matrix.mean()

@torch.no_grad()
def evaluate(model, loader, device, autocaster, eval_type="Validation"):
    model.eval()
    total_groups = 0
    naive_correct = 0
    ebm_correct = 0
    
    if loader is None:
        print(f"Skipping {eval_type} evaluation as data loader is None.")
        return 0.0, 0.0

    pbar_eval = tqdm(loader, desc=f"Evaluating ({eval_type})", leave=False, unit="batch")
    for idsL, maskL, labL in pbar_eval:
        for ids, mask, lab in zip(idsL, maskL, labL):
            if ids.numel() == 0 or lab.numel() == 0 : continue
            ids, mask, lab = ids.to(device), mask.to(device), lab.to(device)
            with autocaster():
                e = model(ids, mask)
            if e.numel() == 0: continue

            best_cand_idx = torch.argmin(e) # Lower energy is better
            if lab.numel() > best_cand_idx and lab[best_cand_idx].item() == 1:
                ebm_correct += 1
            
            # Naive: assumes the first candidate might be a baseline or a common choice
            # The original code's naive check assumes the first candidate in the list is the one to check.
            # If the input data `e` in `lab[0]` refers to the first candidate of the current group.
            if lab.numel() > 0 and lab[0].item() == 1: 
                naive_correct += 1
            
            total_groups += 1
            current_acc = 100.0 * ebm_correct / total_groups if total_groups > 0 else 0.0
            pbar_eval.set_postfix(acc=f"{current_acc:.2f}%")

    if total_groups == 0:
        print(f"Warning: No groups processed during {eval_type} evaluation for loader.")
        return 0.0, 0.0
    naive_acc = 100.0 * naive_correct / total_groups
    ebm_acc = 100.0 * ebm_correct / total_groups
    return naive_acc, ebm_acc

def load_q2cands_from_jsonl(path, description="data"):
    q2cands = defaultdict(list)
    if path and os.path.exists(path) and os.path.isfile(path):
        print(f"Reading {description} from: {path}")
        line_count = 0
        entries_added_from_file = 0
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        ex = json.loads(line)
                        if "question" in ex and "label" in ex and ("gen_text" in ex or "generated_full_text" in ex):
                            q2cands[ex["question"]].append(ex)
                            entries_added_from_file +=1
                        else:
                            print(f"Warning: Skipping malformed line {line_count+1} in {path}. Missing keys or text field.")
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping malformed JSON on line {line_count+1} in {path}.")
                    line_count += 1
            print(f"Finished reading from {path}. Added {entries_added_from_file} candidate entries from {line_count} lines.")
            print(f"Found {len(q2cands)} unique questions in {path}.")
            return q2cands
        except Exception as e:
            print(f"An error occurred while reading {path}: {e}. Skipping this file.")
            return None # Indicates failure
    elif path:
        print(f"Warning: {description.capitalize()} file specified but not found or not a file: {path}. Skipping.")
    return None # Indicates failure or skip