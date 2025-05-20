# dataset.py
import json
import random
from collections import defaultdict
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

class BaseChunkDS(Dataset):
    """Base class for loading and tokenizing grouped data."""
    def __init__(self, tokenizer, max_length, cls_id, pad_id, path=None, q2cands_data=None, dataset_name=""):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cls_id = cls_id
        self.pad_id = pad_id # Store pad_id
        self.groups = []
        self.q2cands = defaultdict(list)

        if q2cands_data:
            self.q2cands = q2cands_data
            print(f"Initialized '{dataset_name}' with pre-loaded q2cands: {len(self.q2cands)} unique questions.")
        elif path:
            print(f"Reading data for '{dataset_name}' set from {path}...")
            line_count = 0
            try:
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        try:
                            ex = json.loads(line)
                            if "question" in ex and "label" in ex and ("gen_text" in ex or "generated_full_text" in ex):
                                self.q2cands[ex["question"]].append(ex)
                            else:
                                print(f"Warning: Skipping malformed line {line_count+1} in {path}. Missing keys or text field.")
                        except json.JSONDecodeError:
                            print(f"Warning: Skipping malformed JSON on line {line_count+1} in {path}.")
                        line_count += 1
                print(f"Finished reading {dataset_name}. Found {len(self.q2cands)} unique questions from {line_count} lines.")
            except FileNotFoundError:
                print(f"Error: Data file not found at {path}")
                raise
            except Exception as e:
                print(f"An error occurred while reading {path}: {e}")
                raise
        else:
            print(f"Warning: Initializing '{dataset_name}' with no path and no pre-loaded q2cands. Dataset will be empty.")

    def _process_and_tokenize(self):
        if not self.q2cands:
            print(f"No q2cands data to tokenize for this dataset part. Groups will be empty.")
            self.groups = []
            return

        print(f"Tokenizing candidate answers for {len(self.q2cands)} questions...")
        num_skipped_encoding = 0
        for q, cands in tqdm(self.q2cands.items(), desc="Tokenizing Groups", total=len(self.q2cands), unit="group"):
            enc_grp = []
            group_meta = {"has_correct": False}
            sep_token_str = self.tokenizer.eos_token
            if sep_token_str is None:
                sep_token_str = "\n"

            for e in cands:
                if e["label"] == 1:
                    group_meta["has_correct"] = True
                try:
                    answer_text = e.get("gen_text") or e.get("generated_full_text")
                    if answer_text is None:
                        continue
                    combined_text = f"{q}{sep_token_str}{answer_text}"
                    ids = self.tokenizer.encode(
                        combined_text,
                        add_special_tokens=False,
                        truncation=True,
                        max_length=self.max_length - 1, # -1 for CLS
                    )
                    ids_with_cls = [self.cls_id] + ids
                    enc_grp.append({
                        "ids": torch.tensor(ids_with_cls, dtype=torch.long),
                        "lab": torch.tensor(e["label"], dtype=torch.float),
                    })
                except Exception as encode_err:
                    num_skipped_encoding += 1
                    continue
            
            if enc_grp:
                self.groups.append({"candidates": enc_grp, "meta": group_meta})

        print(f"Finished tokenizing. Processed {len(self.groups)} groups.")
        if num_skipped_encoding > 0:
            print(f"Warning: Skipped {num_skipped_encoding} candidates due to encoding errors during processing.")

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        if not self.groups:
            raise IndexError("Dataset is empty or groups not processed yet.")
        return self.groups[idx]["candidates"]


class TrainValChunkDS(BaseChunkDS):
    """Dataset for Training/Validation: Uses q2cands, filters, tokenizes, then splits."""
    def __init__(self, tokenizer, max_length, cls_id, pad_id, q2cands_data, split="train", holdout=0.2, dataset_name_log_prefix=""):
        super().__init__(tokenizer, max_length, cls_id, pad_id, q2cands_data=q2cands_data, dataset_name=f"{dataset_name_log_prefix}{split}")

        filtered_q2cands_after_diversity = {}
        if self.q2cands:
            print(f"Filtering groups for diversity (at least one positive and one negative label) for {split} split...")
            for q, cands_list in self.q2cands.items():
                has_pos = any(e["label"] == 1 for e in cands_list)
                has_neg = any(e["label"] == 0 for e in cands_list)
                if has_pos and has_neg:
                    filtered_q2cands_after_diversity[q] = cands_list
            print(f"Filtered for diversity: Kept {len(filtered_q2cands_after_diversity)} / {len(self.q2cands)} groups.")
            self.q2cands = filtered_q2cands_after_diversity
        else:
            print(f"No q2cands data to filter for {split} split.")
            self.q2cands = {}

        if not self.q2cands:
            print(f"Warning: No groups remaining after filtering for diversity for {split} split. Dataset part will be empty.")
            self.groups = []
        else:
            self._process_and_tokenize()
        
        random.seed(42)
        all_groups_temp = list(self.groups)
        random.shuffle(all_groups_temp)

        if len(all_groups_temp) > 0:
            cut = int((1.0 - holdout) * len(all_groups_temp))
            if split == "train":
                self.groups = all_groups_temp[:cut]
                if cut == 0:
                    print(f"Warning: Training split for {dataset_name_log_prefix} resulted in 0 groups with holdout {holdout}.")
            elif split == "val":
                self.groups = all_groups_temp[cut:]
                if cut == len(all_groups_temp):
                     print(f"Warning: Validation split for {dataset_name_log_prefix} resulted in 0 groups with holdout {holdout}.")
            else: # Should not happen with current usage
                self.groups = []
        else:
            self.groups = []

        print(f"Split '{split}' (holdout={holdout}) for {dataset_name_log_prefix}: {len(self.groups)} groups.")
        if len(self.groups) == 0 and len(filtered_q2cands_after_diversity) > 0 :
             print(f"Warning: The '{split}' split for {dataset_name_log_prefix} has 0 groups after splitting, even though groups passed filtering.")


class TestChunkDS(BaseChunkDS):
    """Dataset for Testing: Loads from path, tokenizes all groups (no filtering/splitting)."""
    def __init__(self, tokenizer, max_length, cls_id, pad_id, path, dataset_name="test_dataset"):
        super().__init__(tokenizer, max_length, cls_id, pad_id, path=path, dataset_name=dataset_name)
        if self.q2cands:
            self._process_and_tokenize()
        else:
            print(f"No data loaded for {dataset_name} from {path}. Test set will be empty.")
            self.groups = []


def collate_fn(batch, pad_id):
    idsL, maskL, labL = [], [], []
    for grp in batch:
        if not isinstance(grp, (list, tuple)) or not grp: continue
        valid_elements = [e for e in grp if isinstance(e, dict) and "ids" in e and "lab" in e]
        if not valid_elements: continue
        
        ids  = [e["ids"] for e in valid_elements]
        labs = [e["lab"] for e in valid_elements]
        if not ids: continue

        ids_cpu = [t.cpu() for t in ids]
        try:
            pad_val = pad_sequence(ids_cpu, batch_first=True, padding_value=pad_id)
            mask_val = (pad_val != pad_id).long()
            idsL.append(pad_val)
            maskL.append(mask_val)
            labL.append(torch.stack(labs))
        except Exception as e:
            print(f"Error during padding/collating: {e}. Skipping group.")
            continue
    return idsL, maskL, labL