import json
import re
import argparse
from datasets import load_dataset
from tqdm import tqdm
import os

def extract_numerical_answer_from_gsm8k_solution(answer_string):
    """
    Extracts the numerical value from the GSM8K answer string.
    """
    try:
        boxed_match = re.search(r"\\boxed\{([\d,.-]+)\}", answer_string)
        if boxed_match:
            ans_str = boxed_match.group(1).replace(",", "").strip()
            return float(ans_str)

        parts = answer_string.split('####')
        if len(parts) > 1:
            ans_str = parts[-1].replace(",", "").strip()
            number_match = re.match(r"([\d.-]+)", ans_str)
            if number_match:
                return float(number_match.group(1))
        
        numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", answer_string)
        if numbers:
            last_num_str = numbers[-1].replace(",", "")
            return float(last_num_str)
            
        # print(f"Warning: Could not parse numerical true answer from: {answer_string}")
        return None
    except ValueError:
        # print(f"Warning: ValueError parsing numerical true answer from '{answer_string}'")
        return None
    except Exception:
        # print(f"Warning: Unexpected error parsing numerical true answer from '{answer_string}'")
        return None

def extract_numerical_answer_from_gentext(text):
    """
    Extracts the numerical answer from the generated text.
    """
    try:
        boxed_match = re.search(r"\\boxed\{([\d,.-]+)\}", text)
        if boxed_match:
            ans_str = boxed_match.group(1).replace(",", "").strip()
            return float(ans_str)

        numbers = re.findall(r"[-+]?\d{1,3}(?:,\d{3})*\.\d+|[-+]?\.\d+|[-+]?\d{1,3}(?:,\d{3})*|[-+]?\d+", text)
        
        if numbers:
            valid_numbers = []
            for num_str in numbers:
                try:
                    cleaned_num_str = num_str.replace(",", "")
                    valid_numbers.append(float(cleaned_num_str))
                except ValueError:
                    continue
            
            if valid_numbers:
                return valid_numbers[-1]
        return None
    except ValueError:
        # print(f"Warning: ValueError parsing generated numerical answer from text.")
        return None
    except Exception:
        # print(f"Warning: Unexpected error parsing generated numerical answer.")
        return None

def correct_labels_in_file(input_file_path, output_file_path):
    """
    Reads entries from input_file_path, corrects their labels based on numerical match
    with GSM8K ground truth, and writes to output_file_path.
    """
    print("Loading GSM8K dataset from Hugging Face...")
    try:
        # Determine which split to load based on the input filename
        if "train" in input_file_path.lower():
            gsm8k_split_name = "train"
        elif "test" in input_file_path.lower():
            gsm8k_split_name = "test"
        else:
            print(f"Error: Cannot determine GSM8K split (train/test) from filename: {input_file_path}")
            print("Please ensure 'train' or 'test' is in the input filename.")
            return
            
        print(f"Attempting to load GSM8K '{gsm8k_split_name}' split...")
        gsm8k_dataset = load_dataset("gsm8k", trust_remote_code=True)
        
        if gsm8k_split_name not in gsm8k_dataset:
            print(f"Error: Split '{gsm8k_split_name}' not found in loaded GSM8K dataset. Available splits: {list(gsm8k_dataset.keys())}")
            # Fallback or specific loading if 'default' was the issue previously
            if 'default' in gsm8k_dataset and gsm8k_split_name in gsm8k_dataset['default']: # This is unlikely structure
                 gsm8k_target_split_data = gsm8k_dataset['default'][gsm8k_split_name]
            else: # Try loading the specific split if the main load was too general
                try:
                    split_dataset = load_dataset("gsm8k", name=None, split=gsm8k_split_name, trust_remote_code=True)
                    gsm8k_target_split_data = list(split_dataset) # Convert to list for indexing
                except Exception as e_split:
                    print(f"Failed to load specific split '{gsm8k_split_name}' directly: {e_split}")
                    return
        else:
            gsm8k_target_split_data = list(gsm8k_dataset[gsm8k_split_name]) # Convert to list for indexing

        # Create a dictionary for quick lookup if question_id is reliable
        # Assuming question_id in the file is the 0-based index into the corresponding GSM8K split
        gsm8k_samples_map = {i: sample for i, sample in enumerate(gsm8k_target_split_data)}
        print(f"Loaded {len(gsm8k_samples_map)} samples from GSM8K '{gsm8k_split_name}' split.")

    except Exception as e:
        print(f"Error loading GSM8K dataset: {e}")
        return

    corrected_entries_count = 0
    total_entries = 0
    unparsed_true_answers = 0
    unparsed_gen_answers = 0
    missing_gsm8k_entries = 0

    print(f"\nProcessing input file: {input_file_path}")
    print(f"Writing corrected entries to: {output_file_path}")

    with open(input_file_path, 'r', encoding='utf-8') as infile, \
         open(output_file_path, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(tqdm(infile, desc=f"Correcting labels for {os.path.basename(input_file_path)}")):
            try:
                entry = json.loads(line)
                total_entries += 1

                question_id = entry.get("question_id")
                gen_text = entry.get("gen_text", "")

                if question_id is None:
                    print(f"Warning: Line {line_num+1} missing 'question_id'. Copying as is.")
                    outfile.write(json.dumps(entry) + "\n")
                    continue

                gsm8k_sample = gsm8k_samples_map.get(question_id)
                if gsm8k_sample is None:
                    # print(f"Warning: No GSM8K sample found for question_id {question_id} (line {line_num+1}). Original label kept.")
                    missing_gsm8k_entries += 1
                    # Keep original label or set to 0 if missing? For now, keep original if present.
                    # entry['label'] = entry.get('label', 0) # Default to 0 if no true answer
                    outfile.write(json.dumps(entry) + "\n")
                    continue
                
                true_answer_solution_str = gsm8k_sample['answer']
                true_numerical_answer = extract_numerical_answer_from_gsm8k_solution(true_answer_solution_str)
                
                if true_numerical_answer is None:
                    unparsed_true_answers += 1
                    entry['label'] = 0 # Cannot verify, assume incorrect
                    outfile.write(json.dumps(entry) + "\n")
                    continue

                llama3_numerical_answer = extract_numerical_answer_from_gentext(gen_text)
                if llama3_numerical_answer is None:
                    unparsed_gen_answers += 1
                    entry['label'] = 0 # Cannot extract model's answer, assume incorrect
                    outfile.write(json.dumps(entry) + "\n")
                    continue
                
                new_label = 0
                if abs(llama3_numerical_answer - true_numerical_answer) < 1e-4: # Epsilon for float comparison
                    new_label = 1
                
                if 'label' not in entry or entry['label'] != new_label:
                    corrected_entries_count +=1
                entry['label'] = new_label
                outfile.write(json.dumps(entry) + "\n")

            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed JSON on line {line_num + 1} in {input_file_path}")
                # Optionally write the original malformed line to output if desired
                # outfile.write(line) 
            except Exception as e:
                print(f"Error processing line {line_num+1}: {e}")
                # Write original entry if processing fails for other reasons
                try:
                    outfile.write(json.dumps(entry) + "\n")
                except: # If entry itself is problematic from parsing
                    outfile.write(line)


    print("\n--- Label Correction Summary ---")
    print(f"Total entries processed from '{os.path.basename(input_file_path)}': {total_entries}")
    print(f"Labels potentially changed/set: {corrected_entries_count}")
    if unparsed_true_answers > 0:
        print(f"Warnings: Could not parse true numerical answer for {unparsed_true_answers} GSM8K entries (labeled as 0).")
    if unparsed_gen_answers > 0:
        print(f"Warnings: Could not parse Llama3 numerical answer from gen_text for {unparsed_gen_answers} entries (labeled as 0).")
    if missing_gsm8k_entries > 0:
        print(f"Warnings: Original GSM8K entry not found for {missing_gsm8k_entries} question_ids (original label kept if present).")
    print(f"Corrected file saved to: {output_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Corrects labels in Llama3 GSM8K output files based on numerical match.")
    parser.add_argument("input_file", help="Path to the input .jsonl file.")
    parser.add_argument("output_file", help="Path to save the output .jsonl file with corrected labels.")
    
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
    else:
        correct_labels_in_file(args.input_file, args.output_file)
