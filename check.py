import json
import re
import argparse
from datasets import load_dataset
from tqdm import tqdm

def extract_numerical_answer_from_gsm8k_solution(answer_string):
    """
    Extracts the numerical value from the GSM8K answer string.
    The standard format is "#### <answer>".
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
            
        print(f"Warning: Could not parse numerical true answer from: {answer_string}")
        return None
    except ValueError as e:
        print(f"Warning: ValueError parsing numerical true answer from '{answer_string}': {e}")
        return None
    except Exception as e:
        print(f"Warning: Unexpected error parsing numerical true answer from '{answer_string}': {e}")
        return None

def extract_final_answer_string_from_gsm8k_solution(answer_string):
    """
    Extracts the final answer string (the part after '#### ') from the GSM8K answer string.
    Example: "#### 70" -> "70"
    """
    try:
        # Try to get the most precise part first (boxed)
        boxed_match = re.search(r"\\boxed\{([\d,.-]+)\}", answer_string)
        if boxed_match:
            return boxed_match.group(1).replace(",", "").strip()

        # Then try the '####' marker
        parts = answer_string.split('####')
        if len(parts) > 1:
            final_part = parts[-1].strip()
            # Extract just the number part if there's trailing text like "dollars"
            match = re.match(r"([\d,.-]+)", final_part)
            if match:
                return match.group(1).replace(",", "").strip()
            return final_part # Fallback to the whole part after ####
        
        # As a last resort, if no markers, try to find the last number and return it as string
        numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", answer_string)
        if numbers:
            return numbers[-1].replace(",", "").strip()

        print(f"Warning: Could not extract final answer string from: {answer_string}")
        return None
    except Exception as e:
        print(f"Warning: Unexpected error extracting final answer string from '{answer_string}': {e}")
        return None


def extract_numerical_answer_from_gentext(text):
    """
    Extracts the numerical answer from the generated text.
    Prioritizes \boxed{} then tries to find the last number.
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
    except ValueError as e:
        print(f"Warning: ValueError parsing generated numerical answer from text: {e}")
        return None
    except Exception as e:
        print(f"Warning: Unexpected error parsing generated numerical answer: {e}")
        return None

def verify_labels(llama3_output_file_path):
    """
    Verifies labels in the Llama3 output file against true GSM8K answers.
    """
    print("Loading GSM8K test dataset from Hugging Face...")
    try:
        # Using name='default' as per the error message, or omitting name if 'default' is the actual default.
        # Let's try omitting 'name' first, then 'default' if it still fails.
        # Most GSM8K configs are 'main' or 'socratic'. If 'default' is what user's env sees, use that.
        gsm8k_dataset = load_dataset("gsm8k", trust_remote_code=True) # Try without name first
        # If the above fails with the same error, uncomment the next line and comment the one above:
        # gsm8k_dataset = load_dataset("gsm8k", name="default", trust_remote_code=True)
        
        gsm8k_test_samples = {i: sample for i, sample in enumerate(gsm8k_dataset['test'])}
        print(f"Loaded {len(gsm8k_test_samples)} samples from GSM8K test set.")
    except Exception as e:
        print(f"Error loading GSM8K dataset: {e}")
        print("Please ensure you have an internet connection and the 'datasets' library is installed correctly.")
        print("If the error persists, try specifying `name=\"default\"` in `load_dataset`.")
        return

    total_entries = 0
    llama3_numerical_match_correct_count = 0
    llama3_contains_true_answer_string_correct_count = 0
    original_label_claims_correct_count = 0
    original_label_verified_by_numerical_match_count = 0
    original_label_verified_by_containment_count = 0
    
    unparsed_gentext_numerical_answers = 0
    unparsed_true_numerical_answers = 0
    unparsed_true_answer_strings = 0
    mismatched_question_ids = 0

    print(f"\nProcessing Llama3 output file: {llama3_output_file_path}")
    try:
        with open(llama3_output_file_path, 'r', encoding='utf-8') as f_llama3:
            for line_num, line in enumerate(tqdm(f_llama3, desc="Verifying entries")):
                try:
                    json_entry = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed JSON on line {line_num + 1}")
                    continue

                total_entries += 1
                
                question_id = json_entry.get('question_id')
                original_label = json_entry.get('label')
                gen_text = json_entry.get('gen_text', "")

                if question_id is None or original_label is None:
                    print(f"Warning: Missing 'question_id' or 'label' in entry on line {line_num + 1}. Skipping.")
                    continue

                if original_label == 1:
                    original_label_claims_correct_count += 1

                gsm8k_sample = gsm8k_test_samples.get(question_id)
                if gsm8k_sample is None:
                    print(f"Warning: No GSM8K sample found for question_id {question_id} (line {line_num + 1}). Skipping.")
                    mismatched_question_ids += 1
                    continue
                
                true_answer_solution_str = gsm8k_sample['answer']
                
                # 1. Numerical comparison
                true_numerical_answer = extract_numerical_answer_from_gsm8k_solution(true_answer_solution_str)
                if true_numerical_answer is None:
                    unparsed_true_numerical_answers += 1
                
                llama3_numerical_answer = extract_numerical_answer_from_gentext(gen_text)
                if llama3_numerical_answer is None:
                    unparsed_gentext_numerical_answers +=1

                is_llama3_correct_by_numerical_match = False
                if llama3_numerical_answer is not None and true_numerical_answer is not None:
                    if abs(llama3_numerical_answer - true_numerical_answer) < 1e-4:
                        is_llama3_correct_by_numerical_match = True
                
                if is_llama3_correct_by_numerical_match:
                    llama3_numerical_match_correct_count += 1

                # Verify original label against numerical match
                original_label_predicted_llama3_correct = (original_label == 1)
                if original_label_predicted_llama3_correct == is_llama3_correct_by_numerical_match:
                    original_label_verified_by_numerical_match_count += 1

                # 2. "Contains true answer string" comparison
                true_final_answer_string = extract_final_answer_string_from_gsm8k_solution(true_answer_solution_str)
                if true_final_answer_string is None:
                    unparsed_true_answer_strings +=1
                
                is_llama3_correct_by_containment = False
                if gen_text and true_final_answer_string:
                    # Ensure we are checking for the number as a whole word or clearly demarcated
                    # to avoid partial matches (e.g., "10" in "100").
                    # Using regex for word boundaries: \btrue_final_answer_string\b
                    # Need to escape true_final_answer_string if it contains regex special chars,
                    # but for numbers, it's usually fine.
                    pattern = r"\b" + re.escape(true_final_answer_string) + r"\b"
                    if re.search(pattern, gen_text):
                        is_llama3_correct_by_containment = True
                    # Fallback for cases where \b might not be ideal (e.g. $70)
                    # A simpler check if the above is too strict:
                    elif true_final_answer_string in gen_text:
                         # This is a broader check, consider if it's too lenient.
                         # For example, if true answer is "7" and gen_text has "70".
                         # The regex with \b is generally better for whole numbers.
                         # Let's refine this: if not found with \b, do a simpler check but be wary.
                         # For now, sticking to the regex for better precision.
                         pass # is_llama3_correct_by_containment remains False from \b check

                if is_llama3_correct_by_containment:
                    llama3_contains_true_answer_string_correct_count += 1
                
                # Verify original label against containment
                if original_label_predicted_llama3_correct == is_llama3_correct_by_containment:
                    original_label_verified_by_containment_count += 1
                    
    except FileNotFoundError:
        print(f"Error: Llama3 output file not found at {llama3_output_file_path}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n--- Verification Results ---")
    if total_entries == 0:
        print("No entries processed.")
        return

    # Calculate accuracies
    def get_accuracy(correct_count, total):
        return (correct_count / total) * 100 if total > 0 else 0

    llama3_accuracy_numerical_check = get_accuracy(llama3_numerical_match_correct_count, total_entries)
    llama3_accuracy_containment_check = get_accuracy(llama3_contains_true_answer_string_correct_count, total_entries)
    original_label_stated_accuracy = get_accuracy(original_label_claims_correct_count, total_entries)
    original_label_verif_acc_numerical = get_accuracy(original_label_verified_by_numerical_match_count, total_entries)
    original_label_verif_acc_containment = get_accuracy(original_label_verified_by_containment_count, total_entries)


    print(f"Total entries processed: {total_entries}")
    if mismatched_question_ids > 0:
        print(f"Entries skipped due to mismatched/missing question_id: {mismatched_question_ids}")
    if unparsed_true_numerical_answers > 0:
        print(f"Warnings: Could not parse true numerical answer for {unparsed_true_numerical_answers} GSM8K entries.")
    if unparsed_true_answer_strings > 0:
        print(f"Warnings: Could not parse true final answer string for {unparsed_true_answer_strings} GSM8K entries.")
    if unparsed_gentext_numerical_answers > 0:
        print(f"Warnings: Could not parse Llama3 numerical answer from gen_text for {unparsed_gentext_numerical_answers} entries.")
    
    print(f"\n1. Llama3 Accuracy (Numerical Match):")
    print(f"   - Correct by numerical match: {llama3_numerical_match_correct_count}/{total_entries}")
    print(f"   - Accuracy: {llama3_accuracy_numerical_check:.2f}%")

    print(f"\n2. Llama3 Accuracy (gen_text CONTAINS true final answer string):")
    print(f"   - Correct by containment: {llama3_contains_true_answer_string_correct_count}/{total_entries}")
    print(f"   - Accuracy: {llama3_accuracy_containment_check:.2f}%")

    print(f"\n3. Original Label Stated Accuracy (from input file's 'label' field):")
    print(f"   - Claimed correct by original label: {original_label_claims_correct_count}/{total_entries}")
    print(f"   - Accuracy: {original_label_stated_accuracy:.2f}%")

    print(f"\n4. Original Label Verification (vs. Numerical Match):")
    print(f"   - Original labels accurately reflecting numerical correctness: {original_label_verified_by_numerical_match_count}/{total_entries}")
    print(f"   - Accuracy: {original_label_verif_acc_numerical:.2f}%")

    print(f"\n5. Original Label Verification (vs. Containment Check):")
    print(f"   - Original labels accurately reflecting containment correctness: {original_label_verified_by_containment_count}/{total_entries}")
    print(f"   - Accuracy: {original_label_verif_acc_containment:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify labels in Llama3 GSM8K output against true answers.")
    parser.add_argument("llama3_output_file", 
                        help="Path to the .jsonl file containing Llama3 outputs (with 'question_id', 'label', 'gen_text').")
    
    args = parser.parse_args()
    verify_labels(args.llama3_output_file)


# python check.py /mnt/shared/ericjiang/ebm_cot/need_modification/results_gsm8k_llama3_test_n4_temp0.7_p0.9_test.jsonl
