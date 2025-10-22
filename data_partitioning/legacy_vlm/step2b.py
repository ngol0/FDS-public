import torch
import os
import json
from utils import model_loader
from typing import List
import logging
import sys, os
from dotenv import load_dotenv, find_dotenv
from utils.argument import args
from pathlib import Path

# ----- Configure logging -----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- Set up directory path ----
env_path = find_dotenv()
load_dotenv(env_path)
home_path = os.getenv("HOME_PATH")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# print(">>> sys.path includes:", sys.path[-3:])

# --- Load captions from JSONL ---
# --- Step 1: Load captions from JSONL ---
def post_process(json_path: str):
    """
    Load labels from JSON and count their occurrences.
    
    Args:
        json_path: Path to JSON file with format [{"label": "...", ...}, ...]
    
    Returns:
        Dictionary with label counts
    """
    answer_list = {}
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for item in data:
        label = item["label"]
        
        # Count occurrences
        if label not in answer_list:
            answer_list[label] = 1
        else:
            answer_list[label] += 1
    
    return answer_list

# Read from from txt file
def read_file_to_string(filename):
    with open(filename, 'r') as file:
        content = file.read()
    return content

# ------------------------- Step 2 funcs ------------------------------------------------------
# ---- Step 1: Prepare captions in batch----
def format_prompt(labels, num_of_class) -> str:
    """Format captions into a prompt asking for exact description."""
    return f'''
    List of labels: {labels}. 
    Num_classes: {num_of_class}
    Your response:
'''

# --- Build chat prompt using messages format ---
def build_chat_prompt(model, tokenizer, system_prompt: str, labels, num_class) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": format_prompt(labels, num_class)}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# --- Step 2: Query LLaMA model ---
def query_llm(model, tokenizer, prompts: List[str], max_new_tokens: int = 512) -> List[str]:
    """Process one prompt."""
    inputs = tokenizer(
        prompts, 
        return_tensors="pt", 
        truncation=True,
        padding=True,
        return_attention_mask=True
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #print("RESPONSE: ", response)
    marker = "Your response:assistant"
    idx = response.find(marker)
    if idx != -1:
        return response[idx + len(marker):].strip()
    else:
        return "Can't find marker!"

# ------ Run flow ------
def main(model, tokenizer):
    """
    Process all labels and save results.
    """

    # Load data
    input_file=args.step2a_result_path

    # Usage
    label_counts = post_process(input_file)
    #print(label_counts)

    logger.info(f"Loaded {len(label_counts)} items")

    # 
    # threshold
    if args.num_classes == 20:
        threshold = 35
    else:
        threshold = 30
    label_counts = dict(sorted({k: v for k, v in label_counts.items() if v > threshold}.items(), key=lambda item: item[1], reverse=True))

    print("Post-processed dictionary: ", label_counts)
    
    system_prompt = read_file_to_string(args.step2b_prompt_path)
    system_prompt = system_prompt.replace("[__NUM_CLASSES_CLUSTER__]", str(args.num_classes))
    system_prompt = system_prompt.replace("[__LEN__]", str(len(label_counts)))

    prompt = build_chat_prompt(model, tokenizer, system_prompt, label_counts, args.num_classes)
    response = query_llm(model, tokenizer, prompt, max_new_tokens=500)
    print(response)

    # save results
    with open(args.step2b_result_path, 'w', encoding='utf-8') as file:
        file.write(response)
    
    logger.info(f"Results saved to {args.step2b_result_path}")

# Usage
if __name__ == "__main__":
    # ----- Run flow -----
    model = model_loader.llm_model
    tokenizer = model_loader.llm_tokenizer

    main(model=model, tokenizer=tokenizer)