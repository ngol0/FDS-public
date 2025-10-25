import torch
import os
import json
from typing import List, Dict
import logging
import sys, os
from dotenv import load_dotenv, find_dotenv
from utils.argument import args
from utils import util_loader


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
def load_data(jsonl_path: str) -> List[Dict]:
    """Load data from JSONL file."""
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Read from from txt file
def read_file_to_string(filename):
    with open(filename, 'r') as file:
        content = file.read()
    return content

# ------------------------- Step 2 funcs ------------------------------------------------------
# ---- Step 1: Prepare captions in batch----
def format_prompt(caption: List[str]) -> str:
    """Format captions into a prompt asking for exact description."""
    return f'''
    Image description: "{caption}" 
    Your response:
'''

# --- Build chat prompt using messages format ---
def build_chat_prompt(model, tokenizer, system_prompt: str, captions: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": format_prompt(captions)}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# --- Step 2: Query LLaMA model ---
def query_llm_batch(model, tokenizer, prompts: List[str], max_new_tokens: int = 512) -> List[str]:
    """Process multiple prompts in one batch."""
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
    
    responses = []
    for output in outputs:
        response = tokenizer.decode(output, skip_special_tokens=True)
        print("RESPONSE: ", response)
        marker = "assistant"  
        idx = response.find(marker)
        if idx != -1:
            responses.append(response[idx + len(marker):].strip())
        else:
            responses.append(response.strip())
    
    return responses

# ------ Run flow ------
def main(model, tokenizer, inference_batch_size: int = 16):
    """
    Process all descriptions and save results.
    
    Args:
        model, tokenizer: LLM
        inference_batch_size: Number of descriptions to process simultaneously
    """
    
    # Load data
    input_path=args.step1_result_path
    output_path=args.step2a_result_path

    data = load_data(input_path)
    logger.info(f"Loaded {len(data)} items")
    
    system_prompt = read_file_to_string(args.step2a_prompt_path)
    
    # Process in batches
    logger.info("Starting LLM inference...")
    for i in range(0, len(data), inference_batch_size):
        batch = data[i:i+inference_batch_size]
        
        # Build prompts for this batch
        prompts = [
            build_chat_prompt(model, tokenizer, system_prompt, item["description"]) for item in batch]
        
        print(f"Processing batch {i // inference_batch_size + 1}/{(len(data) + inference_batch_size - 1) // inference_batch_size}")
        
        # Run batch inference
        responses = query_llm_batch(model, tokenizer, prompts, max_new_tokens=100)
        
        # Update descriptions with LLM responses
        for j, response in enumerate(responses):
            batch[j]["label"] = response
        
        print(f"  - Completed {min(i + inference_batch_size, len(data))}/{len(data)} items\n")
    
    # Save results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    logger.info(f"Results saved to {output_path}")

# Usage
if __name__ == "__main__":
    model, tokenizer = util_loader.load_llm(args.llama_ver)

    # ----- Run flow -----
    main(model=model, tokenizer=tokenizer, inference_batch_size=64)