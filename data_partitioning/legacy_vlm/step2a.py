import torch
import os
import json
from typing import List, Dict
import logging
import sys, os
from utils.argument import args
from utils import util_loader
from utils import helper

# ----- Configure logging -----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------- Step 2 funcs ------------------------------------------------------
# ---- Step 1: Prepare captions in batch----
def format_prompt(caption: List[str]) -> str:
    """Format captions into a prompt asking for exact description."""
    return f'''
Task: Read the image description and respond with one word about the "{args.prompt_label}".
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
    tokenizer.padding_side = 'left'
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
            do_sample=False,
            temperature=None,
            top_p=None,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    responses = []
    for output in outputs:
        response = tokenizer.decode(output, skip_special_tokens=True).strip()
        #print("RESPONSE: ", response)
        marker = "assistant"  
        idx = response.find(marker)
        if idx != -1:
            response = response[idx + len(marker):].strip()
            response = helper.clean_response(response)
        else:
           response = response.strip()
    
        print("Cleaned response: ", response)
        # Debug blank responses
        if response == "":
            logger.warning(f"Blank response")
        
        responses.append(response)

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

    data = helper.load_data(input_path)
    logger.info(f"Loaded {len(data)} items")
    
    system_prompt = helper.read_file_to_string(args.step2a_prompt_path)
    system_prompt = system_prompt.replace("[__CRITERION__]", str(args.prompt_label.lower()))
    
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