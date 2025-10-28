import torch
import json
from typing import List
import logging
import os
from utils.argument import args
from utils import util_loader
from utils import helper

# ----- Configure logging -----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------- Step 3 funcs ------------------------------------------------------
# ---- Step 1: Prepare prompt ----
def format_prompt(caption, class_list):
    """Format captions into a prompt asking for exact description and list of class."""
    return f'''
Task: Based on the image description, determine the category for the "{args.prompt_label}" in the image. You must choose one option from this list: {class_list}

Image description: "{caption}"
Your response:
'''

# --- Build chat prompt using messages format ---
def build_chat_prompt(model, tokenizer, system_prompt, class_list, captions):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": format_prompt(captions, class_list)}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# --- Step 2: Query LLaMA model ---
def query_llm_batch(model, tokenizer, prompts: List[str], max_new_tokens: int = 1024) -> List[str]:
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
        response = tokenizer.decode(output, skip_special_tokens=True)
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
            response = "Unknown"
        
        responses.append(response)
    
    return responses

# ------ Run flow ------
def main(model, tokenizer, inference_batch_size: int = 16):
    """
    Classification and save results.
    
    Args:
        model, tokenizer: LLM
        inference_batch_size: Number of descriptions to process simultaneously
    """
    
    # Load data
    input_path=args.step1_result_path
    output_path=args.step3_result_path

    data = helper.load_data(input_path)
    logger.info(f"Loaded {len(data)} items")
    
    # Process in batches
    logger.info("Starting LLM inference...")
    for i in range(0, len(data), inference_batch_size):
        batch = data[i:i+inference_batch_size]
        
        # Build prompts for this batch
        class_list = helper.extract_categories(args.step2b_result_path)
        #class_list = ["landscape", "human", "animal", "plant", "inanimate object", "unknown"]
        system_prompt = helper.read_file_to_string(args.step3_prompt_path)
        system_prompt = system_prompt.replace("[__CLASSES__]", str(class_list))
        system_prompt = system_prompt.replace("[__CRITERION__]", str((args.prompt_label).lower()))

        prompts = [
            build_chat_prompt(model, tokenizer, system_prompt, class_list, item["description"]) for item in batch]
        
        print(f"Processing batch {i // inference_batch_size + 1}/{(len(data) + inference_batch_size - 1) // inference_batch_size}")
        
        # Run batch inference
        responses = query_llm_batch(model, tokenizer, prompts, max_new_tokens=100)
        
        # Update descriptions with LLM responses
        for j, response in enumerate(responses):
            batch[j]["class"] = response
        
        print(f"  - Completed {min(i + inference_batch_size, len(data))}/{len(data)} items\n")
    
    # Save results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    logger.info(f"Results saved to {output_path}")

# Usage
if __name__ == "__main__":
    # ----- Run flow -----

    model, tokenizer = util_loader.load_llm(args.llama_ver)

    main(model=model, tokenizer=tokenizer, inference_batch_size=64)