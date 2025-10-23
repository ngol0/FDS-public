import torch
import os
import json
from datasets import Dataset, load_dataset
from typing import List, Dict, Tuple, Any
import logging
import sys, os
from dotenv import load_dotenv, find_dotenv
from utils.argument import args
from utils import util_loader
from PIL import Image

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset import BaseDataset

# ---- Set up directory path ----
env_path = find_dotenv()
load_dotenv(env_path)
home_path = os.getenv("HOME_PATH")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# print(">>> sys.path includes:", sys.path[-3:])

# ----- Configure logging -----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----- Other utils functions -----------
def read_file_to_string(filename):
    with open(filename, 'r') as file:
        content = file.read()
    return content

# ------- Main VLM pipeline functions -------------
# STEP 1: Prepare batch question, questions passed in, read from txt file
def prepare_batch_questions(data: Dataset,
                            question: str, max_samples: int = None, 
                            skip_grayscale: bool = True) -> List[List[Dict[str, Any]]]:
    """
    Prepare batch questions in the format expected by MiniCPM.
    
    Args:
        data: Dataset containing images and labels
        label_set: List of label names
        question: Question to ask about each image
        max_samples: Maximum number of samples to process (None for all)
        skip_grayscale: Whether to skip grayscale images
        
    Returns:
        List of questions in batch format
    """
    questions = []
    processed_count = 0
    
    # # ---- For TinyImagenet -----
    for i in range(len(data.data)):
        if max_samples and processed_count >= max_samples:
            break
        
        image_path = data.data[i] 
        label = data.labels[i]

        with Image.open(image_path) as image:
            image_copy = image.copy()

        # Skip grayscale images if requested
        if skip_grayscale and image_copy.mode != 'RGB':
            logger.info(f"Skipping grayscale image at index {i}")
            continue
        # else: 
        #     # -------> todo: convert to RGB
        #     image.mode = 'RGB'?
        
        # Prepare question in the required format
        question_batch = [{"role": "user", "content": [image, question]}]
        
        questions.append({
            'question': question_batch,
            'image': image,
            'label': label,
            'index': i,
            'filename': f"{label}_{processed_count:04}.jpg"
        })
        
        processed_count += 1
        
    logger.info(f"Prepared {len(questions)} questions for processing")
    return questions

# STEP 3: Process questions by batch through batch API
def process_questions_batch(questions: List[List[Dict[str, Any]]], model, tokenizer, 
                            temperature: float, sampling: bool) -> List[str]:
    """
    Process questions in true batch mode using MiniCPM's batch API.
    
    Args:
        questions: List of question batches in MiniCPM format
        temperature: Sampling temperature
        sampling: Whether to use sampling
        
    Returns:
        List of response strings
    """
    try:
        # Use the batch processing format: msgs=[msgs, msgs] with image=None
        res = model.chat(
            image=None,
            msgs=questions,  # Pass the list of message batches directly
            tokenizer=tokenizer,
            sampling=sampling,
            temperature=temperature,
            stream=False
        )
        
        responses = res
        
        logger.info(f"Successfully processed batch of {len(questions)} questions")
        return responses
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise e

# ----- Error handling for batch, used only when batch processing failed ------
def process_individually(batch_data: List[Dict[str, Any]], model, tokenizer, 
                         temperature: float, sampling: bool) -> List[Dict[str, str]]:
    """Fallback method to process questions individually."""
    results = []
    
    for item in batch_data:
        try:
            question = item['question']
            image = question[0]['content'][0]
            text_question = question[0]['content'][1]
            
            msgs = [{'role': 'user', 'content': text_question}]
            
            res, context, _ = model.chat(
                image=image,
                msgs=msgs,
                context=None,
                tokenizer=tokenizer,
                sampling=sampling,
                temperature=temperature
            )
            
            results.append({
                "image": item['filename'],
                "label": item['label'],
                "description": res,
                "original_index": item['index']
            })
            
        except Exception as e:
            logger.error(f"Error processing individual item {item['index']}: {e}")
            results.append({
                "image": item['filename'],
                "label": item['label'],
                "description": f"Error: {str(e)}",
                "original_index": item['index']
            })
            
    return results

def process_batch(batch_data: List[Dict[str, Any]], model, tokenizer,
                  temperature: float = 0.7, sampling: bool = True) -> List[Dict[str, str]]:
    """
    Process a batch of questions using MiniCPM's batch API.
    
    Args:
        batch_data: List of prepared question data
        temperature: Sampling temperature
        sampling: Whether to use sampling
        
    Returns:
        List of results with descriptions
    """
    results = []
    
    # Extract questions for batch processing
    questions = [item['question'] for item in batch_data]
    
    try:
        # Process batch using batch API
        responses = process_questions_batch(questions, model, tokenizer, temperature, sampling)
        
        # Ensure there's a right number of responses
        if len(responses) != len(batch_data):
            logger.warning(f"Expected {len(batch_data)} responses, got {len(responses)}")
            # Pad with error messages if needed
            while len(responses) < len(batch_data):
                responses.append("Error: No response received")
        
        # Combine results with metadata
        for i, (batch_item, response) in enumerate(zip(batch_data, responses)):
            results.append({
                "image": batch_item['filename'],
                "label": batch_item['label'],
                "description": response,
                "original_index": batch_item['index']
            })
            
    except Exception as e:
        logger.error(f"Batch processing failed, falling back to individual processing: {e}")
        # Fallback to individual processing
        results = process_individually(batch_data, model, tokenizer, temperature, sampling)
        
    return results

# STEP 5: Save results
def save_to_json(path: str, data: Any) -> None:
    """Save data to JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"Results saved to {path}")

#================================= ENTIRE FLOW FOR STEP 1 ========================================
def main(model, tokenizer, data: BaseDataset,
        max_samples: int = 6, 
        batch_size: int = 4) -> None:
    """
    Main method to process the entire dataset.
    
    Args:
        model, tokenizer: VLM
        max_samples: Maximum number of samples to process
        batch_size: Number of samples to process in each batch
        dataset_split: Dataset split to use
    """
    logger.info("Starting dataset processing...")
    
    # Prepare questions
    question = read_file_to_string(args.step1_prompt_path)
    print("Preparing batch question.... Question: ", question)
    prepared_data = prepare_batch_questions(data, question, max_samples)
    
    # Process in batches
    all_results = []
    for i in range(0, len(prepared_data), batch_size):
        batch = prepared_data[i:i + batch_size]
        batch_num = i//batch_size + 1
        total_batches = (len(prepared_data) + batch_size - 1)//batch_size
        
        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} samples)")
        
        batch_results = process_batch(batch, model, tokenizer)
        all_results.extend(batch_results)
        
        # Log progress
        logger.info(f"Batch {batch_num} complete. Total processed: {len(all_results)}/{len(prepared_data)}")
        
        if batch_num % 5 == 0:  # Log every 5 batches
            logger.info(f"Progress update: {len(all_results)} samples processed so far...")
    
    # Save all results to JSON
    json_path = os.path.expanduser(args.step1_result_path)
    save_to_json(json_path, all_results)
    logger.info(f"Processing complete! Processed {len(all_results)} samples. Saved to {args.output_path}")


if __name__ == "__main__":
    """Main execution function."""

    model, tokenizer = util_loader.load_minicpm()
    data = util_loader.load_data(args.dataset)
    
    # Process dataset
    main(model=model, tokenizer=tokenizer, data=data, max_samples=None, batch_size=50)



