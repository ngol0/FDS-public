import gc, torch
import os
import json
from datasets import Dataset
from typing import List, Dict, Tuple, Any
import logging
import sys, os
from utils.argument import args
from utils import util_loader
from utils import helper
from PIL import Image
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from dataset import BaseDataset
from utils.constants import INTERNVL3_38B, INTERNVL3_5_8B, MINICPM_26

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# --- Utils for InternVL3 (referenced from Hugging Face: https://huggingface.co/OpenGVLab/InternVL3-38B) ---
# Could ignore these as it's only used for InternVL3 inference
def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# ---- Set up directory path ----
# from dotenv import load_dotenv, find_dotenv
# env_path = find_dotenv()
# load_dotenv(env_path)
# home_path = os.getenv("HOME_PATH")
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# print(">>> sys.path includes:", sys.path[-3:])

# ----- Configure logging -----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def print_gpu_memory():
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

# ------- Main VLM pipeline functions -------------
# STEP 1: Prepare batch question, questions passed in, read from txt file
def prepare_batch_questions(data: Dataset,
                            question: str, max_samples: int = None, 
                            convert_to_RGB: bool = True) -> List[List[Dict[str, Any]]]:
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

    for i in range(len(data.data)):
        if max_samples and processed_count >= max_samples:
            break
        
        image_path = data.data[i] 
        label = data.labels[i]
        
        questions.append({
            'question': question,
            'label': label,
            'index': i,
            'filename': image_path,  # Original path
            #'saved_name': f"{label}_{processed_count:04}.jpg"
        })
        
        processed_count += 1
        
    logger.info(f"Prepared {len(questions)} questions for processing")
    return questions

# STEP 2: Process questions by batch through batch API
# ------------- For MiniCPM ------------------------------
def process_questions_batch_minicpm(questions: List[List[Dict[str, Any]]], model, tokenizer, 
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
def process_batch_minicpm(model, tokenizer, batch_data: List[Dict[str, Any]], 
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
    
    # Load images and reconstruct questions for batch processing
    questions = []
    for item in batch_data:
        # Load the actual image
        pil_image = Image.open(item['filename']).convert('RGB')
        
        # Get question text
        question_text = item['question']
        
        # Reconstruct question with real image
        question_batch = [{"role": "user", "content": [pil_image, question_text]}]
        questions.append(question_batch)
    
    try:
        # Process batch using batch API
        responses = process_questions_batch_minicpm(questions, model, tokenizer, temperature, sampling)
        
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
        logger.error(f"Batch processing failed...: {e}")
        
    return results

# ------------ For InternVL3 --------------------------------
def process_batch_internvl(model, tokenizer, batch_data):
    """Process with InternVL in batch"""
    print_gpu_memory()
    results = []
    generation_config = dict(max_new_tokens=512, do_sample=False)
    pixel_values_list = []
    num_patches_list = []

    # Load and process all images in the batch
    for batch_item in batch_data:    
        # Load image
        pixel_values = load_image(batch_item['filename'], input_size=448, max_num=12).to(torch.bfloat16).cuda()
        
        # Track number of patches
        num_patches_list.append(pixel_values.size(0))
        pixel_values_list.append(pixel_values)

    # Concatenate all pixel values
    pixel_values = torch.cat(pixel_values_list, dim=0).cuda()

    # Extract question text
    question_text = batch_data[0]['question']
    question_with_image = f'<image>\n{question_text}'
    questions = [question_with_image] * len(num_patches_list)
    
    # Batch inference
    with torch.no_grad():
        responses = model.batch_chat(
            tokenizer, 
            pixel_values,
            num_patches_list=num_patches_list,
            questions=questions,
            generation_config=generation_config)

    print_gpu_memory()

    # Combine results with metadata
    for i, (batch_item, response) in enumerate(zip(batch_data, responses)):
        results.append({
            "image": batch_item['filename'],
            "label": batch_item['label'],
            "description": response,
            "original_index": batch_item['index']})

    # Cleanup
    del pixel_values, pixel_values_list
    gc.collect()
    torch.cuda.empty_cache()

    return results

# Save results
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
    all_results = []
    logger.info("Starting dataset processing...")
    question = helper.read_file_to_string(args.step1_prompt_path)
    question = question.replace("[__CRITERION__]", str((args.prompt_label).lower()))

    # Prepare questions
    print("Preparing batch question.... ")
    print(f"Question: {question}")
    prepared_data = prepare_batch_questions(data, question, max_samples)
    
    # Process in batches
    for i in range(0, len(prepared_data), batch_size):
        batch = prepared_data[i:i + batch_size]
        batch_num = i//batch_size + 1
        total_batches = (len(prepared_data) + batch_size - 1)//batch_size
        
        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} samples)")

        # Different inference method for different model
        # todo: refactor this to not use if else
        if args.vlm_model == MINICPM_26:
            batch_results = process_batch_minicpm(model=model, tokenizer=tokenizer, batch_data=batch)
        if args.vlm_model == INTERNVL3_5_8B or args.vlm_model == INTERNVL3_38B:
            batch_results = process_batch_internvl(model=model, tokenizer=tokenizer, batch_data=batch)
    
        all_results.extend(batch_results)
        
        # Log progress
        logger.info(f"Batch {batch_num} complete. Total processed: {len(all_results)}/{len(prepared_data)}")
        
        if batch_num % 5 == 0:  # Log every 5 batches
            logger.info(f"Progress update: {len(all_results)} samples processed so far...")

    # Save all results to JSON
    json_path = os.path.expanduser(args.step1_result_path)
    save_to_json(json_path, all_results)
    logger.info(f"Processing complete! Processed {len(all_results)} samples. Saved to {json_path}")


if __name__ == "__main__":
    """Main execution function."""

    model, tokenizer = util_loader.load_vlm(args.vlm_model)
    data = util_loader.load_data(args.dataset)
    
    # Process dataset
    main(model=model, tokenizer=tokenizer, data=data, max_samples=None, batch_size=16)



