# Read from from txt file
import json
from typing import List, Dict
import re

def read_file_to_string(filename):
    with open(filename, 'r') as file:
        content = file.read()
    return content

def clean_response(response: str) -> str:
    """Remove common artifacts from model responses."""
    # Remove "assistant" prefix (case-insensitive)
    response = response.strip()
    
    # Pattern 1: "assistant\n\n{text}"
    if response.lower().startswith("assistant"):
        response = response[len("assistant"):].strip()
    
    # Pattern 2: Remove leading newlines
    response = response.lstrip('\n').strip()
    
    # Pattern 3: If starts with "assistant:" or "Assistant:"
    import re
    response = re.sub(r'^assistant\s*:?\s*', '', response, flags=re.IGNORECASE)
    
    return response.strip()

# --- Load captions from JSONL ---
def load_data(json_path: str) -> List[Dict]:
    """Load data from JSONL file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# ---- Read class list for step 3 ---
def read_class_list(step2b_result_path):
    with open(step2b_result_path, 'r') as file:
        file_read = file.readlines()
        class_list = []
        for lab in file_read:
            if "Reason" not in lab and lab.strip() != "" and ":" in lab:
                lab = lab.split(":")[1].strip().lower()
                class_list.append(lab)

    return class_list

def extract_categories(step2b_result_path) -> List[str]:
    """
    Extract category names from the Answer lines.
    
    Args:
        text: The full response text containing Answer 1, Answer 2, etc.
    
    Returns:
        List of category names in order
    """
    categories = []
    """Load and extract categories from step2b output file."""
    with open(step2b_result_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all lines that start with "Answer X:"
    pattern = r'Answer\s+\d+:\s*(.+?)(?:\n|$)'
    matches = re.findall(pattern, content, re.IGNORECASE)
    
    for match in matches:
        category = match.strip()
        categories.append(category)
    
    return categories