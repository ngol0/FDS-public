from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModel, AutoTokenizer
import sys
import os
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from dataset import BaseDataset

# model and tokenizer
#----- Load model ----
llm_model = None
llm_tokenizer = None 
vlm_model = None
vlm_tokenizer = None

minicpm_model_dir = "/users/sbsh771/archive/vision-saved/minicpm26"
llama_model_path = "/users/sbsh771/archive/vision-saved/llama3.1-instruct"

def load_llama():
    tokenizer = AutoTokenizer.from_pretrained(llama_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        llama_model_path,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer

def load_minicpm():
    """Load the (MiniCPM) model and tokenizer."""
        
    model = AutoModel.from_pretrained(
        minicpm_model_dir, 
        trust_remote_code=True, 
        dtype=torch.bfloat16
    )
    model = model.to(device='cuda', dtype=torch.bfloat16)
    model.eval()
        
    tokenizer = AutoTokenizer.from_pretrained(
        minicpm_model_dir, 
        trust_remote_code=True
    )
        
    return model, tokenizer

def load_data(dataset: str):
    # Load dataset
    # ---- For TinyImagenet -----
    if dataset == "imagenet":
        print("Dataset: TinyImagenet")
        data = BaseDataset("tiny_imagenet", "/users/sbsh771/gtai/FDS", "val")
    if dataset == "food101":
        print("Dataset: Food 101")
        data = BaseDataset("food101", "/users/sbsh771/gtai/FDS", "test")
    #>>> todo: add other dataset here:

    return data

# def load_models_for_script():
#     """Load models based on which script is being executed."""
#     script_name = os.path.basename(sys.argv[0]).lower()

#     llm = vlm = None
#     llm_tokenizer = vlm_tokenizer = None

#     if "step1" in script_name:
#         print("Loading MiniCPM...")
#         vlm, vlm_tokenizer = load_minicpm()

#     elif "step2" in script_name or "step3" in script_name:
#         print("Loading LLaMA...")
#         llm, llm_tokenizer = load_llama()

#     elif "full_pipeline" in script_name:
#         print("Loading both models...")
#         llm, llm_tokenizer = load_llama()
#         vlm, vlm_tokenizer = load_minicpm()

#     else:
#         print(f"Unknown script: {script_name}, no models loaded.")

#     return llm, llm_tokenizer, vlm, vlm_tokenizer