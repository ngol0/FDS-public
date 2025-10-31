from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModel, AutoTokenizer, AutoConfig
import sys
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from dataset import BaseDataset
import math
import torchvision.transforms as T
from utils.constants import FOOD101, INTERNVL3_38B, LLAMA_33_70B, LLAMA31_8B_INSTRUCT , MINICPM_26, INTERNVL3_5_8B


def split_model(path):
    device_map = {}
    world_size = torch.cuda.device_count()
    print("Num of GPU: ", world_size)
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0

    #device_map['language_model.model.tok_embeddings'] = 0
    #device_map['language_model.output'] = 0
    #device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

# model and tokenizer
#----- Load model ----
# note: change to custom path when run
# Model paths
MODEL_PATHS = {
    MINICPM_26: "/users/sbsh771/archive/vision-saved/minicpm26",
    INTERNVL3_38B: "/users/sbsh771/archive/vision-saved/internVl3",
    INTERNVL3_5_8B: "/users/sbsh771/archive/vision-saved/internVl3_5",
    LLAMA31_8B_INSTRUCT: "/users/sbsh771/archive/vision-saved/llama3.1-instruct",
    LLAMA_33_70B: "/users/sbsh771/archive/vision-saved/llama3.3"
}

# Model Registry
MODEL_REGISTRY = {}

def register_model(name: str):
    """Decorator to register a model loader function."""
    def decorator(load_func):
        MODEL_REGISTRY[name] = load_func
        return load_func
    return decorator

def get_model_loader(name: str):
    """Get the registered model loader function."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found in registry. Available models: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]

# Register VLM models
@register_model(MINICPM_26)
def load_minicpm():
    """Load MiniCPM 2.6"""
    print(f"Loading {MINICPM_26}...")
    model_path = MODEL_PATHS[MINICPM_26]
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, dtype=torch.bfloat16)
    model = model.to(device='cuda', dtype=torch.bfloat16)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer

@register_model(INTERNVL3_38B)
def load_internvl3_38b():
    """Load InternVL3-38B with custom device map"""
    print(f"Loading {INTERNVL3_38B}...")
    model_path = MODEL_PATHS[INTERNVL3_38B]
    device_map = split_model(model_path)
    
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        use_fast=False)
    
    return model, tokenizer

@register_model(INTERNVL3_5_8B)
def load_internvl3_5_8b():
    """Load InternVL3.5-8B with auto device map"""
    print(f"Loading {INTERNVL3_5_8B}...")
    model_path = MODEL_PATHS[INTERNVL3_5_8B]
    device_map = split_model(model_path)
    
    model = AutoModel.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        #local_files_only=True,  # Add this!
        device_map="auto").eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        use_fast=False)
    
    return model, tokenizer

# Register LLM models
@register_model(LLAMA31_8B_INSTRUCT)
def load_llama_8b():
    """Load LLaMA 3.1-Instruct 8B"""
    print(f"Loading {LLAMA31_8B_INSTRUCT}...")
    model_path = MODEL_PATHS[LLAMA31_8B_INSTRUCT]
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer

@register_model(LLAMA_33_70B)
def load_llama_70b():
    """Load LLaMA 3.3 70B"""
    print(f"Loading {LLAMA_33_70B}...")
    model_path = MODEL_PATHS[LLAMA_33_70B]
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer

# Main loading functions
def load_vlm(ver: str = INTERNVL3_5_8B):
    """Load the VLM model and tokenizer."""
    loader_func = get_model_loader(ver)
    return loader_func()

def load_llm(ver: str = LLAMA_33_70B):
    """Load the LLM model and tokenizer."""
    loader_func = get_model_loader(ver)
    return loader_func()

# Dataset loading
def load_data(dataset: str):
    """Load dataset based on params set when run script."""
    path = "/users/sbsh771/gtai/FDS"
    split = "val"
    if dataset == FOOD101:
        split = "test"
    
    print("Dataset: ", dataset)
    data = BaseDataset(dataset, path, split)
    return data

# Optional: Print available models
def list_available_models():
    """List all registered models."""
    return list(MODEL_REGISTRY.keys())
