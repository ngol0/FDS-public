from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModel, AutoTokenizer, AutoConfig
import sys
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from dataset import BaseDataset
import math

# model and tokenizer
#----- Load model ----

# VLM
minicpm_model_dir = "/users/sbsh771/archive/vision-saved/minicpm26"
internVL3_model_dir = "/users/sbsh771/archive/vision-saved/internVl3"

# LLM
llama_7b_model_dir = "/users/sbsh771/archive/vision-saved/llama3.1-instruct"
llama_70b_model_dir = "/users/sbsh771/archive/vision-saved/llama3.3"

def split_model():
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(internVL3_model_dir, trust_remote_code=True)
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
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

def load_internvl3(path):
    """Load InternVL3 model split across 2 GPUs"""    
    # Get device map for 2 GPUs
    device_map = split_model()
    
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,  # Set to True if need to save memory
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        path, 
        trust_remote_code=True, 
        use_fast=False)
    
    return model, tokenizer

def load_llm(ver: str = "llama_7b"):
    # Pick the model
    if (ver == "llama_7b"):
        print("Loading LLaMA 3.1-Instruct 7B...")
        model_path = llama_7b_model_dir
    if (ver == "llama_70b"):
        print("Loading LLaMA 3.3 70B...")
        model_path = llama_70b_model_dir

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

def load_vlm(choice: str = "minicpm"):
    """Load the (MiniCPM) model and tokenizer."""
    # Pick the model
    if (choice == "minicpm"):
        print("Loading MiniCPM 2.6...")
        model_path = minicpm_model_dir
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, dtype=torch.bfloat16)
        model = model.to(device='cuda', dtype=torch.bfloat16)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if (choice == "internVl3"):
        print("Loading InternVL3-38B...")
        model_path = internVL3_model_dir
        model, tokenizer = load_internvl3(model_path)
        
    return model, tokenizer

def load_data(dataset: str):
    # Load dataset
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