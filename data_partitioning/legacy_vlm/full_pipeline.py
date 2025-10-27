import step1
import step2a
import step2b
import step3
from utils import util_loader
from utils.argument import args
import gc
import torch

def main():
    # Load model 
    print("=============== Starting full ICTC pipeline... ========================")
    print("--- Loading VLM ...")
    vlm_model, vlm_tokenizer = util_loader.load_vlm(args.vlm_model)

    print("----Loading dataset...")
    # Load data: get dataset_name from argument when run `python script.py --dataset <name>`, default is: imagenet
    data = util_loader.load_data(args.dataset)

    # Run pipeline
    print("1) === Step 1...")
    step1.main(vlm_model, vlm_tokenizer, data=data, max_samples=None, batch_size=64)

    # Unload vlm here:
    del vlm_model
    del vlm_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    print("--- VLM unloaded, loading LLM...")
    llm_model, llm_tokenizer = util_loader.load_llm(args.llama_ver)

    print("2a) === Step 2a...")
    step2a.main(llm_model, llm_tokenizer, inference_batch_size=64)

    print("2b) === Step 2b...")
    step2b.main(llm_model, llm_tokenizer)

    print("3) === Step 3...")
    step3.main(llm_model, llm_tokenizer, inference_batch_size=64)
    
    print("Full pipeline finished successfully!")

if __name__ == "__main__":
    main()