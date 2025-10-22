import step1
import step2a
import step2b
import step3
from utils import model_loader

def main():
    # Load model 
    llm_model, llm_tokenizer, vlm_model, vlm_tokenizer = model_loader.load_models_for_script() 

    # Run pipeline
    print("Starting full ICTC pipeline...")
    step1.main(vlm_model, vlm_tokenizer, max_samples=None, batch_size=50)
    step2a.main(llm_model, llm_tokenizer, inference_batch_size=64)
    step2b.main(llm_model, llm_tokenizer)
    step3.main(llm_model, llm_tokenizer, inference_batch_size=64)
    print("Full pipeline finished successfully!")

if __name__ == "__main__":
    main()