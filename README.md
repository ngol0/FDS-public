# TODO

## VLM pipeline

### Run scripts:
4 scripts for each step: `step1.py`, `step2a.py`, `step2b.py`, `step3.py`. To run full pipeline, use: `full_pipeline.py`

To run script:

`python <script>.py --dataset <data> --criteria <criterion> --exp_name <name_of_experiment> --llama_ver <llama_model> --vlm_model <vlm_model>`

Default params if not specified:
* dataset: Tiny ImageNet
* criteria: Main Object
* exp_name: test
* llama_ver: LLaMA 3.3 70B
* vlm_model: InternVL3 38B

### Output: 
`data/<dataset_name>/<output_criteria_name_of_experiment>`

Example: data/imagenet/output_main_object_test

### Prompt:
Prompts for each dataset are set up in a .txt file. They are located in: `data/<dataset_name>`. After running, the prompts for that run are also copied into the output folder.

### Set up: 
* Model and dataset: In `utils/utils_loader.py`. Edit to user's custom path for VLM/LLM and dataset when run.
* Criteria for prompt customization: In `utils/argument.py`. 




