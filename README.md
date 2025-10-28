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

Example: data/tiny_imagenet/output_main_object_test

### Prompt:
Prompts for each dataset are set up in .txt files. They are located in: `data/<dataset_name>`. 

Each step’s prompt is dynamically customized based on the selected criterion. The script automatically embeds the corresponding values from the `criteria_prompt` (all steps), `examplar_criteria` (step 2a), and `num_classes` (step 3) lists defined in `utils/argument.py`, to make sure that each step uses the appropriate wording for the chosen criterion. 

After running, the prompts for that run are also copied into the output folder.

### Set up: 
* Model and dataset: In `utils/utils_loader.py`. Edit to user's custom path for VLM/LLM and dataset when run.
* Arguments settings: In `utils/argument.py`. 




