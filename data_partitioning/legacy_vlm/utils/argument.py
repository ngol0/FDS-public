import os
import shutil
import argparse
from pathlib import Path
from dotenv import load_dotenv
from time import strftime, localtime

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description="Experiment arguments for ICTC")

# ---datadir
parser.add_argument("--dataset", type=str, default="imagenet", choices=["imagenet", "food101", "pascalvoc", "ade20k"])
# ---parse initial arguments first
args, _ = parser.parse_known_args()

# ---output dir
#parser.add_argument("--exp_name", type=str, default="test")

# ---criteria choices (depends on dataset)
if args.dataset == "imagenet":
    criteria_choices = ["main_object", "size", "time", "color", "location"]
elif args.dataset == "food101":
    criteria_choices = ["ingredient", "category"]
else:
    criteria_choices = ["main_object"]

parser.add_argument("--criteria", type=str, default="main_object", choices=criteria_choices)

# ---model choices
# for llama
parser.add_argument("--llama_ver", type=str, default="llama_70b", choices = ["llama_7b", "llama_70b"])
# for vlm
parser.add_argument("--vlm_model", type=str, default="internVl3", choices = ["minicpm", "internVl3"])

args = parser.parse_args()

load_dotenv()
args.home_path = os.getenv("HOME_PATH")

if not args.home_path:
    raise EnvironmentError("HOME_PATH not found in .env file. Please define it before running.")

#------ Setting the params ------- #
args.general_data_dir = f"{args.home_path}/data_partitioning/legacy_vlm/data/"
args.specific_dataset_dir = args.general_data_dir + args.dataset
args.output_path = args.specific_dataset_dir + f"/output_{args.criteria}"

# ------ Create output folder for each run ------
Path(args.output_path).mkdir(parents=True, exist_ok=True)
args.step1_result_path = f"{args.output_path}/step1_result.jsonl"
args.step2a_result_path = f"{args.output_path}/step2a_result.json"
args.step2b_result_path = f"{args.output_path}/step2b_result.txt"
args.step3_result_path = f"{args.output_path}/step3_result.json"

# ----- Prompt setting
# Direction for each prompt
args.prompt_path = args.specific_dataset_dir
args.step1_prompt_path = args.prompt_path + f"/step1_prompt_" + args.criteria 
args.step1_prompt_path += ".txt"
args.step2a_prompt_path = args.prompt_path + f"/step2a_prompt_" + args.criteria 
args.step2a_prompt_path += ".txt"
args.step2b_prompt_path = args.prompt_path + f"/step2b_prompt_" + args.criteria 
args.step2b_prompt_path += ".txt"
args.step3_prompt_path = args.prompt_path + f"/step3_prompt_" + args.criteria 
args.step3_prompt_path += ".txt"


# Copy the prompts for each experiment
shutil.copy(args.step1_prompt_path, f"{args.output_path}/step1_prompt.txt")
shutil.copy(args.step2a_prompt_path, f"{args.output_path}/step2a_prompt.txt")
shutil.copy(args.step2b_prompt_path, f"{args.output_path}/step2b_prompt.txt")
shutil.copy(args.step3_prompt_path, f"{args.output_path}/step3_prompt.txt")

# Define the number of classes for step 3
if args.dataset == "imagenet":
    args.num_classes = 8

#--------------------------------------------------------------------------------------