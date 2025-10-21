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

# datadir
parser.add_argument("--dataset", type=str, default="imagenet", choices=["imagenet", "food101", "pascalvoc", "ade20k"])

# output dir
parser.add_argument("--exp_name", type=str, default="test")

args = parser.parse_args()

load_dotenv()
args.home_path = os.getenv("HOME_PATH")
args.image_folder = f"{args.home_path}/data/"

#------ Setting the params ------- #
args.general_data_dir = f"{args.home_path}/data_partitioning/legacy_vlm/data/"
args.specific_dataset_dir = args.general_data_dir + args.dataset
args.output_path = args.specific_dataset_dir + f"/{args.exp_name}"

# ------ Create output folder for each run ------
Path(args.output_path).mkdir(parents=True, exist_ok=True)
args.step1_result_path = f"{args.output_path}/step1_result.jsonl"
args.step2a_result_path = f"{args.output_path}/step2a_result.json"
args.step2b_result_path = f"{args.output_path}/step2b_result.json"
args.step3_result_path = f"{args.output_path}/step3_result.json"

# ------ Setup dataset and clustering criteria
if args.dataset == "imagenet":
    #args.image_folder += "imagenet/"
    args.num_classes = 10

# ----- For copying ----
# Direction for each prompt to use in the copying step
args.step1_prompt_path = args.specific_dataset_dir + f"/step1_prompt"
args.step1_prompt_path += ".txt"
args.step2a_prompt_path = args.specific_dataset_dir + f"/step2a_prompt"
args.step2a_prompt_path += ".txt"


# Copy the prompts for each experiment
if not os.path.exists(f"{args.output_path}/step1_prompt.txt"):
    shutil.copy(args.step1_prompt_path, f"{args.output_path}/step1_prompt.txt")
if not os.path.exists(f"{args.output_path}/step2a_prompt.txt"):
    shutil.copy(args.step2a_prompt_path, f"{args.output_path}/step2a_prompt.txt")
# if not os.path.exists(f"{args.output_path}/step2b_prompt.txt"):
#     shutil.copy(args.step2b_prompt_path, f"{args.output_path}/step2b_prompt.txt")
# if not os.path.exists(f"{args.output_path}/step3_prompt.txt"):
#     shutil.copy(args.step3_prompt_path, f"{args.output_path}/step3_prompt.txt")


#--------------------------------------------------------------------------------------
# elif args.dataset == "cifar100":
#     args.image_folder += "cifar100/"
#     args.num_classes = 20
# elif args.dataset == "stl10":
#     args.image_folder += "stl10/test/"
#     args.num_classes = 10
# elif args.dataset == "ppmi":
#     args.image_folder += "ppmi/"
#     if args.num_classes == 12:
#         args.image_folder += "12_classes/"
#     elif args.num_classes == 7:
#         args.image_folder += "7_classes/"
#     elif args.num_classes == 2:
#         args.image_folder += "2_classes/"
# elif args.dataset == "stanford-40-actions":
#     args.image_folder += "stanford-40-actions/JPEGImages/"
#     if args.cl_criteria == "action":
#         args.num_classes = 40