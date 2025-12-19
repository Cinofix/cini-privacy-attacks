"""
This file is the script to run the CMIA framework (MIA shadow modeling part)
"""

import os
import argparse
from utils.loader import load_config
from datetime import datetime

parser = argparse.ArgumentParser()

######### env configuration ########
parser.add_argument("--cuda", "-c", default=0, type=int)
parser.add_argument("--dataset", default="cifar10", type=str)
parser.add_argument("--lr", default=0.1, type=float)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument("--model_type", default="resnet", type=str)
# mia parameters
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--n_queries", default=None, type=int)
parser.add_argument("--n_shadows", default=257, type=int)
parser.add_argument("--savedir", default="./results/", type=str)

args = parser.parse_args()
load_config(args)
print(args)
data_dir = os.path.join("data",args.dataset)
save_dir = os.path.join("results", args.dataset, datetime.now().strftime('%Y-%m-%d'))
save_dir = os.path.join(save_dir, str(args.shadow_id))


# Step 1: Train shadow models in parallel
for shadow_id in range(0, args.n_shadows):
    save_dir = os.path.join(save_dir, str(args.shadow_id))

    if shadow_id % args.n_gpus == args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
        print(f"Training model {shadow_id}/{args.n_shadows} ...")
        os.system(
            f"python shadow.py --dataset {args.dataset} --shadow_id {shadow_id} --data_dir {data_dir} --savedir {save_dir} --seed {args.seed} --model_type {args.model_type} --n_shadows {args.n_shadows} --n_queries {args.n_queries}"
        )
