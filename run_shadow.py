"""
This file is the script to run the CMIA framework (MIA shadow modeling part)
"""

import os
import argparse
from datetime import datetime
import subprocess
from multiprocessing import Pool

parser = argparse.ArgumentParser()

######### env configuration ########
parser.add_argument("--cuda", "-c", default=0, type=int)
parser.add_argument("--n_gpus", default=1, type=int)
parser.add_argument("--n_parallel", default=10, type=int)


parser.add_argument("--dataset", default="cifar10", type=str)
parser.add_argument("--lr", default=0.1, type=float)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument("--model_type", default="resnet", type=str)
# mia parameters
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--n_queries", default=2, type=int) # data augmentations per sample during inference
parser.add_argument("--n_shadows", default=257, type=int)
parser.add_argument("--savedir", default="./results/", type=str)

args = parser.parse_args()

print(args)
data_dir = os.path.join("data",args.dataset)
save_dir = os.path.join("results", args.dataset, datetime.now().strftime('%Y-%m-%d'))

"""
# Step 1: Train shadow models in parallel
for shadow_id in range(0, args.n_shadows):
    if shadow_id % args.n_gpus == args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
        print(f"Training model {shadow_id}/{args.n_shadows} ...")
        os.system(
            f"python shadow.py --dataset {args.dataset} --shadow_id {shadow_id} --data_dir {data_dir} --savedir {save_dir} --seed {args.seed} --model_type {args.model_type} --n_shadows {args.n_shadows} --n_queries {args.n_queries}"
        )
"""

def run_shadow(shadow_id: int):
    gpu_id = shadow_id % args.n_gpus

    # Per-process environment so each job sees only one GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        "python", "shadow.py",
        "--dataset", args.dataset,
        "--shadow_id", str(shadow_id),
        "--data_dir", data_dir,
        "--savedir", save_dir,
        "--seed", str(args.seed),
        "--model_type", args.model_type,
        "--n_shadows", str(args.n_shadows),
        "--n_queries", str(args.n_queries),
        "--lr", str(args.lr),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
    ]

    print(f"[PID {os.getpid()}] Training model {shadow_id}/{args.n_shadows} on GPU {gpu_id} ...", flush=True)

    # Run the process and raise if it fails (so you notice crashes)
    subprocess.run(cmd, env=env, check=True)

if __name__ == "__main__":
    # Run at most n_gpus training processes concurrently
    with Pool(processes=args.n_gpus) as pool:
        pool.map(run_shadow, range(args.n_shadows))