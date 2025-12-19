"""
This file is the script to run the CMIA framework (attack part).
"""

import os
import argparse
from utils.loader import load_config
import datetime

parser = argparse.ArgumentParser()

######### env configuration ########
parser.add_argument("--dataset", default="cifar10", type=str)
parser.add_argument("--attack", default="lira", type=str)
parser.add_argument("--n_shadows", default=17, type=int)
parser.add_argument("--savedir", default="./results/", type=str)

args = parser.parse_args()
print(args)
save_dir = os.path.join("results", args.dataset, datetime.now().strftime('%Y-%m-%d'))


command = "python attack.py"

command += f" --dataset {args.dataset} --n_shadows {args.n_shadows} --attack {args.attack} --save_dir {save_dir}"

os.system(command)
