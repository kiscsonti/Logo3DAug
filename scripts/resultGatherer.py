import json
import pandas as pd
import numpy as np
import os
from typing import List, Set, Dict, Tuple
from argparse import ArgumentParser
import sys


def read_first_line(path):
    with open(path, "r") as in_f:
        for line in in_f:
            return line


def get_best_epoch(path):
    best_epoch = 0
    best_val = 0
    with open(path, "r") as in_f:
        for line in in_f:
            splitted = line.split(" ")
            if float(splitted[1]) > best_val:
                best_val = float(splitted[1])
                best_epoch = int(splitted[2])

    return best_val, best_epoch


def get_ith_epoch(path, ith):
    with open(path, "r") as in_f:
        for line in in_f:
            splitted = line.split(" ")
            if int(splitted[2]) == ith:
                return float(splitted[1])

    return None


def read_test_results(path):
    with open(path, "r") as in_f:
        for line in in_f:
            loaded_data = json.loads(line)
            return loaded_data["test_acc"], loaded_data["test_top5_acc"]


def earlystop_epoch(path):
    artifact_files = os.listdir(path)
    artifact_files = list(filter(lambda x: x.endswith(".ckpt"), artifact_files))
    artifact_files = sorted(artifact_files, key=lambda x: len(x), reverse=True)
    epoch_nmb = int(artifact_files[0].split("epoch=")[-1].split(".")[0])
    return epoch_nmb

parser = ArgumentParser()
parser.add_argument('--mlruns', type=str,
                    help="Directory where the runs are located. eg: mlruns/1")
parser.add_argument('--output', type=str,
                    help="Name of output file. eg: results.xls")
args = parser.parse_args()

runs_tmp = os.listdir(args.mlruns)
runs = list()
for item in runs_tmp:
    if item.__contains__("meta.yaml"):
        continue
    runs.append(item)

container = list()
for run in runs:
    record = dict()
    "RUN NAME"
    record["run_id"] = read_first_line(os.path.join(args.mlruns, run, "params", "run_id"))
    record["train_time"] = read_first_line(os.path.join(args.mlruns, run, "params", "train_time_took"))
    best_val, best_epoch = get_best_epoch(os.path.join(args.mlruns, run, "metrics", "val_acc"))
    record["val_acc"] = best_val
    record["train_acc"] = get_ith_epoch(os.path.join(args.mlruns, run, "metrics", "epoch_train_acc"), best_epoch)
    top1, top5 = read_test_results(os.path.join(args.mlruns, run, "metrics", "testtest_original"))
    record["test_acc"] = top1
    record["test_top5"] = top5
    record["bestmodel_epoch"] = earlystop_epoch(os.path.join(args.mlruns, run, "artifacts"))
    container.append(record)

df = pd.DataFrame(container)
print(df.head())
df.to_excel(args.output)
