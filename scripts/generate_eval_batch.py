import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--mlruns', type=str,
                    default="../mlruns/1",
                    help="Directory where the runs are located. eg: mlruns/1")
parser.add_argument('--model', type=str,
                    help="Name of output file. eg: wide_resnet50")
parser.add_argument('--dataset', type=str,
                    default="/home/kardosp/logo_projek/datasets/flickrlogos32/test/",
                    help="Path to dataset folder")
parser.add_argument('--gpu', type=int,
                    default=None,
                    help="Which gpu to use")
parser.add_argument('--test_name', type=str,
                    default="test_original",
                    help="Which gpu to use")
parser.add_argument('--append', action="store_true",
                    help="Use append or write")
parser.add_argument('--lit_model', type=str,
                    default="original",
                    help="Choose from [original, nts]")

args = parser.parse_args()


base = """python ../evaluateOnDataset.py --model_path {} --dataset {} --model_name {} --run_id {} --lit_model {} --gpu {} \n"""
list_of_files_tmp = os.listdir(args.mlruns)
list_of_files = list()

for item in list_of_files_tmp:
    if item.__contains__("meta.yaml"):
        continue
    list_of_files.append(item)

with open("eval_batch.sh", "a" if args.append else "w") as out_f:
    for item in list_of_files:
        ckpt_files = os.listdir(os.path.join(args.mlruns, item, "artifacts"))
        lengthiest = ""
        for ckpt_f in ckpt_files:
            if len(ckpt_f) > len(lengthiest):
                lengthiest = ckpt_f
        command = base.format(os.path.join(args.mlruns, item, "artifacts", lengthiest),
                                args.dataset, args.model, args.test_name, args.lit_model, args.gpu)
        if args.gpu is None:
            command = command.replace("--gpu None", "")
        out_f.write(command)
