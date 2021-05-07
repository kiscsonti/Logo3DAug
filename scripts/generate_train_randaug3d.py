import os
from argparse import ArgumentParser
import argparse

parser = ArgumentParser()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser.add_argument('--im_per_class', type=int, default=4)
parser.add_argument('--batch', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--earlystop', type=int, default=30)
parser.add_argument('--opt_patience', type=int, default=20)
parser.add_argument('--smoothing', type=float, default=0.1)
parser.add_argument('--optimizer', type=str, default="SGD", help="You can choose from [Adam, SGD]")
parser.add_argument('--opt_step', nargs='+', type=int, default=[20, 100])
parser.add_argument('--model_name', type=str, default="convnet")
parser.add_argument('--path', type=str, default="./running")
parser.add_argument('--test_dir', type=str, default=None)
parser.add_argument('--train_dir', type=str, default=None)
parser.add_argument('--run_id', type=str, default="petigep")
parser.add_argument('--gpu', nargs='+', type=int, default=[])
parser.add_argument('--step_m', type=int, default=1)
parser.add_argument('--low_n', type=int, default=1)
parser.add_argument('--up_n', type=int, default=4)
parser.add_argument('--lit_model', type=str, default="original", help="Choose from [original, nts, wsdan, dcl]")
parser.add_argument('--randaug_augs', type=str, default="all", help="Choose from [pt, gen, all]")
parser.add_argument('--port', type=str, default=12583)
parser.add_argument('--save_dir', type=str, default="mlruns")


args = parser.parse_args()


assert args.randaug_augs in ["pt", "gen", "all"]

base = """python train.py --val_n 1 --epoch {} --opt_step {} --lr {} --im_per_class {} --batch {} --earlystop {} --opt_patience {} --smoothing {} --optimizer {} --path {} --run_id {} --is_generator True --test_dir {} --train_dir {} --pregen True --model_name {} --gpu {} --randaug_N {} --randaug_M {} --randaug True --lit_model {} --randaug_augs {} --port {} --save_dir {}\n"""

opt_step = " ".join(list(map(str, args.opt_step)))
gpu = None if len(args.gpu) == 0 else " ".join(list(map(str, args.gpu)))

with open("randaug_train_gridsearch.sh", "w") as out_f:
    for m in range(1, 11, args.step_m):
        for n in range(args.low_n, args.up_n + 1):
            command = base.format(args.epoch, opt_step, args.lr, args.im_per_class, args.batch, args.earlystop,
                                  args.opt_patience, args.smoothing, args.optimizer, args.path,
                                  args.run_id + "_" + str(m) + "_" + str(n),
                                  args.test_dir,
                                  args.train_dir, 
                                  args.model_name, gpu, n, m, args.lit_model, args.randaug_augs,
                                  args.port,
                                  args.save_dir)
            if args.gpu is None:
                command = command.replace("--gpu None", "")
            out_f.write(command)
