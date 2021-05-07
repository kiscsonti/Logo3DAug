from argparse import ArgumentParser
import argparse
import os
import pytorch_lightning as pl
from dataloader import  get_test_data
import torch
import json
import utils
from dataset.data_modules import LogoDataModule

parser = ArgumentParser()


class DataloaderParams:

    def __init__(self, path, test_dir, batch):
        self.test_dir = test_dir
        self.path = path
        self.batch = batch


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':

    parser.add_argument('--notf_log', type=str2bool, default=False)
    parser.add_argument('--pretrained', type=str2bool, default=True)
    parser.add_argument('--fine_tune', type=str2bool, default=False)
    parser.add_argument('--optimizer', type=str, default="Adam", help="You can choose from [Adam, SGD]")
    parser.add_argument('--debug', type=str2bool, default=False)
    parser.add_argument('--is_generator', type=str2bool, default=False)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--earlystop', type=int, default=10)
    parser.add_argument('--opt_patience', type=int, default=5)
    parser.add_argument('--opt_step', nargs='+', type=int, default=[])
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--smoothing', type=float, default=0)
    parser.add_argument('--model_name', type=str, default="resnet50")
    parser.add_argument('--model_path', type=str, default="mlruns/1/")
    parser.add_argument('--dataset', type=str, default="/home/kardosp/logo_projek/datasets/logo2k")
    parser.add_argument('--run_id', type=str, default="petigep")
    parser.add_argument('--path', type=str, default=None)
    parser.add_argument('--randaug', type=str2bool, default=False)
    parser.add_argument('--gpu', nargs='+', type=int, default=[])
    parser.add_argument('--classes', nargs='*', type=str, default=["Kroger", "Volvo"], help="This parameter doesnt matter as the test dataset classes will be used")
    parser.add_argument('--lit_model', type=str, default="original", help="Choose from [original, nts, wsdan, dcl]")

    # WSDAN
    parser.add_argument('--wsdan_beta', type=float, default=5e-2)
    parser.add_argument('--wsdan_num_attentions', type=float, default=32)

    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    if len(args.gpu) == 0:
        args.gpu = None

    print(args)
    dataset_path = os.path.split(args.dataset)
    mlruns_base_dir = os.path.join(*(args.model_path.split(os.sep)[:-2]))
    
    if not mlruns_base_dir.startswith("/") and mlruns_base_dir.startswith("home"):
        mlruns_base_dir = str(os.sep) + mlruns_base_dir
    print(mlruns_base_dir)
    
    if not os.path.exists(os.path.join(mlruns_base_dir, "artifacts", "confusion_matrices")):
        os.mkdir(os.path.join(mlruns_base_dir, "artifacts", "confusion_matrices"))
    
    confusion_matrix_all_files = os.listdir(os.path.join(mlruns_base_dir, "artifacts", "confusion_matrices"))
    classes = None
    with open(os.path.join(mlruns_base_dir, "artifacts", "confusion_matrices", confusion_matrix_all_files[0]), "r")as conf_matrix_file:
        for line in conf_matrix_file:
            last_line = line
    # print(last_line)
    # classes = last_line.split(" ") # Not good because there are classes with space in them. Separator should be something else
    # print(os.path.join(*dataset_path[:-1]), dataset_path[-1], args.batch)
    # eval_dl = get_test_data(224, DataloaderParams(os.path.join(*dataset_path[:-1]), dataset_path[-1], args.batch))
    # args.classes = [item[0] for item in sorted(eval_dl.dataset.class_to_idx.items(), key=lambda x: int(x[1]))]

    args.path = os.path.join(*dataset_path[:-1])
    transform_controller = utils.get_augmentor(args, None, [])

    trainer = pl.Trainer(gpus=args.gpu)

    data_module = LogoDataModule(args, None, transform_controller, None)
    eval_dl = data_module.test_dataloader()
    args.classes = [item[0] for item in sorted(eval_dl.dataset.class_to_idx.items(), key=lambda x: int(x[1]))]

    model = utils.get_lit_model(args, None, trainer, transform_controller)

    checkpoint = torch.load(
        args.model_path,
        map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint['state_dict'])
    if args.lit_model == "wsdan":
        if 'feature_center' in checkpoint:
            model.feature_center = checkpoint['feature_center']
        else:
            print("NO FEATURE CENTER IN LOADED MODEL!. Creating one...")
            model.feature_center = torch.zeros(len(args.classes),
                                               args.wsdan_num_attentions * model.model.num_features
                                               )

    res = trainer.test(model, test_dataloaders=eval_dl)
    print("RES: ", res)
    res = res[0]
    with open(os.path.join(mlruns_base_dir, "metrics", "test" + args.run_id), "w") as out_f:
        jsonable = res
        jsonable["conf"] = str(jsonable["conf"])
        jsonable["test_acc"] = str(jsonable["test_acc"])
        jsonable["test_loss"] = str(jsonable["test_loss"])
        jsonable["test_top5_acc"] = str(jsonable["test_top5_acc"])
        json.dump(res, out_f)
    print(res)
    # print("Original classes:", classes)
    # print("Eval dl  classes:", " ".join(args.classes))
    with open(os.path.join(mlruns_base_dir, "artifacts", "confusion_matrices", "test_confusion_matrix.txt"), "w") as confusion_out:
        confusion_out.write(str(res["conf"]))
        confusion_out.write("\r\n")
        confusion_out.write(",,".join(args.classes))
    # acc, acc_top5 = intristic_eval_w_top5(model, eval_dl)
    # print("Accuracy:", acc)
    # print("Accuracy TOP 5:", acc_top5)

