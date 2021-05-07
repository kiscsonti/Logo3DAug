from argparse import ArgumentParser
import argparse
import pytorch_lightning as pl

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


# parametrize the network
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--im_per_class', type=int, default=16)
parser.add_argument('--batch', type=int, default=32)
parser.add_argument('--hidden_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--val_n', type=int, default=1)
parser.add_argument('--earlystop', type=int, default=10)
parser.add_argument('--opt_patience', type=int, default=5)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--smoothing', type=float, default=0.1)
parser.add_argument('--optimizer', type=str, default="Adam", help="You can choose from [Adam, SGD]")
parser.add_argument('--opt_step', nargs='+', type=int, default=[])
parser.add_argument('--model_name', type=str, default="convnet")
parser.add_argument('--lit_model', type=str, default="original", help="Choose from [original, nts, wsdan, dcl]")
parser.add_argument('--pregen', type=str2bool, default=False)
parser.add_argument('--fine_tune', type=str2bool, default=False)
parser.add_argument('--pretrained', type=str2bool, default=True)
parser.add_argument('--augment', type=str2bool, default=False)
parser.add_argument('--synthetic', type=str2bool, default=False)
parser.add_argument('--is_generator', type=str2bool, default=True)
parser.add_argument('--debug', type=str2bool, default=False)
parser.add_argument('--notf_log', type=str2bool, default=False)
parser.add_argument('--save_path', type=str, default="saved_models/LVL1_Logo_NoLogo_{epoch}")
parser.add_argument('--path', type=str, default="./running")
parser.add_argument('--test_dir', type=str, default=None)
parser.add_argument('--train_dir', type=str, default=None)
parser.add_argument('--run_id', type=str, default="petigep")
parser.add_argument('--gpu', nargs='+', type=int, default=[])
parser.add_argument('--classes', nargs='*', type=str, default=["Kroger", "Volvo"], help="This parameter doesnt matter as the test dataset classes will be used")
parser.add_argument('--aug_preset', type=str, default="default")
parser.add_argument('--ckpt', type=str, default=None)
parser.add_argument('--save_dir', type=str, default="mlruns")
parser.add_argument('--port', type=int, default=12583)

#RandAug
parser.add_argument('--randaug', type=str2bool, default=False)
parser.add_argument('--randaug_augs', type=str, default="all", help="Choose from [pt, gen, all]")
parser.add_argument('--randaug_M', type=float, default=1)
parser.add_argument('--randaug_N', type=int, default=1)


parser.add_argument('--pt_rotate', type=int, default=0)
parser.add_argument('--pt_x_translate', type=float, default=0)
parser.add_argument('--pt_y_translate', type=float, default=0)
parser.add_argument('--pt_scale_min', type=float, default=1)
parser.add_argument('--pt_scale_max', type=float, default=1)


#WSDAN
parser.add_argument('--wsdan_beta', type=float, default=5e-2)
parser.add_argument('--wsdan_num_attentions', type=float, default=32)
parser.add_argument('--bap_alpha', default=0.95, type=float, help='weight for BAP loss')
#GENERATOR PARAMETERS

#Rotation XY
parser.add_argument('--rotXY', type=str2bool, default=False)
parser.add_argument('--rotXYminX', type=float, default=0)
parser.add_argument('--rotXYmaxX', type=float, default=0)
parser.add_argument('--rotXYminY', type=float, default=0)
parser.add_argument('--rotXYmaxY', type=float, default=0)

#Rotation Z
parser.add_argument('--rotZ', type=str2bool, default=False)
parser.add_argument('--rotZmin', type=int, default=0)
parser.add_argument('--rotZmax', type=int, default=0)

#Translate
parser.add_argument('--translate', type=str2bool, default=False)
parser.add_argument('--translateminX', type=float, default=0.5)
parser.add_argument('--translatemaxX', type=float, default=0.5)
parser.add_argument('--translateminY', type=float, default=0.5)
parser.add_argument('--translatemaxY', type=float, default=0.5)

#Scale
parser.add_argument('--scale', type=str2bool, default=False)
parser.add_argument('--scaleminX', type=float, default=1)
parser.add_argument('--scalemaxX', type=float, default=1)
parser.add_argument('--scaleminY', type=float, default=1)
parser.add_argument('--scalemaxY', type=float, default=1)

#Uniform Scale
parser.add_argument('--uniformScale', type=str2bool, default=False)
parser.add_argument('--uniformScalemin', type=float, default=1)
parser.add_argument('--uniformScalemax', type=float, default=1)

parser = pl.Trainer.add_argparse_args(parser)

args = parser.parse_args()

if len(args.gpu) == 0:
    args.gpu = None

print(args)
