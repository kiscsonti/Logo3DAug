import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain, combinations
import pytorch_lightning as pl
from transformation_pipes.AugmentorBase import AugmentorBase


class LabelSmoothLossV2(nn.Module):
    
    def __init__(self, smoothing=0.0):
        super(LabelSmoothLossV2, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
            self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
        

class CenterLoss(nn.Module):
    def __init__(self):
        super(CenterLoss, self).__init__()
        self.l2_loss = nn.MSELoss(reduction='sum')

    def forward(self, outputs, targets):
        return self.l2_loss(outputs, targets) / outputs.size(0)


def check_classes(client, args):
    client_classes = set(client.GetClasses())
    args_classes = set(args.classes)

    missing = (client_classes - args_classes).union(args_classes - client_classes)
    if len(missing) == 0:
        print("Classes are valid!")
    else:
        print("MISSING: ", missing)
    return missing


def setup_client(client, args, transformation_pipeline=None):
    # print("Classes to debug:", client.GetClasses())
    # print("Set classes:", args.classes)

    client_classes = set(client.GetClasses())
    args_classes = set(args.classes)

    missing = (client_classes-args_classes).union(args_classes-client_classes)
    if len(missing) == 0:
        print("Classes are valid!")
    else:
        print("MISSING: ", missing)
    # client.setRandomStrategyToNormalDistribution()
    client.setRandomStrategyToUniformDistribution()

    if transformation_pipeline is None:
        last = get_last_transformation(args)

        if args.translate:
            client.setTranslation(args.translateminX, args.translatemaxX, args.translateminY, args.translatemaxY)
            if last != "translate":
                client.newTransformationBatch()

        if args.rotZ:
            client.setRotationRangeZ(args.rotZmin, args.rotZmax)
            if last != "rotateZ":
                client.newTransformationBatch()

        if args.rotXY:
            client.setRotationRangeXY(args.rotXYminX, args.rotXYmaxX, args.rotXYminY, args.rotXYmaxY)
            if last != "rotateXY":
                client.newTransformationBatch()

        if args.scale:
            client.setScale(args.scaleminX, args.scalemaxX, args.scaleminY, args.scalemaxY)
            if last != "scale":
                client.newTransformationBatch()

        if args.uniformScale:
            client.setUniformScale(args.uniformScalemin, args.uniformScalemax)
            if last != "scaleuniform":
                client.newTransformationBatch()
    else:
        for element in transformation_pipeline:
            if element[1] is None:
                element[0]()
            else:
                element[0](*element[1])
    return client


counter = 0


def kirajzol(images, size_x=4, size_y=4, channel=3):
    global counter
    # item = item.cpu().numpy()
    print("Kirajzol")
    print(images.shape)
    imgs = np.moveaxis(images, 1, 3)
    print(imgs.shape)
    # images = images.reshape(-1, 224, 224, channel)
    fig1, fig1Axes = plt.subplots(size_x, size_y)
    for x in range(0, size_x):
        for y in range(0, size_y):
            # image = imgs[x * size + y]
            # image = np.flipud(image)
            # print(image.shape)
            # print("Ezelott: ", x * size_x + y)
            fig1Axes[x, y].imshow(imgs[x * size_y + y])
            fig1Axes[x, y].axis("off")
    fig1.set_facecolor('black')
    fig1.savefig("kuka/"+str(counter)+".png")
    counter += 1
    plt.show()


def intristic_eval(model, dataloader):
    good = 0
    total = 0
    for imgs, targets in dataloader:
        imgs = imgs.cuda()
        output = model(imgs)
        _, prediction = torch.max(output, 1)
        if targets.is_cuda:
            gold = targets.cpu().numpy().astype(int)
        else:
            gold = targets.numpy().astype(int)
        for a, b in zip(prediction, gold):
            if a == b:
                good += 1
            total += 1

    return good/total


def get_top5_accuracy(all_gold, all_preds):
    correct = 0
    total = 0
    for a, b in zip(all_preds, all_gold):
        # print("Top5: ", a)
        # print("To be matched to: ", b)
        if b in a:
            correct += 1
        total += 1
    return correct/total


def get_accuracy(gold, pred):
    if len(gold) != 0:
        correct = 0
        for a, b in zip(gold, pred):
            if a == b:
                correct += 1
        return correct / len(gold)
    else:
        return 0


def intristic_eval_w_top5(model, dataloader):
    good = 0
    top5_good = 0
    total = 0
    for imgs, targets in dataloader:
        # imgs = imgs.cuda()
        output = model(imgs)
        sorted, indices = torch.sort(output)
        indices = indices.cpu().numpy().astype(int)[:, :5]
        _, prediction = torch.max(output, 1)
        if targets.is_cuda:
            gold = targets.cpu().numpy().astype(int)
        else:
            gold = targets.numpy().astype(int)
        for a, b in zip(prediction, gold):
            if a == b:
                good += 1
            total += 1
        for a, b in zip(indices, gold):
            if b in a:
                top5_good += 1

    return good/total, top5_good/total


def custom_evaluate(model, dataloader, class_to_idx):

    good = 0
    total = 0

    idx_to_class = dict()
    for key, value in class_to_idx.items():
        idx_to_class[value] = key

    dl_idx_to_class = dict()
    for key, value in dataloader.dataset.class_to_idx.items():
        dl_idx_to_class[value] = key

    target_correct_labeling = dict()
    # print("Gold keys: ", class_to_idx.keys())
    # print("DL keys: ", dataloader.dataset.class_to_idx.keys())
    for key in class_to_idx.keys():
        target_correct_labeling[dataloader.dataset.class_to_idx[key]] = class_to_idx[key]

    for imgs, targets in dataloader:
        output = model(imgs)

        _, prediction = torch.max(output, 1)
        if targets.is_cuda:

            gold = targets.cpu().numpy().astype(int)
        else:
            gold = targets.numpy().astype(int)

        for a, b in zip(prediction, gold):
            if a == target_correct_labeling[b]:
                good += 1
            total += 1

    return good/total


def extend_with_instance(fields, prediction, gold, url, idx_to_class, dl_idx_to_class, iteration, prediction_np, correct_labeling_map):

    if "path" in fields:
        fields["path"].append(os.sep.join(os.path.normpath(url).split(os.sep)[-2:]))
    else:
        fields["path"] = [os.sep.join(os.path.normpath(url).split(os.sep)[-2:])]

    if "pred" in fields:
        fields["pred"].append(dl_idx_to_class[prediction])
    else:
        fields["pred"] = [dl_idx_to_class[prediction]]

    if "gold" in fields:
        fields["gold"].append(idx_to_class[gold])
    else:
        fields["gold"] = [idx_to_class[gold]]

    misclass = 0 if prediction == correct_labeling_map[gold] else 1
    if "misclassified" in fields:
        fields["misclassified"].append(misclass)
    else:
        fields["misclassified"] = [misclass]

    for c in range(len(prediction_np[iteration])):
        if idx_to_class[c] in fields:
            fields[idx_to_class[c]].append("{:.5f}".format(round(prediction_np[iteration][c], 5)))
        else:
            fields[idx_to_class[c]] = ["{:.5f}".format(round(prediction_np[iteration][c], 5))]


def get_misclassification(model, dataloader, class_to_idx, all=False):

    idx_to_class = dict()
    for key, value in class_to_idx.items():
        idx_to_class[value] = key

    dl_idx_to_class = dict()
    for key, value in dataloader.dataset.class_to_idx.items():
        dl_idx_to_class[value] = key

    target_correct_labeling = dict()
    print("Gold keys: ", class_to_idx.keys())
    print("DL keys: ", dataloader.dataset.class_to_idx.keys())
    for key in class_to_idx.keys():
        target_correct_labeling[dataloader.dataset.class_to_idx[key]] = class_to_idx[key]

    fields = dict()

    for imgs, targets, url in dataloader:
        output = model(imgs)

        _, prediction = torch.max(output, 1)
        if targets.is_cuda:
            prediction = prediction.cpu().numpy()
            prediction_np = output.detach().cpu().numpy()
            gold = targets.cpu().numpy().astype(int)
        else:
            prediction = prediction.numpy()
            prediction_np = output.detach().numpy()
            gold = targets.numpy().astype(int)

        iteration = 0
        for a, b, c in zip(prediction, gold, url):

            if all or (not all and a != target_correct_labeling[b]):
                extend_with_instance(fields, a, b, c, idx_to_class, dl_idx_to_class, iteration, prediction_np, target_correct_labeling)
                iteration += 1

    return fields


def get_last_transformation(args):

    if args.uniformScale:
        return "scaleuniform"
    elif args.scale:
        return "scale"
    elif args.rotXY:
        return "rotateXY"
    elif args.rotZ:
        return "rotateZ"
    elif args.translate:
        return "translate"
    else:
        return "notransform"


def powerset(iterable, empty=True):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    start = 0 if empty else 1
    return chain.from_iterable(combinations(s, r) for r in range(start, len(s)+1))


def get_all_subset_pipeline(available_tfs, padding_tfs, empty=True):
    all_subsets = list(powerset(available_tfs, empty))
    transformation_pipeline = list()
    for i, tf_batch in enumerate(all_subsets):
        if i != 0:
            transformation_pipeline.extend(padding_tfs)
        transformation_pipeline.extend(tf_batch)
    return transformation_pipeline


def get_lit_model(args, client, trainer, transformation_controller) -> pl.LightningModule:
    from models.original_model import LitModel
    from nts_net.pl_ntsnet import NtsNetLightningModel
    from ws_dan.pl_wsdan import WSDANLightningModel
    if args.lit_model.lower() == "nts":
        return NtsNetLightningModel(args, client, trainer, transformation_controller)
    elif args.lit_model.lower() == "wsdan":
        return WSDANLightningModel(args, client, trainer, transformation_controller)
    else:
        return LitModel(args, client, trainer, transformation_controller)


def get_augmentor(args, client, transformation_pipeline=None) -> AugmentorBase:
    from transformation_pipes.Augmentors import RandAugment
    from transformation_pipes.Augmentors import TransformationController
    if args.randaug:
        return RandAugment(client, args.randaug_N, args.randaug_M, args.randaug_augs)
    else:
        assert transformation_pipeline is not None
        return TransformationController(client, transformation_pipeline)


def load_model(args, model):
    checkpoint = torch.load(
        args.ckpt,
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
