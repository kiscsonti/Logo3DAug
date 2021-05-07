import os
import random
from typing import Tuple, Union, List

import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

import utils
from models.pl_base import BasePLModel
from transformation_pipes.Augmentors import TransformationController
from utils import CenterLoss, LabelSmoothingLoss

from ws_dan.model.inception_bap import inception_v3_bap
from ws_dan.model.resnet import resnet50

from ws_dan.utils import calculate_pooling_center_loss, mask2bbox
from ws_dan.utils import attention_crop, attention_drop, attention_crop_drop
from torch import nn


class WSDANWrapper(nn.Module):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.feature_center = torch.zeros(len(self.hparams.classes),
                                          self.hparams.wsdan_num_attentions * 512
                                          )
        self.model = self._get_model()

    def _get_model(self):

        if self.hparams.model_name == 'inception':
            net = inception_v3_bap(pretrained=True, aux_logits=False, num_parts=self.hparams.wsdan_num_attentions)
        elif self.hparams.model_name == 'resnet50':
            net = resnet50(pretrained=True, use_bap=True)
        else:
            raise NotImplementedError
        in_features = net.fc_new.in_features
        new_linear = torch.nn.Linear(
            in_features=in_features, out_features=len(self.hparams.classes))
        net.fc_new = new_linear
        return net

    def forward(self, x):
        if x.device != self.feature_center.device:
            self.feature_center = self.feature_center.to(x.device)
            print("Fasz kivan!", x.device, " != ", self.feature_center.device)
        return self.model(x)


class WSDANLightningModel(BasePLModel):

    def __init__(self, hparams, client, trainer, transformation_controller: TransformationController):
        super().__init__(hparams, client, trainer, transformation_controller)

    def setup_model(self) -> Tuple[torch.nn.Module, int]:
        return WSDANWrapper(hparams=self.hparams), 224

    def setup_criterion(self) -> Union[List[torch.nn.Module], torch.nn.Module]:
        return utils.LabelSmoothingLoss(len(self.hparams.classes), self.hparams.smoothing)

    def on_epoch_start(self):
        # if self.init_epoch:
        #     self.feature_center = self.feature_center.to(self.device)
        super().on_epoch_start()

    def training_step(self, batch, batch_idx, optimizer_idx=None):

        img, label = batch
        # compute output
        attention_maps, raw_features, output1 = self.forward(img)
        features = raw_features.reshape(raw_features.shape[0], -1)

        feature_center_loss, center_diff = calculate_pooling_center_loss(
            features, self.model.feature_center, label, alfa=self.hparams.bap_alpha)

        center_diff = center_diff.to(self.device)
        # update model.centers
        self.model.feature_center[label] += center_diff

        # compute refined loss
        # img_drop = attention_drop(attention_maps,input)
        # img_crop = attention_crop(attention_maps, input)
        img_crop, img_drop = attention_crop_drop(attention_maps, img)
        _, _, output2 = self.forward(img_drop)
        _, _, output3 = self.forward(img_crop)

        loss1 = self.criterion(output1, label)
        loss2 = self.criterion(output2, label)
        loss3 = self.criterion(output3, label)

        loss = (loss1+loss2+loss3)/3 + feature_center_loss

        _, prediction = torch.max(output1, 1)

        if self.hparams.debug:
            import pickle
            if not os.path.exists("saved_img_states"):
                os.mkdir("saved_img_states")
            with open(os.path.join("saved_img_states", self.hparams.run_id + "_" +
                                                       str(self.current_epoch) +
                                                       "_" + str(self.sub_epoch) + ".pickle"), "wb") as out_f:
                pickle.dump((img, label), out_f)
            self.sub_epoch += 1

        if label.is_cuda:

            gold = label.cpu().numpy().astype(int)
        else:
            gold = label.numpy().astype(int)

        self.train_gold.extend(gold)
        self.train_preds.extend(prediction)

        train_acc = utils.get_accuracy(gold, prediction)

        self.log("train_acc", train_acc, prog_bar=True, logger=False)

        return {'loss': loss}

    def batch_augment(self, images, attention_map, mode='crop', theta=0.5, padding_ratio=0.1):
        batches, _, imgH, imgW = images.size()

        if mode == 'crop':
            crop_images = []
            for batch_index in range(batches):
                atten_map = attention_map[batch_index:batch_index + 1]
                if isinstance(theta, tuple):
                    theta_c = random.uniform(*theta) * atten_map.max()
                else:
                    theta_c = theta * atten_map.max()

                crop_mask = F.interpolate(atten_map, size=(imgH, imgW), mode='bilinear', align_corners=True) >= theta_c
                nonzero_indices = torch.nonzero(crop_mask[0, 0, ...])
                height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
                height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
                width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
                width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)

                crop_images.append(
                    F.interpolate(
                        images[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max],
                        size=(imgH, imgW), mode='bilinear', align_corners=True))
            crop_images = torch.cat(crop_images, dim=0)
            return crop_images

        elif mode == 'drop':
            drop_masks = []
            for batch_index in range(batches):
                atten_map = attention_map[batch_index:batch_index + 1]
                if isinstance(theta, tuple):
                    theta_d = random.uniform(*theta) * atten_map.max()
                else:
                    theta_d = theta * atten_map.max()

                drop_masks.append(F.interpolate(atten_map, size=(imgH, imgW), mode='bilinear', align_corners=True) < theta_d)
            drop_masks = torch.cat(drop_masks, dim=0)
            drop_images = images * drop_masks.float()
            return drop_images

        else:
            raise ValueError(
                'Expected mode in [\'crop\', \'drop\'], but received unsupported augmentation method %s' % mode)

    def validation_step(self, val_batch, batch_idx):
        img, label = val_batch
        # compute output
        attention_maps, raw_features, output1 = self.forward(img)
        features = raw_features.reshape(raw_features.shape[0], -1)

        feature_center_loss, _ = calculate_pooling_center_loss(
            features, self.model.feature_center, label, alfa=self.hparams.bap_alpha)

        # compute refined loss
        # img_drop = attention_drop(attention_maps,input)
        # img_crop = attention_crop(attention_maps, input)
        img_crop, img_drop = attention_crop_drop(attention_maps, img)
        _, _, output2 = self.forward(img_drop)
        _, _, output3 = self.forward(img_crop)

        loss1 = self.criterion(output1, label)
        loss2 = self.criterion(output2, label)
        loss3 = self.criterion(output3, label)

        loss = (loss1+loss2+loss3)/3 + feature_center_loss

        if self.hparams.debug:
            import pickle
            if not os.path.exists("saved_img_states"):
                os.mkdir("saved_img_states")
            with open(os.path.join("saved_img_states", "VALIDATION_" +
                                                       self.hparams.run_id + "_" +
                                                       str(self.debug_val_sub_epoch) + ".pickle"), "wb") as out_f:
                pickle.dump((img, label), out_f)
            self.debug_val_sub_epoch += 1

        if label.is_cuda:
            gold = label.cpu().numpy().astype(int)
        else:
            gold = label.numpy().astype(int)

        _, prediction = torch.max(output1, 1)
        return {'val_loss': loss, "predictions": prediction, "gold": gold}

    def validation_epoch_end(self, outputs):

        self.debug_val_sub_epoch = 0

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        all_preds = list()
        all_gold = list()
        for x in outputs:
            all_preds.extend(x['predictions'])
            all_gold.extend(x["gold"])

        # self.logger.log_metrics({"val_accuracy": correct/all_}, step=self.current_epoch)
        if not self.init_epoch:
            # self.init_epoch = False
            # self.feature_center = self.feature_center.to(self.device)
        # else:
            self.logger.log_metrics({"val_loss": avg_loss.item()}, step=self.current_epoch)
            self.logger.log_metrics({"val_acc": utils.get_accuracy(all_gold, all_preds)}, step=self.current_epoch)
            # self.logger.log_metrics({"confusion": confusion_matrix(all_gold, all_preds)}, step=self.current_epoch)

            conf = confusion_matrix([item.item() for item in all_gold], [item.item() for item in all_preds])
            conf_file_name = os.path.join("conf_matrices", "confusion_matrix_epoch_" + str(self.current_epoch) + ".txt")
            classes = ",,".join([item[0] for item in sorted(self.trainer.train_dataloader.dataset.class_to_idx.items(),
                                                            key=lambda x: x[1])])
            with open(conf_file_name, "w") as out_f:
                out_f.write(str(conf))
                out_f.write("\r\n")
                out_f.write(classes)
            self.logger.experiment.log_artifact(os.path.join(self.logger.run_id), conf_file_name,
                                                "confusion_matrices")

        if self.hparams.notf_log:
            self.log_notf()
        val_acc = utils.get_accuracy(all_gold, all_preds)
        val_acc = torch.tensor(val_acc, dtype=torch.float64)

        self.log("val_acc", val_acc, prog_bar=True, logger=False)
        self.log("val_loss", avg_loss, prog_bar=True)

    def test_step(self, batch, batch_idx):

        img, label = batch
        # forward
        attention_maps, _, output1 = self.forward(img)
        refined_input = mask2bbox(attention_maps, img)
        _, _, output2 = self.forward(refined_input)
        output = (F.softmax(output1, dim=-1) + F.softmax(output2, dim=-1)) / 2

        batch_loss = self.criterion(output2, label)
        _, prediction = torch.max(output, 1)

        sorted, indices = torch.sort(output, descending=True)
        indices = indices.cpu().numpy().astype(int)[:, :5]

        if label.is_cuda:
            gold = label.cpu().numpy().astype(int)
        else:
            gold = label.numpy().astype(int)

        return {'test_loss': batch_loss, "predictions": prediction, "gold": gold, "top5": indices}
