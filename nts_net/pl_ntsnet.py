import torch
from sklearn.metrics import confusion_matrix
import utils
import os

import nts_net.config as nts_config
from nts_net.core import model as nts_model
from torch.optim.lr_scheduler import MultiStepLR
from transformation_pipes.Augmentors import TransformationController
from models.pl_base import BasePLModel
from typing import Tuple


class NtsNetLightningModel(BasePLModel):

    def __init__(self, hparams, client, trainer, transformation_controller: TransformationController):
        super().__init__(hparams, client, trainer, transformation_controller)

    def setup_model(self) -> Tuple[torch.nn.Module, int]:
        return nts_model.AttentionNet(top_n=nts_config.PROPOSAL_NUM, classes=len(self.hparams.classes)), 224

    def setup_criterion(self) -> torch.nn.Module:
        return utils.LabelSmoothingLoss(len(self.hparams.classes), self.hparams.smoothing)

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        img, label = batch
        batch_size = img.size(0)

        raw_logits, concat_logits, part_logits, _, top_n_prob = self.forward(img)
        part_loss = nts_model.list_loss(part_logits.view(batch_size * nts_config.PROPOSAL_NUM, -1),
                                        label.unsqueeze(1).repeat(1, nts_config.PROPOSAL_NUM).view(-1)).view(batch_size,
                                                                                                             nts_config.PROPOSAL_NUM)
        raw_loss = self.criterion(raw_logits, label)
        concat_loss = self.criterion(concat_logits, label)
        rank_loss = nts_model.ranking_loss(top_n_prob, part_loss)
        partcls_loss = self.criterion(part_logits.view(batch_size * nts_config.PROPOSAL_NUM, -1),
                                 label.unsqueeze(1).repeat(1, nts_config.PROPOSAL_NUM).view(-1))

        total_loss = raw_loss + rank_loss + concat_loss + partcls_loss

        if self.hparams.debug:
            import pickle
            if not os.path.exists("saved_img_states"):
                os.mkdir("saved_img_states")
            with open(os.path.join("saved_img_states", self.hparams.run_id + "_" +
                                                       str(self.current_epoch) +
                                                       "_" + str(self.sub_epoch) + ".pickle"), "wb") as out_f:
                pickle.dump((img, label), out_f)
            self.sub_epoch += 1

        _, prediction = torch.max(concat_logits, 1)
        if label.is_cuda:

            gold = label.cpu().numpy().astype(int)
        else:
            gold = label.numpy().astype(int)

        self.train_gold.extend(gold)
        self.train_preds.extend(prediction)

        train_acc = utils.get_accuracy(gold, prediction)

        self.log("train_acc", train_acc, prog_bar=True, logger=False)

        return {'loss': total_loss}

    def validation_step(self, val_batch, batch_idx):
        img, label = val_batch
        _, concat_logits, _, _, _ = self.forward(img)
        concat_loss = self.criterion(concat_logits, label)
        _, prediction = torch.max(concat_logits, 1)

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

        return {'val_loss': concat_loss, "predictions": prediction, "gold": gold}

    def validation_epoch_end(self, outputs):

        self.debug_val_sub_epoch = 0

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        all_preds = list()
        all_gold = list()
        for x in outputs:
            all_preds.extend(x['predictions'])
            all_gold.extend(x["gold"])

        # self.logger.log_metrics({"val_accuracy": correct/all_}, step=self.current_epoch)
        if self.init_epoch:
            self.init_epoch = False
        else:
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
        _, concat_logits, _, _, _ = self.forward(img)
        concat_loss = self.criterion(concat_logits, label)
        _, prediction = torch.max(concat_logits, 1)

        sorted, indices = torch.sort(concat_logits, descending=True)
        indices = indices.cpu().numpy().astype(int)[:, :5]

        if label.is_cuda:
            gold = label.cpu().numpy().astype(int)
        else:
            gold = label.numpy().astype(int)

        return {'test_loss': concat_loss, "predictions": prediction, "gold": gold, "top5": indices}

    def configure_optimizers(self):
        raw_parameters = list(self.model.pretrained_model.parameters())
        part_parameters = list(self.model.proposal_net.parameters())
        concat_parameters = list(self.model.concat_net.parameters())
        partcls_parameters = list(self.model.partcls_net.parameters())

        raw_optimizer = torch.optim.SGD(raw_parameters, lr=nts_config.LR, momentum=0.9, weight_decay=nts_config.WD)
        concat_optimizer = torch.optim.SGD(concat_parameters, lr=nts_config.LR, momentum=0.9, weight_decay=nts_config.WD)
        part_optimizer = torch.optim.SGD(part_parameters, lr=nts_config.LR, momentum=0.9, weight_decay=nts_config.WD)
        partcls_optimizer = torch.optim.SGD(partcls_parameters, lr=nts_config.LR, momentum=0.9, weight_decay=nts_config.WD)
        schedulers = [MultiStepLR(raw_optimizer, milestones=self.hparams.opt_step, gamma=0.1),
                      MultiStepLR(concat_optimizer, milestones=self.hparams.opt_step, gamma=0.1),
                      MultiStepLR(part_optimizer, milestones=self.hparams.opt_step, gamma=0.1),
                      MultiStepLR(partcls_optimizer, milestones=self.hparams.opt_step, gamma=0.1)]

        return [raw_optimizer, concat_optimizer, part_optimizer, partcls_optimizer], schedulers
