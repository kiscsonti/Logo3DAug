import os
import shutil
import timeit
from abc import abstractmethod
from typing import Tuple, List, Union

import numpy as np
import pytorch_lightning as pl
import torch
from adabound.adabound import AdaBound
from sklearn.metrics import confusion_matrix

import utils
from dataloader import get_notf_dataloader
from transformation_pipes.Augmentors import TransformationController
from dataloader import get_real_train_data, get_generator_train_data, get_test_data
import copy


class BasePLModel(pl.LightningModule):

    def __init__(self, hparams, client, trainer, transformation_controller: TransformationController):
        super(BasePLModel, self).__init__()
        self.transformation_controller = transformation_controller
        self.trainer = trainer
        self.client = client
        self.hparams = hparams

        self.is_generator = hparams.is_generator
        self.train_preds = list()
        self.train_gold = list()
        self.model, self.input_size = self.setup_model()
        self.sanity_check = True
        self.init_epoch = True
        self.stop_gen = False
        self.criterion = self.setup_criterion()

        self.debug_val_sub_epoch = 0
        # set_parameter_requires_grad(self.model, hparams.fine_tune)

        self.freeze_layers()

        if os.path.exists("conf_matrices"):
            shutil.rmtree("conf_matrices")
        os.mkdir("conf_matrices")

        if hparams.notf_log:
            self.notf_dl = get_notf_dataloader(self.input_size, self.hparams)

    def freeze_layers(self):
        print("Params to learn:")
        self.params_to_update = self.model.parameters()
        if self.hparams.fine_tune:
            self.params_to_update = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.params_to_update.append(param)
                    print("\t", name)
        else:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    print("\t", name)

    @abstractmethod
    def setup_model(self) -> Tuple[torch.nn.Module, int]:
        pass

    @abstractmethod
    def setup_criterion(self) -> Union[List[torch.nn.Module], torch.nn.Module]:
        pass

    def forward(self, images):
        return self.model(images)

    def on_epoch_start(self):
        lr = 0
        for param_group in self.trainer.optimizers[0].param_groups:
            lr = param_group["lr"]
        self.logger.log_metrics({"learning rate": lr}, step=self.current_epoch)
        self.epoch_start = timeit.default_timer()

        if self.init_epoch:
            self.init_epoch = False
            with open("conf_matrices/classes.txt", "w") as out_f:
                x = sorted(self.trainer.train_dataloader.dataset.class_to_idx.items(), key=lambda y: y[1])
                out_f.write(" ".join([item[0] for item in x]))
            self.logger.experiment.log_artifact(os.path.join(self.logger.run_id), "conf_matrices/classes.txt")
        self.sub_epoch = 0

    def on_epoch_end(self):
        acc = utils.get_accuracy(self.train_gold, self.train_preds)
        self.logger.log_metrics({"epoch_train_acc": acc}, step=self.current_epoch)
        self.train_preds = list()
        self.train_gold = list()
        self.logger.log_metrics({"train_time": timeit.default_timer() - self.epoch_start}, step=self.current_epoch)
        if self.is_generator:
            self.trainer.train_dataloader.dataset.reset()

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        x, y = batch
        y_hat = self.forward(x)

        if self.hparams.debug:
            import pickle
            if not os.path.exists("saved_img_states"):
                os.mkdir("saved_img_states")
            with open(os.path.join("saved_img_states", self.hparams.run_id + "_" +
                                                       str(self.current_epoch) +
                                                       "_" + str(self.sub_epoch) + ".pickle"), "wb") as out_f:
                pickle.dump((x, y), out_f)
            self.sub_epoch += 1

        _, prediction = torch.max(y_hat, 1)
        if y.is_cuda:

            gold = y.cpu().numpy().astype(int)
        else:
            gold = y.numpy().astype(int)

        self.train_gold.extend(gold)
        self.train_preds.extend(prediction)

        train_acc = utils.get_accuracy(gold, prediction)

        self.log("train_acc", train_acc, prog_bar=True, logger=False)

        return {'loss': self.criterion(y_hat, y)}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self.forward(x)
        _, prediction = torch.max(y_hat, 1)

        if self.hparams.debug:
            import pickle
            if not os.path.exists("saved_img_states"):
                os.mkdir("saved_img_states")
            with open(os.path.join("saved_img_states", "VALIDATION_" +
                                                       self.hparams.run_id + "_" +
                                                       str(self.debug_val_sub_epoch) + ".pickle"), "wb") as out_f:
                pickle.dump((x, y), out_f)
            self.debug_val_sub_epoch += 1

        if y.is_cuda:
            gold = y.cpu().numpy().astype(int)
        else:
            gold = y.numpy().astype(int)

        loss = self.criterion(y_hat, y)

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

    def log_notf(self):
        accuracy = utils.intristic_eval(self, self.notf_dl)
        self.logger.log_metrics({"notf_accuracy": accuracy}, step=self.current_epoch)
        # return {'val_loss': avg_loss, "log": {"val_F1": torch.tensor(f1), "val_loss": avg_loss}}
        # return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        _, prediction = torch.max(y_hat, 1)

        sorted, indices = torch.sort(y_hat, descending=True)
        indices = indices.cpu().numpy().astype(int)[:, :5]

        if y.is_cuda:
            gold = y.cpu().numpy().astype(int)
        else:
            gold = y.numpy().astype(int)

        loss = self.criterion(y_hat, y)
        return {'test_loss': loss, "predictions": prediction, "gold": gold, "top5": indices}

    def test_epoch_end(self, outputs):

        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        all_preds = list()
        all_gold = list()
        top5s = None
        for x in outputs:
            all_preds.extend(x['predictions'])
            all_gold.extend(x["gold"])
            if top5s is None:
                top5s = x["top5"]
            else:
                top5s = np.concatenate((top5s, x["top5"]), axis=0)
        conf = confusion_matrix([item.item() for item in all_gold], [item.item() for item in all_preds])

        test_acc = utils.get_accuracy(all_gold, all_preds)
        test_acc = torch.tensor(test_acc, dtype=torch.float64)
        test_top5_acc = utils.get_top5_accuracy(all_gold=all_gold, all_preds=top5s)

        self.log("test_acc", test_acc, prog_bar=True)
        self.log("test_loss", avg_loss, prog_bar=True)
        self.log("test_top5_acc", test_top5_acc, prog_bar=True)

        return {'test_loss': avg_loss, "test_acc": test_acc, "test_top5_acc": test_top5_acc, "conf": conf}

    def prepare_data(self):
        pass

    def configure_optimizers(self):
        if self.hparams.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.params_to_update, lr=self.hparams.lr, momentum=0.9, weight_decay=1e-4)
            scheduler = self.get_scheduler(optimizer)
            return [optimizer], [
                scheduler
            ]

        elif self.hparams.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.params_to_update, lr=self.hparams.lr)
            return [optimizer], [
                torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=self.hparams.opt_patience)
            ]

        elif self.hparams.optimizer == "AdaBound":
            optimizer = AdaBound(self.params_to_update, lr=self.hparams.lr, final_lr=self.hparams.lr, weight_decay=1e-4)
            scheduler = self.get_scheduler(optimizer)
            return [optimizer], [
                scheduler
            ]
        else:
            raise ValueError("Bad argument --optimizer " + self.hparams.optimizer + ". Please choose from [Adam, SGD]")

    def get_scheduler(self, optimizer):
        if len(self.hparams.opt_step) != 0:
            return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.hparams.opt_step, gamma=0.1)
        else:
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=self.hparams.opt_patience)

    def train_dataloader(self):
        if self.hparams.is_generator:
            return get_generator_train_data(hparams=self.hparams, client=self.client, model=self,
                                            generator=self.transformation_controller)
        else:
            return get_real_train_data(self.input_size, self.hparams)

    def val_dataloader(self):
        return get_test_data(self.input_size, self.hparams)

    def test_dataloader(self):
        args_copy = copy.deepcopy(self.hparams)
        args_copy.test_dir = None
        return get_test_data(self.input_size, args_copy)

