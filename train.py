#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.profiler import AdvancedProfiler
from dataloader import get_test_data
from config import args
from transformation_pipes.augment_defs import augmentations
from dataset.data_modules import LogoDataModule
import utils

import random
import torch

import SyntheticImageGeneratorClient
import numpy as np
import time

seed = args.seed
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join("./saved_models/", args.run_id + "_{epoch}"),
    save_top_k=1,
    verbose=True,
    monitor='val_acc',
    mode='max',
    prefix='',
    save_last=True
)

early_stop_callback = EarlyStopping(
    monitor='val_loss',
    # monitor='val_acc',
    min_delta=0.00,
    patience=args.earlystop,
    verbose=True,
    # mode='max',
    mode='min',
)

profiler = AdvancedProfiler(output_filename="stats.log")
# s_profiler = BaseProfiler()
os.environ["MLFLOW_RUN_ID"] = args.run_id
mlf_logger = MLFlowLogger(
    experiment_name="Testing",
    tracking_uri="file:{}".format(args.save_dir),
)

client = SyntheticImageGeneratorClient.Client(port=args.port)
with client:

    classes = [item[0] for item in sorted(get_test_data(224, args).dataset.class_to_idx.items(), key=lambda x: int(x[1]))]
    args.classes = list(classes)
    print("CLASSES: ", args.classes)

    # utils.setup_client(client, args)
    utils.check_classes(client, args)

    transform_pipeline = augmentations[args.aug_preset](client)
    transform_controller = utils.get_augmentor(args, client, transform_pipeline)

    timeStart = time.time()

    mlf_logger.log_hyperparams(args)

    trainer = pl.Trainer(gpus=args.gpu,
                         max_epochs=args.epoch,
                         check_val_every_n_epoch=args.val_n,
                         logger=mlf_logger,
                         profiler=profiler,
                         callbacks=[early_stop_callback, checkpoint_callback]
                         )

    model = utils.get_lit_model(args, client, trainer, transform_controller)

    if args.ckpt is not None:
        utils.load_model(args, model)

    data_module = LogoDataModule(args, client, transform_controller, model)

    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()

    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)

    mlf_logger.experiment.log_artifact(mlf_logger.run_id, checkpoint_callback.kth_best_model_path)
    mlf_logger.experiment.log_artifact(mlf_logger.run_id, os.path.join(*args.save_path.split(os.path.sep)[:-1], "last.ckpt"))
    train_time = time.time() - timeStart
    print("Time elapsed:", train_time)
    mlf_logger.log_hyperparams({"train_time_took": train_time})
    if args.is_generator:
        train_dataloader.dataset.stop_generating()

# trainer.test(model)

