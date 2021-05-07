import torch
import utils
from transformation_pipes.Augmentors import TransformationController
from models.pl_base import BasePLModel
from typing import Tuple
from models.model_utils import initialize_model


class LitModel(BasePLModel):

    def __init__(self, hparams, client, trainer, transformation_controller: TransformationController):
        super().__init__(hparams, client, trainer, transformation_controller)

    def setup_model(self) -> Tuple[torch.nn.Module, int]:
        return initialize_model(self.hparams.model_name, len(self.hparams.classes), self.hparams.fine_tune,
                                                       self.hparams, use_pretrained=self.hparams.pretrained)

    def setup_criterion(self) -> torch.nn.Module:
        return utils.LabelSmoothingLoss(len(self.hparams.classes), self.hparams.smoothing)

