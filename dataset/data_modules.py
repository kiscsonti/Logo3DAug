from pytorch_lightning import LightningDataModule
from dataloader import get_real_train_data, get_generator_train_data, get_test_data
import copy


class LogoDataModule(LightningDataModule):

    def __init__(self, args, client, augmentor, model):
        super().__init__()
        self.args = args
        self.client = client
        self.model = model
        self.augmentor = augmentor

    def train_dataloader(self):
        if self.args.is_generator:
            return get_generator_train_data(hparams=self.args, client=self.client, model=self.model,
                                            generator=self.augmentor)
        else:
            return get_real_train_data(self.model.input_size, self.args)

    def val_dataloader(self):
        return get_test_data(self.model.input_size, self.args)

    def test_dataloader(self, test_dir=None):
        args_copy = copy.deepcopy(self.args)
        args_copy.test_dir = test_dir
        if self.model is None:
            input_size = 224
        else:
            input_size = self.model.input_size
        return get_test_data(input_size, args_copy)
