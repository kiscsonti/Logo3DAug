from abc import ABC, abstractmethod


class AugmentorBase(ABC):

    def __call__(self, img, index):
        pass

    def get_generator_transform_ids(self, class_name: str):
        pass
