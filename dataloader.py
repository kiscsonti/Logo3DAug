import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np
from SyntheticImageGeneratorClient import Client
from typing import List
import threading
import time
import math
from transformation_pipes.Augmentors import RandAugment
from transformation_pipes.AugmentorBase import AugmentorBase


class MemoryDataset(Dataset):

    def __init__(self, client, args, model, generator: AugmentorBase, target_transform=None):
        super().__init__()
        self.generator = generator
        self.generator.dataset = self
        self.stop_generate = False
        self.generation_flag = False
        self.target_transform = target_transform
        self.args = args
        self.model = model
        self.classes = args.classes
        self.class_to_idx = dict()
        for item in self.classes:
            self.class_to_idx[item] = len(self.class_to_idx)
        self.client = client
        # self.transform = transform
        self.transform = get_generator_transform(input_size=self.model.input_size)
        self.reset_counter = 0
        # self.samples = np.moveaxis(data[:, :, :, :3], 0, -1)

        self.samples = list()
        self.tf_ids = list()
        self.generating_thread = self.new_images_request_new(transformation_container=self.tf_ids,
                                                             augmented_container=self.samples,
                                                             images_per_class=self.args.im_per_class,
                                                             generator_client=self.client)
        self.wait_for_generator()
        self.samples = self.samples[0]
        self.tf_ids = self.tf_ids[0]

        if self.args.pregen:
            self.samples2 = list()
            self.tf_ids2 = list()
            self.generating_thread = self.new_images_request_new(transformation_container=self.tf_ids2,
                                                                 augmented_container=self.samples2,
                                                                 images_per_class=self.args.im_per_class,
                                                                 generator_client=self.client)

        self.samples = (np.array(self.samples)
                        .reshape((len(self.classes) * self.args.im_per_class, 224, 224, 4))[:, :, :, :3])
        self.samples = np.flip(self.samples, axis=1)
        # save_samples(self.samples)
        self.targets = list()
        for c in self.classes:
            self.targets.extend([self.class_to_idx[c]] * self.args.im_per_class)
        assert len(self.targets) == len(self.samples)

        if self.args.debug:
            import pickle
            if not os.path.exists("saved_img_states"):
                os.mkdir("saved_img_states")
            with open(os.path.join("saved_img_states", "GENERATOR_" +
                                                       self.args.run_id + "_" +
                                                       str(self.reset_counter) + ".pickle"), "wb") as out_f:
                pickle.dump((self.samples, self.targets), out_f)

        print("Instantiated MemoryDataset")

    def reset(self):
        print("RESET")
        self.reset_counter += 1
        if self.args.pregen:
            self.wait_for_generator()
            self.samples = self.samples2[0]
            self.samples = (np.array(self.samples)
                            .reshape((len(self.classes) * self.args.im_per_class, 224, 224, 4))[:, :, :, :3])
            self.samples = np.flip(self.samples, axis=1)
            self.samples2 = list()
            self.tf_ids2 = list()
            self.generating_thread = self.new_images_request_new(transformation_container=self.tf_ids2,
                                                                 augmented_container=self.samples2,
                                                                 images_per_class=self.args.im_per_class,
                                                                 generator_client=self.client)
        else:
            self.samples = list()
            self.tf_ids = list()

            self.generating_thread = self.new_images_request_new(transformation_container=self.tf_ids,
                                                                 augmented_container=self.samples,
                                                                 images_per_class=self.args.im_per_class,
                                                                 generator_client=self.client)
            self.wait_for_generator()
            self.samples = self.samples[0]
            self.samples = (np.array(self.samples)
                            .reshape((len(self.classes) * self.args.im_per_class, 224, 224, 4))[:, :, :, :3])
            self.samples = np.flip(self.samples, axis=1)

        if self.args.debug:
            import pickle
            if not os.path.exists("saved_img_states"):
                os.mkdir("saved_img_states")
            with open(os.path.join("saved_img_states", "GENERATOR_" +
                                                       self.args.run_id + "_" +
                                                       str(self.reset_counter) + ".pickle"), "wb") as out_f:
                pickle.dump((self.samples, self.targets), out_f)

    def get_train_from_generator(self, first=False):
        if first:
            self.client.RenderImagesDontWait(self.args.im_per_class)

    def stop_generating(self):
        self.stop_generate = True
        self.generating_thread.join()

    def new_images_request_new(self, transformation_container:list, augmented_container: list, images_per_class: int, generator_client: Client):
        image_receiver_thread = threading.Thread(target=self.receive_and_concat_image_batches,
                                                 args=(transformation_container, augmented_container, images_per_class, generator_client))
        image_receiver_thread.start()
        return image_receiver_thread

    def receive_and_concat_image_batches(self, transformation_container: list, augmented_container: list, images_per_class: int, generator_client: Client):
        hardcoded_request_batch = 100
        buffer = bytearray()
        tf_ids_buffer = list()
        self.generation_flag = True
        for sub_batch in range(math.ceil(len(self.classes) / hardcoded_request_batch)):
            sub_batch_classes = self.classes[
                                sub_batch * hardcoded_request_batch:(sub_batch + 1) * hardcoded_request_batch]
            generator_ids, transformation_ids = list(), list()
            for i in range(len(sub_batch_classes) * images_per_class):
                a, b = self.generator.get_generator_transform_ids(sub_batch_classes[i // images_per_class])
                generator_ids.append(a), transformation_ids.append(b)
            generator_client.RenderImagesDontWait(generator_ids)
            if self.stop_generate:
                self.generation_flag = False
                return
            received = generator_client.GetRenderedImages()
            buffer.extend(received)
            tf_ids_buffer.extend(transformation_ids)
        augmented_container.append(buffer)
        transformation_container.append(tf_ids_buffer)
        self.generation_flag = False

    def wait_for_generator(self):
        while self.generation_flag:
            time.sleep(0.5)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        sample, target = self.samples[index], self.targets[index]
        if self.transform is not None:
            # save_img(sample)
            sample = transforms.ToPILImage()(sample)
            sample = self.generator(sample, index=index)
            # print("Generator: ", type(self.generator))
            # save_img(sample)
            sample = self.transform(sample)
            # save_img(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)


def save_samples(imgs, path="randomimg"):
    for item in imgs:
        new_file_id = len(os.listdir(path))
        im = Image.fromarray(item)
        im.save(os.path.join(path, "{}.jpeg".format(new_file_id)))


def save_img(img, path="randomimg"):
    new_file_id = len(os.listdir(path))
    img.save(os.path.join(path, "{}.jpeg".format(new_file_id)))


def get_generator_transform(input_size: int):
    transform_pipeline = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize((input_size, input_size), Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform_pipeline


def get_real_transform(input_size, args, is_train=True):
    if is_train:
        augment = args.augment
        if augment:
            print("augment")
            transform_pipeline = transforms.Compose([
                transforms.Resize((input_size, input_size), Image.BILINEAR),
                transforms.RandomAffine(args.rotate, (args.x_translate, args.y_translate),
                                        (args.scale_min, args.scale_max)),
                transforms.RandomCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            print("no augment")
            transform_pipeline = transforms.Compose([
                transforms.Resize((input_size, input_size), Image.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    else:
        transform_pipeline = transforms.Compose([
            transforms.Resize((input_size, input_size), Image.BILINEAR),
            # transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transform_pipeline


def get_generator_train_data(hparams, client: Client, model, generator):
    trainset = MemoryDataset(client=client, args=hparams, model=model, generator=generator)
    trainloader = DataLoader(dataset=trainset,
                             # batch_size=len(hparams.classes) * hparams.im_per_class,
                             batch_size=hparams.batch,
                             shuffle=True,
                             num_workers=4,
                             drop_last=False)
    return trainloader


def get_real_train_data(input_size, hparams):
    trainset = ImageFolder(root=os.path.join(hparams.path, "train" if hparams.train_dir is None else hparams.train_dir),
                           transform=get_real_transform(input_size, hparams, is_train=False))
    trainloader = DataLoader(dataset=trainset,
                             batch_size=hparams.batch,
                             shuffle=True,
                             num_workers=4,
                             drop_last=False)
    return trainloader


def get_test_data(input_size, hparams):
    print("Validation data folder: ", "test" if hparams.test_dir is None else hparams.test_dir)
    testset = ImageFolder(root=os.path.join(hparams.path, "test" if hparams.test_dir is None else hparams.test_dir),
                          transform=get_real_transform(input_size, hparams, is_train=False))
    testloader = DataLoader(dataset=testset,
                            batch_size=hparams.batch,
                            shuffle=False,
                            num_workers=4,
                            drop_last=False)
    return testloader


def get_notf_dataloader(input_size, args):
    transform_pipeline = transforms.Compose([
        transforms.Resize((input_size, input_size), Image.BILINEAR),
        # transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    testset = ImageFolder(root="/home/kardosp/docker/OnlineGen/Player/logos",
                          # testset = ImageFolder(root="/media/kiscsonti/521493CD1493B289/egyetem/kutatas/Data_Augment/vm_setup/SyntheticImageGeneratorDocker/Player/logos",
                          transform=transform_pipeline)
    testloader = DataLoader(dataset=testset,
                            # batch_size=len(hparams.classes) * hparams.im_per_class,
                            batch_size=args.batch,
                            shuffle=False,
                            num_workers=4,
                            drop_last=False)
    return testloader
