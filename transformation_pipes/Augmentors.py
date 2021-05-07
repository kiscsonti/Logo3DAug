import random
from inspect import signature
from typing import List, Tuple
from transformation_pipes import transformations as tf
from SyntheticImageGeneratorClient import Client
from transformation_pipes.AugmentorBase import AugmentorBase
import itertools


class RandAugment(AugmentorBase):

    def __init__(self, client: Client, n, m, aug_list: str):

        self.client = client
        self.n = n
        self.m = m
        self.dataset = None

        self.pt_aug_list = [(item, "pt") for item in [tf.AutoContrast,
                                                      tf.Brightness,
                                                      tf.Color,
                                                      tf.Contrast,
                                                      tf.Cutout,
                                                      # tf.CutoutAbs,
                                                      tf.Equalize,
                                                      # tf.Flip,
                                                      # tf.Identity,  # I dont think this is needed for us
                                                      tf.Invert,
                                                      tf.Posterize,
                                                      tf.TranslateY,
                                                      # tf.TranslateYabs,
                                                      tf.TranslateX,
                                                      # tf.TranslateXabs,
                                                      # tf.SolarizeAdd,
                                                      tf.Solarize,
                                                      tf.ShearY,
                                                      tf.ShearX,
                                                      tf.Sharpness,
                                                      # tf.SamplePairing,
                                                      tf.Rotate]]
        self.generator_aug_list = [(item, "gen") for item in [self.client.LogoTranslation,
                                   self.client.LogoRotationZ,
                                   self.client.LogoRotationXY,
                                   self.client.LogoScale, self.client.LogoScaleTrigonometric, self.client.LogoScaleUniform]]
        self.augment_list = self.init_aug_list(aug_list)
        self.tf_batches = None
        self.tf_gen_batches = None
        self.generate_transformation_batches()
        assert self.tf_batches is not None
        assert self.tf_gen_batches is not None
        assert len(self.tf_batches) == len(self.tf_gen_batches)

        print("RandAugment augmentor has been created!")

    def init_aug_list(self, aug_list: str) -> List:
        if aug_list == "all":
            return self.pt_aug_list + self.generator_aug_list
        elif aug_list == "pt":
            return self.pt_aug_list
        elif aug_list == "gen":
            return self.generator_aug_list
        else:
            raise Exception("Invalid argument for randaug_augs: {}".format(aug_list))

    def generate_transformation_batches(self):
        self.tf_batches = list(itertools.combinations(self.augment_list, self.n))
        # print("TMP: ", self.tf_batches)
        self.setup_gen_batches()

        tmp = list()
        for tf_batch in self.tf_batches:
            b = list()
            for item in tf_batch:
                if item[1] == "pt":
                    b.append(item[0])
            tmp.append(b)
        self.tf_batches = tmp

    def setup_gen_batches(self):
        gen_batches = list()
        for tf_batch in self.tf_batches:
            gen_b = list()
            for item in tf_batch:
                # print(item)
                if item[1] == "gen":
                    gen_b.append(item[0])
            gen_batches.append(gen_b)
        self.tf_gen_batches = gen_batches

    def get_generator_transform_ids(self, class_name: str) -> Tuple[List, int]:
        batch_index = random.randrange(0, len(self.tf_batches))
        random_trans_batch = self.tf_gen_batches[batch_index]
        trafo_bytes = [self.client.LogoClass(class_name)]
        for item in random_trans_batch:
            trafo_bytes.extend(self.apply_tf_gen(item))
        return trafo_bytes, batch_index

    def apply_tf_gen(self, client_function):
        sig = signature(client_function)

        if " Client.LogoTranslation " in str(client_function):
            # print("setTranslation")
            magnitude = 0.045 * self.m  # + 0.5
            lower_bound = -magnitude
            upper_bound = magnitude
            return [client_function(*[get_random_val((lower_bound, upper_bound)) for i in range(len(sig.parameters))]),
                    self.client.LogoLookAtCamera()]

        elif "Client.LogoRotationZ" in str(client_function):
            # print("setRotationRangeZ")
            magnitude = 18 * self.m
            lower_bound = -magnitude
            upper_bound = magnitude
            return [client_function(*[get_random_val((lower_bound, upper_bound)) for i in range(len(sig.parameters))])]

        elif "Client.LogoRotationXY" in str(client_function):
            # print("setRotationRangeXY")
            magnitude = 7 * self.m
            lower_bound = 0
            upper_bound = magnitude
            return [client_function(*[get_random_val((lower_bound, upper_bound)) for i in range(len(sig.parameters))])]

        elif "Client.LogoScaleTrigonometric" in str(client_function):
            # print("setScalePolarAngle")
            magnitude = 3.5 * self.m
            lower_bound = 45 - magnitude
            upper_bound = 45 + magnitude
            return [client_function(*[get_random_val((lower_bound, upper_bound)) for i in range(len(sig.parameters))])]

        elif "Client.LogoScaleUniform" in str(client_function):
            # print("setUniformScale")
            magnitude = 0.08 * self.m
            lower_bound = 1 - magnitude
            upper_bound = 1 + magnitude
            return [client_function(*[get_random_val((lower_bound, upper_bound)) for i in range(len(sig.parameters))])]

        elif "Client.LogoScale" in str(client_function):
            # print("setScale")
            magnitude = 0.08 * self.m
            lower_bound = 1 - magnitude
            upper_bound = 1 + magnitude
            return [client_function(*[get_random_val((lower_bound, upper_bound)) for i in range(len(sig.parameters))])]

        raise ValueError("Wrong parameter as client function to apply")

    def apply_tf_pt(self, pt_tf, img):

        if " AutoContrast " in str(pt_tf):
            # print("AutoContrast")
            return pt_tf(img, 0)
        elif " Equalize " in str(pt_tf):
            # print("Equalize")
            return pt_tf(img, 0)
        elif " Flip " in str(pt_tf):
            # print(str(pt_tf))
            return pt_tf(img, 0)
        elif " Invert " in str(pt_tf):
            # print(str(pt_tf))
            return pt_tf(img, 0)
        elif " Brightness " in str(pt_tf):
            # print(str(pt_tf))
            magnitude = 0.09 * self.m
            lower_bound = 1 - magnitude
            upper_bound = 1 + magnitude
            rand_val = get_random_val((lower_bound, upper_bound))
            return pt_tf(img, rand_val)
        elif " Color " in str(pt_tf):
            # print(str(pt_tf))
            magnitude = 0.09 * self.m
            lower_bound = 1 - magnitude
            upper_bound = 1 + magnitude
            rand_val = get_random_val((lower_bound, upper_bound))
            return pt_tf(img, rand_val)
        elif " Contrast " in str(pt_tf):
            magnitude = 0.09 * self.m
            lower_bound = 1 - magnitude
            upper_bound = 1 + magnitude
            # print(str(pt_tf))
            rand_val = get_random_val((lower_bound, upper_bound))
            return pt_tf(img, rand_val)
        # elif " CutoutAbs " in str(pt_tf):
        #     return pt_tf(img, 0)
        elif " Cutout " in str(pt_tf):
            # print(str(pt_tf))
            magnitude = 0.02 * self.m
            lower_bound = 0
            upper_bound = magnitude
            rand_val = get_random_val((lower_bound, upper_bound))
            return pt_tf(img, rand_val)
        elif " Posterize " in str(pt_tf):
            # print(str(pt_tf))
            magnitude = 0.4 * self.m
            lower_bound = 8 - magnitude
            upper_bound = 8
            rand_val = get_random_val((lower_bound, upper_bound))
            return pt_tf(img, rand_val)
        elif " TranslateY " in str(pt_tf):
            # print(str(pt_tf))
            magnitude = 0.045 * self.m
            lower_bound = -magnitude
            upper_bound = magnitude
            rand_val = get_random_val((lower_bound, upper_bound))
            return pt_tf(img, rand_val)
        # elif " TranslateYabs " in str(pt_tf):
        #     return pt_tf(img, 0)
        elif " TranslateX " in str(pt_tf):
            # print(str(pt_tf))
            magnitude = 0.045 * self.m
            lower_bound = -magnitude
            upper_bound = magnitude
            rand_val = get_random_val((lower_bound, upper_bound))
            return pt_tf(img, rand_val)
        # elif " TranslateXabs " in str(pt_tf):
        #     return pt_tf(img, 0)
        elif " SolarizeAdd " in str(pt_tf):
            # print(str(pt_tf))
            lower_bound = 256 - (256/10 * self.m)
            upper_bound = 256
            rand_val = get_random_val((lower_bound, upper_bound))
            return pt_tf(img, rand_val)
        elif " Solarize " in str(pt_tf):
            # print(str(pt_tf))
            lower_bound = 256 - (256/10 * self.m)
            upper_bound = 256
            rand_val = get_random_val((lower_bound, upper_bound))
            return pt_tf(img, rand_val)
        elif " ShearY " in str(pt_tf):
            # print(str(pt_tf))
            magnitude = 0.03 * self.m
            lower_bound = -magnitude
            upper_bound = magnitude
            rand_val = get_random_val((lower_bound, upper_bound))
            return pt_tf(img, rand_val)
        elif " ShearX " in str(pt_tf):
            # print(str(pt_tf))
            magnitude = 0.03 * self.m
            lower_bound = -magnitude
            upper_bound = magnitude
            rand_val = get_random_val((lower_bound, upper_bound))
            return pt_tf(img, rand_val)
        elif " Sharpness " in str(pt_tf):
            # print(str(pt_tf))
            magnitude = 0.09 * self.m
            lower_bound = 1 - magnitude
            upper_bound = 1 + magnitude
            rand_val = get_random_val((lower_bound, upper_bound))
            return pt_tf(img, rand_val)
        # elif " SamplePairing " in str(pt_tf):
        #     return pt_tf(img, 0)
        elif " Rotate " in str(pt_tf):
            # print(str(pt_tf))
            magnitude = 3 * self.m
            lower_bound = -magnitude
            upper_bound = magnitude
            rand_val = get_random_val((lower_bound, upper_bound))
            return pt_tf(img, rand_val)

        raise ValueError("Invalid transformation function {}".format(str(pt_tf)))

    def __call__(self, img, index):
        assert self.dataset is not None
        # print("aug batch: ", self.dataset.tf_ids[index], " -> ", self.tf_batches[self.dataset.tf_ids[index]])
        for pt_tf in self.tf_batches[self.dataset.tf_ids[index]]:
            img = self.apply_tf_pt(pt_tf, img)
        return img


class TransformationController(AugmentorBase):

    def __init__(self, client: Client, transformation_batches: List[List[Tuple]]):
        self.client = client
        self.tf_batches = transformation_batches
        self.dataset = None
        print("TransformationController augmentor has been created!")

    def get_generator_transform_ids(self, class_name: str) -> Tuple[List, int]:
        # TODO - What if no transformation is allowed
        if len(self.tf_batches) == 0:
            return [self.client.LogoClass(class_name)], -1
        batch_index = random.randrange(0, len(self.tf_batches))
        random_trans_batch = self.tf_batches[batch_index]
        return [self.client.LogoClass(class_name)] + [self.call_func_with_random(item[0], item[1]) for item in random_trans_batch], batch_index

    def call_func_with_random(self, client_func, interval: Tuple[float, float]):
        sig = signature(client_func)
        return client_func(*[get_random_val(interval=interval) for i in range(len(sig.parameters))])

    def __call__(self, img, index):
        return img


def get_random_val(interval: Tuple[float, float]):
    return random.uniform(interval[0], interval[1])
