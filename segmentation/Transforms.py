import sys
sys.path.append('/home/aistudio/workspace/ESPNetv2.pytorch.paddle/utils')
import paddle_aux
import paddle
import numpy as np
import random
import cv2
__author__ = 'Sachin Mehta'
__license__ = 'MIT'
__maintainer__ = 'Sachin Mehta'


class Scale(object):
    """
    Randomly crop and resize the given PIL image with a probability of 0.5
    """

    def __init__(self, wi, he):
        """

        :param wi: width after resizing
        :param he: height after reszing
        """
        self.w = wi
        self.h = he

    def __call__(self, img, label):
        """
        :param img: RGB image
        :param label: semantic label image
        :return: resized images
        """
        img = cv2.resize(img, (self.w, self.h))
        label = cv2.resize(label, (self.w, self.h), interpolation=cv2.
            INTER_NEAREST)
        return [img, label]


class RandomCropResize(object):
    """
    Randomly crop and resize the given PIL image with a probability of 0.5
    """

    def __init__(self, size):
        """
        :param crop_area: area to be cropped (this is the max value and we select between o and crop area
        """
        self.size = size

    def __call__(self, img, label):
        h, w = img.shape[:2]
        x1 = random.randint(0, int(w * 0.1))
        y1 = random.randint(0, int(h * 0.1))
        img_crop = img[y1:h - y1, x1:w - x1]
        label_crop = label[y1:h - y1, x1:w - x1]
        img_crop = cv2.resize(img_crop, self.size)
        label_crop = cv2.resize(label_crop, self.size, interpolation=cv2.
            INTER_NEAREST)
        return img_crop, label_crop


class RandomCrop(object):
    """
    This class if for random cropping
    """

    def __init__(self, cropArea):
        """
        :param cropArea: amount of cropping (in pixels)
        """
        self.crop = cropArea

    def __call__(self, img, label):
        if random.random() < 0.5:
            h, w = img.shape[:2]
            img_crop = img[self.crop:h - self.crop, self.crop:w - self.crop]
            label_crop = label[self.crop:h - self.crop, self.crop:w - self.crop
                ]
            return img_crop, label_crop
        else:
            return [img, label]


class RandomFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __call__(self, image, label):
        if random.random() < 0.5:
            x1 = 0
            if x1 == 0:
                image = cv2.flip(image, 0)
                label = cv2.flip(label, 0)
            else:
                image = cv2.flip(image, 1)
                label = cv2.flip(label, 1)
        return [image, label]


class Normalize(object):
    """Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    """

    def __init__(self, mean, std):
        """
        :param mean: global mean computed from dataset
        :param std: global std computed from dataset
        """
        self.mean = mean
        self.std = std

    def __call__(self, image, label):
        image = image.astype(np.float32)
        for i in range(3):
            image[:, :, (i)] -= self.mean[i]
        for i in range(3):
            image[:, :, (i)] /= self.std[i]
        return [image, label]


class ToTensor(object):
    """
    This class converts the data to tensor so that it can be processed by PyTorch
    """

    def __init__(self, scale=1):
        """
        :param scale: ESPNet-C's output is 1/8th of original image size, so set this parameter accordingly
        """
        self.scale = scale

    def __call__(self, image, label):
        if self.scale != 1:
            h, w = label.shape[:2]
            image = cv2.resize(image, (int(w), int(h)))
            label = cv2.resize(label, (int(w / self.scale), int(h / self.
                scale)), interpolation=cv2.INTER_NEAREST)
        image = image.transpose((2, 0, 1))
        image_tensor = paddle.to_tensor(data=image).div(255)
        label_tensor = paddle.to_tensor(data=np.array(label, dtype=np.int),
            dtype='int64')
        return [image_tensor, label_tensor]


class Compose(object):
    """Composes several transforms together.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args
