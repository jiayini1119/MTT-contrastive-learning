"""
Adapted from https://github.com/sjoshi804/sas-data-efficient-contrastive-learning
"""

from enum import Enum
from PIL import Image, ImageFilter
from torchvision import transforms
import kornia

class SupportedDatasets(Enum):
    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"
    MNIST = "mnist"
    TINY_IMAGENET = "tiny_imagenet"
    IMAGENET = "imagenet"
    STL10 = "stl10"

CACHED_MEAN_STD = {
    SupportedDatasets.CIFAR10.value: ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    SupportedDatasets.CIFAR100.value: ((0.5071, 0.4865, 0.4409), (0.2009, 0.1984, 0.2023)),
    SupportedDatasets.MNIST.value:((0.1307,), (0.3081,)),
    SupportedDatasets.STL10.value: ((0.4409, 0.4279, 0.3868), (0.2309, 0.2262, 0.2237)),
    SupportedDatasets.TINY_IMAGENET.value: ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    SupportedDatasets.IMAGENET.value: ((0.485, 0.456, 0.3868), (0.2309, 0.2262, 0.2237))
}

def ColourDistortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort


def BlurOrSharpen(radius=2.):
    blur = GaussianBlur(radius=radius)
    full_transform = transforms.RandomApply([blur], p=.5)
    return full_transform

def KorniaAugmentation(dataset):
    if dataset == SupportedDatasets.MNIST.value:
        return kornia.augmentation.AugmentationSequential(
            kornia.augmentation.RandomResizedCrop((28, 28), scale=(0.08, 1.0), same_on_batch=True, keepdim=True),
            kornia.augmentation.RandomHorizontalFlip(same_on_batch=True, keepdim=True),
            kornia.augmentation.Normalize(*CACHED_MEAN_STD[dataset],keepdim=True),
        )
    else:
        return kornia.augmentation.AugmentationSequential(
            kornia.augmentation.RandomResizedCrop((32, 32), scale=(0.08, 1.0), same_on_batch=True, keepdim=True),
            kornia.augmentation.RandomHorizontalFlip(same_on_batch=True, keepdim=True),
            kornia.augmentation.ColorJiggle(0.4, 0.4, 0.4, 0.1, same_on_batch=True, p=0.8, keepdim=True),
            kornia.augmentation.RandomGrayscale(same_on_batch=True, p=0.2, keepdim=True),
            kornia.augmentation.Normalize(*CACHED_MEAN_STD[dataset],keepdim=True),
        )

def CustomAugmentation(dataset):
    # only for CIFAR 10 and CIFAR 100
    return transforms.Compose([
        transforms.RandomResizedCrop(32, interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        ColourDistortion(s=0.5),
        # transforms.ToTensor(),
        transforms.Normalize(*CACHED_MEAN_STD[dataset]),
    ])


class ImageFilterTransform(object):

    def __init__(self):
        raise NotImplementedError

    def __call__(self, img):
        return img.filter(self.filter)


class GaussianBlur(ImageFilterTransform):

    def __init__(self, radius=2.):
        self.filter = ImageFilter.GaussianBlur(radius=radius)


class Sharpen(ImageFilterTransform):

    def __init__(self):
        self.filter = ImageFilter.SHARPEN
