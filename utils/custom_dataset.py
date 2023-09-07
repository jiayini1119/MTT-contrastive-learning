import numpy as np
import torch
from utils.dataset import *
from utils.supported_dataset import *
from utils.augmentation import CustomAugmentation 
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage  


class CustomDatasetAugment(Dataset):
    def __init__(self, images, device, transform, n_augmentations: 2):
        self.images = images
        self.device = device
        self.transform = transform
        self.n_augmentations = n_augmentations

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        to_pil = ToPILImage()
        pil_img = to_pil(img)
        imgs = []
        for _ in range(self.n_augmentations):
           imgs.append(self.transform(pil_img))
        return imgs

def get_custom_dataset(dataset_images, device, dataset, num_positive=2):
    augmentation = CustomAugmentation(dataset)
    trainset = CustomDatasetAugment(images=dataset_images, device=device, transform=augmentation, n_augmentations=num_positive)

    return trainset

def get_init_syn_data(method: str, dataset: str, ipc: int, path: str):
    if method == "random":
        ori_datasets = get_datasets(dataset)
        return torch.randn(ori_datasets.num_classes * ipc, ori_datasets.channel, ori_datasets.img_size, ori_datasets.img_size)
    elif method == "real":
        ori_datasets = get_datasets(dataset, need_train_ori=True)
        train_set_ori = ori_datasets.trainset_ori
        all_labels = [label for (_, label) in train_set_ori]
        unique_labels = list(set(all_labels))

        sampled_data = []
        for c in unique_labels:
            class_indices = [i for i, (_, label) in enumerate(train_set_ori) if label == c]
            if len(class_indices) >= ipc:
                sampled_indices = np.random.choice(class_indices, ipc, replace=False)
                for i in sampled_indices:
                    sampled_data.append(train_set_ori[i][0])
            else:
                raise ValueError("The class has fewer samples than ipc!")
        return torch.stack(sampled_data)
    elif method == "path":
        try:
            return torch.load(path)
        except ValueError:
            raise ValueError(f"Failed to load images from path: {path}")
    else:
        raise NotImplementedError("Initialize data either with Gaussian noise or real dataset")