import os
import json
import torch
import torchvision
import numpy as np
from PIL import Image
import os
import torch
import torchvision
import numpy as np
from PIL import Image

class Cityscapes(torch.utils.data.Dataset):
    ignore_index = 255
    void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
    valid_classes = [ignore_index, 7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 
                     24, 25, 26, 27, 28, 31, 32, 33]
    class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', 
                   'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', 
                   'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 
                   'motorcycle', 'bicycle']

    # Mapping class indices
    class_map = dict(zip(valid_classes, range(len(valid_classes))))
    n_classes = len(valid_classes)

    label_colours = np.asarray([
        [0, 0, 0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ], dtype=np.uint8)

    def __init__(self, root, split='train', transform=None):
        super().__init__()
        self.transform = transform
        self.dataset = torchvision.datasets.Cityscapes(root, split, 'fine', 'semantic')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.getdata(idx)
        if self.transform:
            image, label = self.transform(image, label)
        return image, label

    def getdata(self, idx):
        image, target = self.dataset[idx]
        image = np.asarray(image)
        target = np.asarray(target)
        
        # Encode labels
        label = self.encode_segmap(target)
        return image, label

    def encode_segmap(self, mask):
        """Remap labels to have only valid classes."""
        mask = np.array(mask, dtype=np.int64)
        
        # Remove unwanted classes
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        
        # Map valid classes
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]

        return mask

    @classmethod
    def get_color(cls, label):
        """Convert grayscale segmentation mask to color."""
        color_mask = np.zeros((*label.shape, 3), dtype=np.uint8)
        valid_mask = label != cls.ignore_index  # Ignore label stays black
        color_mask[valid_mask] = cls.label_colours[label[valid_mask]]
        return color_mask

    @classmethod
    def decode_segmap(cls, mask):
        """Convert grayscale labels to RGB for visualization."""
        mask = mask.numpy() if isinstance(mask, torch.Tensor) else mask
        r = np.zeros_like(mask, dtype=np.uint8)
        g = np.zeros_like(mask, dtype=np.uint8)
        b = np.zeros_like(mask, dtype=np.uint8)

        for l in range(cls.n_classes):
            r[mask == l] = cls.label_colours[l][0]
            g[mask == l] = cls.label_colours[l][1]
            b[mask == l] = cls.label_colours[l][2]

        return np.stack([r, g, b], axis=-1)  # Shape: (H, W, 3)

