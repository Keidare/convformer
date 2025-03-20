import os 
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

import pandas as pd
from PIL import Image  # For image loading
from torch.utils.data import Dataset
from torchvision import transforms  # For image transformations

class BoneDataset(Dataset):
  def __init__(self, csv_file, root_dir, transform=None):
    self.annotations = pd.read_csv(csv_file)
    self.root_dir = root_dir
    self.transform = transform
    self.label_dict = {0: "normal", 1: "abnormal"}  # Assuming label mapping

  def __len__(self):
    return len(self.annotations)

  def __getitem__(self, index):
    img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])

    # Load image using Pillow (PIL Fork) for ViT compatibility
    image = Image.open(img_path).convert('RGB')  # Convert to RGB if needed

    y_label = int(self.annotations.iloc[index, 1])

    if self.transform:
      image = self.transform(image)


    return image, y_label