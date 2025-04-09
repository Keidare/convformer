import os 
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

import pandas as pd
from PIL import Image  # For image loading
from torch.utils.data import Dataset, Subset
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
    image = Image.open(img_path).convert("L")
    y_label = int(self.annotations.iloc[index, 1])

    if self.transform:
      image = self.transform(image)

    # Min-max normalization
    image = np.array(image).astype(np.float32) / 255.0

    return image, y_label
  
  

def prepare_ds(batch_size=16):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor()
    ])

    # Load dataset without applying transforms yet
    full_dataset = BoneDataset(csv_file='data/ground.csv', root_dir='data/bones/', transform=transform)

    # Split dataset
    train_pct = 0.68
    test_pct = 0.20
    val_pct = 0.12

    x_train, x_test_val, y_train, y_test_val = train_test_split(
        full_dataset.annotations['filename'].values,
        full_dataset.annotations['label'].values,
        random_state=0,
        train_size=train_pct
    )

    x_test, x_val, y_test, y_val = train_test_split(
        x_test_val,
        y_test_val,
        random_state=0,
        train_size=test_pct / (test_pct + val_pct)
    )

    train_indices = full_dataset.annotations[full_dataset.annotations['filename'].isin(x_train)].index.tolist()
    test_indices = full_dataset.annotations[full_dataset.annotations['filename'].isin(x_test)].index.tolist()
    val_indices = full_dataset.annotations[full_dataset.annotations['filename'].isin(x_val)].index.tolist()

    # Use Subset to apply the correct split
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
