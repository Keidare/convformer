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
  
def prepare_ds(batch_size=32, csv_file='data/GT_Final.csv', root_dir='data/augmented/'):
    import re
    from collections import defaultdict

    # Load CSV
    df = pd.read_csv(csv_file)

    # Extract base names (remove _aug_X etc.)
    def extract_base(name):
        return re.sub(r'_aug_\d+.*$', '', name)

    df['base'] = df['filename'].apply(extract_base)

    # Group by base
    base_groups = df.groupby('base')

    # Build base-level entries with majority label
    base_entries = []
    for base, group in base_groups:
        label = group['label'].mode()[0]  # assume consistent labels
        base_entries.append((base, label, group['filename'].tolist()))

    # Separate by class for stratified splitting
    yes_group = [entry for entry in base_entries if entry[1] == 1]
    no_group = [entry for entry in base_entries if entry[1] == 0]

    def stratified_split(groups, train_pct=0.68, test_pct=0.20, val_pct=0.12):
        from sklearn.model_selection import train_test_split

        bases = [g[0] for g in groups]

        train_bases, temp_bases = train_test_split(bases, train_size=train_pct, random_state=42)
        test_bases, val_bases = train_test_split(temp_bases, train_size=test_pct / (test_pct + val_pct), random_state=42)

        return set(train_bases), set(val_bases), set(test_bases)

    yes_train, yes_val, yes_test = stratified_split(yes_group)
    no_train, no_val, no_test = stratified_split(no_group)

    train_bases = yes_train.union(no_train)
    val_bases = yes_val.union(no_val)
    test_bases = yes_test.union(no_test)

    # Assign filenames to splits
    split_map = {'train': [], 'val': [], 'test': []}
    for base, label, files in base_entries:
        if base in train_bases:
            split_map['train'].extend(files)
        elif base in val_bases:
            split_map['val'].extend(files)
        elif base in test_bases:
            split_map['test'].extend(files)

    df.set_index('filename', inplace=True)

    def get_indices(file_list):
        return [df.index.get_loc(f) for f in file_list if f in df.index]

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor()
    ])
    test_val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Datasets
    dataset_full = BoneDataset(csv_file=csv_file, root_dir=root_dir, transform=None)

    train_dataset = Subset(BoneDataset(csv_file=csv_file, root_dir=root_dir, transform=train_transform),
                           get_indices(split_map['train']))
    val_dataset = Subset(BoneDataset(csv_file=csv_file, root_dir=root_dir, transform=test_val_transform),
                         get_indices(split_map['val']))
    test_dataset = Subset(BoneDataset(csv_file=csv_file, root_dir=root_dir, transform=test_val_transform),
                          get_indices(split_map['test']))

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Optional: print stats
    def count_labels(dataset):
        counts = {0: 0, 1: 0}
        for _, label in dataset:
            counts[label] += 1
        return counts

    train_counts = count_labels(train_dataset)
    val_counts = count_labels(val_dataset)
    test_counts = count_labels(test_dataset)

    print(f"\n📊 Split Statistics:")
    print(f"Train: {len(train_dataset)} images (YES: {train_counts[1]}, NO: {train_counts[0]})")
    print(f"Val:   {len(val_dataset)} images (YES: {val_counts[1]}, NO: {val_counts[0]})")
    print(f"Test:  {len(test_dataset)} images (YES: {test_counts[1]}, NO: {test_counts[0]})")


    return train_loader, val_loader, test_loader

 
def prepare_ds_base(batch_size=32, csv_file='data/ground.csv', root_dir='data/bones/'):
    # Define transforms

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor()
    ])
    test_val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Load full dataset with no transform for splitting
    full_dataset = BoneDataset(csv_file=csv_file, root_dir=root_dir, transform=None)

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

    # Get indices for Subset
    train_indices = full_dataset.annotations[full_dataset.annotations['filename'].isin(x_train)].index.tolist()
    test_indices = full_dataset.annotations[full_dataset.annotations['filename'].isin(x_test)].index.tolist()
    val_indices = full_dataset.annotations[full_dataset.annotations['filename'].isin(x_val)].index.tolist()

    # Create new datasets with appropriate transforms
    train_dataset = BoneDataset(csv_file=csv_file, root_dir=root_dir, transform=train_transform)
    val_dataset = BoneDataset(csv_file=csv_file, root_dir=root_dir, transform=test_val_transform)
    test_dataset = BoneDataset(csv_file=csv_file, root_dir=root_dir, transform=test_val_transform)

    # Apply Subset to dataset
    train_dataset = Subset(train_dataset, train_indices)
    val_dataset = Subset(val_dataset, val_indices)
    test_dataset = Subset(test_dataset, test_indices)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader



