import torchvision.transforms as transforms
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch

def tiny_imagenet():
    train_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),  # Convert grayscale to RGB
        transforms.RandomHorizontalFlip(),  # Random horizontal flip
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),  # Convert grayscale to RGB
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def apply_transform(batch, transform):
        batch['image'] = torch.stack([transform(img) for img in batch['image']])
        batch['label'] = torch.tensor(batch['label'])
        return batch

    # Load dataset
    train_ds = load_dataset('zh-plus/tiny-imagenet', split='train').map(lambda batch: apply_transform(batch, train_transform), batched=True).with_format('torch')
    val_ds = load_dataset('zh-plus/tiny-imagenet', split='valid').map(lambda batch: apply_transform(batch, val_transform), batched=True).with_format('torch')

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)  
    
    return train_loader, val_loader
