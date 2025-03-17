from torchvision import transforms, datasets
import os

DATA_DIR = './data/tiny-imagenet-200/'

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(64),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def prep():
    train_set = datasets.ImageFolder(root=os.path.join(DATA_DIR, 'train'), transform=train_transform)
    val_set = datasets.ImageFolder(root=os.path.join(DATA_DIR, 'val/organized'), transform=val_transform)
    return train_set, val_set
