import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
from sklearn.preprocessing import LabelEncoder
import os
import time
import psutil
import GPUtil
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm


def get_efficientnet_b0(pretrained=True):
    """Returns an EfficientNet-B0 model with optional ImageNet pretrained weights."""
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
    return model

def get_vit_base_16(pretrained=True):
    """Returns a ViT-Base/16 model with optional ImageNet pretrained weights."""
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
    return model

def load_caltech101():
    train_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    ds = datasets.Caltech101(root='./data', transform=train_transform, download=True)
    train_size = int(0.8 * len(ds))
    val_size = len(ds) - train_size
    train_dataset, val_dataset = random_split(ds, [train_size, val_size])
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    return train_loader, val_loader

def monitor_resources():
    """Monitor CPU, RAM, and GPU usage."""
    cpu_usage = psutil.cpu_percent()
    ram_usage = psutil.virtual_memory().percent
    gpus = GPUtil.getGPUs()
    gpu_usage = gpus[0].load * 100 if gpus else 0
    gpu_memory = gpus[0].memoryUsed if gpus else 0
    return cpu_usage, ram_usage, gpu_usage, gpu_memory

def make_logger(name = None, filename='test.log'):
    """
    Creates and returns a fresh logger with both console and file logging.
    Ensures there are no leftover handlers from previous runs.
    """
    logger = logging.getLogger(name)

    # Check if logger already has handlers, remove them
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(message)s")

    # Console handler (INFO level)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler (DEBUG level)
    file_handler = logging.FileHandler(filename=filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def train(logs_root, model, num_classes=101, num_epochs=10, lr=0.001, wd=1e-4, model_name="EfficientNetB0"):
    train_loader, val_loader = load_caltech101()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = torch.nn.CrossEntropyLoss()
    os.makedirs(logs_root, exist_ok=True)
    model_path = os.path.join(logs_root, 'models')
    os.makedirs(model_path, exist_ok=True)
    logger = make_logger(filename=os.path.join(logs_root, f'train_{model_name}.log'))

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Monitor resources
        cpu, ram, gpu, gpu_mem = monitor_resources()
        
        # Evaluation phase
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Compute metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=1)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=1)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=1)

        log_message = (f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, "
                       f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, "
                       f"F1: {f1:.4f}, "
                       f"CPU: {cpu:.2f}%, RAM: {ram:.2f}%, GPU: {gpu:.2f}%, GPU Mem: {gpu_mem}MB")
        print(log_message)
        logger.info(log_message)
    
    # Save final model
    final_model_path = os.path.join(model_path, f'{model_name}_final.pth')
    torch.save({
        'epoch': num_epochs,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss.item(),
    }, final_model_path)
    print(f"Final model saved at {final_model_path}")
    logger.info(f"Final model saved at {final_model_path}")


# Example usage
model = get_efficientnet_b0(pretrained=True)
train(model = model, num_epochs=20, logs_root="./logs/caltech/EfficientNetB0Pretrained", model_name = "EfficientNetB0Pretrained")
model = get_efficientnet_b0(pretrained=False)
train(model = model, num_epochs=20, logs_root="./logs/caltech/EfficientNetB0Scratch", model_name = "EfficientNetB0Scratch")
model = get_vit_base_16(pretrained=True)
train(model = model, num_epochs=20, logs_root="./logs/caltech/VitBase16Pretrained", model_name = "VitBase16Pretrained")
model = get_vit_base_16(pretrained=False)
train(model = model, num_epochs=20, logs_root="./logs/caltech/VitBase16Scratch", model_name = "VitBase16Scratch")
