from sklearn.metrics import f1_score
import torch
import time,os,logging,random
from tqdm import tqdm
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from mit import MiT
from convmit import ConvMiT, MiT
from sklearn.preprocessing import LabelEncoder
import numpy as np
from convformer import ConvFormer


# Define transformations
train_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((128, 128)),  # Resize images
    transforms.RandomHorizontalFlip(p=0.5),  # Augmentation: Flip image
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0)  # Random Erasing
])

val_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Caltech 101 dataset
dataset = datasets.Caltech101(root="./data", transform=train_transform, download=True)
print(f"Number of classes: {len(dataset.categories)}")
# Encode labels
label_encoder = LabelEncoder()
dataset.y = label_encoder.fit_transform(dataset.y) 

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

val_dataset.dataset.transform = val_transform  # Overwrite transform for validation

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,num_workers=0)

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # Important for full determinism


def make_logger(name=None, filename="test.log"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(message)s")

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(filename=filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def mixup_data(x, y, alpha=1.0):
    """Apply Mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    """Apply CutMix augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size, _, H, W = x.size()
    index = torch.randperm(batch_size).to(x.device)

    # Generate random bounding box
    rx, ry = np.random.randint(W), np.random.randint(H)
    rw, rh = int(W * np.sqrt(1 - lam)), int(H * np.sqrt(1 - lam))
    
    # Ensure valid coordinates
    x1, y1 = max(rx - rw // 2, 0), max(ry - rh // 2, 0)
    x2, y2 = min(rx + rw // 2, W), min(ry + rh // 2, H)

    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    y_a, y_b = y, y[index]

    # Adjust lambda based on actual patch area
    lam = 1 - ((x2 - x1) * (y2 - y1) / (H * W))
    
    return x, y_a, y_b, lam

def train(logs_root):
    set_seed(1234)
    os.makedirs(logs_root, exist_ok=True)
    model_path = os.path.join(logs_root, 'models')
    os.makedirs(model_path, exist_ok=True)

    logger = make_logger(filename=os.path.join(logs_root, 'train.log'))
    net = ConvFormer(num_classes=101)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epoch_begin = 0
    model_files = sorted(os.listdir(model_path))
    net = net.to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.05)
    
    if len(model_files):
        checkpoint_path = os.path.join(model_path, model_files[-1])
        print('Load Checkpoint -> ', checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_begin = checkpoint['epoch']

    # Loss function with label smoothing
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    num_epochs = 150

    num_params = sum(p.numel() for p in net.parameters())

    logger.info(f'Number of parameters: {num_params}')
    logger.info(net)
    print(f'Starting epoch: {epoch_begin}')
    print(f'Total epochs: {num_epochs}')
    logger.info('| %12s | %12s | %12s | %12s | %12s |' %
                ('epoch', 'time_train', 'loss_train', 'loss_val', 'accuracy'))
    
    for epoch in range(epoch_begin, num_epochs):
        t0 = time.time()
        net.train()
        losses = 0

        for x, y in tqdm(train_loader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            # Apply Mixup or CutMix with 50% probability each
            if np.random.rand() < 0.5:
                x, y_a, y_b, lam = mixup_data(x, y, alpha=1.0)
                out = net(x)
                loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
            else:
                x, y_a, y_b, lam = cutmix_data(x, y, alpha=1.0)
                out = net(x)
                loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)

            loss.backward()
            optimizer.step()

            losses += loss.detach()

        loss_train = losses / len(train_loader)
        t1 = time.time()
        time_train = t1 - t0

        # Validation Phase
        t0 = time.time()
        net.eval()
        losses = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in tqdm(val_loader):
                x = x.to(device)
                y = y.to(device)

                out = net(x)
                loss = criterion(out, y)

                losses += loss.detach()
                
                preds = out.argmax(dim=1)
                
                # Count correct predictions
                correct += (preds == y).sum().item()
                total += y.size(0)
                batch_accuracy = (preds == y).sum().item() / y.size(0) * 100
                print(f"Batch Accuracy: {batch_accuracy:.2f}%")

        accuracy = correct / total * 100  # Compute validation accuracy
        loss_val = losses / len(val_loader)
        t1 = time.time()
        time_val = t1 - t0

        print(f"Epoch {epoch+1}: Validation Accuracy = {accuracy:.2f}%")

        time_total = time_train + time_val
        logger.info('| %12d | %12.4f | %12.4f | %12.4f | %12.4f' %
                    (epoch + 1, time_total, loss_train, loss_val, accuracy))

        model_file = os.path.join(model_path, 'model_%03d.pt' % (epoch + 1))
        torch.save({
            'epoch': epoch + 1,
            'model': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': loss_val.item(),
        }, model_file)

def eval():
    set_seed(1234)
    net = ConvFormer(num_classes=101)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net.to(device)
    
    checkpoint = torch.load('./logs/caltech/06032025ConvFormerv224x224/models/model_096.pt')
    net.load_state_dict(checkpoint['model'])
    
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    
    losses = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    net.eval()
    with torch.no_grad():
        for x, y in tqdm(val_loader):
            x = x.to(device)
            y = y.to(device)

            out = net(x)
            loss = criterion(out, y)
            losses += loss.detach().item()

            # Get predicted class (assuming classification task)
            preds = out.argmax(dim=1)

            # Store predictions and labels for F1 calculation
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

            # Count correct predictions
            correct += (preds == y).sum().item()
            total += y.size(0)

    # Compute final accuracy
    accuracy = correct / total * 100  # Percentage
    loss_val = losses / len(val_loader)

    # Compute F1 score (macro average for multi-class classification)
    f1 = f1_score(all_labels, all_preds, average='macro') * 100  # Convert to percentage

    print(f"Total Loss: {losses:.4f}, Avg Loss: {loss_val:.4f}, Accuracy: {accuracy:.2f}%, F1 Score: {f1:.2f}%")



def trainconvmit(logs_root):
    set_seed(1234)
    os.makedirs(logs_root, exist_ok=True)
    model_path = os.path.join(logs_root, 'models')
    os.makedirs(model_path, exist_ok=True)

    logger = make_logger(filename=os.path.join(logs_root, 'train.log'))
    net = ConvMiT(model_name='B1', num_classes=101)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epoch_begin = 0
    model_files = sorted(os.listdir(model_path))
    net = net.to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0.05)
    if len(model_files):
        checkpoint_path = os.path.join(model_path, model_files[-1])
        print('Load Checkpoint -> ', checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_begin = checkpoint['epoch']
    criterion = torch.nn.CrossEntropyLoss()
    num_epochs = 35

    num_params = sum(p.numel() for p in net.parameters())

    logger.info(f'Number of parameters: {num_params}')
    logger.info(net)
    print(f'Starting epoch: {epoch_begin}')
    print(f'Total epochs: {num_epochs}')
    logger.info('| %12s | %12s | %12s | %12s | %12s |' %
                ('epoch', 'time_train', 'loss_train', 'loss_val', 'accuracy'))
    
    for epoch in range(epoch_begin,num_epochs):
        t0 = time.time()
        net.train()
        losses = 0
        for x, y in tqdm(train_loader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            out = net(x)
            loss = criterion(out, y)

            loss.backward()
            optimizer.step()

            losses += loss.detach()

        loss_train = losses / len(train_loader)
        t1 = time.time()
        time_train = t1 - t0

        t0 = time.time()
        net.eval()
        losses = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in tqdm(val_loader):
                x = x.to(device)
                y = y.to(device)

                out = net(x)
                loss = criterion(out, y)

                losses += loss.detach()
                
                preds = out.argmax(dim=1)
                
                # Count correct predictions
                correct += (preds == y).sum().item()
                total += y.size(0)

        # Compute final accuracy
        accuracy = correct / total * 100  # Percentage
        loss_val = losses / len(val_loader)
        loss_val = losses / len(val_loader)
        t1 = time.time()
        time_val = t1 - t0

        time_total = time_train + time_val
        logger.info('| %12d | %12.4f | %12.4f | %12.4f | %12.4f' %
                    (epoch + 1, time_total, loss_train, loss_val, accuracy))

        model_file = os.path.join(model_path, 'model_%03d.pt' % (epoch + 1))
        torch.save({
            'epoch': epoch + 1,
            'model': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': loss_val.item(),
        }, model_file)

def trainMIT(logs_root):
    set_seed(1234)
    os.makedirs(logs_root, exist_ok=True)
    model_path = os.path.join(logs_root, 'models')
    os.makedirs(model_path, exist_ok=True)

    logger = make_logger(filename=os.path.join(logs_root, 'train.log'))
    net = MiT(num_classes=101,model_name="B1")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epoch_begin = 0
    model_files = sorted(os.listdir(model_path))
    net = net.to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0.05)
    if len(model_files):
        checkpoint_path = os.path.join(model_path, model_files[-1])
        print('Load Checkpoint -> ', checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_begin = checkpoint['epoch']
    criterion = torch.nn.CrossEntropyLoss()
    num_epochs = 35

    num_params = sum(p.numel() for p in net.parameters())

    logger.info(f'Number of parameters: {num_params}')
    logger.info(net)
    print(f'Starting epoch: {epoch_begin}')
    print(f'Total epochs: {num_epochs}')
    logger.info('| %12s | %12s | %12s | %12s | %12s |' %
                ('epoch', 'time_train', 'loss_train', 'loss_val', 'accuracy'))
    
    for epoch in range(epoch_begin,num_epochs):
        t0 = time.time()
        net.train()
        losses = 0
        for x, y in tqdm(train_loader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            out = net(x)
            loss = criterion(out, y)

            loss.backward()
            optimizer.step()

            losses += loss.detach()

        loss_train = losses / len(train_loader)
        t1 = time.time()
        time_train = t1 - t0

        t0 = time.time()
        net.eval()
        losses = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in tqdm(val_loader):
                x = x.to(device)
                y = y.to(device)

                out = net(x)
                loss = criterion(out, y)

                losses += loss.detach()
                
                preds = out.argmax(dim=1)
                
                # Count correct predictions
                correct += (preds == y).sum().item()
                total += y.size(0)

        # Compute final accuracy
        accuracy = correct / total * 100  # Percentage
        loss_val = losses / len(val_loader)
        loss_val = losses / len(val_loader)
        t1 = time.time()
        time_val = t1 - t0

        time_total = time_train + time_val
        logger.info('| %12d | %12.4f | %12.4f | %12.4f | %12.4f' %
                    (epoch + 1, time_total, loss_train, loss_val, accuracy))

        model_file = os.path.join(model_path, 'model_%03d.pt' % (epoch + 1))
        torch.save({
            'epoch': epoch + 1,
            'model': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': loss_val.item(),
        }, model_file)

if __name__ == '__main__':
    eval()