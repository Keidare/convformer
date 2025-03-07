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
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
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
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


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



def train(logs_root):
    set_seed(1234)
    os.makedirs(logs_root, exist_ok=True)
    model_path = os.path.join(logs_root, 'models')
    os.makedirs(model_path, exist_ok=True)

    logger = make_logger(filename=os.path.join(logs_root, 'train.log'))
    net = ConvFormer(num_classes = 101)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epoch_begin = 0
    model_files = sorted(os.listdir(model_path))
    net = net.to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-5, weight_decay=0.05)
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

def eval():
    net = MiT(model_name='B1', num_classes=101)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net.to(device)
    checkpoint = torch.load('./logs/caltech/0225025ConvFormerv1/models/model_015.pt')
    net.load_state_dict(checkpoint['model'])
    criterion = torch.nn.CrossEntropyLoss()
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

            # Get predicted class (assuming classification task)
            preds = out.argmax(dim=1)
            
            # Count correct predictions
            correct += (preds == y).sum().item()
            total += y.size(0)

    # Compute final accuracy
    accuracy = correct / total * 100  # Percentage
    loss_val = losses / len(val_loader)

    print(loss_val, accuracy)

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
    train('./logs/caltech/06032025ConvFormerv2')