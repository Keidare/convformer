import tiny_imagenet
import torch
import time
from tqdm import tqdm
import random, os, logging
import numpy as np
from convmit import ConvMiT
from convformer import ConvFormer
import torchvision
from torchmetrics.classification import MulticlassF1Score



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

def mixup_data(x, y, alpha=1.0):
    """Apply Mixup augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    # Ensure tensor format
    if not isinstance(x, torch.Tensor):
        x = torchvision.transforms.ToTensor()(x)

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """Apply CutMix augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    # Ensure the image is a tensor
    if isinstance(x, torch.Tensor):
        batch_size, _, H, W = x.size()
    else:
        x = torchvision.transforms.ToTensor()(x)  # Convert PIL image to tensor
        batch_size, _, H, W = x.size()

    index = torch.randperm(batch_size).to(x.device)

    # Random bbox
    cx, cy = np.random.randint(W), np.random.randint(H)
    bw, bh = int(W * np.sqrt(1 - lam)), int(H * np.sqrt(1 - lam))
    x1, x2 = max(cx - bw // 2, 0), min(cx + bw // 2, W)
    y1, y2 = max(cy - bh // 2, 0), min(cy + bh // 2, H)

    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


def train(logs_root,learning_rate,weight_decay,num_classes,num_epochs, img_height= 64, img_width= 64):
    train_loader, val_loader = tiny_imagenet.tiny_imagenet()
    set_seed(1234)
    os.makedirs(logs_root, exist_ok=True)
    model_path = os.path.join(logs_root, 'models')
    os.makedirs(model_path, exist_ok=True)

    logger = make_logger(filename=os.path.join(logs_root, 'train.log'))
    net = ConvFormer(num_classes=num_classes, img_height = img_height, img_width = img_width)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epoch_begin = 0
    model_files = sorted(os.listdir(model_path))
    net = net.to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if len(model_files):
        checkpoint_path = os.path.join(model_path, model_files[-1])
        print('Load Checkpoint -> ', checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_begin = checkpoint['epoch']

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    num_epochs = num_epochs
    f1_metric = MulticlassF1Score(num_classes=num_classes, average='macro').to(device)

    print(f'Starting epoch: {epoch_begin}')
    print(f'Total epochs: {num_epochs}')
    logger.info('| %12s | %12s | %12s | %12s | %12s | %12s | %12s |' % 
                ('epoch', 'time_train', 'loss_train', 'loss_val', 'acc_val', 'top5_acc', 'f1_score'))

    for epoch in range(epoch_begin, num_epochs):
        t0 = time.time()
        net.train()
        losses = 0

        for batch in tqdm(train_loader, desc="Training"):
            images, labels = batch['image'].to(device), batch['label'].to(device)

            optimizer.zero_grad(set_to_none=True)

            out = net(images)
            loss = criterion(out, labels)  # Missing in your code

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
        correct_top1 = 0
        correct_top5 = 0
        total = 0
        f1_metric.reset()

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                images, labels = batch['image'].to(device), batch['label'].to(device)

                out = net(images)
                loss = criterion(out, labels)
                losses += loss.detach()

                preds = out.argmax(dim=1)
                f1_metric.update(preds, labels)

                correct_top1 += (preds == labels).sum().item()
                top5_preds = torch.topk(out, 5, dim=1).indices
                correct_top5 += top5_preds.eq(labels.view(-1, 1)).sum().item()

                total += labels.size(0)

        accuracy = correct_top1 / total * 100
        top5_accuracy = correct_top5 / total * 100
        f1_score = f1_metric.compute().item()
        loss_val = losses / len(val_loader)
        t1 = time.time()
        time_val = t1 - t0

        time_total = time_train + time_val

        logger.info('| %12d | %12.4f | %12.4f | %12.4f | %12.4f | %12.4f | %12.4f |' % 
                    (epoch + 1, time_total, loss_train, loss_val, accuracy, top5_accuracy, f1_score))

        model_file = os.path.join(model_path, 'model_%03d.pt' % (epoch + 1))
        torch.save({
            'epoch': epoch + 1,
            'model': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': loss_val.item(),
        }, model_file)

def evaluate(val_loader, checkpoint_path, device, img_height = 64, img_width = 64):
    """
    Loads a model checkpoint and evaluates it on the validation dataset.
    
    Args:
        val_loader (DataLoader): Validation data loader.
        checkpoint_path (str): Path to the model checkpoint.
        device (str): Device to run the evaluation on ('cuda' or 'cpu').

    Returns:
        dict: Dictionary containing validation loss, top-1 accuracy, top-5 accuracy, and F1 score.
    """
    set_seed(1234)
    model = ConvFormer(num_classes=200, img_height = img_height, img_width = img_width).to(device)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    
    model.eval()
    losses = 0
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    f1_metric = MulticlassF1Score(num_classes=200, average='macro').to(device)
    f1_metric.reset()
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Evaluating'):
            images, labels = batch['image'].to(device), batch['label'].to(device)

            out = model(images)
            loss = criterion(out, labels)
            losses += loss.item()

            preds = out.argmax(dim=1)
            f1_metric.update(preds, labels)

            correct_top1 += (preds == labels).sum().item()
            top5_preds = torch.topk(out, 5, dim=1).indices
            correct_top5 += top5_preds.eq(labels.view(-1, 1)).sum().item()
            total += labels.size(0)
    
    return {
        "loss": losses / len(val_loader),
        "top1_accuracy": correct_top1 / total * 100,
        "top5_accuracy": correct_top5 / total * 100,
        "f1_score": f1_metric.compute().item()
    }



if __name__ == '__main__':
    train('./logs/tinyImageNet/ConvFormerV1',learning_rate=1e-3,weight_decay=0.02,num_classes=200,num_epochs=100, img_height = 64, img_width = 64)



