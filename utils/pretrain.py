import tiny_imagenet
import torch
import time
from tqdm import tqdm
import random, os, logging
import numpy as np
from convmit import ConvMiT
from convformer import ConvFormer
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

def train(logs_root):
    train_loader, val_loader = tiny_imagenet.tiny_imagenet()
    set_seed(1234)
    os.makedirs(logs_root, exist_ok=True)
    model_path = os.path.join(logs_root, 'models')
    os.makedirs(model_path, exist_ok=True)

    logger = make_logger(filename=os.path.join(logs_root, 'train.log'))
    net = ConvFormer(num_classes=200)
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

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    num_epochs = 150
    f1_metric = MulticlassF1Score(num_classes=200, average='macro').to(device)

    print(f'Starting epoch: {epoch_begin}')
    print(f'Total epochs: {num_epochs}')
    logger.info('| %12s | %12s | %12s | %12s | %12s | %12s | %12s |' % 
                ('epoch', 'time_train', 'loss_train', 'loss_val', 'acc_val', 'top5_acc', 'f1_score'))

    for epoch in range(epoch_begin, num_epochs):
        t0 = time.time()
        net.train()
        losses = 0

        for batch in tqdm(train_loader):
            images, labels = batch['image'].to(device), batch['label'].to(device)

            optimizer.zero_grad(set_to_none=True)

            # Apply Mixup or CutMix with 50% probability each
            if np.random.rand() < 0.5:
                images, y_a, y_b, lam = mixup_data(images, labels, alpha=1.0)
                out = net(images)
                loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
            else:
                images, y_a, y_b, lam = cutmix_data(images, labels, alpha=1.0)
                out = net(images)
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
        correct_top1 = 0
        correct_top5 = 0
        total = 0
        f1_metric.reset()

        with torch.no_grad():
            for batch in tqdm(val_loader):
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

if __name__ == '__main__':
    train(logs_root = 'logs/tinyImageNet/ConvFormerV1')


