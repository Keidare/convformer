from helpers import *
import torch
import time
from tqdm import tqdm
import random, os, logging
import numpy as np
from convmit import ConvMiT
from mit import MiT
from convformer import ConvFormer, ConvFormerBN
from bonemeta import *
import torchvision
from torchmetrics.classification import MulticlassF1Score
from torch.utils.data import DataLoader

def train(logs_root,learning_rate,weight_decay,num_classes,num_epochs, batch_size):
    train_ds, val_ds = BoneDataset()
    train_loader = DataLoader(train_ds, batch_size, shuffle=True)
    val_loader = DataLoader(val_ds,batch_size,shuffle=False)
    set_seed(1234)
    os.makedirs(logs_root, exist_ok=True)
    model_path = os.path.join(logs_root, 'models')
    os.makedirs(model_path, exist_ok=True)

    logger = make_logger(filename=os.path.join(logs_root, 'train.log'))
    net = ConvFormer(num_classes=2)
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
    logger.info(sum(p.numel() for p in net.parameters()))

    print(f'Starting epoch: {epoch_begin}')
    print(f'Total epochs: {num_epochs}')
    logger.info('| %12s | %12s | %12s | %12s | %12s | %12s | %12s |' % 
                ('epoch', 'time_train', 'loss_train', 'loss_val', 'acc_val', 'top5_acc', 'f1_score'))

    for epoch in range(epoch_begin, num_epochs):
        t0 = time.time()
        net.train()
        losses = 0

        for x,y in tqdm(train_loader, desc="Training"):
            images, labels = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            # if np.random.rand() < 0.5:
            #     images, y_a, y_b, lam = mixup_data(images, labels, alpha=1.0)
            #     out = net(images)
            #     loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
            # else:
            #     images, y_a, y_b, lam = cutmix_data(images, labels, alpha=1.0)
            #     out = net(images)
            #     loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
            out = net(images)
            loss = criterion(out, labels)
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
            for x,y in tqdm(val_loader, desc="Evaluating"):
                images, labels = x.to(device), y.to(device)
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