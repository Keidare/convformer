from tiny_imagenet_local import prep
import torch
import time
from tqdm import tqdm
import random, os, logging
import numpy as np
from convmit import ConvMiT
from mit import MiT
from convformer import ConvFormer, ConvFormerBN
from convnextformer import ConvNextFormer
import torchvision
from torchmetrics.classification import MulticlassF1Score
from torch.utils.data import DataLoader
from convnext import ConvNeXt
from helpers import *
from caltech import load_caltech256, load_caltech101



def train(logs_root,learning_rate,weight_decay,num_classes,num_epochs, batch_size):
    train_ds, val_ds = prep()
    train_loader = DataLoader(train_ds, batch_size, shuffle=True)
    val_loader = DataLoader(val_ds,batch_size,shuffle=False)
    set_seed(1234)
    os.makedirs(logs_root, exist_ok=True)
    model_path = os.path.join(logs_root, 'models')
    os.makedirs(model_path, exist_ok=True)

    logger = make_logger(filename=os.path.join(logs_root, 'train.log'))
    net = ConvNextFormer(num_classes=num_classes, img_height=64,img_width=64)
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


def train_caltech256(logs_root,lr,wd,num_classes,num_epochs,model = "MiT"):
    train_loader, val_loader = load_caltech256()
    set_seed(1234)
    os.makedirs(logs_root, exist_ok=True)
    model_path = os.path.join(logs_root, 'models')
    os.makedirs(model_path, exist_ok=True)

    logger = make_logger(filename=os.path.join(logs_root, 'train.log'))
    if model == "MiT":
        net = MiT(model_name="B1", num_classes= num_classes)
    elif model == "ConvMiT":
        net = ConvMiT(model_name="B1", num_classes= num_classes)
    elif model == "ConvNextFormer":
        net = ConvNextFormer(num_classes, img_height=224, img_width=224)
    elif model == "ConvFormer":
        net = ConvFormer(num_classes)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epoch_begin = 0
    model_files = sorted(os.listdir(model_path))
    net = net.to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    
    if len(model_files):
        checkpoint_path = os.path.join(model_path, model_files[-1])
        print('Load Checkpoint -> ', checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_begin = checkpoint['epoch']

    # Loss function with label smoothing
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    num_params = sum(p.numel() for p in net.parameters())

    logger.info(f'Number of parameters: {num_params}')
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

            # # Apply Mixup or CutMix with 50% probability each
            # if np.random.rand() < 0.5:
            #     x, y_a, y_b, lam = mixup_data(x, y, alpha=1.0)
            #     out = net(x)
            #     loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
            # else:
            #     x, y_a, y_b, lam = cutmix_data(x, y, alpha=1.0)
            #     out = net(x)
            #     loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
            out = net(x)
            loss = criterion(out,y)

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

        # Compute final accuracy
        accuracy = correct / total * 100  # Percentage
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

def train_caltech101(logs_root,lr,wd,num_classes,num_epochs,model = "MiT"):
    train_loader, val_loader = load_caltech101()
    set_seed(1234)
    os.makedirs(logs_root, exist_ok=True)
    model_path = os.path.join(logs_root, 'models')
    os.makedirs(model_path, exist_ok=True)

    logger = make_logger(filename=os.path.join(logs_root, f'train_{model}.log'))
    if model == "MiT":
        net = MiT(model_name="B2", num_classes= num_classes)
    elif model == "ConvMiT":
        net = ConvMiT(model_name="B1", num_classes= num_classes)
    elif model == "ConvNextFormer":
        net = ConvNextFormer(num_classes, img_height=224, img_width=224)
    elif model == "ConvFormer":
        net = ConvFormer(num_classes)
    elif model == "ConvNeXt":
        net = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], num_classes=num_classes, in_chans=3)
    elif model == "ConvFormerBN":
        net = ConvFormerBN(num_classes)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epoch_begin = 0
    model_files = sorted(os.listdir(model_path))
    net = net.to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    
    if len(model_files):
        checkpoint_path = os.path.join(model_path, model_files[-1])
        print('Load Checkpoint -> ', checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_begin = checkpoint['epoch']

    # Loss function with label smoothing
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    num_params = sum(p.numel() for p in net.parameters())

    logger.info(f'Number of parameters: {num_params}')
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
            # out = net(x)
            # loss = criterion(out,y)

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

        # Compute final accuracy
        accuracy = correct / total * 100  # Percentage
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
    # train('./logs/tinyImageNet/ConvNextFormer',learning_rate=1.5e-4,weight_decay=0.05,num_classes=200,num_epochs=100, batch_size=512)
    # train_caltech101('./logs/caltech/MiT224',1.5e-4,0.05,101,150,model = "MiT")
    train_caltech101('./logs/caltech/ConvFormerModification1',1e-4,0.05,101,125,model = "ConvFormer")

