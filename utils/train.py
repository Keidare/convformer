import os
import time
import logging
import random
import torch
import torch.nn as nn
import numpy as np
import segmentation_models_pytorch as smp
from tqdm import tqdm
import UNet
import utils.cityscapes
import utils.transforms



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


def train(
    logs_root,
    data_root='data/cityscapes',
    learning_rate=0.0003,
    size=(1024, 512),
    weight_decay=0,
    batch_size = 4,
    epochs= 50,
    num_workers=4,
    backbone='efficientnet-b0',
    model_name='unet',
    num_classes=20):
    set_seed(1234)
    os.makedirs(logs_root, exist_ok=True)
    model_path = os.path.join(logs_root, 'models')
    os.makedirs(model_path, exist_ok=True)

    logger = make_logger(filename=os.path.join(logs_root, 'train.log'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_name == 'unet-new':
        net = smp.Unet(backbone, classes=num_classes, encoder_weights='imagenet')
    elif model_name == 'vit':
        net = smp.Segformer(backbone, classes=num_classes, encoder_weights='imagenet')
    elif model_name == 'unet':
        net = UNet(in_channels=3, out_channels=num_classes)
    
    net = net.to(device)

    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay)

    epoch_begin = 0
    model_files = sorted(os.listdir(model_path))
    if len(model_files):
        checkpoint_path = os.path.join(model_path, model_files[-1])
        print('Load Checkpoint -> ', checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_begin = checkpoint['epoch']

    T_train = utils.transforms.Compose([
        utils.transforms.RandomResizedCrop(size, scale=(0.5, 1.0)),
        utils.transforms.RandomHorizontalFlip(),
        utils.transforms.ToTensor(),
        utils.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])])

    T_val = utils.transforms.Compose([
        utils.transforms.Resize(size),
        utils.transforms.ToTensor(),
        utils.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])])

    train_dataset = utils.cityscapes.Cityscapes(data_root, 'train', T_train)
    val_dataset = utils.cityscapes.Cityscapes(data_root, 'val', T_val)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    criterion = nn.CrossEntropyLoss()

    logger.info(net)
    logger.info(device)
    logger.info(optimizer)

    logger.info('| %12s | %12s | %12s | %12s |' %
                ('epoch', 'time_train', 'loss_train', 'loss_val'))
    print(epoch_begin)
    print(epochs)
    for epoch in range(epoch_begin, epochs):
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
        with torch.no_grad():
            for x, y in tqdm(val_loader):
                x = x.to(device)
                y = y.to(device)

                out = net(x)
                loss = criterion(out, y)

                losses += loss.detach()

        loss_val = losses / len(val_loader)
        t1 = time.time()
        time_val = t1 - t0

        time_total = time_train + time_val

        logger.info('| %12d | %12.4f | %12.4f | %12.4f |' %
                    (epoch + 1, time_total, loss_train, loss_val))

        model_file = os.path.join(model_path, 'model_%03d.pt' % (epoch + 1))
        torch.save({
            'epoch': epoch + 1,
            'model': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': loss_val.item(),
        }, model_file)

