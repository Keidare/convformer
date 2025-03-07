import tiny_imagenet
import torch
import time
from tqdm import tqdm
import random, os, logging
import numpy as np
from convmit import ConvMiT
from convformer import ConvFormer
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
    train_loader, val_loader = tiny_imagenet.tiny_imagenet()
    set_seed(1234)
    os.makedirs(logs_root, exist_ok=True)
    model_path = os.path.join(logs_root, 'models')
    os.makedirs(model_path, exist_ok=True)

    logger = make_logger(filename=os.path.join(logs_root, 'train.log'))
    net = ConvFormer(num_classes= 200)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epoch_begin = 0
    model_files = sorted(os.listdir(model_path))
    net = net.to(device)
    if len(model_files):
        checkpoint_path = os.path.join(model_path, model_files[-1])
        print('Load Checkpoint -> ', checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_begin = checkpoint['epoch']
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.0001, weight_decay=0.02)
    criterion = torch.nn.CrossEntropyLoss()
    num_epochs = 100

    print(f'Starting epoch: {epoch_begin}')
    print(f'Total epochs: {num_epochs}')
    logger.info('| %12s | %12s | %12s | %12s | %12s |' %
                ('epoch', 'time_train', 'loss_train', 'loss_val', 'acc_val'))
    for epoch in range(epoch_begin,num_epochs):
        t0 = time.time()
        net.train()
        losses = 0
        for batch in tqdm(train_loader):

            images, labels = batch['image'].to(device), batch['label'].to(device)  # Move to GPU/CPU
            
            optimizer.zero_grad(set_to_none=True)

            out = net(images)
            loss = criterion(out, labels)

            loss.backward()
            optimizer.step()

            losses += loss.detach()

        loss_train = losses / len(train_loader)
        t1 = time.time()
        time_train = t1 - t0

        t0 = time.time()
        net.eval()
        losses = 0
        correct_top5 = 0
        total = 0
        with torch.no_grad():
            for batch in tqdm(val_loader):
                images, labels = batch['image'].to(device), batch['label'].to(device)  # Move to GPU/CPU

                out = net(images)
                loss = criterion(out, labels)
                # Get top-5 predictions
                top5_preds = torch.topk(out, 5, dim=1).indices  

                # Check if the correct label is in the top 5 predictions
                top5_correct = top5_preds.eq(labels.view(-1, 1)).sum().item()  

                # Track correct predictions
                correct_top5 += top5_correct

                total += labels.size(0)
                losses += loss.detach()
        accuracy = correct_top5 / total * 100  # Percentage
        loss_val = losses / len(val_loader)
        t1 = time.time()
        time_val = t1 - t0

        time_total = time_train + time_val

        logger.info('| %12d | %12.4f | %12.4f | %12.4f | %12.4f |' %
                    (epoch + 1, time_total, loss_train, loss_val, accuracy))

        model_file = os.path.join(model_path, 'model_%03d.pt' % (epoch + 1))
        torch.save({
            'epoch': epoch + 1,
            'model': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': loss_val.item(),
        }, model_file)

if __name__ == '__main__':
    train(logs_root = 'logs/tinyImageNet/ConvFormerTest')


