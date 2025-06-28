from helpers import *
import torch
import time
from tqdm import tqdm
import random, os, logging
import numpy as np
from convmit import ConvMiT
from mit import MiT
from convformer import *
from bonemeta import *
from effNetCBAM import *
from MagbooCBAMTransformer import *
import timm 
from CMT import *
from coatnet import *
from vit_effnet import *
def train(logs_root,lr,wd,num_classes,num_epochs,model = "MiT", ds_mode='normal', heads = 4):
    if ds_mode == "origds":
        train_loader, val_loader, test_loader = prepare_ds(batch_size=16, csv_file='data/ground_original.csv', root_dir='data/bones_original/', crop = True)
    else:
        train_loader, val_loader, test_loader = prepare_ds_base(batch_size=8)
    set_seed(1232)
    os.makedirs(logs_root, exist_ok=True)
    model_path = os.path.join(logs_root, 'models')
    os.makedirs(model_path, exist_ok=True)

    logger = make_logger(filename=os.path.join(logs_root, f'train_{model}.log'))
    if model == "MiT":
        net = MiT(model_name="B2", num_classes= num_classes)
    elif model == "ConvMiT":
        net = ConvMiT(model_name="B1", num_classes= num_classes)
    elif model == "ConvFormer":
        net = ConvFormer(num_classes)
    elif model == "ConvFormerV2":
        net = ConvFormerV2(num_classes)
    elif model == "MCNNCBAMTransformer":
        net = CNNTransformer(num_classes= num_classes, in_channels=1, heads=heads)
    elif model == "EfficientNet_CBAMb0":
        net = EfficientNet_CBAM(num_classes=num_classes, variant="b0")
    elif model == "DSCCBAMTransformer":
        net = DSCCBAMTransformer(num_classes=num_classes, num_transformers=3)
    elif model == "MCNNCBAMTransformerMultiBlock":
        net = CNNTransformerMultiBlock(num_classes= num_classes, in_channels=1, num_transformers=3)
    elif model == "MagbooCBAMTransformerAndBlock":
        net = CNNTransformerMultiCNNAndBlock(num_classes=num_classes, in_channels=1, num_transformers=1)
    elif model == "MCNNCBAMTransformerPosEnc":
        net = CNNTransformerPosEnc(num_classes=num_classes, in_channels=1)
    elif model == "MCNNCBAMTransformerCPE":
        net = CNNTransformerRepCPE(num_classes=num_classes, in_channels=1)
    elif model == "CNNCBAM":
        net = CNN_NoTransformer(num_classes=num_classes, in_channels=1)
    elif model == "CNN_NoCBAM":
        net = CNNTransformer_NoCBAM(num_classes=num_classes, in_channels=1)
    elif model == "Base":
        net = BaseModel(num_classes=num_classes, in_channels=1)
    elif model == "GELU":
        net = CNNTransformerGELU(num_classes=num_classes, in_channels=1)
    elif model == "swin":
        net = timm.create_model('swin_small_patch4_window7_224', pretrained=False)

        # Modify input to accept 1 channel instead of 3
        original_proj = net.patch_embed.proj
        net.patch_embed.proj = nn.Conv2d(
            in_channels=1,
            out_channels=original_proj.out_channels,
            kernel_size=original_proj.kernel_size,
            stride=original_proj.stride,
            padding=original_proj.padding,
            bias=original_proj.bias is not None
        )

        # Initialize new conv layer weights
        nn.init.kaiming_normal_(net.patch_embed.proj.weight, mode='fan_out', nonlinearity='relu')

        # Modify classification head
        net.head = nn.Linear(net.head.in_features, num_classes)
                # Modify the forward method to include global average pooling
        def swin_forward(self, x):
            x = self.forward_features(x)  # Output shape: [B, H_patch, W_patch, C]
            x = x.mean(dim=(1, 2))        # Global average pool over height and width (H, W)
            x = self.head(x)              # Output shape: [B, num_classes]
            return x

        # Replace the forward method in the model
        net.forward = swin_forward.__get__(net)
    elif model == "CMT":
        net = CMT_S()
    elif model == "COAT":
        net = coatnet_1()
    elif model == "vit":
        net = vit()
    elif model == "effnet":
        net = efficientnet()
    logging.getLogger('PIL').setLevel(logging.CRITICAL)  # This will suppress all Pillow logs


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epoch_begin = 0
    model_files = sorted(os.listdir(model_path))
    net = net.to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    val_cm = np.zeros((num_classes, num_classes))
    if len(model_files):
        checkpoint_path = os.path.join(model_path, model_files[-1])
        print('Load Checkpoint -> ', checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_begin = checkpoint['epoch']

    # Loss function 
    criterion = torch.nn.CrossEntropyLoss()
    num_params = sum(p.numel() for p in net.parameters())
    
    logger.info(f'Number of parameters: {num_params}')
    print(f'Starting epoch: {epoch_begin}')
    print(f'Total epochs: {num_epochs}')
    logger.info('| %12s | %12s | %12s | %12s | %12s | %12s | %12s | %12s | %12s |' %
                ('epoch', 'time_train', 'loss_train', 'loss_val', 'accuracy', 'precision', 'recall', 'f1', 'specificity'))
    
    for epoch in range(epoch_begin, num_epochs):
        t0 = time.time()
        net.train()
        losses = 0

        for x, y in tqdm(train_loader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            out = net(x)
            loss = criterion(out,y)

            loss.backward()
            optimizer.step()

            losses += loss.detach()

        loss_train = losses / len(train_loader)
        t1 = time.time()
        time_train = t1 - t0

        net.eval()
        total = 0
        true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
        final_pos, final_neg = 0, 0
        val_losses = 0

        with torch.no_grad():
            for x, y in tqdm(val_loader):
                x, y = x.to(device), y.to(device)

                out = net(x)
                loss = criterion(out, y)
                val_losses += loss.detach()

                _, predicted = torch.max(out, dim=1)  # Get predicted labels
                total += y.size(0)  # Count total samples

                for index in range(len(y)):
                    true_pos += (predicted[index] == 1 and y[index] == 1).item()
                    true_neg += (predicted[index] == 0 and y[index] == 0).item()
                    false_pos += (predicted[index] == 1 and y[index] == 0).item()
                    false_neg += (predicted[index] == 0 and y[index] == 1).item()

                    # Track class distribution
                    final_pos += (y[index] == 1).item()
                    final_neg += (y[index] == 0).item()

        # Compute validation loss
        loss_val = val_losses / len(val_loader)
        t2 = time.time()
        time_val = t2 - t1
        time_total = time_train + time_val

        # Compute evaluation metrics
        val_accuracy = (true_pos + true_neg) / total if total != 0 else 0
        val_precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) != 0 else 0
        val_recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) != 0 else 0  # Sensitivity
        val_specificity = true_neg / (true_neg + false_pos) if (true_neg + false_pos) != 0 else 0
        val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall) if (val_precision + val_recall) != 0 else 0

        # Print class distribution
        print(f"Class Composition: Positive: {final_pos} ({(final_pos/total)*100:.2f}%) Negative: {final_neg} ({(final_neg/total)*100:.2f}%)")
        print(f"TP: {true_pos}, TN: {true_neg}, FP: {false_pos}, FN: {false_neg}, Total: {total}")

        # Log results
        logger.info('| %12d | %12.4f | %12.4f | %12.4f | %12.4f | %12.4f | %12.4f | %12.4f | %12.4f |' %
                    (epoch + 1, time_total, loss_train, loss_val, val_accuracy, val_precision, val_recall, val_f1, val_specificity))

        model_file = os.path.join(model_path, 'model_%03d.pt' % (epoch + 1))
        torch.save({
            'epoch': epoch + 1,
            'model': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': loss_val.item(),
        }, model_file)    
    

if __name__ == "__main__":
    # train('./logs/BoneDataset/EFFNETCBAMB0',1e-5,0.2,2,300,model = "EfficientNet_CBAMb0")
    # train('./logs/BoneDataset/EFFNETCBAMB1',1e-5,0.2,2,300,model = "EfficientNet_CBAMb1")
    # train('./logs/BoneDataset/MagbooCBAM-1-CPE',1e-5,0.02,2,400,model = "MCNNCBAMTransformerCPE")
    # train('./logs/BoneDataset/MagbooCBAM-1-PosEnc',1e-5,0.02,2,400,model = "MCNNCBAMTransformerPosEnc")
    # train('./logs/BoneDatasetO/MiT',1e-5,0.02,2,150,model = "MiT", ds_mode='origds')
    # train('./logs/BoneDatasetO/CoAt',1e-5,0.02,2,150,model = "COAT", ds_mode='origds')
    # train('./logs/BoneDatasetO/CMT',1e-5,0.2,2,150,model = "CMT")
    # train('./logs/BoneDatasetO/SWIN',1e-5,0.2,2,150,model = "swin")
    # train('./logs/BoneDatasetO/ViT',1e-5,0.2,2,150,model = "vit", ds_mode='origds')
    # train('./logs/BoneDatasetO/effnet',1e-5,0.2,2,150,model = "effnet", ds_mode='origds')
    # train('./logs/BoneDataset/MNCNNCBAM-GELU1TBlock',1e-5,0.2,2,300,model = "GELU")
    train('./logs/BoneDataset/FinalModel224',1e-5,0.15,2,600,model = "MCNNCBAMTransformer", heads = 4)
    




# SAVE 278
# save 167/213/263/267/441/438/435/434/433/431/426/413/387/373/365/359/342/336 on Finalv2