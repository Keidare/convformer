import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from tqdm import tqdm
import UNet
import utils.cityscapes
import utils.transforms


def evaluate_cityscapes(
    data_root='./data/cityscapes',
    model_path='./logs/cityscapes/220720/models/model_100.pth',
    size=(512, 256),
    batch_size=8,
    num_workers=4,
    num_classes=19,
    backbone='mobilenet_v2',
    model_name='unet',
):
    device = device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if model_name == 'unet-new':
        net = smp.Unet(backbone, classes=num_classes, encoder_weights='imagenet')
    elif model_name == 'vit':
        net = smp.Segformer(backbone, classes=num_classes, encoder_weights='imagenet')
    elif model_name == 'unet':
        net = UNet(in_channels=3, out_channels=num_classes)
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['model'])
    net = net.to(device)
    net = net.eval()

    T_val = utils.transforms.Compose([
        utils.transforms.Resize(size),
        utils.transforms.ToTensor(),
        utils.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])])

    val_dataset = utils.cityscapes.Cityscapes(data_root, 'val', T_val)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    pack = []
    for c in range(num_classes):
        pack.append({
            'mask': 0,
            'pred': 0,
            'inter': 0,
            'union': 0,
        })

    criterion = nn.CrossEntropyLoss()
    cityscapes_classes = [
        "unlabeled", "road", "sidewalk", "building", "wall", "fence",
        "pole", "traffic light", "traffic sign", "vegetation", "terrain",
        "sky", "person", "rider", "car", "truck",
        "bus", "train", "motorcycle", "bicycle"
    ]
    losses = 0
    with torch.inference_mode():
        for x, y in tqdm(val_loader):
            x = x.to(device)
            y = y.to(device)

            out = net(x)
            loss = criterion(out, y)

            losses += loss.detach()

            mask = y
            pred = torch.argmax(out, 1)

            true = mask == pred
            for c in range(num_classes):
                mask_c = mask == c
                pred_c = pred == c
                true_mask_c = true[mask_c]
                true_pred_c = true[pred_c]

                area_mask = len(true_mask_c)
                area_pred = len(true_pred_c)
                inter = torch.sum(true_mask_c).item()
                union = area_mask + area_pred - inter

                pack[c]['mask'] += area_mask
                pack[c]['pred'] += area_pred
                pack[c]['inter'] += inter
                pack[c]['union'] += union

    losses_val = losses / len(val_loader)
    print(losses_val)

    print('%8s %12s %12s %12s %12s %8s %8s %8s' %
          ('class', 'mask', 'pred', 'inter', 'union', 'acc', 'iou', 'dice'))
    acc_list = []
    iou_list = []
    dice_list = []
    for c, r in enumerate(pack):
        acc = r['inter'] / (r['mask'] + 1e-9)
        iou = r['inter'] / (r['union'] + 1e-9)
        dice = 2 * r['inter'] / (r['mask'] + r['pred'] + 1e-9)
        acc_list.append(acc)
        iou_list.append(iou)
        dice_list.append(dice)
        print('%8s %12d %12d %12d %12d %8.6f %8.6f %8.6f' %
              (cityscapes_classes[c], r['mask'], r['pred'], r['inter'], r['union'], acc, iou, dice))

    accuracy = np.mean(acc_list)
    miou = np.mean(iou_list)
    dice_coeff = np.mean(dice_list)

    print('acc  ->', accuracy)
    print('miou ->', miou)
    print('dice ->', dice_coeff)

