from torchvision import models
import torch.nn as nn
import timm
def efficientnet():
    # Load pretrained EfficientNet-B2
    model = models.efficientnet_b2(weights='IMAGENET1K_V1')

    # Modify input convolution layer (for 1 channel)
    model.features[0][0] = nn.Conv2d(
        1, 32, kernel_size=3, stride=2, padding=1, bias=False
    )

    # Modify classifier layer (for 2 classes)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)  # Output 2 logits
    
    return model
# Load pretrained ViT

def vit():
    model = timm.create_model('vit_base_patch16_224', pretrained=True)

    # Modify input layer
    model.patch_embed.proj = nn.Conv2d(
        1, model.embed_dim, kernel_size=16, stride=16
    )

    # Modify classifier
    model.head = nn.Linear(model.embed_dim, 2)
    return model
