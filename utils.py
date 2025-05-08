import torch
import torchvision.models as models
import torch.nn as nn


def load_chexnet_model(pth_tar_path, num_classes=14):
    # Load DenseNet121
    model = models.densenet121(pretrained=False)

    # Replace classifier for CheXNet (14 diseases)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)

    # Load the .pth.tar file
    checkpoint = torch.load(pth_tar_path, map_location='cpu')

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Fix keys if they have 'module.' or 'features.' prefixes
    new_state_dict = {}
    for key in state_dict:
        new_key = key
        if key.startswith('module.'):
            new_key = key[len('module.'):]
        if new_key.startswith('features.') or new_key.startswith('classifier'):
            new_state_dict[new_key] = state_dict[key]

    # Load the cleaned state_dict
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    return model
