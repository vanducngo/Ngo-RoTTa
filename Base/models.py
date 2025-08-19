import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights, mobilenet_v3_small, MobileNet_V3_Small_Weights, densenet121, DenseNet121_Weights
import torch
import os

from RoTTA.core.utils.constants import DEVICE

def get_pretrained_model(cfg):
    model_path = './ckpt/Resnet18_60_aug18_00h00.pth'
    print(f"Loading fine-tuned weights from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please run the training script first.")
    
    print(f"Found fine-tuned model at {model_path}")
    # Load the pre-trained model architecture
    model = get_model(cfg, feature_extract=False, useWeight = False, numclasses=5)
    # Load the fine-tuned weights
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    print(f"Loaded fine-tuned model from {model_path}")
    
    print("Fine-tuned model loaded successfully.")
    return model

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def get_model_chexpert(cfg):
    return get_model(cfg, feature_extract=False, useWeight = False, numclasses=5)

def get_model(cfg, feature_extract=False, useWeight=True, numclasses=5):
    model = None
    arch = cfg.MODEL.ARCH.lower()
    
    print(f">>> Loading model: {arch} | useWeight: {useWeight} | num_classes: {numclasses}")
    
    # Step 1: Load model
    if arch == 'resnet18':
        weights = ResNet18_Weights.IMAGENET1K_V1 if useWeight else None
        model = resnet18(weights=weights)
    elif arch == 'resnet50':
        weights = ResNet50_Weights.IMAGENET1K_V1 if useWeight else None
        model = resnet50(weights=weights)
    elif arch == 'mobilenet_v3_small':
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if useWeight else None
        model = mobilenet_v3_small(weights=weights)
    elif arch == 'densenet121':
        weights = DenseNet121_Weights.IMAGENET1K_V1 if useWeight else None
        model = densenet121(weights=weights)
    else:
        raise ValueError(f"Model architecture {arch} not supported.")

    # Step 2: Freeze if needed
    # if feature_extract:
    #     set_parameter_requires_grad(model, True)

    # Step 3: Determine features and replace classifier
    if hasattr(model, 'fc'): # ResNet
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_ftrs, numclasses)
        )
    elif hasattr(model, 'classifier'): # DenseNet và MobileNet
        # DenseNet (classifier is a Linear)
        if isinstance(model.classifier, nn.Linear):
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(num_ftrs, numclasses)
            )
        # MobileNet (classifier is a Sequential)
        elif isinstance(model.classifier, nn.Sequential):
            num_ftrs = model.classifier[-1].in_features
            model.classifier = nn.Sequential(
                nn.Linear(model.classifier[0].in_features, 512), # Lớp ẩn mới
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(512, numclasses)
            )
        else:
            raise TypeError(f"Unsupported classifier type: {type(model.classifier)}")
    else:
        raise AttributeError("Model does not have 'fc' or 'classifier' attribute.")


    print(f"Model pre-trained on ImageNet loaded.")
    if feature_extract:
        print("Feature extracting mode: All layers frozen except the final classifier.")
    else:
        print("Fine-tuning mode: All layers are trainable.")
        
    print(f"Model adapted for {numclasses} classes.")
    return model

def unfreeze_specific_layers(model, layers_to_unfreeze=['layer4', 'fc']):
    print(f"\nUnfreezing specific layers: {layers_to_unfreeze}")
    for name, param in model.named_parameters():
        for layer_name in layers_to_unfreeze:
            if name.startswith(layer_name):
                param.requires_grad = True
                break
    
    print("Trainable parameters after unfreezing:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)