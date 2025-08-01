import os
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights, mobilenet_v3_small, MobileNet_V3_Small_Weights, densenet121, DenseNet121_Weights

import torch
import torch.nn as nn
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model

def get_model(cfg, feature_extract=False, useWeight=True, numclasses=5):
    """
    Tải mô hình và điều chỉnh lớp classifier cuối cùng một cách chính xác và nhất quán.
    """
    model = None
    arch = cfg.MODEL.ARCH.lower()
    
    print(f">>> Loading model: {arch} | useWeight: {useWeight} | num_classes: {numclasses}")
    
    # Bước 1: Tải mô hình gốc
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

    # Đóng băng (nếu cần)
    # if feature_extract:
    #     set_parameter_requires_grad(model, True)

    # Xác định số features và thay thế classifier
    if hasattr(model, 'fc'): # Dành cho ResNet
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_ftrs, numclasses)
        )
    elif hasattr(model, 'classifier'): # Dành cho DenseNet và MobileNet
        # Xử lý DenseNet (classifier là một lớp Linear)
        if isinstance(model.classifier, nn.Linear):
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(num_ftrs, numclasses)
            )
        # Xử lý MobileNet (classifier là một Sequential)
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


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model_chexpert_14(cfg):
    return get_model(cfg, feature_extract=False, useWeight = True, numclasses=5)

def get_pretrained_model(cfg):
    # model_path = "./ckpt/resnet_14class_jul17_7h00.pth"
    model_path = './ckpt/resnet_j5_class_jul21_7h00.pth'
    # model_path = './ckpt/mobile_net_5_class_jul9_16h34.pth'
    print(f"Loading fine-tuned weights from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please run the training script first.")
    
    print(f"Found fine-tuned model at {model_path}")
    # Load the pre-trained model architecture
    model = get_model_chexpert_14(cfg)
    # Load the fine-tuned weights
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    print(f"Loaded fine-tuned model from {model_path}")
    
    print("Fine-tuned model loaded successfully.")
    return model

def build_model(cfg):
    """
    Xây dựng mô hình.
    - Nếu là dataset gốc (cifar10, cifar100), tải mô hình đã huấn luyện từ robustbench.
    - Nếu là dataset đa nhãn (như CXR), tải một mô hình nền và thay thế lớp classifier.
    """
    dataset_name = cfg.DATASET.NAME
    num_classes = cfg.MODEL.NUM_CLASSES

    if dataset_name in ["cifar10", "cifar100"]:
        # --- Logic gốc cho CIFAR ---
        print(f"Building pre-trained model for {dataset_name}...")
        base_model = load_model(
            cfg.MODEL.ARCH, 
            cfg.CKPT_DIR,
            dataset_name, 
            ThreatModel.corruptions
        ).cpu()

    elif 'CXR' in dataset_name:
        base_model = get_pretrained_model(cfg)
    else:
        raise NotImplementedError(f"Model building logic not implemented for dataset: {dataset_name}")

    return base_model