import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, mobilenet_v3_small, MobileNet_V3_Small_Weights

def set_parameter_requires_grad(model, feature_extracting):
    """
    Hàm helper để đóng băng các tham số.
    Nếu feature_extracting = True, tất cả các tham số sẽ bị đóng băng.
    """
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def get_model(cfg, feature_extract=False, useWeight = True):
    """
    Tải mô hình và tùy chọn đóng băng các lớp đầu để chỉ fine-tune các lớp cuối.

    Args:
        cfg (OmegaConf): Đối tượng cấu hình.
        feature_extract (bool): Nếu True, đóng băng tất cả các lớp trừ lớp cuối.
    """
    model = None
    num_ftrs = 0
    
    print(f">>> Loading model: {cfg.MODEL.ARCH} -> useWeight: {useWeight}")
    
    if cfg.MODEL.ARCH == 'resnet18':
        weights = ResNet18_Weights.IMAGENET1K_V1 if useWeight else None
        model = resnet18(weights=weights)
        
        # Đóng băng các lớp nếu cần
        set_parameter_requires_grad(model, feature_extract)
        
        # Thay thế lớp cuối và đảm bảo nó luôn có thể huấn luyện
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_ftrs, cfg.MODEL.NUM_CLASSES)
            # Bỏ Sigmoid, BCEWithLogitsLoss sẽ xử lý
        )

    elif cfg.MODEL.ARCH == 'mobilenet_v3_small':
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if useWeight else None
        model = mobilenet_v3_small(weights=weights)
        
        set_parameter_requires_grad(model, feature_extract)
        
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(
             nn.Dropout(p=0.5),
             nn.Linear(num_ftrs, cfg.MODEL.NUM_CLASSES)
        )
        
    else:
        raise ValueError(f"Model architecture {cfg.MODEL.ARCH} not supported.")

    print(f"Model pre-trained on ImageNet loaded.")
    if feature_extract:
        print("Feature extracting mode: All layers frozen except the final classifier.")
    else:
        print("Fine-tuning mode: All layers are trainable.")
        
    print(f"Model adapted for {cfg.MODEL.NUM_CLASSES} classes.")
    
    return model

def unfreeze_specific_layers(model, layers_to_unfreeze=['layer4', 'fc']):
    """
    "Mở băng" các lớp cụ thể để fine-tune.
    
    Args:
        model (nn.Module): Mô hình đã bị đóng băng.
        layers_to_unfreeze (list): Danh sách tên các lớp cần mở băng.
    """
    print(f"\nUnfreezing specific layers: {layers_to_unfreeze}")
    for name, param in model.named_parameters():
        for layer_name in layers_to_unfreeze:
            # Kiểm tra xem tên tham số có bắt đầu bằng tên lớp cần mở không
            # Ví dụ: 'layer4.0.conv1.weight' sẽ khớp với 'layer4'
            if name.startswith(layer_name):
                param.requires_grad = True
                break # Đã tìm thấy, không cần kiểm tra các tên lớp khác cho tham số này
    
    print("Trainable parameters after unfreezing:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

# --- Phần test file ---
if __name__ == '__main__':
    from omegaconf import OmegaConf
    
    # Giả định có file config.yaml
    cfg = OmegaConf.create({
        "MODEL": {
            "ARCH": "resnet18",
            "NUM_CLASSES": 6
        }
    })

    print("\n--- TEST 1: FINE-TUNING MODE (DEFAULT) ---")
    model_ft = get_model(cfg, feature_extract=False)
    # In ra để xem tất cả các tham số đều có requires_grad = True
    # for name, param in model_ft.named_parameters():
    #     print(f"{name}: requires_grad={param.requires_grad}")

    print("\n--- TEST 2: FEATURE EXTRACTING MODE ---")
    model_fe = get_model(cfg, feature_extract=True)
    # In ra để xem chỉ lớp cuối có requires_grad = True
    # print("Trainable parameters in feature extracting mode:")
    # for name, param in model_fe.named_parameters():
    #     if param.requires_grad:
    #         print(name)
            
    print("\n--- TEST 3: UNFREEZING layer4 and fc ---")
    # Bắt đầu với mô hình bị đóng băng hoàn toàn
    model_unfrozen = get_model(cfg, feature_extract=True)
    unfreeze_specific_layers(model_unfrozen, layers_to_unfreeze=['layer4', 'fc'])