import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, mobilenet_v3_small, MobileNet_V3_Small_Weights, densenet121, DenseNet121_Weights
from transformers import AutoModelForImageClassification, AutoConfig

def set_parameter_requires_grad(model, feature_extracting):
    """
    Hàm helper để đóng băng các tham số.
    Nếu feature_extracting = True, tất cả các tham số sẽ bị đóng băng.
    """
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def get_model_chexpert_14(cfg):
    return get_model(cfg, feature_extract=False, useWeight = True, numclasses=14)

def get_model(cfg, feature_extract=False, useWeight = True, numclasses = 6):
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
            nn.Dropout(p=0.7),
            nn.Linear(num_ftrs, numclasses)
            # Bỏ Sigmoid, BCEWithLogitsLoss sẽ xử lý
        )

    elif cfg.MODEL.ARCH == 'mobilenet_v3_small':
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if useWeight else None
        model = mobilenet_v3_small(weights=weights)
        
        set_parameter_requires_grad(model, feature_extract)
        
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(
             nn.Dropout(p=0.7),
             nn.Linear(num_ftrs, numclasses)
        )
    elif cfg.MODEL.ARCH == 'densenet121':
        weights = DenseNet121_Weights.IMAGENET1K_V1 if useWeight else None
        model = densenet121(weights=weights)
        
        # set_parameter_requires_grad(model, feature_extract)
        
        # Lớp cuối của DenseNet có tên là 'classifier'
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Sequential(
            # nn.Dropout(p=0.5),
            nn.Linear(num_ftrs, numclasses),
            nn.Sigmoid()
        )
    elif cfg.MODEL.ARCH == 'cxr_foundation':
        # Load CXR Foundation model from Hugging Face
        model_name = "google/cxr-foundation"
        if useWeight:
            try:
                model = AutoModelForImageClassification.from_pretrained(model_name)
                print(f"Loaded pre-trained CXR Foundation model: {model_name}")
            except Exception as e:
                raise ValueError(f"Failed to load CXR Foundation model: {e}")
        else:
            # Load model architecture without pre-trained weights
            config = AutoConfig.from_pretrained(model_name)
            model = AutoModelForImageClassification(config)
            print(f"Initialized CXR Foundation model without pre-trained weights: {model_name}")
        
        # Freeze layers if needed
        set_parameter_requires_grad(model, feature_extract)
        
        # Replace classifier head to match numclasses
        # CXR Foundation model typically uses a ViT or ResNet backbone
        # We'll assume the classifier is accessible as model.classifier
        if hasattr(model, 'classifier'):
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(num_ftrs, numclasses)
            )
        else:
            raise ValueError("CXR Foundation model does not have a 'classifier' attribute. Check model architecture.")
    else:
        raise ValueError(f"Model architecture {cfg.MODEL.ARCH} not supported.")

    print(f"Model pre-trained on ImageNet loaded.")
    if feature_extract:
        print("Feature extracting mode: All layers frozen except the final classifier.")
    else:
        print("Fine-tuning mode: All layers are trainable.")
        
    print(f"Model adapted for {numclasses} classes.")
    
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
    
    # Tạo một config giả để test
    def create_test_config(arch_name):
        return OmegaConf.create({
            "MODEL": { "ARCH": arch_name, "NUM_CLASSES": 6 }
        })

    print("\n--- TESTING ResNet18 ---")
    cfg_resnet = create_test_config('resnet18')
    model_resnet = get_model(cfg_resnet)
    print(model_resnet.fc)

    print("\n" + "="*40)
    print("\n--- TESTING MobileNetV3-Small ---")
    cfg_mobile = create_test_config('mobilenet_v3_small')
    model_mobile = get_model(cfg_mobile)
    print(model_mobile.classifier)

    print("\n" + "="*40)
    print("\n--- TESTING DenseNet121 ---")
    cfg_dense = create_test_config('densenet121')
    model_dense = get_model(cfg_dense)
    print(model_dense.classifier)